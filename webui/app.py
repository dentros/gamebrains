"""
Flask app for the GameBrains web demo (see package docstring in __init__.py for scope/honesty
note: a local, form-driven demo interface -- a first, functional step toward the full gamified
"houses" UI, not that UI itself).

Two pages:
  /       a single heterogeneous match, every parameter chosen through a dynamic roster builder.
  /evolve the Markov-brain genetic algorithm, plus a "bake-off" mode that compares the evolved
          animat against separately-pretrained-then-frozen RL agents in one evaluation match.

Run from the "gametheoretic platform" directory:

    "./gamebrains/.venv/Scripts/python.exe" -m gamebrains.webui.app

then open http://127.0.0.1:5000/ in a browser.
"""

from __future__ import annotations

import base64
import contextlib
import csv
import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from flask import Flask, render_template, request

from ..agents.classic import AllC, AllD, MajorityTFT, RandomAgent
from ..agents.qlearning import QLearningAgent
from ..engine.agent import Agent
from ..engine.console import LiveConsole
from ..engine.eventlog import EventLog
from ..engine.evolution import EvolutionConfig, evolve
from ..engine.runner import run_match
from ..games.congestion import from_preset as congestion_from_preset
from ..games.public_goods import PublicGoodsGame
from ..metrics import equilibrium, graph, information, social, social_alt
from ..repository.cas import ContentStore
from ..repository.ledger import Ledger
from ..repository.record import (
    DEFAULT_CODE_VERSION, _config_game_desc, _feature_vector, _PARAM_ATTRS, _roster_description,
    protocol_from_run, record_experiment,
)
from ..repository.smart_filter import lookup as smart_filter_lookup

app = Flask(__name__)


@app.context_processor
def _inject_active_tab():
    if request.path.startswith("/evolve"):
        return {"active_tab": "evolve"}
    if request.path.startswith("/analytics"):
        return {"active_tab": "analytics"}
    if request.path.startswith("/spacemap"):
        return {"active_tab": "spacemap"}
    return {"active_tab": "match"}


_GAMEBRAINS_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _GAMEBRAINS_ROOT / "results"
_REPO_ROOT = _GAMEBRAINS_ROOT / "repo_store"

_CLASSIC_STRATEGIES = ["AllC", "AllD", "Random", "MajorityTFT"]
_NASH_MAX_AGENTS = 10  # enumpure_solve builds a 2^n table -- keep this bounded in a web request

# One "player piece" (glyph + colour) per cognitive architecture -- shared between the roster
# builder and the results page so a "kind" always looks the same wherever it appears. The colour
# matches the --piece-<kind> CSS custom property in base.html; the glyph matches an <symbol id=
# "glyph-<kind>"> defined there. The technical kind name is always shown alongside it.
KIND_META = {
    "qlearning":    {"glyph": "glyph-qlearning", "label": "Q-learning",       "color": "#3d6b8a"},
    "dqn":          {"glyph": "glyph-dqn",       "label": "Deep Q-Network",   "color": "#7a4a8a"},
    "fep":          {"glyph": "glyph-fep",       "label": "FEP / Active Inference", "color": "#6b8a3d"},
    "markov_brain": {"glyph": "glyph-markov_brain", "label": "Markov-brain (evolutionary)", "color": "#b5541f"},
    "classic":      {"glyph": "glyph-classic",   "label": "Classic (fixed)",  "color": "#5c5346"},
}
_KIND_ORDER = ["qlearning", "dqn", "fep", "markov_brain", "classic"]

_METRIC_META = [
    ("cooperation_rate", "Cooperation rate"),
    ("efficiency", "Efficiency"),
    ("payoff_gini", "Payoff Gini"),
    ("action_entropy_bits", "Action entropy (bits)"),
    ("mutual_information_bits", "Mutual info I(obs;action) (bits)"),
    ("transfer_entropy_bits", "Transfer entropy (bits, significant pairs)"),
    ("predictive_information_bits", "Predictive info I(past;future) (bits)"),
    ("coop_graph_weight", "Co-cooperation graph weight"),
    ("coop_graph_clustering", "Co-cooperation clustering"),
    ("influence_graph_density", "TE-influence graph density"),
]
# Roadmap placeholders: shown disabled so the tool documents where the platform is going, not just
# what it does today. The whole CLAUDE.md section-5 metric family has now graduated out of this
# list (TE/MI 2026-07-17, predictive information + graph metrics 2026-07-19) -- what remains
# planned lives at the game level (_GAME_ROADMAP), not the metric level.
_METRIC_ROADMAP: list[str] = []
_GAME_ROADMAP = ["Battle of the Sexes (coming soon)", "Custom game builder (coming soon)"]

#: Temporal-fairness measures, shown only for episodic games since they are defined per episode.
#: `alt_efficiency` is renamed at the merge point: the papers' Efficiency is per episode, while
#: `social.py`'s is per round, and two different numbers must not share one label.
_ALT_METRIC_META = [
    ("CALT", "CALT (primary alternation)"),
    ("EALT", "EALT (exclusivity)"),
    ("AALT", "AALT (strictest)"),
    ("FALT", "FALT (reaches)"),
    ("qFALT", "qFALT"),
    ("qEALT", "qEALT"),
    ("RP_excl", "RP (exclusive wins)"),
    ("RS_excl", "RS rhythm (exclusive)"),
    ("WPE_excl", "WPE frequency (exclusive)"),
    ("RP_reach", "RP (reaches)"),
    ("RS_reach", "RS rhythm (reaches)"),
    ("WPE_reach", "WPE frequency (reaches)"),
    ("alt_efficiency", "Efficiency (per episode)"),
    ("reward_fairness", "Reward Fairness"),
    ("tt_fairness", "Turn-Taking Fairness"),
    ("fairness", "Fairness (exclusive wins)"),
]

#: Selectable games. Each congestion entry is a preset of the one parametrized family; the label
#: says which are published configurations so an exploratory run is never mistaken for one.
_GAME_CHOICES = [
    ("public_goods", "Public Goods Game (n-player Prisoner's Dilemma)"),
    ("congestion:hjg", "Honey-Jar Game, ILF (published main)"),
    ("congestion:hjg_iqf", "Honey-Jar Game, IQF (published)"),
    ("congestion:hjg_k", "Honey-Jar Game, k-variant (published robustness check)"),
    ("congestion:hjg_memory", "Honey-Jar Game, Type-B memory"),
    ("congestion:market_entry", "Market entry (one-shot / ballistic)"),
]

#: Reward denominators for the congestion family. "n" divides by the whole population, "k" by the
#: number who actually collided; the latter is literal Rosenthal congestion. Only the first four
#: appear in the papers, and the label says so.
_REWARD_RULE_CHOICES = [
    ("", "keep the preset's rule"),
    ("ILF", "ILF: r/n, linear in population (published)"),
    ("IQF", "IQF: r/n squared (published)"),
    ("KLF", "KLF: r/k, per claimant (published)"),
    ("KQF", "KQF: r/k squared (published)"),
    ("ICF", "ICF: r/n cubed (exploratory)"),
    ("KCF", "KCF: r/k cubed (exploratory)"),
]


# --- a frozen wrapper for the bake-off (Section /evolve) -----------------------------------------

class _Frozen(Agent):
    """Wraps an already-trained agent so it acts greedily (epsilon forced to 0 for the duration
    of the call) and never learns further. Used only to build a fair bake-off evaluation match:
    an agent that finished its own training/evolution process should not keep adapting *during*
    the comparison, or the comparison would conflate "final policy quality" with "how much it
    adapted mid-evaluation" -- the exact confound a bake-off is meant to avoid.
    """
    training_mode = "fixed"

    def __init__(self, agent: Any, name: str) -> None:
        self._agent = agent
        self.name = name
        self.kind = agent.kind

    def act(self, observation: int) -> int:
        old_eps = getattr(self._agent, "epsilon", None)
        if old_eps is not None:
            self._agent.epsilon = 0.0
        try:
            return self._agent.act(observation)
        finally:
            if old_eps is not None:
                self._agent.epsilon = old_eps

    def on_match_start(self, game: Any) -> None:
        # Forward, or a frozen classic strategy never learns which action plays its role.
        self._agent.on_match_start(game)

    def update(self, *args, **kwargs) -> None:
        return None

    def inspect(self) -> dict:
        return self._agent.inspect()

    def render_brain(self) -> dict:
        return self._agent.render_brain()


# --- roster construction ---------------------------------------------------------------------

def _classic_agent(strategy: str, i: int, n: int, seed: int, extra: dict[str, Any] | None = None):
    extra = extra or {}
    if strategy == "AllC":
        return AllC(f"AllC {i}")
    if strategy == "AllD":
        return AllD(f"AllD {i}")
    if strategy == "Random":
        return RandomAgent(f"Random {i}", p_cooperate=extra.get("p_cooperate", 0.5), seed=seed + 400 + i)
    if strategy == "MajorityTFT":
        return MajorityTFT(f"TFT {i}", n_agents=n)
    raise ValueError(f"unknown classic strategy {strategy}")


def _make_agent(kind: str, i: int, game: PublicGoodsGame, seed: int, classic_strategy: str,
                reciprocity: float, markov_hidden: int, epsilon_decay: float = 0.9995,
                epsilon_min: float = 0.02, extra: dict[str, Any] | None = None):
    """`extra` carries kind-specific hyperparameter overrides beyond the handful already exposed
    as their own arguments (the roster builder's "Advanced" section) -- see `_ADVANCED_PARAMS`
    for the full list per kind and each one's default. Any key not present in `extra` falls back
    to the underlying Agent class's own constructor default, not a value duplicated here."""
    extra = extra or {}
    labels, actions = game.state_labels(), game.action_names
    if kind == "qlearning":
        return QLearningAgent(
            name=f"Q-learner {i}", n_states=game.n_states, n_actions=game.n_actions,
            alpha=extra.get("alpha", 0.1), gamma=extra.get("gamma", 0.95),
            epsilon=1.0, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
            seed=seed + 100 + i, state_labels=labels, action_labels=actions,
        )
    if kind == "dqn":
        from ..agents.dqn import DQNAgent
        kwargs = {}
        for key in ("hidden", "lr", "gamma", "buffer_size", "batch_size", "train_every", "target_sync_every"):
            if key in extra:
                kwargs[key] = extra[key]
        return DQNAgent(
            name=f"DeepQ {i}", n_states=game.n_states, n_actions=game.n_actions,
            epsilon=1.0, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
            seed=seed + 200 + i, state_labels=labels, action_labels=actions, **kwargs,
        )
    if kind == "fep":
        from ..agents.fep import FEPAgent
        kwargs = {k: extra[k] for k in ("obs_noise", "drift", "precision") if k in extra}
        return FEPAgent(
            name=f"FEP {i}", n_agents=game.n_agents, mpcr=game.mpcr, cost=game.cost,
            reciprocity=reciprocity, seed=seed + 300 + i, start_state=game.start_state, **kwargs,
        )
    if kind == "markov_brain":
        from ..agents.markov_brain import MarkovBrainAgent
        return MarkovBrainAgent(
            name=f"MarkovBrain {i}", n_states=game.n_states, n_actions=game.n_actions,
            n_hidden=markov_hidden, seed=seed + 500 + i, start_state=game.start_state,
        )
    if kind == "classic":
        return _classic_agent(classic_strategy, i, game.n_agents, seed, extra)
    raise ValueError(f"unknown kind {kind}")


# Advanced (collapsed-by-default) per-kind hyperparameters exposed in the roster builder, beyond
# the always-visible ones (epsilon schedule, reciprocity, hidden nodes, classic strategy). Each
# tuple is (form-field suffix, default, help text) -- the default is shown as the input's value
# and matches the underlying Agent class's own constructor default exactly, so leaving a field
# untouched reproduces today's behaviour bit-for-bit.
_ADVANCED_PARAMS: dict[str, list[tuple[str, float, str]]] = {
    "qlearning": [("alpha", 0.1, "learning rate"), ("gamma", 0.95, "discount factor")],
    "dqn": [("hidden", 64, "hidden layer size"), ("lr", 0.001, "learning rate"),
           ("gamma", 0.95, "discount factor"), ("buffer_size", 10000, "replay buffer size"),
           ("batch_size", 64, "training batch size"), ("train_every", 1, "train every N steps"),
           ("target_sync_every", 200, "sync target network every N steps")],
    "fep": [("obs_noise", 0.75, "observation likelihood spread"),
           ("drift", 0.1, "belief drift toward uniform per round"),
           ("precision", 4.0, "softmax precision over expected value")],
    "classic": [("p_cooperate", 0.5, "P(Cooperate) for the Random strategy only")],
}


def _compute_phi_sequential(agent: Any, n_t: int = 5):
    """Φ/causal-autonomy, forcing PyPhi's PARALLEL_CUT_EVALUATION off for the duration of this
    call. `metrics/phi_autonomy.py` deliberately leaves that optimization ON by default (85x
    speedup, safe from real script/module invocations -- see its module docstring and
    CLAUDE.md §9d). It is NOT safe from inside this long-lived Flask worker process: a request
    with a large-ish Markov-brain (n_hidden=3 observed) reproducibly leaked orphaned
    spawn-multiprocessing workers and crashed with MemoryError while those workers tried to
    re-import the vendored pyphi package -- a second, distinct instance of the same
    Windows-spawn-based-multiprocessing caution already on file (see the Flask-reloader
    incident noted in webui/__init__.py's history / CLAUDE.md §9e), not a new failure mode.
    Trading the speedup for reliability here is the right call in a server context."""
    from ..metrics.phi_autonomy import causal_autonomy, compute_phi  # imports pyphi + sets sys.path
    import pyphi
    with pyphi.config.override(PARALLEL_CUT_EVALUATION=False):
        phi_result = compute_phi(agent)
        autonomy = causal_autonomy(agent, n_t=n_t)
    return phi_result, autonomy


def _epsilon_rounds_to_min(decay: float, emin: float) -> float:
    """Rounds needed for epsilon=1.0 to multiplicatively decay down to `emin` (see qlearning.py /
    dqn.py: `epsilon *= epsilon_decay` each round, floored at `epsilon_min`)."""
    if not (0 < decay < 1) or emin <= 0:
        return float("nan")
    return float(np.log(emin) / np.log(decay))


# --- HTML rendering for one agent's brain (creature card) ---------------------------------------

def _heat_color(value: float, vmax: float) -> str:
    """Diverging red(-)/green(+) background, intensity scaled by |value| / vmax."""
    if vmax <= 0:
        return "background:#2a3050"
    t = min(abs(value) / vmax, 1.0)
    if value >= 0:
        return f"background:rgba(34,160,80,{0.12 + 0.55 * t:.2f})"
    return f"background:rgba(210,60,60,{0.12 + 0.55 * t:.2f})"


def _mono_color(value: float, vmax: float) -> str:
    """Single-hue (green) intensity for non-negative magnitudes (weight strengths, etc.)."""
    if vmax <= 0:
        return "background:#2a2f3a"
    t = min(value / vmax, 1.0)
    return f"background:rgba(46,204,113,{0.15 + 0.7 * t:.2f})"


def _qtable_html(brain: dict, q_key: str) -> str:
    q = np.asarray(brain[q_key])
    states = brain.get("state_labels") or [f"s{i}" for i in range(len(q))]
    actions = brain.get("action_labels") or [f"a{j}" for j in range(q.shape[1])]
    policy = brain.get("greedy_policy")
    vmax = float(np.abs(q).max()) if q.size else 1.0
    head = "".join(f"<th>{a}</th>" for a in actions)
    rows = []
    for s, row in enumerate(q):
        cells = "".join(
            f"<td style='{_heat_color(v, vmax)}'>{v:+.2f}</td>" for v in row
        )
        pick = f"<td class='pick'>&rarr; {actions[policy[s]]}</td>" if policy is not None else ""
        rows.append(f"<tr><th>{states[s]}</th>{cells}{pick}</tr>")
    eps = brain.get("epsilon")
    eps_txt = f"<p class='meta'>&epsilon; = {eps:.3f}</p>" if eps is not None else ""
    return f"{eps_txt}<table class='qtable'><tr><th>state</th>{head}<th></th></tr>{''.join(rows)}</table>"


def _dqn_layer_row(W: np.ndarray, label: str, cap: int = 24) -> str:
    magnitudes = np.abs(W).mean(axis=1)  # per-output-neuron mean |weight| of incoming connections
    vmax = float(magnitudes.max()) if magnitudes.size else 1.0
    n = len(magnitudes)
    dots = "".join(
        f"<span class='neuron' style='{_mono_color(float(m), vmax)}' title='{m:.3f}'></span>"
        for m in magnitudes[:cap]
    )
    more = f"<span class='moretag'>+{n - cap}</span>" if n > cap else ""
    return f"<div class='netlayer'><span class='netlabel'>{label} ({n})</span>{dots}{more}</div>"


def _dqn_network_html(agent: Any) -> str:
    """A genuine network-structure view -- NOT the derived Q-table (that's rendered separately).
    `_MLP.net` is always `Linear, ReLU, Linear, ReLU, Linear` (see agents/dqn.py), so fixed
    indices 0/2/4 are safe without needing to import torch here just for an isinstance check."""
    net = agent.policy.net
    layer_specs = [(net[0], "input &rarr; hidden1"), (net[2], "hidden1 &rarr; hidden2"),
                   (net[4], "hidden2 &rarr; output")]
    rows = "".join(
        _dqn_layer_row(lin.weight.detach().cpu().numpy(), label) for lin, label in layer_specs
    )
    n_params = sum(p.numel() for p in agent.policy.parameters())
    return (f"<div class='dqnnet'>{rows}</div>"
           f"<p class='meta'>{n_params:,} parameters. Each circle is an output node; colour is "
           f"the mean absolute weight of its incoming connections (darker = stronger). "
           f"This is the network's architecture, not the Q-values it produces (see table below).</p>")


# One shared slide-out drawer (base.html) replaces what used to be a <details> accordion repeated
# inline in every single creature card of the same kind (e.g. three Q-learners each carried their
# own copy of the same explanation). The content below is now rendered exactly once per page
# (results.html's <template id="help-tmpl-*"> blocks); each card just gets a small trigger button
# (`_help_trigger`) pointing at the shared topic. Less scrolling, no duplicated text.
_HELP_TOPICS: dict[str, tuple[str, str]] = {
    "fep": ("FEP / Active Inference", (
        "<p><b>Theory.</b> The FEP agent holds a categorical belief over how many other players are "
        "cooperating, updated by exact Bayesian filtering each round. It picks an action by minimizing "
        "expected free energy: a softmax over expected value, not reward-maximizing learning like "
        "Q-learning.</p>"
        "<p><b>Reading it.</b> Each bar is the probability the agent assigns to that many cooperators. "
        "E[others] is the expectation of that distribution. <code>reciprocity</code> controls how "
        "social the agent is: 0 is purely selfish (maximizes only its own payoff); higher values make "
        "it prefer cooperating when it believes others will too: a simple Theory-of-Mind rule, since "
        "what I expect others to do shapes what I do.</p>"
    )),
    "nash": ("Nash equilibrium (pygambit)", (
        "<p><b>Theory.</b> Computed analytically (via pygambit) on the single-round stage game: which "
        "action profile is stable when no player benefits from unilaterally deviating, given mpcr/cost/n. "
        "It ignores repeated-game effects such as reputation and whatever the actual agents learned.</p>"
        "<p><b>Reading it.</b> This is where a fully rational, one-shot player would end up. Compare it "
        "with the cooperation rate this run actually reached: if your agents haven't converged there yet "
        "(e.g. epsilon is still high), they likely need more rounds.</p>"
    )),
}


def _help_trigger(topic: str) -> str:
    title = _HELP_TOPICS[topic][0]
    return (f"<button type='button' class='help-trigger' onclick=\"openHelpDrawer('{topic}')\">"
           f"&#8505; How to read this ({title})</button>")


_MB_ROLE_COLORS = {"sensor": "#3d6b8a", "hidden": "#b5541f", "motor": "#6b8a3d"}


def _mb_role(k: int, ns: int, nh: int) -> str:
    return "sensor" if k < ns else ("hidden" if k < ns + nh else "motor")


def _markov_wiring_svg(W: np.ndarray, labels: list[str], ns: int, nh: int) -> str:
    """The evolved wiring as a bipartite from->to diagram: every node on the left (sources,
    including the recurrent hidden/motor nodes), the controlled nodes (hidden+motor, the ones the
    genome actually drives) on the right. Edge thickness/opacity scale with |weight|; green =
    excitatory (positive), red = inhibitory (negative). Weak edges (|w| < 20% of the strongest)
    are omitted so the picture stays readable -- the full matrix is in the table below it."""
    n_controlled, n_nodes = W.shape
    row_h, pad_top, lx, rx, width = 30, 18, 70, 230, 300
    height = pad_top + row_h * max(n_nodes, n_controlled) + 6
    vmax = float(np.abs(W).max()) or 1.0
    ly = {k: pad_top + row_h * k for k in range(n_nodes)}
    ry = {t: pad_top + row_h * t for t in range(n_controlled)}
    parts = []
    for t in range(n_controlled):
        for k in range(n_nodes):
            w = float(W[t, k])
            if abs(w) < 0.2 * vmax:
                continue
            color = "#2e8b57" if w >= 0 else "#c0392b"
            parts.append(
                f"<line x1='{lx + 8}' y1='{ly[k]}' x2='{rx - 8}' y2='{ry[t]}' "
                f"stroke='{color}' stroke-width='{1 + 2.5 * abs(w) / vmax:.1f}' "
                f"opacity='{0.3 + 0.6 * abs(w) / vmax:.2f}'><title>{labels[k]} &rarr; "
                f"{labels[ns + t]}: {w:+.2f}</title></line>"
            )
    for k in range(n_nodes):
        c = _MB_ROLE_COLORS[_mb_role(k, ns, nh)]
        parts.append(f"<circle cx='{lx}' cy='{ly[k]}' r='7' fill='{c}'/>")
        parts.append(f"<text x='{lx - 14}' y='{ly[k] + 4}' text-anchor='end' font-size='11' "
                     f"fill='currentColor'>{labels[k]}</text>")
    for t in range(n_controlled):
        c = _MB_ROLE_COLORS[_mb_role(ns + t, ns, nh)]
        parts.append(f"<circle cx='{rx}' cy='{ry[t]}' r='7' fill='{c}'/>")
        parts.append(f"<text x='{rx + 14}' y='{ry[t] + 4}' font-size='11' "
                     f"fill='currentColor'>{labels[ns + t]}</text>")
    return (f"<svg viewBox='0 0 {width} {height}' width='{width}' height='{height}' "
           f"class='mbwiring' role='img'>{''.join(parts)}</svg>")


def _markov_weight_table(W: np.ndarray, bias: np.ndarray, labels: list[str], ns: int) -> str:
    vmax = float(max(np.abs(W).max(), np.abs(bias).max())) or 1.0
    head = "".join(f"<th>{lab}</th>" for lab in labels) + "<th>bias</th>"
    rows = []
    for t, row in enumerate(W):
        cells = "".join(f"<td style='{_heat_color(float(v), vmax)}'>{v:+.2f}</td>" for v in row)
        cells += f"<td style='{_heat_color(float(bias[t]), vmax)}'>{bias[t]:+.2f}</td>"
        rows.append(f"<tr><th>{labels[ns + t]}</th>{cells}</tr>")
    return f"<table class='qtable'><tr><th>to \\ from</th>{head}</tr>{''.join(rows)}</table>"


def _markov_tpm_html(W: np.ndarray, bias: np.ndarray, state: list[int],
                     labels: list[str], ns: int) -> str:
    """The TPM over the controlled (hidden+motor) subsystem, conditioned on the sensors as
    currently read -- the same subsystem metrics/phi_autonomy.py analyzes. Each row is one
    possible current (hidden,motor) configuration; each cell is P(that node = 1 next step).
    Guarded to small brains: 2^n_controlled rows explode fast, and this view is only legible
    when it fits on a screen."""
    n_controlled = W.shape[0]
    if n_controlled > 5:
        return ("<p class='meta'>TPM omitted: 2^" + str(n_controlled) +
                " rows is beyond what a table can usefully show.</p>")
    sensor_bits = np.array(state[:ns], dtype=float)
    combos = 2 ** n_controlled
    head = "".join(f"<th>P({labels[ns + t]}=1)</th>" for t in range(n_controlled))
    rows = []
    for c in range(combos):
        bits = np.array([(c >> k) & 1 for k in range(n_controlled)], dtype=float)  # little-endian
        full_state = np.concatenate([sensor_bits, bits])
        p = 1.0 / (1.0 + np.exp(-(W @ full_state + bias)))
        label = "".join(str(int(b)) for b in bits)
        cells = "".join(f"<td style='{_mono_color(float(v), 1.0)}'>{v:.2f}</td>" for v in p)
        rows.append(f"<tr><th><code>{label}</code></th>{cells}</tr>")
    return (f"<table class='qtable'><tr><th>state ({''.join(labels[ns:])})</th>{head}</tr>"
           f"{''.join(rows)}</table>"
           f"<p class='meta'>Conditioned on the sensors as currently read "
           f"(<code>{''.join(str(int(b)) for b in sensor_bits)}</code>); this is the same "
           f"hidden+motor subsystem the &Phi;/autonomy panel analyzes.</p>")


def _markov_brain_html(brain: dict) -> str:
    state = brain["state"]
    ns, nh = brain["n_sensor"], brain["n_hidden"]
    boxes = "".join(
        f"<span class='bit {'on' if b else 'off'} {_mb_role(k, ns, nh)}'>{b}</span>"
        for k, b in enumerate(state)
    )
    out = (f"<div class='mbrain'>{boxes}</div>"
          f"<p class='meta'>sensor={ns} hidden={nh} motor={brain['n_motor']} "
          f"<span class='legend'>(sensor / hidden / motor)</span>. A snapshot of the final "
          f"state, not something it learned during this match: training_mode is "
          f"\"evolutionary\", meaning it evolves between generations, not within a match "
          f"(see the Evolutionary tab).</p>")
    # Older genomes loaded from the CAS may predate render_brain() exposing W/bias -- degrade
    # gracefully to the state-bits view rather than crashing the whole results page.
    if "W" in brain and "bias" in brain:
        W = np.asarray(brain["W"], dtype=float)
        bias = np.asarray(brain["bias"], dtype=float)
        labels = brain.get("node_labels") or [f"n{k}" for k in range(W.shape[1])]
        out += (
            f"<h4>Evolved wiring</h4>{_markov_wiring_svg(W, labels, ns, nh)}"
            f"<p class='meta'>Green = excitatory, red = inhibitory; thickness = |weight|. Edges "
            f"below 20% of the strongest are hidden here but shown in the matrix.</p>"
            f"<details class='explain'><summary>Weight matrix (the genome itself)</summary>"
            f"{_markov_weight_table(W, bias, labels, ns)}</details>"
            f"<details class='explain'><summary>Transition probabilities (TPM)</summary>"
            f"{_markov_tpm_html(W, bias, state, labels, ns)}</details>"
        )
    return out


def _classic_html(brain: dict) -> str:
    extra = f" (p={brain['p_cooperate']:g})" if "p_cooperate" in brain else ""
    return f"<p class='rule'><b>{brain['strategy']}</b>{extra}: {brain['rule']}</p>"


def _policy_vs_behavior_html(brain: dict, coop_rate: float | None) -> str:
    """Explainability correlation for RL agents: what the learned greedy policy WOULD do (how many
    states it picks Cooperate in) next to what the agent ACTUALLY did this match. A large gap with
    high epsilon is exploration, not a bug; a large gap with epsilon near its floor means the
    policy shifted late in the match and the run-average no longer reflects it."""
    policy = brain.get("greedy_policy")
    if policy is None or coop_rate is None:
        return ""
    n_states = len(policy)
    n_coop = int(sum(1 for a in policy if a == 1))
    eps = brain.get("epsilon")
    eps_note = f" with &epsilon; = {eps:.3f} still forcing random moves" if eps and eps > 0.1 else ""
    return (f"<p class='meta'><b>Policy vs behavior:</b> the greedy policy cooperates in "
           f"{n_coop}/{n_states} states; this agent actually cooperated in "
           f"{coop_rate * 100:.1f}% of rounds{eps_note}.</p>")


_HELP_TOPICS["qlearning"] = ("Q-learning", (
    "<p><b>Theory.</b> Tabular Q-learning keeps one row per observed state (here: how many players "
    "cooperated last round) and one column per action; each cell estimates the long-run value of "
    "taking that action in that state. The table IS the agent's entire mind -- nothing is hidden.</p>"
    "<p><b>Reading it.</b> Green = higher value. The arrow marks the greedy choice per state. "
    "With &epsilon;-greedy exploration the agent sometimes acts against its own table on purpose; "
    "the policy-vs-behavior line quantifies exactly how often that happened this match.</p>"
))

_HELP_TOPICS["alt"] = ("Temporal fairness: ALT and RP", (
    "<p><b>Why these exist.</b> Efficiency and fairness are time-averaged: they ask how much each "
    "agent ended up with. That cannot see <i>whether access rotated</i>. Two agents who collide "
    "in every single episode and split the reduced share can post near-perfect Reward Fairness "
    "while never once taking turns. ALT and RP ask the question the totals cannot: which agent "
    "got access, and when.</p>"
    "<p><b>Reading them.</b> All are 0 to 1, higher is better. <b>CALT</b> is the primary "
    "measure; <b>AALT</b> is the strictest, counting only agents with exactly one solo win per "
    "window; <b>FALT</b> is the loosest, counting arrivals including ties. <b>RP</b> is the "
    "cheap proxy, the mean of <b>RS</b> (rhythm: are the gaps between an agent's wins even?) and "
    "<b>WPE</b> (frequency: did it win its fair share?). Those two are meant to be independent, "
    "so a run can score well on one and badly on the other.</p>"
    "<p><b>Reach vs exclusive.</b> Every measure is reported both ways, because an agent can "
    "reach the terminal constantly and never once arrive alone. That reads as healthy access by "
    "one definition and total failure by the other, and neither is designated the answer.</p>"
    "<p><b>A caution for anti-coordination games.</b> No constant strategy is collectively good "
    "here. If everyone gives way the payoff is zero, exactly as it is if everyone rushes. The "
    "best behaviour is taking turns, which no fixed strategy can express, so an always-concede "
    "agent is a baseline rather than a cooperative one.</p>"
))

_HELP_TOPICS["dqn"] = ("Deep Q-Network", (
    "<p><b>Theory.</b> The DQN replaces the table with a neural network that maps a one-hot state "
    "to Q-values, trained by experience replay against a target network. The network diagram is "
    "its actual architecture; the table below it is the network <i>evaluated</i> at every state, "
    "so it stays directly comparable with the tabular agent's table.</p>"
    "<p><b>Reading it.</b> If the DQN's evaluated table and a tabular Q-learner's table disagree "
    "sharply on the same match, that difference is the function approximation itself -- same "
    "algorithm family, different representation. loss is the last training-batch TD error.</p>"
))


def render_creature(agent: Any, coop_rate: float | None = None) -> dict[str, Any]:
    brain = agent.render_brain()
    kind = brain.get("kind")
    if kind == "qlearning":
        body = (_qtable_html(brain, "q_table") + _policy_vs_behavior_html(brain, coop_rate)
                + _help_trigger("qlearning"))
    elif kind == "dqn":
        net = f"<p class='meta'>net {'-'.join(map(str, brain.get('layers', [])))} " \
             f"&middot; loss={brain.get('last_loss', 0):.4f} &middot; {brain.get('device')}</p>"
        body = (net + _dqn_network_html(agent) + _qtable_html(brain, "q_values")
                + _policy_vs_behavior_html(brain, coop_rate) + _help_trigger("dqn"))
    elif kind == "fep":
        body = _fep_html(brain) + _help_trigger("fep")
    elif kind == "markov_brain":
        body = _markov_brain_html(brain)
    elif kind == "classic":
        body = _classic_html(brain)
    else:
        body = f"<pre>{brain}</pre>"
    meta = KIND_META.get(kind, {"glyph": "glyph-classic", "label": kind, "color": "#5c5346"})
    return {"name": agent.name, "kind": kind, "mode": getattr(agent, "training_mode", "?"),
           "body": body, "glyph": meta["glyph"], "color": meta["color"], "mascot": _mascot_svg(kind)}


# --- "Horsey Lab" gamified-theme mascots: one cute face per kind, CSS-hidden under the
# Scientific theme (see base.html's .creature-mascot rule) and shown under Gamified. Shape per
# kind matches the architecture it stands for (table/network/animat/belief-funnel); edge color is
# fixed (not theme-driven) since these only ever render on the bright Gamified background. ------

_MASCOT_EDGE = "#2b2440"


def _mascot_svg(kind: str) -> str:
    e = _MASCOT_EDGE
    if kind == "qlearning":
        return (
            "<svg viewBox='0 0 100 100'>"
            f"<rect x='16' y='16' width='68' height='68' rx='28' fill='var(--piece-qlearning)' stroke='{e}' stroke-width='5'/>"
            f"<text x='50' y='27' text-anchor='middle' dominant-baseline='central' font-family='var(--font-display)' font-size='17' font-weight='700' fill='{e}'>Q</text>"
            f"<circle cx='35' cy='46' r='7.5' fill='#fff' stroke='{e}' stroke-width='3'/>"
            f"<circle cx='65' cy='46' r='7.5' fill='#fff' stroke='{e}' stroke-width='3'/>"
            f"<circle cx='38.5' cy='46' r='3' fill='{e}'/><circle cx='61.5' cy='46' r='3' fill='{e}'/>"
            f"<path d='M38 66 Q50 74 62 66' fill='none' stroke='{e}' stroke-width='4.5' stroke-linecap='round'/>"
            "<line x1='67' y1='72' x2='74' y2='81' stroke='#ff8fa3' stroke-width='10' stroke-linecap='round'/>"
            "</svg>"
        )
    if kind == "dqn":
        return (
            "<svg viewBox='0 0 100 100'>"
            f"<rect x='18' y='18' width='64' height='64' rx='18' fill='var(--piece-dqn)' stroke='{e}' stroke-width='5' transform='rotate(45 50 50)'/>"
            f"<text x='50' y='26' text-anchor='middle' dominant-baseline='central' font-family='var(--font-display)' font-size='17' font-weight='700' fill='{e}'>D</text>"
            f"<rect x='29' y='41' width='19' height='15' rx='4' fill='none' stroke='{e}' stroke-width='3.5'/>"
            f"<rect x='52' y='41' width='19' height='15' rx='4' fill='none' stroke='{e}' stroke-width='3.5'/>"
            f"<line x1='48' y1='48.5' x2='52' y2='48.5' stroke='{e}' stroke-width='3.5'/>"
            f"<circle cx='38.5' cy='48.5' r='2.8' fill='{e}'/><circle cx='61.5' cy='48.5' r='2.8' fill='{e}'/>"
            f"<path d='M38 69 Q50 74 62 69' fill='none' stroke='{e}' stroke-width='4.5' stroke-linecap='round'/>"
            "</svg>"
        )
    if kind == "markov_brain":
        return (
            "<svg viewBox='0 0 100 100'>"
            f"<rect x='32' y='80' width='14' height='17' rx='6' fill='var(--piece-markov_brain)' stroke='{e}' stroke-width='4'/>"
            f"<rect x='54' y='80' width='14' height='17' rx='6' fill='var(--piece-markov_brain)' stroke='{e}' stroke-width='4'/>"
            f"<circle cx='50' cy='52' r='34' fill='var(--piece-markov_brain)' stroke='{e}' stroke-width='5'/>"
            f"<circle cx='24' cy='28' r='11' fill='var(--piece-markov_brain)' stroke='{e}' stroke-width='4'/>"
            f"<circle cx='76' cy='28' r='11' fill='var(--piece-markov_brain)' stroke='{e}' stroke-width='4'/>"
            f"<text x='50' y='29' text-anchor='middle' dominant-baseline='central' font-family='var(--font-display)' font-size='16' font-weight='700' fill='{e}'>M</text>"
            f"<circle cx='37' cy='48' r='7.3' fill='#fff' stroke='{e}' stroke-width='3'/>"
            f"<circle cx='63' cy='48' r='7.3' fill='#fff' stroke='{e}' stroke-width='3'/>"
            f"<circle cx='37' cy='49' r='3' fill='{e}'/><circle cx='63' cy='49' r='3' fill='{e}'/>"
            f"<path d='M48 66 Q50 81 55 72' fill='#ff8fa3' stroke='{e}' stroke-width='3'/>"
            "</svg>"
        )
    if kind == "fep":
        return (
            "<svg viewBox='0 0 100 100'>"
            f"<path d='M30,18 L70,18 Q85,18 77.6,31.1 L57.4,67 Q50,80 42.6,67 L22.4,31.1 Q15,18 30,18 Z' fill='var(--piece-fep)' stroke='{e}' stroke-width='5'/>"
            f"<path d='M46 26 L46 39 M46 26 L54 26 M46 32 L52 32' fill='none' stroke='{e}' stroke-width='2.6' stroke-linecap='round'/>"
            f"<path d='M31 44 q7 -4.5 13 0' fill='none' stroke='{e}' stroke-width='2.6' stroke-linecap='round'/>"
            f"<path d='M56 44 q7 -4.5 13 0' fill='none' stroke='{e}' stroke-width='2.6' stroke-linecap='round'/>"
            f"<circle cx='37' cy='51' r='6.2' fill='#fff' stroke='{e}' stroke-width='3'/>"
            f"<circle cx='63' cy='51' r='6.2' fill='#fff' stroke='{e}' stroke-width='3'/>"
            f"<circle cx='37' cy='51' r='2.9' fill='none' stroke='var(--psy)' stroke-width='1.2' opacity='.8'/>"
            f"<circle cx='63' cy='51' r='2.9' fill='none' stroke='var(--psy)' stroke-width='1.2' opacity='.8'/>"
            f"<circle cx='37' cy='51' r='1.5' fill='{e}'/><circle cx='63' cy='51' r='1.5' fill='{e}'/>"
            f"<path d='M41 60 Q50 63 59 60' fill='none' stroke='{e}' stroke-width='4' stroke-linecap='round'/>"
            "</svg>"
        )
    if kind == "classic":
        return (
            "<svg viewBox='0 0 100 100'>"
            f"<rect x='36' y='38' width='28' height='24' rx='6' fill='var(--piece-classic)' stroke='{e}' stroke-width='5'/>"
            f"<line x1='50' y1='38' x2='50' y2='28' stroke='{e}' stroke-width='3' stroke-linecap='round'/>"
            f"<circle cx='50' cy='26' r='3' fill='{e}'/>"
            f"<line x1='37' y1='40' x2='17' y2='19' stroke='{e}' stroke-width='4' stroke-linecap='round'/>"
            f"<line x1='63' y1='40' x2='83' y2='19' stroke='{e}' stroke-width='4' stroke-linecap='round'/>"
            f"<line x1='37' y1='60' x2='17' y2='81' stroke='{e}' stroke-width='4' stroke-linecap='round'/>"
            f"<line x1='63' y1='60' x2='83' y2='81' stroke='{e}' stroke-width='4' stroke-linecap='round'/>"
            f"<circle cx='17' cy='19' r='9' fill='var(--panel)' stroke='{e}' stroke-width='3'/>"
            f"<circle cx='83' cy='19' r='9' fill='var(--panel)' stroke='{e}' stroke-width='3'/>"
            f"<circle cx='17' cy='81' r='9' fill='var(--panel)' stroke='{e}' stroke-width='3'/>"
            f"<circle cx='83' cy='81' r='9' fill='var(--panel)' stroke='{e}' stroke-width='3'/>"
            f"<path d='M10 19 L24 19 M17 12 L17 26' stroke='{e}' stroke-width='2' opacity='.6'/>"
            f"<path d='M76 19 L90 19 M83 12 L83 26' stroke='{e}' stroke-width='2' opacity='.6'/>"
            f"<path d='M10 81 L24 81 M17 74 L17 88' stroke='{e}' stroke-width='2' opacity='.6'/>"
            f"<path d='M76 81 L90 81 M83 74 L83 88' stroke='{e}' stroke-width='2' opacity='.6'/>"
            "<circle cx='50' cy='50' r='4' fill='var(--sun)'/>"
            "</svg>"
        )
    return ""


def _fep_html(brain: dict) -> str:
    belief = brain.get("belief", [])
    labels = brain.get("state_labels") or [f"s{i}" for i in range(len(belief))]
    bars = "".join(
        f"<div class='barrow'><span class='lab'>{lab}</span>"
        f"<span class='bar'><span class='fill' style='width:{p * 100:.1f}%'></span></span>"
        f"<span class='val'>{p:.2f}</span></div>"
        for lab, p in zip(labels, belief)
    )
    probs = brain.get("action_probs", {})
    footer = (f"<p class='meta'>E[others cooperating]={brain.get('expected_others', 0):.2f} &middot; "
             f"P(Cooperate)={probs.get('Cooperate', 0):.2f} &middot; "
             f"P(Defect)={probs.get('Defect', 0):.2f} &middot; "
             f"reciprocity={brain.get('reciprocity', 0):g}</p>")
    return f"<div class='beliefs'>{bars}</div>{footer}"


# --- cooperation-rate chart, as a plain server-rendered SVG polyline (no JS/CDN needed) ----------

def _polyline_svg(series: list[float], width: int = 720, height: int = 140, color: str = "#2ecc71",
                  css_class: str = "coopchart") -> str:
    if not series:
        return ""
    pts = [
        f"{(idx / max(len(series) - 1, 1)) * width:.1f},{height - v * height:.1f}"
        for idx, v in enumerate(series)
    ]
    gridlines = "".join(
        f"<line x1='0' y1='{height * frac:.1f}' x2='{width}' y2='{height * frac:.1f}' "
        f"stroke='#ffffff30' stroke-width='1'/>" for frac in (0.0, 0.5, 1.0)
    )
    return (f"<svg viewBox='0 0 {width} {height}' class='{css_class}'>"
           f"{gridlines}<polyline points='{' '.join(pts)}' fill='none' stroke='{color}' "
           f"stroke-width='2'/></svg>")


def cooperation_svg(coop_series: np.ndarray, width: int = 720, height: int = 140, bins: int = 150) -> str:
    n = len(coop_series)
    if n == 0:
        return ""
    bin_size = max(n // bins, 1)
    binned = [float(np.mean(coop_series[i:i + bin_size])) for i in range(0, n, bin_size)]
    return _polyline_svg(binned, width, height)


def fitness_svg(history: list[dict], width: int = 720, height: int = 160) -> str:
    if not history:
        return ""
    best = [h["best"] for h in history]
    avg = [h["avg"] for h in history]
    vmax = max(max(best), max(avg), 1e-9)
    vmin = min(min(best), min(avg), 0.0)
    span = (vmax - vmin) or 1.0

    def pts(series):
        return [
            f"{(idx / max(len(series) - 1, 1)) * width:.1f},{height - ((v - vmin) / span) * height:.1f}"
            for idx, v in enumerate(series)
        ]
    return (f"<svg viewBox='0 0 {width} {height}' class='coopchart'>"
           f"<polyline points='{' '.join(pts(avg))}' fill='none' stroke='#6ec6ff' stroke-width='2'/>"
           f"<polyline points='{' '.join(pts(best))}' fill='none' stroke='#2ecc71' stroke-width='2'/>"
           f"</svg><p class='meta'><span style='color:#2ecc71'>&#9644;</span> best fitness &middot; "
           f"<span style='color:#6ec6ff'>&#9644;</span> avg fitness</p>")


# --- CSV / ZIP export helpers: every chart and table on the results page, downloadable in one
# click as real files (not screenshots), so they can be dropped directly into a paper. -----------

def _text_download_href(text: str, mime: str = "text/plain") -> str:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"data:{mime};charset=utf-8;base64,{encoded}"


def _metrics_csv(metrics: dict[str, Any], metric_meta: list[tuple[str, str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["metric", "label", "value"])
    for key, label in metric_meta:
        if key in metrics and isinstance(metrics[key], (int, float)):
            writer.writerow([key, label, metrics[key]])
    return buf.getvalue()


def _leaderboard_csv(leaderboard: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["agent", "kind", "cumulative_payoff"])
    for row in leaderboard:
        writer.writerow([row["name"], row["kind"], row["payoff"]])
    return buf.getvalue()


def _mi_te_csv(metrics: dict[str, Any], roster: list[Any]) -> str:
    """One CSV combining both information-theoretic breakdowns (mutual information per agent,
    transfer entropy per pair with its surrogate p-value) -- empty string if neither was computed
    this run, so the ZIP simply omits the file rather than including an empty/misleading one."""
    mi = metrics.get("mutual_information_detail")
    te = metrics.get("transfer_entropy_detail")
    pi = metrics.get("predictive_information_detail")
    gd = metrics.get("graph_detail")
    if not mi and not te:
        return ""
    buf = io.StringIO()
    writer = csv.writer(buf)
    if mi:
        writer.writerow(["mutual_information_and_predictive_information"])
        writer.writerow(["agent", "kind", "I(obs;action)_bits", "PI(past;future)_bits"])
        for i, bits in enumerate(mi["bits_by_agent"]):
            pi_bits = pi["bits_by_agent"][i] if pi else ""
            writer.writerow([roster[i].name, roster[i].kind, bits, pi_bits])
        writer.writerow([])
    if te:
        writer.writerow(["transfer_entropy"])
        writer.writerow(["source", "target", "TE_bits", "p_value", "significant_p<0.05"])
        for (i, j), d in te["by_pair"].items():
            writer.writerow([roster[i].name, roster[j].name, d["bits"], d["p_value"],
                             d["p_value"] < 0.05])
        writer.writerow([])
    if gd:
        writer.writerow(["co_cooperation_graph"])
        writer.writerow(["agent_a", "agent_b", "weight"])
        for e in gd["coop_edges"]:
            writer.writerow([roster[e["i"]].name, roster[e["j"]].name, e["weight"]])
        writer.writerow([])
        writer.writerow(["influence_graph_significant_te_edges"])
        writer.writerow(["source", "target", "TE_bits", "p_value"])
        for e in gd["influence_edges"]:
            writer.writerow([roster[e["source"]].name, roster[e["target"]].name,
                             e["bits"], e["p_value"]])
    return buf.getvalue()


def _build_results_zip(chart_svg: str, metrics_csv: str, leaderboard_csv: str, mi_te_csv: str,
                       console_log: str) -> str:
    """Every chart/CSV/log on the results page, bundled into one ZIP, returned as a downloadable
    data URI -- no server-side temp file or extra route needed, matching how the single-chart SVG
    download already works (embed the file's own bytes directly in the link)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if chart_svg:
            zf.writestr("cooperation_rate.svg", chart_svg)
        if metrics_csv:
            zf.writestr("metrics_summary.csv", metrics_csv)
        if leaderboard_csv:
            zf.writestr("leaderboard.csv", leaderboard_csv)
        if mi_te_csv:
            zf.writestr("mutual_information_transfer_entropy.csv", mi_te_csv)
        if console_log:
            zf.writestr("log.txt", console_log)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:application/zip;base64,{encoded}"


# --- routes: single match ------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def form():
    return render_template(
        "form.html", kinds=_KIND_ORDER, kind_meta=KIND_META,
        classic_strategies=_CLASSIC_STRATEGIES, metric_meta=_METRIC_META,
        metric_roadmap=_METRIC_ROADMAP, game_roadmap=_GAME_ROADMAP,
        game_choices=_GAME_CHOICES, reward_rules=_REWARD_RULE_CHOICES,
        advanced_params=_ADVANCED_PARAMS,
    )


@app.route("/run", methods=["POST"])
def run():
    f = request.form
    if f.get("batch_mode") == "on":
        return _run_batch(f)
    result = _execute_single_run(f)
    if isinstance(result, tuple):
        message, status = result
        return render_template("error.html", message=message), status
    return render_template("results.html", **result)


def _execute_single_run(f) -> dict[str, Any] | tuple[str, int]:
    """Runs one match end-to-end and returns the kwargs `results.html` needs, or an
    `(error_message, http_status)` pair. Factored out of the `/run` route so batch mode
    (`_run_batch`) can call it repeatedly with one form field swept, without duplicating the
    pipeline."""
    game_kind = f.get("game_kind", "public_goods")
    if game_kind not in {key for key, _ in _GAME_CHOICES}:
        return f"Unknown game {game_kind!r}.", 400

    rounds = int(f.get("rounds", 1500))
    seed = int(f.get("seed", 0))
    mpcr_raw = f.get("mpcr", "").strip()
    mpcr = float(mpcr_raw) if mpcr_raw else None
    log_every = max(rounds // int(f.get("log_lines", 50) or 50), 1)
    do_repo = f.get("record_repo") == "on"
    do_nash = f.get("compute_nash") == "on"
    do_phi = f.get("compute_phi") == "on"
    show_metrics = set(f.getlist("show_metrics")) or {k for k, _ in _METRIC_META}

    rows_kind = f.getlist("row_kind[]")
    rows_count = f.getlist("row_count[]")
    rows_classic = f.getlist("row_classic_strategy[]")
    rows_recip = f.getlist("row_reciprocity[]")
    rows_mh = f.getlist("row_markov_hidden[]")
    rows_ed = f.getlist("row_epsilon_decay[]")
    rows_em = f.getlist("row_epsilon_min[]")
    rows_gcid = f.getlist("row_genome_cid[]")
    # One parallel list per (kind, advanced-param) pair -- see _ADVANCED_PARAMS. Row N's own kind
    # picks out only the entries relevant to it; the rest are simply unused for that row.
    adv_lists: dict[tuple[str, str], list[str]] = {
        (kind, pname): f.getlist(f"row_adv_{kind}_{pname}[]")
        for kind, params in _ADVANCED_PARAMS.items() for pname, _default, _help in params
    }

    def _count(s):
        try:
            return int(s or 0)
        except ValueError:
            return 0

    n_agents = sum(_count(c) for k, c in zip(rows_kind, rows_count) if k)
    if n_agents < 2:
        return "You need at least 2 agents in total. Add rows to the roster.", 400

    try:
        if game_kind == "public_goods":
            game = PublicGoodsGame(n_agents=n_agents, rounds=rounds, mpcr=mpcr)
        else:
            # Preset first, then any explicit override from the form, so the published
            # configurations stay one click away while every axis is still reachable.
            overrides: dict[str, Any] = {}
            for field, cast in (("num_positions", int), ("memory_episodes", int),
                                ("episode_max_rounds", int), ("full_reward", float)):
                raw = (f.get(f"cong_{field}") or "").strip()
                if raw:
                    overrides[field] = cast(raw)
            rule = (f.get("cong_reward_rule") or "").strip()
            if rule:
                overrides["reward_rule"] = rule
                overrides["collapse_at_full"] = f.get("cong_collapse_at_full") == "on"
            game = congestion_from_preset(game_kind.split(":", 1)[1], n_agents, **overrides)
    except ValueError as exc:
        return str(exc), 400

    roster: list[Any] = []
    i = 0
    for row_idx, (kind, count_s, strat, recip_s, mh_s, ed_s, em_s, gcid) in enumerate(zip(
        rows_kind, rows_count, rows_classic, rows_recip, rows_mh, rows_ed, rows_em, rows_gcid
    )):
        if not kind or _count(count_s) <= 0:
            continue
        eps_decay = float(ed_s or 0.9995)
        eps_min = float(em_s or 0.02)
        extra: dict[str, Any] = {}
        for pname, default, _help in _ADVANCED_PARAMS.get(kind, []):
            values = adv_lists.get((kind, pname), [])
            raw = values[row_idx] if row_idx < len(values) else ""
            extra[pname] = type(default)(raw) if str(raw).strip() else default
        for _ in range(_count(count_s)):
            if kind == "markov_brain" and gcid.strip():
                agent = _load_genome_agent(gcid.strip(), i, game, seed)
                if agent is None:
                    return (f"Genome CID '{gcid.strip()}' was not found or doesn't match this "
                           f"game (it may have evolved for a different n_states/n_actions)."), 400
                roster.append(agent)
            else:
                roster.append(_make_agent(kind, i, game, seed, strat, float(recip_s or 0.0),
                                          int(mh_s or 2), eps_decay, eps_min, extra=extra))
            i += 1

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _RESULTS_DIR / f"webui_n{n_agents}_{stamp}.jsonl"

    console = LiveConsole()
    console.use_emoji, console.coop_sym, console.defect_sym = True, "\U0001f7e2", "\U0001f534"
    snap_idx = next((i for i, a in enumerate(roster) if a.kind in ("qlearning", "dqn", "fep")), 0)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with EventLog(path=log_path) as log:
            records = run_match(
                game, roster, rounds=rounds, seed=seed, eventlog=log, console=console,
                log_every=log_every, snapshot_every=max(rounds // 3, 1), snapshot_agent=snap_idx,
            )
        print()
        shown: set[str] = set()
        for ag in roster:
            if ag.kind in ("qlearning", "dqn", "fep") and ag.kind not in shown:
                console.brain_snapshot(ag, title="final brain")
                shown.add(ag.kind)
    console_log = buf.getvalue()

    metrics = social.compute_all(records, game.max_welfare_per_round())
    metrics.update(information.compute_all(records, seed=seed))
    metrics.update(graph.compute_all(records, metrics["transfer_entropy_detail"]))

    # Temporal-fairness measures need a sequence of contests to look at. A single-stage game
    # produces one episode for the whole match, so there is nothing to alternate over and the
    # panel is simply absent rather than showing a confident zero.
    alt_detail = None
    episodes = records.get("episodes", [])
    if len(episodes) > 1:
        alt_detail = social_alt.compute_all(
            episodes, n_agents=n_agents, full_reward=getattr(game, "full_reward", 100.0))
        # The papers' Efficiency is per episode; social.py's is per round. Same word, different
        # quantity, so the alternation one is renamed rather than silently overwriting it.
        alt_detail["alt_efficiency"] = alt_detail.pop("efficiency")
        metrics.update({k: v for k, v in alt_detail.items() if not k.startswith("detail_")})
    per_agent_coop = records["actions"].mean(axis=0)
    creatures = [render_creature(a, coop_rate=float(per_agent_coop[i]))
                 for i, a in enumerate(roster)]
    chart_svg = cooperation_svg(metrics["cooperation_series"])

    leaderboard = sorted(
        zip(roster, metrics["cumulative_payoffs"]), key=lambda pair: pair[1], reverse=True
    )
    leaderboard = [{"name": a.name, "kind": a.kind, "payoff": float(p)} for a, p in leaderboard]

    epsilon_hints = []
    for kind, ed_s, em_s in zip(rows_kind, rows_ed, rows_em):
        if kind in ("qlearning", "dqn"):
            decay, emin = float(ed_s or 0.9995), float(em_s or 0.02)
            r = _epsilon_rounds_to_min(decay, emin)
            epsilon_hints.append({"kind": kind, "decay": decay, "emin": emin, "rounds_needed": r,
                                  "reached": rounds >= r})

    nash_html = None
    if do_nash:
        if not isinstance(game, PublicGoodsGame):
            # The solver builds its 2^n payoff table from the Public Goods Game's closed-form
            # payoff. An episodic game has no single-round normal form to enumerate, so this is
            # a genuine gap rather than a size limit.
            nash_html = ("(skipped: equilibrium detection is currently built on the Public Goods "
                         "Game's closed-form payoff and has no normal form for an episodic game)")
        elif n_agents <= _NASH_MAX_AGENTS:
            eqs = equilibrium.pure_nash_equilibria(game)
            nash_html = equilibrium.describe_equilibria(game, eqs)
        else:
            nash_html = f"(skipped: n_agents > {_NASH_MAX_AGENTS} makes the 2^n table too large)"

    phi_info = None
    markov_agents = [a for a in roster if getattr(a, "kind", None) == "markov_brain"]
    if do_phi and markov_agents:
        target = markov_agents[0]
        phi_result, autonomy = _compute_phi_sequential(target)
        phi_info = {"agent": target.name, "phi": phi_result["phi"], "autonomy": autonomy}

    repo_info, filter_info, genome_cids = None, None, []
    for a in markov_agents:
        genome_cids.append({"name": a.name, "cid": _save_genome(a)})

    if do_repo:
        game_desc = _config_game_desc(game)
        roster_desc = _roster_description(roster)
        feature_vector = _feature_vector(game, roster, rounds)
        ledger = Ledger(_REPO_ROOT)
        raw_hits = smart_filter_lookup(ledger, game_desc, roster_desc, DEFAULT_CODE_VERSION, rounds,
                                       seed, feature_vector, k=5,
                                       protocol=protocol_from_run(records))
        filter_info = {
            "exact": raw_hits["exact"],
            "similar": [
                {"content_cid": r["content_cid"][:12], "config_hash": r["config_hash"][:12],
                 "rounds": r["horizon"]["rounds"], "seed": r["seeds"]["master"],
                 "cooperation_rate": r["metrics_summary"].get("cooperation_rate"), "distance": dist}
                for r, dist in raw_hits["similar"]
            ],
        }
        record = record_experiment(_REPO_ROOT, game, roster, log_path, rounds=rounds, seed=seed,
                                   metrics=metrics, code_version=DEFAULT_CODE_VERSION,
                                   protocol=protocol_from_run(records))
        lineage_bits = [k for k, v in record["lineage"].items() if v]
        repo_info = {
            "config_hash": record["config_hash"], "content_cid": record["content_cid"],
            "contributor": record["contributor"][:16] + "...",
            "signature_valid": Ledger.verify_record(record),
            "lineage": ", ".join(lineage_bits) if lineage_bits else "none (first of its kind)",
        }

    metrics_csv = _metrics_csv(metrics, _METRIC_META)
    leaderboard_csv = _leaderboard_csv(leaderboard)
    mi_te_csv = _mi_te_csv(metrics, roster)
    zip_download = _build_results_zip(chart_svg, metrics_csv, leaderboard_csv, mi_te_csv, console_log)
    log_download = _text_download_href(console_log)

    return {
        "game": game, "roster": roster, "seed": seed, "rounds": rounds, "metrics": metrics,
        "show_metrics": show_metrics, "metric_meta": _METRIC_META, "creatures": creatures,
        "chart_svg": chart_svg, "console_log": console_log, "leaderboard": leaderboard,
        "nash_html": nash_html, "nash_help_trigger": _help_trigger("nash"), "phi_info": phi_info,
        "help_topics": _HELP_TOPICS,
        "alt_metrics": alt_detail, "alt_metric_meta": _ALT_METRIC_META,
        "alt_help_trigger": _help_trigger("alt") if alt_detail else "",
        "repo_info": repo_info, "filter_info": filter_info, "log_path": log_path.name,
        "epsilon_hints": epsilon_hints, "genome_cids": genome_cids,
        "zip_download": zip_download, "log_download": log_download,
    }


def _run_batch(f):
    """Batch/sweep mode: rerun `_execute_single_run` once per value of one swept field, each a
    small variation on the same base configuration, each recorded to the ledger like any other
    run. Kept deliberately simple: a fixed menu of sweepable fields (not arbitrary ones) so the
    override logic below stays a couple of lines instead of a general form-mutation engine."""
    param = f.get("batch_param", "rounds")
    values = [v.strip() for v in f.get("batch_values", "").split(",") if v.strip()]
    if not values:
        return render_template("error.html", message=(
            "Batch mode needs at least one value in the values list (comma-separated).")), 400

    rows = []
    for v in values:
        f2 = f.copy()
        if param == "row0_count":
            counts = f.getlist("row_count[]")
            if counts:
                f2.setlist("row_count[]", [v] + counts[1:])
        else:
            f2[param] = v
        result = _execute_single_run(f2)
        if isinstance(result, tuple):
            rows.append({"value": v, "error": result[0]})
            continue
        repo = result["repo_info"]
        rows.append({
            "value": v, "cooperation_rate": result["metrics"]["cooperation_rate"],
            "efficiency": result["metrics"]["efficiency"],
            "config_hash": repo["config_hash"][:12] if repo else None,
            "content_cid": repo["content_cid"][:12] if repo else None,
        })

    coop_series = [r["cooperation_rate"] for r in rows if "error" not in r]
    chart_svg = _polyline_svg(coop_series, color="#3d6b8a") if coop_series else ""
    return render_template("batch_result.html", param=param, rows=rows, chart_svg=chart_svg)


# --- genome hand-off (Smart-Filter-adjacent: reuse the CAS as a tiny genome store) ----------------

def _save_genome(agent: Any) -> str:
    package = {"W": agent.W.tolist(), "bias": agent.bias.tolist(), "n_hidden": agent.n_hidden,
              "n_sensor": agent.n_sensor, "n_motor": agent.n_motor}
    return ContentStore(_REPO_ROOT).put(package)


def _load_genome_agent(cid: str, i: int, game: PublicGoodsGame, seed: int):
    from ..agents.markov_brain import MarkovBrainAgent
    store = ContentStore(_REPO_ROOT)
    if not store.exists(cid):
        return None
    genome = store.get(cid)
    W = np.asarray(genome["W"])
    n_sensor_needed = max(1, (game.n_states - 1).bit_length())
    if genome["n_sensor"] != n_sensor_needed or W.shape[1] != genome["n_sensor"] + genome["n_hidden"] + genome["n_motor"]:
        return None
    return MarkovBrainAgent(
        name=f"Evolved-{i}", n_states=game.n_states, n_actions=game.n_actions,
        n_hidden=genome["n_hidden"], W=W, bias=np.asarray(genome["bias"]),
        seed=seed + 500 + i, start_state=game.start_state,
    )


# --- routes: evolutionary tab ---------------------------------------------------------------------

@app.route("/evolve", methods=["GET"])
def evolve_form():
    return render_template("evolve_form.html")


def _pretrain_representative(kind: str, n_agents: int, mpcr, seed: int, rounds: int):
    """Self-play a full match among `n_agents` copies of one online-learning kind, then return
    agent 0 -- already trained purely through the ordinary run_match/update() path, no special
    pretraining code. Used only to build a frozen baseline for the bake-off."""
    pretrain_game = PublicGoodsGame(n_agents=n_agents, rounds=rounds, mpcr=mpcr)
    roster = [_make_agent(kind, i, pretrain_game, seed, "MajorityTFT", 0.0, 2) for i in range(n_agents)]
    run_match(pretrain_game, roster, rounds=rounds, seed=seed)
    return roster[0]


def _run_bakeoff(n_agents: int, mpcr, best_genome, pretrain_rounds: int, eval_rounds: int, seed: int) -> dict:
    """One evaluation match: the GA's best genome, a frozen self-play-pretrained Q-learner and
    DQN, a fresh FEP (it has no cross-match parameters to pretrain -- see agents/fep.py), and an
    AllD baseline, cycled to fill n_agents seats. This is comparison Mode 1 ("bake-off"): every
    architecture reaches its own natural convergence process *separately*, then all are frozen (or
    fresh, for FEP/classic) for one shared match -- the clean way to compare a population-evolved
    architecture against online learners without the confound of simultaneous co-adaptation.
    """
    cycle = ["markov_brain", "qlearning", "dqn", "fep", "classic"]
    kinds = [cycle[i % len(cycle)] for i in range(n_agents)]
    eval_game = PublicGoodsGame(n_agents=n_agents, rounds=eval_rounds, mpcr=mpcr)

    roster: list[Any] = []
    for i, kind in enumerate(kinds):
        if kind == "markov_brain":
            roster.append(best_genome.clone(name=f"Evolved-{i}", seed=seed + 900 + i))
        elif kind in ("qlearning", "dqn"):
            trained = _pretrain_representative(kind, n_agents, mpcr, seed + 700 + i, pretrain_rounds)
            roster.append(_Frozen(trained, name=f"{trained.name} (frozen)"))
        elif kind == "fep":
            roster.append(_make_agent("fep", i, eval_game, seed + 800 + i, "MajorityTFT", 0.0, 2))
        else:
            roster.append(AllD(f"AllD-{i}"))

    records = run_match(eval_game, roster, rounds=eval_rounds, seed=seed + 999)
    metrics = social.compute_all(records, eval_game.max_welfare_per_round())
    metrics.update(information.compute_all(records, seed=seed + 999))
    metrics.update(graph.compute_all(records, metrics["transfer_entropy_detail"]))
    leaderboard = sorted(zip(roster, metrics["cumulative_payoffs"]), key=lambda p: p[1], reverse=True)
    return {
        "kinds": kinds, "cooperation_rate": metrics["cooperation_rate"],
        "leaderboard": [{"name": a.name, "kind": getattr(a, "kind", "?"), "payoff": float(p)}
                        for a, p in leaderboard],
        "pretrain_rounds": pretrain_rounds, "eval_rounds": eval_rounds,
    }


@app.route("/evolve", methods=["POST"])
def evolve_run():
    f = request.form
    n_agents = int(f.get("n_agents_evo", 4))
    mpcr_raw = f.get("mpcr", "").strip()
    mpcr = float(mpcr_raw) if mpcr_raw else None
    cfg = EvolutionConfig(
        population_size=int(f.get("population_size", 20)),
        generations=int(f.get("generations", 30)),
        match_rounds=int(f.get("match_rounds", 200)),
        matches_per_generation=int(f.get("matches_per_generation", 3)),
        n_hidden=int(f.get("n_hidden", 2)),
        tournament_size=int(f.get("tournament_size", 3)),
        mutation_rate=float(f.get("mutation_rate", 0.15)),
        mutation_scale=float(f.get("mutation_scale", 0.4)),
        elite_count=int(f.get("elite_count", 2)),
        seed=int(f.get("seed", 0)),
    )
    do_bakeoff = f.get("bakeoff") == "on"
    pretrain_rounds = int(f.get("pretrain_rounds", 4000) or 4000)
    do_phi = f.get("compute_phi") == "on"

    if cfg.population_size % n_agents != 0:
        return render_template("error.html", message=(
            f"population_size ({cfg.population_size}) must divide evenly by n_agents "
            f"({n_agents}) -- the population is split into groups of n_agents per match.")), 400

    def game_factory():
        return PublicGoodsGame(n_agents=n_agents, rounds=cfg.match_rounds, mpcr=mpcr)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"Evolving: population={cfg.population_size} generations={cfg.generations} "
             f"match_rounds={cfg.match_rounds} n_hidden={cfg.n_hidden} seed={cfg.seed}")
        result = evolve(game_factory, cfg)
    evo_log = buf.getvalue()

    best = result.best
    best.act(best.start_state)  # settle into one concrete, reachable state (see run_evolution.py)

    phi_info = None
    if do_phi:
        phi_result, autonomy = _compute_phi_sequential(best, n_t=3)
        phi_info = {"phi": phi_result["phi"], "autonomy": autonomy,
                   "max_autonomy": best.n_hidden + best.n_motor}

    bakeoff = None
    if do_bakeoff:
        bakeoff = _run_bakeoff(n_agents, mpcr, best, pretrain_rounds,
                               eval_rounds=cfg.match_rounds, seed=cfg.seed)

    genome_cid = _save_genome(best)

    return render_template(
        "evolve_result.html", cfg=cfg, n_agents=n_agents,
        best_fitness=float(result.fitness.max()), avg_fitness=float(result.fitness.mean()),
        fitness_svg=fitness_svg(result.fitness_history), evo_log=evo_log, phi_info=phi_info,
        bakeoff=bakeoff, genome_cid=genome_cid,
    )


# --- Analytics: flexible querying/correlation over every recorded run ------------------------

_ANALYTICS_SIMPLE_FIELDS = [
    ("n_agents", "Number of agents"), ("mpcr", "MPCR"), ("cost", "Cost"),
    ("rounds", "Rounds"), ("seed", "Seed"),
]


def _analytics_field_options() -> list[tuple[str, str]]:
    """(field_key, label) pairs offered in the filter dropdown: simple game/run fields, every
    scalar metric, and every per-kind hyperparameter actually stored in the ledger (see
    repository/record.py's `_PARAM_ATTRS`, the authoritative list of what `roster[i]["params"]`
    can contain -- reused here rather than duplicated so this list can never drift out of sync
    with what a record actually stores)."""
    options = list(_ANALYTICS_SIMPLE_FIELDS)
    options += [(f"metric:{k}", label) for k, label in _METRIC_META]
    for kind, params in _PARAM_ATTRS.items():
        for p in params:
            options.append((f"roster:{kind}:{p}", f"{KIND_META[kind]['label']} · {p}"))
    return options


def _analytics_row(record: dict[str, Any]) -> dict[str, Any]:
    game = record.get("game", {})
    roster = record.get("roster", [])
    metrics = record.get("metrics_summary", {})
    kind_counts: dict[str, int] = {}
    for a in roster:
        kind_counts[a["kind"]] = kind_counts.get(a["kind"], 0) + 1
    return {
        "config_hash": record["config_hash"], "content_cid": record["content_cid"],
        "timestamp": record.get("timestamp", ""),
        "n_agents": game.get("n_agents"), "mpcr": game.get("mpcr"), "cost": game.get("cost"),
        "rounds": record.get("horizon", {}).get("rounds"),
        "seed": record.get("seeds", {}).get("master"),
        "kind_counts": kind_counts,
        "kind_summary": ", ".join(f"{v}x {k}" for k, v in kind_counts.items()),
        "roster": roster, "metrics": metrics, "lineage": record.get("lineage", {}),
    }


def _row_field_candidates(row: dict[str, Any], field: str) -> list[float]:
    """Numeric candidate value(s) for `field` on this row. `roster:<kind>:<param>` fields return
    one value per matching agent (empty if the roster has none of that kind, or the agent lacks
    that param) -- so filtering means "at least one agent of this kind satisfies the condition",
    the natural reading of a request like "runs with an RL agent whose epsilon_decay is above 0.8".
    """
    if field.startswith("metric:"):
        v = row["metrics"].get(field.split(":", 1)[1])
        return [v] if isinstance(v, (int, float)) else []
    if field.startswith("roster:"):
        _, kind, param = field.split(":", 2)
        return [a["params"][param] for a in row["roster"]
                if a["kind"] == kind and isinstance(a.get("params", {}).get(param), (int, float))]
    v = row.get(field)
    return [v] if isinstance(v, (int, float)) else []


_ANALYTICS_OPS = {
    ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b, "<": lambda a, b: a < b,
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
}


def _row_matches_filters(row: dict[str, Any], filters: list[dict[str, Any]]) -> bool:
    for f in filters:
        candidates = _row_field_candidates(row, f["field"])
        op = _ANALYTICS_OPS[f["op"]]
        if not any(op(v, f["value"]) for v in candidates):
            return False
    return True


def _row_matches_kind_filter(row: dict[str, Any], include_kinds: set[str]) -> bool:
    return not include_kinds or bool(include_kinds & set(row["kind_counts"].keys()))


def _format_tick(v: float) -> str:
    """Integer-looking values (n_agents, rounds, seed) print without decimals; everything else
    (metrics, hyperparameters) prints to 3 significant figures -- readable at both scales without
    a bare point number's false precision (e.g. `2` not `2.000`, but `0.00347` not `0.0`)."""
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.3g}"


def _scatter_svg(points: list[tuple[float, float]], x_label: str, y_label: str,
                 width: int = 720, height: int = 360, color: str = "#3d6b8a",
                 n_ticks: int = 5) -> tuple[str, str]:
    """Returns (inline_html, download_href): `inline_html` embeds the chart in the results page;
    `download_href` is a self-contained `data:image/svg+xml` URI of the *same* chart as a
    standalone SVG file (its own `xmlns`, no CSS-class dependency on this page's stylesheet), so a
    "Download SVG" link works as a real, portable vector figure suitable for a paper -- not a
    screenshot of the page.
    """
    if not points:
        return "", ""
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    xspan, yspan = (xmax - xmin) or 1.0, (ymax - ymin) or 1.0
    pad_left, pad_right, pad_top, pad_bottom = 60, 20, 30, 50

    def sx(x: float) -> float:
        return pad_left + (x - xmin) / xspan * (width - pad_left - pad_right)

    def sy(y: float) -> float:
        return height - pad_bottom - (y - ymin) / yspan * (height - pad_top - pad_bottom)

    def ticks(vmin: float, vmax: float) -> list[float]:
        if vmin == vmax:
            return [vmin]
        step = (vmax - vmin) / (n_ticks - 1)
        return [vmin + i * step for i in range(n_ticks)]

    x_ticks, y_ticks = ticks(xmin, xmax), ticks(ymin, ymax)

    gridlines = "".join(
        f"<line x1='{sx(t):.1f}' y1='{pad_top}' x2='{sx(t):.1f}' y2='{height - pad_bottom}' "
        f"stroke='#8884' stroke-dasharray='2,3'/>"
        f"<text x='{sx(t):.1f}' y='{height - pad_bottom + 16}' text-anchor='middle' "
        f"font-size='10' fill='currentColor'>{_format_tick(t)}</text>"
        for t in x_ticks
    ) + "".join(
        f"<line x1='{pad_left}' y1='{sy(t):.1f}' x2='{width - pad_right}' y2='{sy(t):.1f}' "
        f"stroke='#8884' stroke-dasharray='2,3'/>"
        f"<text x='{pad_left - 8}' y='{sy(t):.1f}' text-anchor='end' dominant-baseline='middle' "
        f"font-size='10' fill='currentColor'>{_format_tick(t)}</text>"
        for t in y_ticks
    )

    dots = "".join(
        f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='4' fill='{color}' fill-opacity='0.75'/>"
        for x, y in points
    )
    axes = (
        f"<line x1='{pad_left}' y1='{height - pad_bottom}' x2='{width - pad_right}' "
        f"y2='{height - pad_bottom}' stroke='#8888' stroke-width='1.5'/>"
        f"<line x1='{pad_left}' y1='{pad_top}' x2='{pad_left}' y2='{height - pad_bottom}' "
        f"stroke='#8888' stroke-width='1.5'/>"
    )
    title = (
        f"<text x='{width / 2}' y='16' text-anchor='middle' font-size='13' font-weight='700' "
        f"fill='currentColor'>{y_label} vs {x_label} (n={len(points)})</text>"
    )
    labels = (
        f"<text x='{width / 2}' y='{height - 10}' text-anchor='middle' font-size='12' "
        f"fill='currentColor'>{x_label}</text>"
        f"<text x='16' y='{height / 2}' text-anchor='middle' font-size='12' fill='currentColor' "
        f"transform='rotate(-90 16 {height / 2})'>{y_label}</text>"
    )
    body = f"{title}{gridlines}{axes}{dots}{labels}"
    # `currentColor` (used by title/labels/ticks above) inherits the CSS `color` of whatever
    # wraps the SVG. The dark chart background (`.scatterchart`, matching `.coopchart`) sits
    # inside a `.panel` whose text color is `--ink` (bright, meant to sit on the panel's own dark
    # background) -- inherited as-is, that text would blend into the equally-dark chart
    # background. Setting `style="color:..."` directly on the `<svg>` root fixes `currentColor`
    # locally, independent of the surrounding page: `--felt-text` inline, plain dark gray for the
    # standalone downloadable file (which sets its own white background below, deliberately
    # print/paper-appropriate regardless of the on-page theme).
    inline_html = (
        f"<svg viewBox='0 0 {width} {height}' class='scatterchart' style='color:#c7cbe6'>"
        f"{body}</svg>"
    )
    standalone_svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
        f"width='{width}' height='{height}' style='background:#fff;color:#222;font-family:sans-serif'>"
        f"{body}</svg>"
    )
    download_href = "data:image/svg+xml;charset=utf-8," + quote(standalone_svg)
    return inline_html, download_href


@app.route("/analytics", methods=["GET", "POST"])
def analytics():
    records = Ledger(_REPO_ROOT).load_all()
    rows = [_analytics_row(r) for r in records]

    include_kinds = set(request.values.getlist("include_kind"))
    filter_fields = request.values.getlist("filter_field[]")
    filter_ops = request.values.getlist("filter_op[]")
    filter_values = request.values.getlist("filter_value[]")
    filters: list[dict[str, Any]] = []
    for field, op, value in zip(filter_fields, filter_ops, filter_values):
        if not field or not value.strip():
            continue
        try:
            filters.append({"field": field, "op": op, "value": float(value)})
        except ValueError:
            continue

    filtered = [row for row in rows if _row_matches_kind_filter(row, include_kinds)
               and _row_matches_filters(row, filters)]

    chart_axis_options = list(_ANALYTICS_SIMPLE_FIELDS) + [(f"metric:{k}", l) for k, l in _METRIC_META]
    axis_labels = dict(chart_axis_options)
    chart_x = request.values.get("chart_x", "n_agents")
    chart_y = request.values.get("chart_y", "metric:cooperation_rate")
    points = []
    for row in filtered:
        xs, ys = _row_field_candidates(row, chart_x), _row_field_candidates(row, chart_y)
        if xs and ys:
            points.append((xs[0], ys[0]))
    chart_svg, chart_download = _scatter_svg(
        points, axis_labels.get(chart_x, chart_x), axis_labels.get(chart_y, chart_y),
    )

    # Pearson r needs at least 3 points and variance on both axes to mean anything; below that,
    # say so instead of printing a coefficient computed from a degenerate sample.
    correlation = None
    if len(points) >= 3:
        xs = np.array([p[0] for p in points], dtype=float)
        ys = np.array([p[1] for p in points], dtype=float)
        if xs.std() > 0 and ys.std() > 0:
            correlation = {"r": float(np.corrcoef(xs, ys)[0, 1]), "n": len(points)}

    return render_template(
        "analytics.html", rows=filtered, total_count=len(rows), kind_meta=KIND_META,
        field_options=_analytics_field_options(), chart_axis_options=chart_axis_options,
        include_kinds=include_kinds, chart_download=chart_download,
        active_filters=list(zip(filter_fields, filter_ops, filter_values)),
        chart_x=chart_x, chart_y=chart_y, chart_svg=chart_svg, correlation=correlation,
    )


# --- Parameter-space map: coverage over a declared grid + auto-fill of missing cells -------------

_SPACE_DEFAULTS = {
    "n": [2, 3, 4, 5, 6],
    "mpcr": [0.4, 0.5, 0.6, 0.75],
    "rounds": [500, 1500, 3000],
    "seeds": [0, 1, 2],
}

# markov_brain is deliberately absent from the default mixes: a random, un-evolved genome does not
# learn within a match (training_mode="evolutionary"), so a grid cell running one says nothing
# about the architecture -- evolve first on the Evolutionary tab, then study that genome via the
# Match tab's genome-CID hand-off instead.
_SPACE_MIXES: dict[str, dict[str, Any]] = {
    "all_qlearning": {"label": "All Q-learning", "counts": lambda n: {"qlearning": n}},
    "all_dqn": {"label": "All DQN", "counts": lambda n: {"dqn": n}},
    "all_fep": {"label": "All FEP", "counts": lambda n: {"fep": n}},
    "one_of_each": {"label": "One of each (Q, DQN, FEP, Classic) -- n=4 only",
                    "counts": lambda n: ({"qlearning": 1, "dqn": 1, "fep": 1, "classic": 1}
                                         if n == 4 else None)},
}


def _parse_num_list(raw: str | None, default: list, cast) -> list:
    if raw is None or not raw.strip():
        return list(default)
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            try:
                out.append(cast(tok))
            except ValueError:
                continue
    return out or list(default)


def _space_cells(ns: list[int], mpcrs: list[float], roundss: list[int], seeds: list[int],
                 mix_keys: list[str]) -> list[dict[str, Any]]:
    """Enumerate every cell of the declared grid. A cell is `valid` only when the mix is defined
    at that population size AND the mpcr satisfies the social-dilemma condition 1/n < mpcr < 1
    (the same constraint PublicGoodsGame itself enforces) -- invalid cells are shown as such on
    the map rather than silently dropped, so the user can see WHY that corner is empty."""
    cells = []
    for mix_key in mix_keys:
        mix = _SPACE_MIXES[mix_key]
        for n in ns:
            counts = mix["counts"](n)
            for mpcr in mpcrs:
                valid = counts is not None and (1.0 / n) < mpcr < 1.0
                for rounds in roundss:
                    for seed in seeds:
                        cells.append({"mix": mix_key, "n": n, "mpcr": mpcr, "rounds": rounds,
                                      "seed": seed, "valid": valid, "counts": counts})
    return cells


def _cell_covered(cell: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    """A ledger row covers a cell when it matches on every grid coordinate (n, mpcr, rounds, seed)
    and its roster has exactly the cell's kind composition. Hyperparameters are deliberately NOT
    part of the match: the map's dimensions are the grid's coordinates, and a run with custom
    alpha at those coordinates is still a run at those coordinates."""
    for row in rows:
        if (row["n_agents"] == cell["n"] and row["rounds"] == cell["rounds"]
                and row["seed"] == cell["seed"] and row["mpcr"] is not None
                and abs(row["mpcr"] - cell["mpcr"]) < 1e-9
                and row["kind_counts"] == cell["counts"]):
            return True
    return False


def _run_cell(cell: dict[str, Any]) -> None:
    """Run one missing cell with default hyperparameters and record it to the ledger -- the lean
    pipeline (match + metrics + record), not the full results-page pipeline (no Nash/Phi/creature
    rendering: nobody is looking at this run's page, its purpose is to exist in the ledger so the
    Analytics page can query it)."""
    game = PublicGoodsGame(n_agents=cell["n"], rounds=cell["rounds"], mpcr=cell["mpcr"])
    seed = cell["seed"]
    roster: list[Any] = []
    i = 0
    for kind, cnt in cell["counts"].items():
        for _ in range(cnt):
            roster.append(_make_agent(kind, i, game, seed, "AllD", 0.0, 2))
            i += 1
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = _RESULTS_DIR / f"webui_space_n{cell['n']}_{stamp}.jsonl"
    with contextlib.redirect_stdout(io.StringIO()):
        with EventLog(path=log_path) as log:
            records = run_match(game, roster, rounds=cell["rounds"], seed=seed, eventlog=log)
    metrics = social.compute_all(records, game.max_welfare_per_round())
    metrics.update(information.compute_all(records, seed=seed))
    metrics.update(graph.compute_all(records, metrics["transfer_entropy_detail"]))
    record_experiment(_REPO_ROOT, game, roster, log_path, rounds=cell["rounds"], seed=seed,
                      metrics=metrics, code_version=DEFAULT_CODE_VERSION,
                      protocol=protocol_from_run(records))


@app.route("/spacemap", methods=["GET", "POST"])
def spacemap():
    v = request.values
    ns = _parse_num_list(v.get("grid_n"), _SPACE_DEFAULTS["n"], int)
    mpcrs = _parse_num_list(v.get("grid_mpcr"), _SPACE_DEFAULTS["mpcr"], float)
    roundss = _parse_num_list(v.get("grid_rounds"), _SPACE_DEFAULTS["rounds"], int)
    seeds = _parse_num_list(v.get("grid_seeds"), _SPACE_DEFAULTS["seeds"], int)
    mix_keys = [m for m in v.getlist("grid_mix") if m in _SPACE_MIXES] or list(_SPACE_MIXES)

    rows = [_analytics_row(r) for r in Ledger(_REPO_ROOT).load_all()]
    cells = _space_cells(ns, mpcrs, roundss, seeds, mix_keys)
    for c in cells:
        c["covered"] = c["valid"] and _cell_covered(c, rows)

    ran = 0
    if request.method == "POST" and request.form.get("action") == "run_missing":
        cap = max(1, min(int(request.form.get("cap", 5) or 5), 50))
        for c in cells:
            if ran >= cap:
                break
            if c["valid"] and not c["covered"]:
                _run_cell(c)
                c["covered"] = True
                ran += 1

    # Per-mix summary: an n x mpcr table whose cells aggregate over rounds x seeds.
    tables = []
    for mix_key in mix_keys:
        grid: dict[int, dict[float, dict[str, Any]]] = {}
        for c in cells:
            if c["mix"] != mix_key:
                continue
            slot = grid.setdefault(c["n"], {}).setdefault(
                c["mpcr"], {"covered": 0, "total": 0, "valid": c["valid"]})
            if c["valid"]:
                slot["total"] += 1
                slot["covered"] += int(c["covered"])
        tables.append({"key": mix_key, "label": _SPACE_MIXES[mix_key]["label"], "grid": grid})

    n_valid = sum(1 for c in cells if c["valid"])
    n_covered = sum(1 for c in cells if c["covered"])
    n_invalid = len(cells) - n_valid
    return render_template(
        "spacemap.html", tables=tables, ns=ns, mpcrs=mpcrs,
        grid_n=",".join(map(str, ns)), grid_mpcr=",".join(map(str, mpcrs)),
        grid_rounds=",".join(map(str, roundss)), grid_seeds=",".join(map(str, seeds)),
        mix_keys=mix_keys, all_mixes=[(k, m["label"]) for k, m in _SPACE_MIXES.items()],
        n_cells=len(cells), n_valid=n_valid, n_covered=n_covered,
        n_missing=n_valid - n_covered, n_invalid=n_invalid, ran=ran,
    )


if __name__ == "__main__":
    # debug/reloader off on purpose: Werkzeug's reloader re-execs this module in a second
    # process, and on Windows that confused PyPhi's spawn-based multiprocessing pool (used by
    # compute_phi's parallel cut evaluation) into leaking worker processes -- observed directly
    # as a MemoryError from the LP solver plus several orphaned "--multiprocessing-fork" python
    # processes after a request that had "compute_phi" checked. A single plain process avoids it.
    app.run(debug=False, use_reloader=False)
