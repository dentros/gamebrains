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

import contextlib
import io
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
from ..games.public_goods import PublicGoodsGame
from ..metrics import equilibrium, information, social
from ..repository.cas import ContentStore
from ..repository.ledger import Ledger
from ..repository.record import (
    DEFAULT_CODE_VERSION, _config_game_desc, _feature_vector, _PARAM_ATTRS, _roster_description,
    record_experiment,
)
from ..repository.smart_filter import lookup as smart_filter_lookup

app = Flask(__name__)


@app.context_processor
def _inject_active_tab():
    if request.path.startswith("/evolve"):
        return {"active_tab": "evolve"}
    if request.path.startswith("/analytics"):
        return {"active_tab": "analytics"}
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
]
# Roadmap placeholders: real code doesn't exist yet (see CLAUDE.md §5) -- shown disabled so the
# tool itself documents where the platform is going, not just what it does today. Transfer entropy
# and mutual information graduated out of this list (2026-07-17): metrics/information.py.
_METRIC_ROADMAP = [
    "Predictive information (planned)",
    "Graph-theoretic metrics (planned, networkx)",
]
_GAME_ROADMAP = ["Honey-Jar Game, formerly MBoE (coming soon)"]


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
        return "background:#eee"
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


_FEP_EXPLAIN = (
    "<details class='explain'><summary>&#8505; How to read this</summary>"
    "<p><b>Theory.</b> The FEP agent holds a categorical belief over how many other players are "
    "cooperating, updated by exact Bayesian filtering each round. It picks an action by minimizing "
    "expected free energy: a softmax over expected value, not reward-maximizing learning like "
    "Q-learning.</p>"
    "<p><b>Reading it.</b> Each bar is the probability the agent assigns to that many cooperators. "
    "E[others] is the expectation of that distribution. <code>reciprocity</code> controls how "
    "social the agent is: 0 is purely selfish (maximizes only its own payoff); higher values make "
    "it prefer cooperating when it believes others will too: a simple Theory-of-Mind rule, since "
    "what I expect others to do shapes what I do.</p></details>"
)

_NASH_EXPLAIN = (
    "<details class='explain'><summary>&#8505; How to read this</summary>"
    "<p><b>Theory.</b> Computed analytically (via pygambit) on the single-round stage game: which "
    "action profile is stable when no player benefits from unilaterally deviating, given mpcr/cost/n. "
    "It ignores repeated-game effects such as reputation and whatever the actual agents learned.</p>"
    "<p><b>Reading it.</b> This is where a fully rational, one-shot player would end up. Compare it "
    "with the cooperation rate this run actually reached: if your agents haven't converged there yet "
    "(e.g. epsilon is still high), they likely need more rounds.</p></details>"
)


def _markov_brain_html(brain: dict) -> str:
    state = brain["state"]
    ns, nh = brain["n_sensor"], brain["n_hidden"]
    boxes = "".join(
        f"<span class='bit {'on' if b else 'off'} {'sensor' if k < ns else ('hidden' if k < ns + nh else 'motor')}'>"
        f"{b}</span>"
        for k, b in enumerate(state)
    )
    return (f"<div class='mbrain'>{boxes}</div>"
           f"<p class='meta'>sensor={ns} hidden={nh} motor={brain['n_motor']} "
           f"<span class='legend'>(sensor / hidden / motor)</span>. A snapshot of the final "
           f"state, not something it learned during this match: training_mode is "
           f"\"evolutionary\", meaning it evolves between generations, not within a match "
           f"(see the Evolutionary tab).</p>")


def _classic_html(brain: dict) -> str:
    extra = f" (p={brain['p_cooperate']:g})" if "p_cooperate" in brain else ""
    return f"<p class='rule'><b>{brain['strategy']}</b>{extra}: {brain['rule']}</p>"


def render_creature(agent: Any) -> dict[str, Any]:
    brain = agent.render_brain()
    kind = brain.get("kind")
    if kind == "qlearning":
        body = _qtable_html(brain, "q_table")
    elif kind == "dqn":
        net = f"<p class='meta'>net {'-'.join(map(str, brain.get('layers', [])))} " \
             f"&middot; loss={brain.get('last_loss', 0):.4f} &middot; {brain.get('device')}</p>"
        body = net + _dqn_network_html(agent) + _qtable_html(brain, "q_values")
    elif kind == "fep":
        body = _fep_html(brain) + _FEP_EXPLAIN
    elif kind == "markov_brain":
        body = _markov_brain_html(brain)
    elif kind == "classic":
        body = _classic_html(brain)
    else:
        body = f"<pre>{brain}</pre>"
    meta = KIND_META.get(kind, {"glyph": "glyph-classic", "label": kind, "color": "#5c5346"})
    return {"name": agent.name, "kind": kind, "mode": getattr(agent, "training_mode", "?"),
           "body": body, "glyph": meta["glyph"], "color": meta["color"]}


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
        f"stroke='#3336' stroke-width='1'/>" for frac in (0.0, 0.5, 1.0)
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


# --- routes: single match ------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def form():
    return render_template(
        "form.html", kinds=_KIND_ORDER, kind_meta=KIND_META,
        classic_strategies=_CLASSIC_STRATEGIES, metric_meta=_METRIC_META,
        metric_roadmap=_METRIC_ROADMAP, game_roadmap=_GAME_ROADMAP,
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
    if f.get("game_kind", "public_goods") != "public_goods":
        return "That game isn't implemented yet.", 400

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
        game = PublicGoodsGame(n_agents=n_agents, rounds=rounds, mpcr=mpcr)
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
    creatures = [render_creature(a) for a in roster]
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
        if n_agents <= _NASH_MAX_AGENTS:
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
                                       seed, feature_vector, k=5)
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
                                   metrics=metrics, code_version=DEFAULT_CODE_VERSION)
        lineage_bits = [k for k, v in record["lineage"].items() if v]
        repo_info = {
            "config_hash": record["config_hash"], "content_cid": record["content_cid"],
            "contributor": record["contributor"][:16] + "...",
            "signature_valid": Ledger.verify_record(record),
            "lineage": ", ".join(lineage_bits) if lineage_bits else "none (first of its kind)",
        }

    return {
        "game": game, "roster": roster, "seed": seed, "rounds": rounds, "metrics": metrics,
        "show_metrics": show_metrics, "metric_meta": _METRIC_META, "creatures": creatures,
        "chart_svg": chart_svg, "console_log": console_log, "leaderboard": leaderboard,
        "nash_html": nash_html, "nash_explain": _NASH_EXPLAIN, "phi_info": phi_info,
        "repo_info": repo_info, "filter_info": filter_info, "log_path": log_path.name,
        "epsilon_hints": epsilon_hints, "genome_cids": genome_cids,
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
    # wraps the SVG. The felt-dark chart background (`.scatterchart`, matching `.coopchart`) sits
    # inside a light `.panel` whose text color is `--ink` (dark brown, meant for a light
    # background) -- inherited as-is, that text would be near-invisible against the dark chart
    # background. Setting `style="color:..."` directly on the `<svg>` root fixes `currentColor`
    # locally, independent of the surrounding page: `--felt-text` (light cream) inline, plain dark
    # gray for the standalone downloadable file (which sets its own white background below).
    inline_html = (
        f"<svg viewBox='0 0 {width} {height}' class='scatterchart' style='color:#e6e0c8'>"
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

    return render_template(
        "analytics.html", rows=filtered, total_count=len(rows), kind_meta=KIND_META,
        field_options=_analytics_field_options(), chart_axis_options=chart_axis_options,
        include_kinds=include_kinds, chart_download=chart_download,
        active_filters=list(zip(filter_fields, filter_ops, filter_values)),
        chart_x=chart_x, chart_y=chart_y, chart_svg=chart_svg,
    )


if __name__ == "__main__":
    # debug/reloader off on purpose: Werkzeug's reloader re-execs this module in a second
    # process, and on Windows that confused PyPhi's spawn-based multiprocessing pool (used by
    # compute_phi's parallel cut evaluation) into leaking worker processes -- observed directly
    # as a MemoryError from the LP solver plus several orphaned "--multiprocessing-fork" python
    # processes after a request that had "compute_phi" checked. A single plain process avoids it.
    app.run(debug=False, use_reloader=False)
