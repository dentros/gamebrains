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
from ..metrics import equilibrium, social
from ..repository.cas import ContentStore
from ..repository.ledger import Ledger
from ..repository.record import _config_game_desc, _feature_vector, _roster_description, record_experiment
from ..repository.smart_filter import lookup as smart_filter_lookup

app = Flask(__name__)

_GAMEBRAINS_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _GAMEBRAINS_ROOT / "results"
_REPO_ROOT = _GAMEBRAINS_ROOT / "repo_store"

_CLASSIC_STRATEGIES = ["AllC", "AllD", "Random", "MajorityTFT"]
_NASH_MAX_AGENTS = 10  # enumpure_solve builds a 2^n table -- keep this bounded in a web request

# One badge per cognitive architecture -- shared between the roster builder and the results
# page so a "kind" always looks the same wherever it appears. Purely cosmetic (§CLAUDE.md's
# "gamified & cute on top" goal); the technical kind name is always shown alongside it.
KIND_META = {
    "qlearning":    {"emoji": "\U0001f423", "label": "Q-learning",       "color": "#f1c40f"},
    "dqn":          {"emoji": "\U0001f916", "label": "Deep Q-Network",   "color": "#3498db"},
    "fep":          {"emoji": "\U0001f52e", "label": "FEP / Active Inference", "color": "#9b59b6"},
    "markov_brain": {"emoji": "\U0001f9ec", "label": "Markov-brain (evolutionary)", "color": "#2ecc71"},
    "classic":      {"emoji": "\U0001f4cf", "label": "Classic (fixed)",  "color": "#e67e22"},
}
_KIND_ORDER = ["qlearning", "dqn", "fep", "markov_brain", "classic"]

_METRIC_META = [
    ("cooperation_rate", "Cooperation rate"),
    ("efficiency", "Efficiency"),
    ("payoff_gini", "Payoff Gini"),
    ("action_entropy_bits", "Action entropy (bits)"),
]
# Roadmap placeholders: real code doesn't exist yet (see CLAUDE.md §5) -- shown disabled so the
# tool itself documents where the platform is going, not just what it does today.
_METRIC_ROADMAP = [
    "Transfer Entropy (TODO -- IDTxl/JIDT)",
    "Mutual Information (TODO -- IDTxl/dit)",
    "Predictive Information (TODO)",
    "Graph-theoretic μετρικές (TODO -- networkx)",
]
_GAME_ROADMAP = ["Honey-Jar Game (πρώην MBoE) -- έρχεται σύντομα"]


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

def _classic_agent(strategy: str, i: int, n: int, seed: int):
    if strategy == "AllC":
        return AllC(f"AllC {i}")
    if strategy == "AllD":
        return AllD(f"AllD {i}")
    if strategy == "Random":
        return RandomAgent(f"Random {i}", seed=seed + 400 + i)
    if strategy == "MajorityTFT":
        return MajorityTFT(f"TFT {i}", n_agents=n)
    raise ValueError(f"unknown classic strategy {strategy}")


def _make_agent(kind: str, i: int, game: PublicGoodsGame, seed: int, classic_strategy: str,
                reciprocity: float, markov_hidden: int, epsilon_decay: float = 0.9995,
                epsilon_min: float = 0.02):
    labels, actions = game.state_labels(), game.action_names
    if kind == "qlearning":
        return QLearningAgent(
            name=f"Q-learner {i}", n_states=game.n_states, n_actions=game.n_actions,
            alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
            seed=seed + 100 + i, state_labels=labels, action_labels=actions,
        )
    if kind == "dqn":
        from ..agents.dqn import DQNAgent
        return DQNAgent(
            name=f"DeepQ {i}", n_states=game.n_states, n_actions=game.n_actions,
            gamma=0.95, epsilon=1.0, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
            seed=seed + 200 + i, state_labels=labels, action_labels=actions,
        )
    if kind == "fep":
        from ..agents.fep import FEPAgent
        return FEPAgent(
            name=f"FEP {i}", n_agents=game.n_agents, mpcr=game.mpcr, cost=game.cost,
            reciprocity=reciprocity, seed=seed + 300 + i, start_state=game.start_state,
        )
    if kind == "markov_brain":
        from ..agents.markov_brain import MarkovBrainAgent
        return MarkovBrainAgent(
            name=f"MarkovBrain {i}", n_states=game.n_states, n_actions=game.n_actions,
            n_hidden=markov_hidden, seed=seed + 500 + i, start_state=game.start_state,
        )
    if kind == "classic":
        return _classic_agent(classic_strategy, i, game.n_agents, seed)
    raise ValueError(f"unknown kind {kind}")


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
           f"<p class='meta'>{n_params:,} παράμετροι &middot; κάθε κύκλος = κόμβος εξόδου, "
           f"χρώμα = μέση |βάρος| εισερχόμενων συνδέσεων (πιο σκούρο = πιο ισχυρό) &mdash; "
           f"η αρχιτεκτονική του δικτύου, όχι τα παραγόμενα Q-values (βλ. πίνακα από κάτω)</p>")


_FEP_EXPLAIN = (
    "<details class='explain'><summary>&#8505; Τι σημαίνει (θεωρία + πράξη)</summary>"
    "<p><b>Θεωρητικά:</b> ο FEP agent κρατάει μια κατηγορική πίστη (belief) πάνω στο &laquo;πόσοι "
    "από τους άλλους συνεργάζονται&raquo και την ενημερώνει με ακριβές Bayesian filtering κάθε "
    "γύρο. Επιλέγει ενέργεια ελαχιστοποιώντας το expected free energy (softmax πάνω στις "
    "αναμενόμενες αξίες) &mdash; active inference, όχι reward-μεγιστοποίηση με μάθηση όπως το "
    "Q-learning.</p>"
    "<p><b>Πρακτικά:</b> οι μπάρες δείχνουν πόσο πιθανό θεωρεί κάθε δυνατό αριθμό συνεργατών· το "
    "E[others] είναι η προσδοκία της. Το <code>reciprocity</code> ελέγχει πόσο &laquo;κοινωνικός"
    "&raquo; είναι: 0 = εγωιστικός (μεγιστοποιεί μόνο τη δική του αμοιβή), μεγαλύτερο = προτιμά να "
    "συνεργάζεται όταν πιστεύει ότι θα συνεργαστούν κι οι άλλοι &mdash; ένα πρώιμο μοντέλο "
    "Theory-of-Mind: «τι πιστεύω ότι θα κάνουν οι άλλοι, και πώς αλλάζει αυτό τι κάνω εγώ».</p></details>"
)

_NASH_EXPLAIN = (
    "<details class='explain'><summary>&#8505; Τι σημαίνει (θεωρία + πράξη)</summary>"
    "<p><b>Θεωρητικά:</b> υπολογίζεται αναλυτικά (pygambit) πάνω στο <i>μονο-γύρου</i> παιχνίδι: "
    "ποιο προφίλ ενεργειών είναι σταθερό όταν κανείς δεν έχει κίνητρο να αλλάξει μονομερώς τη "
    "στρατηγική του, δεδομένων mpcr/cost/n. Δεν λαμβάνει υπόψη επαναλαμβανόμενο παιχνίδι, φήμη, ή "
    "ό,τι έμαθαν οι πραγματικοί agents.</p>"
    "<p><b>Πρακτικά:</b> δείχνει πού θα κατέληγε ένας εντελώς ορθολογικός, μονο-γύρου παίκτης. "
    "Σύγκρινέ το με το πραγματικό cooperation rate του run σου &mdash; αν οι agents σου δεν έχουν "
    "συγκλίνει ακόμα εκεί (π.χ. λόγω υψηλού ε), σημαίνει ότι χρειάζονται περισσότερα rounds.</p></details>"
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
           f"<span class='legend'>(sensor / hidden / motor)</span> &mdash; στιγμιότυπο της "
           f"τελικής κατάστασης, ΟΧΙ κάτι που &laquo;έμαθε&raquo; μέσα σε αυτό το match "
           f"(training_mode=evolutionary: εξελίσσεται μεταξύ generations, όχι εντός ενός match "
           f"&mdash; βλ. tab &laquo;Evolutionary&raquo;)</p>")


def _classic_html(brain: dict) -> str:
    extra = f" (p={brain['p_cooperate']:g})" if "p_cooperate" in brain else ""
    return f"<p class='rule'><b>{brain['strategy']}</b>{extra} &mdash; {brain['rule']}</p>"


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
    meta = KIND_META.get(kind, {"emoji": "❓", "label": kind, "color": "#888"})
    return {"name": agent.name, "kind": kind, "mode": getattr(agent, "training_mode", "?"),
           "body": body, "emoji": meta["emoji"], "color": meta["color"]}


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
    )


@app.route("/run", methods=["POST"])
def run():
    f = request.form
    if f.get("game_kind", "public_goods") != "public_goods":
        return render_template("error.html", message="Αυτό το παιχνίδι δεν είναι ακόμα υλοποιημένο."), 400

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

    def _count(s):
        try:
            return int(s or 0)
        except ValueError:
            return 0

    n_agents = sum(_count(c) for k, c in zip(rows_kind, rows_count) if k)
    if n_agents < 2:
        return render_template("error.html", message=(
            "Χρειάζονται τουλάχιστον 2 agents συνολικά -- πρόσθεσε γραμμές στο roster.")), 400

    try:
        game = PublicGoodsGame(n_agents=n_agents, rounds=rounds, mpcr=mpcr)
    except ValueError as exc:
        return render_template("error.html", message=str(exc)), 400

    roster: list[Any] = []
    i = 0
    for kind, count_s, strat, recip_s, mh_s, ed_s, em_s, gcid in zip(
        rows_kind, rows_count, rows_classic, rows_recip, rows_mh, rows_ed, rows_em, rows_gcid
    ):
        if not kind or _count(count_s) <= 0:
            continue
        eps_decay = float(ed_s or 0.9995)
        eps_min = float(em_s or 0.02)
        for _ in range(_count(count_s)):
            if kind == "markov_brain" and gcid.strip():
                agent = _load_genome_agent(gcid.strip(), i, game, seed)
                if agent is None:
                    return render_template("error.html", message=(
                        f"Δεν βρέθηκε ή δεν ταιριάζει το genome CID '{gcid.strip()}' "
                        f"(ίσως εξελίχθηκε για διαφορετικό n_states/n_actions).")), 400
                roster.append(agent)
            else:
                roster.append(_make_agent(kind, i, game, seed, strat, float(recip_s or 0.0),
                                          int(mh_s or 2), eps_decay, eps_min))
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
            nash_html = f"(παραλείφθηκε -- n_agents>{_NASH_MAX_AGENTS} κάνει 2^n το table)"

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
        raw_hits = smart_filter_lookup(ledger, game_desc, roster_desc, "dev-webui", rounds, seed,
                                       feature_vector, k=5)
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
                                   metrics=metrics, code_version="dev-webui")
        lineage_bits = [k for k, v in record["lineage"].items() if v]
        repo_info = {
            "config_hash": record["config_hash"], "content_cid": record["content_cid"],
            "contributor": record["contributor"][:16] + "...",
            "signature_valid": Ledger.verify_record(record),
            "lineage": ", ".join(lineage_bits) if lineage_bits else "κανένα (πρώτο του είδους)",
        }

    return render_template(
        "results.html", game=game, roster=roster, seed=seed, rounds=rounds, metrics=metrics,
        show_metrics=show_metrics, metric_meta=_METRIC_META, creatures=creatures,
        chart_svg=chart_svg, console_log=console_log, leaderboard=leaderboard,
        nash_html=nash_html, nash_explain=_NASH_EXPLAIN, phi_info=phi_info, repo_info=repo_info,
        filter_info=filter_info, log_path=log_path.name, epsilon_hints=epsilon_hints,
        genome_cids=genome_cids,
    )


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
            f"population_size ({cfg.population_size}) πρέπει να διαιρείται ακριβώς με το "
            f"n_agents ({n_agents}) -- ο πληθυσμός χωρίζεται σε ομάδες των n_agents ανά match.")), 400

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


if __name__ == "__main__":
    # debug/reloader off on purpose: Werkzeug's reloader re-execs this module in a second
    # process, and on Windows that confused PyPhi's spawn-based multiprocessing pool (used by
    # compute_phi's parallel cut evaluation) into leaking worker processes -- observed directly
    # as a MemoryError from the LP solver plus several orphaned "--multiprocessing-fork" python
    # processes after a request that had "compute_phi" checked. A single plain process avoids it.
    app.run(debug=False, use_reloader=False)
