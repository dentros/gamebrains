"""
Glue: turn one completed match (its saved event-log + computed metrics) into a repository record.

Kept deliberately separate from `experiments/run_pgg.py` so the repository package has no
knowledge of any particular experiment script, and any future script (run_evolution.py, a future
run_honey_jar.py, ...) can call the same one function.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from ..engine.agent import Agent
from ..engine.game import Game
from .cas import ContentStore
from .ledger import Ledger

# Fixed kind order for the feature_vector's per-kind counts (see docs/repository-schema.md §3).
#
# `llm` was appended on 2026-09-18, which lengthens the vector from 9 to 10. Records written before
# that keep their 9-element vectors and `smart_filter.nearest` skips them as incomparable, which is
# its documented behaviour for a schema change and is the honest one: a shorter vector does not
# state that no language model played, it states that the question was not asked. Append here
# rather than insert, so at least the positions of the existing dimensions never move.
_KIND_ORDER = ["qlearning", "dqn", "fep", "markov_brain", "classic", "llm"]


def _git_code_version() -> str:
    """`git:<short-hash>` of whatever commit is actually checked out, computed fresh each call
    (not hardcoded) so it never goes stale as the repo (now public: github.com/dentros/gamebrains)
    keeps moving. Falls back to "dev-nogit" for a non-git checkout (e.g. a plain zip/pip install
    with no .git directory) rather than failing the whole recording step over it."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5, check=True,
        )
        return f"git:{out.stdout.strip()}"
    except Exception:
        return "dev-nogit"


DEFAULT_CODE_VERSION = _git_code_version()


# Which attributes count as "the config" for each kind -- read back from the live agent object
# (not the constructor call) so this works uniformly whether the agent was built by run_pgg.py,
# the webui, or anything else. Missing attributes are simply omitted, not errors: an older agent
# class without one of these fields still hashes fine, just without that dimension.
_PARAM_ATTRS: dict[str, tuple[str, ...]] = {
    "qlearning": ("alpha", "gamma", "epsilon_decay", "epsilon_min"),
    "dqn": ("hidden", "lr", "gamma", "epsilon_decay", "epsilon_min", "buffer_size",
            "batch_size", "train_every", "target_sync_every"),
    "fep": ("reciprocity", "obs_noise", "drift", "precision"),
    "markov_brain": ("n_hidden",),
    "classic": ("strategy", "p"),
    # For an LLM seat the model tag and whether the server constrained decoding to the schema are
    # design decisions, not runtime detail: the same prompt answered by a different model, or by
    # the same model without the constraint, is a different experiment and must not share a
    # config_hash. `persona` is here for the same reason, since it is an experimental manipulation.
    # `profile` names what the model was told (agents/llm_prompt.py). It belongs here above all the
    # others: two models compared under different profiles were not given the same board, and a
    # repository that let those two runs share a config_hash would be offering one as the other's
    # answer. Adding it lengthens the parameter set for llm seats only, so llm records written
    # before this line carry a different config_hash from equivalent ones written after it.
    "llm": ("backend_name", "model", "profile", "native_schema", "history", "persona"),
}


def _agent_params(a: Agent) -> dict[str, Any]:
    attrs = _PARAM_ATTRS.get(getattr(a, "kind", None), ())
    return {attr: getattr(a, attr) for attr in attrs if hasattr(a, attr)}


#: Weakest first. A run's tier is the weakest tier any of its agents declares, because a promise
#: the whole run makes cannot be stronger than the component least able to keep it.
_TIER_ORDER = ("none", "replay", "byte")


def reproducibility_of(roster: list[Agent]) -> dict[str, Any]:
    """What a rerun of this roster would give back, as the weakest tier any of its agents declares.

    The fourth place this codebase declines to proceed as though nothing were wrong (see the
    paper's refusals section, and `metrics/information.py` for the third). The claim is that one
    `config_hash` plus one seed implies one event log, and it holds because every agent draws from
    a generator the runner seeded. An agent that declares a weaker tier (`Agent.reproducibility`)
    breaks it for the run it takes part in, and the useful move is neither to drop the claim
    everywhere nor to keep making it: it is to record, on the run itself, what the claim covers
    here and why.

    Downstream that matters twice. A reader of the record knows what a rerun would and would not
    give them, and `Ledger.find_exact` refuses to offer such a record as a finished answer for the
    same configuration, because reusing it would silently substitute one sample of a random
    process for another.
    """
    declared = [(a, a.reproducibility()) for a in roster]
    tier = min((t for _, t in declared), key=_TIER_ORDER.index, default="byte")
    offenders = [
        {"name": getattr(a, "name", "?"), "kind": getattr(a, "kind", "?"),
         "tier": t, "reason": a.reproducibility_note()}
        for a, t in declared if t != "byte"
    ]
    return {
        "tier": tier,
        # Kept alongside the tier because records written before tiers existed carry it, and
        # `Ledger.find_exact` reads it to decide what may be offered as finished work. A replayed
        # run is byte-identical and still not a substitute for running the thing, which is why it
        # is false here and the tier says why.
        "byte_identical_claimed": tier == "byte",
        "nondeterministic_agents": offenders,
    }


def _roster_description(roster: list[Agent]) -> list[dict[str, Any]]:
    """Feeds `config_hash` (via normalize.build_config) -- so two runs whose rosters differ only
    in hyperparameters (e.g. Q-learning alpha, DQN learning rate) must produce different
    descriptions here, not just different "kind" labels, or they would silently collide onto the
    same config_hash. Bug found 2026-07-16: this previously returned only {kind, training_mode},
    so hyperparameter changes were invisible to the ledger entirely."""
    # `information` joins the identity of a run because two agents of the same kind that condition
    # on different things are not the same agent, whatever their hyperparameters say. It is
    # constant per class today, so this changes every hash exactly once and never again, which is
    # the cheapest moment to do it: the alternative is doing it after records are shared.
    return [
        {"kind": a.kind, "training_mode": getattr(a, "training_mode", "?"),
         "information": getattr(a, "information", "observation"), "params": _agent_params(a)}
        for a in roster
    ]


def _config_game_desc(game: Game) -> dict[str, Any]:
    """`game.describe()` embeds `rounds` (a per-run parameter, not part of the experimental
    design) alongside the design fields (mpcr, cost, ...). `rounds` is already passed to
    `Ledger.append()` as its own `rounds=` argument, so it must be stripped here before the dict
    is hashed into `config_hash` -- otherwise two runs of the same design at different lengths
    would get different config_hash values and `extends` lineage detection (see
    repository/normalize.py's `build_config` docstring and repository/ledger.py's
    `_detect_lineage`) would never fire."""
    desc = game.describe()
    desc.pop("rounds", None)
    return desc


def protocol_from_run(run_result: dict[str, Any]) -> dict[str, Any]:
    """The scoring conventions a finished match actually used, read back off the runner's own
    output rather than re-stated by the caller, so a recorded protocol can never drift from the
    one that produced the numbers.

    This is the single definition of what "protocol" covers (see normalize.build_config for why
    it is hashed). Anything added to it later belongs here, once, not at each call site.
    """
    return {"partial_episode": run_result.get("partial_episode_policy", "drop")}


def _feature_vector(game: Game, roster: list[Agent], rounds: int) -> list[float]:
    counts = [sum(1 for a in roster if a.kind == kind) for kind in _KIND_ORDER]
    base = [game.n_agents, getattr(game, "mpcr", 0.0), getattr(game, "cost", 0.0), rounds]
    return base + counts


def _scalar_metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Drop non-scalar entries (per-round series, per-agent arrays) -- see docs/
    repository-schema.md §3: `metrics_summary` is a small cache, not the full record.

    Three kinds survive, and the second two are here for one reason. A metric that declined to
    report itself (`metrics/information.py` withholds transfer entropy over a non-stationary
    window) must reach the ledger as an explicit null carrying an explicit reason, because a field
    that is simply missing is indistinguishable from a bug for anyone reading the record later.

    Note that `np.isscalar` answers True for strings, so the previous one-line version would have
    raised on the first diagnostic string it met rather than skipping it.
    """
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if value is None:
            out[key] = None                                   # withheld, deliberately
        elif isinstance(value, str):
            if len(value) <= 300:                             # a reason, not a payload
                out[key] = value
        elif isinstance(value, (bool, np.bool_)):
            out[key] = bool(value)
        elif np.isscalar(value):
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                pass
    return out


def record_experiment(
    repo_root: str | Path,
    game: Game,
    roster: list[Agent],
    log_path: str | Path,
    rounds: int,
    seed: int,
    metrics: dict[str, Any],
    code_version: str = DEFAULT_CODE_VERSION,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Package a completed match's event-log + metrics and append it to the repository.

    `protocol` is the scoring conventions this run used (see normalize.build_config): pass the
    ones the runner was actually given, not the defaults, or two runs scored differently will
    collide onto the same config_hash and the Smart Filter will call them the same experiment.

    Returns the full signed ledger record (see repository/ledger.py).
    """
    with open(log_path, encoding="utf-8") as f:
        event_log = [json.loads(line) for line in f if line.strip()]

    # Any agent that journals its decisions has them stored with the run, which is what makes an
    # exact rerun possible for a component that cannot promise one on its own
    # (see agents/llm_replay.py). Keyed by agent name, since a match may seat more than one.
    decisions = {a.name: a.journal() for a in roster
                 if callable(getattr(a, "journal", None)) and a.journal()}
    package = {"manifest": game.describe(), "event_log": event_log}
    if decisions:
        package["llm_decisions"] = decisions
    content_cid = ContentStore(repo_root).put(package)

    ledger = Ledger(repo_root)
    return ledger.append(
        game_desc=_config_game_desc(game),
        roster_desc=_roster_description(roster),
        code_version=code_version,
        rounds=rounds,
        master_seed=seed,
        content_cid=content_cid,
        metrics_summary=_scalar_metrics_summary(metrics),
        feature_vector=_feature_vector(game, roster, rounds),
        reproducibility=reproducibility_of(roster),
        protocol=protocol,
    )
