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
_KIND_ORDER = ["qlearning", "dqn", "fep", "markov_brain", "classic"]


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


def _roster_description(roster: list[Agent]) -> list[dict[str, Any]]:
    return [{"kind": a.kind, "training_mode": getattr(a, "training_mode", "?")} for a in roster]


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


def _feature_vector(game: Game, roster: list[Agent], rounds: int) -> list[float]:
    counts = [sum(1 for a in roster if a.kind == kind) for kind in _KIND_ORDER]
    base = [game.n_agents, getattr(game, "mpcr", 0.0), getattr(game, "cost", 0.0), rounds]
    return base + counts


def _scalar_metrics_summary(metrics: dict[str, Any]) -> dict[str, float]:
    """Drop non-scalar entries (per-round series, per-agent arrays) -- see docs/
    repository-schema.md §3: `metrics_summary` is a small cache, not the full record."""
    return {k: float(v) for k, v in metrics.items() if np.isscalar(v)}


def record_experiment(
    repo_root: str | Path,
    game: Game,
    roster: list[Agent],
    log_path: str | Path,
    rounds: int,
    seed: int,
    metrics: dict[str, Any],
    code_version: str = DEFAULT_CODE_VERSION,
) -> dict[str, Any]:
    """Package a completed match's event-log + metrics and append it to the repository.

    Returns the full signed ledger record (see repository/ledger.py).
    """
    with open(log_path, encoding="utf-8") as f:
        event_log = [json.loads(line) for line in f if line.strip()]

    package = {"manifest": game.describe(), "event_log": event_log}
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
    )
