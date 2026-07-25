"""
Smart Filter (lite): the two lookups over the ledger described in docs/repository-schema.md §6.

  1. Exact  -> `Ledger.find_exact` (already implemented in ledger.py) -- hash the normalized
     config, look up `config_hash` + seed + rounds. Hit means "skip the rerun".
  2. Similar -> `nearest` below -- k-NN over the record's `feature_vector`
     ([n, mpcr, cost, rounds, #qlearning, #dqn, #fep, #markov_brain, #classic], see
     repository/record.py's `_feature_vector`), ranked by closeness.

Honesty note on the distance metric: this is plain Euclidean distance on the *raw* feature
vector, not a normalized/whitened one. `rounds` (order ~10^2-10^4) and `n_agents` (order ~1-10)
live on wildly different scales, so in practice `rounds` dominates the distance and "similar"
mostly means "similar length + similar design" rather than a carefully weighted notion of
closeness. Good enough for a first "existing experiments found: X" signal (per the schema's own
scope for this lookup); a real version would normalize each dimension (e.g. z-score across the
ledger) before computing distance -- left as a documented future improvement, not hidden.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .ledger import Ledger


def nearest(records: list[dict[str, Any]], feature_vector: Sequence[float],
           k: int = 5) -> list[tuple[dict[str, Any], float]]:
    """The k nearest prior records to `feature_vector`, closest first.

    Records whose own feature_vector has a different length (e.g. an older schema version, or a
    roster with a kind not in the current `_KIND_ORDER`) are skipped as incomparable rather than
    padded/truncated -- silently comparing mismatched dimensions would be misleading, not lenient.
    """
    fv = np.asarray(feature_vector, dtype=float)
    scored = []
    for record in records:
        rv = np.asarray(record["feature_vector"], dtype=float)
        if rv.shape != fv.shape:
            continue
        scored.append((record, float(np.linalg.norm(rv - fv))))
    scored.sort(key=lambda pair: pair[1])
    return scored[:k]


def lookup(ledger: Ledger, game_desc: dict[str, Any], roster_desc: list[dict[str, Any]],
          code_version: str, rounds: int, master_seed: int, feature_vector: Sequence[float],
          k: int = 5, protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    """Both Smart Filter lookups against the ledger's *current* state (call before appending the
    new run's own record, or it will trivially appear as its own nearest neighbor at distance 0).

    Pass the same `protocol` the run will be recorded under. Skipping it would make the exact
    lookup advertise a reuse hit for a run scored by different conventions, which is the one
    thing this lookup exists to prevent.
    """
    exact = ledger.find_exact(game_desc, roster_desc, code_version, rounds, master_seed, protocol)
    similar = nearest(ledger.load_all(), feature_vector, k=k)
    return {"exact": exact, "similar": similar}
