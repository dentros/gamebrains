"""
Smart Filter: the two lookups over the ledger described in docs/repository-schema.md §6.

  1. Exact  -> `Ledger.find_exact` (in ledger.py) -- hash the normalized config, look up
     `config_hash` + seed + rounds. A hit means "skip the rerun".
  2. Similar -> `nearest` below -- k-NN over the record's `feature_vector`
     ([n, mpcr, cost, rounds, #qlearning, #dqn, #fep, #markov_brain, #classic], see
     repository/record.py's `_feature_vector`), ranked by closeness.

## Why the distance is not plain Euclidean

It used to be, and that was a real defect rather than a stylistic one. The raw feature vector mixes
quantities that differ by four orders of magnitude: `rounds` runs from 10^2 to 10^5 while
`n_agents` runs from 2 to 10 and `mpcr` from 0 to 1. Under an unweighted Euclidean norm the round
count is the only dimension that can contribute a distance worth noticing, so "similar" collapsed
into "roughly the same length", and two experiments with different games, different populations and
different agent mixes ranked as neighbours because both happened to run 20,000 rounds. The lookup
existed, returned results, and answered a question nobody asked.

Two changes fix it, and they address different halves of the problem.

**Scale.** Each dimension is standardised against the ledger's own spread before distances are
taken, so a dimension contributes in proportion to how much it actually varies across recorded
experiments rather than to the units it happens to be measured in. Standardising against the
corpus rather than against fixed bounds means the filter adapts as a ledger fills up: a repository
where everyone ran n=3 stops treating population size as informative, correctly, because it
carries no information there.

**Meaning.** Even standardised, the dimensions are not equally relevant to the question the lookup
is asked. A prior run at a different MPCR is a different experiment; a prior run of different
length is the *same* experiment, observed for longer, and the ledger already models that
relationship explicitly as `extends`. So design fields are weighted above scale fields rather than
being given equal say. The weights are stated as constants below, deliberately visible, because
they encode a judgement about what makes two experiments alike and that judgement should be
arguable rather than buried in a norm.

A degenerate dimension -- one where every record in the ledger holds the same value -- has zero
spread and cannot be standardised. It is dropped rather than divided by an epsilon, since a
constant field distinguishes nothing and a small epsilon would turn floating-point noise in it into
the dominant term, which is the original defect wearing different clothes.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .ledger import Ledger

#: Per-dimension weights, applied after standardisation, in `_feature_vector` order:
#: [n_agents, mpcr, cost, rounds, #qlearning, #dqn, #fep, #markov_brain, #classic].
#:
#: Three tiers, matching the tiers `config_hash` already uses (docs/repository-schema.md §4a):
#: design fields that define the experiment, roster composition that defines who played it, and
#: scale, which the `extends` relation already handles better than a distance ever could.
FEATURE_WEIGHTS = np.array([
    1.0,    # n_agents        design
    1.0,    # mpcr            design
    1.0,    # cost            design
    0.25,   # rounds          scale -- deliberately quietened, see the module docstring
    0.75,   # #qlearning      roster composition
    0.75,   # #dqn
    0.75,   # #fep
    0.75,   # #markov_brain
    0.75,   # #classic
], dtype=float)


def _standardiser(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and scale for each column, with zero-spread columns marked by a scale of 0.

    Returned rather than applied so the same transform can be used on the query vector, which is
    not part of the corpus. Standardising the query against its own value would be meaningless.
    """
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-12] = 0.0            # 0 means "drop this dimension", handled by the caller
    return mean, std


def nearest(records: list[dict[str, Any]], feature_vector: Sequence[float],
            k: int = 5, weights: Sequence[float] | None = None) -> list[tuple[dict[str, Any], float]]:
    """The k nearest prior records to `feature_vector`, closest first.

    Distance is weighted Euclidean over dimensions standardised against `records` themselves, so
    the ranking answers "which recorded experiment is most like this one" rather than "which one
    ran for a similar number of rounds". Returned distances are in standardised units and are
    comparable within one call, not across ledgers of different composition.

    Records whose own feature_vector has a different length (an older schema version, or a roster
    with a kind absent from the current `_KIND_ORDER`) are skipped as incomparable rather than
    padded or truncated: silently comparing mismatched dimensions would be misleading rather than
    lenient.
    """
    query = np.asarray(feature_vector, dtype=float)
    usable = [r for r in records
              if np.asarray(r["feature_vector"], dtype=float).shape == query.shape]
    if not usable:
        return []

    matrix = np.array([r["feature_vector"] for r in usable], dtype=float)
    mean, std = _standardiser(matrix)

    live = std > 0.0                   # dimensions that actually vary across the ledger
    if not live.any():
        # Every recorded run is identical in every feature. Nothing can be ranked, and returning an
        # arbitrary order would imply a similarity judgement that was never made.
        return [(r, 0.0) for r in usable[:k]]

    w = np.asarray(FEATURE_WEIGHTS if weights is None else weights, dtype=float)[live]
    z_matrix = (matrix[:, live] - mean[live]) / std[live]
    z_query = (query[live] - mean[live]) / std[live]

    distances = np.linalg.norm((z_matrix - z_query) * w, axis=1)
    order = np.argsort(distances, kind="stable")
    return [(usable[i], float(distances[i])) for i in order[:k]]


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
