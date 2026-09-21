"""What the repository has covered, and which of its numbers move together.

`meta_analysis.py` answers a question about one design. This module answers the two questions that
only exist once a repository has many: **where are the holes**, and **which quantities move
together across everything recorded so far**. Both are computed from the ledger as it stands, so a
metric, an agent kind or a game added next month appears here without a line being changed. Nothing
in this file names a metric, an agent or a game.

Three choices make the answers honest rather than impressive.

**The unit is a design, not a run.** Twenty runs of one configuration at twenty seeds are twenty
samples of one point, and a correlation over them measures that configuration's seed noise while
looking like a relationship between variables. So rows are collapsed to one point per
`(config_hash, protocol)` by default, which is the same grouping `meta_analysis` pools over.
`unit="run"` is available and the view says which is in use.

**Significance is permuted, not assumed.** A matrix over twenty-six metrics is three hundred and
twenty-five tests, and at n near ten a good handful clear r = 0.6 on noise alone. The null is built
the way this platform already builds one for transfer entropy: shuffle the second column, recompute
the statistic, many times. Benjamini-Hochberg is then applied over the whole matrix, because the
matrix is the family and testing pair by pair would report exactly the false discoveries the
permutation was there to remove.

**A cell that cannot be computed says which reason.** Too few designs, no variation in one of the
two columns, and nothing measured are three different findings, and a blank hides all three.

No scientific-stack dependency, matching the rest of `repository/`: Pearson, the permutation null
and Benjamini-Hochberg are a few lines of arithmetic each, and the alternative is making the layer
that reads the ledger depend on numpy.
"""

from __future__ import annotations

import math
import random
import statistics
from operator import mul
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

#: Below this many points no coefficient is reported. Not a statistical threshold: it is the point
#: below which a permutation null has too few distinct arrangements to place a coefficient in,
#: since 5! is 120 and the smallest p-value reachable at n=5 is therefore 1/120.
MIN_POINTS = 6

#: Shuffles per pair. The smallest p-value a permutation test can report is 1/(permutations+1), so
#: this bounds what the matrix can ever claim. Deliberately not large: the question here is which
#: cells survive a correction, not the precise size of a tail.
PERMUTATIONS = 2000

#: Benjamini-Hochberg level. Looser than 0.05 on purpose, because this view is for deciding what to
#: look at next rather than for reporting a finding, and the page says so.
FDR_Q = 0.10


@dataclass(frozen=True)
class Cell:
    """One pair of fields, carrying the reason when there is no coefficient."""

    x: str
    y: str
    n: int
    r: Optional[float] = None
    p: Optional[float] = None
    significant: bool = False
    note: str = ""


@dataclass
class Matrix:
    fields: list[str]
    cells: dict[tuple[str, str], Cell] = field(default_factory=dict)
    unit: str = "design"
    n_points: int = 0
    n_tested: int = 0
    n_significant: int = 0
    fdr_q: float = FDR_Q

    def get(self, x: str, y: str) -> Optional[Cell]:
        """Either orientation: a matrix is symmetric and only one triangle is stored."""
        return self.cells.get((x, y)) or self.cells.get((y, x))


# --- what is in the ledger ----------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    """Booleans are integers in Python and are not quantities. A True averaged with a False gives
    0.5, which would enter a correlation looking exactly like a measurement."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def numeric_fields(rows: Sequence[dict[str, Any]]) -> list[tuple[str, int]]:
    """Every numeric field these rows offer, with how many rows have it, most populated first.

    Names are the ones the analytics view already understands: a bare design field, `metric:<name>`
    or `roster:<kind>:<param>`. Discovered from the data, which is the point, since the list has to
    grow when the platform does.
    """
    counts: dict[str, int] = {}
    for row in rows:
        seen: set[str] = set()
        for key in ("n_agents", "mpcr", "cost", "rounds"):
            if _is_number(row.get(key)):
                seen.add(key)
        for key, value in (row.get("metrics") or {}).items():
            if _is_number(value):
                seen.add(f"metric:{key}")
        for agent in row.get("roster") or []:
            for key, value in (agent.get("params") or {}).items():
                if _is_number(value):
                    seen.add(f"roster:{agent.get('kind')}:{key}")
        for name in seen:
            counts[name] = counts.get(name, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def field_values(row: dict[str, Any], name: str) -> list[float]:
    """The numeric value or values this row offers for a field name.

    A `roster:` field can match several seats and the mean is taken rather than the first, because
    the learning rate of the DQN seats is a property of the roster, and reading seat 0 would report
    a mixed roster as though it were uniform.
    """
    if name.startswith("metric:"):
        value = (row.get("metrics") or {}).get(name.split(":", 1)[1])
        return [float(value)] if _is_number(value) else []
    if name.startswith("roster:"):
        _, kind, param = name.split(":", 2)
        return [float(agent["params"][param]) for agent in row.get("roster") or []
                if agent.get("kind") == kind and _is_number((agent.get("params") or {}).get(param))]
    value = row.get(name)
    return [float(value)] if _is_number(value) else []


def _row_value(row: dict[str, Any], name: str) -> Optional[float]:
    values = field_values(row, name)
    return statistics.fmean(values) if values else None


# --- the unit of analysis -----------------------------------------------------------------------

def collapse_to_designs(rows: Sequence[dict[str, Any]],
                        fields: Sequence[str]) -> list[dict[str, float]]:
    """One point per `(config_hash, protocol)`, each field averaged over that design's runs.

    A design whose runs disagree about whether a field exists contributes the mean of the runs that
    have it, since the alternative is dropping a whole design over a metric added midway through a
    sweep.
    """
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row.get("config_hash"), row.get("protocol")), []).append(row)

    points = []
    for members in groups.values():
        point: dict[str, float] = {}
        for name in fields:
            values = [v for v in (_row_value(m, name) for m in members) if v is not None]
            if values:
                point[name] = statistics.fmean(values)
        points.append(point)
    return points


def as_points(rows: Sequence[dict[str, Any]], fields: Sequence[str],
              unit: str = "design") -> list[dict[str, float]]:
    if unit == "design":
        return collapse_to_designs(rows, fields)
    points = []
    for row in rows:
        point: dict[str, float] = {}
        for name in fields:
            value = _row_value(row, name)
            if value is not None:
                point[name] = value
        points.append(point)
    return points


# --- the statistic ------------------------------------------------------------------------------

def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """None when either column has no variation, which is a fact about the sample rather than a
    correlation of zero."""
    if len(xs) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def permutation_p(xs: Sequence[float], ys: Sequence[float], observed: float,
                  permutations: int = PERMUTATIONS, seed: int = 0) -> float:
    """Two-sided p from shuffling one column, with the observed arrangement counted in.

    Counting it is what stops the p-value reaching zero, which a resampled test cannot support: the
    floor is 1/(permutations+1) and is reported as that floor.

    Shuffling `ys` changes neither mean, so both columns are centred once and the denominator is
    computed once. Each permutation then costs one dot product rather than a fresh Pearson, which
    is what makes a matrix of a few hundred pairs a page that loads rather than a job that runs.
    """
    n = len(xs)
    if n < 2:
        return 1.0
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    xc = [x - mx for x in xs]
    yc = [y - my for y in ys]
    if sum(v * v for v in xc) <= 0 or sum(v * v for v in yc) <= 0:
        return 1.0

    # The observed numerator, computed the same way the permuted ones will be, rather than
    # reconstructed from `observed` through the denominator.
    target = abs(sum(map(mul, xc, yc)))
    # **Ties have to count, and floating point makes equality unreliable.** A column with few
    # distinct values (a population size, a role count) produces permutations whose numerator is
    # genuinely identical to the observed one, and whether the sum comes out equal depends on the
    # order the terms were added in. Without this tolerance those rearrangements are counted as
    # less extreme, the p-value comes out smaller than the data supports, and the matrix reports
    # discoveries it has not made: on this repository's own ledger that turned 3 surviving cells
    # into 12.
    threshold = target * (1.0 - 1e-9)

    rng = random.Random(seed)
    at_least = 1
    for _ in range(permutations):
        rng.shuffle(yc)
        if abs(sum(map(mul, xc, yc))) >= threshold:
            at_least += 1
    return at_least / (permutations + 1)


def benjamini_hochberg(p_values: Sequence[float], q: float = FDR_Q) -> list[bool]:
    """Which of these p-values survive at false-discovery rate `q`.

    Ordered by p, the largest rank k with p_(k) <= k*q/m sets the cutoff and everything at or below
    it survives. The step-up is the point: testing each pair at its own alpha would accept, in a
    three-hundred-cell matrix at alpha 0.05, about fifteen cells of noise.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    cutoff_rank = 0
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= rank * q / m:
            cutoff_rank = rank
    survives = [False] * m
    for rank, index in enumerate(order, start=1):
        if rank <= cutoff_rank:
            survives[index] = True
    return survives


def correlation_matrix(rows: Sequence[dict[str, Any]], fields: Sequence[str],
                       unit: str = "design", min_points: int = MIN_POINTS,
                       permutations: int = PERMUTATIONS, q: float = FDR_Q,
                       seed: int = 0) -> Matrix:
    """Every pair of the given fields, over the records as they stand.

    A pair is computed on the points where *both* fields exist, so a metric added later does not
    cost the matrix its other cells, and every cell carries its own n because those n differ.
    """
    fields = list(dict.fromkeys(fields))
    points = as_points(rows, fields, unit=unit)
    matrix = Matrix(fields=fields, unit=unit, n_points=len(points), fdr_q=q)

    pending: list[tuple[tuple[str, str], list[float], list[float], float]] = []
    for i, x in enumerate(fields):
        for y in fields[i + 1:]:
            pairs = [(point[x], point[y]) for point in points if x in point and y in point]
            n = len(pairs)
            if n < min_points:
                matrix.cells[(x, y)] = Cell(x, y, n,
                                            note=f"{min_points} {unit}s needed, {n} present")
                continue
            xs = [a for a, _ in pairs]
            ys = [b for _, b in pairs]
            r = pearson(xs, ys)
            if r is None:
                matrix.cells[(x, y)] = Cell(x, y, n, note="no variation in one of the two")
                continue
            pending.append(((x, y), xs, ys, r))

    p_values = [permutation_p(xs, ys, r, permutations=permutations, seed=seed)
                for _, xs, ys, r in pending]
    survives = benjamini_hochberg(p_values, q=q)
    for (key, xs, _ys, r), p, ok in zip(pending, p_values, survives):
        matrix.cells[key] = Cell(key[0], key[1], len(xs), r=r, p=p, significant=ok)

    matrix.n_tested = len(pending)
    matrix.n_significant = sum(survives)
    return matrix


# --- where the holes are ------------------------------------------------------------------------

def coverage_grid(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Which game and agent-kind combinations have been run, and what was measured there.

    The empty squares are the point. A platform that can report only what it has run cannot be
    asked what it has not, and the question of what is missing is the one a research programme is
    actually organised around.
    """
    games: dict[str, int] = {}
    kinds: dict[str, int] = {}
    metrics: dict[str, int] = {}
    cells: dict[tuple[str, str], dict[str, Any]] = {}

    for row in rows:
        game = str(row.get("game_name") or "?")
        games[game] = games.get(game, 0) + 1
        present = {k for k, v in (row.get("metrics") or {}).items() if _is_number(v)}
        for name in present:
            metrics[name] = metrics.get(name, 0) + 1
        for kind in {agent.get("kind", "?") for agent in row.get("roster") or []}:
            kinds[kind] = kinds.get(kind, 0) + 1
            cell = cells.setdefault((game, kind), {"runs": 0, "designs": set(), "metrics": set()})
            cell["runs"] += 1
            cell["designs"].add(row.get("config_hash"))
            cell["metrics"] |= present

    return {
        "games": sorted(games, key=lambda g: (-games[g], g)),
        "kinds": sorted(kinds, key=lambda k: (-kinds[k], k)),
        "metrics": sorted(metrics, key=lambda m: (-metrics[m], m)),
        "cells": {key: {"runs": value["runs"], "designs": len(value["designs"]),
                        "metrics": len(value["metrics"])} for key, value in cells.items()},
        "filled": len(cells),
        "total": len(games) * len(kinds),
    }
