"""Pooling results across runs, across seeds, and across installations.

The repository stores one record per match. The questions a researcher actually asks are one level
up: what does this design give on average, how much does it move between seeds, and does somebody
else's installation get the same thing. This module answers those three and nothing else, over
records that `bundle.load_records` has already tagged with the source they came from.

Three deliberate choices.

**A design is `(config_hash, protocol)`.** That pairing is exactly what the repository already
treats as one experiment (see `normalize.build_config`): same game, same roster, same code
version, same scoring conventions, with round count and seed deliberately outside. So runs that
differ only in seed group together, which is the grouping a seed average needs, and runs scored
under different conventions never pool, which is the grouping mistake that would quietly average
two different quantities.

**Dispersion is reported, never smoothed.** A mean over seeds with no interval is the form in
which a multi-agent result is most often overstated, and this platform's own bake-off produced a
mean that no individual run exhibits (seven seeds cooperative, three defecting). So every group
carries its standard deviation, its interval and its n, and a group of one gets no interval rather
than a comfortable-looking zero.

**Heterogeneity is a question about sources, not a summary statistic.** `heterogeneity` runs over
the *per-source* means for one design, so what it measures is whether two installations running
the same declared experiment agree. Applying the same formula across designs would answer nothing,
since designs are meant to differ. Cochran's Q and I^2 are the standard pair for this and are
reported with their own caveat: with two or three sources, which is what a small collaboration
has, I^2 is too noisy to act on and is there to be looked at rather than thresholded.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Optional

#: Two-sided 95% t multipliers by degrees of freedom (n-1), for the small n a seed sweep has.
#: A table rather than scipy: the repository layer has no scientific-stack dependency, the values
#: are fixed, and the alternative of reaching for a normal 1.96 at n=3 would understate the
#: interval by nearly a factor of two.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042,
        40: 2.021, 60: 2.000, 120: 1.980}


def _t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    for key in sorted(_T95):
        if df <= key:
            return _T95[key]
    return 1.960


def design_key(record: dict[str, Any]) -> tuple[str, str]:
    """What makes two records the same experiment: the config hash and the scoring protocol."""
    protocol = record.get("protocol") or {}
    rendered = ",".join(f"{k}={protocol[k]}" for k in sorted(protocol))
    return record.get("config_hash", ""), rendered


def _metric_value(record: dict[str, Any], metric: str) -> Optional[float]:
    value = (record.get("metrics_summary") or {}).get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None            # a withheld metric arrives as None or as its own reason, in words
    return float(value)


def summarise(values: Iterable[float]) -> dict[str, Any]:
    """Mean, spread and a 95% interval, or an honest absence of one.

    A single observation has a mean and no dispersion at all. Reporting `ci=0` there would be a
    statement that the quantity does not vary, made from data that could not have detected
    variation.
    """
    data = [float(v) for v in values]
    n = len(data)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "ci95": None, "low": None, "high": None}
    mean = statistics.fmean(data)
    if n == 1:
        return {"n": 1, "mean": mean, "sd": None, "ci95": None, "low": None, "high": None,
                "note": "one run, so nothing can be said about spread"}
    sd = statistics.stdev(data)
    half = _t95(n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "ci95": half,
            "low": mean - half, "high": mean + half,
            "min": min(data), "max": max(data)}


def group_by_design(records: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    """One row per design, largest group first, each with its own summary and its sources.

    Rows carry `values` so a caller can draw the individual runs beside the pooled estimate. A
    forest plot that shows only intervals hides the case this platform has already met, where the
    mean sits in a region no single run occupied.
    """
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        value = _metric_value(record, metric)
        if value is None:
            continue
        key = design_key(record)
        group = groups.setdefault(key, {
            "config_hash": key[0], "protocol": key[1], "values": [], "records": [],
            "by_source": {}, "seeds": set(),
        })
        group["values"].append(value)
        group["records"].append(record)
        group["seeds"].add(record.get("seeds", {}).get("master"))
        label = (record.get("source") or {}).get("label", "this installation")
        group["by_source"].setdefault(label, []).append(value)

    rows = []
    for key, group in groups.items():
        first = group["records"][0]
        rows.append({
            "config_hash": key[0],
            "protocol": key[1],
            "game": (first.get("game") or {}).get("name", "?"),
            "n_agents": (first.get("game") or {}).get("n_agents"),
            "roster": _roster_summary(first.get("roster") or []),
            "rounds": sorted({r.get("horizon", {}).get("rounds") for r in group["records"]}),
            "seeds": sorted(s for s in group["seeds"] if s is not None),
            "values": group["values"],
            "summary": summarise(group["values"]),
            "by_source": {label: summarise(values)
                          for label, values in group["by_source"].items()},
            "sources": sorted(group["by_source"]),
            "reproducible": all(
                (r.get("reproducibility") or {}).get("byte_identical_claimed", True)
                for r in group["records"]),
        })
    rows.sort(key=lambda row: (-row["summary"]["n"], row["config_hash"]))
    return rows


def _roster_summary(roster: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for agent in roster:
        counts[agent.get("kind", "?")] = counts.get(agent.get("kind", "?"), 0) + 1
    return ", ".join(f"{count}x {kind}" for kind, count in sorted(counts.items()))


def heterogeneity(by_source: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Do independent sources running one design agree, in Cochran's Q and I^2.

    Each source contributes its own mean and standard error, weighted by the inverse of that
    error's square, which is the usual fixed-effect weighting. Sources whose mean rests on a single
    run have no standard error and are excluded from the statistic, though they stay visible in the
    table: a point estimate with no dispersion cannot be weighted against one that has it, and
    substituting a nominal value for the missing spread would let a single run dominate.

    Returns `None` for the statistics when fewer than two sources qualify. That is the common case
    in a small collaboration, and saying so is more useful than printing an I^2 computed from two
    numbers.
    """
    usable = {label: s for label, s in by_source.items()
              if s.get("n", 0) > 1 and s.get("sd") and s["sd"] > 0}
    if len(usable) < 2:
        return {"sources": len(by_source), "usable": len(usable), "q": None, "i2": None,
                "pooled": None,
                "note": "at least two sources with more than one run each are needed"}

    weights, means = [], []
    for summary in usable.values():
        se = summary["sd"] / math.sqrt(summary["n"])
        weights.append(1.0 / (se * se))
        means.append(summary["mean"])

    pooled = sum(w * m for w, m in zip(weights, means)) / sum(weights)
    q = sum(w * (m - pooled) ** 2 for w, m in zip(weights, means))
    df = len(usable) - 1
    i2 = max(0.0, (q - df) / q) * 100 if q > 0 else 0.0
    return {
        "sources": len(by_source), "usable": len(usable), "q": q, "df": df, "i2": i2,
        "pooled": pooled,
        "note": ("I^2 from a handful of sources is noisy; read it alongside the per-source "
                 "intervals rather than as a threshold"),
    }


def compare_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The designs more than one source has run, with their agreement statistics.

    This is the question federation exists to answer, and the reason the repository keeps one
    chain per installation instead of merging them.
    """
    shared = []
    for row in rows:
        if len(row["sources"]) < 2:
            continue
        shared.append({**row, "heterogeneity": heterogeneity(row["by_source"])})
    return shared
