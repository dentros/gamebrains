"""Does a record's stored evidence actually support the numbers it reports?

A signature proves who wrote a record and that nobody edited it afterwards. It proves nothing
about whether the numbers are the ones the run produced, because the person holding the key can
sign anything at all. That gap is the whole reason a shared repository is hard: everybody can
verify everybody's signatures and still be looking at invented results.

This module closes as much of that gap as arithmetic can close. Every headline measure in this
platform is a **pure function of the event log** (CLAUDE.md section 5, and it is the reason the
metrics were built that way), and the log travels inside the record's own package. So the numbers
can simply be recomputed from the evidence and compared against what the record claims. Editing a
reported cooperation rate now requires editing the log it came from, which changes the package,
which changes the `content_cid` the signed record points at.

**What this establishes, stated precisely**, because overclaiming here would be worse than not
checking at all:

- The package is the one the record points at, since its `content_cid` is recomputed from it.
- The reported measures are the ones this log produces, to the last decimal we store.
- The log is internally consistent: the roster size matches the width of every round, the round
  count matches the declared horizon, and the payoff arithmetic the game declares holds round by
  round.

**What it does not and cannot establish:** that the log came from a real run. A determined author
can fabricate a log and derive honest metrics from it, and no signature or checksum will ever
notice. Only independent replication answers that, which is what `config_hash` and the
`extends`/`replicates` detection exist for: a fabricated result is exposed when somebody else runs
the same declared experiment and gets something else. The two mechanisms are complementary, and
the report below says which of the two it is speaking for.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from .normalize import canonical_json

#: How far a recomputed measure may sit from the stored one before it counts as a disagreement.
#: Stored values are full-precision floats written by the same code path that recomputes them, so
#: the only expected difference is the last bit or two of floating-point summation order. A looser
#: tolerance would quietly accept an edited number, which is the thing being checked.
TOLERANCE = 1e-9

#: Measures recomputed here. Each is a pure function of the round events, which is what makes the
#: check possible; anything needing the live agents (Phi, the brain snapshots) cannot be rechecked
#: from a log and is deliberately absent rather than waved through.
RECHECKED = ("cooperation_rate", "payoff_gini", "action_entropy_bits")


@dataclass
class Check:
    """One question asked of a record, with the answer and why it matters."""

    name: str
    passed: bool
    detail: str


@dataclass
class Report:
    """Every check run against one record, and whether the record may be trusted as evidence."""

    record_hash: str
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if not check.passed]

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(Check(name, passed, detail))


def _canonical_cid(package: dict[str, Any]) -> str:
    """The CID this package addresses.

    Deliberately uses the project's own `canonical_json` rather than a second spelling of it. A
    checksum has to agree byte for byte on how the bytes are produced, so an "independent"
    canonicalisation here would not be a second opinion, it would simply always disagree. The
    independence that matters is in the *measures*, which are recomputed from the evidence rather
    than copied from the record.
    """
    blob = canonical_json(package)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _rounds(package: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in package.get("event_log", []) or [] if e.get("type") == "round"]


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def _gini(values: list[float]) -> float:
    """The same Gini the platform reports, rewritten here rather than imported.

    Importing `metrics/social.py` would make this check agree with that module by construction,
    including if that module were ever wrong. An independent implementation is the point: two
    routes to the same number is evidence, one route computed twice is not.
    """
    if not values:
        return 0.0
    # Payoffs go negative in a social dilemma, and the Gini coefficient is undefined there, so
    # the platform's measure shifts the whole vector up by its own minimum before computing. That
    # is part of what the published number *means*, not an implementation detail, so this check
    # shares it. Leaving it out made this function disagree with every stored record on the first
    # run, which is how the convention got found and written down here.
    floor = min(values)
    ordered = sorted(value - floor for value in values)
    n = len(ordered)
    total = sum(ordered)
    if total <= 0:
        return 0.0
    weighted = sum((i + 1) * value for i, value in enumerate(ordered))
    return (2 * weighted) / (n * total) - (n + 1) / n


def recompute_metrics(package: dict[str, Any]) -> dict[str, float]:
    """The measures this log produces, computed from the log alone."""
    rounds = _rounds(package)
    if not rounds:
        return {}

    n_agents = max((len(r.get("actions") or []) for r in rounds), default=0)
    if not n_agents:
        return {}

    coop_fractions, entropies = [], []
    totals = [0.0] * n_agents
    for event in rounds:
        actions = event.get("actions") or []
        rewards = event.get("rewards") or []
        for i, reward in enumerate(rewards[:n_agents]):
            totals[i] += float(reward)
        # Cooperation comes from the game's own per-round count, because what counts as
        # cooperating is the game's declaration rather than this module's reading of an index.
        if isinstance(event.get("cooperators"), (int, float)):
            coop_fractions.append(float(event["cooperators"]) / n_agents)
        if actions:
            entropies.append(_binary_entropy(sum(float(a) for a in actions) / len(actions)))

    out: dict[str, float] = {}
    if coop_fractions:
        out["cooperation_rate"] = sum(coop_fractions) / len(coop_fractions)
    if any(totals):
        out["payoff_gini"] = _gini(totals)
    if entropies:
        out["action_entropy_bits"] = sum(entropies) / len(entropies)
    return out


def verify(record: dict[str, Any], package: Optional[dict[str, Any]],
           record_hash: str = "") -> Report:
    """Check one record against the evidence it carries.

    `package` is the stored object the record's `content_cid` addresses, or `None` when the
    evidence did not travel with the record. A missing package is reported as such rather than
    passed over: a record nobody can check is exactly what this is for.
    """
    report = Report(record_hash=record_hash or (record.get("content_cid") or "")[:16])

    if package is None:
        report.add("evidence present", False,
                   "the record's package is not here, so nothing it reports can be rechecked. "
                   "This happens when a bundle was shared without the objects its records point "
                   "at")
        return report

    claimed_cid = record.get("content_cid") or ""
    actual_cid = _canonical_cid(package)
    report.add("package is the one signed", claimed_cid == actual_cid,
               "the package hashes to the content_cid the signed record names"
               if claimed_cid == actual_cid else
               f"the record points at {claimed_cid[:16]}... and this package hashes to "
               f"{actual_cid[:16]}..., so one of the two was changed after signing")

    rounds = _rounds(package)
    declared_rounds = (record.get("horizon") or {}).get("rounds")
    if declared_rounds is not None:
        report.add("round count matches", len(rounds) == declared_rounds,
                   f"{len(rounds)} rounds in the log, {declared_rounds} declared"
                   if len(rounds) != declared_rounds else
                   f"the log holds the {declared_rounds} rounds the record declares")

    declared_agents = (record.get("game") or {}).get("n_agents")
    widths = {len(r.get("actions") or []) for r in rounds}
    if declared_agents is not None and widths:
        consistent = widths == {declared_agents}
        report.add("roster size matches the log", consistent,
                   f"every round records {declared_agents} actions"
                   if consistent else
                   f"the record declares {declared_agents} players and the log has rounds with "
                   f"{sorted(widths)}")

    recomputed = recompute_metrics(package)
    claimed = record.get("metrics_summary") or {}
    for name in RECHECKED:
        if name not in recomputed or claimed.get(name) is None:
            continue
        gap = abs(recomputed[name] - float(claimed[name]))
        report.add(f"{name} recomputes", gap <= TOLERANCE,
                   f"the log gives {recomputed[name]:.10f}, matching what is reported"
                   if gap <= TOLERANCE else
                   f"the record reports {float(claimed[name]):.10f} and this log produces "
                   f"{recomputed[name]:.10f}, a gap of {gap:.2e}")
    return report


def verify_all(records: list[dict[str, Any]], packages: dict[str, dict[str, Any]]) -> list[Report]:
    """One report per record, in the order given. `packages` maps `content_cid` to the object."""
    return [verify(r, packages.get(r.get("content_cid") or ""), (r.get("content_cid") or "")[:16])
            for r in records]
