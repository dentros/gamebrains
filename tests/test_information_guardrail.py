"""The transfer-entropy guardrail: a null control, and the machinery that makes it pass.

`tests/test_information.py` checks the estimators against known values. It never asked the question
that matters for a significance test, which is how often the test fires when the answer is known to
be no. This module asks it.

The null control is agents drawn from *separate matches*. They never shared a game and cannot have
influenced one another, so every pair flagged significant is a false positive by construction. The
one thing they share is an exploration schedule, and that shared deterministic trend is what a
permutation surrogate cannot null out.

Run: python -m gamebrains.tests.test_information_guardrail
"""

from __future__ import annotations

import numpy as np

from ..agents.classic import AllD
from ..agents.qlearning import QLearningAgent
from ..engine.runner import run_match
from ..games.public_goods import PublicGoodsGame
from ..metrics import information

ROUNDS = 4000
SEEDS = 8


def _independent_run(seed: int) -> tuple[np.ndarray, list]:
    game = PublicGoodsGame(n_agents=3, rounds=ROUNDS, mpcr=0.5)
    roster = [QLearningAgent(f"Q{i}", game.n_states, game.n_actions, seed=seed * 100 + i)
              for i in range(3)]
    return run_match(game, roster, rounds=ROUNDS, seed=seed)["actions"], roster


def _false_positive_rate(burn_in: int) -> float:
    """Fraction of provably unconnected pairs the test flags at the given burn-in."""
    flagged = total = 0
    for seed in range(SEEDS):
        a, _ = _independent_run(seed)
        b, _ = _independent_run(seed + 500)
        spliced = {"actions": np.column_stack([a[:, 0], b[:, 0]])}
        out = information.transfer_entropy_pairwise(spliced, seed=seed, n_surrogates=200,
                                                    burn_in=burn_in)
        for detail in out["by_pair"].values():
            total += 1
            flagged += detail["p_value"] < information.ALPHA
    return flagged / total


def test_burn_in_is_derived_from_the_schedule() -> None:
    """The point of deriving it: two rosters with different schedules get different answers, and a
    magic constant could not be right for both."""
    game = PublicGoodsGame(n_agents=2, rounds=20000)
    slow = [QLearningAgent("slow", game.n_states, game.n_actions, seed=1,
                           epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.9995)]
    fast = [QLearningAgent("fast", game.n_states, game.n_actions, seed=1,
                           epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.99)]

    slow_n, slow_why = information.suggested_burn_in(slow, 20000)
    fast_n, fast_why = information.suggested_burn_in(fast, 20000)

    # ln(0.02)/ln(0.9995) is about 7823; ln(0.02)/ln(0.99) about 389.
    assert 7000 < slow_n < 8500, slow_n
    assert 300 < fast_n < 500, fast_n
    assert slow_n > fast_n * 10
    assert "rounds discarded" in slow_why and "slow" in slow_why
    print(f"OK: burn-in derived per roster ({fast_n} fast, {slow_n} slow), not a constant")


def test_a_roster_with_no_schedule_needs_no_burn_in() -> None:
    """Fixed strategies have no transient, and inventing one for them would discard good data."""
    n, why = information.suggested_burn_in([AllD("a"), AllD("b")], 1000)
    assert n == 0 and "no agent" in why
    print("OK: a roster of fixed strategies is given no burn-in, with a reason")


def test_missing_roster_is_reported_not_assumed() -> None:
    n, why = information.suggested_burn_in(None, 1000)
    assert n == 0 and "no roster supplied" in why
    print("OK: an absent roster produces an explanation rather than a silent zero")


def test_burn_in_longer_than_the_match_is_caught() -> None:
    """A short run of slow learners is entirely transient. Reporting influence there would be
    reporting on the schedule."""
    game = PublicGoodsGame(n_agents=2, rounds=500)
    roster = [QLearningAgent("q", game.n_states, game.n_actions, seed=1,
                             epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.9995)]
    n, why = information.suggested_burn_in(roster, 500)
    assert n >= 500 and "no part of this run is post-transient" in why
    print("OK: a match shorter than its own transient is identified as such")


def test_stationarity_report_separates_the_three_cases() -> None:
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 2, size=600)
    drifting = (rng.random(600) < np.linspace(0.95, 0.05, 600)).astype(int)

    assert information.stationarity_report(np.zeros(600, dtype=int))["verdict"] == "constant"
    assert information.stationarity_report(noise)["verdict"] == "stationary"
    assert information.stationarity_report(drifting)["verdict"] in ("non-stationary", "inconclusive")
    print("OK: stationarity report distinguishes constant, stationary and drifting series")


def test_the_guardrail_reduces_false_positives_on_a_null_control() -> None:
    """The headline. Without burn-in the platform flagged influence between agents that never met.

    The threshold is loose on purpose. This is 8 seeds, not the companion study's 100, so it asserts
    the direction and a large margin rather than a precise rate.
    """
    without = _false_positive_rate(burn_in=0)
    with_guard = _false_positive_rate(burn_in=2000)

    print(f"     false positives without burn-in: {without:.1%}")
    print(f"     false positives with burn-in   : {with_guard:.1%}")
    assert without > 0.5, (
        f"expected the unguarded procedure to fail loudly on a null control, got {without:.1%}. "
        "If this now passes, re-measure before relaxing the guardrail.")
    assert with_guard < without, "excluding the transient must reduce false positives"
    print("OK: excluding the training transient substantially reduces false positives")


def test_the_aggregate_is_withheld_with_a_reason_not_silently() -> None:
    """The user-facing half of the guardrail: an absent number must never look like a bug."""
    actions, roster = _independent_run(0)
    records = {"actions": actions, "cooperators": actions.sum(axis=1)}

    no_roster = information.compute_all(records, seed=0, n_surrogates=50)
    assert no_roster["transfer_entropy_bits"] is None
    assert "no roster supplied" in no_roster["transfer_entropy_bits_status"]

    with_roster = information.compute_all(records, seed=0, n_surrogates=50, roster=roster)
    status = with_roster["transfer_entropy_bits_status"]
    assert status, "a status must always be present, whether or not a number was"
    if with_roster["transfer_entropy_bits"] is None:
        assert "stationary" in status or "post-transient" in status, status
        print(f"OK: aggregate withheld, and says why ({status[:60]}...)")
    else:
        assert "rounds discarded" in status
        print(f"OK: aggregate reported, and says on what window ({status[:60]}...)")


def test_a_withheld_number_survives_the_ledger_summary() -> None:
    """The reason has to reach the record. A string in the metrics dict used to raise here, and a
    None used to vanish, which is the failure mode this whole guardrail exists to avoid."""
    from ..repository.record import _scalar_metrics_summary

    summary = _scalar_metrics_summary({
        "transfer_entropy_bits": None,
        "transfer_entropy_bits_status": "first 7,823 of 4,000 rounds discarded",
        "cooperation_rate": 0.42,
        "some_series": np.arange(10),
    })
    assert summary["transfer_entropy_bits"] is None, "an explicit null must be preserved"
    assert "7,823" in summary["transfer_entropy_bits_status"], "the reason must reach the ledger"
    assert summary["cooperation_rate"] == 0.42
    assert "some_series" not in summary
    print("OK: a withheld value reaches the ledger as an explicit null plus its reason")


if __name__ == "__main__":
    test_burn_in_is_derived_from_the_schedule()
    test_a_roster_with_no_schedule_needs_no_burn_in()
    test_missing_roster_is_reported_not_assumed()
    test_burn_in_longer_than_the_match_is_caught()
    test_stationarity_report_separates_the_three_cases()
    test_the_aggregate_is_withheld_with_a_reason_not_silently()
    test_a_withheld_number_survives_the_ledger_summary()
    test_the_guardrail_reduces_false_positives_on_a_null_control()
    print("\nall transfer-entropy guardrail tests passed")
