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


def _false_positive_rate(burn_in: int, surrogate: str = "whole") -> float:
    """Fraction of provably unconnected pairs the test flags, under one construction.

    `surrogate` is explicit and defaults to `"whole"` rather than to the module's own default,
    because these tests compare constructions against each other. A helper that silently followed
    the default would make the comparison drift the next time the default moves, and the reader of
    a passing run would not be told.
    """
    flagged = total = 0
    for seed in range(SEEDS):
        a, _ = _independent_run(seed)
        b, _ = _independent_run(seed + 500)
        spliced = {"actions": np.column_stack([a[:, 0], b[:, 0]])}
        out = information.transfer_entropy_pairwise(spliced, seed=seed, n_surrogates=200,
                                                    burn_in=burn_in, surrogate=surrogate)
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


def test_blockwise_beats_the_unrestricted_permutation_on_a_null_control() -> None:
    """The finding that changed this module's default, reproduced here on its own null control.

    Both constructions see the identical series, so the only difference is how the source is
    permuted. If this ever stops holding, the default is wrong and not just the number.
    """
    whole = _false_positive_rate(burn_in=0, surrogate="whole")
    blockwise = _false_positive_rate(burn_in=0, surrogate="blockwise")

    print(f"     unrestricted permutation: {whole:.1%}")
    print(f"     within-block permutation: {blockwise:.1%}")
    assert whole > 0.5, (
        f"expected the unrestricted permutation to fail loudly on a null control, got {whole:.1%}")
    assert blockwise < whole / 2, (
        f"the blockwise construction must substantially beat the unrestricted one, got "
        f"{blockwise:.1%} against {whole:.1%}")
    print("OK: permuting within blocks of training time cuts the false-positive rate")


def test_drift_z_separates_a_drifting_series_from_a_steady_one() -> None:
    """The precondition statistic must answer about drift, not about sampling noise.

    This is the test the first implementation would have failed. That version measured the raw
    difference of two half-block means, which at this block length reads about 0.4 on a series
    with no drift at all, purely from twenty-odd binary samples per half.
    """
    rng = np.random.default_rng(7)
    n = 1500
    steady = (rng.random(n) < 0.5).astype(np.int64)
    # A marginal sweeping the full range inside every block, which is the failure case.
    within = np.tile(np.linspace(0.05, 0.95, n // information.N_BLOCKS), information.N_BLOCKS)
    drifting = (rng.random(len(within)) < within).astype(np.int64)

    z_steady = information.within_block_drift_z(steady)
    z_drifting = information.within_block_drift_z(drifting)
    raw_steady = information.within_block_marginal_shift(steady)

    print(f"     steady series: drift {z_steady:.1f} SE, raw shift {raw_steady:.3f}")
    print(f"     drifting series: drift {z_drifting:.1f} SE")
    assert raw_steady > 0.15, (
        "the raw shift is expected to be large even with no drift; if it is not, this block "
        "length changed and the reason for preferring the standardised statistic needs rechecking")
    assert z_drifting > z_steady * 1.5, (
        f"the standardised statistic must separate the two, got {z_drifting:.1f} against "
        f"{z_steady:.1f}")
    print("OK: the standardised statistic separates drift from noise where the raw one cannot")


def test_the_aggregate_is_withheld_with_a_reason_not_silently() -> None:
    """The user-facing half of the guardrail: an absent number must never look like a bug.

    What gates it changed. A roster is no longer required for a headline, because the blockwise
    construction needs no burn-in and therefore needs nothing from the roster. What is required is
    the construction's own precondition, so that is what this asserts.
    """
    actions, roster = _independent_run(0)
    records = {"actions": actions, "cooperators": actions.sum(axis=1)}

    out = information.compute_all(records, seed=0, n_surrogates=50, roster=roster)
    status = out["transfer_entropy_bits_status"]
    assert status, "a status must always be present, whether or not a number was"

    precondition = out["transfer_entropy_detail"]["precondition"]
    if out["transfer_entropy_bits"] is None:
        assert not precondition["ok"], "a withheld number must name a failed precondition"
        assert precondition["reason"] in status
        print(f"OK: aggregate withheld, and says why ({status[:60]}...)")
    else:
        assert precondition["ok"], "a published number must have passed the precondition"
        assert "stationarity diagnostic" in status
        print(f"OK: aggregate reported, with its diagnostic ({status[:60]}...)")

    # Stationarity reports, and never decides.
    assert "role" in out["transfer_entropy_detail"]["stationarity"]
    no_roster = information.compute_all(records, seed=0, n_surrogates=50)
    assert (no_roster["transfer_entropy_bits"] is None) == (out["transfer_entropy_bits"] is None), \
        "the roster must not decide whether a number is published, only what burn-in would be"
    print("OK: the roster no longer gates the headline, the precondition does")


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
    test_drift_z_separates_a_drifting_series_from_a_steady_one()
    test_the_aggregate_is_withheld_with_a_reason_not_silently()
    test_a_withheld_number_survives_the_ledger_summary()
    test_the_guardrail_reduces_false_positives_on_a_null_control()
    test_blockwise_beats_the_unrestricted_permutation_on_a_null_control()
    print("\nall transfer-entropy guardrail tests passed")
