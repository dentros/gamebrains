"""Conditioning on training time, as a second route when the surrogate can only refuse.

Two agents learning on the same schedule drift together. An unconditional transfer entropy reads
that shared drift as influence, and on a null control where no influence exists it flags
essentially every pair. The module's existing answer is a blockwise surrogate plus a precondition
that refuses to publish when its own assumption fails, which on a short default run means
publishing nothing.

Conditioning answers the same question differently. Instead of cutting the transient away or
declining, it puts the known driver into the statistic: T(X->Y | Z) with Z a discretisation of
training time. The companion study measured the false-positive rate collapsing from 100% to the
nominal rate as the blocks get finer, over the full untrimmed series, and the tests below pin
that behaviour rather than any particular number.

**Why the assertions are about direction rather than values.** The exact rate depends on the
generator, the round count and the surrogate count, so pinning 6.0% would make this test a
record of one machine's run. What has to hold is that the unconditional statistic fails badly on
a drifting null, that conditioning fixes it, and that it does not achieve this by reporting
nothing at all.

Run:  "$GB/gamebrains/.venv/Scripts/python.exe" -m gamebrains.tests.test_conditional_te
"""

from __future__ import annotations

import sys

import numpy as np

from ..metrics.information import (N_BLOCKS, _conditional_te_bits,
                                   conditional_transfer_entropy_pairwise)

TRIALS = 24
SURROGATES = 60
ROUNDS = 1500


def _drifting_null(seed: int, rounds: int = ROUNDS) -> dict[str, np.ndarray]:
    """Two independent agents whose bias drifts on a shared schedule.

    Nothing flows from one to the other, so every pair this flags is a false positive. The drift
    stands in for exploration decaying over training, which is the real confound in a match.
    """
    rng = np.random.default_rng(seed)
    p = 0.15 + 0.7 * (np.arange(rounds) / rounds)
    a = (rng.random(rounds) < p).astype(np.int64)
    b = (rng.random(rounds) < p).astype(np.int64)
    return {"actions": np.column_stack([a, b])}


def _false_positive_rate(n_blocks: int) -> float:
    flagged = 0
    for seed in range(TRIALS):
        result = conditional_transfer_entropy_pairwise(
            _drifting_null(seed), seed=seed, n_surrogates=SURROGATES, n_blocks=n_blocks)
        flagged += result["n_significant_pairs"]
    return flagged / (TRIALS * 2)


def test_without_conditioning_a_drifting_null_is_flagged_almost_always() -> None:
    """One block is the unconditional statistic, and it is what the second route exists for."""
    rate = _false_positive_rate(1)
    assert rate > 0.5, (
        f"the unconditional statistic flagged only {rate:.0%} of a null control built entirely "
        f"from shared drift. If this has become well behaved, the second route is solving a "
        f"problem that no longer exists and should be reconsidered rather than kept")
    print(f"OK: unconditioned, the drifting null is flagged {rate:.0%} of the time")


def test_conditioning_brings_it_back_to_about_nominal() -> None:
    rate = _false_positive_rate(N_BLOCKS)
    assert rate <= 0.15, (
        f"conditioning on {N_BLOCKS} training-time blocks left {rate:.0%} false positives, which "
        f"is not close to the 5% the test is run at")
    print(f"OK: conditioned on {N_BLOCKS} blocks, false positives fall to {rate:.0%}")


def test_finer_blocks_do_not_make_it_worse() -> None:
    """The rate should fall as training time is held constant more tightly, not wander."""
    coarse = _false_positive_rate(4)
    fine = _false_positive_rate(N_BLOCKS)
    assert fine <= coarse + 0.05, (
        f"going from 4 blocks to {N_BLOCKS} moved the false-positive rate from {coarse:.0%} to "
        f"{fine:.0%}, so the block count is not doing what it is supposed to do")
    print(f"OK: 4 blocks {coarse:.0%} against {N_BLOCKS} blocks {fine:.0%}")


def test_a_real_link_is_still_found() -> None:
    """Guards the guard: a statistic that always returns nothing would pass the tests above.

    The copier reproduces the source's previous move, so there is a genuine lag-1 dependence on
    top of the same shared drift.
    """
    found = 0
    for seed in range(TRIALS):
        rng = np.random.default_rng(1000 + seed)
        p = 0.15 + 0.7 * (np.arange(ROUNDS) / ROUNDS)
        source = (rng.random(ROUNDS) < p).astype(np.int64)
        target = np.empty(ROUNDS, dtype=np.int64)
        target[0] = int(rng.random() < p[0])
        for t in range(1, ROUNDS):
            # Follows the source most of the time, and otherwise drifts like everybody else.
            target[t] = source[t - 1] if rng.random() < 0.8 else int(rng.random() < p[t])
        records = {"actions": np.column_stack([source, target])}
        result = conditional_transfer_entropy_pairwise(
            records, seed=seed, n_surrogates=SURROGATES, n_blocks=N_BLOCKS)
        if result["by_pair"][(0, 1)]["p_value"] < 0.05:
            found += 1
    rate = found / TRIALS
    assert rate > 0.8, (
        f"a deliberately planted link was detected in only {rate:.0%} of trials, so the low "
        f"false-positive rate above may simply be a statistic that never reports anything")
    print(f"OK: a planted link is still detected {rate:.0%} of the time")


def test_the_estimator_is_zero_when_the_source_adds_nothing() -> None:
    """With the target independent of the source given its own past, the statistic is near zero."""
    rng = np.random.default_rng(7)
    n = 4000
    x = rng.integers(0, 2, n)
    y = rng.integers(0, 2, n)
    z = np.zeros(n, dtype=np.int64)
    bits = _conditional_te_bits(x[:-1], y[:-1], y[1:], z[:-1], 1)
    assert abs(bits) < 0.01, f"independent series gave {bits:.4f} bits"
    print(f"OK: independent series give {bits:+.4f} bits")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}\n     {exc}")
    print(f"\n{'all conditional-TE tests passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
