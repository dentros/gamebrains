"""The payoff Gini must measure inequality, and a record must say which Gini it reports.

The platform used to shift the payoff vector up by its own minimum before computing the Gini, to
cope with the negative payoffs a social dilemma produces. That made the measure translation
invariant: two players on 100 and 100.01 read 0.5, exactly as two players on 0 and 1 do. An
external review of the code found it. No test had, because none checked a vector whose minimum was
not already close to zero.

`metrics/social.py` now uses the normalisation of Raffinetti, Siletti and Vernizzi (2015), which is
the ordinary Gini on non-negative data and stays within [0, (n-1)/n] with negatives. The values
below are computed by hand, so the module cannot agree with them by construction.

The second half is about stored records. Every record written before the fix carries a number
computed under the old definition, and `repository/verify.py` recomputes stored numbers from their
logs. Checking an old record against the new definition would report it as altered when nothing
was, so a record names its definition and the checker uses the one named.

Run:  "$GB/gamebrains/.venv/Scripts/python.exe" -m gamebrains.tests.test_gini
"""

from __future__ import annotations

import sys

import numpy as np

from ..agents.classic import AllC, AllD
from ..engine.runner import run_match
from ..games.public_goods import PublicGoodsGame
from ..metrics import social
from ..repository import verify
from ..repository.record import _scalar_metrics_summary

TIGHT = 1e-12


def _close(a: float, b: float, tol: float = TIGHT) -> bool:
    return abs(a - b) <= tol


def test_equal_payoffs_read_zero() -> None:
    for vector in ([5.0, 5.0, 5.0], [-3.0, -3.0], [0.0, 0.0, 0.0, 0.0]):
        value = social.payoff_gini(np.array(vector))
        assert _close(value, 0.0), f"{vector} are all equal and read {value}"
    print("OK: equal payoffs read 0, positive, negative or zero")


def test_a_shift_of_the_whole_vector_is_not_ignored() -> None:
    """The review's case. Two players almost exactly level must read almost exactly equal."""
    value = social.payoff_gini(np.array([100.0, 100.01]))
    assert value < 1e-4, (
        f"two players on 100 and 100.01 read {value:.6f}; a Gini near 0.5 here means the vector "
        "was shifted by its minimum, which turns this pair into 0 and 0.01")
    print(f"OK: 100 against 100.01 reads {value:.2e}, not 0.5")


def test_ordinary_gini_on_non_negative_payoffs() -> None:
    """With no negative payoff the measure is the textbook one, mean absolute difference over
    twice the mean, computed here over pairs rather than by the sorted closed form."""
    assert _close(social.payoff_gini(np.array([0.0, 0.0, 0.0, 1.0])), 0.75)
    rng = np.random.default_rng(7)
    for _ in range(50):
        x = rng.uniform(0.0, 50.0, size=int(rng.integers(2, 9)))
        textbook = np.abs(x[:, None] - x[None, :]).mean() / (2.0 * x.mean())
        assert _close(social.payoff_gini(x), float(textbook), 1e-10), x
    print("OK: on non-negative payoffs it is the ordinary Gini, checked on 50 random vectors")


def test_hand_computed_values_with_negatives() -> None:
    """G = sum_i sum_j |x_i - x_j| / (2 n (T+ + T-)), worked by hand for each vector."""
    cases = [
        # [-1, 2, 5]: ordered pair differences sum to 24, absolute total 8, so 24 / (2*3*8).
        ([-1.0, 2.0, 5.0], 0.5),
        # One player holds all the gains and one all the losses: the maximum, (n-1)/n.
        ([-2.0, 0.0, 0.0, 2.0], 0.75),
        # [-4, -1]: differences sum to 6, absolute total 5, so 6 / (2*2*5).
        ([-4.0, -1.0], 0.3),
    ]
    for vector, expected in cases:
        value = social.payoff_gini(np.array(vector))
        assert _close(value, expected, 1e-12), f"{vector}: expected {expected}, got {value}"
    print("OK: three hand-worked vectors with negative payoffs, including the maximum")


def test_bounded_by_n_minus_one_over_n() -> None:
    rng = np.random.default_rng(11)
    for _ in range(200):
        n = int(rng.integers(2, 12))
        x = rng.normal(0.0, 30.0, size=n)
        value = social.payoff_gini(x)
        assert -TIGHT <= value <= (n - 1) / n + TIGHT, f"n={n}: {value} outside [0, (n-1)/n]"
    print("OK: 200 random vectors with mixed signs all stay within [0, (n-1)/n]")


def test_the_checker_agrees_with_the_platform() -> None:
    """`verify._gini` is written independently (over pairs, without numpy) so that agreement is
    evidence. If the two ever disagree, one of them is wrong."""
    rng = np.random.default_rng(3)
    for _ in range(100):
        x = rng.normal(5.0, 40.0, size=int(rng.integers(2, 10)))
        ours = social.payoff_gini(x)
        theirs = verify._gini(x.tolist(), "rsv-2015")
        assert _close(ours, theirs, 1e-10), f"{x}: platform {ours}, checker {theirs}"
    print("OK: the independent checker matches the platform on 100 random vectors")


def test_a_new_record_names_its_definition() -> None:
    game = PublicGoodsGame(n_agents=3)
    roster = [AllC("c0"), AllD("d0"), AllD("d1")]
    records = run_match(game, roster, rounds=20, seed=1)
    metrics = social.compute_all(records, game.max_welfare_per_round())
    assert metrics.get("payoff_gini_definition") == social.GINI_DEFINITION == "rsv-2015"
    summary = _scalar_metrics_summary(metrics)
    assert summary.get("payoff_gini_definition") == "rsv-2015", (
        "the definition does not survive into the stored summary, so a reader of the record "
        "cannot tell which Gini it reports")
    print("OK: a new run names rsv-2015 next to its Gini, and the name reaches the record")


def _package_and_totals() -> tuple[dict, list[float]]:
    rewards = [[50.0, 50.005], [50.0, 50.005]]
    package = {"manifest": {}, "event_log": [
        {"type": "round", "round": t, "actions": [1, 0], "rewards": r, "cooperators": 1}
        for t, r in enumerate(rewards)]}
    totals = [sum(r[i] for r in rewards) for i in range(2)]
    return package, totals


def _gini_check(report: "verify.Report") -> "verify.Check":
    found = [c for c in report.checks if c.name == "payoff_gini recomputes"]
    assert found, f"no Gini check ran: {[c.name for c in report.checks]}"
    return found[0]


def test_an_old_record_is_checked_under_the_definition_it_was_made_with() -> None:
    package, totals = _package_and_totals()
    legacy = verify._gini(totals, verify.LEGACY_GINI)
    assert legacy > 0.4, "the fixture no longer exposes the shift, so this test checks nothing"
    record = {"content_cid": verify._canonical_cid(package),
              "metrics_summary": {"payoff_gini": legacy}}
    check = _gini_check(verify.verify(record, package))
    assert check.passed, f"an untouched pre-fix record was reported as altered: {check.detail}"
    print(f"OK: a pre-fix record reporting {legacy:.3f} still verifies, under min-shift")


def test_a_record_naming_the_new_definition_is_held_to_it() -> None:
    package, totals = _package_and_totals()
    legacy = verify._gini(totals, verify.LEGACY_GINI)
    cid = verify._canonical_cid(package)
    stale = {"content_cid": cid, "metrics_summary": {"payoff_gini": legacy,
                                                     "payoff_gini_definition": "rsv-2015"}}
    assert not _gini_check(verify.verify(stale, package)).passed, (
        "a record claiming rsv-2015 while reporting the min-shift number was accepted")
    honest = {"content_cid": cid, "metrics_summary": {
        "payoff_gini": social.payoff_gini(np.array(totals)), "payoff_gini_definition": "rsv-2015"}}
    assert _gini_check(verify.verify(honest, package)).passed
    print("OK: a record naming rsv-2015 passes with the rsv number and fails with the old one")


def test_an_unknown_definition_is_refused_by_name() -> None:
    package, totals = _package_and_totals()
    record = {"content_cid": verify._canonical_cid(package), "metrics_summary": {
        "payoff_gini": 0.1, "payoff_gini_definition": "theil"}}
    report = verify.verify(record, package)
    refused = [c for c in report.checks if c.name == "payoff_gini definition known"]
    assert refused and not refused[0].passed and "theil" in refused[0].detail, (
        "a definition the checker does not know was not reported as such")
    assert not any(c.name == "payoff_gini recomputes" for c in report.checks), (
        "the checker compared a number against a definition it does not implement")
    print("OK: an unknown Gini definition is reported by name rather than checked as another")


if __name__ == "__main__":
    failures = 0
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {fn_name}\n     {exc}")
            except Exception as exc:  # an import-level change shows up here, not as a pass
                failures += 1
                print(f"ERROR {fn_name}\n     {type(exc).__name__}: {exc}")
    print(f"\n{'all Gini tests passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
