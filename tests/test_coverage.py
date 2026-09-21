"""Reading the whole repository at once: where the holes are, and what moves with what.

The dangerous thing about a correlation matrix is that it always produces something. Over twenty-six
measures it is three hundred and twenty-five tests, most of the records differ only by seed, and a
page that printed the strongest cell would print noise on an empty ledger and look exactly the same
as one that had found something. So the tests here are mostly about what the module refuses to say:

  * runs that differ only by seed collapse to one point, because twenty seeds of one design are one
    point measured twenty times
  * a pair below the minimum is reported with its reason instead of a coefficient
  * ties count toward a permutation p-value, which is the difference between a p of 0.022 and a p
    of 0.0005 on a column with few distinct values
  * the correction runs over the whole matrix, so a cell clears a bar set by every other cell

Run: python -m gamebrains.tests.test_coverage
"""

from __future__ import annotations

from itertools import permutations
from typing import Any

from ..repository import coverage


def _row(config: str, metrics: dict[str, Any], roster: list[dict[str, Any]] | None = None,
         game: str = "public_goods", **fields: Any) -> dict[str, Any]:
    """An analytics row, in the shape `webui/app.py::_analytics_row` produces."""
    return {"config_hash": config, "protocol": "scoring=v1", "game_name": game,
            "metrics": metrics, "roster": roster or [], **fields}


def test_the_field_list_is_read_from_the_data_and_not_from_a_list() -> None:
    """A metric or an agent parameter added later has to appear without editing this module."""
    rows = [
        _row("a", {"cooperation_rate": 0.3, "a_measure_invented_tomorrow": 7.0},
             [{"kind": "qlearning", "params": {"alpha": 0.1}}], n_agents=4),
        _row("b", {"cooperation_rate": 0.9},
             [{"kind": "brand_new_kind", "params": {"whatever": 2.0}}], n_agents=5),
    ]
    found = dict(coverage.numeric_fields(rows))

    assert found["metric:a_measure_invented_tomorrow"] == 1
    assert found["roster:brand_new_kind:whatever"] == 1
    assert found["metric:cooperation_rate"] == 2
    assert found["n_agents"] == 2
    print("OK: fields are discovered from the ledger, so the platform can grow without this file")


def test_a_flag_and_a_status_string_are_not_quantities() -> None:
    """Booleans are integers in Python. A True averaged with a False is 0.5, which would enter a
    correlation looking exactly like a measurement."""
    rows = [_row("a", {"precondition_overridden": True, "status": "withheld", "coop": 0.3}),
            _row("b", {"precondition_overridden": False, "status": "ok", "coop": 0.9})]
    names = dict(coverage.numeric_fields(rows))

    assert "metric:precondition_overridden" not in names
    assert "metric:status" not in names
    assert "metric:coop" in names
    print("OK: a flag and a withheld-reason string stay out of the numbers")


def test_twenty_seeds_of_one_design_are_one_point() -> None:
    """The choice the whole module turns on."""
    rows = [_row("same", {"coop": value}) for value in (0.1, 0.2, 0.3, 0.4)]
    rows.append(_row("other", {"coop": 0.9}))

    points = coverage.collapse_to_designs(rows, ["metric:coop"])
    assert len(points) == 2, points
    assert sorted(round(p["metric:coop"], 6) for p in points) == [0.25, 0.9]

    assert len(coverage.as_points(rows, ["metric:coop"], unit="run")) == 5
    print("OK: seeds of one design collapse to their mean, and the run view still exists")


def test_a_roster_parameter_is_the_roster_s_and_not_seat_zero_s() -> None:
    rows = [_row("a", {}, [{"kind": "dqn", "params": {"lr": 0.001}},
                           {"kind": "dqn", "params": {"lr": 0.003}},
                           {"kind": "fep", "params": {"lr": 99.0}}])]
    assert coverage.field_values(rows[0], "roster:dqn:lr") == [0.001, 0.003]
    assert coverage._row_value(rows[0], "roster:dqn:lr") == 0.002
    print("OK: a per-kind parameter averages over that kind's seats, not over the whole roster")


def test_no_variation_is_said_rather_than_reported_as_zero() -> None:
    assert coverage.pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None

    rows = [_row(str(i), {"flat": 1.0, "moves": float(i)}) for i in range(8)]
    matrix = coverage.correlation_matrix(rows, ["metric:flat", "metric:moves"], permutations=200)
    cell = matrix.get("metric:flat", "metric:moves")

    assert cell.r is None and "no variation" in cell.note
    print("OK: a constant column is named as constant instead of correlating at zero")


def test_a_pair_below_the_minimum_carries_its_reason() -> None:
    rows = [_row(str(i), {"x": float(i), "y": float(i)}) for i in range(3)]
    matrix = coverage.correlation_matrix(rows, ["metric:x", "metric:y"], permutations=200)
    cell = matrix.get("metric:x", "metric:y")

    assert cell.r is None and cell.n == 3
    assert "6 designs needed, 3 present" in cell.note
    assert matrix.n_tested == 0
    print("OK: too little data is reported as too little data, with the counts")


def test_ties_count_toward_the_p_value() -> None:
    """A column with few distinct values produces permutations whose statistic is genuinely equal
    to the observed one. They are not less extreme, and dropping them makes the p-value smaller
    than the data supports. Checked against the exact enumeration over all 720 arrangements."""
    xs = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    ys = list(xs)
    r = coverage.pearson(xs, ys)
    exact = sum(1 for perm in permutations(ys)
                if abs(coverage.pearson(xs, list(perm))) >= abs(r) - 1e-12) / 720

    p = coverage.permutation_p(xs, ys, r, permutations=4000)
    assert abs(p - exact) < 0.01, (p, exact)
    assert p > 0.005, "the tie-blind version of this returns about 0.0005"
    print(f"OK: a perfect correlation on a three-valued column gives p = {p:.4f}, "
          f"against an exact {exact:.4f}")


def test_the_correction_is_applied_over_the_whole_matrix() -> None:
    """Step-up, not a threshold per cell: a p-value above its own rank's bar survives when a
    later one passes, and the family is every pair drawn."""
    assert coverage.benjamini_hochberg([0.001, 0.02, 0.03, 0.5], q=0.05) == [True, True, True, False]
    assert coverage.benjamini_hochberg([0.01, 0.04, 0.04, 0.04, 0.04], q=0.05) == [True] * 5
    assert coverage.benjamini_hochberg([0.2, 0.3], q=0.05) == [False, False]
    assert coverage.benjamini_hochberg([], q=0.05) == []
    print("OK: Benjamini-Hochberg steps up over the family rather than testing cells one by one")


def test_a_real_relationship_survives_and_noise_does_not() -> None:
    """The matrix still has to find something when something is there."""
    rows = []
    for i in range(14):
        x = float(i)
        rows.append(_row(f"design-{i}", {"driver": x, "follows": 2.0 * x + 1.0,
                                         "unrelated": float((i * 7919) % 13)}))
    fields = ["metric:driver", "metric:follows", "metric:unrelated"]
    matrix = coverage.correlation_matrix(rows, fields, permutations=2000)

    strong = matrix.get("metric:driver", "metric:follows")
    assert strong.r > 0.99 and strong.significant, strong
    assert matrix.n_points == 14 and matrix.n_tested == 3
    print(f"OK: an exact relationship is found (r = {strong.r:.2f}, p = {strong.p:.4f}) "
          f"and {matrix.n_significant} of {matrix.n_tested} pairs survive")


def test_the_board_reports_the_squares_nobody_has_played() -> None:
    """The empty squares are the point of the view."""
    rows = [_row("a", {"coop": 0.3}, [{"kind": "qlearning", "params": {}}], game="public_goods"),
            _row("b", {"coop": 0.4}, [{"kind": "dqn", "params": {}}], game="public_goods"),
            _row("c", {"coop": 0.5}, [{"kind": "qlearning", "params": {}}], game="congestion")]
    grid = coverage.coverage_grid(rows)

    assert grid["games"] == ["public_goods", "congestion"]
    assert set(grid["kinds"]) == {"qlearning", "dqn"}
    assert grid["total"] == 4 and grid["filled"] == 3
    assert ("congestion", "dqn") not in grid["cells"]
    assert grid["cells"][("public_goods", "qlearning")]["designs"] == 1
    print("OK: the board says three squares of four, and which one is empty")


if __name__ == "__main__":
    test_the_field_list_is_read_from_the_data_and_not_from_a_list()
    test_a_flag_and_a_status_string_are_not_quantities()
    test_twenty_seeds_of_one_design_are_one_point()
    test_a_roster_parameter_is_the_roster_s_and_not_seat_zero_s()
    test_no_variation_is_said_rather_than_reported_as_zero()
    test_a_pair_below_the_minimum_carries_its_reason()
    test_ties_count_toward_the_p_value()
    test_the_correction_is_applied_over_the_whole_matrix()
    test_a_real_relationship_survives_and_noise_does_not()
    test_the_board_reports_the_squares_nobody_has_played()
    print("\nall coverage tests passed")
