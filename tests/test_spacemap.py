"""
Tests for the webui's parameter-space map helpers (webui/app.py): grid enumeration with the
social-dilemma validity constraint, and ledger-row coverage matching. Pure-logic tests -- no
Flask request context and no ledger on disk needed, same spirit as test_graph.py's hand-built
te_detail dicts.
"""

from gamebrains.webui.app import _cell_covered, _parse_num_list, _space_cells


def test_grid_validity_respects_social_dilemma_and_mix_definition():
    cells = _space_cells(ns=[2, 4], mpcrs=[0.4, 0.75], roundss=[500], seeds=[0],
                         mix_keys=["all_qlearning", "one_of_each"])
    by_key = {(c["mix"], c["n"], c["mpcr"]): c for c in cells}

    # mpcr=0.4 violates 1/n < mpcr at n=2 (1/2 = 0.5 > 0.4) but is fine at n=4 (1/4 = 0.25).
    assert not by_key[("all_qlearning", 2, 0.4)]["valid"]
    assert by_key[("all_qlearning", 4, 0.4)]["valid"]
    assert by_key[("all_qlearning", 2, 0.75)]["valid"]

    # one_of_each is only defined at n=4.
    assert not by_key[("one_of_each", 2, 0.75)]["valid"]
    assert by_key[("one_of_each", 4, 0.75)]["valid"]
    assert by_key[("one_of_each", 4, 0.75)]["counts"] == {
        "qlearning": 1, "dqn": 1, "fep": 1, "classic": 1}


def test_cell_coverage_matches_on_coordinates_and_roster_composition():
    cell = {"mix": "all_qlearning", "n": 3, "mpcr": 0.5, "rounds": 500, "seed": 1,
            "valid": True, "counts": {"qlearning": 3}}
    base_row = {"n_agents": 3, "mpcr": 0.5, "rounds": 500, "seed": 1,
                "kind_counts": {"qlearning": 3}}

    assert _cell_covered(cell, [base_row])
    assert not _cell_covered(cell, [{**base_row, "seed": 2}])
    assert not _cell_covered(cell, [{**base_row, "rounds": 1500}])
    assert not _cell_covered(cell, [{**base_row, "kind_counts": {"qlearning": 2, "dqn": 1}}])
    assert not _cell_covered(cell, [{**base_row, "mpcr": None}])
    assert not _cell_covered(cell, [])


def test_parse_num_list_falls_back_and_skips_junk():
    assert _parse_num_list("2, 3,junk, 5", [1], int) == [2, 3, 5]
    assert _parse_num_list("", [1, 2], int) == [1, 2]
    assert _parse_num_list(None, [0.5], float) == [0.5]
    assert _parse_num_list("junk,also junk", [7], int) == [7]


if __name__ == "__main__":
    test_grid_validity_respects_social_dilemma_and_mix_definition()
    print("OK: grid validity respects the social-dilemma constraint and per-mix definitions")
    test_cell_coverage_matches_on_coordinates_and_roster_composition()
    print("OK: cell coverage matches on coordinates and roster composition")
    test_parse_num_list_falls_back_and_skips_junk()
    print("OK: grid-list parsing falls back to defaults and skips junk tokens")
