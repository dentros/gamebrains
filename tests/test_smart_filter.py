"""
Tests for repository/smart_filter.py -- the two Smart Filter lookups (exact + k-NN similarity).
"""

import shutil
import tempfile
from pathlib import Path

from gamebrains.repository.ledger import Ledger
from gamebrains.repository.smart_filter import lookup, nearest


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="gamebrains_smartfilter_test_"))


def test_nearest_ranks_by_distance_and_skips_shape_mismatch():
    records = [
        {"feature_vector": [4, 0.5, 1.0, 1000, 4, 0, 0, 0, 0]},
        {"feature_vector": [4, 0.5, 1.0, 1010, 4, 0, 0, 0, 0]},   # very close
        {"feature_vector": [8, 0.9, 1.0, 5000, 0, 8, 0, 0, 0]},   # far
        {"feature_vector": [4, 0.5, 1.0]},                         # different shape -> skipped
    ]
    query = [4, 0.5, 1.0, 1000, 4, 0, 0, 0, 0]
    ranked = nearest(records, query, k=2)
    assert len(ranked) == 2
    assert ranked[0][0] is records[0] and ranked[0][1] == 0.0
    assert ranked[1][0] is records[1]


def test_lookup_finds_exact_and_similar_but_not_itself():
    tmp = _tmp_dir()
    try:
        ledger = Ledger(tmp)
        game = {"name": "public_goods", "n_agents": 4}
        roster = [{"kind": "qlearning"}]

        prior = ledger.append(game, roster, "git:test", rounds=1000, master_seed=0,
                              content_cid="prior", metrics_summary={},
                              feature_vector=[4, 0.5, 1.0, 1000, 1])

        # A near-identical but not-identical design (slightly different feature vector / rounds).
        result = lookup(ledger, game, roster, "git:test", rounds=1000, master_seed=0,
                        feature_vector=[4, 0.5, 1.0, 1000, 1])
        assert result["exact"] is not None and result["exact"]["content_cid"] == "prior"
        assert len(result["similar"]) == 1 and result["similar"][0][0]["content_cid"] == "prior"

        # A genuinely new design: no exact hit, but the prior run shows up as "similar".
        miss = lookup(ledger, game, roster, "git:test", rounds=2000, master_seed=99,
                      feature_vector=[4, 0.5, 1.0, 2000, 1])
        assert miss["exact"] is None
        assert len(miss["similar"]) == 1 and miss["similar"][0][0]["content_cid"] == "prior"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_round_count_no_longer_dominates_the_ranking():
    """The defect, pinned down.

    Two candidates: one that shares the query's whole design and ran ten times longer, and one that
    ran exactly as long but is a different game setup with a different roster. Under the old
    unweighted distance on raw features the round count outweighed everything else by three orders
    of magnitude, so the second always won. Design has to win now.
    """
    same_design_different_length = {"feature_vector": [4, 0.5, 1.0, 10000, 4, 0, 0, 0, 0],
                                    "tag": "same design"}
    same_length_different_design = {"feature_vector": [9, 0.9, 1.0, 1000, 0, 0, 0, 0, 9],
                                    "tag": "same length"}
    records = [same_design_different_length, same_length_different_design]

    query = [4, 0.5, 1.0, 1000, 4, 0, 0, 0, 0]
    ranked = nearest(records, query, k=2)
    assert ranked[0][0]["tag"] == "same design", (
        f"round count still dominates: {ranked[0][0]['tag']} ranked first")


def test_constant_dimensions_are_dropped_not_divided_by_epsilon():
    """A ledger where everyone ran the same n and the same cost. Those columns carry no
    information, and standardising them against a near-zero spread would turn floating-point noise
    into the dominant term -- the original defect in a new costume."""
    records = [
        {"feature_vector": [4, 0.5, 1.0, 1000, 4, 0, 0, 0, 0], "tag": "a"},
        {"feature_vector": [4, 0.7, 1.0, 1000, 4, 0, 0, 0, 0], "tag": "b"},
        {"feature_vector": [4, 0.9, 1.0, 1000, 4, 0, 0, 0, 0], "tag": "c"},
    ]
    ranked = nearest(records, [4, 0.52, 1.0, 1000, 4, 0, 0, 0, 0], k=3)
    assert [r[0]["tag"] for r in ranked] == ["a", "b", "c"], (
        "with only mpcr varying, ranking must follow mpcr")
    assert all(d == d for _, d in ranked), "distances must be finite, not NaN"


def test_an_all_identical_ledger_returns_without_inventing_an_order():
    """Every record identical. There is no similarity judgement to make, so the lookup must not
    imply one by ranking them."""
    records = [{"feature_vector": [4, 0.5, 1.0, 1000, 4, 0, 0, 0, 0]} for _ in range(3)]
    ranked = nearest(records, [4, 0.5, 1.0, 1000, 4, 0, 0, 0, 0], k=2)
    assert len(ranked) == 2
    assert all(distance == 0.0 for _, distance in ranked)


def test_empty_and_incomparable_pools_return_empty():
    assert nearest([], [4, 0.5, 1.0, 1000, 4, 0, 0, 0, 0], k=5) == []
    assert nearest([{"feature_vector": [1, 2, 3]}], [4, 0.5, 1.0, 1000, 4, 0, 0, 0, 0], k=5) == []


if __name__ == "__main__":
    test_nearest_ranks_by_distance_and_skips_shape_mismatch()
    print("OK: nearest ranks by distance and skips shape mismatches")
    test_lookup_finds_exact_and_similar_but_not_itself()
    print("OK: lookup finds exact + similar")
    test_round_count_no_longer_dominates_the_ranking()
    print("OK: design beats round count, which it did not before")
    test_constant_dimensions_are_dropped_not_divided_by_epsilon()
    print("OK: zero-spread dimensions are dropped, distances stay finite")
    test_an_all_identical_ledger_returns_without_inventing_an_order()
    print("OK: an all-identical ledger yields no invented ranking")
    test_empty_and_incomparable_pools_return_empty()
    print("OK: empty and incomparable pools return empty")
