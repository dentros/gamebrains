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


if __name__ == "__main__":
    test_nearest_ranks_by_distance_and_skips_shape_mismatch()
    print("OK: nearest ranks by distance and skips shape mismatches")
    test_lookup_finds_exact_and_similar_but_not_itself()
    print("OK: lookup finds exact + similar")
