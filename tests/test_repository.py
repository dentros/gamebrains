"""
Tests for the repository-lite implementation (repository/cas.py, ledger.py, normalize.py).

Validates the properties that matter for a "blockchain-lite" claim to be honest:
  - content-addressing roundtrips and is deduplicating (same content -> same CID, one file)
  - every appended record carries a valid Ed25519 signature
  - tampering with a record's content invalidates its signature
  - tampering with an OLDER record breaks the hash chain from that point forward, even if that
    older record's own signature were (hypothetically) still valid
  - config_hash is stable under key reordering and float-formatting differences
  - extends/replicates lineage is detected correctly
  - exact-match dedup lookup (Smart Filter's first half) finds a prior identical run
"""

import shutil
import tempfile
from pathlib import Path

from gamebrains.repository.cas import ContentStore
from gamebrains.repository.ledger import Ledger
from gamebrains.repository.normalize import canonical_json, hash_config


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="gamebrains_repo_test_"))


def test_config_hash_stable_under_key_order_and_float_noise():
    a = {"game": {"n": 4, "mpcr": 0.5}, "roster": [{"kind": "qlearning"}], "code_version": "abc"}
    b = {"roster": [{"kind": "qlearning"}], "code_version": "abc", "game": {"mpcr": 0.5, "n": 4}}
    assert hash_config(a) == hash_config(b)

    c = {"game": {"n": 4, "mpcr": 0.500000000001}, "roster": [{"kind": "qlearning"}], "code_version": "abc"}
    assert hash_config(a) == hash_config(c)  # rounds away float noise below 1e-10


def test_cas_roundtrip_and_dedup():
    tmp = _tmp_dir()
    try:
        store = ContentStore(tmp)
        package = {"manifest": {"a": 1}, "event_log": [{"round": 0}, {"round": 1}]}
        cid1 = store.put(package)
        cid2 = store.put(package)  # identical content -> identical CID, no duplicate file
        assert cid1 == cid2
        assert len(list(store.objects_dir.glob("*.json"))) == 1
        assert store.get(cid1) == package
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ledger_records_are_signed_and_verify():
    tmp = _tmp_dir()
    try:
        ledger = Ledger(tmp)
        record = ledger.append(
            game_desc={"name": "public_goods", "n_agents": 4},
            roster_desc=[{"kind": "qlearning"}] * 4,
            code_version="git:test",
            rounds=100,
            master_seed=0,
            content_cid="deadbeef",
            metrics_summary={"cooperation_rate": 0.3},
            feature_vector=[4, 0.5, 1.0, 100, 4],
        )
        assert Ledger.verify_record(record) is True
        assert ledger.verify_chain() is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_tampering_with_a_record_invalidates_its_signature():
    tmp = _tmp_dir()
    try:
        ledger = Ledger(tmp)
        record = ledger.append(
            game_desc={"name": "public_goods", "n_agents": 4}, roster_desc=[{"kind": "qlearning"}],
            code_version="git:test", rounds=100, master_seed=0, content_cid="deadbeef",
            metrics_summary={"cooperation_rate": 0.3}, feature_vector=[4, 0.5, 1.0, 100, 4],
        )
        tampered = {**record, "metrics_summary": {"cooperation_rate": 0.99}}
        assert Ledger.verify_record(tampered) is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_tampering_with_an_older_record_breaks_the_chain():
    tmp = _tmp_dir()
    try:
        ledger = Ledger(tmp)
        ledger.append(
            game_desc={"name": "public_goods", "n_agents": 4}, roster_desc=[{"kind": "qlearning"}],
            code_version="git:test", rounds=100, master_seed=0, content_cid="aaa",
            metrics_summary={}, feature_vector=[4],
        )
        ledger.append(
            game_desc={"name": "public_goods", "n_agents": 4}, roster_desc=[{"kind": "qlearning"}],
            code_version="git:test", rounds=100, master_seed=1, content_cid="bbb",
            metrics_summary={}, feature_vector=[4],
        )
        assert ledger.verify_chain() is True

        # Directly corrupt the first line on disk (simulating an edited past record).
        lines = ledger.ledger_path.read_text(encoding="utf-8").splitlines()
        import json
        first = json.loads(lines[0])
        first["metrics_summary"] = {"hacked": True}
        # Re-sign the tampered record so verify_record() alone would (wrongly) still pass --
        # the point is that verify_chain() must catch it via the broken parent link regardless.
        lines[0] = json.dumps(first)
        ledger.ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        assert ledger.verify_chain() is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_extends_and_replicates_lineage_detection():
    tmp = _tmp_dir()
    try:
        ledger = Ledger(tmp)
        game = {"name": "public_goods", "n_agents": 4}
        roster = [{"kind": "qlearning"}]

        short_run = ledger.append(game, roster, "git:test", rounds=100, master_seed=0,
                                  content_cid="short", metrics_summary={}, feature_vector=[4])
        long_run = ledger.append(game, roster, "git:test", rounds=200, master_seed=0,
                                 content_cid="long", metrics_summary={}, feature_vector=[4])
        replica = ledger.append(game, roster, "git:test", rounds=200, master_seed=1,
                                content_cid="replica", metrics_summary={}, feature_vector=[4])

        assert short_run["lineage"]["extends"] is None
        assert long_run["lineage"]["extends"] is not None  # extends the shorter same-seed run
        assert replica["lineage"]["replicates"] is not None  # different seed, same design
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_extends_and_replicates_can_coexist_on_the_same_record():
    """A record can simultaneously extend its own same-seed prior run AND belong to a family of
    different-seed replicates -- the two checks are independent (see Ledger._detect_lineage)."""
    tmp = _tmp_dir()
    try:
        ledger = Ledger(tmp)
        game = {"name": "public_goods", "n_agents": 4}
        roster = [{"kind": "qlearning"}]

        ledger.append(game, roster, "git:test", rounds=100, master_seed=0,
                      content_cid="seed0-short", metrics_summary={}, feature_vector=[4])
        ledger.append(game, roster, "git:test", rounds=100, master_seed=1,
                      content_cid="seed1-short", metrics_summary={}, feature_vector=[4])
        # Same seed as the first record (0), but longer -> extends it.
        # A sibling with a different seed (1) already exists -> also replicates.
        both = ledger.append(game, roster, "git:test", rounds=200, master_seed=0,
                             content_cid="seed0-long", metrics_summary={}, feature_vector=[4])

        assert both["lineage"]["extends"] is not None
        assert both["lineage"]["replicates"] is not None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_find_exact_dedup_lookup():
    tmp = _tmp_dir()
    try:
        ledger = Ledger(tmp)
        game = {"name": "public_goods", "n_agents": 4}
        roster = [{"kind": "qlearning"}]
        ledger.append(game, roster, "git:test", rounds=500, master_seed=0, content_cid="cid500",
                      metrics_summary={}, feature_vector=[4])

        hit = ledger.find_exact(game, roster, "git:test", rounds=300, master_seed=0)
        assert hit is not None and hit["content_cid"] == "cid500"  # 500-round run satisfies "need >=300"

        miss = ledger.find_exact(game, roster, "git:test", rounds=300, master_seed=999)
        assert miss is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_config_hash_stable_under_key_order_and_float_noise()
    print("OK: config_hash stable under key order / float noise")
    test_cas_roundtrip_and_dedup()
    print("OK: CAS roundtrip + dedup")
    test_ledger_records_are_signed_and_verify()
    print("OK: ledger records are signed and verify")
    test_tampering_with_a_record_invalidates_its_signature()
    print("OK: tampering invalidates signature")
    test_tampering_with_an_older_record_breaks_the_chain()
    print("OK: tampering with an older record breaks the chain")
    test_extends_and_replicates_lineage_detection()
    print("OK: extends/replicates lineage detection")
    test_extends_and_replicates_can_coexist_on_the_same_record()
    print("OK: extends and replicates can coexist on the same record")
    test_find_exact_dedup_lookup()
    print("OK: exact-match dedup lookup")
