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

import json
import shutil
import tempfile
from pathlib import Path

from gamebrains.repository.cas import ContentStore
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gamebrains.repository.ledger import Ledger, _record_hash
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
        #
        # This block used to carry a comment saying the record was re-signed, so that the parent
        # link rather than the signature would be what caught it. It never re-signed anything, so
        # verify_chain() returned False at the signature check and the parent link was never
        # reached. The assertion passed for a reason the comment denied. Both cases are separate
        # tests below now, and each states which check it exercises.
        lines = ledger.ledger_path.read_text(encoding="utf-8").splitlines()
        import json
        first = json.loads(lines[0])
        first["metrics_summary"] = {"hacked": True}
        lines[0] = json.dumps(first)
        ledger.ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        assert ledger.verify_chain() is False, "an edited record must fail the signature check"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_resigned_record_is_caught_by_the_parent_link():
    """The case the older test's comment described and its code did not create.

    Here the tampered record really is re-signed, with the ledger's own key, so the signature
    check passes and the parent link is the only thing left to catch it. That is the division of
    labour the chain exists for, and it was never actually exercised until this test.
    """
    tmp = tempfile.mkdtemp()
    try:
        ledger = Ledger(root_dir=tmp)
        for i in range(3):
            ledger.append(
                game_desc={"name": "public_goods", "n_agents": 4},
                roster_desc=[{"kind": "qlearning"}], code_version="git:test",
                rounds=100, master_seed=i, content_cid=f"cid{i}",
                metrics_summary={"coop_rate": 0.42}, feature_vector=[4],
            )
        assert ledger.verify_chain() is True

        records = ledger.load_all()
        records[0]["metrics_summary"] = {"coop_rate": 0.99}
        payload = {k: v for k, v in records[0].items() if k != "signature"}
        records[0]["signature"] = ledger._private_key.sign(
            canonical_json(payload).encode("utf-8")).hex()

        # The signature is now genuinely valid, which is the whole point of this test.
        assert Ledger.verify_record(records[0]) is True

        ledger.ledger_path.write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        assert Ledger(root_dir=tmp).verify_chain() is False, (
            "the second record still names the pre-edit hash of the first, so the chain must break")
        print("OK: a validly re-signed edit is caught by the parent link")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_substituted_identity_is_caught_without_knowing_the_real_one():
    """An adversary who never had the signing key can still produce a chain that verifies.

    They generate their own identity, write its public half into `contributor`, and re-sign the
    edited record and everything after it. Every signature is then valid, every parent link is
    correct, and the only thing wrong is who signed. We ran this against the platform before
    `verify_chain` compared contributors, and it reported True on a ledger whose reported metric
    had been changed from 0.42 to 0.99.

    Note what the assertion does *not* claim. Requiring one identity across the chain catches a key
    swapped for part of it. A chain rewritten end to end under a new identity still verifies, and
    nothing inside a file could decide otherwise, which is why `expected_contributor` exists and
    why federation needs key distribution.
    """
    tmp = tempfile.mkdtemp()
    try:
        ledger = Ledger(root_dir=tmp)
        for i in range(3):
            ledger.append(
                game_desc={"name": "public_goods", "n_agents": 4},
                roster_desc=[{"kind": "qlearning"}], code_version="git:test",
                rounds=100, master_seed=i, content_cid=f"cid{i}",
                metrics_summary={"coop_rate": 0.42}, feature_vector=[4],
            )
        honest_key = ledger.public_key_hex
        assert ledger.verify_chain() is True

        adversary = Ed25519PrivateKey.generate()
        adversary_pub = adversary.public_key().public_bytes_raw().hex()

        records = ledger.load_all()
        records[1]["metrics_summary"] = {"coop_rate": 0.99}
        for i in range(1, len(records)):
            if i > 1:
                records[i]["parents"] = [_record_hash(records[i - 1])]
            records[i]["contributor"] = adversary_pub
            payload = {k: v for k, v in records[i].items() if k != "signature"}
            records[i]["signature"] = adversary.sign(
                canonical_json(payload).encode("utf-8")).hex()

        ledger.ledger_path.write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        reopened = Ledger(root_dir=tmp)

        # Every individual record still carries a valid signature under the key it names.
        assert all(Ledger.verify_record(r) for r in reopened.load_all())

        assert reopened.verify_chain() is False, (
            "the chain changes identity midway, which is what the contributor check exists for")
        assert reopened.verify_chain(expected_contributor=honest_key) is False
        print("OK: a substituted signing identity is caught, with or without the expected key")
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
    test_a_resigned_record_is_caught_by_the_parent_link()
    test_a_substituted_identity_is_caught_without_knowing_the_real_one()
    test_extends_and_replicates_lineage_detection()
    print("OK: extends/replicates lineage detection")
    test_extends_and_replicates_can_coexist_on_the_same_record()
    print("OK: extends and replicates can coexist on the same record")
    test_find_exact_dedup_lookup()
    print("OK: exact-match dedup lookup")
