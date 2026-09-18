"""Exchanging records between installations, and the four ways that must fail.

Every refusal here is a case the ledger's own history says matters. A chain rewritten under a
fresh identity verified perfectly against itself until `verify_chain` started comparing signers,
so a bundle is checked against a key the caller supplies and a mismatch is refused. A half-written
import would look exactly like a verified one, so failures leave nothing behind. And a private key
travelling inside a bundle is the kind of mistake discovered by its recipient.

Run: python -m gamebrains.tests.test_bundle
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from ..agents.classic import AllD
from ..games.public_goods import PublicGoodsGame
from ..repository.bundle import (
    BundleRejected, export_bundle, import_bundle, list_sources, load_records,
)
from ..repository.ledger import Ledger
from ..repository.record import record_experiment


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="gamebrains_bundle_"))


def _install(root: Path, seeds=(0, 1), rounds: int = 50) -> list[dict]:
    """A small installation with a real signed chain."""
    log = root / "log.jsonl"
    log.write_text('{"type": "meta", "seed": 0}\n', encoding="utf-8")
    game = PublicGoodsGame(n_agents=3, rounds=rounds)
    roster = [AllD(f"AllD {i}") for i in range(3)]
    return [record_experiment(root, game, roster, log, rounds=rounds, seed=seed,
                              metrics={"cooperation_rate": 0.1 * (seed + 1)})
            for seed in seeds]


def test_a_bundle_travels_and_verifies_against_the_key_the_reader_was_given() -> None:
    lab_a, lab_b = _tmp(), _tmp()
    try:
        _install(lab_a)
        key = Ledger(lab_a).public_key_hex
        bundle = lab_a / "lab_a.zip"
        manifest = export_bundle(lab_a, bundle, title="Lab A, September", doi="10.5281/zenodo.0")
        assert manifest["records"] == 2 and manifest["complete_chain"] is True

        _install(lab_b, seeds=(7,))
        source = import_bundle(bundle, lab_b, trusted_key=key, label="Lab A")
        assert source["verified"] is True and source["records"] == 2

        # The imported chain is its own chain, not merged into the local one. Merging is what
        # would need consensus the moment two installations wrote at once.
        assert len(Ledger(lab_b).load_all()) == 1
        tagged = load_records(lab_b)
        assert len(tagged) == 3
        labels = {r["source"]["label"] for r in tagged}
        assert labels == {"this installation", "Lab A"}

        # Verifying somebody else's chain must not create a signing identity inside their folder.
        imported_dir = lab_b / source["path"]
        assert not (imported_dir / "identity_private.pem").exists(), (
            "a reader needs to verify, not to sign")

        sources = list_sources(lab_b)
        assert [s["verified"] for s in sources] == [True, True]
        print(f"OK: {source['records']} records imported as a separate chain and verified")
    finally:
        shutil.rmtree(lab_a, ignore_errors=True)
        shutil.rmtree(lab_b, ignore_errors=True)


def test_a_bundle_never_carries_the_private_key() -> None:
    root = _tmp()
    try:
        _install(root)
        bundle = root / "out.zip"
        export_bundle(root, bundle)
        with zipfile.ZipFile(bundle) as zf:
            names = zf.namelist()
            blob = b"".join(zf.read(n) for n in names)
        assert not any("identity_private" in n for n in names), names
        assert b"PRIVATE KEY" not in blob, "the signing key must never leave the machine"
        print("OK: the bundle carries the public key only")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_wrong_key_is_refused_and_nothing_is_kept() -> None:
    lab_a, lab_b = _tmp(), _tmp()
    try:
        _install(lab_a)
        bundle = lab_a / "lab_a.zip"
        export_bundle(lab_a, bundle)

        other_key = Ledger(lab_b).public_key_hex        # a real key, but not the signer's
        try:
            import_bundle(bundle, lab_b, trusted_key=other_key)
        except BundleRejected as exc:
            assert "different key" in str(exc)
        else:
            raise AssertionError("a bundle signed by another key must be refused")

        try:
            import_bundle(bundle, lab_b, trusted_key="")
        except BundleRejected as exc:
            assert "no trusted key" in str(exc)
        else:
            raise AssertionError("importing without a trusted key must be refused")

        assert not (lab_b / "imports").exists(), "a refused import must leave nothing behind"
        print("OK: the wrong key and no key are both refused, and nothing is written")
    finally:
        shutil.rmtree(lab_a, ignore_errors=True)
        shutil.rmtree(lab_b, ignore_errors=True)


def test_an_edited_record_is_caught_on_import() -> None:
    """The attack the ledger was hardened against, now arriving as a file from elsewhere. The
    editor here holds the bundle and not the signing key, which is the realistic case."""
    lab_a, lab_b = _tmp(), _tmp()
    try:
        _install(lab_a)
        key = Ledger(lab_a).public_key_hex
        bundle, tampered = lab_a / "lab_a.zip", lab_a / "tampered.zip"
        export_bundle(lab_a, bundle)

        with zipfile.ZipFile(bundle) as src, zipfile.ZipFile(tampered, "w") as dst:
            for name in src.namelist():
                data = src.read(name)
                if name == "ledger.jsonl":
                    lines = data.decode("utf-8").splitlines()
                    first = json.loads(lines[0])
                    first["metrics_summary"]["cooperation_rate"] = 0.99
                    lines[0] = json.dumps(first, ensure_ascii=True)
                    data = ("\n".join(lines) + "\n").encode("utf-8")
                dst.writestr(name, data)

        try:
            import_bundle(tampered, lab_b, trusted_key=key)
        except BundleRejected as exc:
            assert "did not verify" in str(exc)
        else:
            raise AssertionError("an edited record must be refused")

        # What must be true is that nothing was kept: no chain readable, no source registered,
        # and no leftover directory from the attempt.
        assert load_records(lab_b) == []
        assert [s["local"] for s in list_sources(lab_b)] == [True]
        assert not (lab_b / "imports").exists()
        print("OK: a record edited in transit is refused, with nothing kept")
    finally:
        shutil.rmtree(lab_a, ignore_errors=True)
        shutil.rmtree(lab_b, ignore_errors=True)


def test_a_read_only_ledger_refuses_to_sign_or_to_guess_who_to_trust() -> None:
    root = _tmp()
    try:
        _install(root, seeds=(0,))
        reader = Ledger(root, read_only=True)
        assert reader.public_key_hex == ""

        try:
            reader.verify_chain()
        except ValueError as exc:
            assert "no key of its own" in str(exc)
        else:
            raise AssertionError("a read-only ledger has no key to default its trust set to")

        assert reader.verify_chain(trusted_contributors=[Ledger(root).public_key_hex])

        try:
            reader.append({}, [], "git:test", 1, 0, "cid", {}, [0.0])
        except RuntimeError as exc:
            assert "read-only" in str(exc)
        else:
            raise AssertionError("a read-only ledger must refuse to append")
        print("OK: read-only means neither signing nor guessing whose keys to accept")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_a_bundle_travels_and_verifies_against_the_key_the_reader_was_given()
    test_a_bundle_never_carries_the_private_key()
    test_the_wrong_key_is_refused_and_nothing_is_kept()
    test_an_edited_record_is_caught_on_import()
    test_a_read_only_ledger_refuses_to_sign_or_to_guess_who_to_trust()
    print("\nall bundle tests passed")
