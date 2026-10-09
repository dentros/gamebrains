"""Sign a file with this installation's identity key, and say honestly what that proves.

Written for one concrete need: sending an unpublished manuscript to a colleague and wanting
something on record about it. The temptation with a tool like this is to let it imply more than it
does, so the statement it writes is as careful as the arithmetic.

**What a signature here establishes.** The file hashes to a particular value, and whoever holds
the private key put their name to that value. Change one byte of the file and the check fails. So
it answers "is this the same document" and "did the key holder vouch for it", and it answers both
completely.

**What it does not establish, and cannot.** When. A signature carries no time: signing something
today produces a signature as valid as one made last year, and nothing in it says which. That is
not a limitation of this implementation, it is what a signature is, and it is the open question
the project's own letter to a cryptographer puts third on its list. For time you need a party you
do not control, a timestamping authority or a public record.

It also does not establish authorship. It establishes that the key holder signed these bytes, and
the link from the key to a person is a thing people arrange socially, by publishing a fingerprint
somewhere others already trust.

The statement written alongside the signature says all of this in plain words, because a file
named `signature.txt` that a reader cannot interpret is worse than none: it invites them to assume
the strongest reading.

    py -m gamebrains.repository.attest "SPE GAMEBRAINS/root.pdf"
    py -m gamebrains.repository.attest paper.pdf --verify paper.pdf.attestation.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import (Encoding, NoEncryption, PrivateFormat,
                                                          PublicFormat, load_pem_private_key)

#: Where the platform keeps the key it already uses to sign ledger records. The same identity is
#: used on purpose: a second key would mean a second thing for a reader to be told to trust.
DEFAULT_KEY = Path(__file__).resolve().parents[1] / "repo_store" / "identity_private.pem"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_key(key_path: Path) -> Ed25519PrivateKey:
    if not key_path.exists():
        raise FileNotFoundError(
            f"no identity key at {key_path}. The platform writes one the first time it records a "
            f"run, so run an experiment first, or point --key at the key you want to use")
    return load_pem_private_key(key_path.read_bytes(), password=None)


def _payload(path: Path, file_hash: str, note: str) -> dict[str, Any]:
    """Exactly what gets signed. Everything a reader needs is inside the signed bytes.

    The filename is included because a signature over a bare hash says nothing about what the
    document was called, and the timestamp is included *and labelled as a claim* rather than
    omitted: a reader is better served by a stated claim they can weigh than by a silence that
    invites them to guess.
    """
    return {
        "statement": "the holder of the key below vouches for the bytes of this file",
        "filename": path.name,
        "sha256": file_hash,
        "bytes": path.stat().st_size,
        "signed_at_claimed": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "signed_at_is_a_claim": ("this time is written by the signer and is not independently "
                                 "witnessed, so it is evidence only to whoever trusts the signer"),
        "note": note,
    }


def sign(path: Path, key_path: Path = DEFAULT_KEY, note: str = "") -> dict[str, Any]:
    """Sign one file and return the attestation that will be written beside it."""
    private_key = _load_key(key_path)
    public_hex = private_key.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw).hex()

    payload = _payload(path, _hash_file(path), note)
    signed_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = private_key.sign(signed_bytes)

    return {
        "payload": payload,
        "public_key": public_hex,
        "signature": signature.hex(),
        "algorithm": "Ed25519 over the canonical JSON of payload",
        "what_this_shows": [
            "the file hashes to payload.sha256, so one changed byte breaks this",
            "the holder of public_key signed that hash together with the rest of payload",
        ],
        "what_this_does_not_show": [
            "when it was signed: a signature carries no time, and payload.signed_at_claimed is "
            "the signer's own word. For time you need a party you do not control",
            "who wrote the document: linking this key to a person is done by publishing the key "
            "somewhere people already trust, not by this file",
        ],
    }


def verify(path: Path, attestation: dict[str, Any]) -> tuple[bool, str]:
    """Check a file against an attestation. Returns whether it holds and why."""
    payload = attestation.get("payload") or {}
    actual = _hash_file(path)
    if actual != payload.get("sha256"):
        return False, (f"the file hashes to {actual[:16]}... and the attestation names "
                       f"{str(payload.get('sha256'))[:16]}..., so this is not that file")
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(attestation["public_key"]))
        key.verify(bytes.fromhex(attestation["signature"]),
                   json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    except (InvalidSignature, KeyError, ValueError) as exc:
        return False, f"the signature does not check out against that key ({exc})"
    return True, (f"the file is unchanged and the holder of {attestation['public_key'][:16]}... "
                  f"signed it. This says nothing about when")


def _readable(attestation: dict[str, Any]) -> str:
    """The same thing in prose, for whoever opens the folder in six months."""
    payload = attestation["payload"]
    lines = [
        f"{payload['filename']}",
        f"  sha256     {payload['sha256']}",
        f"  size       {payload['bytes']} bytes",
        f"  signed by  {attestation['public_key']}",
        f"  claimed at {payload['signed_at_claimed']}",
        "",
        "What this shows:",
    ]
    lines += [f"  - {item}" for item in attestation["what_this_shows"]]
    lines += ["", "What it does not show:"]
    lines += [f"  - {item}" for item in attestation["what_this_does_not_show"]]
    if payload.get("note"):
        lines += ["", f"Note: {payload['note']}"]
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file")
    parser.add_argument("--key", default=str(DEFAULT_KEY))
    parser.add_argument("--note", default="", help="a sentence stored inside the signed payload")
    parser.add_argument("--verify", metavar="ATTESTATION",
                        help="check the file against an existing attestation instead of signing")
    args = parser.parse_args(argv)

    path = Path(args.file)
    if not path.exists():
        print(f"no such file: {path}")
        return 1

    if args.verify:
        attestation = json.loads(Path(args.verify).read_text(encoding="utf-8"))
        ok, reason = verify(path, attestation)
        print(("holds: " if ok else "does not hold: ") + reason)
        return 0 if ok else 1

    attestation = sign(path, Path(args.key), args.note)
    json_path = path.with_suffix(path.suffix + ".attestation.json")
    text_path = path.with_suffix(path.suffix + ".attestation.txt")
    json_path.write_text(json.dumps(attestation, indent=2, ensure_ascii=False), encoding="utf-8")
    text_path.write_text(_readable(attestation), encoding="utf-8")

    print(_readable(attestation))
    print(f"wrote {json_path.name} and {text_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
