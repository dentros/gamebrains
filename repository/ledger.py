"""
The signed, append-only, hash-linked ledger -- GameBrains's "blockchain-lite" (see package
docstring in `__init__.py` for what this deliberately is and is not).

Each record is immutable once written. Tamper-evidence comes from two mechanisms:
  1. Every record is Ed25519-signed by its contributor's private key (`verify_record`), so its
     content cannot be altered without invalidating the signature.
  2. Every record's `parents` field names the hash of the record before it, so editing *any* past
     record breaks the hash chain from that point forward (`verify_chain`).
No consensus mechanism is needed on top of this because records are purely additive (new
experiments), not competing claims -- see the module docstring in `__init__.py`.

## What the two mechanisms do not give you, which took us two tries to state

An earlier version of this docstring called them *independent*. They are not. Mechanism 2 is worth
nothing once an adversary can produce signatures, because re-signing the edited record and every
record after it repairs the chain as well: the hash pointers are part of the signed content. The
chain protects the sequence against someone who cannot sign, and against nobody else.

Two separate things therefore have to hold, and only the first is in this file's control.

**Key custody.** `_load_or_create_identity` writes the private key unencrypted next to the ledger,
so write access to `ledger.jsonl` implies read access to the key signing it. That is a reasonable
trade for a local single-user tool and the wrong one the moment a record travels.

**Key distribution.** `verify_record` takes the public key from the `contributor` field of the
record it is verifying, so used alone it asks only whether a record was signed by whoever it says
signed it. Anyone can generate an identity in microseconds. Deciding *which* identity should have
signed a given ledger cannot be answered from inside the file, and `verify_chain` takes
`expected_contributor` for callers who know the answer from elsewhere.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding, NoEncryption, PrivateFormat, PublicFormat, load_pem_private_key,
)

from .normalize import build_config, canonical_json, hash_config

SCHEMA = "gamebrains/experiment@1"


def _record_hash(record: dict[str, Any]) -> str:
    """Identifies one immutable, already-signed record (used as the next record's `parents`)."""
    import hashlib
    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


class Ledger:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.root_dir / "ledger.jsonl"
        self._private_key = self._load_or_create_identity()
        self.public_key_hex = self._private_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        ).hex()

    # --- identity -----------------------------------------------------------------

    def _load_or_create_identity(self) -> Ed25519PrivateKey:
        key_path = self.root_dir / "identity_private.pem"
        if key_path.exists():
            return load_pem_private_key(key_path.read_bytes(), password=None)
        private_key = Ed25519PrivateKey.generate()
        pem = private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        key_path.write_bytes(pem)
        return private_key

    # --- reading --------------------------------------------------------------------

    def load_all(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        with open(self.ledger_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _last_record_hash(self) -> Optional[str]:
        records = self.load_all()
        return _record_hash(records[-1]) if records else None

    # --- lineage detection -----------------------------------------------------------

    def _detect_lineage(self, config_hash: str, rounds: int, master_seed: int) -> dict[str, Any]:
        """`extends` and `replicates` are checked independently and MAY both be set on the same
        record: e.g. a record can extend its own same-seed prior (shorter) run while also
        belonging to a family of same-config, different-seed replicates. `derives_from` is not
        auto-detected here -- it is set explicitly by a future post-hoc-metric/meta-analysis
        writer, and per docs/repository-schema.md §4 may be a *list* of parent hashes (a
        meta-analysis over several replicates derives_from all of them, without itself being an
        `extends` or a `replicates` of any single one)."""
        extends = None
        replicates = None
        family = [r for r in self.load_all() if r["config_hash"] == config_hash]

        same_seed = [r for r in family if r["seeds"]["master"] == master_seed]
        shorter = [r for r in same_seed if r["horizon"]["rounds"] < rounds]
        if shorter:
            longest_shorter = max(shorter, key=lambda r: r["horizon"]["rounds"])
            extends = _record_hash(longest_shorter)

        other_seed = [r for r in family if r["seeds"]["master"] != master_seed]
        if other_seed:
            replicates = _record_hash(other_seed[0])  # point back to the family's first sibling

        return {"extends": extends, "replicates": replicates, "derives_from": None}

    # --- writing --------------------------------------------------------------------

    def append(
        self,
        game_desc: dict[str, Any],
        roster_desc: list[dict[str, Any]],
        code_version: str,
        rounds: int,
        master_seed: int,
        content_cid: str,
        metrics_summary: dict[str, Any],
        feature_vector: list[float],
        license: str = "TBD (code) / CC-BY (data)",
        protocol: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one immutable experiment record. Returns the full signed record.

        `protocol` carries the scoring conventions that change the reported numbers without
        changing the game or the roster (see normalize.build_config for the three-tier rule).
        """
        # Built through build_config rather than assembled here, so there is exactly one
        # definition of what a config_hash covers. These two used to be separate literals, which
        # is precisely how a new tier could get added in one place and silently missed in the other.
        config = build_config(game_desc, roster_desc, code_version, protocol)
        config_hash = hash_config(config)
        lineage = self._detect_lineage(config_hash, rounds, master_seed)

        unsigned = {
            "schema": SCHEMA,
            "config_hash": config_hash,
            "content_cid": content_cid,
            "game": game_desc,
            "roster": roster_desc,
            "protocol": protocol or {},
            "horizon": {"rounds": rounds},
            "seeds": {"master": master_seed},
            "code_version": code_version,
            "metrics_summary": metrics_summary,
            "feature_vector": feature_vector,
            "contributor": self.public_key_hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "license": license,
            "parents": [self._last_record_hash()] if self._last_record_hash() else [],
            "lineage": lineage,
        }
        signature = self._private_key.sign(canonical_json(unsigned).encode("utf-8"))
        record = {**unsigned, "signature": signature.hex()}

        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
        return record

    # --- verification -----------------------------------------------------------------

    @staticmethod
    def verify_record(record: dict[str, Any]) -> bool:
        """Checks the Ed25519 signature only (not the chain link -- see verify_chain)."""
        try:
            payload = {k: v for k, v in record.items() if k != "signature"}
            public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(record["contributor"]))
            public_key.verify(bytes.fromhex(record["signature"]), canonical_json(payload).encode("utf-8"))
            return True
        except Exception:
            return False

    def verify_chain(self, expected_contributor: Optional[str] = None) -> bool:
        """Every signature is valid, every record names its predecessor, and one identity signed.

        **Pass `expected_contributor` whenever you have it.** Without it this answers a weaker
        question than it appears to, and the gap is not subtle. `verify_record` reads the public
        key out of the record it is checking, from the `contributor` field, so on its own it asks
        whether a record was signed by whoever the record claims signed it. An adversary who never
        had our key can generate a fresh one, write its public half into `contributor`, re-sign the
        edited record and every record after it, and walk out with a ledger that verifies. We
        demonstrated exactly that before adding this parameter, on a three-record ledger, changing
        a reported metric from 0.42 to 0.99.

        Three checks now run per record, and what each buys is worth separating:

          1. the signature matches the content under the key the record names
          2. the record names its predecessor's hash, so nothing was edited, reordered or removed
          3. the contributor is the same across the whole chain, and equal to
             `expected_contributor` when one is supplied

        Check 3 without the argument catches a key substituted for *part* of a chain, which is
        what an adversary editing one old record would otherwise do. It cannot catch a chain
        rewritten end to end under a new identity, because nothing inside a file can establish
        which identity ought to have written it. That is a key-distribution problem and it is why
        federation needs one, not a defect this function can close.
        """
        records = self.load_all()
        previous_hash: Optional[str] = None
        contributor = expected_contributor

        for record in records:
            if not self.verify_record(record):
                return False

            if contributor is None:
                contributor = record.get("contributor")      # first record sets the identity
            elif record.get("contributor") != contributor:
                return False

            expected_parents = [previous_hash] if previous_hash else []
            if record["parents"] != expected_parents:
                return False
            previous_hash = _record_hash(record)

        return True

    # --- Smart Filter's exact-match lookup (the other half lives in metrics/filter later) ---

    def find_exact(self, game_desc: dict, roster_desc: list[dict], code_version: str,
                   rounds: int, master_seed: int,
                   protocol: dict[str, Any] | None = None) -> Optional[dict[str, Any]]:
        """Has this precise experiment (design + protocol + seed + rounds) already been run?

        Must hash the same way `append` does, or the Smart Filter reports a reuse hit for a run
        that was actually scored under a different convention.
        """
        config_hash = hash_config(build_config(game_desc, roster_desc, code_version, protocol))
        for record in self.load_all():
            if (record["config_hash"] == config_hash
                    and record["seeds"]["master"] == master_seed
                    and record["horizon"]["rounds"] >= rounds):
                return record
        return None
