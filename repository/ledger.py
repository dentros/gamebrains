"""
The signed, append-only, hash-linked ledger -- GameBrains's "blockchain-lite" (see package
docstring in `__init__.py` for what this deliberately is and is not).

Each record is immutable once written. Tamper-evidence comes from two independent mechanisms:
  1. Every record is Ed25519-signed by its contributor's private key (`verify_record`), so its
     content cannot be altered without invalidating the signature.
  2. Every record's `parents` field names the hash of the record before it, so editing *any* past
     record breaks the hash chain from that point forward (`verify_chain`).
No consensus mechanism is needed on top of this because records are purely additive (new
experiments), not competing claims -- see the module docstring in `__init__.py`.
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

from .normalize import canonical_json, hash_config

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
    ) -> dict[str, Any]:
        """Append one immutable experiment record. Returns the full signed record."""
        config = {"game": game_desc, "roster": roster_desc, "code_version": code_version}
        config_hash = hash_config(config)
        lineage = self._detect_lineage(config_hash, rounds, master_seed)

        unsigned = {
            "schema": SCHEMA,
            "config_hash": config_hash,
            "content_cid": content_cid,
            "game": game_desc,
            "roster": roster_desc,
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

    def verify_chain(self) -> bool:
        """Every record's signature is valid AND every record correctly names its predecessor."""
        records = self.load_all()
        previous_hash: Optional[str] = None
        for record in records:
            if not self.verify_record(record):
                return False
            expected_parents = [previous_hash] if previous_hash else []
            if record["parents"] != expected_parents:
                return False
            previous_hash = _record_hash(record)
        return True

    # --- Smart Filter's exact-match lookup (the other half lives in metrics/filter later) ---

    def find_exact(self, game_desc: dict, roster_desc: list[dict], code_version: str,
                   rounds: int, master_seed: int) -> Optional[dict[str, Any]]:
        """Has this precise experiment (design + seed + rounds) already been run?"""
        config_hash = hash_config({"game": game_desc, "roster": roster_desc, "code_version": code_version})
        for record in self.load_all():
            if (record["config_hash"] == config_hash
                    and record["seeds"]["master"] == master_seed
                    and record["horizon"]["rounds"] >= rounds):
                return record
        return None
