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
signed it. Anyone can generate an identity in microseconds. Deciding *which* identities a reader
should accept cannot be answered from inside the file, so `verify_chain` takes the set of trusted
public keys from its caller and defaults to this installation's own.

A set rather than a single key, and this was once got wrong: an earlier version required one
identity across the whole chain. That is the single-user case, and the repository schema exists
for the other one, where a second researcher's record extends or replicates the first's. Requiring
one identity would have refused exactly the exchange the design is for.

**One thing the set does not settle.** If several installations appended to *one* shared chain,
two could append at once naming the same parent, and the chain would fork. Choosing a branch is
what a consensus mechanism is for. The claim that no consensus is needed holds for one chain per
installation, with records referring across chains through lineage, and not for a chain shared by
several writers. `_detect_lineage` currently reads only the local ledger, so that design choice
has not been made yet.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

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
        reproducibility: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one immutable experiment record. Returns the full signed record.

        `protocol` carries the scoring conventions that change the reported numbers without
        changing the game or the roster (see normalize.build_config for the three-tier rule).

        `reproducibility` says whether a rerun of this configuration would give this event log
        back (see `record.reproducibility_of`). It is deliberately **not** part of `config_hash`:
        it is a property of what the run could promise, not of the design, and two runs of one
        design must keep sharing a hash so that lineage still works. It is signed with the rest of
        the record, so the caveat cannot be stripped from a record that travels.
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
            "reproducibility": reproducibility or {"byte_identical_claimed": True,
                                                   "nondeterministic_agents": []},
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

    def verify_chain(self, trusted_contributors: Optional[Iterable[str]] = None) -> bool:
        """Every signature is valid, every signer is trusted, and every record names its predecessor.

        `verify_record` reads the public key out of the record it is checking, from the
        `contributor` field, so on its own it asks whether a record was signed by whoever the
        record claims signed it. Anyone can generate an identity in microseconds, so that question
        has no useful answer: an adversary who never had our key generates one, writes its public
        half into `contributor`, re-signs the edited record and every record after it, and the
        chain verifies. We demonstrated exactly that on a three-record ledger, changing a reported
        metric from 0.42 to 0.99, before this function compared signers against anything.

        Three checks run per record:

          1. the signature matches the content under the key the record names
          2. that key is one the caller trusts
          3. the record names its predecessor's hash, so nothing was edited, reordered or removed

        **`trusted_contributors` defaults to this installation's own key**, which is right for the
        only case the platform currently has: a local ledger verified by the machine that wrote it.
        A reader of records written elsewhere must pass the set of keys they accept, because which
        identities to accept cannot be read out of the file being checked. That is the key
        distribution problem, and it is the reason federation needs it rather than inheriting it.

        **A set, not one identity.** An earlier version of this check required the same contributor
        across the whole chain. That refuses precisely what the repository schema is for: a second
        researcher's record extending or replicating the first's. A set admits that case and is
        also stricter than the single-identity version in the case that matters, since a chain
        rewritten end to end under a fresh identity passed that check and fails this one.

        What this still does not detect is an edit made with a trusted key. The local key is
        stored unencrypted beside the ledger, so anyone who can write `ledger.jsonl` can also sign
        as its owner. That is key custody, and it is stated in the module docstring.
        """
        trusted = ({self.public_key_hex} if trusted_contributors is None
                   else set(trusted_contributors))
        records = self.load_all()
        previous_hash: Optional[str] = None

        for record in records:
            if not self.verify_record(record):
                return False
            if record.get("contributor") not in trusted:
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

        A record that declined the byte-identical claim (see `record.reproducibility_of`) is never
        returned, however exactly its configuration matches. The whole value of an exact hit is
        that rerunning would produce the same thing, so offering one here would hand the caller a
        single sample of a random process in place of the run they asked for, and nothing
        downstream would mark the substitution. Such a record is still found by the similarity
        lookup, where it is presented as prior work rather than as a finished answer.
        """
        config_hash = hash_config(build_config(game_desc, roster_desc, code_version, protocol))
        for record in self.load_all():
            if (record["config_hash"] == config_hash
                    and record["seeds"]["master"] == master_seed
                    and record["horizon"]["rounds"] >= rounds
                    and record.get("reproducibility", {}).get("byte_identical_claimed", True)):
                return record
        return None
