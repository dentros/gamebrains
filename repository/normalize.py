"""
Canonical config normalization + hashing.

Resolves the open item in `docs/repository-schema.md` §7 ("canonical normalization of config
before hashing — define before implementing dedup"): the same experiment must always produce the
same `config_hash`, regardless of dict key order or floating-point formatting quirks.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

_FLOAT_ROUND_DECIMALS = 10  # avoid float-representation drift changing the hash


def _round_floats(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, _FLOAT_ROUND_DECIMALS)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, fixed float precision, no whitespace."""
    return json.dumps(_round_floats(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_config(config: dict) -> str:
    """The `config_hash` field: sha256 of the canonical JSON, hex-encoded."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def build_config(game_desc: dict, roster_desc: list[dict], code_version: str) -> dict:
    """Assemble the dict that becomes `config_hash` -- deliberately EXCLUDES `horizon`/`seeds`.

    Per docs/repository-schema.md §4, `extends` ("same config_hash + same seeds, larger rounds")
    and `replicates` ("same config_hash, different seed") are both defined *in terms of* matching
    config_hash while rounds/seed vary independently -- so config_hash itself must identify the
    experimental *design* (game + roster + code version) only. Horizon and seed are compared as
    their own record fields, not folded into the hash. An *exact* rerun (identical design, seed,
    and rounds) is then simply "same config_hash AND same seeds.master AND same horizon.rounds" --
    and, since GameBrains event-logs are already byte-identical for identical seed+config (see
    engine/eventlog.py), an exact rerun also yields the same content_cid for free.
    """
    return {"game": game_desc, "roster": roster_desc, "code_version": code_version}
