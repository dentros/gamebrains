"""
Local content-addressed store — stands in for IPFS's role in `docs/repository-schema.md` §1-2.

Simplification, stated plainly: the design doc's IPFS layer stores a package as a directory of
files (manifest.json, event_log.jsonl, metrics/*.json) addressed as a Merkle-DAG of chunks, so a
longer run's log can share the shorter run's prefix chunks. This local store instead addresses the
*whole package* (manifest + event log + metrics, embedded as one JSON object) as a single content
hash. That gives up automatic prefix-sharing between an "extends" pair (see `docs/
repository-schema.md` §4) at the storage layer -- lineage-based reuse still works via the ledger's
`extends` edges, just without the storage-level dedup a real chunked Merkle-DAG would add for free.
Migrating to a real IPFS backend later would not change any caller of this module: `put`/`get`
keep the same signature, only the CID computation and where bytes physically live would change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .normalize import canonical_json


class ContentStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.objects_dir = Path(root_dir) / "objects"
        self.objects_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, cid: str) -> Path:
        return self.objects_dir / f"{cid}.json"

    def put(self, package: dict[str, Any]) -> str:
        """Store `package`; returns its content_cid (sha256 of the canonical JSON, hex)."""
        blob = canonical_json(package)
        cid = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        path = self._path_for(cid)
        if not path.exists():  # content-addressed: identical content already stored -> no-op
            path.write_text(blob, encoding="utf-8")
        return cid

    def get(self, cid: str) -> dict[str, Any]:
        path = self._path_for(cid)
        if not path.exists():
            raise KeyError(f"no object with content_cid {cid!r} in {self.objects_dir}")
        return json.loads(path.read_text(encoding="utf-8"))

    def exists(self, cid: str) -> bool:
        return self._path_for(cid).exists()
