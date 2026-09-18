"""Moving one installation's records to another, without pretending the hard part is solved.

A bundle is a zip holding a chain, the content-addressed objects its records point at, the public
key that signed them, and a manifest saying where it came from. Importing one verifies the chain
against **a key the reader supplies**, then keeps it as a separate chain under `imports/`.

Three decisions worth stating, because each is a claim the paper makes.

**One chain per installation, never a shared one.** An imported chain is stored beside the local
one, not merged into it. Merging would mean two writers appending to a single chain, which forks
the moment they write concurrently, and choosing between forks is the consensus problem. Records
refer across chains through `lineage.extends`, which is a content hash and needs no coordination.
So the repository stays verifiable without anything resembling a blockchain's agreement protocol,
and that is why: not because consensus was solved here, but because the structure avoids needing it.

**The trusted key comes from the reader, and there is no default.** `import_bundle` requires
`trusted_key` as an argument, compares it against the key the bundle declares, and refuses on any
mismatch, showing both. What it cannot do is check *where the reader got that key*. A reader who
opens the zip, copies the key out of it and passes it back has verified only that the bundle is
internally consistent, which is precisely the hole `Ledger.verify_chain` was rewritten to close:
anyone can generate an identity, sign a chain of invented results with it, and ship the two
together. The defence against that is social rather than cryptographic, and the only thing this
module can contribute is to never supply the key itself. How a reader comes by the right key is
the key distribution problem, open here and named as such in the paper's future work.

**The private key never leaves.** `export_bundle` reads only `ledger.jsonl`, the objects and the
public key. A test asserts the absence, because this is the kind of mistake that is found by the
person who receives the bundle.

    from gamebrains.repository.bundle import export_bundle, import_bundle

    export_bundle("gamebrains/repo_store", "lab_a.zip", title="Lab A, September runs")
    summary = import_bundle("lab_a.zip", "gamebrains/repo_store", trusted_key="9f2c...")
"""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .cas import ContentStore
from .ledger import Ledger

#: Where imported chains live, one directory per contributor key, under the local repo root.
IMPORTS_DIR = "imports"
#: The registry of what has been imported and whether it verified.
SOURCES_FILE = "sources.json"

MANIFEST_NAME = "manifest.json"
LEDGER_NAME = "ledger.jsonl"
KEY_NAME = "public_key.txt"
OBJECTS_DIR = "objects"


def _short(key: str) -> str:
    """A directory name from a key. Sixteen hex characters, which is enough to keep two
    installations apart in a folder listing and short enough to read in one."""
    return key[:16] or "unknown"


def export_bundle(repo_root: str | Path, out_path: str | Path, title: str = "",
                  doi: str = "", records: Optional[Iterable[dict[str, Any]]] = None) -> dict[str, Any]:
    """Write a shareable zip of this installation's chain and the objects it refers to.

    `records` defaults to the whole chain. A subset is accepted, but note what it costs: the chain
    is verified by each record naming its predecessor's hash, so a bundle missing records in the
    middle cannot be verified as a chain by the receiver. Exporting a prefix keeps that property,
    which is why the argument exists at all.
    """
    repo_root, out_path = Path(repo_root), Path(out_path)
    ledger = Ledger(repo_root)
    all_records = ledger.load_all()
    chosen = list(records) if records is not None else all_records

    manifest = {
        "schema": "gamebrains-bundle-1",
        "title": title,
        "doi": doi,
        "contributor": ledger.public_key_hex,
        "records": len(chosen),
        "complete_chain": len(chosen) == len(all_records),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    store = ContentStore(repo_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
        zf.writestr(LEDGER_NAME, "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in chosen))
        zf.writestr(KEY_NAME, ledger.public_key_hex)
        # Content addressing means two records of the same package share one CID, which happens
        # for real whenever a design is rerun at another seed on the same log. Writing it once is
        # the whole point of addressing by content; writing it twice makes a zip with duplicate
        # entries that readers handle differently.
        written: set[str] = set()
        for record in chosen:
            cid = record.get("content_cid")
            if cid and cid not in written and store.exists(cid):
                zf.writestr(f"{OBJECTS_DIR}/{cid}.json", json.dumps(store.get(cid),
                                                                    ensure_ascii=True))
                written.add(cid)
    return manifest


def _discard(staging: Path) -> None:
    """Remove a failed import, including the `imports/` parent if this attempt created it.

    A rejected import should leave the repository exactly as it found it. An empty directory is
    harmless in itself, but it is the kind of leftover that makes a later reader wonder whether
    something half-arrived.
    """
    shutil.rmtree(staging, ignore_errors=True)
    parent = staging.parent
    try:
        if parent.name == IMPORTS_DIR and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass


class BundleRejected(Exception):
    """The bundle did not verify, so nothing was kept.

    An import that half-succeeded would be worse than one that failed: the records would sit in the
    repository looking exactly like verified ones.
    """


def import_bundle(bundle_path: str | Path, repo_root: str | Path, trusted_key: str,
                  label: str = "") -> dict[str, Any]:
    """Verify a bundle against `trusted_key` and keep it as its own chain. Refuse otherwise.

    Raises `BundleRejected` when no key is given, when the key the bundle declares is not the one
    the caller trusts, or when the chain does not verify against it, which covers an edited record,
    a reordering, a removal, and a record signed by somebody else.

    Nothing is kept unless all of that passes. The staging directory is removed on every failure
    path, because a half-imported chain would sit in the repository looking exactly like a
    verified one.
    """
    bundle_path, repo_root = Path(bundle_path), Path(repo_root)
    trusted_key = (trusted_key or "").strip().lower()
    if not trusted_key:
        raise BundleRejected(
            "no trusted key given. A chain can only be verified against a key obtained some other "
            "way than from the file being checked.")

    with zipfile.ZipFile(bundle_path) as zf:
        names = set(zf.namelist())
        for required in (MANIFEST_NAME, LEDGER_NAME, KEY_NAME):
            if required not in names:
                raise BundleRejected(f"the bundle has no {required}")
        manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
        claimed_key = zf.read(KEY_NAME).decode("utf-8").strip().lower()

        if claimed_key != trusted_key:
            raise BundleRejected(
                "the bundle was signed by a different key from the one you trust.\n"
                f"  bundle says: {claimed_key[:32]}...\n"
                f"  you trusted: {trusted_key[:32]}...\n"
                "Either you were given the wrong key, or this is not the chain you were promised.")

        target = repo_root / IMPORTS_DIR / _short(trusted_key)
        staging = target.with_name(target.name + ".incoming")
        if staging.exists():
            shutil.rmtree(staging)
        (staging / OBJECTS_DIR).mkdir(parents=True, exist_ok=True)

        try:
            staging.joinpath(LEDGER_NAME).write_bytes(zf.read(LEDGER_NAME))
            staging.joinpath(KEY_NAME).write_bytes(zf.read(KEY_NAME))
            staging.joinpath(MANIFEST_NAME).write_bytes(zf.read(MANIFEST_NAME))
            for name in names:
                if name.startswith(f"{OBJECTS_DIR}/") and name.endswith(".json"):
                    staging.joinpath(name).write_bytes(zf.read(name))

            # Read-only, so verifying somebody else's chain never creates a signing identity inside
            # their folder. See Ledger's docstring.
            incoming = Ledger(staging, read_only=True)
            if not incoming.verify_chain(trusted_contributors=[trusted_key]):
                raise BundleRejected(
                    "the chain did not verify against that key. Either a record was edited, the "
                    "order changed, a record was removed, or something in it was signed by "
                    "somebody else.")
            records = incoming.load_all()
        except BundleRejected:
            _discard(staging)
            raise
        except Exception as exc:
            _discard(staging)
            raise BundleRejected(f"the bundle could not be read: {exc}") from exc

    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)

    source = {
        "key": trusted_key,
        "label": label or manifest.get("title") or _short(trusted_key),
        "doi": manifest.get("doi", ""),
        "records": len(records),
        "complete_chain": bool(manifest.get("complete_chain", False)),
        "exported_at": manifest.get("exported_at", ""),
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "path": str(Path(IMPORTS_DIR) / _short(trusted_key)),
        "verified": True,
    }
    _register_source(repo_root, source)
    return source


def _register_source(repo_root: Path, source: dict[str, Any]) -> None:
    path = repo_root / SOURCES_FILE
    sources = []
    if path.exists():
        try:
            sources = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            sources = []
    sources = [s for s in sources if s.get("key") != source["key"]] + [source]
    path.write_text(json.dumps(sources, indent=2), encoding="utf-8")


def list_sources(repo_root: str | Path) -> list[dict[str, Any]]:
    """Every chain this installation can read, its own first.

    The local entry carries `verified` as the result of checking it now, rather than a stored
    flag. A verification result that was true last week says nothing about the file on disk today.
    """
    repo_root = Path(repo_root)
    local = Ledger(repo_root)
    sources = [{
        "key": local.public_key_hex,
        "label": "this installation",
        "local": True,
        "records": len(local.load_all()),
        "verified": local.verify_chain(),
        "path": ".",
    }]

    path = repo_root / SOURCES_FILE
    if path.exists():
        try:
            for entry in json.loads(path.read_text(encoding="utf-8")):
                folder = repo_root / entry.get("path", "")
                entry = dict(entry, local=False)
                if (folder / LEDGER_NAME).exists():
                    imported = Ledger(folder, read_only=True)
                    entry["records"] = len(imported.load_all())
                    entry["verified"] = imported.verify_chain(
                        trusted_contributors=[entry.get("key", "")])
                else:
                    entry["verified"] = False
                    entry["missing"] = True
                sources.append(entry)
        except json.JSONDecodeError:
            pass
    return sources


def load_records(repo_root: str | Path, include_imports: bool = True) -> list[dict[str, Any]]:
    """Every record this installation can read, each tagged with the source it came from.

    The tag is added here rather than stored in the record, because a record is signed and
    `source` is the reader's own bookkeeping. Anything written into the record itself would either
    break the signature or have to be excluded from it, and a field excluded from a signature is a
    field an adversary may set.
    """
    repo_root = Path(repo_root)
    tagged: list[dict[str, Any]] = []
    for source in list_sources(repo_root):
        if not include_imports and not source.get("local"):
            continue
        folder = repo_root if source.get("local") else repo_root / source.get("path", "")
        if not (folder / LEDGER_NAME).exists():
            continue
        ledger = Ledger(folder, read_only=not source.get("local"))
        for record in ledger.load_all():
            tagged.append({**record, "source": {"key": source.get("key", ""),
                                                "label": source.get("label", ""),
                                                "local": bool(source.get("local")),
                                                "verified": bool(source.get("verified"))}})
    return tagged
