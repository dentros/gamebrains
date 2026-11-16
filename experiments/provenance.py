"""What produced a stored result: code version, library versions, command line and every setting.

The paper states that the settings behind a reported number can be recovered. For a run recorded in
the ledger that was always true. For the result files the experiment scripts write it was not: the
30-seed, bake-off and lesion files held numbers and nothing about the exploration decay, the
genetic-algorithm settings or the commit that produced them, so a reader holding the file could not
say what it measured. Found by an external review of the repository (C2-05, 2026-11-16). Every
script now stores this block beside its results.

Determinism is unaffected: the block is written next to the numbers and never feeds back into them.
"""

from __future__ import annotations

import datetime as _dt
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _code_version() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=Path(__file__).resolve().parents[1], timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            # Only uncommitted *code* makes a version unrecoverable. Result files are excluded: a
            # chain of scripts rewrites them one after another, and counting them marked every
            # script after the first as "+uncommitted" while the code was the committed version.
            dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--",
                                    ".", ":(exclude)experiments/results"],
                                   capture_output=True, text=True, timeout=5,
                                   cwd=Path(__file__).resolve().parents[1]).stdout.strip()
            return "git:" + out.stdout.strip() + ("+uncommitted" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _versions() -> dict[str, Any]:
    out: dict[str, Any] = {"python": sys.version.split()[0], "numpy": np.__version__,
                           "platform": platform.platform()}
    for name in ("torch", "gymnasium", "pettingzoo", "networkx", "scipy"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = None
    return out


def provenance(settings: dict[str, Any]) -> dict[str, Any]:
    """The block to store under `"provenance"` in a result file.

    `settings` is every constant and argument that changes a number in the file. A setting left
    out of it is a setting nobody can recover later, so a script passes its whole configuration.
    """
    return {
        "code_version": _code_version(),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "written_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "versions": _versions(),
        "settings": settings,
    }
