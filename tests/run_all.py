"""
Run every test module in this package, in one command.

    python -m gamebrains.tests.run_all
    python -m gamebrains.tests.run_all --skip-slow

Each module is executed the same way it runs standalone, by invoking its `__main__` block, so
there is one definition of what a module's tests are and no risk of this runner drifting from it.

Modules whose optional dependency is missing print SKIP and exit 0 on their own (the interop
modules for Stable-Baselines3, RLlib and MLPro do this), so a machine without those libraries
still gets a green run over everything it can actually check.

Exit code is 0 only if every module that ran passed.
"""

from __future__ import annotations

import runpy
import sys
import time
import traceback
from pathlib import Path

#: Modules that take appreciably longer than the rest, measured rather than guessed: on the
#: development machine these two cost ~16s each while every other module is under 2s. Ray's
#: cluster startup dominates the first, the animat's TPM construction the second. Re-measure from
#: this runner's own per-module timings before editing this set.
SLOW = {"test_interop_rllib", "test_markov_brain"}


def module_names() -> list[str]:
    here = Path(__file__).parent
    return sorted(p.stem for p in here.glob("test_*.py"))


def run_one(name: str) -> tuple[bool, str, float]:
    """Returns (passed, note, seconds). A module that SKIPs counts as passed."""
    started = time.perf_counter()
    note = ""
    try:
        runpy.run_module(f"gamebrains.tests.{name}", run_name="__main__")
    except SystemExit as exit_signal:
        code = exit_signal.code or 0
        if code != 0:
            return False, f"exited {code}", time.perf_counter() - started
        note = "skipped or exited cleanly"
    except BaseException:
        traceback.print_exc()
        return False, "raised", time.perf_counter() - started
    return True, note, time.perf_counter() - started


def main(argv: list[str]) -> int:
    skip_slow = "--skip-slow" in argv
    names = [n for n in module_names() if not (skip_slow and n in SLOW)]

    print(f"Running {len(names)} test modules"
          f"{' (--skip-slow: omitting ' + ', '.join(sorted(SLOW)) + ')' if skip_slow else ''}\n")

    failures: list[str] = []
    total = 0.0
    for name in names:
        print(f"--- {name} " + "-" * max(0, 66 - len(name)))
        passed, note, seconds = run_one(name)
        total += seconds
        if passed:
            print(f"    PASS  {seconds:6.2f}s{'  (' + note + ')' if note else ''}\n")
        else:
            print(f"    FAIL  {seconds:6.2f}s  ({note})\n")
            failures.append(name)

    print("=" * 72)
    print(f"{len(names) - len(failures)}/{len(names)} modules passed in {total:.1f}s")
    if failures:
        print("failed: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
