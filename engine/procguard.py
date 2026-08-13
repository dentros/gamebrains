"""Deciding whether spawn-based parallelism is safe here, by trying it rather than guessing.

Background. Computing $\\Phi$ uses PyPhi's parallel cut evaluation, which starts worker processes.
On Windows those are *spawned*: the child re-imports `__main__` and re-executes the module rather
than inheriting a forked address space. That works from a normal script and fails in two situations
this project actually hit:

  1. `python -c "..."` one-liners, where `__main__` is not an importable module, so the child hangs.
  2. Under Flask's development reloader, which re-executes the serving process. In our environment
     the re-executed process resolved to a different interpreter than the one hosting the platform's
     dependencies, and workers spawned across that boundary leaked: orphaned
     `--multiprocessing-fork` processes accumulated and the LP solver eventually raised `MemoryError`.

The project's earlier responses to both were avoidance. Parallel evaluation was force-disabled
repo-wide for (1), which silently cost parallelism in every context where it was fine, and cost a
factor of several in $\\Phi$ runtime until it was noticed. The reloader was switched off for (2),
which works but gives up automatic reloading during development, and only for as long as nobody
turns it back on.

This module replaces both with a decision made from evidence. `spawn_is_safe()` starts one real
worker and waits for it to answer. A process that can spawn gets parallelism; a process that cannot
gets a serial fallback and a reason. Three mechanisms, in the order they matter:

  `pin_interpreter()`  points multiprocessing at this interpreter explicitly, so a child cannot
                       inherit whichever `python.exe` a re-exec happened to resolve to
  `spawn_is_safe()`    the probe, run at most once per process and cached
  `reap_orphans()`     an atexit sweep, so a process that dies mid-computation does not leave
                       workers behind holding memory

Nothing here is Windows-specific in its logic. The probe simply passes everywhere spawn works.
"""

from __future__ import annotations

import atexit
import multiprocessing
import os
import sys
from typing import Optional

#: Filled by `spawn_is_safe` the first time it runs: (verdict, human-readable reason).
_VERDICT: Optional[tuple[bool, str]] = None

#: Escape hatch for callers who know better than the probe, and for reproducing either branch in a
#: test. "1"/"0" force the verdict; unset means probe.
_OVERRIDE_ENV = "GAMEBRAINS_PARALLEL"


def pin_interpreter() -> None:
    """Make spawned children use *this* interpreter, not whatever the platform resolves to.

    `multiprocessing` defaults to `sys.executable` already, so on a healthy process this is a
    no-op. It stops being a no-op when a parent process was re-executed by something else (a
    reloader, a launcher, a shim) and the inherited default no longer points at the interpreter
    holding this project's dependencies -- which is precisely the case that leaked workers here.
    """
    multiprocessing.set_executable(sys.executable)


def _probe_worker(q) -> None:
    """Answer from inside a spawned child. Deliberately trivial: this tests the mechanism, not the
    workload. Anything heavier would confuse "cannot spawn" with "workload failed"."""
    q.put(os.getpid())


def spawn_is_safe(timeout: float = 30.0) -> bool:
    """Can this process start a spawned worker and hear back from it?

    Runs at most once per process; the answer is cached. A child that never answers is terminated
    before returning, so a negative probe cannot itself leave the orphan it was checking for.

    The timeout is generous on purpose. A false negative costs a slower serial computation, while a
    false positive costs a hung request and leaked memory, so the asymmetry is worth waiting for.
    """
    global _VERDICT
    if _VERDICT is not None:
        return _VERDICT[0]

    forced = os.environ.get(_OVERRIDE_ENV)
    if forced in ("0", "1"):
        _VERDICT = (forced == "1", f"forced by {_OVERRIDE_ENV}={forced}")
        return _VERDICT[0]

    # We may already *be* a spawned worker. That happens because a spawning child re-imports the
    # parent's `__main__`, so any probe reachable from module level runs again inside the child --
    # and starting a process from there raises "an attempt has been made to start a new process
    # before the current process has finished its bootstrapping phase". This check was added after
    # exactly that: an earlier version of this module probed at import time and became the
    # recursive-spawn bug it was written to detect. Workers do not need workers of their own, so
    # the answer here is always no, and it is cheap and correct.
    if multiprocessing.parent_process() is not None:
        _VERDICT = (False, "already running inside a spawned worker; workers do not nest")
        return False

    pin_interpreter()
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    child = ctx.Process(target=_probe_worker, args=(queue,), daemon=True)

    try:
        child.start()
        child.join(timeout)
        if child.is_alive():
            child.terminate()
            child.join(5.0)
            _VERDICT = (False, f"a spawned worker did not report back within {timeout:g}s")
        elif queue.empty():
            _VERDICT = (False, f"a spawned worker exited with code {child.exitcode} without reporting")
        else:
            queue.get_nowait()
            _VERDICT = (True, "a spawned worker started and reported back")
    except Exception as exc:                       # noqa: BLE001 -- any failure means "not safe"
        _VERDICT = (False, f"spawning raised {type(exc).__name__}: {exc}")
    finally:
        queue.close()

    return _VERDICT[0]


def spawn_verdict_reason() -> str:
    """Why `spawn_is_safe` answered as it did, for logging next to a degraded computation."""
    if _VERDICT is None:
        return "not yet probed"
    return _VERDICT[1]


def reap_orphans(timeout: float = 5.0) -> int:
    """Terminate any worker processes still running under us. Returns how many were reaped.

    `active_children()` sees only this process's own children, so this cannot touch anything it did
    not start. Registered at exit below, and safe to call directly after a computation that may
    have been interrupted.
    """
    reaped = 0
    for child in multiprocessing.active_children():
        child.terminate()
        child.join(timeout)
        reaped += 1
    return reaped


atexit.register(reap_orphans)
