"""Matches that are still being played, because a person cannot answer inside one HTTP request.

Every other seat decides in microseconds, so a match runs start to finish while the browser waits.
A person takes seconds per round and may walk away mid-match, which breaks that shape completely.

The match therefore runs in a thread and this module holds the handle. The browser polls a page
that shows the board and posts a move; `agents/human.py` blocks on a queue until one arrives. What
matters about this arrangement is what it avoids: `engine/runner.run_match` is reused whole, with
its semantics binding, its event log, its episode handling and its metrics, instead of the web
layer driving `game.step()` round by round. A second copy of the runner is how two code paths
quietly stop agreeing about what a match is.

**A thread, not a process.** The agents are live Python objects holding Q-tables and torch modules,
and the person's seat is a queue somebody else has to reach. Nothing here is CPU-bound while it
waits, so the GIL costs nothing.

**Finished matches are kept, abandoned ones are not.** A result nobody collected is still the
person's result, so it stays until it is fetched or until `MAX_AGE` passes. A match whose player
left is holding a thread blocked on a queue; it ends itself when `HumanAgent.wait_seconds` runs
out, which is how a closed browser tab stops costing anything.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

#: How long a finished result stays collectable. Long enough to survive a reload and a coffee,
#: short enough that a server left running for days is not an archive of other people's matches.
MAX_AGE = 60 * 60

#: At most this many at once. A cap exists because every live match holds a thread and a roster of
#: trained agents; without one, a reload loop would quietly accumulate both.
MAX_LIVE = 8


@dataclass
class LiveMatch:
    """One match in progress, and whatever it ended as."""

    token: str
    thread: threading.Thread
    humans: list[Any] = field(default_factory=list)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    @property
    def prompt(self) -> Optional[dict[str, Any]]:
        """What the first waiting person is being asked, or None if nobody is waiting.

        One prompt at a time even with several people seated, because the runner asks each seat in
        turn within a round and two prompts on one page would be two different rounds.
        """
        for person in self.humans:
            pending = person.pending
            if pending is not None:
                return {"name": person.name, **pending}
        return None

    def submit(self, role: str) -> None:
        """Give the waiting seat its move, in the words the page showed."""
        for person in self.humans:
            if person.pending is not None:
                person.submit_role(role)
                return
        raise RuntimeError("no seat is waiting for a move right now")


_MATCHES: dict[str, LiveMatch] = {}
_LOCK = threading.Lock()


def _prune(now: Optional[float] = None) -> None:
    """Drop what nobody can still use. Called on every start, so it needs no timer of its own."""
    now = now or time.time()
    for token, match in list(_MATCHES.items()):
        if match.running:
            continue
        age = now - (match.finished or match.started)
        if age > MAX_AGE:
            del _MATCHES[token]


def start(work: Callable[[list], dict[str, Any]]) -> str:
    """Begin a match in the background and return the token the browser will poll.

    `work` is handed a list to append the human seats to as it builds the roster. That indirection
    is what lets the existing run pipeline stay a single function: it does not need to know a
    person is involved, it only needs to hand back the roster it built.
    """
    with _LOCK:
        _prune()
        if sum(1 for m in _MATCHES.values() if m.running) >= MAX_LIVE:
            raise RuntimeError(
                f"{MAX_LIVE} matches are already being played on this server. They end on their "
                f"own when nobody answers, so either finish one or wait for a seat to time out.")
        token = secrets.token_urlsafe(9)

    humans: list[Any] = []

    def run() -> None:
        match = _MATCHES[token]
        try:
            match.result = work(humans)
        except BaseException as exc:                       # noqa: BLE001
            # Including the timeout a walked-away player causes. The message is what the page
            # shows, so it has to read as an explanation rather than as a traceback.
            match.error = str(exc) or exc.__class__.__name__
        finally:
            match.finished = time.time()

    thread = threading.Thread(target=run, daemon=True, name=f"live-{token}")
    with _LOCK:
        _MATCHES[token] = LiveMatch(token=token, thread=thread, humans=humans)
    thread.start()
    return token


def get(token: str) -> Optional[LiveMatch]:
    return _MATCHES.get(token)


def collect(token: str) -> Optional[dict[str, Any]]:
    """Take a finished match's result and forget it, so a result is rendered once and released."""
    match = _MATCHES.get(token)
    if match is None or match.running or match.result is None:
        return None
    with _LOCK:
        _MATCHES.pop(token, None)
    return match.result


def live_count() -> int:
    return sum(1 for m in _MATCHES.values() if m.running)
