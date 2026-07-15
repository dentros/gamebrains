"""
Structured event-log — the single contract between the engine and everything downstream.

The same stream of events (a) drives the live console / web visualization, (b) is the
on-disk reproducibility record, and (c) is the data source for the paper figures. Format is
JSON Lines (one JSON object per line): the first line is a `meta` header (config + seed),
each subsequent line is a `round` (or other) event.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional


class EventLog:
    """Collects events in memory and (optionally) streams them to a .jsonl file.

    Pass `on_event` to forward every event live (e.g. to the console or a WebSocket).
    """

    def __init__(
        self,
        path: Optional[str | Path] = None,
        on_event: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        self.events: list[dict[str, Any]] = []
        self.on_event = on_event
        self._fh = None
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "w", encoding="utf-8")
        self.path = path

    def log(self, event: dict[str, Any]) -> None:
        # NB: no wall-clock timestamp is injected, so a given seed+config yields a
        # byte-identical log (reproducibility). Callers may add their own fields.
        self.events.append(event)
        if self._fh is not None:
            self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._fh.flush()
        if self.on_event is not None:
            self.on_event(event)

    def meta(self, **fields: Any) -> None:
        """Write the run header. Call once, first."""
        self.log({"type": "meta", **fields})

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
