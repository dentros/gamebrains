"""Vary one thing across a range, run the cells in parallel, and watch every measure respond.

The question this answers is the one a researcher actually has: *as this parameter moves, what
happens to everything?* Asking it one match at a time is possible and nobody does it, because
twenty matches is twenty waits. These are independent runs over a handful of cores, so they should
be twenty at once, and the point of the view above this module is that the answer fills in while
you watch.

Three decisions hold it together.

**The workers compute and the parent records.** A signed, append-only chain has exactly one writer
by construction: two processes appending would interleave records whose `parents` hashes each
other's, and the chain would be unverifiable with nothing saying when it broke. So a worker runs
its match, writes its own event log and hands back metrics, and the parent walks the completed
cells in order and appends them. Recording is microseconds against a match's seconds, so
serialising it costs nothing worth measuring.

**A cell is a full experimental configuration, not a diff.** Each carries its game, size, horizon,
seed, roster composition and hyperparameter overrides, so a worker needs nothing from the parent's
memory and a cell can be rerun, sent elsewhere or stored as it is. The parent rebuilds the same
roster to record it, which is exact because `agents/factory.py` is deterministic given the seed and
because every parameter the recorder stores is a construction-time value that training never
changes. A hyperparameter that mutated during a match would break that, which is why the recorder's
parameter list and this assumption are named together here.

**Interrupting is expected.** The machine this is meant to scale onto has no uninterruptible power
supply, and a long sweep will meet a power cut. So every cell is recorded and journalled as it
lands rather than at the end, and starting the same sweep again runs only what is missing. The
journal holds cell labels rather than configuration hashes, because a hash is known only after a
cell has run and the point is to not run it.

    python -m gamebrains.experiments.sweep --vary mpcr --values 0.3,0.4,0.5,0.6,0.75
    python -m gamebrains.experiments.sweep --vary param:qlearning:alpha \\
        --values 0.02,0.05,0.1,0.2,0.4 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

from ..agents import factory
from ..engine.eventlog import EventLog
from ..engine.runner import run_match
from ..games.congestion import from_preset as congestion_from_preset
from ..games.public_goods import PublicGoodsGame
from ..metrics import graph, information, social
from ..repository.record import DEFAULT_CODE_VERSION, protocol_from_run, record_experiment

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

#: Roster compositions offered by name. A mix is a function of population size so that the same
#: name means the same thing at every size, and returns None where it has no meaning, which is how
#: "one of each" declines a population of three rather than quietly dropping an architecture.
MIXES: dict[str, Callable[[int], Optional[dict[str, int]]]] = {
    "all_qlearning": lambda n: {"qlearning": n},
    "all_dqn": lambda n: {"dqn": n},
    "all_fep": lambda n: {"fep": n},
    "all_classic": lambda n: {"classic": n},
    "one_of_each": lambda n: ({"qlearning": 1, "dqn": 1, "fep": 1, "classic": 1}
                              if n == 4 else None),
    "learners_vs_classic": lambda n: ({"qlearning": n // 2, "classic": n - n // 2}
                                      if n >= 2 else None),
}

#: What may be varied. `param:<kind>:<name>` reaches any hyperparameter the factory accepts, which
#: is what makes a cognitive parameter a sweepable axis rather than a constant nobody moved.
AXES = ("n_agents", "mpcr", "rounds", "seed")


@dataclass(frozen=True)
class Cell:
    """One run, described completely enough to be executed anywhere."""

    game: str = "public_goods"
    n_agents: int = 4
    rounds: int = 500
    seed: int = 0
    mpcr: float = 0.5
    mix: str = "all_qlearning"
    overrides: tuple[tuple[str, Any], ...] = ()
    """Hyperparameter overrides as a sorted tuple of `("<kind>:<name>", value)`, since a cell has
    to be hashable to be compared and deduplicated."""

    def label(self) -> str:
        parts = [self.game, f"n{self.n_agents}", f"r{self.rounds}", f"s{self.seed}", self.mix]
        if self.game == "public_goods":
            parts.insert(2, f"mpcr{self.mpcr:g}")
        parts += [f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.overrides]
        return " ".join(parts)

    def overrides_for(self, kind: str) -> dict[str, Any]:
        prefix = f"{kind}:"
        return {k[len(prefix):]: v for k, v in self.overrides if k.startswith(prefix)}


def set_axis(cell: Cell, axis: str, value: Any) -> Cell:
    """The same cell with one thing changed, which is the whole idea of a sweep."""
    if axis in ("n_agents", "rounds", "seed"):
        return replace(cell, **{axis: int(value)})
    if axis == "mpcr":
        return replace(cell, mpcr=float(value))
    if axis.startswith("param:"):
        _, kind, name = axis.split(":", 2)
        key = f"{kind}:{name}"
        kept = tuple((k, v) for k, v in cell.overrides if k != key)
        return replace(cell, overrides=tuple(sorted(kept + ((key, value),))))
    raise ValueError(f"cannot vary {axis!r}: expected one of {', '.join(AXES)} "
                     f"or param:<kind>:<name>")


def axis_value(cell: Cell, axis: str) -> Optional[float]:
    """What this cell's value on the swept axis is, for plotting it against everything else.

    None when the cell carries no value for that axis, which happens when a sweep is read back
    against a different axis than it was run on. Drawing it as zero would put a point on the chart
    that no run produced.
    """
    if axis in ("n_agents", "rounds", "seed"):
        return float(getattr(cell, axis))
    if axis == "mpcr":
        return float(cell.mpcr)
    if axis.startswith("param:"):
        _, kind, name = axis.split(":", 2)
        value = cell.overrides_for(kind).get(name)
        return float(value) if isinstance(value, (int, float)) else None
    return None


def build_cells(base: Cell, axis: str, values: Iterable[Any],
                seeds: Iterable[int] = (0,)) -> list[Cell]:
    """Every combination of the varied values and the seeds, with invalid cells left out.

    A mix that has no meaning at a population size is dropped here rather than failing in a worker,
    because a sweep that dies on its eleventh cell has wasted ten.
    """
    cells = []
    for value in values:
        for seed in seeds:
            cell = set_axis(replace(base, seed=int(seed)), axis, value)
            if MIXES[cell.mix](cell.n_agents) is not None:
                cells.append(cell)
    return cells


def make_game(cell: Cell):
    if cell.game == "public_goods":
        return PublicGoodsGame(n_agents=cell.n_agents, rounds=cell.rounds, mpcr=cell.mpcr)
    return congestion_from_preset(cell.game, cell.n_agents)


def make_roster(cell: Cell, game: Any) -> list[Any]:
    """Deterministic given the cell, which is what lets the parent rebuild what a worker ran."""
    counts = MIXES[cell.mix](cell.n_agents)
    if counts is None:
        raise ValueError(f"the {cell.mix!r} mix has no meaning at {cell.n_agents} agents")
    roster, index = [], 0
    for kind, count in counts.items():
        for _ in range(count):
            roster.append(factory.build(kind, index, game, cell.seed,
                                        overrides=cell.overrides_for(kind)))
            index += 1
    return roster


def run_cell(cell: Cell) -> dict[str, Any]:
    """One match with its measures, in this process. The entry point the pool calls.

    Returns the cell, the event log's path and the metrics. Nothing is appended to the ledger from
    here, because the chain has one writer.
    """
    started = time.perf_counter()
    game = make_game(cell)
    roster = make_roster(cell, game)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = RESULTS_DIR / f"sweep_{os.getpid()}_{stamp}.jsonl"

    # The runner narrates to stdout, and eleven workers narrating at once is noise that hides the
    # one line that matters, which is the progress the caller prints.
    with contextlib.redirect_stdout(io.StringIO()):
        with EventLog(path=log_path) as log:
            records = run_match(game, roster, rounds=cell.rounds, seed=cell.seed, eventlog=log)
        metrics = social.compute_all(records, game.max_welfare_per_round())
        metrics.update(information.compute_all(records, seed=cell.seed, roster=roster))
        metrics.update(graph.compute_all(records, metrics["transfer_entropy_detail"]))
        protocol = protocol_from_run(records)

    return {"cell": cell, "log_path": str(log_path), "metrics": metrics,
            "protocol": protocol, "seconds": time.perf_counter() - started}


@dataclass
class Progress:
    """What a caller needs to draw a loader, and what the command line prints."""

    total: int = 0
    done: int = 0
    failed: int = 0
    skipped: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    rows: list[dict[str, Any]] = field(default_factory=list)
    last: str = ""

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def remaining_estimate(self) -> Optional[float]:
        """Seconds left at the rate so far, or None before anything has finished.

        Deliberately the simple estimate. A sweep whose cells differ in cost will mislead it, and
        a number that says "about" is more use than no number while the cheap cells finish first.
        """
        if self.done < 1:
            return None
        return (self.elapsed / self.done) * max(0, self.total - self.done - self.skipped)


def read_journal(path: Optional[Path]) -> set[str]:
    """Cell labels this sweep has already finished.

    A journal of labels rather than a check against the ledger's configuration hashes, because a
    hash is only known after a cell has run and the whole point is to not run it. The label is
    complete: it carries every field of the cell, so two cells share one only if they are the same
    experiment.
    """
    if path is None or not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return set()                    # a half-written journal costs a rerun, never a crash


def _append_journal(path: Optional[Path], done: set[str], label: str) -> None:
    """Written after every cell, not at the end. The machine this is built to scale onto has no
    uninterruptible power supply, so a journal flushed at the end is a journal that is empty
    exactly when it is needed."""
    if path is None:
        return
    done.add(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(done), indent=0), encoding="utf-8")


def run_sweep(cells: list[Cell], repo_root: str | Path, workers: Optional[int] = None,
              record: bool = True, journal: Optional[Path] = None,
              progress: Optional[Progress] = None) -> Iterator[Progress]:
    """Run the cells across processes, recording each as it lands. Yields after every cell.

    A generator rather than a callback, so the command line and the web view consume the same
    thing: one prints a line and the other moves a bar.
    """
    progress = progress or Progress()
    progress.total = len(cells)

    finished = read_journal(journal)
    todo = [cell for cell in cells if cell.label() not in finished]
    progress.skipped = len(cells) - len(todo)
    if not todo:
        progress.last = ("everything in this sweep is already done" if progress.skipped
                         else "nothing to run")
        yield progress
        return

    # One worker per core bar one. The spare is not politeness: a fully committed machine makes
    # the interface drawing the progress bar stop responding, and the whole point of this module
    # is that the sweep is watchable while it runs.
    workers = workers or max(1, (os.cpu_count() or 2) - 1)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_cell, cell): cell for cell in todo}
        for future in as_completed(futures):
            cell = futures[future]
            try:
                result = future.result()
            except Exception as failure:                     # a cell, not the sweep
                progress.failed += 1
                progress.last = f"{cell.label()}: {type(failure).__name__}: {failure}"
                yield progress
                continue

            metrics = result["metrics"]
            if record:
                game = make_game(cell)
                roster = make_roster(cell, game)
                record_experiment(
                    repo_root, game, roster, result["log_path"], rounds=cell.rounds,
                    seed=cell.seed, metrics=metrics, code_version=DEFAULT_CODE_VERSION,
                    protocol=result["protocol"])

            progress.done += 1
            progress.last = f"{cell.label()} in {result['seconds']:.1f}s"
            progress.rows.append({
                "label": cell.label(), "cell": cell, "seconds": result["seconds"],
                "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            })
            # Journalled only once the record is in the chain, so a power cut between the two
            # reruns the cell rather than losing it.
            _append_journal(journal, finished, cell.label())
            yield progress


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vary", default="mpcr",
                        help="what to move: n_agents, mpcr, rounds, seed, "
                             "or param:<kind>:<name> for a hyperparameter")
    parser.add_argument("--values", default="0.3,0.4,0.5,0.6,0.75",
                        help="comma separated values for that axis")
    parser.add_argument("--seeds", default="0", help="comma separated seeds per value")
    parser.add_argument("--game", default="public_goods",
                        help="public_goods, or a congestion preset such as hjg")
    parser.add_argument("--mix", default="all_qlearning", choices=sorted(MIXES))
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-record", action="store_true",
                        help="run without writing to the ledger, for a dry look at the cost")
    parser.add_argument("--name", default="",
                        help="journal name, so an interrupted sweep of this shape continues "
                             "instead of repeating itself. Defaults to the axis and the mix")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1] / "repo_store"))
    args = parser.parse_args(argv)

    def parse(raw: str) -> list[Any]:
        out = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            out.append(float(token) if "." in token else int(token))
        return out

    base = Cell(game=args.game, n_agents=args.agents, rounds=args.rounds, mix=args.mix)
    cells = build_cells(base, args.vary, parse(args.values), [int(s) for s in parse(args.seeds)])
    if not cells:
        print("no valid cells: that roster mix has no meaning at this population size")
        return 1

    journal = None
    if not args.no_record:
        name = args.name or f"{args.vary}_{args.mix}_{args.game}".replace(":", "-")
        journal = RESULTS_DIR / f"sweep_journal_{name}.json"

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"{len(cells)} cells over {workers} workers, varying {args.vary}")
    for progress in run_sweep(cells, args.repo, workers=workers,
                              record=not args.no_record, journal=journal):
        left = progress.remaining_estimate
        eta = f", about {left / 60:.1f} min left" if left else ""
        counted = progress.done + progress.failed + progress.skipped
        print(f"  [{counted}/{progress.total}] {progress.last}{eta}")

    verb = "recorded" if not args.no_record else "run without recording"
    already = f", {progress.skipped} already done" if progress.skipped else ""
    print(f"\n{progress.done} {verb}, {progress.failed} failed{already}, in "
          f"{progress.elapsed / 60:.1f} min")
    return 1 if progress.failed and not progress.done else 0


if __name__ == "__main__":
    raise SystemExit(main())
