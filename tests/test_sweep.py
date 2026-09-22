"""Varying one thing across a range, in parallel, without losing work to a power cut.

Most of what can go wrong in a parallel sweep is invisible in its output. Two processes appending
to a signed chain produce a chain that verifies until somebody checks it. A roster rebuilt in the
parent to record what a worker ran is only correct while every recorded parameter is a
construction-time value. A journal flushed at the end is empty exactly when it is needed. So these
tests are about the seams rather than about the arithmetic:

  * the roster the parent rebuilds is the roster the worker ran, parameter for parameter
  * a cell varies in exactly one thing, and its label says every other thing
  * a finished cell is not run again, and one that failed is
  * a mix that has no meaning at a population size is dropped before anything starts

The parallel path itself is exercised with a two-cell sweep over real processes, because a sweep
that only ever runs in one process is not the thing being tested.

Run: python -m gamebrains.tests.test_sweep
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
from pathlib import Path

from ..agents import factory
from ..engine.runner import run_match
from ..experiments import sweep
from ..repository.ledger import Ledger
from ..repository.record import _agent_params


def test_one_axis_moves_and_the_label_says_everything_else() -> None:
    base = sweep.Cell(n_agents=4, rounds=200, mpcr=0.5, mix="all_qlearning")
    moved = sweep.set_axis(base, "mpcr", 0.75)

    assert moved.mpcr == 0.75
    assert (moved.n_agents, moved.rounds, moved.mix) == (4, 200, "all_qlearning")
    assert "mpcr0.75" in moved.label() and "n4" in moved.label() and "r200" in moved.label()
    assert base.label() != moved.label()
    print("OK: a swept cell differs in one field and its label carries the rest")


def test_a_cognitive_parameter_is_an_axis_like_any_other() -> None:
    """The point of the feature: not just game size, but what the agent's mind is set to."""
    cell = sweep.set_axis(sweep.Cell(mix="all_qlearning"), "param:qlearning:alpha", 0.3)

    assert cell.overrides_for("qlearning") == {"alpha": 0.3}
    assert cell.overrides_for("dqn") == {}
    assert "qlearning:alpha=0.3" in cell.label()

    again = sweep.set_axis(cell, "param:qlearning:alpha", 0.4)
    assert again.overrides_for("qlearning") == {"alpha": 0.4}, "a swept axis must replace, not add"
    print("OK: a hyperparameter is a sweepable axis and sweeping it replaces rather than piles up")


def test_an_unknown_axis_is_refused_by_name() -> None:
    try:
        sweep.set_axis(sweep.Cell(), "temperature", 0.7)
    except ValueError as exc:
        assert "n_agents" in str(exc) and "param:" in str(exc)
    else:
        raise AssertionError("an unknown axis was accepted")
    print("OK: an axis that does not exist is refused, and the message lists the ones that do")


def test_a_mix_with_no_meaning_at_this_size_is_dropped_before_anything_runs() -> None:
    """A sweep that dies on its eleventh cell has wasted ten."""
    base = sweep.Cell(mix="one_of_each")
    cells = sweep.build_cells(base, "n_agents", [2, 3, 4, 5])

    assert [c.n_agents for c in cells] == [4], [c.n_agents for c in cells]
    print("OK: the one-of-each mix declines every population but four, and does it up front")


def test_the_parent_rebuilds_exactly_the_roster_the_worker_ran() -> None:
    """The assumption the whole design rests on. If a recorded parameter were mutated by training,
    the record would describe a roster that never played."""
    cell = sweep.Cell(n_agents=4, rounds=60, mix="one_of_each",
                      overrides=(("qlearning:alpha", 0.3),))
    game = sweep.make_game(cell)

    first = sweep.make_roster(cell, game)
    with contextlib.redirect_stdout(io.StringIO()):        # play it, so anything mutable moves
        run_match(game, first, rounds=cell.rounds, seed=cell.seed)

    second = sweep.make_roster(cell, sweep.make_game(cell))
    assert [a.name for a in first] == [a.name for a in second]
    assert [a.kind for a in first] == [a.kind for a in second]
    for played, fresh in zip(first, second):
        assert _agent_params(played) == _agent_params(fresh), (
            f"{played.name}: {_agent_params(played)} after a match, "
            f"{_agent_params(fresh)} when rebuilt")
    assert _agent_params(first[0])["alpha"] == 0.3, "the override has to survive into the record"
    print("OK: every recorded parameter survives a match unchanged, so the rebuild is exact")


def test_a_language_model_seat_is_refused_rather_than_fanned_out() -> None:
    """It needs a live backend and keeps a journal only its own process holds."""
    try:
        factory.build("llm", 0, sweep.make_game(sweep.Cell()), 0)
    except ValueError as exc:
        assert "journal" in str(exc) or "backend" in str(exc), str(exc)
    else:
        raise AssertionError("a model seat was built by the batch factory")
    print("OK: the batch factory will not seat a language model")


def test_a_finished_cell_is_not_run_again() -> None:
    root = Path(tempfile.mkdtemp())
    try:
        journal = root / "journal.json"
        cells = sweep.build_cells(sweep.Cell(rounds=40), "mpcr", [0.4, 0.6])
        journal.write_text(json.dumps([cells[0].label()]), encoding="utf-8")

        final = None
        for final in sweep.run_sweep(cells, root, workers=2, record=False, journal=journal):
            pass
        assert final.skipped == 1 and final.done == 1, (final.skipped, final.done)

        for final in sweep.run_sweep(cells, root, workers=2, record=False, journal=journal):
            pass
        assert final.skipped == 2 and final.done == 0
        assert "already done" in final.last
        print("OK: a journalled cell is skipped, and a fully journalled sweep runs nothing")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_corrupt_journal_costs_a_rerun_and_never_a_crash() -> None:
    root = Path(tempfile.mkdtemp())
    try:
        journal = root / "journal.json"
        journal.write_text("{ this is not json", encoding="utf-8")
        assert sweep.read_journal(journal) == set()
        assert sweep.read_journal(root / "nothing.json") == set()
        assert sweep.read_journal(None) == set()
        print("OK: an unreadable journal reruns the sweep instead of stopping it")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_two_cell_sweep_runs_in_parallel_and_lands_in_the_chain() -> None:
    """The real path: separate processes compute, this process writes, the chain verifies."""
    root = Path(tempfile.mkdtemp())
    try:
        cells = sweep.build_cells(sweep.Cell(rounds=60, n_agents=3), "mpcr", [0.4, 0.7])
        final = None
        for final in sweep.run_sweep(cells, root, workers=2, record=True,
                                     journal=root / "journal.json"):
            pass

        assert final.done == 2 and final.failed == 0, final.last
        assert len(final.rows) == 2
        assert all("cooperation_rate" in row["metrics"] for row in final.rows)

        records = Ledger(root, read_only=True).load_all()
        assert len(records) == 2
        assert len({r["config_hash"] for r in records}) == 2, "two designs, two hashes"
        assert json.loads((root / "journal.json").read_text(encoding="utf-8"))
        print(f"OK: two cells ran in two processes and the chain holds {len(records)} records")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_one_axis_moves_and_the_label_says_everything_else()
    test_a_cognitive_parameter_is_an_axis_like_any_other()
    test_an_unknown_axis_is_refused_by_name()
    test_a_mix_with_no_meaning_at_this_size_is_dropped_before_anything_runs()
    test_the_parent_rebuilds_exactly_the_roster_the_worker_ran()
    test_a_language_model_seat_is_refused_rather_than_fanned_out()
    test_a_finished_cell_is_not_run_again()
    test_a_corrupt_journal_costs_a_rerun_and_never_a_crash()
    test_a_two_cell_sweep_runs_in_parallel_and_lands_in_the_chain()
    print("\nall sweep tests passed")
