"""A long run must resume where it died, and resuming must not change the answer.

The failure this guards is mundane and expensive: a hundred-seed job dies at seed 67 and the whole
night is gone. `experiments/sweep.py` already journalled per cell for exactly this reason; the
multi-seed runner, which is the longer job of the two, did not.

The assertions are about the property rather than the plumbing. A journal that resumed but gave
different numbers would be worse than none at all, because the run would finish quickly and be
wrong, so every case below checks that the resumed answer equals the straight-through one.

Run:  "$GB/gamebrains/.venv/Scripts/python.exe" -m gamebrains.tests.test_multiseed_journal
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from ..experiments.run_multiseed import SeedJournal


def _counting_source():
    """A compute function that records how many times it really ran."""
    calls: list[str] = []

    def compute(key: str, value):
        def run():
            calls.append(key)
            return value
        return run

    return calls, compute


def test_a_stored_seed_is_not_recomputed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "j.jsonl"
        calls, compute = _counting_source()

        first = SeedJournal(str(path), "v1")
        got = [first.get_or_run(f"s{i}", compute(f"s{i}", [float(i), 1.0])) for i in range(4)]
        first.close()
        assert len(calls) == 4, "the first run should have computed every seed"

        calls.clear()
        second = SeedJournal(str(path), "v1")
        again = [second.get_or_run(f"s{i}", compute(f"s{i}", [float(i), 1.0])) for i in range(4)]
        second.close()
        assert not calls, f"a resumed run recomputed {calls}"
        assert again == got, "the resumed values differ from the originals"
    print("OK: a stored seed is read back rather than recomputed, with the same value")


def test_only_the_missing_seeds_are_computed() -> None:
    """The case that matters: the job died partway and the rest must still run."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "j.jsonl"
        calls, compute = _counting_source()

        partial = SeedJournal(str(path), "v1")
        for i in range(2):
            partial.get_or_run(f"s{i}", compute(f"s{i}", [float(i)]))
        partial.close()
        calls.clear()

        resumed = SeedJournal(str(path), "v1")
        values = [resumed.get_or_run(f"s{i}", compute(f"s{i}", [float(i)])) for i in range(5)]
        resumed.close()
        assert calls == ["s2", "s3", "s4"], f"recomputed the wrong set: {calls}"
        assert values == [[float(i)] for i in range(5)]
    print("OK: only the seeds that were missing are computed")


def test_a_journal_from_other_code_is_discarded_rather_than_merged() -> None:
    """Numbers from two versions of the code in one run are not interpretable."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "j.jsonl"
        old = SeedJournal(str(path), "v1")
        old.get_or_run("s0", lambda: [1.0])
        old.close()

        calls, compute = _counting_source()
        new = SeedJournal(str(path), "v2")
        value = new.get_or_run("s0", compute("s0", [2.0]))
        new.close()
        assert calls == ["s0"], "the stale entry was reused across a code change"
        assert value == [2.0]
    print("OK: a journal written under other code is started over, not merged")


def test_a_torn_final_line_costs_one_seed_and_not_the_rest() -> None:
    """A kill mid-append leaves half a line; everything before it is still good."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "j.jsonl"
        good = SeedJournal(str(path), "v1")
        for i in range(3):
            good.get_or_run(f"s{i}", lambda i=i: [float(i)])
        good.close()

        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"key": "s3", "val')

        calls, compute = _counting_source()
        resumed = SeedJournal(str(path), "v1")
        values = [resumed.get_or_run(f"s{i}", compute(f"s{i}", [float(i)])) for i in range(4)]
        resumed.close()
        assert calls == ["s3"], f"a torn line cost more than the one seed it holds: {calls}"
        assert values == [[float(i)] for i in range(4)]
    print("OK: a torn final line costs the one seed it was writing")


def test_no_path_means_no_journal_and_no_files() -> None:
    """A quick check should not leave files behind, so journalling is opt-in."""
    with tempfile.TemporaryDirectory() as tmp:
        before = set(Path(tmp).iterdir())
        calls, compute = _counting_source()
        off = SeedJournal(None, "v1")
        off.get_or_run("s0", compute("s0", [1.0]))
        off.get_or_run("s0", compute("s0", [1.0]))
        off.close()
        assert set(Path(tmp).iterdir()) == before, "a journal file appeared without a path"
        # Still memoised within the process, which is free and keeps one run self-consistent.
        assert calls == ["s0"], f"computed twice in one process: {calls}"
    print("OK: without a path nothing is written, and one process still computes each seed once")


def test_the_header_names_the_code_version() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "j.jsonl"
        journal = SeedJournal(str(path), "git:abc1234")
        journal.get_or_run("s0", lambda: [1.0])
        journal.close()
        header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert header.get("code_version") == "git:abc1234", (
            "without the version in the header nothing can refuse a stale journal")
    print("OK: the journal records the code version it was written under")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}\n     {exc}")
    print(f"\n{'all journal tests passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
