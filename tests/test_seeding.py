"""Replicates at different master seeds must not share any agent's random stream.

The factory used to give seat `index` of a kind the seed `master + offset + index`, so master 1's
seats 0 to 3 drew exactly the streams of master 0's seats 1 to 4. A batch of "independent" replicates
run from the web interface or a sweep at consecutive master seeds shared most of their randomness,
and an interval computed over them was narrower than the data warranted. Nothing failed. Found by an
external review of the code.

The checks build real agents through the factory and read the seed each one was given, so they
cover every kind the factory builds without restating its arithmetic.

Run:  "$GB/gamebrains/.venv/Scripts/python.exe" -m gamebrains.tests.test_seeding
"""

from __future__ import annotations

import sys

from ..agents import factory
from ..games.public_goods import PublicGoodsGame

KINDS = ("qlearning", "fep", "markov_brain", "dac", "markov_gates")
SEATS = 10


def _seat_seeds(master: int) -> dict[str, list[int]]:
    """The seed every seat received, read back from the factory's own arithmetic per kind."""
    return {kind: [factory._seat_seed(master, kind, i) for i in range(SEATS)] for kind in KINDS}


def test_no_two_master_seeds_share_a_seat_seed() -> None:
    seen: dict[int, tuple[int, str, int]] = {}
    for master in range(20):
        for kind, seeds in _seat_seeds(master).items():
            for i, seed in enumerate(seeds):
                assert seed not in seen, (
                    f"master {master} {kind} seat {i} and master {seen[seed][0]} "
                    f"{seen[seed][1]} seat {seen[seed][2]} share seed {seed}")
                seen[seed] = (master, kind, i)
    print("OK: twenty master seeds, four kinds, ten seats each, and no seed is used twice")


def test_agents_built_at_consecutive_master_seeds_never_share_a_seed() -> None:
    """The behaviour itself, read off built agents, so it fails on any scheme with overlap."""
    game = PublicGoodsGame(n_agents=5)
    owner: dict[int, tuple[int, int]] = {}
    for master in range(6):
        for seat in range(5):
            seed = factory.build("dac", seat, game, master).seed
            assert seed not in owner, (
                f"master {master} seat {seat} was built with seed {seed}, already used by master "
                f"{owner[seed][0]} seat {owner[seed][1]}")
            owner[seed] = (master, seat)
    print("OK: agents built at six consecutive master seeds carry thirty distinct seeds")


def test_the_factory_uses_the_seat_seed() -> None:
    """Guards the guard: the agents the factory builds must carry the seeds checked above."""
    game = PublicGoodsGame(n_agents=4)
    agent = factory.build("dac", 3, game, 7)
    assert agent.seed == factory._seat_seed(7, "dac", 3), (
        "the factory builds agents with a seed other than the one this test checks")
    print("OK: an agent built by the factory carries its seat seed")


def test_the_same_master_seed_reproduces_its_seats() -> None:
    assert _seat_seeds(5) == _seat_seeds(5)
    print("OK: one master seed always gives the same seat seeds")


def test_a_seat_index_that_would_collide_is_refused() -> None:
    try:
        factory._seat_seed(0, "qlearning", factory._MAX_SEATS_PER_KIND)
    except ValueError:
        print("OK: a seat index that would reach the next kind's seeds is refused")
        return
    raise AssertionError("a seat index past the per-kind range was accepted")


if __name__ == "__main__":
    failures = 0
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {fn_name}\n     {exc}")
            except Exception as exc:
                failures += 1
                print(f"ERROR {fn_name}\n     {type(exc).__name__}: {exc}")
    print(f"\n{'all seeding tests passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
