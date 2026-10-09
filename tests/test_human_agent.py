"""A person in a seat, under the same contract as every other architecture.

The interesting assertions here are the ones that would pass for the wrong reason if the seat
cheated. A human seat is easy to make "work" by handing it an action index and letting the page
print "press 1", and easy to make reproducible-looking by giving it a default move. Both would be
false in ways nothing else in the suite would notice, so each has a test that fails on it.

Run:  "$GB/gamebrains/.venv/Scripts/python.exe" -m gamebrains.tests.test_human_agent
"""

from __future__ import annotations

import sys
import threading
import time

from ..agents.classic import AllD
from ..agents.human import HumanAgent, WaitingForPerson
from ..engine.runner import run_match
from ..games.congestion import CongestionGame
from ..games.public_goods import PublicGoodsGame
from ..metrics import social
from ..repository import record as record_mod


def _games():
    return [("public_goods", PublicGoodsGame(n_agents=3)),
            ("congestion", CongestionGame(n_agents=3))]


def _play(game, rounds: int, answer: str, wait_seconds: float = 5.0):
    """Drive a match from another thread, answering `answer` once per round.

    Waits for a *new* round rather than for a non-empty prompt, because the seat clears its
    prompt only after the move is consumed and answering twice at one board is refused.
    """
    person = HumanAgent("You", wait_seconds=wait_seconds)
    roster = [person, AllD("B"), AllD("C")]
    out: dict = {}
    thread = threading.Thread(
        target=lambda: out.update(result=run_match(game, roster, rounds=rounds, seed=0)),
        daemon=True)
    thread.start()
    seen = -1
    while thread.is_alive():
        prompt = person.pending
        if prompt and prompt["round"] != seen:
            seen = prompt["round"]
            person.submit_role(answer)
        time.sleep(0.005)
    thread.join(timeout=15)
    return person, out.get("result")


def test_one_seat_plays_both_games_through_the_words() -> None:
    """Conceding must mean conceding in both games, which number that move oppositely.

    This is the assertion that a seat handing out raw indices cannot pass: the same instruction
    resolves to action 1 in one game and action 0 in the other.
    """
    chosen = {}
    for name, game in _games():
        person, result = _play(game, 8, "concede")
        rate = social.compute_all(result, game.max_welfare_per_round())["cooperation_rate"]
        assert abs(rate - 1 / 3) < 1e-9, (
            f"{name}: one of three players conceded every round, so the cooperation rate must be "
            f"1/3 and reads {rate:.4f}")
        chosen[name] = {int(row[0]) for row in result["actions"]}
        assert len(chosen[name]) == 1, f"{name}: the person did not play one constant move"
    assert chosen["public_goods"] != chosen["congestion"], (
        "the two games now number conceding the same way, so this test no longer shows that the "
        f"seat went through the words rather than through an index: {chosen}")
    print(f"OK: one instruction, opposite indices in the two games {chosen}")


def test_the_seat_refuses_to_move_before_it_knows_the_game() -> None:
    person = HumanAgent("You")
    try:
        person.act(0)
    except RuntimeError as exc:
        assert "has not been matched to a game" in str(exc)
        print("OK: acting before the game is known is refused, with a reason")
        return
    raise AssertionError("the seat played without knowing what its moves mean")


def test_nothing_is_played_on_a_persons_behalf() -> None:
    """A timeout must stop the match, not invent a move.

    The failure this guards is the tempting one: a default action keeps the demo running and puts
    a move in the event log that nobody made, indistinguishable from one they did.
    """
    game = PublicGoodsGame(n_agents=3)
    person = HumanAgent("You", wait_seconds=0.15)
    roster = [person, AllD("B"), AllD("C")]
    error: dict = {}
    thread = threading.Thread(
        target=lambda: error.setdefault("raised", _capture(game, roster)), daemon=True)
    thread.start()
    thread.join(timeout=10)
    assert isinstance(error.get("raised"), WaitingForPerson), (
        f"the match should have stopped when nobody answered, and it {error.get('raised')!r}")
    print("OK: an unanswered prompt stops the match rather than substituting a move")


def _capture(game, roster):
    try:
        run_match(game, roster, rounds=4, seed=0)
    except BaseException as exc:            # noqa: BLE001 - the type is the assertion
        return exc
    return None


def test_two_moves_at_one_board_are_refused() -> None:
    """Answering twice before the first move is consumed would play against an unseen board."""
    game = PublicGoodsGame(n_agents=3)
    person = HumanAgent("You", wait_seconds=5.0)
    person.on_match_start(game)
    person.submit_role("concede")
    try:
        person.submit_role("claim")
    except RuntimeError as exc:
        assert "already waiting" in str(exc)
        print("OK: a second move at the same board is refused")
        return
    raise AssertionError("a second move was accepted and would be played a round later")


def test_a_run_with_a_person_withholds_the_reproduction_claim() -> None:
    """The tier written for a served model is the right answer for a person too."""
    assert HumanAgent("You").reproducibility() == "none"
    roster = [HumanAgent("You"), AllD("B")]
    claim = record_mod.reproducibility_of(roster)
    assert claim["tier"] == "none", f"a roster with a person claimed {claim['tier']!r}"
    assert not claim["byte_identical_claimed"]
    assert any(entry["kind"] == "human" for entry in claim["nondeterministic_agents"]), (
        "the record must name the seat that withheld the claim, not just withhold it")
    print("OK: a run containing a person records tier 'none' and names the seat")


def test_the_panel_says_there_is_nothing_to_show_rather_than_showing_nothing() -> None:
    game = PublicGoodsGame(n_agents=3)
    person = HumanAgent("You")
    person.on_match_start(game)
    panel = person.render_brain()
    assert panel["opaque"] is True and panel["note"], (
        "every other seat renders internal state; this one has none to render and must say so")
    print("OK: the transparency panel states the absence rather than drawing an empty box")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}\n     {exc}")
    print(f"\n{'all human-seat tests passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
