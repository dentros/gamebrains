"""Playing a seat from the browser: the match runs in a thread and the page drives it.

A person cannot answer inside one HTTP request, so a roster containing one takes a different
route through the web layer. What must stay the same is everything after that: the same pipeline,
the same metrics, the same record. These tests drive the real routes with Flask's test client, so
a change that quietly gave human matches their own path would fail here.

Run:  "$GB/gamebrains/.venv/Scripts/python.exe" -m gamebrains.tests.test_live_play
"""

from __future__ import annotations

import sys
import time

from ..webui import live
from ..webui.app import app

#: The roster form the match page posts. Every parallel list has to be present and the same
#: length, which is the form's own contract: the roster loop zips them and stops at the shortest.
FORM = {
    "game_kind": "public_goods", "seed": "0", "log_lines": "5",
    "row_kind[]": ["human", "classic"], "row_count[]": ["1", "2"],
    "row_classic_strategy[]": ["", "AllD"], "row_reciprocity[]": ["", ""],
    "row_markov_hidden[]": ["", ""], "row_epsilon_decay[]": ["", ""],
    "row_epsilon_min[]": ["", ""], "row_genome_origin[]": ["", ""],
    "row_genome_cid[]": ["", ""],
}


def _start(client, rounds: int = 6) -> str:
    response = client.post("/run", data={**FORM, "rounds": str(rounds)})
    assert response.status_code == 302, (
        f"a roster with a person must redirect to the board, not render a result "
        f"({response.status_code})")
    return response.headers["Location"].rsplit("/", 1)[-1]


def _answer_until_done(client, token: str, role: str = "concede", limit: int = 200) -> int:
    """Answer every prompt until the match ends. Returns how many moves were made."""
    answered, seen = 0, -1
    for _ in range(limit):
        match = live.get(token)
        if match is None or not match.running:
            return answered
        prompt = match.prompt
        # Answer each round once. The prompt stays visible until the seat consumes the move, so
        # posting on every pass would count rejected double submits as moves -- which is exactly
        # what an earlier version of this helper did, reporting 161 moves in a six-round match.
        if prompt is not None and prompt["round"] != seen:
            seen = prompt["round"]
            client.post(f"/play/{token}", data={"role": role})
            answered += 1
        else:
            time.sleep(0.02)
    raise AssertionError(f"the match was still running after {limit} turns")


def test_a_human_roster_goes_to_the_board_and_plays_through() -> None:
    client = app.test_client()
    token = _start(client, rounds=6)

    # The first page can land before the thread has reached the first decision, in which case it
    # says so and refreshes itself. That is the real behaviour, so the test waits the way a
    # browser does rather than asserting against a race it would sometimes lose.
    deadline = time.time() + 10
    while time.time() < deadline:
        board = client.get(f"/play/{token}")
        assert board.status_code == 200
        if 'name="role"' in board.get_data(as_text=True):
            break
        assert "refresh" in board.get_data(as_text=True).lower(), (
            "a board with nothing to answer must refresh itself, or the page is a dead end")
        time.sleep(0.05)
    else:
        raise AssertionError("the board never offered a move")

    answered = _answer_until_done(client, token)
    assert answered == 6, f"a six-round match asked for {answered} moves"
    print(f"OK: the board asked for and received {answered} moves")


def test_the_result_is_the_ordinary_results_page() -> None:
    """A human match must end in the same page every other match ends in.

    The point of running the existing pipeline in a thread rather than writing a second one is
    that the result is not a special case, so this asserts the page carries the same metric the
    synchronous path produces.
    """
    client = app.test_client()
    token = _start(client, rounds=6)
    _answer_until_done(client, token)

    final = client.get(f"/play/{token}")
    assert final.status_code == 200, f"collecting the result gave {final.status_code}"
    page = final.get_data(as_text=True)
    assert "glyph-metrics" in page, "the results panel is not on the page the match ended in"
    assert "cooperation" in page.lower() or "συνεργασ" in page, (
        "the ordinary measures are missing from a human match's result")
    print("OK: a human match ends in the same results page as any other")


def test_a_collected_match_is_released() -> None:
    """A result is rendered once and the match stops holding a thread and a roster."""
    client = app.test_client()
    token = _start(client, rounds=4)
    _answer_until_done(client, token)
    client.get(f"/play/{token}")
    assert live.get(token) is None, "the finished match is still being held"
    again = client.get(f"/play/{token}")
    assert again.status_code == 404, (
        f"a collected match should be gone, and the page returned {again.status_code}")
    print("OK: a collected result is released, and asking again says so")


def test_a_roster_without_a_person_still_runs_in_the_request() -> None:
    """The ordinary path must be untouched: no redirect, no thread, a result straight back."""
    client = app.test_client()
    response = client.post("/run", data={
        **FORM, "rounds": "20",
        "row_kind[]": ["classic", "classic"], "row_count[]": ["2", "2"],
        "row_classic_strategy[]": ["AllC", "AllD"],
    })
    assert response.status_code == 200, (
        f"a roster of ordinary agents should render its result directly, got "
        f"{response.status_code}")
    print("OK: a roster with no person is unaffected and answers in the request")


def test_a_move_offered_when_nothing_is_waiting_is_reported() -> None:
    """A stale tab posting twice must be told, not silently ignored."""
    client = app.test_client()
    token = _start(client, rounds=4)
    deadline = time.time() + 5
    while live.get(token) and live.get(token).prompt is None and time.time() < deadline:
        time.sleep(0.02)
    client.post(f"/play/{token}", data={"role": "concede"})
    # The seat is now computing the next round; a second post has nothing to answer.
    response = client.post(f"/play/{token}", data={"role": "concede"})
    body = response.get_data(as_text=True)
    assert response.status_code in (200, 302), f"unexpected status {response.status_code}"
    if response.status_code == 200:
        assert "badge warn" in body or "waiting" in body.lower(), (
            "a move with nothing to answer was swallowed without a word")
    _answer_until_done(client, token)
    print("OK: a move posted with nothing waiting is acknowledged rather than dropped")


def test_no_pair_ever_cooperating_is_a_result_and_not_a_crash() -> None:
    """One AllC against two AllD: no two agents ever cooperate in the same round.

    Every co-cooperation weight is then zero, which networkx's weighted clustering divides by.
    The interface answered 500 for that perfectly ordinary roster until the measure returned zero
    itself. Kept here rather than only in the metric's own tests because this is the path a person
    actually hits.
    """
    client = app.test_client()
    response = client.post("/run", data={
        **FORM, "rounds": "30",
        "row_kind[]": ["classic", "classic"], "row_count[]": ["1", "2"],
        "row_classic_strategy[]": ["AllC", "AllD"],
    })
    assert response.status_code == 200, (
        f"a roster where no pair ever co-cooperates returned {response.status_code}")
    print("OK: a match with no co-cooperation at all reports zero rather than failing")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}\n     {exc}")
    print(f"\n{'all live-play tests passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
