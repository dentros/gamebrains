"""The cooperation rate must mean cooperation in every game, not "action index 1".

This is the platform's own headline defect, found a second time one layer above where it was
fixed. `agents/` was migrated to ask each game what its indices mean, and that migration worked:
`AllC` really does play STAY in the congestion family now. `metrics/social.py` was never part of
that migration, so it went on counting `actions == 1`, which is cooperating in the Public Goods
Game and *claiming the resource* in a congestion game. A match in which every agent conceded in
every round was reported as zero cooperation.

Nothing failed. The number had the right type, the right range and a plausible value, and the only
way to see it was to ask a game with the opposite convention. That is the same reason the original
defect survived: a platform whose games share a convention cannot discover that the convention was
carrying the meaning.

The tests below pin the property rather than the number. A roster that cooperates in every round
must read 1.0 and one that never cooperates must read 0.0, **in both games**, whichever way each
game happens to number its actions. A future game with a third convention is covered by the same
assertions for free, which a test written against a stored value would not be.

Run:  "$GB/gamebrains/.venv/Scripts/python.exe" -m gamebrains.tests.test_cooperation_semantics
"""

from __future__ import annotations

import sys

import numpy as np

from ..agents.classic import AllC, AllD
from ..engine.runner import run_match
from ..games.congestion import CongestionGame
from ..games.public_goods import PublicGoodsGame
from ..metrics import social

ROUNDS = 60


def _games():
    """One game per action convention. The point of the pair is that they disagree."""
    return [("public_goods", PublicGoodsGame(n_agents=3)),
            ("congestion", CongestionGame(n_agents=3))]


def _play(game, agent_cls):
    roster = [agent_cls(f"A{i}") for i in range(3)]
    return run_match(game, roster, rounds=ROUNDS, seed=0)


def test_all_cooperators_read_as_full_cooperation() -> None:
    for name, game in _games():
        records = _play(game, AllC)
        rate = social.compute_all(records, game.max_welfare_per_round())["cooperation_rate"]
        assert abs(rate - 1.0) < 1e-9, (
            f"{name}: every agent conceded in every round and the cooperation rate reads "
            f"{rate:.3f}. The measure is counting an action index rather than the game's own "
            f"declaration of what cooperating is")
    print("OK: a roster that always cooperates reads 1.0 in both action conventions")


def test_all_defectors_read_as_no_cooperation() -> None:
    """The mirror, so the check cannot be satisfied by a measure that always returns 1."""
    for name, game in _games():
        records = _play(game, AllD)
        rate = social.compute_all(records, game.max_welfare_per_round())["cooperation_rate"]
        assert abs(rate) < 1e-9, (
            f"{name}: no agent ever conceded and the cooperation rate reads {rate:.3f}")
    print("OK: a roster that never cooperates reads 0.0 in both action conventions")


def test_the_two_games_really_do_disagree_about_the_index() -> None:
    """Guards the guard: if both games ever numbered cooperation the same way, the tests above
    would pass while testing nothing, and this defect could return unnoticed."""
    conventions = {}
    for name, game in _games():
        records = _play(game, AllC)
        taken = sorted(set(np.asarray(records["actions"]).flatten().tolist()))
        assert len(taken) == 1, f"{name}: AllC did not play one constant action, it played {taken}"
        conventions[name] = taken[0]
    assert conventions["public_goods"] != conventions["congestion"], (
        "both games now number cooperation the same way, so the pair no longer tests anything. "
        f"Add a game with the opposite convention: {conventions}")
    print(f"OK: the two games disagree about which index means cooperate ({conventions})")


def test_the_measure_follows_the_game_not_the_index() -> None:
    """The direct statement of the property, read off the game's own per-round count."""
    for name, game in _games():
        records = _play(game, AllC)
        declared = np.asarray(records["cooperators"], dtype=float).mean() / 3.0
        reported = social.compute_all(records, game.max_welfare_per_round())["cooperation_rate"]
        assert abs(declared - reported) < 1e-9, (
            f"{name}: the game counted {declared:.3f} of its players as cooperating and the "
            f"measure reported {reported:.3f}")
    print("OK: the reported rate is the game's own count, in both games")


def test_the_series_matches_the_headline() -> None:
    """The per-round series is drawn on the results page, so it has to carry the same meaning."""
    for name, game in _games():
        records = _play(game, AllC)
        out = social.compute_all(records, game.max_welfare_per_round())
        series = np.asarray(out["cooperation_series"], dtype=float)
        assert series.shape[0] == ROUNDS, f"{name}: the series has {series.shape[0]} rounds"
        assert abs(float(series.mean()) - out["cooperation_rate"]) < 1e-9, (
            f"{name}: the chart and the headline number disagree")
    print("OK: the per-round series agrees with the headline rate")


if __name__ == "__main__":
    failures = 0
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {fn_name}\n     {exc}")
    print(f"\n{'all cooperation-semantics tests passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
