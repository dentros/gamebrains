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
from ..engine.agent import Agent
from ..engine.game import Game, StepResult
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


# --- The runner's own count, for a game that reports none ----------------------------------------
#
# Both shipped games report `info["cooperators"]`, so the runner's fallback never ran in any test
# and read action 1 as cooperating for as long as it existed. These two games are what a third-party
# author following docs/adding-a-game.md could plausibly write: one declares only its roles, the
# other declares nothing.


class _ConcedeAtZero(Game):
    """Declares `concede` at index 0 and reports no count, so only the role can say who conceded."""

    name = "concede_at_zero"
    n_agents = 3
    n_actions = 2
    action_names = ["Give way", "Take"]
    n_states = 1
    action_roles = {"concede": 0, "claim": 1}

    def reset(self) -> list[int]:
        return [0] * self.n_agents

    def step(self, actions: list[int]) -> StepResult:
        return StepResult(observations=[0] * self.n_agents,
                          rewards=[float(a) for a in actions], done=False)

    def max_welfare_per_round(self) -> float:
        return float(self.n_agents)


class _Undeclared(_ConcedeAtZero):
    """Declares nothing at all, so nobody can say who cooperated."""

    name = "undeclared"
    action_roles = {}


class _IndexPlayer(Agent):
    """Plays one fixed index without interpreting it, which is all an undeclared game admits."""

    kind = "test"
    semantics = "index-agnostic"
    information = "none"

    def __init__(self, name: str, index: int) -> None:
        self.name, self._index = name, index

    def act(self, observation: int) -> int:
        return self._index


def test_a_game_reporting_no_count_is_counted_by_its_concede_role() -> None:
    game = _ConcedeAtZero()
    records = run_match(game, [AllC(f"C{i}") for i in range(3)], rounds=ROUNDS, seed=0)
    assert set(np.asarray(records["actions"]).flatten().tolist()) == {0}, (
        "AllC did not play the declared concede action, so this case tests nothing")
    rate = social.compute_all(records, game.max_welfare_per_round())["cooperation_rate"]
    assert abs(rate - 1.0) < 1e-9, (
        f"every player conceded in every round and the rate reads {rate:.3f}. The runner counted "
        f"an action index instead of the role the game declared")
    print("OK: a game with no count of its own is counted by the concede role it declares")


def test_a_game_declaring_neither_is_refused() -> None:
    try:
        run_match(_Undeclared(), [_IndexPlayer(f"P{i}", 1) for i in range(3)], rounds=5, seed=0)
    except ValueError as exc:
        assert "concede" in str(exc) and "cooperators" in str(exc), (
            f"refused, but the message does not say what to declare: {exc}")
        print("OK: a game that cannot say who cooperated is refused, and told how to fix it")
        return
    raise AssertionError("a game declaring neither a count nor a concede role ran, so its "
                         "cooperation rate was a reading of action indices")


def test_the_measure_refuses_records_without_a_count() -> None:
    actions = np.ones((4, 3), dtype=int)
    try:
        social.cooperation_rate(actions, None)
    except ValueError:
        print("OK: the cooperation measure refuses to read action indices as cooperation")
        return
    raise AssertionError("cooperation_rate accepted records with no count and read index 1")


def test_the_policy_panel_counts_the_concede_action() -> None:
    """The results page's "policy vs behavior" line counted `a == 1` in the greedy policy as
    cooperating, which in the congestion family is the number of states where it claims."""
    from ..webui.app import _policy_vs_behavior_html
    brain = {"greedy_policy": [0, 0, 1], "epsilon": 0.0}
    text = _policy_vs_behavior_html(brain, 0.5, concede=0)
    assert "2/3 states" in text, f"a policy conceding in 2 of 3 states was described as: {text}"
    # A tie is not a preference: argmax reports index 0 for it, and the agent plays either.
    tied = {"greedy_policy": [0, 0, 1], "q_table": [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]],
            "epsilon": 0.0}
    text = _policy_vs_behavior_html(tied, 0.5, concede=0)
    assert "1/2 states" in text and "1 tied" in text, f"a tied state was counted: {text}"
    assert _policy_vs_behavior_html(brain, 0.5) == "", (
        "with no declared concede action the panel still claimed to know what cooperating is")
    print("OK: the policy panel counts the game's concede action, and says nothing without one")


if __name__ == "__main__":
    failures = 0
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {fn_name}\n     {exc}")
            except Exception as exc:  # a changed signature is a failure, not a crash
                failures += 1
                print(f"ERROR {fn_name}\n     {type(exc).__name__}: {exc}")
    print(f"\n{'all cooperation-semantics tests passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
