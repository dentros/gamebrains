"""
Tests for the game-declared semantics mechanism (engine/game.py): action roles and observation
kinds, and the classic strategies that resolve themselves through them.

The regression this locks down is a real shipped bug. `agents/classic.py` used to import
COOPERATE from the Public Goods Game, pinning "cooperate" to action 1. Action 1 in the congestion
game is MOVE, grabbing the contested resource, so AllC played the most aggressive strategy
available there while still being labelled cooperative. The first assertion below is exactly that
case: the same strategy, two games, opposite action indices, one consistent meaning.
"""

from gamebrains.agents.classic import AllC, AllD, MajorityTFT, RandomAgent
from gamebrains.engine.game import Game, StepResult
from gamebrains.games.congestion import MOVE, STAY, CongestionGame
from gamebrains.games.public_goods import COOPERATE, DEFECT, PublicGoodsGame


class _RolelessGame(Game):
    """A game that declares nothing, to prove agents refuse rather than guess."""

    name = "roleless"
    n_agents = 2
    n_actions = 2
    action_names = ["A", "B"]
    n_states = 2

    def reset(self) -> list[int]:
        return [0, 0]

    def step(self, actions: list[int]) -> StepResult:
        return StepResult(observations=[0, 0], rewards=[0.0, 0.0], done=False)


def test_the_same_strategy_maps_to_opposite_indices_in_the_two_games():
    pgg = PublicGoodsGame(n_agents=3, rounds=10)
    hjg = CongestionGame(n_agents=3)

    conceder, claimer = AllC("c"), AllD("d")

    conceder.on_match_start(pgg)
    claimer.on_match_start(pgg)
    assert conceder.act(0) == COOPERATE == 1
    assert claimer.act(0) == DEFECT == 0

    conceder.on_match_start(hjg)
    claimer.on_match_start(hjg)
    # The indices flip, because holding back is the conceding act when a resource jams.
    assert conceder.act(0) == STAY == 0
    assert claimer.act(0) == MOVE == 1


def test_an_agent_refuses_a_game_that_declares_no_roles():
    agent = AllC("c")
    try:
        agent.on_match_start(_RolelessGame())
    except ValueError as exc:
        assert "concede" in str(exc) and "does not declare" in str(exc)
    else:
        raise AssertionError("an undeclared role must be refused, not guessed at")


def test_an_unbound_agent_says_so_instead_of_playing_something_arbitrary():
    for agent in (AllC("c"), AllD("d"), RandomAgent("r"), MajorityTFT("t", n_agents=3)):
        try:
            agent.act(0)
        except RuntimeError as exc:
            assert "has not been matched to a game" in str(exc)
        else:
            raise AssertionError(f"{agent.strategy} played without knowing the game")


def test_reciprocating_strategies_also_check_they_can_read_the_observation():
    """Majority-TFT counts what others did, so a role alone is not enough: it needs an
    observation that is actually a count. The congestion board index is not one."""
    tft = MajorityTFT("t", n_agents=3)
    tft.on_match_start(PublicGoodsGame(n_agents=3, rounds=10))   # concede_count, fine

    try:
        tft.on_match_start(CongestionGame(n_agents=3))            # board_index, not readable
    except ValueError as exc:
        assert "board_index" in str(exc) and "concede_count" in str(exc)
    else:
        raise AssertionError("a counting strategy must refuse a positional observation")


def test_majority_tft_reciprocates_in_the_declared_roles():
    pgg = PublicGoodsGame(n_agents=4, rounds=10)
    tft = MajorityTFT("t", n_agents=4)
    tft.on_match_start(pgg)

    assert tft.act(tft.start_state) == COOPERATE   # opens by conceding
    assert tft.act(3) == COOPERATE                 # 3 of 4 conceded last round, >= 2
    assert tft.act(1) == DEFECT                    # only 1 did, below the threshold


def test_random_agent_mixes_between_the_two_declared_roles():
    hjg = CongestionGame(n_agents=2)
    always_concede = RandomAgent("r", p_cooperate=1.0, seed=0)
    always_claim = RandomAgent("r", p_cooperate=0.0, seed=0)
    always_concede.on_match_start(hjg)
    always_claim.on_match_start(hjg)

    assert [always_concede.act(0) for _ in range(5)] == [STAY] * 5
    assert [always_claim.act(0) for _ in range(5)] == [MOVE] * 5


def test_declared_semantics_are_part_of_what_a_run_records():
    """A game that assigned its roles to different actions would make every role-aware agent in
    the roster behave differently, so the declaration belongs in the recorded design."""
    for game in (PublicGoodsGame(n_agents=3, rounds=10), CongestionGame(n_agents=3)):
        described = game.describe()
        assert described["action_roles"] == game.action_roles
        assert described["observation_kind"] == game.observation_kind

    assert PublicGoodsGame(n_agents=3, rounds=10).describe()["action_roles"]["concede"] == 1
    assert CongestionGame(n_agents=3).describe()["action_roles"]["concede"] == 0


if __name__ == "__main__":
    test_the_same_strategy_maps_to_opposite_indices_in_the_two_games()
    print("OK: one strategy, two games, opposite indices, one meaning")
    test_an_agent_refuses_a_game_that_declares_no_roles()
    print("OK: an agent refuses a game that declares no roles")
    test_an_unbound_agent_says_so_instead_of_playing_something_arbitrary()
    print("OK: an unbound agent says so instead of playing something arbitrary")
    test_reciprocating_strategies_also_check_they_can_read_the_observation()
    print("OK: reciprocating strategies also check they can read the observation")
    test_majority_tft_reciprocates_in_the_declared_roles()
    print("OK: Majority-TFT reciprocates in the declared roles")
    test_random_agent_mixes_between_the_two_declared_roles()
    print("OK: the random agent mixes between the two declared roles")
    test_declared_semantics_are_part_of_what_a_run_records()
    print("OK: declared semantics are part of what a run records")
