"""
Tests for games/congestion.py, the Honey-Jar Game family.

The reward cases are checked against the source papers' own stated arithmetic rather than against
whatever the implementation happens to do: arriving alone pays in full, a partial tie pays a
reduced share whose denominator depends on the rule family, and under the main rule a unanimous
arrival pays exactly zero. The k-variant's missing zero floor is asserted explicitly, since that
asymmetry between the two published rules is easy to "tidy up" by accident.
"""

from gamebrains.engine.runner import run_match
from gamebrains.agents.classic import AllC, AllD
from gamebrains.games.congestion import (
    MOVE, PRESETS, STAY, CongestionGame, from_preset,
)


def _play(game: CongestionGame, *rounds_of_actions):
    """Run a scripted sequence of joint actions from a fresh episode."""
    game.reset()
    result = None
    for actions in rounds_of_actions:
        result = game.step(list(actions))
    return result


def _reach(game: CongestionGame, actions_last_round):
    """Drive every agent to one step short of the terminal, then apply a final joint action."""
    game.reset()
    for _ in range(game.num_positions - 2):
        game.step([MOVE] * game.n_agents)
    return game.step(list(actions_last_round))


def test_arriving_alone_pays_in_full_and_ends_the_episode():
    game = CongestionGame(n_agents=3, num_positions=3, reward_rule="ILF", full_reward=100.0)
    result = _reach(game, [MOVE, STAY, STAY])

    assert result.done
    assert result.rewards == [100.0, 0.0, 0.0]
    assert result.info["episode"]["top_agents"] == [1, 0, 0]
    assert result.info["episode"]["terminal_occurrences"] == 1
    assert result.info["episode"]["timed_out"] is False


def test_published_reward_rules_match_the_papers_arithmetic():
    # Partial tie: 2 of 3 arrive together. n-family divides by the population, k-family by the
    # number who actually collided, which is the whole difference between the two.
    for rule, expected in [("ILF", 100 / 3), ("IQF", 100 / 9),
                           ("KLF", 100 / 2), ("KQF", 100 / 4)]:
        game = CongestionGame(n_agents=3, num_positions=3, reward_rule=rule, full_reward=100.0)
        result = _reach(game, [MOVE, MOVE, STAY])
        assert result.info["episode"]["terminal_occurrences"] == 2
        assert abs(result.rewards[0] - expected) < 1e-9, rule
        assert result.rewards[0] == result.rewards[1], rule
        assert result.rewards[2] == 0.0, rule


def test_main_rule_collapses_to_zero_when_everyone_arrives_but_the_k_variant_does_not():
    """The papers' main rule keeps universal collision at exactly zero, so total non-coordination
    cannot register as partial success under outcome-based metrics. The k-variant, being literal
    Rosenthal congestion, has no such floor and simply pays r/n."""
    main = CongestionGame(n_agents=4, num_positions=3, reward_rule="ILF", full_reward=100.0)
    assert _reach(main, [MOVE] * 4).rewards == [0.0] * 4

    k_variant = from_preset("hjg_k", n_agents=4)
    k_variant.full_reward = 100.0
    everyone = _reach(k_variant, [MOVE] * 4)
    assert everyone.rewards == [25.0] * 4  # r/k with k = n, small but not zero
    assert everyone.info["episode"]["terminal_occurrences"] == 4


def test_corridor_length_is_the_one_shot_dial():
    one_shot = CongestionGame(n_agents=2, num_positions=2)
    assert one_shot.is_one_shot
    assert one_shot.describe()["timing"] == "one_shot"
    one_shot.reset()
    assert one_shot.step([MOVE, STAY]).done  # a single move decides it

    dynamic = CongestionGame(n_agents=2, num_positions=3)
    assert not dynamic.is_one_shot
    dynamic.reset()
    assert not dynamic.step([MOVE, STAY]).done  # an intermediate cell to be seen approaching from


def test_episode_times_out_when_nobody_ever_moves_and_says_so():
    game = CongestionGame(n_agents=2, num_positions=3, episode_max_rounds=4)
    result = _play(game, *[[STAY, STAY]] * 4)

    assert result.done
    episode = result.info["episode"]
    assert episode["timed_out"] is True
    assert episode["top_agents"] == [0, 0]     # a real outcome: everyone refused
    assert episode["terminal_occurrences"] == 0
    assert result.rewards == [0.0, 0.0]


def test_memory_depth_changes_the_state_space_and_survives_reset():
    plain = CongestionGame(n_agents=3, num_positions=3, memory_episodes=0)
    remembering = CongestionGame(n_agents=3, num_positions=3, memory_episodes=1)

    assert plain.state_type == "Type-A"
    assert remembering.state_type == "Type-B"
    assert plain.n_states == 3 ** 3
    assert remembering.n_states == (3 ** 3) * (2 ** 3)
    assert CongestionGame(n_agents=3, num_positions=3, memory_episodes=2).state_type == "Type-B(2)"

    # Type-A cannot tell two episodes apart at the start line; Type-B can, because it remembers
    # who arrived. That memory has to survive reset() or the whole representation is pointless.
    fresh_plain = plain.reset()[0]
    _reach(plain, [MOVE, STAY, STAY])
    assert plain.reset()[0] == fresh_plain

    fresh_remembering = remembering.reset()[0]
    _reach(remembering, [MOVE, STAY, STAY])
    assert remembering.reset()[0] != fresh_remembering
    # new_match() is the explicit way back to a blank slate.
    remembering.new_match()
    assert remembering.reset()[0] == fresh_remembering


def test_custom_rule_resolves_to_the_same_design_as_its_named_equivalent():
    """describe() reports the resolved (base, exponent), so a custom rule spelling out r/n^2 is
    recognised as the same experimental design as IQF rather than as something new."""
    named = CongestionGame(n_agents=3, reward_rule="IQF").describe()
    custom = CongestionGame(n_agents=3, reward_rule="custom",
                            reward_base="n", reward_exponent=2).describe()

    assert (named["reward_base"], named["reward_exponent"]) == ("n", 2)
    assert (custom["reward_base"], custom["reward_exponent"]) == ("n", 2)
    assert named["reward_published"] is True
    # The cubic rules are real options but must not claim to reproduce a published result.
    assert CongestionGame(n_agents=3, reward_rule="ICF").describe()["reward_published"] is False


def test_impossible_configurations_are_refused_with_a_reason():
    for kwargs, needle in [
        ({"n_agents": 1}, "at least 2 agents"),
        ({"n_agents": 3, "num_positions": 1}, "num_positions"),
        ({"n_agents": 3, "reward_rule": "nonsense"}, "unknown reward_rule"),
        ({"n_agents": 3, "reward_rule": "custom"}, "reward_base"),
        ({"n_agents": 3, "num_positions": 4, "episode_max_rounds": 2}, "below the 3 moves"),
        ({"n_agents": 12, "memory_episodes": 2}, "over the"),   # state-space explosion guard
    ]:
        try:
            CongestionGame(**kwargs)
        except ValueError as exc:
            assert needle in str(exc), f"{kwargs} -> {exc}"
        else:
            raise AssertionError(f"{kwargs} should have been refused")


def test_runs_end_to_end_as_a_real_episodic_match():
    """The point of the episodic runner change: many contests inside one round budget, each one
    logged with the arrival vector the alternation metrics need."""
    game = from_preset("hjg", n_agents=3)
    # AllD claims and AllC concedes, in this game as in every other: the roles are resolved from
    # the game rather than from a hardcoded index, so the two claimers are the ones who move.
    roster = [AllD("Claimer 0"), AllD("Claimer 1"), AllC("Conceder 2")]
    out = run_match(game, roster, rounds=30, seed=0)

    assert len(out["episodes"]) > 1
    for episode in out["episodes"]:
        assert len(episode["top_agents"]) == 3
        assert episode["terminal_occurrences"] == sum(episode["top_agents"])

    # The two claimers collide every single episode and the conceder never arrives: the
    # pathological no-alternation pattern the ALT metrics exist to expose. Under ILF at n=3 a
    # 2-way tie pays r/n each, never the full reward.
    assert all(e["top_agents"] == [1, 1, 0] for e in out["episodes"])
    assert all(e["terminal_occurrences"] == 2 for e in out["episodes"])
    assert all(abs(r - 100.0 / 3) < 1e-9 for r in out["rewards"][:, :2][out["rewards"][:, :2] > 0])
    assert out["rewards"][:, 2].sum() == 0.0     # conceding pays nothing at all here


def test_every_preset_builds_and_declares_whether_it_is_published():
    for key in PRESETS:
        game = from_preset(key, n_agents=3)
        described = game.describe()
        assert described["family"] == "congestion"
        assert described["n_states"] == game.n_states
        assert isinstance(described["reward_published"], bool)
    assert from_preset("market_entry", n_agents=3).is_one_shot
    assert not from_preset("hjg", n_agents=3).is_one_shot


if __name__ == "__main__":
    test_arriving_alone_pays_in_full_and_ends_the_episode()
    print("OK: arriving alone pays in full and ends the episode")
    test_published_reward_rules_match_the_papers_arithmetic()
    print("OK: the four published reward rules match the papers' arithmetic")
    test_main_rule_collapses_to_zero_when_everyone_arrives_but_the_k_variant_does_not()
    print("OK: main rule collapses to zero at full arrival, the k-variant deliberately does not")
    test_corridor_length_is_the_one_shot_dial()
    print("OK: corridor length is the one-shot / dynamic dial")
    test_episode_times_out_when_nobody_ever_moves_and_says_so()
    print("OK: an episode times out when nobody moves, and reports that it did")
    test_memory_depth_changes_the_state_space_and_survives_reset()
    print("OK: memory depth changes the state space and survives reset")
    test_custom_rule_resolves_to_the_same_design_as_its_named_equivalent()
    print("OK: a custom rule resolves to the same design as its named equivalent")
    test_impossible_configurations_are_refused_with_a_reason()
    print("OK: impossible configurations are refused with a reason")
    test_runs_end_to_end_as_a_real_episodic_match()
    print("OK: runs end to end as a real episodic match")
    test_every_preset_builds_and_declares_whether_it_is_published()
    print("OK: every preset builds and declares whether it is published")
