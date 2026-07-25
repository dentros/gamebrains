"""
Tests for episodic support in engine/runner.py: a game that ends episodes early gets restarted
until the round budget runs out, each completed episode emits an `episode` event carrying
whatever the game put in `info["episode"]`, and single-stage games keep behaving exactly as
before. That last part is the point of the regression test at the bottom -- the Public Goods
Game reports `done` on its final round, so it is the case most at risk from this change.
"""

import numpy as np

from gamebrains.agents.classic import AllC
from gamebrains.engine.eventlog import EventLog
from gamebrains.engine.game import Game, StepResult
from gamebrains.engine.runner import run_match
from gamebrains.games.public_goods import PublicGoodsGame


class _RotatingRace(Game):
    """Toy episodic game. Every episode lasts exactly `ep_len` rounds; the agent whose index
    matches the episode counter 'reaches the terminal', so the winner rotates predictably and
    the forwarded payload is trivial to assert on."""

    def __init__(self, n_agents: int = 2, ep_len: int = 3) -> None:
        self.n_agents = n_agents
        self.n_actions = 2
        self.action_names = ["Stay", "Move"]
        self.n_states = 2
        self.name = "rotating_race"
        self.ep_len = ep_len
        self._round = 0
        self._episode = 0

    def reset(self) -> list[int]:
        self._round = 0
        return [0] * self.n_agents

    def step(self, actions: list[int]) -> StepResult:
        self._round += 1
        done = self._round >= self.ep_len
        info: dict = {}
        if done:
            winner = self._episode % self.n_agents
            top = [1 if i == winner else 0 for i in range(self.n_agents)]
            info["episode"] = {"top_agents": top, "terminal_occurrences": sum(top)}
            self._episode += 1
        return StepResult(observations=[1] * self.n_agents,
                          rewards=[1.0] * self.n_agents, done=done, info=info)


def _roster(n: int) -> list:
    return [AllC(f"AllC {i}") for i in range(n)]


def _counting_reset(game):
    """Wrap game.reset with a call counter, so tests can assert on restarts without reaching
    into any game's private round state."""
    calls = {"n": 0}
    original = game.reset

    def counted():
        calls["n"] += 1
        return original()

    game.reset = counted
    return calls


def test_episodic_game_restarts_until_the_round_budget_is_spent():
    game = _RotatingRace(n_agents=2, ep_len=3)
    resets = _counting_reset(game)
    log = EventLog()

    out = run_match(game, _roster(2), rounds=12, seed=0, eventlog=log)

    # 12 rounds of 3 = 4 complete episodes.
    assert len(out["episodes"]) == 4
    assert [e["episode"] for e in out["episodes"]] == [0, 1, 2, 3]
    assert all(e["rounds"] == 3 for e in out["episodes"])

    # One reset to open the match, then one per episode boundary -- except the last, which lands
    # on the final round and must not rewind the game behind the caller's back.
    assert resets["n"] == 4

    events = [e for e in log.events if e["type"] == "episode"]
    assert len(events) == 4


def test_episode_event_carries_the_games_own_payload_and_reward_totals():
    game = _RotatingRace(n_agents=2, ep_len=3)
    log = EventLog()

    out = run_match(game, _roster(2), rounds=6, seed=0, eventlog=log)

    # The game hands the runner its winner vector; the runner forwards it untouched.
    assert out["episodes"][0]["top_agents"] == [1, 0]
    assert out["episodes"][1]["top_agents"] == [0, 1]
    assert all(e["terminal_occurrences"] == 1 for e in out["episodes"])

    # Reward totals are the runner's own contribution: 3 rounds x 1.0 per agent.
    assert out["episodes"][0]["rewards"] == [3.0, 3.0]

    logged = [e for e in log.events if e["type"] == "episode"]
    assert logged[0]["top_agents"] == [1, 0]
    assert logged[0]["rounds"] == 3


def test_trailing_partial_episode_follows_the_chosen_policy():
    # 10 rounds of 3-round episodes = 3 complete contests plus one round of a 4th that the round
    # budget cut short. Whether that half-played contest counts is the caller's call.
    dropped = run_match(_RotatingRace(2, 3), _roster(2), rounds=10, seed=0)
    assert len(dropped["episodes"]) == 3
    assert sum(e["rounds"] for e in dropped["episodes"]) == 9

    kept = run_match(_RotatingRace(2, 3), _roster(2), rounds=10, seed=0,
                     partial_episode="record")
    assert len(kept["episodes"]) == 4
    assert kept["episodes"][3]["rounds"] == 1
    assert kept["episodes"][3]["truncated"] is True
    # The completed ones are never marked, so a consumer can always tell them apart.
    assert all("truncated" not in e for e in kept["episodes"][:3])

    # Either way the truncation is visible rather than silent.
    for out in (dropped, kept):
        assert out["partial_episode_rounds"] == 1
    assert dropped["partial_episode_policy"] == "drop"
    assert kept["partial_episode_policy"] == "record"


def test_unknown_partial_episode_policy_is_rejected():
    try:
        run_match(_RotatingRace(2, 3), _roster(2), rounds=6, seed=0, partial_episode="maybe")
    except ValueError as exc:
        assert "partial_episode" in str(exc)
    else:
        raise AssertionError("an unknown partial_episode policy should not be accepted silently")


def test_single_stage_game_is_unaffected():
    def play():
        game = PublicGoodsGame(n_agents=3, rounds=20, mpcr=0.5)
        resets = _counting_reset(game)
        log = EventLog()
        out = run_match(game, _roster(3), rounds=20, seed=7, eventlog=log)
        return game, resets, log, out

    game_a, resets_a, log_a, out_a = play()
    _, _, log_b, out_b = play()

    # Reproducibility: same seed and config still give identical play.
    assert np.array_equal(out_a["actions"], out_b["actions"])
    assert np.allclose(out_a["rewards"], out_b["rewards"])
    assert np.array_equal(out_a["cooperators"], out_b["cooperators"])

    # PGG reports done on its final round, so the whole match is exactly one episode, and the
    # game must NOT have been rewound afterwards (only the opening reset ran).
    assert len(out_a["episodes"]) == 1
    assert out_a["episodes"][0]["rounds"] == 20
    assert resets_a["n"] == 1

    # The round events themselves are untouched by any of this.
    rounds_a = [e for e in log_a.events if e["type"] == "round"]
    rounds_b = [e for e in log_b.events if e["type"] == "round"]
    assert len(rounds_a) == 20
    assert rounds_a == rounds_b

    # Episode reward totals agree with the per-round arrays they were accumulated from.
    assert np.allclose(out_a["episodes"][0]["rewards"], out_a["rewards"].sum(axis=0))


if __name__ == "__main__":
    test_episodic_game_restarts_until_the_round_budget_is_spent()
    print("OK: an episodic game restarts until the round budget is spent")
    test_episode_event_carries_the_games_own_payload_and_reward_totals()
    print("OK: episode events carry the game's own payload plus runner reward totals")
    test_trailing_partial_episode_follows_the_chosen_policy()
    print("OK: a trailing partial episode follows the chosen drop/record policy")
    test_unknown_partial_episode_policy_is_rejected()
    print("OK: an unknown partial-episode policy is rejected")
    test_single_stage_game_is_unaffected()
    print("OK: the single-stage Public Goods Game is unaffected")
