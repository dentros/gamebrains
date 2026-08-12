"""
Tests for the DQN agent, with reproducibility first.

This module exists because there was none, and its absence had a cost. The platform claims that a
given seed and configuration reproduce a byte-identical event log, and the repository's duplicate
detection depends on that claim: one configuration hash is supposed to imply one result. No test
exercised the DQN agent, and the DQN agent drew its replay batches from the `random` module's global
functions, which are seeded by the interpreter rather than by us.

The claim was therefore false for any roster containing a DQN, and nothing said so. Measured before
the fix, with exploration decayed so that actions depend on the learned network: 119 of 600 rounds
differed between two identically-seeded runs, and 45 of 600 changed in response to the global seed
alone. `test_two_identical_runs_produce_identical_matches` and
`test_match_is_immune_to_the_global_random_seed` are the regression guards.

The nearly-greedy configuration used here is deliberate. At the default exploration schedule most
actions come from the agent's own seeded generator, which masks any nondeterminism in the learning
path: a weak test at those settings reported the matches as identical when they were not.
"""

import random

import numpy as np

from gamebrains.agents.dqn import DQNAgent
from gamebrains.engine.runner import run_match
from gamebrains.games.public_goods import PublicGoodsGame

N_AGENTS = 3
ROUNDS = 400


def _match(agent_seed_base: int = 0, run_seed: int = 0, rounds: int = ROUNDS) -> dict:
    """A nearly-greedy match, so that outcomes depend on the network and not on exploration noise."""
    game = PublicGoodsGame(n_agents=N_AGENTS, rounds=rounds, mpcr=0.5)
    roster = [DQNAgent(f"D{i}", n_states=game.n_states, n_actions=game.n_actions,
                       seed=agent_seed_base + i, epsilon=0.05, epsilon_min=0.0,
                       epsilon_decay=0.99)
              for i in range(N_AGENTS)]
    return run_match(game, roster, rounds=rounds, seed=run_seed)


def test_two_identical_runs_produce_identical_matches():
    a, b = _match(), _match()
    assert np.array_equal(np.asarray(a["actions"]), np.asarray(b["actions"])), (
        "two identically-seeded DQN matches diverged in their actions, so the platform's "
        "byte-identical event-log property does not hold for rosters containing a DQN")
    assert np.array_equal(np.asarray(a["rewards"]), np.asarray(b["rewards"]))


def test_match_is_immune_to_the_global_random_seed():
    """Nothing in a match may depend on interpreter-level RNG state we do not control."""
    random.seed(1)
    x = np.asarray(_match()["actions"])
    random.seed(999_999)
    y = np.asarray(_match()["actions"])
    assert np.array_equal(x, y), (
        "changing the global random seed changed the match, so some sampling path is still "
        "drawing from the global generator instead of a per-agent one")


def test_different_seeds_still_produce_different_behaviour():
    """The guard above must not have been satisfied by freezing the agent into determinism."""
    x = np.asarray(_match(agent_seed_base=10, run_seed=1, rounds=300)["actions"])
    y = np.asarray(_match(agent_seed_base=70, run_seed=2, rounds=300)["actions"])
    assert not np.array_equal(x, y), "different seeds produced identical matches, so seeding is dead"


def test_actions_stay_in_range_and_the_network_trains():
    game = PublicGoodsGame(n_agents=N_AGENTS, rounds=ROUNDS, mpcr=0.5)
    roster = [DQNAgent(f"D{i}", n_states=game.n_states, n_actions=game.n_actions, seed=i)
              for i in range(N_AGENTS)]
    records = run_match(game, roster, rounds=ROUNDS, seed=0)

    actions = np.asarray(records["actions"])
    assert set(np.unique(actions)).issubset({0, 1}), f"out-of-range actions: {np.unique(actions)}"
    for agent in roster:
        assert agent.steps >= ROUNDS, f"{agent.name} took fewer steps than rounds played"
        assert len(agent.buffer) > 0, f"{agent.name} never stored a transition"
        assert agent.epsilon < 1.0, f"{agent.name} never decayed its exploration rate"


def test_render_and_inspect_are_serializable_shapes():
    game = PublicGoodsGame(n_agents=N_AGENTS, rounds=50, mpcr=0.5)
    agent = DQNAgent("D", n_states=game.n_states, n_actions=game.n_actions, seed=0)
    run_match(game, [agent] + [DQNAgent(f"D{i}", n_states=game.n_states,
                                        n_actions=game.n_actions, seed=i) for i in (1, 2)],
              rounds=50, seed=0)
    for payload in (agent.inspect(), agent.render_brain()):
        assert isinstance(payload, dict) and payload, "empty transparency payload"


if __name__ == "__main__":
    test_two_identical_runs_produce_identical_matches()
    print("OK: two identically-seeded DQN matches are identical")
    test_match_is_immune_to_the_global_random_seed()
    print("OK: a DQN match is immune to the global random seed")
    test_different_seeds_still_produce_different_behaviour()
    print("OK: different seeds still produce different behaviour")
    test_actions_stay_in_range_and_the_network_trains()
    print("OK: actions stay in range and the network trains")
    test_render_and_inspect_are_serializable_shapes()
    print("OK: inspect and render_brain return usable payloads")
