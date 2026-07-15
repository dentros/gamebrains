"""
Tests for the PettingZoo Parallel-API adapter.

1. Full conformance against PettingZoo's own `parallel_api_test` (all seats externally
   controlled) — this is what a plain multi-agent RL framework would exercise.
2. A "background agents" scenario — a QLearningAgent occupies a seat we don't control and
   is confirmed to actually learn (its epsilon decays / Q-table changes) purely from being
   stepped through the adapter, with no direct calls into our own engine.runner.
"""

import random

from pettingzoo.test import parallel_api_test

from gamebrains.agents.classic import AllD, MajorityTFT
from gamebrains.agents.qlearning import QLearningAgent
from gamebrains.games.public_goods import COOPERATE, DEFECT
from gamebrains.interop.pettingzoo_adapter import GameBrainsParallelEnv
from gamebrains.games.public_goods import PublicGoodsGame


def test_parallel_api_conformance():
    game = PublicGoodsGame(n_agents=4, rounds=50)
    env = GameBrainsParallelEnv(game, roster=[None] * 4)  # fully externally controlled
    parallel_api_test(env, num_cycles=5)


def test_background_agent_learns_through_adapter():
    game = PublicGoodsGame(n_agents=3, rounds=2000)
    ql = QLearningAgent("QL", n_states=game.n_states, n_actions=game.n_actions, seed=0)
    roster = [ql, AllD("Defector"), MajorityTFT("TFT", n_agents=3)]

    env = GameBrainsParallelEnv(game, roster, controlled_agent_ids=[])  # nobody externally controlled
    initial_epsilon = ql.epsilon
    initial_q = ql.Q.copy()

    rng = random.Random(0)
    obs, _ = env.reset()
    for _ in range(500):
        if not env.agents:
            obs, _ = env.reset()
        obs, rewards, terminateds, truncateds, infos = env.step({})  # no controlled seats

    assert ql.epsilon < initial_epsilon, "background Q-learning agent did not decay epsilon"
    assert not (ql.Q == initial_q).all(), "background Q-learning agent's Q-table never updated"


if __name__ == "__main__":
    test_parallel_api_conformance()
    print("OK: parallel_api_test passed")
    test_background_agent_learns_through_adapter()
    print("OK: background agent learns purely through the adapter")
