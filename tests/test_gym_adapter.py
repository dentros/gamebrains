"""
Tests for the Gymnasium single-agent adapter.

1. Full conformance against Gymnasium's own `check_env` — what Stable-Baselines3 runs before
   accepting a custom environment.
2. A "background agents" scenario: the controlled seat plays randomly while a QLearningAgent
   occupies another seat and is confirmed to learn purely from being stepped through the adapter.
"""

import numpy as np
from gymnasium.utils.env_checker import check_env

from gamebrains.agents.classic import MajorityTFT
from gamebrains.agents.qlearning import QLearningAgent
from gamebrains.games.public_goods import PublicGoodsGame
from gamebrains.interop.gym_adapter import GameBrainsGymEnv


def test_gymnasium_check_env():
    game = PublicGoodsGame(n_agents=3, rounds=50)
    ql = QLearningAgent("QL", n_states=game.n_states, n_actions=game.n_actions, seed=0)
    tft = MajorityTFT("TFT", n_agents=3)
    env = GameBrainsGymEnv(game, roster=[None, ql, tft], controlled_agent_id=0)
    check_env(env, skip_render_check=True)


def test_background_agent_learns_through_gym_adapter():
    game = PublicGoodsGame(n_agents=3, rounds=2000)
    ql = QLearningAgent("QL", n_states=game.n_states, n_actions=game.n_actions, seed=0)
    tft = MajorityTFT("TFT", n_agents=3)
    env = GameBrainsGymEnv(game, roster=[None, ql, tft], controlled_agent_id=0)

    initial_epsilon = ql.epsilon
    initial_q = ql.Q.copy()

    rng = np.random.default_rng(0)
    obs, _ = env.reset()
    for _ in range(500):
        action = int(rng.integers(env.action_space.n))
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    assert ql.epsilon < initial_epsilon, "background Q-learning agent did not decay epsilon"
    assert not (ql.Q == initial_q).all(), "background Q-learning agent's Q-table never updated"


if __name__ == "__main__":
    test_gymnasium_check_env()
    print("OK: gymnasium check_env passed")
    test_background_agent_learns_through_gym_adapter()
    print("OK: background agent learns purely through the gym adapter")
