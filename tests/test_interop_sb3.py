"""
Stable-Baselines3 consumes GameBrains through `interop/gym_adapter.py`.

SB3 is an optional dependency, so this module SKIPS rather than fails when it is absent
(`pip install stable-baselines3`). Everything else in tests/ runs without it.

Three levels of claim, tested separately because they are not the same statement:
  1. SB3's own `check_env` accepts the adapter          -> interface conformance
  2. DQN and PPO train and predict against it           -> the integration actually runs
  3. Background GameBrains agents learn during that     -> an external algorithm was trained
                                                           against our own cognitive brains
Only (3) is the claim the adapters exist to support.
"""

import numpy as np

from ..agents.qlearning import QLearningAgent
from ..games.public_goods import PublicGoodsGame
from ..interop.gym_adapter import GameBrainsGymEnv

N_AGENTS = 3
ROUNDS = 200


def _make_env(seed: int = 0) -> GameBrainsGymEnv:
    """Seat 0 for SB3; seats 1-2 stay real Q-learning brains that keep learning."""
    game = PublicGoodsGame(n_agents=N_AGENTS, rounds=ROUNDS, mpcr=0.5)
    roster = [None] + [
        QLearningAgent(f"Q{i}", n_states=game.n_states, n_actions=game.n_actions, seed=seed + i)
        for i in range(1, N_AGENTS)
    ]
    return GameBrainsGymEnv(game, roster, controlled_agent_id=0)


def test_sb3_check_env_accepts_the_adapter() -> None:
    from stable_baselines3.common.env_checker import check_env

    check_env(_make_env(), warn=True, skip_render_check=True)


def test_sb3_dqn_and_ppo_train() -> None:
    from stable_baselines3 import DQN, PPO

    dqn = DQN("MlpPolicy", _make_env(), learning_starts=50, buffer_size=1000,
              batch_size=32, verbose=0, seed=0)
    dqn.learn(total_timesteps=400)

    ppo = PPO("MlpPolicy", _make_env(), n_steps=64, batch_size=32, verbose=0, seed=0)
    ppo.learn(total_timesteps=256)

    for name, model in (("DQN", dqn), ("PPO", ppo)):
        env = _make_env()
        obs, _ = env.reset(seed=0)
        action = int(model.predict(obs, deterministic=True)[0])
        assert action in (0, 1), f"{name} predicted out-of-range action {action}"


def test_sb3_vecenv_wraps_parallel_instances() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv([lambda: _make_env(0), lambda: _make_env(10)])
    obs = venv.reset()
    assert obs.shape[0] == 2, f"expected 2 stacked envs, got shape {obs.shape}"
    _, rewards, _, _ = venv.step(np.array([0, 1]))
    assert len(rewards) == 2


def test_background_brains_learn_while_sb3_trains() -> None:
    """The research-relevant claim: our agents adapt inside somebody else's training loop."""
    from stable_baselines3 import DQN

    env = _make_env()
    background = [env._env.roster[i] for i in range(1, N_AGENTS)]
    updates_before = [a.updates for a in background]
    epsilon_before = [a.epsilon for a in background]

    DQN("MlpPolicy", env, learning_starts=50, buffer_size=1000, batch_size=32,
        verbose=0, seed=0).learn(total_timesteps=600)

    for i, agent in enumerate(background):
        assert agent.updates > updates_before[i], f"{agent.name} never updated its Q-table"
        assert agent.epsilon < epsilon_before[i], f"{agent.name} epsilon did not decay"
        assert np.abs(agent.Q).sum() > 0, f"{agent.name} Q-table is still all zeros"


if __name__ == "__main__":
    try:
        import stable_baselines3  # noqa: F401
    except ImportError:
        print("SKIP: stable-baselines3 not installed (pip install stable-baselines3)")
        raise SystemExit(0)

    test_sb3_check_env_accepts_the_adapter()
    print("OK: SB3 check_env accepts the Gymnasium adapter")
    test_sb3_dqn_and_ppo_train()
    print("OK: SB3 DQN and PPO train and predict against it")
    test_sb3_vecenv_wraps_parallel_instances()
    print("OK: SB3 DummyVecEnv wraps parallel instances")
    test_background_brains_learn_while_sb3_trains()
    print("OK: GameBrains brains learn while SB3 trains against them")
