"""
RLlib consumes GameBrains through `interop/pettingzoo_adapter.py`, plus one wrapper.

Ray is an optional dependency, so this module SKIPS when absent (`pip install "ray[rllib]"`).

The finding this module pins down, and the reason `interop/wrappers.py` exists: RLlib's wrapper
accepts the adapter and RLlib's own multi-agent pre-check passes, but building an algorithm on the
raw adapter fails with "No default encoder config for obs space=Discrete(n)". RLlib's current API
stack has no default encoder for a discrete observation space. `OneHotObs` resolves it and PPO then
trains normally.

`test_raw_discrete_obs_is_refused` asserts that failure deliberately. If a future Ray release
accepts `Discrete` observations it will fail, which is the signal to simplify the documented setup
and the interoperability section of the paper.
"""

import numpy as np

from ..agents.qlearning import QLearningAgent
from ..games.public_goods import PublicGoodsGame
from ..interop.pettingzoo_adapter import GameBrainsParallelEnv
from ..interop.wrappers import OneHotObs

N_AGENTS = 3
ROUNDS = 200


def _make_parallel(controlled=(0, 1, 2), seed: int = 0) -> GameBrainsParallelEnv:
    game = PublicGoodsGame(n_agents=N_AGENTS, rounds=ROUNDS, mpcr=0.5)
    roster = [
        None if i in controlled else
        QLearningAgent(f"Q{i}", n_states=game.n_states, n_actions=game.n_actions, seed=seed + i)
        for i in range(N_AGENTS)
    ]
    return GameBrainsParallelEnv(game, roster, controlled_agent_ids=list(controlled))


def _ppo_config(env_name: str):
    from ray.rllib.algorithms.ppo import PPOConfig

    return (
        PPOConfig()
        .environment(env_name)
        .framework("torch")
        .env_runners(num_env_runners=0, rollout_fragment_length=64)
        .training(train_batch_size=128, minibatch_size=32, num_epochs=1)
        .multi_agent(policies={"shared"}, policy_mapping_fn=lambda aid, *a, **kw: "shared")
        .debugging(log_level="ERROR")
    )


def test_rllib_wrapper_accepts_and_steps_the_adapter() -> None:
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    env = ParallelPettingZooEnv(_make_parallel())
    obs, _ = env.reset(seed=0)
    assert set(obs) == {f"agent_{i}" for i in range(N_AGENTS)}, f"unexpected ids {sorted(obs)}"
    obs, rewards, _, _, _ = env.step({aid: 0 for aid in obs})
    assert set(rewards) == set(obs)


def test_rllib_multiagent_precheck_passes() -> None:
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.rllib.utils.pre_checks.env import check_multiagent_environments

    check_multiagent_environments(ParallelPettingZooEnv(_make_parallel()))


def test_raw_discrete_obs_is_refused() -> None:
    """Documents why OneHotObs exists. Delete this test only when Ray stops needing the wrapper."""
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.tune.registry import register_env

    register_env("gamebrains_raw", lambda cfg: ParallelPettingZooEnv(_make_parallel()))
    try:
        _ppo_config("gamebrains_raw").build_algo().stop()
    except Exception as exc:
        assert "encoder" in str(exc).lower() or "Discrete" in str(exc), (
            f"expected RLlib's missing-encoder refusal, got a different failure: {exc}"
        )
        return
    raise AssertionError(
        "RLlib accepted a raw Discrete observation space. OneHotObs may no longer be needed; "
        "re-check interop/wrappers.py and the paper's interoperability section."
    )


def test_rllib_ppo_trains_through_onehot_wrapper() -> None:
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.tune.registry import register_env

    register_env("gamebrains_onehot",
                 lambda cfg: ParallelPettingZooEnv(OneHotObs(_make_parallel())))
    algo = _ppo_config("gamebrains_onehot").build_algo()
    try:
        result = algo.train()
        sampled = (result.get("env_runners", {}) or {}).get("num_env_steps_sampled", 0)
        assert sampled > 0, f"PPO reported no sampled steps: {sampled}"
    finally:
        algo.stop()


def test_rllib_drives_a_subset_while_our_brain_learns() -> None:
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    inner = _make_parallel(controlled=(0, 1), seed=100)
    env = ParallelPettingZooEnv(inner)
    background = inner.roster[2]
    assert background is not None, "seat 2 should hold a real GameBrains agent"

    updates_before, epsilon_before = background.updates, background.epsilon
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(150):
        _, _, terminated, truncated, _ = env.step(
            {aid: int(rng.integers(2)) for aid in ("agent_0", "agent_1")})
        if terminated.get("__all__") or truncated.get("__all__"):
            env.reset()

    assert background.updates > updates_before, "background brain never learned"
    assert background.epsilon < epsilon_before, "background brain epsilon did not decay"
    assert np.abs(background.Q).sum() > 0, "background Q-table still all zeros"


if __name__ == "__main__":
    try:
        import ray  # noqa: F401
        import ray.rllib  # noqa: F401
    except ImportError:
        print('SKIP: ray[rllib] not installed (pip install "ray[rllib]")')
        raise SystemExit(0)

    test_rllib_wrapper_accepts_and_steps_the_adapter()
    print("OK: RLlib ParallelPettingZooEnv wraps and steps the adapter")
    test_rllib_multiagent_precheck_passes()
    print("OK: RLlib check_multiagent_environments passes")
    test_raw_discrete_obs_is_refused()
    print("OK: raw Discrete obs is refused, as documented (this is why OneHotObs exists)")
    test_rllib_ppo_trains_through_onehot_wrapper()
    print("OK: RLlib PPO trains through the one-hot wrapper")
    test_rllib_drives_a_subset_while_our_brain_learns()
    print("OK: RLlib drives 2 seats while seat 3 stays a learning GameBrains brain")
