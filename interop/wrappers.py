"""
Compatibility wrappers for downstream consumers whose requirements go past the standard API.

Both adapters in this package expose `Discrete` observation spaces, which is correct: it is what
the observation actually is (an integer state index), and it is what PettingZoo, Gymnasium and
Stable-Baselines3 all accept without comment. One consumer needs more, so the widening lives here
rather than in the adapters, and only callers who need it pay for it.

Empirically established, not assumed (tests/test_interop_*.py):
  - Stable-Baselines3 needs nothing. `check_env` passes, DQN and PPO train, `DummyVecEnv` wraps.
  - RLlib's current API stack refuses to build a model on a `Discrete` observation space at all:
    "No default encoder config for obs space=Discrete(n)". `OneHotObs` fixes it. SB3 performs the
    same one-hot preprocessing internally and never mentions it, which is exactly why this cost is
    invisible until a second consumer is tried.
  - MLPro's Gymnasium bridge needs the env *registered* with Gymnasium, not merely conformant,
    because it reads `env.env.spec.id` and later calls `gymnasium.make` on it. See
    `register_gym_env` below. Its PettingZoo bridge cannot accept any third-party environment
    (it resolves the class by name inside five hardcoded `pettingzoo.*` submodules), so there is
    nothing to wrap for that route.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium
import numpy as np
from pettingzoo.utils.env import ParallelEnv


class OneHotObs(ParallelEnv):
    """One-hot each agent's `Discrete` observation into a `Box`, leaving actions untouched.

    Wraps at the PettingZoo level so `ray.rllib...ParallelPettingZooEnv` can consume the result:

        from gamebrains.interop.wrappers import OneHotObs
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        env = ParallelPettingZooEnv(OneHotObs(GameBrainsParallelEnv(game, roster)))

    Background seats are untouched, so GameBrains agents behind the wrapper keep acting and
    learning on the raw integer observation they expect.
    """

    metadata = {"render_modes": [], "name": "gamebrains_onehot_v0"}

    def __init__(self, env: ParallelEnv) -> None:
        self.env = env
        self.possible_agents = list(env.possible_agents)
        self.agents = list(env.agents)

        self._n: dict[str, int] = {}
        for aid in self.possible_agents:
            space = env.observation_space(aid)
            if not isinstance(space, gymnasium.spaces.Discrete):
                raise TypeError(
                    f"OneHotObs only widens Discrete observations; seat {aid} has {space}. "
                    "A non-discrete observation needs no widening for RLlib."
                )
            self._n[aid] = int(space.n)

        self.observation_spaces = {
            aid: gymnasium.spaces.Box(0.0, 1.0, shape=(self._n[aid],), dtype=np.float32)
            for aid in self.possible_agents
        }
        self.action_spaces = {aid: env.action_space(aid) for aid in self.possible_agents}

    def observation_space(self, agent: str) -> gymnasium.spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.spaces.Space:
        return self.action_spaces[agent]

    def _encode(self, obs: dict[str, int]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for aid, value in obs.items():
            vec = np.zeros(self._n[aid], dtype=np.float32)
            vec[int(value)] = 1.0
            out[aid] = vec
        return out

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.agents = list(self.env.agents)
        return self._encode(obs), infos

    def step(self, actions: dict[str, int]):
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
        self.agents = list(self.env.agents)
        return self._encode(obs), rewards, terminateds, truncateds, infos

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


def register_gym_env(env_id: str, factory: Callable[[], Any],
                     max_episode_steps: Optional[int] = None) -> str:
    """Register a GameBrainsGymEnv factory with Gymnasium and return the id.

    Needed for MLPro's `WrEnvGYM2MLPro`, which identifies an environment by
    `env.env.spec.id` and rebuilds it later via `gymnasium.make(id)`. A bare `gymnasium.Env`
    subclass has no `.spec`, so conformance alone is not enough for that bridge:

        env_id = register_gym_env("GameBrainsPGG-v0", make_env, max_episode_steps=200)
        mlpro_env = WrEnvGYM2MLPro(gymnasium.make(env_id))

    Re-registering the same id is tolerated so repeated calls in one session are safe.
    """
    if env_id not in gymnasium.registry:
        gymnasium.register(id=env_id, entry_point=lambda **kwargs: factory(),
                           max_episode_steps=max_episode_steps)
    return env_id
