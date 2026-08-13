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
    `register_gym_env` below.
  - MLPro's PettingZoo bridge needs two things the standard never mentions, and we previously
    recorded it as closed to third-party environments outright. That was wrong, and the correction
    is instructive: it came from reading the code rather than running it. See `to_mlpro_pettingzoo`.
"""

from __future__ import annotations

import sys
import types
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


def to_rllib(parallel_env: ParallelEnv) -> Any:
    """Present a GameBrains ParallelEnv to RLlib, widened so it can build a model.

    RLlib's current API stack has no default encoder for a `Discrete` observation space and fails
    at algorithm-build time with "No default encoder config for obs space=Discrete(n)". The
    widening is RLlib's requirement, not a defect in the observation, so it stays on this side of
    the boundary and only this route pays for it.

        from gamebrains.interop.wrappers import to_rllib
        env = to_rllib(GameBrainsParallelEnv(game, roster))

    Returns a `ParallelPettingZooEnv` when Ray is installed, so the result is ready to register
    with `ray.tune.register_env`. Without Ray it returns the one-hot `ParallelEnv` alone, which is
    still the useful half and keeps this import-light.
    """
    widened = OneHotObs(parallel_env)
    try:
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    except ImportError:
        return widened
    return ParallelPettingZooEnv(widened)


def to_mlpro_gym(factory: Callable[[], Any], env_id: str = "GameBrains-v0",
                 max_episode_steps: Optional[int] = None, visualize: bool = False) -> Any:
    """Present a GameBrainsGymEnv to MLPro through its Gymnasium bridge, registration included.

    `WrEnvGYM2MLPro` identifies an environment by `env.env.spec.id` and rebuilds it later through
    `gymnasium.make`, so a bare conformant `gymnasium.Env` is refused for want of a `.spec`. That
    is the whole cost, and doing the registration here means a caller never meets it.

        from gamebrains.interop.wrappers import to_mlpro_gym
        mlpro_env = to_mlpro_gym(lambda: GameBrainsGymEnv(game, roster))

    Note for anyone driving the result: MLPro's `compute_reward()` takes no arguments on this
    bridge, and passing states raises `NotImplementedError`. The reward is whatever the underlying
    step produced.
    """
    from mlpro_int_gymnasium.wrappers.basics import WrEnvGYM2MLPro

    registered = register_gym_env(env_id, factory, max_episode_steps=max_episode_steps)
    return WrEnvGYM2MLPro(gymnasium.make(registered), p_visualize=visualize)


def to_mlpro_pettingzoo(parallel_env: ParallelEnv, visualize: bool = False,
                        logging: Any = None) -> Any:
    """Present a GameBrains ParallelEnv to MLPro through its PettingZoo bridge.

    We previously recorded this route as impossible by construction, on the grounds that
    `WrEnvPZOO2MLPro` resolves the environment class by name inside five hardcoded
    `pettingzoo.*` submodules and therefore cannot see anything outside the PettingZoo
    distribution. That conclusion came from reading the bridge, and it was wrong. Running it
    showed two obstacles, both surmountable with mechanisms the two libraries already ship.

    **Obstacle one: the bridge speaks AEC, not Parallel.** `_reset` calls `self._zoo_env.last()`,
    which belongs to PettingZoo's turn-based API, while our adapter is a `ParallelEnv`. PettingZoo
    ships the converter for exactly this, so `parallel_to_aec_wrapper` closes it at no cost to us.

    **Obstacle two: the name lookup.** `C_SUPPORTED_MODULES` is a *class* attribute, so a subclass
    can extend it, and the lookup only needs `<module>.<metadata['name']>` to resolve to something
    non-None. The resolved class is used again only when unpickling a saved environment, which
    this path does not do, so a module holding the wrapper's own type satisfies it honestly.

    One detail of MLPro's loop is worth knowing before relying on this: the search `break`s after
    its first entry whether or not that entry resolved anything, so only the first module in the
    list is ever consulted. Ours is therefore the only one in the subclass below. If a future
    release moves the `break` inside the success branch, prepending still works and this comment
    becomes the explanation for why the list has one element.

        from gamebrains.interop.wrappers import to_mlpro_pettingzoo
        mlpro_env = to_mlpro_pettingzoo(GameBrainsParallelEnv(game, roster))

    Requires `mlpro` and `mlpro_int_pettingzoo`, whose own declared dependencies are incomplete:
    they do not import until `dill` and `multiprocess` are installed by hand.
    """
    from mlpro_int_pettingzoo.wrappers.basics import WrEnvPZOO2MLPro
    from pettingzoo.utils.conversions import parallel_to_aec_wrapper

    aec = parallel_to_aec_wrapper(parallel_env)
    name = aec.metadata["name"]

    shim_name = f"_gamebrains_mlpro_shim_{name}"
    shim = sys.modules.get(shim_name)
    if shim is None:
        shim = types.ModuleType(shim_name)
        sys.modules[shim_name] = shim
    setattr(shim, name, type(aec))

    bridge = type("WrGameBrainsPZOO2MLPro", (WrEnvPZOO2MLPro,),
                  {"C_SUPPORTED_MODULES": [shim_name]})
    if logging is None:
        from mlpro.bf.various import Log
        logging = Log.C_LOG_NOTHING          # this bridge logs every single action otherwise
    return bridge(aec, p_visualize=visualize, p_logging=logging)
