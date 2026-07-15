"""
PettingZoo Parallel-API adapter for a GameBrains `Game`.

Wraps any `engine.Game` + a roster of `engine.Agent` as a `pettingzoo.utils.env.ParallelEnv`.
Agent ids are `"agent_0".."agent_{n-1}"`, matching the seat index in the game/roster.

The point of this adapter is not "make our game generically playable" — it is to let an
*external* RL framework (RLlib, Stable-Baselines3 via the Gym adapter, MLPro) train or evaluate
one or more of its own policies **against GameBrains's own cognitive brains**. Seats listed in
`controlled_agent_ids` expect an action from the caller each `step()`; every other seat is a real
`Agent` from `roster` that acts (and learns, via `update()`) on its own, invisibly, each round.
Passing `controlled_agent_ids=None` (the default) makes every seat externally controlled, which is
what a plain conformance test (`pettingzoo.test.parallel_api_test`) or full self-play expects.

Notes on the mapping to our engine:
  - Our games run a fixed number of rounds and end by length, not by reaching an absorbing state,
    so a finished match is reported as `truncated=True`, `terminated=False` for every agent (per
    Gymnasium's terminated/truncated distinction).
  - Rendering and `state()` are not implemented; `metadata["render_modes"] = []` advertises this.
  - `reset(seed=...)` is accepted for API compliance but not threaded into the underlying `Game` or
    `Agent` RNGs, which are seeded at construction time — see `engine.eventlog` for how GameBrains
    achieves reproducibility instead.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium
from pettingzoo.utils.env import ParallelEnv

from ..engine.agent import Agent
from ..engine.game import Game


class GameBrainsParallelEnv(ParallelEnv):
    metadata = {"render_modes": [], "name": "gamebrains_v0"}

    def __init__(
        self,
        game: Game,
        roster: list[Optional[Agent]],
        controlled_agent_ids: Optional[list[int]] = None,
    ) -> None:
        if len(roster) != game.n_agents:
            raise ValueError(f"roster length {len(roster)} != game.n_agents {game.n_agents}")

        self.game = game
        self.roster = roster
        self._controlled = set(range(game.n_agents) if controlled_agent_ids is None
                                else controlled_agent_ids)
        self._background = {i: roster[i] for i in range(game.n_agents) if i not in self._controlled}
        for i, agent in self._background.items():
            if agent is None:
                raise ValueError(f"seat {i} is a background seat but roster[{i}] is None")

        self.possible_agents = [f"agent_{i}" for i in range(game.n_agents)]
        self.agents: list[str] = list(self.possible_agents)
        self._id_to_seat = {aid: i for i, aid in enumerate(self.possible_agents)}

        self.observation_spaces = {aid: gymnasium.spaces.Discrete(game.n_states)
                                    for aid in self.possible_agents}
        self.action_spaces = {aid: gymnasium.spaces.Discrete(game.n_actions)
                               for aid in self.possible_agents}

        self._last_obs: dict[str, int] = {}

    def observation_space(self, agent: str) -> gymnasium.spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.spaces.Space:
        return self.action_spaces[agent]

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, int], dict[str, dict]]:
        obs_list = self.game.reset()
        self.agents = list(self.possible_agents)
        self._last_obs = {aid: obs_list[i] for i, aid in enumerate(self.possible_agents)}
        infos = {aid: {} for aid in self.possible_agents}
        return dict(self._last_obs), infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict[str, int], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        n = len(self.possible_agents)
        full_actions: list[int] = [0] * n

        for i in range(n):
            aid = self.possible_agents[i]
            if i in self._controlled:
                if aid not in actions:
                    raise KeyError(f"missing action for controlled seat {aid}")
                full_actions[i] = actions[aid]
            else:
                full_actions[i] = self._background[i].act(self._last_obs[aid])

        result = self.game.step(full_actions)

        for i, agent in self._background.items():
            aid = self.possible_agents[i]
            agent.update(self._last_obs[aid], full_actions[i], result.rewards[i],
                        result.observations[i], result.done)

        obs = {aid: result.observations[i] for i, aid in enumerate(self.possible_agents)}
        rewards = {aid: result.rewards[i] for i, aid in enumerate(self.possible_agents)}
        terminateds = {aid: False for aid in self.possible_agents}
        truncateds = {aid: result.done for aid in self.possible_agents}
        infos = {aid: dict(result.info) for aid in self.possible_agents}

        self._last_obs = obs
        if result.done:
            self.agents = []

        return obs, rewards, terminateds, truncateds, infos

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
