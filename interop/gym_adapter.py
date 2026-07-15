"""
Gymnasium single-agent adapter for a GameBrains `Game`.

This is the single-controlled-seat special case of `GameBrainsParallelEnv`: exactly one seat
(`controlled_agent_id`) is exposed to the caller as a standard `gymnasium.Env`; every other seat
is a real GameBrains `Agent` from `roster` acting (and learning) on its own each round. It is
implemented by composing `GameBrainsParallelEnv` rather than duplicating the seat-splitting logic,
so both adapters share one validated implementation.

Getting Gymnasium compatibility is what makes Stable-Baselines3 and MLPro's
`WrEnvGYM2MLPro` usable against GameBrains's own cognitive brains with no further code.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium

from ..engine.agent import Agent
from ..engine.game import Game
from .pettingzoo_adapter import GameBrainsParallelEnv


class GameBrainsGymEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        game: Game,
        roster: list[Optional[Agent]],
        controlled_agent_id: int = 0,
    ) -> None:
        self._env = GameBrainsParallelEnv(game, roster, controlled_agent_ids=[controlled_agent_id])
        self._aid = self._env.possible_agents[controlled_agent_id]

        self.observation_space = self._env.observation_space(self._aid)
        self.action_space = self._env.action_space(self._aid)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)  # Gymnasium API contract: records that a seed was accepted.
        obs, infos = self._env.reset(seed=seed, options=options)
        return obs[self._aid], infos[self._aid]

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        obs, rewards, terminateds, truncateds, infos = self._env.step({self._aid: action})
        return (
            obs[self._aid],
            rewards[self._aid],
            terminateds[self._aid],
            truncateds[self._aid],
            infos[self._aid],
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
