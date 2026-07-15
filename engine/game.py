"""
The Game contract.

Every game in GameBrains is n-player-native (n >= 2). A game is a *repeated* stage
game: `reset()` starts a new match, and each `step(actions)` advances one round in which
all agents move simultaneously. This keeps the engine agnostic to the specific game so
that the N-player Prisoner's Dilemma (Public Goods), MBoE (Battle of the Exes), and future
games all plug into the same loop, metrics, and visualization.

Observations are integers so that tabular agents (Q-learning) can index a Q-table directly;
`n_states` bounds that index. Richer agents (DQN/FEP/LLM) may ignore the integer and read
the full round context from `StepResult.info` instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    """Outcome of one round (one `step`)."""

    observations: list[int]          # per-agent observation index for the *next* round
    rewards: list[float]             # per-agent payoff for this round
    done: bool                       # True when the match is over
    info: dict[str, Any] = field(default_factory=dict)  # rich per-round context for metrics/viz


class Game(ABC):
    """Abstract base class for an n-player repeated game."""

    #: number of players (>= 2)
    n_agents: int
    #: number of discrete actions available to each player
    n_actions: int
    #: human-readable action labels, len == n_actions (e.g. ["Defect", "Cooperate"])
    action_names: list[str]
    #: number of distinct observation indices an agent may receive
    n_states: int
    #: short identifier used in logs and filenames
    name: str

    @abstractmethod
    def reset(self) -> list[int]:
        """Begin a new match. Returns the initial per-agent observation indices."""
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: list[int]) -> StepResult:
        """Advance one round given every agent's action. See `StepResult`."""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """Serializable description of the game configuration (goes into the event-log header)."""
        return {
            "name": self.name,
            "n_agents": self.n_agents,
            "n_actions": self.n_actions,
            "action_names": list(self.action_names),
            "n_states": self.n_states,
        }
