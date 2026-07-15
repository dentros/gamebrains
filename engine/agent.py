"""
The Agent contract — the heart of GameBrains.

Every "brain" (Q-learning, DQN, evolutionary Markov-brain, FEP/Bayesian, LLM, or a fixed
classic strategy) implements this one interface, so new brains plug in without touching the
engine, the games, the metrics, or the viz.

The two transparency hooks are what make the platform special:
  - `inspect()`      -> the raw internal state (Q-table, weights, beliefs, ...)
  - `render_brain()` -> a JSON-serializable payload the frontend knows how to draw as a
                        cute little "brain". Keep it small and stable in shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """Abstract base class for a cognitive agent."""

    #: display name of this brain, e.g. "Q-learning #2" or "Tit-for-Tat"
    name: str
    #: short kind tag used by the frontend to pick a renderer, e.g. "qlearning", "classic"
    kind: str = "agent"
    #: learning paradigm. "online" -> learns each round via update() (RL: Q-learning, DQN, ...);
    #: "fixed" -> never changes within/across matches (classic strategies, a frozen genome);
    #: "evolutionary" -> fixed within a match, evolved between matches by engine/evolution.py.
    training_mode: str = "fixed"

    @abstractmethod
    def act(self, observation: int) -> int:
        """Choose an action given the current observation index."""
        raise NotImplementedError

    def update(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        done: bool,
    ) -> None:
        """Learn / adapt from one transition. No-op for fixed strategies."""
        return None

    def inspect(self) -> dict[str, Any]:
        """Return the raw internal state. Default: nothing to show."""
        return {}

    def render_brain(self) -> dict[str, Any]:
        """Return a small, serializable description for the visualizer.

        Convention: {"kind": <self.kind>, ...payload}. The frontend switches on "kind".
        """
        return {"kind": self.kind}

    def on_match_end(self) -> None:
        """Hook called once when a match finishes (e.g. decay schedules, bookkeeping)."""
        return None
