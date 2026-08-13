"""
Tabular Q-learning agent.

The "starter brain": its internal state is a Q-table you can read directly, which makes it the
clearest little "μυαλάκι" to visualize. Crucially, the learning rule lives *here* (in `update`),
not in the game — that is what lets a Q-learner share a match with a DQN, an FEP agent, or a
classic strategy without the game knowing anything about how any of them learns.

State-action values Q[s, a]; epsilon-greedy action selection with multiplicative epsilon decay.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..engine.agent import Agent


class QLearningAgent(Agent):
    kind = "qlearning"
    training_mode = "online"
    # A Q-table is indexed by action, and the agent never asks what an index stands for:
    # it learns whichever column pays. Reversing a game's action meanings would change
    # what it learns, not whether it is correct.
    semantics = "index-agnostic"

    def __init__(
        self,
        name: str,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.9995,
        seed: int = 0,
        state_labels: Optional[list[str]] = None,
        action_labels: Optional[list[str]] = None,
    ) -> None:
        self.name = name
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        self.Q = np.zeros((n_states, n_actions), dtype=float)
        self.state_labels = state_labels or [f"s{i}" for i in range(n_states)]
        self.action_labels = action_labels or [f"a{j}" for j in range(n_actions)]

        self.updates = 0

    def act(self, observation: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        row = self.Q[observation]
        # argmax with random tie-breaking for unbiased exploration of equal-value actions
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    def update(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        done: bool,
    ) -> None:
        best_next = 0.0 if done else float(self.Q[next_observation].max())
        target = reward + self.gamma * best_next
        td_error = target - self.Q[observation, action]
        self.Q[observation, action] += self.alpha * td_error

        self.updates += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_policy(self) -> list[int]:
        return [int(np.argmax(self.Q[s])) for s in range(self.n_states)]

    def inspect(self) -> dict:
        return {
            "Q": self.Q.tolist(),
            "epsilon": self.epsilon,
            "updates": self.updates,
            "state_labels": self.state_labels,
            "action_labels": self.action_labels,
        }

    def render_brain(self) -> dict:
        return {
            "kind": self.kind,
            "q_table": self.Q.tolist(),
            "greedy_policy": self.greedy_policy(),
            "epsilon": self.epsilon,
            "state_labels": self.state_labels,
            "action_labels": self.action_labels,
        }
