"""
Deep Q-Network (DQN) agent — Q-learning with a neural-network function approximator.

This is the "advanced RL" brain: instead of a lookup table it approximates Q(s, a) with a small
MLP, trained from a replay buffer with a target network (the standard DQN recipe). The design is
adapted from the author's own ALT project (`src/dqn_agents.py`), but the learning rule lives *inside
the agent* (`update`), so a DQN can share a match with Q-learners, FEP agents, or classic
strategies without the game knowing anything about it.

Observations are integer indices (as produced by the games here); they are one-hot encoded before
entering the network. `render_brain()` exposes the network's Q-values for *every* state, giving a
view directly comparable to the tabular Q-learner's Q-table (handy for the "houses" UI).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..engine.agent import Agent


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(Agent):
    kind = "dqn"
    training_mode = "online"

    def __init__(
        self,
        name: str,
        n_states: int,
        n_actions: int,
        hidden: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        train_every: int = 1,
        target_sync_every: int = 200,
        seed: int = 0,
        state_labels: Optional[list[str]] = None,
        action_labels: Optional[list[str]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.name = name
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.train_every = train_every
        self.target_sync_every = target_sync_every

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # Seed for reproducibility (per-agent so a roster is deterministic given the run seed).
        self.rng = np.random.default_rng(seed)
        self._gen = torch.Generator(device="cpu").manual_seed(seed)
        torch.manual_seed(seed)

        self.policy = _MLP(n_states, n_actions, hidden).to(self.device)
        self.target = _MLP(n_states, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer: deque = deque(maxlen=buffer_size)

        self.state_labels = state_labels or [f"s{i}" for i in range(n_states)]
        self.action_labels = action_labels or [f"a{j}" for j in range(n_actions)]
        self.hidden = hidden
        # Stored purely for introspection (repository/record.py's config_hash needs the actual
        # hyperparameter values, not just "kind" -- these aren't otherwise read after __init__).
        self.lr = lr
        self.buffer_size = buffer_size
        self.steps = 0
        self.last_loss: float = 0.0

    # --- encoding -----------------------------------------------------------------
    def _one_hot(self, s: int) -> np.ndarray:
        v = np.zeros(self.n_states, dtype=np.float32)
        v[s] = 1.0
        return v

    # --- policy -------------------------------------------------------------------
    def act(self, observation: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            x = torch.from_numpy(self._one_hot(observation)).unsqueeze(0).to(self.device)
            q = self.policy(x)
            return int(torch.argmax(q, dim=1).item())

    def update(
        self,
        observation: int,
        action: int,
        reward: float,
        next_observation: int,
        done: bool,
    ) -> None:
        self.buffer.append((self._one_hot(observation), action, reward,
                            self._one_hot(next_observation), float(done)))
        self.steps += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if len(self.buffer) >= self.batch_size and self.steps % self.train_every == 0:
            self._train_step()
        if self.steps % self.target_sync_every == 0:
            self.target.load_state_dict(self.policy.state_dict())

    def _train_step(self) -> None:
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s2, d = zip(*batch)
        s = torch.from_numpy(np.stack(s)).to(self.device)
        s2 = torch.from_numpy(np.stack(s2)).to(self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.policy(s).gather(1, a)
        with torch.no_grad():
            q_next = self.target(s2).max(1, keepdim=True)[0]
            target = r + (1.0 - d) * self.gamma * q_next
        loss = nn.functional.mse_loss(q, target)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()
        self.last_loss = float(loss.item())

    # --- introspection ------------------------------------------------------------
    def _all_q(self) -> np.ndarray:
        with torch.no_grad():
            eye = torch.eye(self.n_states, dtype=torch.float32, device=self.device)
            return self.policy(eye).cpu().numpy()

    def greedy_policy(self) -> list[int]:
        return [int(a) for a in np.argmax(self._all_q(), axis=1)]

    def inspect(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "steps": self.steps,
            "last_loss": self.last_loss,
            "buffer": len(self.buffer),
            "q_values": self._all_q().tolist(),
        }

    def render_brain(self) -> dict:
        return {
            "kind": self.kind,
            "layers": [self.n_states, self.hidden, self.hidden, self.n_actions],
            "epsilon": self.epsilon,
            "last_loss": self.last_loss,
            "q_values": self._all_q().tolist(),      # per-state Q, comparable to the Q-table
            "greedy_policy": self.greedy_policy(),
            "state_labels": self.state_labels,
            "action_labels": self.action_labels,
            "device": str(self.device),
        }
