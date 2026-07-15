"""
Free-Energy-Principle / Active-Inference agent (a Theory-of-Mind brain).

This brain does two things, both straight from the FEP:

  1. Perception = variational free-energy minimization. It maintains a categorical posterior
     `qs` over a hidden state — "how many of the OTHER players will cooperate this round" — and
     updates it by exact Bayesian filtering: predict through a slow-drift transition B, then
     multiply by the observation likelihood A and renormalize. That posterior *is* its little
     model of the others' minds, which is why this is our Theory-of-Mind brain.

  2. Action = expected-free-energy minimization. For each action it computes expected value under
     its beliefs (pragmatic value); because its own move does not change the others' disposition,
     the epistemic term is action-independent here, so action selection reduces to a precision-
     weighted softmax over expected value. With `reciprocity=0` the agent is purely self-interested
     (and, correctly, free-rides — defection dominates the Public Goods Game); a positive
     `reciprocity` turns it into a belief-driven conditional cooperator.

We implement the active-inference math directly in numpy (rather than via pymdp's current
JAX API) for robustness and full transparency; a pymdp backend can be added later.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..engine.agent import Agent
from ..games.public_goods import COOPERATE, DEFECT


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


class FEPAgent(Agent):
    kind = "fep"
    training_mode = "online"

    def __init__(
        self,
        name: str,
        n_agents: int,
        mpcr: float,
        cost: float = 1.0,
        obs_noise: float = 0.75,   # σ of the likelihood A over hidden "others cooperating"
        drift: float = 0.1,        # how fast others' disposition is assumed to drift (B)
        precision: float = 4.0,    # γ: how sharply value drives action
        reciprocity: float = 0.0,  # >0 -> prefer cooperating when it believes others do
        seed: int = 0,
        start_state: Optional[int] = None,
    ) -> None:
        self.name = name
        self.n_agents = n_agents
        self.n_others = n_agents - 1
        self.mpcr = mpcr
        self.cost = cost
        self.precision = precision
        self.reciprocity = reciprocity
        self.rng = np.random.default_rng(seed)
        self.start_state = start_state if start_state is not None else n_agents + 1

        # Hidden states s = 0..n_others  (number of OTHER players cooperating this round)
        self.n_states = self.n_others + 1
        states = np.arange(self.n_states)

        # Likelihood A[o, s] ∝ exp(-(o - s)^2 / 2σ^2), normalized over observations per state.
        diff = states[:, None] - states[None, :]
        A = np.exp(-(diff ** 2) / (2.0 * obs_noise ** 2))
        self.A = A / A.sum(axis=0, keepdims=True)

        # Transition B = (1 - drift) I + drift * uniform  (slow, action-independent drift)
        self.B = (1.0 - drift) * np.eye(self.n_states) + drift * (np.ones((self.n_states, self.n_states)) / self.n_states)

        self.qs = np.ones(self.n_states) / self.n_states   # posterior over others cooperating
        self.own_last_action = DEFECT
        self.last_action_probs = np.array([0.5, 0.5])

    # --- perception ---------------------------------------------------------------
    def _perceive(self, others_cooperating: int) -> None:
        pred = self.B @ self.qs                       # prior after drift
        like = self.A[int(np.clip(others_cooperating, 0, self.n_states - 1)), :]
        post = pred * like
        s = post.sum()
        self.qs = post / s if s > 0 else np.ones(self.n_states) / self.n_states

    # --- expected value under beliefs ---------------------------------------------
    def _expected_values(self) -> np.ndarray:
        s = np.arange(self.n_states)
        ev_defect = float(np.sum(self.qs * (self.mpcr * s * self.cost)))
        ev_coop = float(np.sum(self.qs * (self.mpcr * (s + 1) * self.cost - self.cost)))
        if self.reciprocity:
            ev_coop += self.reciprocity * float(np.sum(self.qs * s)) / max(self.n_others, 1)
        return np.array([ev_defect, ev_coop])   # index 0=Defect, 1=Cooperate

    # --- Agent interface ----------------------------------------------------------
    def act(self, observation: int) -> int:
        if observation != self.start_state:
            others = observation - self.own_last_action   # obs = total cooperators last round
            self._perceive(others)
        ev = self._expected_values()
        self.last_action_probs = _softmax(self.precision * ev)
        action = COOPERATE if self.rng.random() < self.last_action_probs[COOPERATE] else DEFECT
        self.own_last_action = action
        return action

    def expected_others(self) -> float:
        return float(np.sum(np.arange(self.n_states) * self.qs))

    def inspect(self) -> dict:
        return {
            "belief_others_cooperating": self.qs.tolist(),
            "expected_others": self.expected_others(),
            "expected_values": self._expected_values().tolist(),
            "action_probs": self.last_action_probs.tolist(),
        }

    def render_brain(self) -> dict:
        return {
            "kind": self.kind,
            "belief": self.qs.tolist(),
            "state_labels": [f"others={s}" for s in range(self.n_states)],
            "expected_others": self.expected_others(),
            "expected_values": {"Defect": self._expected_values()[0], "Cooperate": self._expected_values()[1]},
            "action_probs": {"Defect": self.last_action_probs[DEFECT], "Cooperate": self.last_action_probs[COOPERATE]},
            "reciprocity": self.reciprocity,
        }
