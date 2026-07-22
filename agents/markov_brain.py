"""
Evolutionary Markov-brain animat.

A small stochastic recurrent binary network in the tradition of MABE / Albantakis et al.'s
animats: `n_sensor` nodes encode the game observation (clamped from the environment each step),
`n_hidden` nodes provide memory, and `n_motor` node(s) determine the action. Node dynamics:

    P(node_i = 1 at t+1) = sigmoid(W[i, :] . state_t + bias[i])   for i in hidden/motor
    node_i(t+1) = bit of the current observation                  for i in sensor

The weight matrix `W` and `bias` are the evolvable genome (see engine.evolution); this agent
never learns online (`training_mode = "evolutionary"`, `update()` is a no-op) -- fitness,
mutation, and crossover happen between matches, not within one.

Node order is [sensor_0..sensor_{ns-1}, hidden_0..hidden_{nh-1}, motor_0..motor_{nm-1}], and row
index `s` of `full_tpm()` decodes to a state via the little-endian convention
`state[k] = (s >> k) & 1` -- the same convention as `pyphi.convert.le_index2state`, which is what
`metrics.phi_autonomy` (Φ and causal autonomy) assumes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..engine.agent import Agent


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class MarkovBrainAgent(Agent):
    kind = "markov_brain"
    training_mode = "evolutionary"

    def __init__(
        self,
        name: str,
        n_states: int,
        n_actions: int,
        n_hidden: int = 2,
        W: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None,
        seed: int = 0,
        start_state: Optional[int] = None,
    ) -> None:
        self.name = name
        self.n_sensor = max(1, (n_states - 1).bit_length())
        self.n_motor = max(1, (n_actions - 1).bit_length())
        if n_actions != 2 ** self.n_motor:
            raise ValueError("MarkovBrainAgent requires n_actions to be a power of 2 "
                              f"(got n_actions={n_actions})")
        self.n_hidden = n_hidden
        self.n_nodes = self.n_sensor + self.n_hidden + self.n_motor
        self.n_controlled = self.n_hidden + self.n_motor  # hidden+motor: evolved dynamics

        self.rng = np.random.default_rng(seed)
        self.start_state = start_state if start_state is not None else n_states - 1

        if W is None:
            W = self.rng.normal(scale=0.5, size=(self.n_controlled, self.n_nodes))
        if bias is None:
            bias = self.rng.normal(scale=0.5, size=self.n_controlled)
        self.W = np.asarray(W, dtype=float)
        self.bias = np.asarray(bias, dtype=float)
        if self.W.shape != (self.n_controlled, self.n_nodes):
            raise ValueError(f"W must have shape {(self.n_controlled, self.n_nodes)}, "
                             f"got {self.W.shape}")
        if self.bias.shape != (self.n_controlled,):
            raise ValueError(f"bias must have shape {(self.n_controlled,)}, got {self.bias.shape}")

        self.state = np.zeros(self.n_nodes, dtype=int)

    # --- dynamics ---------------------------------------------------------------

    def _encode_observation(self, observation: int) -> np.ndarray:
        return np.array([(observation >> k) & 1 for k in range(self.n_sensor)], dtype=int)

    def _decode_action(self) -> int:
        motor_bits = self.state[self.n_sensor + self.n_hidden:]
        return int(sum(int(b) << k for k, b in enumerate(motor_bits)))

    def reset_state(self) -> None:
        self.state = np.zeros(self.n_nodes, dtype=int)

    def act(self, observation: int) -> int:
        if observation == self.start_state:
            self.reset_state()
        self.state[:self.n_sensor] = self._encode_observation(observation)

        logits = self.W @ self.state + self.bias
        p = _sigmoid(logits)
        sampled = (self.rng.random(self.n_controlled) < p).astype(int)
        self.state[self.n_sensor:] = sampled

        return self._decode_action()

    def update(self, observation: int, action: int, reward: float,
              next_observation: int, done: bool) -> None:
        return None  # evolutionary: no online learning within a match

    # --- genome / GA hooks -------------------------------------------------------

    def clone(self, name: Optional[str] = None, seed: Optional[int] = None) -> "MarkovBrainAgent":
        """A fresh agent with the identical genome but a reset state and (by default) a new RNG."""
        new_seed = seed if seed is not None else int(self.rng.integers(1 << 30))
        return MarkovBrainAgent(
            name=name or self.name, n_states=2 ** self.n_sensor, n_actions=2 ** self.n_motor,
            n_hidden=self.n_hidden, W=self.W.copy(), bias=self.bias.copy(),
            seed=new_seed, start_state=self.start_state,
        )

    def mutate(self, rng: np.random.Generator, rate: float = 0.15, scale: float = 0.4,
              name: Optional[str] = None, seed: int = 0) -> "MarkovBrainAgent":
        """Return a new agent whose genome is a Gaussian-perturbed copy of this one's.

        `rng` drives which genes mutate (GA-level randomness, for reproducibility of the whole
        run); `seed` seeds the *resulting* agent's own internal stochastic dynamics.
        """
        W = self.W.copy()
        bias = self.bias.copy()
        w_mask = rng.random(W.shape) < rate
        W[w_mask] += rng.normal(scale=scale, size=int(w_mask.sum()))
        b_mask = rng.random(bias.shape) < rate
        bias[b_mask] += rng.normal(scale=scale, size=int(b_mask.sum()))
        return MarkovBrainAgent(
            name=name or self.name, n_states=2 ** self.n_sensor, n_actions=2 ** self.n_motor,
            n_hidden=self.n_hidden, W=W, bias=bias, seed=seed, start_state=self.start_state,
        )

    @classmethod
    def random(cls, name: str, n_states: int, n_actions: int, n_hidden: int = 2,
               seed: int = 0, start_state: Optional[int] = None) -> "MarkovBrainAgent":
        return cls(name, n_states, n_actions, n_hidden=n_hidden, seed=seed, start_state=start_state)

    # --- introspection ------------------------------------------------------------

    def full_tpm(self) -> np.ndarray:
        """State-by-node TPM: tpm[s, i] = P(node i = 1 | current joint state == s), with row
        index `s` decoded via the little-endian convention (matches pyphi.convert.le_index2state).
        Sensor nodes are given an identity (self-copy) transition, since they are clamped from the
        environment rather than evolving autonomously -- the standard treatment of input nodes so
        the network has well-defined intrinsic dynamics for PyPhi's complex search.
        """
        n = self.n_nodes
        tpm = np.zeros((2 ** n, n))
        for s in range(2 ** n):
            state = np.array([(s >> k) & 1 for k in range(n)], dtype=float)
            tpm[s, :self.n_sensor] = state[:self.n_sensor]
            logits = self.W @ state + self.bias
            tpm[s, self.n_sensor:] = _sigmoid(logits)
        return tpm

    def inspect(self) -> dict:
        return {
            "W": self.W.tolist(), "bias": self.bias.tolist(), "state": self.state.tolist(),
            "n_sensor": self.n_sensor, "n_hidden": self.n_hidden, "n_motor": self.n_motor,
        }

    def render_brain(self) -> dict:
        # W/bias ARE the genome (see __init__) -- exposing them here is what lets the webui draw
        # the actual evolved wiring (edges/weights/TPM), not just the current state bits.
        labels = ([f"s{k}" for k in range(self.n_sensor)]
                  + [f"h{k}" for k in range(self.n_hidden)]
                  + [f"m{k}" for k in range(self.n_motor)])
        return {
            "kind": self.kind,
            "n_sensor": self.n_sensor, "n_hidden": self.n_hidden, "n_motor": self.n_motor,
            "state": self.state.tolist(),
            "node_labels": labels,
            "W": self.W.tolist(),
            "bias": self.bias.tolist(),
        }


def crossover(parent_a: MarkovBrainAgent, parent_b: MarkovBrainAgent, rng: np.random.Generator,
             name: str, seed: int) -> MarkovBrainAgent:
    """Uniform crossover: each genome element independently comes from one parent or the other."""
    if parent_a.W.shape != parent_b.W.shape:
        raise ValueError("crossover requires parents with identical genome shape")
    W = np.where(rng.random(parent_a.W.shape) < 0.5, parent_a.W, parent_b.W)
    bias = np.where(rng.random(parent_a.bias.shape) < 0.5, parent_a.bias, parent_b.bias)
    return MarkovBrainAgent(
        name=name, n_states=2 ** parent_a.n_sensor, n_actions=2 ** parent_a.n_motor,
        n_hidden=parent_a.n_hidden, W=W, bias=bias, seed=seed, start_state=parent_a.start_state,
    )
