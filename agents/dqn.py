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
    # Same as the tabular learner: the network's output head is indexed by action and
    # carries no interpretation of what the index means.
    semantics = "index-agnostic"
    #: One observation in, action values out. The network has no recurrence, so like the tabular
    #: agent it conditions on the current state alone.
    information = "observation"

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
        lifetime_learning: bool = True,
        learning_rule: str = "dqn",
    ) -> None:
        self.name = name
        # Both default to the ordinary online learner, so every existing caller gets the agent it
        # got before. Only `spawn` sets them otherwise. See this module's neuroevolution section.
        self.lifetime_learning = lifetime_learning
        self.learning_rule = learning_rule
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        # `epsilon` decays in place, so the schedule's starting point is gone by the end of
        # a match. metrics/information.suggested_burn_in solves for when exploration stops
        # changing, and it needs where it started.
        self.epsilon_start = epsilon
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
        #: Replay sampling has its OWN seeded generator, and this line is load-bearing. It used to
        #: call the `random` module's global functions, which are seeded from the interpreter and not
        #: from us, so two matches with identical agent seeds diverged and the same match changed when
        #: unrelated code touched the global RNG. Measured before the fix: 119 of 600 rounds differed
        #: between two identically-seeded runs once exploration had decayed, and 45 of 600 changed in
        #: response to the global seed alone. That silently falsified the platform's byte-identical
        #: event-log property for any roster containing a DQN, and with it the repository's premise
        #: that one configuration hash implies one result. Do not replace this with `random.sample`.
        self._replay_rng = random.Random(seed)

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
        if not self.lifetime_learning:
            # Under Darwinian evolution the weights are what selection acts on, so a match has to
            # leave them as it found them. Returning here rather than relying on each caller to
            # skip the call keeps the guarantee with the agent.
            return
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
        batch = self._replay_rng.sample(self.buffer, self.batch_size)
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

    # --- The Evolvable protocol (engine/evolution.py) ------------------------------------------
    #
    # Searching over these weights with a genetic algorithm is neuroevolution, not deep
    # Q-learning: no gradient is taken and the Bellman target plays no part. Such an agent
    # reports `learning_rule="evolved"`, because a bake-off row that called it DQN would be
    # naming the architecture while the reader hears a claim about the learning rule.

    def reset_state(self) -> None:
        """Clear what belongs to the previous grouping, keep what selection acts on.

        The weights are the genome and survive. The replay buffer is the opposite: it holds
        transitions against opponents this individual will not face again, and carrying it over
        would let one grouping's experience train the agent during the next one's measurement.
        Empty under the Darwinian default anyway, since `update` returns before filling it.
        """
        self.buffer.clear()

    def _respawn(self, state_dict: dict, name: Optional[str], seed: int) -> "DQNAgent":
        """One constructor call for clone and mutate, so a field cannot be carried by one and
        dropped by the other."""
        twin = DQNAgent(
            name=name or self.name, n_states=self.n_states, n_actions=self.n_actions,
            hidden=self.hidden, lr=self.lr, gamma=self.gamma, epsilon=self.epsilon,
            epsilon_min=self.epsilon_min, epsilon_decay=self.epsilon_decay,
            buffer_size=self.buffer_size, batch_size=self.batch_size,
            train_every=self.train_every, target_sync_every=self.target_sync_every, seed=seed,
            state_labels=list(self.state_labels), action_labels=list(self.action_labels),
            device=str(self.device), lifetime_learning=self.lifetime_learning,
            learning_rule=self.learning_rule,
        )
        twin.policy.load_state_dict(state_dict)
        # The target net is a lagged copy of the policy, so an inherited genome has to reach both.
        # Leaving the target at its own random initialisation would make a cloned elite behave
        # differently from its parent the moment it trained.
        twin.target.load_state_dict(state_dict)
        return twin

    def clone(self, name: Optional[str] = None, seed: Optional[int] = None) -> "DQNAgent":
        """The identical network under a new name and RNG. Used to carry elites over."""
        new_seed = seed if seed is not None else int(self.rng.integers(1 << 30))
        return self._respawn({k: v.clone() for k, v in self.policy.state_dict().items()},
                             name, new_seed)

    def mutate(self, rng: np.random.Generator, rate: float = 0.15, scale: float = 0.4,
               name: Optional[str] = None, seed: int = 0) -> "DQNAgent":
        """A Gaussian-perturbed copy of every weight tensor.

        `rng` is the GA's own generator rather than torch's, deliberately: it is what makes the
        whole run reproducible from `EvolutionConfig.seed`, and drawing the perturbation from
        torch's global state would reintroduce exactly the defect this file's replay-sampling
        comment describes.
        """
        genome = {}
        for key, tensor in self.policy.state_dict().items():
            values = tensor.detach().cpu().numpy().copy()
            mask = rng.random(values.shape) < rate
            values[mask] += rng.normal(scale=scale, size=int(mask.sum()))
            genome[key] = torch.from_numpy(values).to(tensor.dtype)
        return self._respawn(genome, name, seed)


def crossover(parent_a: "DQNAgent", parent_b: "DQNAgent", rng: np.random.Generator,
              name: str, seed: int) -> "DQNAgent":
    """Uniform crossover, element by element, over every weight tensor.

    Recombination rather than averaging, for the same reason as the tabular case: the mean of two
    networks is a genome neither parent had. Note what this inherits from the architecture, since
    it is the standard objection to crossing networks -- two parents can compute the same function
    through differently ordered hidden units, and recombining them cell by cell then produces a
    child that matches neither. Here both parents descend from one spawned population at a fixed
    hidden width, which narrows that mismatch without removing it. Reported, not solved.
    """
    a_state, b_state = parent_a.policy.state_dict(), parent_b.policy.state_dict()
    if [(k, tuple(v.shape)) for k, v in a_state.items()] != \
            [(k, tuple(v.shape)) for k, v in b_state.items()]:
        raise ValueError(
            "crossover requires parents with identical genome shape; these two networks differ "
            "in their parameter tensors, so they were built for different games or widths")
    genome = {}
    for key, tensor in a_state.items():
        a_values = tensor.detach().cpu().numpy()
        b_values = b_state[key].detach().cpu().numpy()
        picked = np.where(rng.random(a_values.shape) < 0.5, a_values, b_values)
        genome[key] = torch.from_numpy(picked).to(tensor.dtype)
    return parent_a._respawn(genome, name, seed)


def spawn(name: str, game, seed: int, config) -> "DQNAgent":
    """Build one random network individual for `engine.evolution.evolve`.

    Variation across the population comes free here, unlike the tabular case: each individual is
    constructed under its own seed and `_MLP`'s initialisation reads it, so the first generation
    already differs. `lifetime_learning` decides the same thing it decides for a Q-table, and
    means more here, since under `"lamarckian"` the GA is searching over starting points for
    gradient descent rather than over policies.
    """
    darwinian = getattr(config, "lifetime_learning", "off") != "lamarckian"
    knobs = getattr(config, "genome", None) or {}
    return DQNAgent(
        name, n_states=game.n_states, n_actions=game.n_actions, seed=seed,
        hidden=int(knobs.get("hidden", 64)),
        epsilon=0.0 if darwinian else float(knobs.get("epsilon", 0.1)),
        epsilon_min=0.0 if darwinian else float(knobs.get("epsilon_min", 0.01)),
        lr=float(knobs.get("lr", 1e-3)), gamma=float(knobs.get("gamma", 0.95)),
        lifetime_learning=not darwinian, learning_rule="evolved",
    )
