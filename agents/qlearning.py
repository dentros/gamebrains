"""
Tabular Q-learning agent.

The "starter brain": its internal state is a Q-table you can read directly, which makes it the
clearest little "μυαλάκι" to visualize. Crucially, the learning rule lives *here* (in `update`),
not in the game — that is what lets a Q-learner share a match with a DQN, an FEP agent, or a
classic strategy without the game knowing anything about how any of them learns.

State-action values Q[s, a]; epsilon-greedy action selection with multiplicative epsilon decay.

**The table is also a genome.** `spawn` and `crossover` at the bottom of this file let
`engine.evolution` search over Q-tables directly, the way it already searches over Markov-brain
weights, which is the point of the GA taking its genome operators as arguments. A table found that
way was not produced by Q-learning, so such an agent reports `learning_rule="evolved"`; the
architecture is what `kind` says and the two are not the same claim.

That raises a question the Markov brain never could. `runner.run_match` calls `update` on every
agent every round, so an agent that learns would rewrite its own genome *during the match that
measures its fitness*, and `_next_generation` would clone the rewritten table into the next
generation. That is Lamarckian inheritance, and nothing in the output distinguishes it from a GA
that simply worked well. `EvolutionConfig.lifetime_learning` makes it a decision with a name;
`spawn` reads it and builds an agent that either learns during evaluation or does not.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..engine.agent import Agent


class QLearningAgent(Agent):
    kind = "qlearning"
    training_mode = "online"
    # A Q-table is indexed by action, and the agent never asks what an index stands for:
    # it learns whichever column pays. Reversing a game's action meanings would change
    # what it learns, not whether it is correct.
    semantics = "index-agnostic"
    #: The Q-table is indexed by the current observation alone, so the agent is Markov in
    #: whatever state the game publishes and sees nothing of the rounds before it.
    information = "observation"

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
        lifetime_learning: bool = True,
        learning_rule: str = "q_learning",
    ) -> None:
        self.name = name
        # Both default to the ordinary online learner, so every existing caller gets exactly the
        # agent it got before. Only `spawn` sets them otherwise.
        self.lifetime_learning = lifetime_learning
        self.learning_rule = learning_rule
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # `epsilon` decays in place, so the schedule's starting point is gone by the end of
        # a match. metrics/information.suggested_burn_in solves for when exploration stops
        # changing, and it needs where it started.
        self.epsilon_start = epsilon
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
        if not self.lifetime_learning:
            # Under Darwinian evolution the table is what selection acts on, so a match must
            # leave it exactly as it found it. Returning here rather than having the caller skip
            # the call keeps the guarantee with the agent: the runner, the adapters and the
            # bake-off wrappers all drive agents and would each have to remember.
            return
        best_next = 0.0 if done else float(self.Q[next_observation].max())
        target = reward + self.gamma * best_next
        td_error = target - self.Q[observation, action]
        self.Q[observation, action] += self.alpha * td_error

        self.updates += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_policy(self) -> list[int]:
        return [int(np.argmax(self.Q[s])) for s in range(self.n_states)]

    # --- The Evolvable protocol (engine/evolution.py) ------------------------------------------

    def reset_state(self) -> None:
        """Nothing to clear, and that is worth saying rather than leaving blank.

        The GA calls this before every match so fitness is not inherited from the previous
        grouping. A Markov brain has hidden nodes carrying the last match into this one; a
        Q-learner's only state across rounds *is* the table, which is the genome and must survive.
        Epsilon is the one thing that drifts, and under the Darwinian default it is zero.
        """

    def _respawn(self, Q: np.ndarray, name: Optional[str], seed: int) -> "QLearningAgent":
        """One constructor call for clone and mutate, so a new field cannot be carried by one
        and dropped by the other. That is how a mutated child quietly becomes a learner."""
        twin = QLearningAgent(
            name=name or self.name, n_states=self.n_states, n_actions=self.n_actions,
            alpha=self.alpha, gamma=self.gamma, epsilon=self.epsilon,
            epsilon_min=self.epsilon_min, epsilon_decay=self.epsilon_decay, seed=seed,
            state_labels=list(self.state_labels), action_labels=list(self.action_labels),
            lifetime_learning=self.lifetime_learning, learning_rule=self.learning_rule,
        )
        twin.Q = Q
        return twin

    def clone(self, name: Optional[str] = None, seed: Optional[int] = None) -> "QLearningAgent":
        """The identical policy under a new name and RNG. Used to carry elites over."""
        new_seed = seed if seed is not None else int(self.rng.integers(1 << 30))
        return self._respawn(self.Q.copy(), name, new_seed)

    def mutate(self, rng: np.random.Generator, rate: float = 0.15, scale: float = 0.4,
               name: Optional[str] = None, seed: int = 0) -> "QLearningAgent":
        """A Gaussian-perturbed copy. `rng` drives which cells mutate, so the whole GA run stays
        reproducible; `seed` seeds the resulting agent's own tie-breaking."""
        Q = self.Q.copy()
        mask = rng.random(Q.shape) < rate
        Q[mask] += rng.normal(scale=scale, size=int(mask.sum()))
        return self._respawn(Q, name, seed)

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
            # Carried separately from `kind` on purpose: a table the GA found and a table
            # Q-learning built look identical here, and only one of them was learned.
            "learning_rule": self.learning_rule,
            "q_table": self.Q.tolist(),
            "greedy_policy": self.greedy_policy(),
            "epsilon": self.epsilon,
            "state_labels": self.state_labels,
            "action_labels": self.action_labels,
        }


def crossover(parent_a: QLearningAgent, parent_b: QLearningAgent, rng: np.random.Generator,
              name: str, seed: int) -> QLearningAgent:
    """Uniform crossover: each cell of the child's table comes from one parent or the other.

    Recombination, never averaging. The mean of two tables is a genome neither parent had and
    that selection has never evaluated, which would make the child's fitness say nothing about
    either lineage. Same rule, and the same refusal on mismatched shapes, as the Markov brain's.
    """
    if parent_a.Q.shape != parent_b.Q.shape:
        raise ValueError(
            f"crossover requires parents with identical genome shape; got {parent_a.Q.shape} "
            f"and {parent_b.Q.shape}. A Q-table is indexed by the game's state and action "
            f"counts, so two parents of different shape were evaluated on different games"
        )
    Q = np.where(rng.random(parent_a.Q.shape) < 0.5, parent_a.Q, parent_b.Q)
    return parent_a._respawn(Q, name, seed)


def spawn(name: str, game: Any, seed: int, config: Any) -> QLearningAgent:
    """Build one random tabular individual for `engine.evolution.evolve`.

    Two decisions live here rather than in the engine, because only this module knows that its
    genome has a learning rule at all.

    **The table starts random, not at zero.** Zero is where Q-learning starts, and it is the one
    initialisation a GA cannot use: a population of identical genomes gives selection nothing to
    act on, so the first generations would drift on mutation alone.

    **`lifetime_learning` decides what is being searched.** Under the default the individual acts
    greedily on the table selection handed it and discards what the match would teach, so its
    fitness is a property of the genome. Under `"lamarckian"` it learns during evaluation and
    passes that on, which is a different experiment and has to be asked for by name.
    """
    darwinian = getattr(config, "lifetime_learning", "off") != "lamarckian"
    knobs = getattr(config, "genome", None) or {}
    agent = QLearningAgent(
        name, n_states=game.n_states, n_actions=game.n_actions, seed=seed,
        # Greedy under the Darwinian reading: exploration noise during evaluation is noise in the
        # fitness of a genome that does not change, which costs selection pressure and buys
        # nothing, since there is no learning for it to feed.
        epsilon=0.0 if darwinian else float(knobs.get("epsilon", 0.1)),
        epsilon_min=0.0 if darwinian else float(knobs.get("epsilon_min", 0.01)),
        alpha=float(knobs.get("alpha", 0.1)), gamma=float(knobs.get("gamma", 0.95)),
        lifetime_learning=not darwinian, learning_rule="evolved",
    )
    agent.Q = np.random.default_rng(seed).normal(
        scale=float(knobs.get("init_scale", 1.0)), size=(game.n_states, game.n_actions))
    return agent
