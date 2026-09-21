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
    #: How this agent relates to what the action indices *mean*. There is no default, because the
    #: whole point is that the author states it rather than inherits it:
    #:
    #:   "index-agnostic"  learns or acts over indices without assigning them meaning, so any
    #:                     game is safe (tabular Q-learning, a DQN, an evolved animat)
    #:   "role-bound"      its behaviour depends on what the indices mean, so it MUST resolve
    #:                     them against the game in `on_match_start`, via `game.action_for` or
    #:                     `game.require_observation_kind`
    #:
    #: `engine.semantics.bind_agents` refuses an agent that leaves this unset, and refuses a
    #: "role-bound" agent that never actually asked. That is the difference between a documented
    #: convention and an enforced contract: a convention catches the lapse when someone rereads
    #: the code, a contract catches it before the first round is played.
    semantics: str | None = None

    #: What this agent conditions on when it acts, which is not the same question as how it learns
    #: and is easy to confuse with it. Agents in one match can differ here by more than they differ
    #: in architecture, and a comparison that does not hold it fixed is partly a comparison of who
    #: was allowed to see what:
    #:
    #:   "none"                  ignores the observation entirely (a constant strategy)
    #:   "observation"           the game's current declared observation and nothing else, so the
    #:                           agent is Markov in the state the game publishes
    #:   "observation+memory"    plus internal state summarising everything before it, as a belief
    #:                           or a recurrent hidden state, never the raw past
    #:   "observation+history"   plus a window of past rounds verbatim; the window length is a
    #:                           parameter and belongs in the record
    #:
    #: Declared rather than inferred, for the reason `semantics` is: the platform can then record
    #: it, and an experiment can hold it constant on purpose rather than by accident.
    information: str = "observation"

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

    def on_match_start(self, game: Any) -> None:
        """Hook called once before a match begins, with the game about to be played.

        This is where a strategy defined in meaning rather than in numbers resolves itself: ask
        `game.action_for("concede")` for the index that plays a role here, or
        `game.require_observation_kind(...)` to refuse a game whose observations you cannot read.
        Doing it here rather than in `__init__` keeps agents constructible without a game, and
        means the check runs at the moment the pairing actually happens.

        No-op by default, for agents that work purely off indices they were told about.
        """
        return None

    def on_match_end(self) -> None:
        """Hook called once when a match finishes (e.g. decay schedules, bookkeeping)."""
        return None

    def reproducibility(self) -> str:
        """What a rerun of the same configuration would give back, as one of three words.

        Declared rather than inferred, for the same reason as `semantics` above: a property nobody
        states is a property nobody can check. Naming a tier rather than answering yes or no is
        what lets the platform say something true about a component whose output is nearly, but
        not exactly, the same every time.

          ``byte``      the same event log down to the byte, given the seed. The default, and true
                        in fact for every agent that draws from a generator the runner seeded.
          ``replay``    the same event log, because the decisions are read back from a recording
                        rather than recomputed. The promise is about the recording, not the
                        component, and a rerun that wanders off the recording fails loudly.
          ``none``      neither. A language model behind a server is the case this exists for:
                        temperature zero, a pinned seed, a reload and a single thread all leave
                        repeated requests returning different text some of the time.

        `repository/record.py` records the weakest tier in the roster on the run itself, so the
        claim is withheld for that run instead of quietly weakening for every run.
        """
        return "byte"

    def reproducibility_note(self) -> str:
        """Why the tier is not `byte`, in a sentence a stored record can carry."""
        return ""
