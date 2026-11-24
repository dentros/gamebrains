"""
A symmetric two-player, two-action game given by its four payoffs, repeated.

    rows: my action, columns: the other's        cooperate    defect
    cooperate                                    R, R         S, T
    defect                                       T, S         P, P

T is what defecting against a cooperator pays, R mutual cooperation, P mutual defection and S
cooperating against a defector. The preset `pd` is T = 5, R = 3, P = 1, S = 0, the Prisoner's
Dilemma in the form most of the experimental literature uses. Any four finite numbers are
accepted.

The observation is the number of players who cooperated in the previous round, 0 to 2, with 3 before
the first round, exactly as the Public Goods Game publishes it, so every agent that reads a count
reads this one. Cooperating is action 0 and defecting action 1, the reverse of the Public Goods
Game, which costs nothing for an agent that resolves its roles and exposes one that does not.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Optional, Sequence

from ..engine.game import Game, StepResult

COOPERATE = 0
DEFECT = 1

#: Named payoff sets. A preset is a convenience, not a claim about the game's family.
PRESETS: dict[str, dict[str, float]] = {
    "pd": {"T": 5.0, "R": 3.0, "P": 1.0, "S": 0.0},
}


class MatrixGame(Game):
    name = "matrix_game"
    n_actions = 2
    action_names = ["Cooperate", "Defect"]
    observation_kind = "concede_count"
    action_roles = {"concede": COOPERATE, "claim": DEFECT,
                    "cooperate": COOPERATE, "defect": DEFECT}

    def __init__(self, T: float, R: float, P: float, S: float, rounds: int = 100,
                 n_agents: int = 2, action_names: Optional[Sequence[str]] = None) -> None:
        if n_agents != 2:
            raise ValueError(f"MatrixGame is a two-player game, got n_agents={n_agents}")
        if action_names is not None:
            names = [str(a).strip() for a in action_names]
            if len(names) != 2 or not all(names) or names[0] == names[1] \
                    or any(len(a) > 40 for a in names):
                raise ValueError(f"action_names must be two different non-empty names of at most "
                                 f"40 characters, the cooperative one first, got {action_names!r}")
            # An instance attribute, so describe() and the language-model prompts use these.
            self.action_names = names
        for label, value in (("T", T), ("R", R), ("P", P), ("S", S)):
            if isinstance(value, bool) or not isinstance(value, numbers.Real) \
                    or not math.isfinite(float(value)):
                raise ValueError(f"payoff {label} must be a finite number, got {value!r}")
        if isinstance(rounds, bool) or not isinstance(rounds, numbers.Integral) or rounds < 1:
            raise ValueError(f"rounds must be a positive integer, got {rounds!r}")
        self.n_agents = 2
        self.rounds = int(rounds)
        self.T, self.R, self.P, self.S = float(T), float(R), float(P), float(S)
        self.start_state = 3
        self.n_states = 4
        self._round = 0

    @property
    def payoffs(self) -> dict[str, float]:
        return {"T": self.T, "R": self.R, "P": self.P, "S": self.S}

    def payoff(self, mine: int, theirs: int) -> float:
        """What `mine` pays against `theirs`, both action indices."""
        if mine == COOPERATE:
            return self.R if theirs == COOPERATE else self.S
        return self.T if theirs == COOPERATE else self.P

    def reset(self) -> list[int]:
        self._round = 0
        return [self.start_state] * 2

    def step(self, actions: list[int]) -> StepResult:
        if len(actions) != 2:
            raise ValueError(f"expected 2 actions, got {len(actions)}")
        for a in actions:
            if a not in (COOPERATE, DEFECT):
                raise ValueError(f"action must be 0 or 1, got {a!r}")
        a, b = actions
        rewards = [self.payoff(a, b), self.payoff(b, a)]
        k = sum(1 for x in actions if x == COOPERATE)
        self._round += 1
        info = {"cooperators": k, "coop_fraction": k / 2, "round": self._round}
        return StepResult(observations=[k, k], rewards=rewards,
                          done=self._round >= self.rounds, info=info)

    def state_labels(self) -> list[str]:
        return ["k=0", "k=1", "k=2", "start"]

    def max_welfare_per_round(self) -> float:
        """The most the two can earn together in a round, whichever profile gives it."""
        return max(2 * self.R, self.T + self.S, 2 * self.P)

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({"rounds": self.rounds, **self.payoffs})
        return d


def from_preset(key: str, rounds: int = 100, **overrides: float) -> MatrixGame:
    if key not in PRESETS:
        raise ValueError(f"unknown preset {key!r}; expected one of {sorted(PRESETS)}")
    return MatrixGame(rounds=rounds, **{**PRESETS[key], **overrides})
