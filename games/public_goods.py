"""
N-player Public Goods Game (PGG) — the standard generalization of the Prisoner's Dilemma.

Each round every one of the n players either Cooperates (contributes a cost `c` to a common pool)
or Defects (contributes nothing). The pool is multiplied by a factor r and split equally among all
n players, so with k cooperators:

    payoff_i = mpcr * k * c - (c if i cooperated else 0),   where mpcr = r / n.

It is a genuine social dilemma iff  1/n < mpcr < 1:
  - mpcr < 1  -> a lone contributor loses money (defection dominates individually);
  - mpcr > 1/n (i.e. r > 1) -> universal cooperation is Pareto-superior to universal defection.
At n = 2 this reduces to the classic Prisoner's Dilemma. The well-known result — rational/selfish
play collapses to universal defection (free-riding) — is our built-in validation target.

Observation given to every agent each round: the number of cooperators in the *previous* round,
k_prev in {0..n}; before the first round a distinct "start" index is used. Hence n_states = n + 2.
This is public information (full information visibility) and keeps a tabular brain's Q-table tiny
and human-readable.
"""

from __future__ import annotations

from ..engine.game import Game, StepResult

DEFECT = 0
COOPERATE = 1


class PublicGoodsGame(Game):
    name = "public_goods"
    n_actions = 2
    action_names = ["Defect", "Cooperate"]
    #: The observation is last round's cooperator count, with `n_agents + 1` for the first round,
    #: so reciprocating strategies may read it (see engine/game.py's vocabulary).
    observation_kind = "concede_count"
    #: Contributing is the action that gives up your own endowment for the group, so it is the
    #: `concede` side of the universal axis; `cooperate`/`defect` are this family's own words for
    #: the same two actions.
    action_roles = {"concede": COOPERATE, "claim": DEFECT,
                    "cooperate": COOPERATE, "defect": DEFECT}

    def __init__(
        self,
        n_agents: int,
        rounds: int = 2000,
        mpcr: float | None = None,
        cost: float = 1.0,
    ) -> None:
        if n_agents < 2:
            raise ValueError("Public Goods Game needs n_agents >= 2.")
        if mpcr is None:
            # Default: a clean social-dilemma value. For n>=3, 0.5 satisfies 1/n < 0.5 < 1;
            # at n=2, 1/n == 0.5 so we bump above it.
            mpcr = 0.5 if n_agents >= 3 else 0.75
        if not (1.0 / n_agents < mpcr < 1.0):
            raise ValueError(
                f"mpcr={mpcr} violates the social-dilemma condition 1/n < mpcr < 1 "
                f"(1/n = {1.0 / n_agents:.3f}) for n={n_agents}."
            )

        self.n_agents = n_agents
        self.rounds = rounds
        self.mpcr = mpcr
        self.cost = cost

        # Observation indices: 0..n encode "k_prev cooperators", n+1 is the start marker.
        self.start_state = n_agents + 1
        self.n_states = n_agents + 2

        self._round = 0

    def reset(self) -> list[int]:
        self._round = 0
        return [self.start_state] * self.n_agents

    def step(self, actions: list[int]) -> StepResult:
        if len(actions) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions, got {len(actions)}")

        k = int(sum(1 for a in actions if a == COOPERATE))
        pool_share = self.mpcr * k * self.cost
        rewards = [pool_share - (self.cost if a == COOPERATE else 0.0) for a in actions]

        self._round += 1
        done = self._round >= self.rounds

        # Everyone observes this round's cooperator count next round (public info).
        next_obs = [k] * self.n_agents
        info = {
            "cooperators": k,
            "coop_fraction": k / self.n_agents,
            "round": self._round,
        }
        return StepResult(observations=next_obs, rewards=rewards, done=done, info=info)

    # --- helpers for nicer visualization / metrics (game-specific, optional) ---

    def state_labels(self) -> list[str]:
        labels = [f"k={i}" for i in range(self.n_agents + 1)]
        labels.append("start")
        return labels

    def max_welfare_per_round(self) -> float:
        """Total social welfare when everyone cooperates (k = n): n * c * (r - 1)."""
        return self.n_agents * self.cost * (self.mpcr * self.n_agents - 1.0)

    def describe(self) -> dict:
        d = super().describe()
        d.update({"rounds": self.rounds, "mpcr": self.mpcr, "cost": self.cost})
        return d
