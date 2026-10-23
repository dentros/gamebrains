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

**Above the crossing is a different game, and asking for it is explicit.** At r = n a contribution
returns exactly what it cost, so nobody is pushed either way and every action profile is an
equilibrium; above it contributing dominates and universal contribution is the unique one. That
regime is reachable with `allow_non_dilemma=True`, never by default, because a cooperation rate
reported without saying which side of the crossing it came from is uninterpretable: below, it is
something the agents had to solve for; above, it is what a payoff-maximiser does without solving
anything. `regime` names the three cases and is derived rather than stored, since `mpcr` and
`n_agents` already identify the design and `describe()` feeds `config_hash` whole.

`r <= 1` stays refused even with that flag. There universal contribution is worse than universal
defection, `max_welfare_per_round` is not positive, and it is the denominator of every efficiency
reading the platform takes.

Observation given to every agent each round: the number of cooperators in the *previous* round,
k_prev in {0..n}; before the first round a distinct "start" index is used. Hence n_states = n + 2.
This is public information (full information visibility) and keeps a tabular brain's Q-table tiny
and human-readable.
"""

from __future__ import annotations

import math

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
        allow_non_dilemma: bool = False,
    ) -> None:
        if n_agents < 2:
            raise ValueError("Public Goods Game needs n_agents >= 2.")
        if mpcr is None:
            # Default: a clean social-dilemma value. For n>=3, 0.5 satisfies 1/n < 0.5 < 1;
            # at n=2, 1/n == 0.5 so we bump above it.
            mpcr = 0.5 if n_agents >= 3 else 0.75
        if allow_non_dilemma:
            # The upper bound lifts; the lower one does not. Above 1/n the pool returns more than
            # it cost to fill, so full contribution is still the welfare maximum and the
            # efficiency denominator stays positive. At or below it that denominator reaches zero
            # and then turns negative, which does not make efficiency a worse number -- it makes
            # every reading of it meaningless, including the ones already in the ledger.
            if mpcr <= 1.0 / n_agents:
                raise ValueError(
                    f"mpcr={mpcr} puts the multiplier r={mpcr * n_agents:.3f} at or below 1, so "
                    f"contributing destroys value and welfare at full contribution is not "
                    f"positive. allow_non_dilemma lifts the upper bound on mpcr, not this one."
                )
        elif not (1.0 / n_agents < mpcr < 1.0):
            raise ValueError(
                f"mpcr={mpcr} violates the social-dilemma condition 1/n < mpcr < 1 "
                f"(1/n = {1.0 / n_agents:.3f}) for n={n_agents}. Pass allow_non_dilemma=True for "
                f"the regime where r = mpcr*n reaches or crosses n and contributing stops being "
                f"dominated; that is a different game, which is what the flag is for saying."
            )

        self.n_agents = n_agents
        self.rounds = rounds
        self.mpcr = mpcr
        self.cost = cost

        # Observation indices: 0..n encode "k_prev cooperators", n+1 is the start marker.
        self.start_state = n_agents + 1
        self.n_states = n_agents + 2

        self._round = 0

    @property
    def multiplier(self) -> float:
        """The pool multiplier `r`, which is how this game is written everywhere but this file.

        `mpcr = r / n` is the per-capita form, and it is the one stored, because it is the form
        the dominance condition is stated in. A caller thinking in multipliers reads this.
        """
        return self.mpcr * self.n_agents

    @property
    def regime(self) -> str:
        """Which side of `r = n` this configuration sits on: the three cases, named.

        Derived, never stored. `describe()` is hashed whole into `config_hash`
        (`repository/record.py`), and this is a function of `mpcr` and `n_agents`, both already
        there -- so recording it would add a key without adding a fact, and would hand every
        Public Goods record ever written a new hash, cutting the lineage of all of them.
        """
        if math.isclose(self.mpcr, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            # A contribution returns exactly its own cost, so no player is pushed either way and
            # every profile is an equilibrium. A reading taken here describes the configuration,
            # not the agents.
            return "knife_edge"
        return "social_dilemma" if self.mpcr < 1.0 else "cooperation_dominant"

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
