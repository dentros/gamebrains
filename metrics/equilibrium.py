"""
Nash equilibrium detection for the Public Goods Game, via `pygambit` (the Gambit project's
Python bindings — see the platform review in the paper). Rather than writing our own
equilibrium solver, we build the game's normal-form payoff tables from its closed-form payoff
function and hand them to Gambit's battle-tested algorithms.

Scope: pure-strategy equilibria for any n (via `enumpure_solve`, which scales to any number of
players), and exact mixed-strategy equilibria for the special case n=2 (via `enummixed_solve`,
which Gambit only supports for two-player games; general n-player mixed equilibria would need the
heavier `enumpoly_solve`/`gnm_solve`/`logit_solve` methods and are out of scope for now).

Note: for the Public Goods Game with mpcr < 1, Defect strictly dominates Cooperate for every
player regardless of the others' actions (raising your own action from C to D always adds `cost`
to your payoff, since the shared mpcr*k term does not depend on your own action). So universal
defection is *always* the unique pure Nash equilibrium whenever the game satisfies our
social-dilemma condition 1/n < mpcr < 1 — this is a theory-grounded check we can validate against.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pygambit as gbt

from ..games.public_goods import COOPERATE, DEFECT, PublicGoodsGame


def build_normal_form(pgg: PublicGoodsGame) -> gbt.Game:
    """Build the pygambit strategic-form game for one stage of a PublicGoodsGame."""
    n = pgg.n_agents
    shape = (2,) * n
    arrays = [np.zeros(shape) for _ in range(n)]

    for joint in product((DEFECT, COOPERATE), repeat=n):
        k = sum(joint)
        pool_share = pgg.mpcr * k * pgg.cost
        for i in range(n):
            arrays[i][joint] = pool_share - (pgg.cost if joint[i] == COOPERATE else 0.0)

    return gbt.Game.from_arrays(*arrays, title=f"PublicGoods(n={n}, mpcr={pgg.mpcr})")


def pure_nash_equilibria(pgg: PublicGoodsGame) -> list[tuple[int, ...]]:
    """Return every pure-strategy Nash equilibrium as a tuple of actions (one per agent)."""
    game = build_normal_form(pgg)
    result = gbt.nash.enumpure_solve(game)

    players = list(game.players)
    equilibria = []
    for eq in result.equilibria:
        profile = tuple(
            int(np.argmax([float(prob) for _strategy, prob in eq[player]]))
            for player in players
        )
        equilibria.append(profile)
    return equilibria


def mixed_nash_equilibria_2p(pgg: PublicGoodsGame) -> list[list[list[float]]]:
    """Exact mixed-strategy Nash equilibria (two-player games only).

    Returns a list of equilibria; each is [player_0_probs, player_1_probs].
    """
    if pgg.n_agents != 2:
        raise ValueError("mixed_nash_equilibria_2p only supports 2-player games "
                          "(pygambit.nash.enummixed_solve is a two-player method)")
    game = build_normal_form(pgg)
    result = gbt.nash.enummixed_solve(game)
    players = list(game.players)
    return [
        [[float(prob) for _strategy, prob in eq[player]] for player in players]
        for eq in result.equilibria
    ]


def describe_equilibria(pgg: PublicGoodsGame, equilibria: list[tuple[int, ...]]) -> str:
    """Human-readable summary, e.g. 'Cooperate/Defect/Defect (payoffs: 0.50, 1.50, 1.50)'."""
    lines = []
    for profile in equilibria:
        names = [pgg.action_names[a] for a in profile]
        k = sum(profile)
        payoffs = [pgg.mpcr * k * pgg.cost - (pgg.cost if a == COOPERATE else 0.0) for a in profile]
        lines.append(f"{'/'.join(names)} (payoffs: {', '.join(f'{p:.2f}' for p in payoffs)})")
    return "\n".join(lines) if lines else "(no equilibria found)"
