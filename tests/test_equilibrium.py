"""
Tests for pygambit-based Nash equilibrium detection on the Public Goods Game.

The Public Goods Game has a clean theoretical prediction whenever mpcr < 1: Defect strictly
dominates Cooperate for every player, so universal defection is *always* the unique pure Nash
equilibrium. We check this holds for several n via pygambit, and cross-check the n=2 exact mixed
equilibrium too (dominance means the unique mixed equilibrium coincides with the pure one).
"""

from gamebrains.games.public_goods import DEFECT, PublicGoodsGame
from gamebrains.metrics.equilibrium import (
    describe_equilibria,
    mixed_nash_equilibria_2p,
    pure_nash_equilibria,
)


def test_universal_defection_is_the_unique_pure_equilibrium():
    for n in (2, 3, 4, 5):
        pgg = PublicGoodsGame(n_agents=n)  # default mpcr satisfies 1/n < mpcr < 1
        equilibria = pure_nash_equilibria(pgg)
        assert equilibria == [tuple([DEFECT] * n)], (
            f"n={n}: expected unique all-Defect equilibrium, got {equilibria}"
        )


def test_two_player_mixed_equilibrium_matches_pure():
    pgg = PublicGoodsGame(n_agents=2, mpcr=0.75)
    mixed = mixed_nash_equilibria_2p(pgg)
    assert len(mixed) == 1
    # Dominance -> the unique mixed equilibrium is degenerate: pure Defect for both.
    for player_probs in mixed[0]:
        assert player_probs[DEFECT] == 1.0
        assert player_probs[1 - DEFECT] == 0.0


if __name__ == "__main__":
    test_universal_defection_is_the_unique_pure_equilibrium()
    print("OK: universal defection confirmed as the unique pure NE for n=2..5")
    test_two_player_mixed_equilibrium_matches_pure()
    print("OK: 2-player mixed equilibrium matches the pure prediction")

    pgg = PublicGoodsGame(n_agents=5)
    print()
    print(f"Public Goods Game (n=5, mpcr={pgg.mpcr}) equilibria:")
    print(describe_equilibria(pgg, pure_nash_equilibria(pgg)))
