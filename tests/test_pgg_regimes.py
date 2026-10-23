"""The multiplier crossing the player count, and why that is a decision rather than a parameter.

`games/public_goods.py` refuses any `mpcr` outside `1/n < mpcr < 1`, which is the condition that
makes the game a genuine social dilemma. That refusal is correct for every experiment the platform
has run so far, and it is also the reason the game cannot currently express the one question a
collaborator asked about: what happens as the multiplier `r = mpcr * n` crosses `n`, where
defection stops dominating and contributing starts to.

So the bound is kept and an opt-in is added beside it. The argument for an opt-in rather than a
wider bound is that the two are not the same game. Below the crossing, cooperation is what the
agents have to solve for; above it, cooperation is what a payoff-maximiser does without solving
anything, and a result reported without saying which side it came from is not interpretable. A
caller who passes `allow_non_dilemma=True` has said which one they mean.

**What is deliberately still refused.** `r <= 1` stays out even with the opt-in. There the pool
returns less than it costs to fill, universal contribution is worse than universal defection, and
`max_welfare_per_round` -- the denominator of the efficiency metric -- goes negative. An
efficiency reading against a negative denominator is not a worse number, it is a meaningless one,
so the constructor keeps that case away from the metrics rather than letting it through and hoping
nobody divides.

**What must not change.** `describe()` feeds `config_hash` whole (see `repository/record.py`'s
`_config_game_desc`), so a new key there would give every Public Goods record ever written a
different hash and silently cut the lineage of all of them. The regime is therefore derived from
`mpcr` and `n_agents` on demand and never stored: it is not new information, so it does not belong
in the identity of a run. The last test here is what holds that line.

Run:  "$GB/gamebrains/.venv/Scripts/python.exe" -m gamebrains.tests.test_pgg_regimes
"""

from __future__ import annotations

import sys

from ..games.public_goods import COOPERATE, DEFECT, PublicGoodsGame
from ..metrics.equilibrium import pure_nash_equilibria


def test_the_dilemma_bound_still_refuses_by_default() -> None:
    """Nothing that worked before changes, and the refusal says how to ask for the other regime."""
    try:
        PublicGoodsGame(n_agents=4, rounds=10, mpcr=1.25)
    except ValueError as exc:
        assert "allow_non_dilemma" in str(exc), (
            f"the refusal does not name the way past it, so a caller who has a reason to cross "
            f"the player count has to read the source to find out it is possible: {exc}")
        return
    raise AssertionError(
        "mpcr=1.25 was accepted without the opt-in, so the default is no longer a social dilemma "
        "and every existing experiment's premise has quietly widened")


def test_the_opt_in_admits_the_regime_where_the_equilibrium_flips() -> None:
    game = PublicGoodsGame(n_agents=4, rounds=10, mpcr=1.25, allow_non_dilemma=True)
    assert game.mpcr == 1.25
    assert game.multiplier == 5.0, f"r should be mpcr*n = 5.0, got {game.multiplier}"
    print("OK: with the opt-in, n=4 at r=5 is constructible")


def test_the_equilibrium_actually_flips_at_the_player_count() -> None:
    """The claim the collaboration rests on, checked by the solver rather than asserted in prose."""
    below = PublicGoodsGame(n_agents=4, rounds=10, mpcr=0.5)
    assert pure_nash_equilibria(below) == [tuple([DEFECT] * 4)], (
        "below the crossing the unique pure equilibrium must still be universal defection")

    above = PublicGoodsGame(n_agents=4, rounds=10, mpcr=1.25, allow_non_dilemma=True)
    assert pure_nash_equilibria(above) == [tuple([COOPERATE] * 4)], (
        "above the crossing contributing dominates, so universal contribution must be the unique "
        "pure equilibrium. If this fails the flip is not happening and the experiment built on it "
        "would be measuring nothing")
    print("OK: r=2 gives all-Defect, r=5 gives all-Cooperate, both from the solver")


def test_at_exactly_r_equals_n_nobody_is_pushed_either_way() -> None:
    """The crossing itself: a contribution returns exactly what it cost, so every profile is an
    equilibrium. This is the point an experiment sweeping the multiplier passes through, and a
    reading taken there says nothing about the agents."""
    n = 3
    edge = PublicGoodsGame(n_agents=n, rounds=10, mpcr=1.0, allow_non_dilemma=True)
    equilibria = pure_nash_equilibria(edge)
    assert len(equilibria) == 2 ** n, (
        f"at r=n every one of the {2 ** n} profiles should be an equilibrium, since no player can "
        f"gain by moving; the solver found {len(equilibria)}")
    print(f"OK: at r=n all {2 ** n} profiles are equilibria, so the crossing is indifference")


def test_a_pool_that_destroys_value_is_refused_even_with_the_opt_in() -> None:
    """`r <= 1` would make the efficiency denominator negative, so it stays out of reach."""
    for mpcr in (0.2, 1.0 / 4):          # r = 0.8 and r = 1.0 at n = 4
        try:
            PublicGoodsGame(n_agents=4, rounds=10, mpcr=mpcr, allow_non_dilemma=True)
        except ValueError:
            continue
        raise AssertionError(
            f"mpcr={mpcr} (r={mpcr * 4}) was accepted, so max_welfare_per_round can reach zero or "
            f"below and every efficiency reading taken against it is meaningless")
    print("OK: r<=1 is refused with the opt-in too, so the efficiency denominator stays positive")


def test_welfare_stays_positive_wherever_construction_succeeds() -> None:
    """Guards the guard above: the refusal is only worth having if what it admits is sound."""
    for n, mpcr in ((4, 0.5), (4, 1.25), (3, 1.0), (5, 2.0), (2, 0.75)):
        game = PublicGoodsGame(n_agents=n, rounds=10, mpcr=mpcr, allow_non_dilemma=True)
        assert game.max_welfare_per_round() > 0.0, (
            f"n={n}, mpcr={mpcr} constructs but offers welfare "
            f"{game.max_welfare_per_round()}, which cannot serve as an efficiency denominator")
    print("OK: every constructible configuration has positive welfare at full contribution")


def test_the_regime_is_named_and_matches_the_solver() -> None:
    cases = [
        (PublicGoodsGame(n_agents=4, rounds=10, mpcr=0.5), "social_dilemma"),
        (PublicGoodsGame(n_agents=4, rounds=10, mpcr=1.0, allow_non_dilemma=True), "knife_edge"),
        (PublicGoodsGame(n_agents=4, rounds=10, mpcr=1.25, allow_non_dilemma=True),
         "cooperation_dominant"),
    ]
    for game, expected in cases:
        assert game.regime == expected, f"mpcr={game.mpcr} named {game.regime}, not {expected}"
    print("OK: the three regimes are named from mpcr and n, with no stored field")


def test_the_recorded_design_is_unchanged_so_old_records_still_match() -> None:
    """The line that protects every Public Goods record already in the ledger.

    `describe()` is hashed whole into `config_hash`. If the regime or the opt-in appeared there,
    a rerun of an old design would produce a different hash, `replicates` would stop firing, and
    the Smart Filter would offer two halves of one experiment as unrelated work. The regime is
    derivable from fields already present, so storing it would add a key without adding a fact.
    """
    described = PublicGoodsGame(n_agents=4, rounds=10, mpcr=0.5).describe()
    assert set(described) == {"name", "n_agents", "n_actions", "n_states", "action_names",
                              "action_roles", "observation_kind", "rounds", "mpcr", "cost"}, (
        f"describe() now carries {sorted(described)}. Any key added or removed here changes "
        f"config_hash for every Public Goods record ever written and cuts their lineage")
    print("OK: describe() is byte-for-byte the same design dict as before")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}\n     {exc}")
    print(f"\n{'all public-goods regime tests passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
