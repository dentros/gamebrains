"""
Tests for the Public Goods Game's `mpcr_schedule` (games/public_goods.py): a return that changes at
a fixed round, so that one match can cross from the dilemma into the regime where contributing
dominates.

The first case is the guarantee the option was added under: a game built without a schedule plays,
logs and hashes exactly as before. The rest pin the switch itself, its place in the recorded design,
and the refusals.
"""

from gamebrains.games.public_goods import COOPERATE, DEFECT, PublicGoodsGame
from gamebrains.repository.normalize import build_config, hash_config
from gamebrains.repository.record import DEFAULT_CODE_VERSION, _config_game_desc


def _play(game, profile_for_round):
    game.reset()
    out = []
    for t in range(1, game.rounds + 1):
        out.append(game.step(profile_for_round(t)))
    return out


def _hash(game):
    return hash_config(build_config(_config_game_desc(game), [], DEFAULT_CODE_VERSION, {}))


def test_without_a_schedule_nothing_changes():
    game = PublicGoodsGame(n_agents=4, rounds=6, mpcr=0.5)
    assert "mpcr_schedule" not in game.describe()
    profile = [COOPERATE, COOPERATE, DEFECT, DEFECT]
    for result in _play(game, lambda t: profile):
        assert result.rewards == [0.5 * 2 - 1.0, 0.5 * 2 - 1.0, 1.0, 1.0]
        assert "mpcr" not in result.info, "a run without a schedule must log what it always did"
    assert game.max_welfare_per_round() == 4 * (0.5 * 4 - 1.0)


def test_the_return_switches_at_the_scheduled_round():
    game = PublicGoodsGame(n_agents=4, rounds=80, mpcr=0.5, allow_non_dilemma=True,
                           mpcr_schedule=[(41, 1.25)])
    results = _play(game, lambda t: [COOPERATE, DEFECT, DEFECT, DEFECT])
    before, after = results[39], results[40]
    assert before.info["mpcr"] == 0.5 and before.rewards[0] == 0.5 - 1.0
    assert after.info["mpcr"] == 1.25 and after.rewards[0] == 1.25 - 1.0
    assert after.rewards[1] == 1.25, "a defector shares the pool at the new return too"
    assert game.mpcr == 0.5, "mpcr stays the round-1 value, since describe() is read after play"
    assert [game.mpcr_at(t) for t in (1, 40, 41, 80)] == [0.5, 0.5, 1.25, 1.25]


def test_reset_starts_again_from_round_one():
    game = PublicGoodsGame(n_agents=4, rounds=10, mpcr=0.5, allow_non_dilemma=True,
                           mpcr_schedule=[(3, 1.25)])
    _play(game, lambda t: [COOPERATE] * 4)
    game.reset()
    assert game.current_mpcr == 0.5
    assert game.step([COOPERATE] * 4).info["mpcr"] == 0.5


def test_the_schedule_is_part_of_the_design():
    plain = PublicGoodsGame(n_agents=4, rounds=80, mpcr=0.5, allow_non_dilemma=True)
    switched = PublicGoodsGame(n_agents=4, rounds=80, mpcr=0.5, allow_non_dilemma=True,
                               mpcr_schedule=[(41, 1.25)])
    later = PublicGoodsGame(n_agents=4, rounds=80, mpcr=0.5, allow_non_dilemma=True,
                            mpcr_schedule=[(61, 1.25)])
    hashes = {_hash(plain), _hash(switched), _hash(later)}
    assert len(hashes) == 3, "a switch, and when it happens, each make a different experiment"
    assert switched.describe()["mpcr_schedule"] == [[41, 1.25]]


def test_max_welfare_averages_over_the_schedule():
    game = PublicGoodsGame(n_agents=4, rounds=80, mpcr=0.5, allow_non_dilemma=True,
                           mpcr_schedule=[(41, 1.25)])
    expected = (40 * 4 * (0.5 * 4 - 1.0) + 40 * 4 * (1.25 * 4 - 1.0)) / 80
    assert abs(game.max_welfare_per_round() - expected) < 1e-12
    # So efficiency of full contribution over the whole match is exactly 1.
    results = _play(game, lambda t: [COOPERATE] * 4)
    realised = sum(sum(r.rewards) for r in results)
    assert abs(realised / (game.max_welfare_per_round() * 80) - 1.0) < 1e-12


def test_bad_schedules_are_refused_with_the_reason():
    def refused(**kwargs):
        try:
            PublicGoodsGame(n_agents=4, rounds=80, mpcr=0.5, **kwargs)
        except ValueError as exc:
            return str(exc)
        raise AssertionError(f"accepted {kwargs}")

    assert "violates the social-dilemma condition" in refused(mpcr_schedule=[(41, 1.25)])
    assert "after round 1" in refused(mpcr_schedule=[(1, 0.6)])
    assert "after round 1" in refused(mpcr_schedule=[(41, 0.6), (30, 0.7)])
    assert "never be paid" in refused(mpcr_schedule=[(81, 0.6)])
    assert "at or below 1" in refused(allow_non_dilemma=True, mpcr_schedule=[(41, 0.2)])
    assert "pair" in refused(mpcr_schedule=[41])
    assert "integer" in refused(mpcr_schedule=[(41.5, 0.6)])


if __name__ == "__main__":
    test_without_a_schedule_nothing_changes()
    print("OK: without a schedule the game plays, logs and hashes as before")
    test_the_return_switches_at_the_scheduled_round()
    print("OK: the return switches at the scheduled round")
    test_reset_starts_again_from_round_one()
    print("OK: reset starts again from round one")
    test_the_schedule_is_part_of_the_design()
    print("OK: the schedule, and when it switches, are part of the design")
    test_max_welfare_averages_over_the_schedule()
    print("OK: max welfare averages over the schedule, so efficiency stays total over total")
    test_bad_schedules_are_refused_with_the_reason()
    print("OK: bad schedules are refused with the reason")
