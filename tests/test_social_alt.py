"""
Tests for metrics/social_alt.py, checked against the source papers' own worked examples rather
than against whatever this implementation happens to produce.

Four independent oracles, all taken from the papers or their code:

  CALT on a hand-built window     the Gap paper's 5 * 0.5625 / 6 = 0.469
  ABABABAB at n=2                 perfect alternation, RS = WPE = RP = 1.00 exactly
  ABBAABBA at n=2                 clumped, RS 0.78 / WPE 1.00 / RP 0.89
  ABCABC at n=3                   perfect but RS 0.89, the disclosed finite-boundary edge effect

The last one matters most: a naive implementation that skips the boundary waiting periods scores
it 1.00 and looks *more* correct. It is not. Ignoring the wait before an agent's first win and
after its last is what hid the real failure mode in the source data, where Q-learning agents stop
winning partway through and never recover.
"""

from gamebrains.metrics.social_alt import (
    alt_metrics, compute_all, coordination_score, periods_boundary_inclusive,
    rp_metrics, traditional_metrics, win_lists,
)


def _vectors(pattern: str, n_agents: int) -> list[list[int]]:
    """Turn "AB{BC}" style shorthand into per-episode arrival flag vectors."""
    episodes, i = [], 0
    while i < len(pattern):
        if pattern[i] == "{":
            end = pattern.index("}", i)
            names, i = pattern[i + 1:end], end + 1
        else:
            names, i = pattern[i], i + 1
        episodes.append([1 if chr(ord("A") + a) in names else 0 for a in range(n_agents)])
    return episodes


def test_calt_matches_the_gap_papers_worked_example():
    """One window of 3 episodes with 4 arrivals, all three agents reaching at least once:
    FALT = 3/4 = 0.75, so CALT = 5 * 0.75^2 / (3*2) = 0.469."""
    window = _vectors("{AB}CA", 3)
    assert [sum(v) for v in window] == [2, 1, 1]      # 4 arrivals

    metrics = alt_metrics(window, n_agents=3)
    assert abs(metrics["FALT"] - 0.75) < 1e-9
    assert abs(metrics["qFALT"] - 0.5625) < 1e-9
    assert abs(metrics["CALT"] - 5 * 0.5625 / 6) < 1e-9
    assert abs(metrics["CALT"] - 0.46875) < 1e-9


def test_perfect_alternation_scores_exactly_one_at_n2():
    episodes = _vectors("ABABABAB", 2)
    rp = rp_metrics(win_lists(episodes, 2, exclusive=True), n_agents=2, n_episodes=8)
    assert abs(rp["RS"] - 1.0) < 1e-9
    assert abs(rp["WPE"] - 1.0) < 1e-9
    assert abs(rp["RP"] - 1.0) < 1e-9


def test_clumping_is_caught_by_rhythm_but_invisible_to_frequency():
    """ABBAABBA gives every agent the same number of wins as ABABABAB, so the frequency
    sub-measure cannot tell them apart at all. Only rhythm can."""
    episodes = _vectors("ABBAABBA", 2)
    rp = rp_metrics(win_lists(episodes, 2, exclusive=True), n_agents=2, n_episodes=8)
    assert abs(rp["WPE"] - 1.00) < 1e-9        # identical win counts, so nothing to see
    assert abs(rp["RS"] - 0.775) < 1e-9        # rhythm drops
    assert abs(rp["RP"] - 0.8875) < 1e-9


def test_the_finite_boundary_edge_effect_is_not_hidden():
    """ABCABC is perfect alternation, yet RS is 0.89 rather than 1.00. The middle agent's wins do
    not bookend the run, so it picks up a leading and a trailing waiting period. This is a real,
    disclosed finite-length effect that vanishes as episodes grow, and an implementation that
    scored it 1.00 would be the one that had dropped boundary counting."""
    episodes = _vectors("ABCABC", 3)
    rp = rp_metrics(win_lists(episodes, 3, exclusive=True), n_agents=3, n_episodes=6)
    assert abs(rp["RS"] - 8 / 9) < 1e-9        # 0.888..., agents score 1.0 / 0.667 / 1.0
    assert abs(rp["WPE"] - 1.0) < 1e-9         # two wins each, exactly the fair share


def test_boundary_periods_are_counted_at_both_ends():
    # Wins at 2 and 4 of an 8-episode run: 2 before the first, 1 between, 3 after the last.
    assert periods_boundary_inclusive([2, 4], 8) == [2, 1, 3]
    assert periods_boundary_inclusive([0, 2], 3) == [1]          # bookended, no boundary periods
    assert periods_boundary_inclusive([], 5) == [5]              # never won: the whole run waits


def test_exclusive_and_reach_disagree_when_an_agent_only_ever_ties():
    """Agent B reaches in every single episode but never once alone. Under one definition it is
    the most active agent on the board and never waits at all; under the other it waits the entire
    run. Same record, opposite readings, which is why both are always reported."""
    episodes = _vectors("{AB}{AB}{AB}{AB}", 2)

    reach = win_lists(episodes, 2, exclusive=False)
    excl = win_lists(episodes, 2, exclusive=True)
    assert reach[1] == [0, 1, 2, 3]      # present every time
    assert excl[1] == []                 # yet holds nothing

    assert rp_metrics(reach, 2, 4)["avg_wait"] == 0.0    # never waits for a reach
    assert rp_metrics(excl, 2, 4)["avg_wait"] == 4.0     # waits the whole run for a solo win


def test_traditional_measures_look_healthy_while_coordination_has_collapsed():
    """The finding this whole module exists for. Two agents collide in every single episode and
    never once take turns, splitting the reduced share equally. Reward Fairness is a perfect 1.0,
    turn-taking fairness is a perfect 1.0, and every alternation measure that depends on winning
    alone is zero."""
    episodes = _vectors("{AB}{AB}{AB}{AB}", 2)
    totals = [50.0 * 4, 50.0 * 4]              # identical takings, split every time

    traditional = traditional_metrics(episodes, 2, totals, full_reward=100.0)
    assert traditional["reward_fairness"] == 1.0
    assert traditional["tt_fairness"] == 1.0
    assert traditional["fairness"] == 0.0      # the one fairness measure that notices
    assert traditional["efficiency"] == 1.0

    alt = alt_metrics(episodes, n_agents=2)
    assert alt["EALT"] == 0.0                  # no episode was ever won alone
    assert alt["AALT"] == 0.0
    assert alt["CALT"] == 0.0


def test_efficiency_and_the_three_fairness_measures_are_separate_quantities():
    # A reaches 3 times and wins 2 alone; B reaches once, alone; rewards deliberately lopsided.
    episodes = _vectors("A{AB}AB", 2)
    traditional = traditional_metrics(episodes, 2, [90.0, 30.0], full_reward=100.0)

    assert abs(traditional["efficiency"] - 120.0 / 400.0) < 1e-9
    assert abs(traditional["reward_fairness"] - 30.0 / 90.0) < 1e-9   # min/max of rewards
    assert abs(traditional["tt_fairness"] - 2.0 / 3.0) < 1e-9         # reaches: A 3, B 2
    assert abs(traditional["fairness"] - 1.0 / 2.0) < 1e-9            # solo wins: A 2, B 1


def test_coordination_score_reports_worse_than_random_as_negative():
    assert abs(coordination_score(0.8, 0.5, perfect=1.0) - 0.6) < 1e-9   # better than random
    assert coordination_score(0.5, 0.5, perfect=1.0) == 0.0              # indistinguishable
    assert coordination_score(0.2, 0.5, perfect=1.0) < 0.0               # the papers' headline
    # Not clamped: a value far below random legitimately exceeds -100%.
    assert abs(coordination_score(0.0, 0.6, perfect=1.0) - (-1.5)) < 1e-9


def test_compute_all_reads_the_runners_episode_records():
    episodes = [
        {"top_agents": [1, 0, 0], "rewards": [100.0, 0.0, 0.0]},
        {"top_agents": [0, 1, 0], "rewards": [0.0, 100.0, 0.0]},
        {"top_agents": [0, 0, 1], "rewards": [0.0, 0.0, 100.0]},
        {"top_agents": [1, 1, 0], "rewards": [33.3, 33.3, 0.0]},
    ]
    out = compute_all(episodes, n_agents=3, full_reward=100.0)

    assert out["n_episodes"] == 4
    assert out["solo_win_episodes"] == 3
    assert out["collision_episodes"] == 1
    assert out["empty_episodes"] == 0
    # Both flavours present for every RP measure, which is the point of not picking a winner.
    for key in ("RS", "WPE", "RP", "AWE", "avg_wait"):
        assert f"{key}_reach" in out and f"{key}_excl" in out
    assert {"FALT", "EALT", "qFALT", "qEALT", "CALT", "AALT"} <= set(out)
    assert {"efficiency", "reward_fairness", "tt_fairness", "fairness"} <= set(out)


def test_a_single_stage_game_yields_no_alternation_measures():
    """One episode for the whole match means there is no sequence to alternate over, and saying
    so beats reporting a confident zero."""
    assert compute_all([{"rounds": 40, "rewards": [1.0, 2.0]}], n_agents=2)["n_episodes"] == 0


if __name__ == "__main__":
    test_calt_matches_the_gap_papers_worked_example()
    print("OK: CALT matches the Gap paper's worked example (0.469)")
    test_perfect_alternation_scores_exactly_one_at_n2()
    print("OK: perfect alternation scores exactly 1.00 at n=2")
    test_clumping_is_caught_by_rhythm_but_invisible_to_frequency()
    print("OK: clumping is caught by rhythm and invisible to frequency")
    test_the_finite_boundary_edge_effect_is_not_hidden()
    print("OK: the finite-boundary edge effect is reported, not hidden")
    test_boundary_periods_are_counted_at_both_ends()
    print("OK: waiting periods are counted at both ends of the run")
    test_exclusive_and_reach_disagree_when_an_agent_only_ever_ties()
    print("OK: reach and exclusive disagree when an agent only ever ties")
    test_traditional_measures_look_healthy_while_coordination_has_collapsed()
    print("OK: traditional measures look healthy while coordination has collapsed")
    test_efficiency_and_the_three_fairness_measures_are_separate_quantities()
    print("OK: efficiency and the three fairness measures are separate quantities")
    test_coordination_score_reports_worse_than_random_as_negative()
    print("OK: the coordination score reports worse-than-random as negative")
    test_compute_all_reads_the_runners_episode_records()
    print("OK: compute_all reads the runner's episode records")
    test_a_single_stage_game_yields_no_alternation_measures()
    print("OK: a single-stage game yields no alternation measures")
