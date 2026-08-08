"""
Temporal-fairness metrics: the ALT family, RP, and the traditional measures they are meant to
expose as insufficient.

The point of this whole file is a single finding from the source papers. Traditional outcome
metrics can look excellent while coordination has completely collapsed: Reward Fairness above 0.99
in runs where Q-learning performs *worse than random* on every alternation measure. That happens
because efficiency and fairness are time-averaged. They ask how much each agent ended up with, and
cannot see whether access rotated fairly or a couple of agents simply collided forever. ALT and RP
ask the question the totals cannot: **which agent got access, and when.**

Everything here is a pure function of the per-episode arrival record, so metrics can be added or
recomputed later over any window without rerunning anything.

  Ported from the authors' own code, in two pieces, deliberately:
    ALT family + traditional measures  <- `2. ALT MEASURES TO MEGALO/src/metrics.py`
    RS / WPE / RP                      <- `RP for Journal/synthetic_experiments/common.py`
                                          (extracted from `compute_rp_full_corrected.py`)

  **Do not port RP from `src/metrics.py`.** Its `compute_rp_metrics` averages AWE with WPE, and
  AWE is the deprecated predecessor of RS. AWE returns 0 for any agent whose average wait reaches
  twice the ideal gap, which is what every collapsed run in the source data does, so that RP
  becomes WPE/2 with no rhythm contribution at all. Corrected values run 1.5-2x higher. Note the
  nasty part: on healthy alternation AWE is nonzero and the bug is invisible, so it only shows up
  on exactly the runs whose failure you are trying to measure. Equally, do not port RS from
  `RP for Journal/compute_rs_all_modes.py`, which predates the boundary fix below and roughly
  doubles RS on real data.

  Cite when publishing results computed here:
    Papadopoulos, Freire, Sanchez-Fibla, Psannis, "The Coordination Gap: Multi-Agent Alternation
      Metrics for Temporal Fairness in Repeated Games". arXiv:2603.05789,
      Zenodo 10.5281/zenodo.18528891
    Papadopoulos, Freire, Sanchez-Fibla, Psannis, "Temporal Fair Division in Multi-Agent Systems:
      From Precise Alternation Metrics to Scalable Coordination Proxies". arXiv:2605.14879;
      conference version MDAI 2025, doi:10.1007/978-3-032-03711-4_16

## Reach and exclusive

Every measure here is computed twice, over two different definitions of "a win":

  reach      the agent got to the terminal, ties included
  exclusive  the agent got there alone

The distinction is the whole subject. An agent can reach constantly and never once win alone,
which reads as healthy access by one definition and total failure by the other. The papers report
the whole family rather than picking a winner, so this module does too: no single number is
designated "the" result.

For the ALT family the split is not an option but the axis the variants are built on. FALT and
qFALT count reaches, EALT / qEALT / AALT count exclusive wins, CALT combines them. For RP it is a
caller-side choice, so RS/WPE/RP/AWE each come in `_reach` and `_excl` form.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

#: Measures where a *higher* value is better, for any caller that needs to know the direction.
HIGHER_IS_BETTER = frozenset({
    "efficiency", "reward_fairness", "tt_fairness", "fairness",
    "FALT", "EALT", "qFALT", "qEALT", "CALT", "AALT",
    "RS_reach", "RS_excl", "WPE_reach", "WPE_excl", "RP_reach", "RP_excl",
    "AWE_reach", "AWE_excl",
})


# --- deriving the two win definitions from one record -------------------------------------------

def win_lists(top_agents_per_episode: Sequence[Sequence[int]], n_agents: int,
              exclusive: bool) -> dict[int, list[int]]:
    """Per agent, the episode indices it won, under one of the two definitions.

    `top_agents_per_episode[k]` is a 0/1 flag **per agent**, not a list of winner ids. An episode
    is an exclusive win when `sum(vector) == 1`; using `len(vector) == 1` instead is a real
    mistake that was made once here, and it silently reports every episode as a tie, since that
    length is always n.
    """
    lists: dict[int, list[int]] = {i: [] for i in range(n_agents)}
    for episode, vector in enumerate(top_agents_per_episode):
        if exclusive and sum(vector) != 1:
            continue
        for i in range(n_agents):
            if vector[i] == 1:
                lists[i].append(episode)
    return lists


# --- RP: rhythm (RS) and frequency (WPE) --------------------------------------------------------

def periods_boundary_inclusive(wins: Sequence[int], n_episodes: int) -> list[int]:
    """Waiting periods for one agent, counting the wait before its first win and after its last.

    Two subtleties, both of which broke this once and are load-bearing:

    The internal gap keeps its `next - prev - 1` form (episodes strictly between two wins), which
    is what makes RS exactly 1 under Perfect Alternation. The boundary periods are raw counts and
    are *added* to that; changing the internal form at the same time breaks the RS=1 property.

    Excluding the boundaries entirely, as an earlier version did, hides exactly the failure mode
    that matters most. A Q-learning agent that stops winning partway through and never recovers
    has an idle tail worth a large fraction of the run, and without boundary counting that tail is
    invisible: real data moved from RS 0.663 to 0.334 once it was counted.
    """
    if not wins:
        return [n_episodes]          # never won at all: the entire run is one waiting period
    periods: list[int] = []
    if wins[0] > 0:
        periods.append(wins[0])                       # before the first win
    for j in range(1, len(wins)):
        periods.append(wins[j] - wins[j - 1] - 1)     # strictly between two wins
    if wins[-1] < n_episodes - 1:
        periods.append(n_episodes - 1 - wins[-1])     # after the last win
    return periods or [0]


def rs_score(avg_wait: float, r_star: float) -> float:
    """Rhythm: how close the average gap between an agent's wins is to the ideal `n - 1`."""
    if max(avg_wait, r_star) <= 0:
        return 0.0
    return min(avg_wait, r_star) / max(avg_wait, r_star)


def wpe_score(wins: int, t_star: float) -> float:
    """Frequency: how close an agent's win count is to its fair share `episodes / n`."""
    if t_star <= 0:
        return 0.0
    if wins < 2 * t_star:
        return max(0.0, 1.0 - abs(wins - t_star) / t_star)
    return 0.0


def awe_score(avg_wait: float, r_star: float) -> float:
    """The deprecated predecessor of RS, kept only so its collapse stays visible.

    Reported for comparison, never as part of RP. It falls to exactly 0 once an agent's average
    wait reaches `2 * r_star`, which is what made the old AWE-based RP silently equal WPE/2 on
    every collapsed run: the cliff is hit precisely by the agents whose behaviour you most want
    measured, while well-alternating agents keep a nonzero AWE and hide the problem.
    """
    if r_star <= 0:
        return 0.0
    if avg_wait < 2 * r_star:
        return max(0.0, 1.0 - abs(avg_wait - r_star) / r_star)
    return 0.0


def rp_metrics(wins_by_agent: dict[int, list[int]], n_agents: int,
               n_episodes: int) -> dict[str, Any]:
    """RP and its parts, averaged over agents. RP is the plain mean of RS and WPE.

    The two are fixed by definition rather than fitted, because Perfect Alternation requires
    exactly two things: constant gaps of `n - 1` (rhythm) and exactly `episodes / n` wins each
    (frequency). Weighting them is not tuned to maximise agreement with ALT; a sweep of the weight
    over the papers' own data is flat anyway.
    """
    r_star = n_agents - 1
    t_star = n_episodes / n_agents if n_agents else 0.0

    waits, rs_all, wpe_all, awe_all = [], [], [], []
    for i in range(n_agents):
        wins = sorted(wins_by_agent.get(i, []))
        avg_wait = float(np.mean(periods_boundary_inclusive(wins, n_episodes)))
        waits.append(avg_wait)
        rs_all.append(rs_score(avg_wait, r_star))
        # The raw win-event count, NOT the length of the periods list: those differ by one
        # whenever both boundary periods happen to be zero, which breaks WPE=1 under Perfect
        # Alternation.
        wpe_all.append(wpe_score(len(wins), t_star))
        awe_all.append(awe_score(avg_wait, r_star))

    rs = float(np.mean(rs_all))
    wpe = float(np.mean(wpe_all))
    return {
        "RS": rs, "WPE": wpe, "RP": (rs + wpe) / 2,
        "AWE": float(np.mean(awe_all)), "avg_wait": float(np.mean(waits)),
        "RS_per_agent": rs_all, "WPE_per_agent": wpe_all,
    }


# --- the ALT family ------------------------------------------------------------------------------

def alt_metrics(top_agents_per_episode: Sequence[Sequence[int]], n_agents: int,
                terminal_occurrences_per_episode: Sequence[int] | None = None
                ) -> dict[str, float]:
    """The six ALT variants, each averaged over every sliding window of `n` consecutive episodes.

    Sliding windows, not disjoint blocks. Under the disjoint reading a clumped sequence such as
    A,B,B,A would wrongly count as perfect alternation; the sliding one is the strict definition
    and the one the source code implements.

    Per window, with `f` distinct agents reaching at least once, `t` total arrivals, `w` episodes
    won by somebody alone, and `g` agents with exactly one solo win:

        FALT  = f / t                          reaches, the loosest
        EALT  = (w * f) / n^2                  exclusivity
        qFALT = FALT^2                         squared, not exponentiated
        qEALT = EALT^2
        CALT  = sum_k[(n - Y_k)] * qFALT / [n(n-1)]     the primary measure
        AALT  = g / t                          strictest: *exactly* one solo win

    `g` counts exactly one solo win, not at least one. With `>= 1` the strictness ordering breaks
    empirically (AALT rises above CALT at n=2), which is how that bug was caught.
    """
    n_episodes = len(top_agents_per_episode)
    if n_agents < 2 or n_episodes < n_agents:
        return {k: 0.0 for k in ("FALT", "EALT", "qFALT", "qEALT", "CALT", "AALT")}

    if terminal_occurrences_per_episode is None:
        terminal_occurrences_per_episode = [sum(v) for v in top_agents_per_episode]

    n_windows = n_episodes - (n_agents - 1)
    beta = {k: np.zeros(n_windows) for k in ("FALT", "EALT", "qFALT", "qEALT", "CALT", "AALT")}

    for w_id in range(n_windows):
        episodes = range(w_id, w_id + n_agents)

        reached_at_all = np.zeros(n_agents)      # did agent i reach at least once in this window
        solo_wins = np.zeros(n_agents)           # how often agent i was the sole arrival
        arrivals = 0                             # t: total arrivals across the window
        solo_episodes = 0                        # w: episodes decided by a single arrival
        crowding = 0                             # sum of (n - arrivals in episode k)

        for k in episodes:
            vector = top_agents_per_episode[k]
            arrivals += terminal_occurrences_per_episode[k]
            in_episode = sum(vector)
            crowding += n_agents - in_episode
            if in_episode == 1:
                solo_episodes += 1
            for i in range(n_agents):
                if vector[i] == 1:
                    reached_at_all[i] = 1
                    if in_episode == 1:
                        solo_wins[i] += 1

        distinct = float(reached_at_all.sum())
        falt = distinct / arrivals if arrivals else 0.0
        ealt = (solo_episodes * distinct) / (n_agents ** 2)

        beta["FALT"][w_id] = falt
        beta["EALT"][w_id] = ealt
        beta["qFALT"][w_id] = falt ** 2
        beta["qEALT"][w_id] = ealt ** 2
        beta["CALT"][w_id] = crowding * (falt ** 2) / (n_agents * (n_agents - 1))
        beta["AALT"][w_id] = (
            float(sum(1 for wins in solo_wins if wins == 1)) / arrivals if arrivals else 0.0)

    return {name: float(values.mean()) for name, values in beta.items()}


# --- the traditional measures the papers argue are insufficient ----------------------------------

def _min_over_max(counts: Iterable[float]) -> float:
    """The shape all three fairness measures share: the worst-off positive share over the best.

    Agents with zero are excluded, which is deliberate but worth knowing when reading the number:
    an agent frozen out entirely does not drag the score down, it simply stops being counted. That
    is part of why these can look healthy while coordination has failed.
    """
    positive = [c for c in counts if c > 0]
    if not positive:
        return 0.0
    return min(positive) / max(positive)


def traditional_metrics(top_agents_per_episode: Sequence[Sequence[int]], n_agents: int,
                        total_reward_per_agent: Sequence[float],
                        full_reward: float) -> dict[str, float]:
    """Efficiency plus the three fairness measures, each over a different quantity.

        efficiency       total reward earned over the most a run could have paid out
        reward_fairness  min/max of total rewards        <- the one the papers show hitting 0.99
        tt_fairness      min/max of reaches, ties included
        fairness         min/max of exclusive wins, ties excluded

    Three fairness numbers rather than one because they disagree, and the disagreement is
    informative: rewards can be near-perfectly equal while exclusive wins are not shared at all.
    """
    n_episodes = len(top_agents_per_episode)
    max_possible = n_episodes * full_reward

    reaches = [0] * n_agents
    solo = [0] * n_agents
    for vector in top_agents_per_episode:
        alone = sum(vector) == 1
        for i in range(n_agents):
            if vector[i] == 1:
                reaches[i] += 1
                if alone:
                    solo[i] += 1

    return {
        "efficiency": (float(sum(total_reward_per_agent)) / max_possible) if max_possible > 0 else 0.0,
        "reward_fairness": _min_over_max(total_reward_per_agent),
        "tt_fairness": _min_over_max(reaches),
        "fairness": _min_over_max(solo),
    }


# --- putting it together --------------------------------------------------------------------------

def compute_all(episodes: Sequence[dict[str, Any]], n_agents: int,
                full_reward: float = 100.0) -> dict[str, Any]:
    """Every measure in this module, from the runner's per-episode records.

    `episodes` is `run_match(...)["episodes"]`: each entry carries `top_agents` (the per-agent
    arrival flags) and `rewards` (that episode's per-agent totals). Only episodic games produce
    these, so a single-stage game yields one episode and the alternation measures are trivially
    undefined, which they should be.
    """
    usable = [e for e in episodes if "top_agents" in e]
    if not usable:
        return {"n_episodes": 0, "note": "no per-episode arrival records; not an episodic game"}

    top_agents = [list(e["top_agents"]) for e in usable]
    arrivals = [sum(v) for v in top_agents]
    totals = [0.0] * n_agents
    for e in usable:
        for i, r in enumerate(e.get("rewards", [])):
            totals[i] += float(r)

    n_episodes = len(top_agents)
    out: dict[str, Any] = {"n_episodes": n_episodes}
    out.update(traditional_metrics(top_agents, n_agents, totals, full_reward))
    out.update(alt_metrics(top_agents, n_agents, arrivals))

    for label, exclusive in (("reach", False), ("excl", True)):
        rp = rp_metrics(win_lists(top_agents, n_agents, exclusive), n_agents, n_episodes)
        for key in ("RS", "WPE", "RP", "AWE", "avg_wait"):
            out[f"{key}_{label}"] = rp[key]
        out[f"detail_{label}"] = {"RS_per_agent": rp["RS_per_agent"],
                                  "WPE_per_agent": rp["WPE_per_agent"]}

    out["solo_win_episodes"] = int(sum(1 for a in arrivals if a == 1))
    out["collision_episodes"] = int(sum(1 for a in arrivals if a > 1))
    out["empty_episodes"] = int(sum(1 for a in arrivals if a == 0))
    return out


def coordination_score(observed: float, random_baseline: float, perfect: float = 1.0) -> float:
    """Where a result sits between a random policy and perfect alternation.

    Positive means better than random, zero means indistinguishable from it, and **negative means
    worse than random**, which is the papers' headline: Q-learning lands there on every ALT
    variant while the traditional measures look fine.

    Not bounded to +/-1. A score of -119% is arithmetically valid and did occur in the source
    data; it means the observed value fell below random by more than the whole random-to-perfect
    span.

    The baseline must come from an actual random-policy run over the same configuration, not from
    a guess. Run one with `RandomAgent`s and pass its value for the same measure.
    """
    if perfect == random_baseline:
        return 0.0 if observed == random_baseline else 1.0
    return (observed - random_baseline) / (perfect - random_baseline)


def coordination_scores(observed: dict[str, Any], random_baseline: dict[str, Any],
                        perfect: float = 1.0) -> dict[str, float]:
    """`coordination_score` for every measure the two runs share."""
    return {
        key: coordination_score(float(observed[key]), float(random_baseline[key]), perfect)
        for key in sorted(HIGHER_IS_BETTER & set(observed) & set(random_baseline))
    }
