"""
Social + a first information-theoretic metric for the Public Goods Game.

Operates on the per-round records produced by engine.runner.run_match:
  - actions:     (rounds, n_agents) int array (0=Defect, 1=Cooperate)
  - rewards:     (rounds, n_agents) float array
  - cooperators: (rounds,) int array (number of cooperators per round)

Metrics:
  - cooperation_rate: overall fraction of Cooperate actions (+ per-round series)
  - efficiency:       realized total welfare / welfare under universal cooperation
  - payoff_gini:      inequality of cumulative payoffs (0 = perfectly equal)
  - action_entropy:   mean binary entropy of the per-round cooperation frequency (bits);
                      an information-theoretic teaser (0 = deterministic, 1 = maximally mixed)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def cooperation_rate(actions: np.ndarray,
                     cooperators: Optional[np.ndarray] = None) -> dict[str, Any]:
    """The share of players who cooperated, per round and overall.

    **`cooperators` is the game's own count and is what this uses when it is given.** Deriving the
    answer from `actions` instead means reading an index as a meaning, and an index means nothing
    across games: cooperating is action 1 in the Public Goods Game and action 0 in the congestion
    family, where the cooperative act is to stay put and let somebody else through. Counting
    `actions == 1` therefore reported the *claim* rate under the name "cooperation rate" for every
    congestion run, which is the complement of the truth.

    This is the same defect the platform's own declaration mechanism was built to prevent, one
    layer up from where it was fixed. `agents/` was migrated to ask the game what its indices mean
    (CLAUDE.md sections 3 and 9g) and `engine/runner.py` already records `result.info["cooperators"]`
    from the game, so the correct number was being carried the whole time and this function was
    the one place that stopped reading it. Found by `repository/verify.py` recomputing stored
    records from their own event logs, not by a test.

    The fallback to `actions` is kept for a game that declares no count, and it is only correct
    for a game whose cooperative action is index 1. A game should declare the count instead.
    """
    if cooperators is not None:
        n_agents = actions.shape[1]
        per_round = np.asarray(cooperators, dtype=float) / float(n_agents)
        return {"overall": float(per_round.mean()), "series": per_round}
    per_round = actions.mean(axis=1)
    return {"overall": float(actions.mean()), "series": per_round}


def efficiency(rewards: np.ndarray, max_welfare_per_round: float) -> float:
    rounds = rewards.shape[0]
    realized = float(rewards.sum())
    ceiling = max_welfare_per_round * rounds
    if ceiling == 0:
        return 0.0
    return realized / ceiling


def payoff_gini(cumulative_rewards: np.ndarray) -> float:
    x = np.asarray(cumulative_rewards, dtype=float)
    # Shift so all values are non-negative (payoffs can be negative in a PGG).
    x = x - x.min()
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * x) / (n * x.sum())) - (n + 1.0) / n)


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def action_entropy(actions: np.ndarray) -> float:
    per_round_p = actions.mean(axis=1)
    return float(np.mean([_binary_entropy(p) for p in per_round_p]))


def compute_all(records: dict[str, np.ndarray], max_welfare_per_round: float) -> dict[str, Any]:
    actions = records["actions"]
    rewards = records["rewards"]
    cum = rewards.sum(axis=0)
    # The game's own per-round count when the runner captured one, which it does for every game
    # that declares it. See `cooperation_rate` for why reading `actions` instead was wrong.
    coop = cooperation_rate(actions, records.get("cooperators"))
    return {
        "cooperation_rate": coop["overall"],
        "cooperation_series": coop["series"],
        "efficiency": efficiency(rewards, max_welfare_per_round),
        "payoff_gini": payoff_gini(cum),
        "action_entropy_bits": action_entropy(actions),
        "cumulative_payoffs": cum,
    }
