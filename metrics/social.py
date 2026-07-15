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

from typing import Any

import numpy as np


def cooperation_rate(actions: np.ndarray) -> dict[str, Any]:
    per_round = actions.mean(axis=1)          # fraction cooperating each round
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
    coop = cooperation_rate(actions)
    return {
        "cooperation_rate": coop["overall"],
        "cooperation_series": coop["series"],
        "efficiency": efficiency(rewards, max_welfare_per_round),
        "payoff_gini": payoff_gini(cum),
        "action_entropy_bits": action_entropy(actions),
        "cumulative_payoffs": cum,
    }
