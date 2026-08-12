"""
Cross-paradigm bake-off: train each architecture separately, freeze them all, play one match.

Regenerates the table in Appendix E of the platform paper.

    python -m gamebrains.experiments.run_bakeoff
    python -m gamebrains.experiments.run_bakeoff --seeds 3 --json out.json

The protocol is the substance here, not the numbers. Comparing a population-evolved architecture
against an online learner by simply putting them in one match conflates final policy quality with
how fast each adapts *during* the comparison. So:

  1. Each architecture trains in pure self-play on its own copy of the game. Nothing co-adapts.
  2. Every trained agent is frozen: acts greedily, discards learning updates.
  3. One evaluation match runs with a single frozen agent of each kind in the roster.

Step 2 is what makes it a comparison. The freezing wrapper is the same construction the web
interface's own bake-off mode uses.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Optional

import numpy as np

from ..agents.classic import AllD
from ..agents.dqn import DQNAgent
from ..agents.fep import FEPAgent
from ..agents.qlearning import QLearningAgent
from ..agents.markov_brain import crossover as mb_crossover, spawn as mb_spawn
from ..engine.agent import Agent
from ..engine.evolution import EvolutionConfig, evolve
from ..engine.runner import run_match
from ..games.public_goods import PublicGoodsGame
from ..metrics import social

N_AGENTS = 5
MPCR = 0.5
PRETRAIN_ROUNDS = 3000
EVAL_ROUNDS = 1000
EPS_DECAY = 0.9997
EPS_MIN = 0.02

KINDS = ["qlearning", "dqn", "fep", "markov_brain", "classic"]
LABELS = {"qlearning": "Tabular Q", "dqn": "DQN", "fep": "FEP (active inf.)",
          "markov_brain": "Markov brain (GA)", "classic": "AllD (reference)"}


class Frozen(Agent):
    """Acts greedily and never learns, so an evaluation match measures the final policy only."""

    training_mode = "fixed"

    def __init__(self, agent: Any, name: str) -> None:
        self._agent = agent
        self.name = name
        self.kind = agent.kind

    def act(self, observation: int) -> int:
        saved = getattr(self._agent, "epsilon", None)
        if saved is not None:
            self._agent.epsilon = 0.0
        try:
            return self._agent.act(observation)
        finally:
            if saved is not None:
                self._agent.epsilon = saved

    def on_match_start(self, game: Any) -> None:
        # Forward, or a frozen role-defined strategy never learns which action plays its role.
        self._agent.on_match_start(game)

    def update(self, *args, **kwargs) -> None:
        return None

    def inspect(self) -> dict:
        return self._agent.inspect()

    def render_brain(self) -> dict:
        return self._agent.render_brain()


def _game(rounds: int) -> PublicGoodsGame:
    return PublicGoodsGame(n_agents=N_AGENTS, rounds=rounds, mpcr=MPCR)


def pretrain(kind: str, seed: int, quiet: bool = False) -> Frozen:
    """Train one architecture in self-play and return one frozen representative of it."""
    game = _game(PRETRAIN_ROUNDS)

    if kind == "classic":
        return Frozen(AllD("AllD"), "AllD")

    if kind == "fep":
        # No trained parameters, only beliefs formed online, so a fresh agent is the honest
        # representative. Reported in the paper as not comparable to the RL rows in the same way.
        return Frozen(FEPAgent("FEP", n_agents=N_AGENTS, mpcr=MPCR, seed=seed), "FEP")

    if kind == "markov_brain":
        config = EvolutionConfig(population_size=20, generations=15, match_rounds=200,
                                 matches_per_generation=3, seed=seed)
        result = evolve(lambda: _game(200), config, spawn=mb_spawn, crossover=mb_crossover)
        best = result.population[int(np.argmax(result.fitness))]
        return Frozen(best, "MarkovBrain")

    make = QLearningAgent if kind == "qlearning" else DQNAgent
    prefix = "Q" if kind == "qlearning" else "D"
    roster = [make(f"{prefix}{i}", n_states=game.n_states, n_actions=game.n_actions,
                   seed=seed * 100 + i, epsilon_decay=EPS_DECAY, epsilon_min=EPS_MIN)
              for i in range(N_AGENTS)]
    run_match(game, roster, rounds=PRETRAIN_ROUNDS, seed=seed)
    return Frozen(roster[0], roster[0].name)


def one_bakeoff(seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    roster = [pretrain(kind, seed) for kind in KINDS]
    game = _game(EVAL_ROUNDS)
    records = run_match(game, roster, rounds=EVAL_ROUNDS, seed=seed + 9000)
    metrics = social.compute_all(records, game.max_welfare_per_round())

    payoff = np.asarray(records["rewards"], dtype=float).mean(axis=0)
    actions = np.asarray(records["actions"])
    cooperation = (actions == 1).mean(axis=0)   # COOPERATE == 1 in the Public Goods Game
    return payoff, cooperation, float(np.mean(metrics["cooperation_series"]))


def ci95(values) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    if a.size < 2:
        return float(a.mean()), 0.0
    return float(a.mean()), float(1.96 * a.std(ddof=1) / math.sqrt(a.size))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--seeds", type=int, default=10, help="independent master seeds")
    parser.add_argument("--json", default="", help="also write results to this path")
    args = parser.parse_args(argv)

    print(f"Bake-off: {len(KINDS)} architectures, {PRETRAIN_ROUNDS} pretrain rounds each, "
          f"{EVAL_ROUNDS} eval rounds, {args.seeds} seeds")
    print("(the genetic algorithm dominates the runtime)\n")

    payoffs, cooperations, overall = [], [], []
    for seed in range(1, args.seeds + 1):
        p, c, o = one_bakeoff(seed)
        payoffs.append(p)
        cooperations.append(c)
        overall.append(o)
        print(f"  seed {seed:2d}  payoff {np.round(p, 3).tolist()}  "
              f"coop {np.round(c, 3).tolist()}")

    payoffs = np.asarray(payoffs)
    cooperations = np.asarray(cooperations)

    print(f"\n  {'architecture':22s} {'payoff/round':>20} {'cooperation':>20}")
    print("  " + "-" * 64)
    results: dict[str, Any] = {}
    for j, kind in enumerate(KINDS):
        pm, ph = ci95(payoffs[:, j])
        cm, ch = ci95(cooperations[:, j])
        print(f"  {LABELS[kind]:22s} {f'{pm:+.4f} +/- {ph:.4f}':>20} "
              f"{f'{cm:.4f} +/- {ch:.4f}':>20}")
        results[kind] = {"label": LABELS[kind], "payoff": [pm, ph], "cooperation": [cm, ch]}

    om, oh = ci95(overall)
    print(f"\n  match cooperation rate: {om:.4f} +/- {oh:.4f}")

    # The split the paper reports: self-play training does not always find the same policy.
    q_coop = cooperations[:, KINDS.index("qlearning")]
    q_pay = payoffs[:, KINDS.index("qlearning")]
    defecting = q_coop < 0.05
    if defecting.any() and (~defecting).any():
        print(f"  tabular Q split: {int(defecting.sum())}/{args.seeds} seeds converged to "
              f"defection (coop {q_coop[defecting].mean():.3f}, payoff "
              f"{q_pay[defecting].mean():+.4f}); the other {int((~defecting).sum())} kept "
              f"cooperating (coop {q_coop[~defecting].mean():.3f}, payoff "
              f"{q_pay[~defecting].mean():+.4f})")

    results["_match_cooperation"] = [om, oh]
    results["_config"] = {"n_agents": N_AGENTS, "mpcr": MPCR, "pretrain_rounds": PRETRAIN_ROUNDS,
                          "eval_rounds": EVAL_ROUNDS, "seeds": args.seeds}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
