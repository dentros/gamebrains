"""
The two headline experiments of the platform paper, over independent seeds with confidence intervals.

Regenerates the Public Goods numbers of the paper's results section, Table 4 (alternation against run
length) and the full measure set in its Appendix C.

    python -m gamebrains.experiments.run_multiseed
    python -m gamebrains.experiments.run_multiseed --seeds 5 --lengths 1000 3000
    python -m gamebrains.experiments.run_multiseed --json results.json

Each master seed derives its own per-agent seeds, so runs are independent samples rather than slices
of one trajectory. Reported as mean with a 95% interval of 1.96*s/sqrt(N), plus, where a direction of
effect is claimed, the count of seeds showing it: a mean can hide a split population and a count
cannot.

The default 30 seeds over five run lengths is a long job, dominated by the two longest lengths.
Use --seeds/--lengths for a quick check that the trend is present before committing to the full run.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Optional

import numpy as np

from ..agents.classic import RandomAgent
from ..agents.qlearning import QLearningAgent
from ..engine.runner import run_match
from ..games.congestion import from_preset
from ..games.public_goods import PublicGoodsGame
from ..metrics import social
from ..metrics.social_alt import compute_all, coordination_score

EPS_DECAY = 0.9997
EPS_MIN = 0.02
DEFAULT_LENGTHS = (1000, 3000, 10_000, 40_000, 100_000)


def ci95(values) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    if a.size < 2:
        return float(a.mean()), 0.0
    return float(a.mean()), float(1.96 * a.std(ddof=1) / math.sqrt(a.size))


def fmt(mean: float, half: float) -> str:
    return f"{mean:.3f} +/- {half:.3f}"


# --- Experiment 1: Public Goods, cooperation collapse --------------------------------------------

def public_goods_run(seed: int, n_agents: int = 5, rounds: int = 3000,
                     mpcr: float = 0.5) -> tuple[float, float]:
    game = PublicGoodsGame(n_agents=n_agents, rounds=rounds, mpcr=mpcr)
    roster = [QLearningAgent(f"Q{i}", n_states=game.n_states, n_actions=game.n_actions,
                             seed=seed * 100 + i, epsilon_decay=EPS_DECAY, epsilon_min=EPS_MIN)
              for i in range(n_agents)]
    records = run_match(game, roster, rounds=rounds, seed=seed)
    series = social.compute_all(records, game.max_welfare_per_round())["cooperation_series"]
    return float(series[:100].mean()), float(series[-100:].mean())


# --- Experiment 2: congestion family, alternation against run length -----------------------------

def congestion_run(seed: int, rounds: int, n_agents: int = 3, learner: bool = True) -> dict:
    game = from_preset("hjg", n_agents=n_agents)
    if learner:
        roster = [QLearningAgent(f"Q{i}", n_states=game.n_states, n_actions=game.n_actions,
                                 seed=seed * 100 + i, epsilon_decay=EPS_DECAY,
                                 epsilon_min=EPS_MIN) for i in range(n_agents)]
    else:
        roster = [RandomAgent(f"R{i}", seed=seed * 100 + 50 + i) for i in range(n_agents)]
    records = run_match(game, roster, rounds=rounds, seed=seed)
    return compute_all(records["episodes"], n_agents=n_agents, full_reward=game.full_reward)


def _efficiency(row: dict) -> float:
    """social_alt's per-episode efficiency is renamed at the webui merge point; accept either key."""
    return float(row["alt_efficiency"] if "alt_efficiency" in row else row["efficiency"])


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    n = args.seeds
    results: dict = {}

    print(f"EXPERIMENT 1  Public Goods, 5 Q-learners, 3000 rounds, N={n} seeds")
    first, last = zip(*[public_goods_run(s) for s in range(1, n + 1)])
    fm, fh = ci95(first)
    lm, lh = ci95(last)
    fell = sum(1 for a, b in zip(first, last) if b < a)
    print(f"  cooperation, first 100 rounds : {fmt(fm, fh)}")
    print(f"  cooperation, last 100 rounds  : {fmt(lm, lh)}")
    print(f"  seeds where it fell           : {fell}/{n}")
    results["public_goods"] = {"first": [fm, fh], "last": [lm, lh], "fell": fell, "seeds": n}

    print(f"\nEXPERIMENT 2  Honey-Jar (ILF), 3 agents, N={n} seeds per length")
    header = (f"  {'rounds':>7} {'CALT (QL)':>16} {'CALT (random)':>16} "
              f"{'CS(CALT)':>16} {'rewFair (QL)':>16}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Per-seed values, kept so the sweep can be checked inside each seed as well as in aggregate.
    # The sweep is paired: the same seeds are reused at every budget and per-step schedules do not
    # depend on the total round count, so a shorter run is a prefix of the longer one with the same
    # seed. That makes "does every seed decline" a stronger and simpler claim than any comparison
    # of confidence intervals between rows. Costs nothing extra: the runs already happen below.
    per_seed: dict[str, list[list[float]]] = {"calt_ql": [], "aalt_ql": [], "calt_rnd": []}

    results["congestion"] = {}
    for rounds in args.lengths:
        learners = [congestion_run(s, rounds, learner=True) for s in range(1, n + 1)]
        randoms = [congestion_run(s, rounds, learner=False) for s in range(1, n + 1)]

        per_seed["calt_ql"].append([float(r["CALT"]) for r in learners])
        per_seed["aalt_ql"].append([float(r["AALT"]) for r in learners])
        per_seed["calt_rnd"].append([float(r["CALT"]) for r in randoms])

        qm, qh = ci95([r["CALT"] for r in learners])
        rm, rh = ci95([r["CALT"] for r in randoms])
        cm, ch = ci95([coordination_score(a["CALT"], b["CALT"])
                       for a, b in zip(learners, randoms)])
        wm, wh = ci95([r["reward_fairness"] for r in learners])
        print(f"  {rounds:7d} {fmt(qm, qh):>16} {fmt(rm, rh):>16} "
              f"{f'{cm:+.1%} +/- {ch:.1%}':>16} {fmt(wm, wh):>16}")

        results["congestion"][rounds] = {
            "calt_learner": [qm, qh], "calt_random": [rm, rh],
            "coordination_score": [cm, ch], "reward_fairness": [wm, wh],
            "aalt_learner": list(ci95([r["AALT"] for r in learners])),
            "tt_fairness": list(ci95([r["tt_fairness"] for r in learners])),
            "fairness": list(ci95([r["fairness"] for r in learners])),
            "efficiency_learner": list(ci95([_efficiency(r) for r in learners])),
            "efficiency_random": list(ci95([_efficiency(r) for r in randoms])),
            "episodes": list(ci95([r["n_episodes"] for r in learners])),
            "collision_rate": list(ci95([r["collision_episodes"] / max(r["n_episodes"], 1)
                                         for r in learners])),
        }

    # --- per-seed monotonicity, the paired reading of the same runs -------------------------------
    if len(args.lengths) > 1:
        print(f"\nPER-SEED MONOTONICITY  (seeds of {n} that decrease at each step)")
        labels = {"calt_ql": "CALT (QL)", "aalt_ql": "AALT (QL)", "calt_rnd": "CALT (random)"}
        head = "  " + f"{'step':>22}" + "".join(f"{labels[k]:>16}" for k in per_seed)
        print(head)
        print("  " + "-" * (len(head) - 2))

        matrices = {k: np.asarray(v).T for k, v in per_seed.items()}   # [seed, length]
        for i in range(len(args.lengths) - 1):
            step = f"{args.lengths[i]} -> {args.lengths[i+1]}"
            cells = "".join(f"{int(np.sum(m[:, i + 1] < m[:, i])):>13}/{n:<2}"
                            for m in matrices.values())
            print(f"  {step:>22}{cells}")

        cells = "".join(f"{int(np.sum(np.all(np.diff(m, axis=1) < 0, axis=1))):>13}/{n:<2}"
                        for m in matrices.values())
        print(f"  {'decreasing at all steps':>22}{cells}")
        for name, m in matrices.items():
            results.setdefault("per_seed", {})[name] = {
                "monotone_all_steps": int(np.sum(np.all(np.diff(m, axis=1) < 0, axis=1))),
                "pairwise": [int(np.sum(m[:, i + 1] < m[:, i]))
                            for i in range(len(args.lengths) - 1)],
                "first_range": [float(m[:, 0].min()), float(m[:, 0].max())],
                "last_range": [float(m[:, -1].min()), float(m[:, -1].max())],
                "ranges_disjoint": bool(m[:, -1].max() < m[:, 0].min()),
                "matrix": m.tolist(),
            }
        for name, m in matrices.items():
            print(f"  {labels[name]}: {m[:,0].min():.3f}-{m[:,0].max():.3f} at the shortest budget, "
                  f"{m[:,-1].min():.3f}-{m[:,-1].max():.3f} at the longest, "
                  f"disjoint={bool(m[:,-1].max() < m[:,0].min())}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
