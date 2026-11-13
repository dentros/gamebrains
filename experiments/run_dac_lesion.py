"""
What each layer of the DAC-style agent contributes, measured by removing it.

Regenerates the DAC paragraph of the platform paper.

    python -m gamebrains.experiments.run_dac_lesion
    python -m gamebrains.experiments.run_dac_lesion --seeds 3 --json out.json

Two settings, each over independent seeds with 95% t intervals:

* **Honey-Jar Game**, three agents, the `hjg` preset. The complete agent against two lesions (the
  reactive layer alone, and everything but the contextual layer), with three tabular Q-learners on
  the same game as the reference. Reported: CALT, AALT, reward fairness, efficiency, and the share
  of moves each layer decided.
* **Public Goods Game**, four agents. One complete agent among three unconditional cooperators,
  among three unconditional defectors, and four complete agents in self-play, with and without the
  contextual layer. Reported: the DAC seats' cooperation and the contextual layer's share.

The point is the comparison between rows, not any row on its own: a lesion that leaves behaviour
unchanged says the removed layer was not producing it, and one that changes it says by how much.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Optional

import numpy as np

from ..agents.classic import AllC, AllD
from ..agents.dac import DACAgent
from ..agents.qlearning import QLearningAgent
from ..engine.runner import run_match
from ..games.congestion import from_preset
from ..games.public_goods import PublicGoodsGame
from ..metrics import social, social_alt
from ..repository.meta_analysis import t95

HJG_AGENTS = 3
HJG_ROUNDS = 3000
PGG_AGENTS = 4
PGG_ROUNDS = 1000

HJG_ROWS = {
    "dac_all_layers": {},
    "dac_reactive_only": {"layers": ("reactive",)},
    "dac_without_contextual": {"layers": ("reactive", "adaptive")},
    "qlearning_reference": None,
}


def ci95(values) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    if a.size < 2:
        return float(a.mean()), 0.0
    return float(a.mean()), float(t95(a.size - 1) * a.std(ddof=1) / math.sqrt(a.size))


def _layer_shares(roster: list) -> dict[str, float]:
    totals: dict[str, int] = {}
    for agent in roster:
        for layer, count in getattr(agent, "layer_counts", {}).items():
            totals[layer] = totals.get(layer, 0) + count
    whole = sum(totals.values())
    return {k: v / whole for k, v in totals.items()} if whole else {}


def hjg_once(row: str, seed: int) -> dict[str, Any]:
    game = from_preset("hjg", HJG_AGENTS)
    kwargs = HJG_ROWS[row]
    if kwargs is None:
        roster = [QLearningAgent(f"Q{i}", game.n_states, game.n_actions, seed=seed * 10 + i)
                  for i in range(HJG_AGENTS)]
    else:
        roster = [DACAgent(f"D{i}", seed=seed * 10 + i, **kwargs) for i in range(HJG_AGENTS)]
    records = run_match(game, roster, rounds=HJG_ROUNDS, seed=seed)
    alt = social_alt.compute_all(records["episodes"], n_agents=HJG_AGENTS,
                                 full_reward=game.full_reward)
    return {"CALT": alt["CALT"], "AALT": alt["AALT"], "reward_fairness": alt["reward_fairness"],
            "efficiency": social.compute_all(records, game.max_welfare_per_round())["efficiency"],
            "episodes": len(records["episodes"]), "layers": _layer_shares(roster)}


def pgg_once(row: str, seed: int) -> dict[str, Any]:
    game = PublicGoodsGame(n_agents=PGG_AGENTS)
    if row in ("among_cooperators", "among_defectors"):
        other = AllC if row == "among_cooperators" else AllD
        roster = [DACAgent("D0", seed=seed * 10)] + [other(f"o{i}") for i in range(PGG_AGENTS - 1)]
        seats = [0]
    else:
        layers = ("reactive", "adaptive") if row == "self_play_without_contextual" else None
        kwargs = {"layers": layers} if layers else {}
        roster = [DACAgent(f"D{i}", seed=seed * 10 + i, **kwargs) for i in range(PGG_AGENTS)]
        seats = list(range(PGG_AGENTS))
    records = run_match(game, roster, rounds=PGG_ROUNDS, seed=seed)
    cooperated = np.asarray(records["cooperated"])[:, seats]
    return {"cooperation": float(cooperated.mean()),
            "layers": _layer_shares([roster[i] for i in seats])}


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in runs[0]:
        if key == "layers":
            names = sorted({name for r in runs for name in r["layers"]})
            out["layers"] = {name: ci95([r["layers"].get(name, 0.0) for r in runs])
                             for name in names}
        else:
            out[key] = ci95([r[key] for r in runs])
    return out


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)
    seeds = range(1, args.seeds + 1)

    results: dict[str, Any] = {"config": {"hjg_agents": HJG_AGENTS, "hjg_rounds": HJG_ROUNDS,
                                          "pgg_agents": PGG_AGENTS, "pgg_rounds": PGG_ROUNDS,
                                          "seeds": args.seeds, "interval": "95% t"}}
    print(f"Honey-Jar Game, {HJG_AGENTS} agents, {HJG_ROUNDS} rounds, {args.seeds} seeds")
    for row in HJG_ROWS:
        s = summarise([hjg_once(row, seed) for seed in seeds])
        results[f"hjg/{row}"] = s
        print(f"  {row:24s} CALT {s['CALT'][0]:.3f} +/- {s['CALT'][1]:.3f}  "
              f"AALT {s['AALT'][0]:.3f} +/- {s['AALT'][1]:.3f}  "
              f"RF {s['reward_fairness'][0]:.3f}  eff {s['efficiency'][0]:.3f}  "
              + " ".join(f"{k}={v[0]:.2f}" for k, v in s["layers"].items() if v[0] >= 0.005))

    print(f"\nPublic Goods Game, {PGG_AGENTS} agents, {PGG_ROUNDS} rounds, {args.seeds} seeds")
    for row in ("among_cooperators", "among_defectors", "self_play", "self_play_without_contextual"):
        s = summarise([pgg_once(row, seed) for seed in seeds])
        results[f"pgg/{row}"] = s
        print(f"  {row:30s} cooperation {s['cooperation'][0]:.3f} +/- {s['cooperation'][1]:.3f}  "
              + " ".join(f"{k}={v[0]:.2f}" for k, v in s["layers"].items() if v[0] >= 0.005))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
