"""The same evolutionary pressure applied to three different substrates.

`run_bakeoff.py` trains each architecture by its own method and freezes the results, which answers
which architecture learns the game best under the rule it was designed for. That comparison cannot
separate the substrate from the learning rule, because each row changes both at once.

This one holds the learning rule fixed. Every architecture is searched by the identical genetic
algorithm: one population size, one generation count, one mutation rate, one set of seeds, one
game. What differs between rows is only what a genome *is* -- a table of state-action values, the
weights of a small recurrent binary network, or the parameters of a feed-forward one. That is the
question the Markov-brain literature asks, and the platform could not ask it until the GA stopped
being tied to one genome.

    python -m gamebrains.experiments.run_evolution_bakeoff
    python -m gamebrains.experiments.run_evolution_bakeoff --seeds 5 --generations 40
    python -m gamebrains.experiments.run_evolution_bakeoff --lifetime-learning lamarckian

**Three things this does not claim, stated here because a table invites all three.**

*Equal pressure is not equal difficulty.* A Q-table for this game holds `n_states x n_actions`
numbers and a network holds thousands, so one generation of mutation explores a very different
fraction of each space. A row that evolves slower may have a larger space rather than a worse
substrate, and nothing here separates those.

*The ranking is of genomes under this GA*, with uniform crossover and Gaussian mutation, on one
game. A different operator set is a different experiment, and crossing networks element by element
is the weakest part of the three (see the note in `agents/dqn.crossover`).

*Fitness is payoff, so the reference point matters more than the ordering.* In the Public Goods
Game universal defection is the unique Nash equilibrium and pays exactly 0, which
`metrics/equilibrium.py` confirms analytically. A population converging toward 0 has found the
equilibrium, which is the expected outcome rather than a failure, and a row sitting well above it
is the result worth looking at twice.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Optional

import numpy as np

from ..agents.factory import EVOLVABLE_KINDS, evolvable
from ..engine.evolution import EvolutionConfig, evolve
from ..games.public_goods import PublicGoodsGame

LABELS = {"markov_brain": "Markov brain", "qlearning": "Evolved Q-table",
          "dqn": "Evolved network"}
#: Name prefixes keep the individuals in one run distinguishable in a log. Cosmetic only.
PREFIXES = {"markov_brain": "MB", "qlearning": "QL", "dqn": "DQ"}


def _ci95(values: Any) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    if a.size < 2:
        return float(a.mean()), 0.0
    return float(a.mean()), float(1.96 * a.std(ddof=1) / math.sqrt(a.size))


def one_run(kind: str, seed: int, args: argparse.Namespace) -> tuple[float, float]:
    """Evolve one architecture once. Returns (best fitness, population mean fitness)."""
    spawn, crossover = evolvable(kind)
    config = EvolutionConfig(
        population_size=args.population, generations=args.generations,
        match_rounds=args.match_rounds, matches_per_generation=args.matches_per_generation,
        mutation_rate=args.mutation_rate, mutation_scale=args.mutation_scale,
        elite_count=args.elite_count, seed=seed, name_prefix=PREFIXES[kind],
        lifetime_learning=args.lifetime_learning,
    )

    def game_factory() -> PublicGoodsGame:
        return PublicGoodsGame(n_agents=args.n_agents, rounds=args.match_rounds, mpcr=args.mpcr)

    result = evolve(game_factory, config, spawn=spawn, crossover=crossover)
    return float(result.fitness.max()), float(result.fitness.mean())


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kinds", nargs="+", default=list(EVOLVABLE_KINDS),
                        help=f"architectures to evolve, from {list(EVOLVABLE_KINDS)}")
    parser.add_argument("--seeds", type=int, default=5, help="independent runs per architecture")
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--match-rounds", type=int, default=200)
    parser.add_argument("--matches-per-generation", type=int, default=3)
    parser.add_argument("--n-agents", type=int, default=4)
    parser.add_argument("--mpcr", type=float, default=0.5)
    parser.add_argument("--mutation-rate", type=float, default=0.15)
    parser.add_argument("--mutation-scale", type=float, default=0.4)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--lifetime-learning", default="off", choices=["off", "lamarckian"],
                        help="'off' searches over policies; 'lamarckian' lets each individual "
                             "learn during its own fitness match and pass that on, so the GA is "
                             "then searching over starting points rather than over policies")
    parser.add_argument("--json", default="", help="also write the results to this path")
    args = parser.parse_args(argv)

    unknown = [k for k in args.kinds if k not in EVOLVABLE_KINDS]
    if unknown:
        print(f"not evolvable: {unknown}. Choose from {list(EVOLVABLE_KINDS)}")
        return 1
    if args.population % args.n_agents != 0:
        print(f"population ({args.population}) must divide by n_agents ({args.n_agents}); the "
              f"population is split into matches of n_agents seats")
        return 1

    print(f"Evolutionary bake-off: {len(args.kinds)} substrates under one GA\n"
          f"  population {args.population}, {args.generations} generations, "
          f"{args.matches_per_generation} match(es) of {args.match_rounds} rounds per generation\n"
          f"  Public Goods Game, n={args.n_agents}, mpcr={args.mpcr}, "
          f"{args.seeds} seed(s) per substrate\n"
          f"  lifetime learning: {args.lifetime_learning}\n")

    results: dict[str, Any] = {}
    for kind in args.kinds:
        bests, means = [], []
        for seed in range(1, args.seeds + 1):
            best, mean = one_run(kind, seed, args)
            bests.append(best)
            means.append(mean)
            print(f"  {LABELS[kind]:18} seed {seed:2d}   best {best:8.3f}   mean {mean:8.3f}")
        results[kind] = {"label": LABELS[kind], "best": bests, "mean": means}
        print()

    print(f"  {'substrate':18} {'best fitness':>22} {'population mean':>22}")
    print("  " + "-" * 64)
    for kind in args.kinds:
        bm, bh = _ci95(results[kind]["best"])
        mm, mh = _ci95(results[kind]["mean"])
        print(f"  {LABELS[kind]:18} {f'{bm:+.3f} +/- {bh:.3f}':>22} "
              f"{f'{mm:+.3f} +/- {mh:.3f}':>22}")
        results[kind]["summary"] = {"best": [bm, bh], "mean": [mm, mh]}

    print("\n  Universal defection is the unique Nash equilibrium of this configuration and pays "
          "exactly 0.\n  A substrate converging there has found it; one sitting above it is the "
          "row worth examining.\n  Read the spread before the ordering: these are different "
          "search spaces under one set of\n  GA constants, not a like-for-like contest.")

    results["_config"] = {
        "kinds": args.kinds, "seeds": args.seeds, "population": args.population,
        "generations": args.generations, "match_rounds": args.match_rounds,
        "matches_per_generation": args.matches_per_generation, "n_agents": args.n_agents,
        "mpcr": args.mpcr, "mutation_rate": args.mutation_rate,
        "mutation_scale": args.mutation_scale, "elite_count": args.elite_count,
        "lifetime_learning": args.lifetime_learning,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
