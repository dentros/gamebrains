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
#: A seed counts as having found the equilibrium when its best individual earns less than this
#: fraction of what universal contribution pays. Loose on purpose: the question is which of two
#: widely separated outcomes a run landed on, and a tight threshold would turn that into a
#: judgement about near-equilibrium play, which this experiment does not measure.
EQUILIBRIUM_BAND = 0.10


def _ci95(values: Any) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    if a.size < 2:
        return float(a.mean()), 0.0
    return float(a.mean()), float(1.96 * a.std(ddof=1) / math.sqrt(a.size))


def genome_size(kind: str, agent: Any) -> int:
    """How many free numbers selection is searching over, per individual.

    Stated explicitly per kind rather than discovered by scanning attributes, because a wrong
    count here would quietly misreport the one caveat this table most needs: the three substrates
    do not present the GA with comparably sized spaces.
    """
    if kind == "qlearning":
        return int(agent.Q.size)
    if kind == "markov_brain":
        return int(agent.W.size + agent.bias.size)
    if kind == "dqn":
        return int(sum(t.numel() for t in agent.policy.state_dict().values()))
    raise ValueError(f"no genome size defined for {kind!r}")


def still_climbing(history: list[dict], tail: float = 0.25, margin: float = 0.02) -> bool:
    """Was the population still improving when the run was stopped?

    Compares the mean best fitness over the last `tail` of generations against the block before
    it. A run that is still climbing has not converged, and its final number says where the search
    had got to rather than where it ends up -- which is exactly how a short run produces a ranking
    that a longer one reverses.
    """
    n = len(history)
    window = max(1, int(n * tail))
    if n < 2 * window:
        return True
    late = float(np.mean([h["best"] for h in history[-window:]]))
    earlier = float(np.mean([h["best"] for h in history[-2 * window:-window]]))
    return late > earlier + margin * max(1.0, abs(earlier))


def one_run(kind: str, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    """Evolve one architecture once, with the diagnostics the headline numbers need."""
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
    return {
        "best": float(result.fitness.max()),
        "mean": float(result.fitness.mean()),
        "genome": genome_size(kind, result.best),
        "still_climbing": still_climbing(result.fitness_history),
    }


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

    # The game's own two reference points, which is what makes the split below a reading of the
    # game rather than a clustering choice. Universal defection is the unique Nash equilibrium and
    # pays 0; universal contribution pays (mpcr*n - 1) per round to everyone.
    coop_payoff = args.match_rounds * (args.mpcr * args.n_agents - 1.0)
    at_equilibrium = EQUILIBRIUM_BAND * coop_payoff
    print(f"  reference points: universal defection pays 0, universal contribution pays "
          f"{coop_payoff:.1f}\n  a seed counts as having found the equilibrium below "
          f"{at_equilibrium:.1f}\n")

    results: dict[str, Any] = {}
    for kind in args.kinds:
        runs = []
        for seed in range(1, args.seeds + 1):
            run = one_run(kind, seed, args)
            runs.append(run)
            flag = "  still climbing" if run["still_climbing"] else ""
            print(f"  {LABELS[kind]:18} seed {seed:2d}   best {run['best']:8.3f}   "
                  f"mean {run['mean']:8.3f}{flag}")
        results[kind] = {"label": LABELS[kind], "runs": runs,
                         "genome": runs[0]["genome"],
                         "best": [r["best"] for r in runs],
                         "mean": [r["mean"] for r in runs]}
        print()

    print(f"  {'substrate':18} {'genome':>8} {'best fitness':>22} {'population mean':>22}")
    print("  " + "-" * 74)
    for kind in args.kinds:
        bm, bh = _ci95(results[kind]["best"])
        mm, mh = _ci95(results[kind]["mean"])
        print(f"  {LABELS[kind]:18} {results[kind]['genome']:>8} "
              f"{f'{bm:+.3f} +/- {bh:.3f}':>22} {f'{mm:+.3f} +/- {mh:.3f}':>22}")
        results[kind]["summary"] = {"best": [bm, bh], "mean": [mm, mh]}

    # The part that decides whether the row above means anything. A set of seeds that lands in two
    # places has no centre, so its mean describes neither; `run_bakeoff.py` already reports the
    # tabular split this way rather than averaging over it.
    print()
    for kind in args.kinds:
        bests = np.asarray(results[kind]["best"], dtype=float)
        found = bests < at_equilibrium
        results[kind]["found_equilibrium"] = int(found.sum())
        if found.any() and (~found).any():
            print(f"  {LABELS[kind]}: SPLIT, {int(found.sum())}/{len(bests)} seeds found the "
                  f"equilibrium (mean {bests[found].mean():.2f}); the other "
                  f"{int((~found).sum())} sustained contribution (mean {bests[~found].mean():.2f}).")
            print(f"    The interval above is averaging over two outcomes and describes neither. "
                  f"Report the split.")
        elif found.all():
            print(f"  {LABELS[kind]}: every seed found the equilibrium.")
        else:
            print(f"  {LABELS[kind]}: no seed found the equilibrium.")

    climbing = {kind: sum(r["still_climbing"] for r in results[kind]["runs"])
                for kind in args.kinds}
    if any(climbing.values()):
        print(f"\n  NOT CONVERGED: " + ", ".join(
            f"{LABELS[k]} {climbing[k]}/{args.seeds}" for k in args.kinds if climbing[k]))
        print(f"  Those runs were still improving at generation {args.generations}, so their "
              f"numbers say where the\n  search had reached, not where it settles. A ranking "
              f"taken here can reverse with more generations.")

    print("\n  Genome sizes above are per individual. One generation of mutation at a fixed rate "
          "explores a\n  very different fraction of each space, so a slower substrate may have a "
          "larger space and not a\n  worse one. Nothing in this table separates those.")

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
