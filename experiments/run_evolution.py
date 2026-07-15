"""
Evolve a population of Markov-brain animats to play the N-player Public Goods Game.

Run from the "gametheoretic platform" directory:

    python -m gamebrains.experiments.run_evolution
    python -m gamebrains.experiments.run_evolution --agents 4 --population 20 --generations 40

Reports per-generation fitness (avg/best/worst) live, then computes Φ and causal autonomy (see
metrics.phi_autonomy) for the fittest evolved genome, evaluated at a concrete state reached by
running it one step from a fresh reset (IIT's Φ is always a property of a system *at a state*,
not of the transition rules alone). This last step runs PyPhi's major-complex search and can take
a little while even for these small animats.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ..engine.console import LiveConsole
from ..engine.eventlog import EventLog
from ..engine.evolution import EvolutionConfig, evolve
from ..games.public_goods import PublicGoodsGame


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GameBrains — evolve Markov-brain animats on the Public Goods Game")
    ap.add_argument("--agents", type=int, default=4, help="game.n_agents; must divide --population")
    ap.add_argument("--population", type=int, default=20)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--match-rounds", type=int, default=200)
    ap.add_argument("--n-hidden", type=int, default=2)
    ap.add_argument("--mpcr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-phi", action="store_true", help="skip the Phi/autonomy step at the end")
    args = ap.parse_args()

    def game_factory():
        return PublicGoodsGame(n_agents=args.agents, rounds=args.match_rounds, mpcr=args.mpcr)

    config = EvolutionConfig(
        population_size=args.population, generations=args.generations,
        match_rounds=args.match_rounds, n_hidden=args.n_hidden, seed=args.seed,
    )

    results_dir = Path(__file__).resolve().parents[1] / "results"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"evolution_pgg_n{args.agents}_{stamp}.jsonl"

    console = LiveConsole()
    console.rule("═")
    print(f"  GameBrains · evolving Markov-brain animats · Public Goods Game n={args.agents}")
    print(f"  population={args.population}  generations={args.generations}  "
          f"match_rounds={args.match_rounds}  n_hidden={args.n_hidden}  seed={args.seed}")
    console.rule("═")

    with EventLog(path=log_path) as log:
        log.meta(game=game_factory().describe(), config=vars(config))
        result = evolve(game_factory, config, console=console, eventlog=log)

    print()
    console.rule("═")
    print("  EVOLUTION COMPLETE")
    console.rule()
    print(f"  best fitness: {result.fitness.max():.2f}   avg fitness: {result.fitness.mean():.2f}")
    print(f"  event-log: {log_path}")

    if not args.skip_phi:
        from ..metrics.phi_autonomy import causal_autonomy, compute_phi  # lazy: only needs pyphi here

        print()
        print("  Computing Phi and causal autonomy for the fittest evolved animat "
              "(runs PyPhi's major-complex search, may take a moment)...")
        best = result.best
        best.act(best.start_state)   # settle into one concrete, reachable state to analyze
        phi_result = compute_phi(best)
        autonomy = causal_autonomy(best, n_t=3)
        print(f"  Phi = {phi_result['phi']:.4f}   (main complex node indices: "
              f"{phi_result['complex_nodes']})")
        print(f"  causal autonomy = {autonomy:.4f} bits "
              f"(max possible = {best.n_hidden + best.n_motor} bits)")


if __name__ == "__main__":
    main()
