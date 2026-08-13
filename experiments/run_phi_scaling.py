"""Where PyPhi's parallel cut evaluation starts paying for itself, measured rather than assumed.

This exists because the project got the question wrong twice in opposite directions. Parallel cut
evaluation was first force-disabled repo-wide to dodge a Windows spawn failure, which cost
parallelism everywhere it would have been fine. It was then re-enabled globally, which leaked
worker processes in a re-executed process and, at the animat sizes this project actually uses, was
*slower* than serial anyway. Neither answer was measured.

The finding is that there is a crossover, and it is sharp. Worker start-up on Windows is a fixed
cost of seconds paid per cut, while cut evaluation itself grows with the state space, so below a
few nodes the overhead dominates completely and above it the work does. `metrics/phi_autonomy.py`
consults `PARALLEL_MIN_NODES` for exactly this reason.

Wall-clock on a contended laptop is noisy -- a previous benchmark in this project saw 56% spread on
an operation that had not changed -- so a single timed pair is worthless here. This reports the
median of several repetitions per condition and asserts that the computed value is identical either
way, since a speedup that changed the answer would not be a speedup.

Run:
    python -m gamebrains.experiments.run_phi_scaling
    python -m gamebrains.experiments.run_phi_scaling --hidden 2,3,4 --reps 5
"""

from __future__ import annotations

import argparse
import multiprocessing
import statistics
import time

from ..agents.markov_brain import MarkovBrainAgent
from ..engine import procguard
from ..games.public_goods import PublicGoodsGame
from ..metrics import phi_autonomy      # must precede `import pyphi`: it puts vendor/ on sys.path

import pyphi  # noqa: E402


def measure(n_hidden: int, reps: int, parallel: bool) -> tuple[int, float, list[float], float]:
    """Time `reps` identical Phi computations. Returns (n_nodes, median, all times, phi)."""
    game = PublicGoodsGame(n_agents=3, rounds=10)
    times: list[float] = []
    phi = float("nan")
    n_nodes = 0

    for _ in range(reps):
        brain = MarkovBrainAgent("mb", game.n_states, game.n_actions, n_hidden=n_hidden, seed=7)
        brain.on_match_start(game)
        brain.act(game.reset()[0])
        n_nodes = brain.n_nodes

        # Set the flag *after* phi_autonomy would have settled it, so this script measures the
        # condition it names rather than whatever the module's own threshold chose.
        phi_autonomy._settle_parallelism(n_nodes)
        pyphi.config.PARALLEL_CUT_EVALUATION = parallel

        start = time.time()
        phi = phi_autonomy.compute_phi(brain)["phi"]
        times.append(time.time() - start)

    return n_nodes, statistics.median(times), times, phi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", default="2,3",
                        help="comma-separated hidden-node counts to sweep (default 2,3)")
    parser.add_argument("--reps", type=int, default=3,
                        help="repetitions per condition; the median is reported (default 3)")
    args = parser.parse_args()

    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]

    # Pay the probe's cost once, up front, so it is not charged to the first timed run.
    safe = procguard.spawn_is_safe()
    print(f"spawn probe: {safe} ({procguard.spawn_verdict_reason()})")
    if not safe:
        print("Workers cannot be started here, so the parallel condition is not measurable.")
        return

    print(f"\n{'nodes':>6} {'parallel':>9} {'median s':>10} {'observed':>26}   phi")
    crossover = None
    for n_hidden in hidden:
        medians = {}
        values = {}
        n_nodes = 0
        for parallel in (True, False):
            n_nodes, median, times, phi = measure(n_hidden, args.reps, parallel)
            medians[parallel] = median
            values[parallel] = phi
            observed = "[" + ", ".join(f"{t:.2f}" for t in times) + "]"
            print(f"{n_nodes:>6} {str(parallel):>9} {median:>10.2f} {observed:>26}   {phi:.6f}")

        if values[True] != values[False]:
            raise AssertionError(
                f"parallelism changed the computed value at {n_nodes} nodes: "
                f"{values[True]} vs {values[False]}. That is a correctness bug, not a speedup.")

        ratio = medians[False] / medians[True]
        winner = "parallel" if ratio > 1 else "serial"
        factor = ratio if ratio > 1 else 1 / ratio
        print(f"       -> {winner} wins by {factor:.1f}x, identical value {values[True]:.6f}")
        if winner == "parallel" and crossover is None:
            crossover = n_nodes

    if crossover is None:
        print("\nSerial won at every size measured. Widen --hidden to find the crossover.")
    else:
        print(f"\nCrossover at {crossover} nodes. metrics/phi_autonomy.PARALLEL_MIN_NODES is "
              f"{phi_autonomy.PARALLEL_MIN_NODES}.")

    leftover = len(multiprocessing.active_children())
    print(f"orphaned workers after the sweep: {leftover} (reaped {procguard.reap_orphans()})")


if __name__ == "__main__":
    main()
