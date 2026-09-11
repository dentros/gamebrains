"""Does the blockwise surrogate reach the size of the test at *this platform's* defaults?

The companion validation study establishes the construction over 12,000-round runs, where 32
blocks leave 375 rounds each. GameBrains ships a 1,500-round default with an epsilon decay of
0.9995, so the same 32 blocks leave about 47 rounds, and epsilon falls from 1.0 to roughly 0.47
across the run. That study is explicit that the construction has one precondition, the marginal
being near-constant inside a block, and equally explicit about what happens when a fast schedule
breaks it: 6.50% becomes 31.75%. Porting the method without measuring it here would be assuming
the precondition holds at a block length four times shorter than the one it was established on.

The design is a null control: every pair is drawn from agents in **separate matches**, so no pair
can have influenced the other and every flagged pair is a false positive by construction. Three
cells, so the answer separates the surrogate from the trimming:

  whole            unrestricted permutation, no trimming      the ablation
  blockwise        within-block permutation, no trimming      the candidate default
  whole+burn_in    unrestricted permutation, transient cut    what this platform shipped

Writes results/te_surrogate_null.json and prints the table. The producing script lives here
because the numbers reach the paper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gamebrains.agents.qlearning import QLearningAgent
from gamebrains.engine.runner import run_match
from gamebrains.games.congestion import CongestionGame
from gamebrains.metrics.information import (
    ALPHA, N_BLOCKS, _te_bits, _block_segments, blockwise_precondition, suggested_burn_in,
    within_block_drift_z,
)

SEEDS = list(range(20))
#: (label, epsilon_decay). The second is deliberately fast: epsilon reaches its floor inside a
#: couple of blocks, so the marginal moves appreciably within one and the construction should
#: break. Without a case where it breaks, a threshold on the drift statistic is invented.
SCHEDULES = [("default", 0.9995), ("fast", 0.95)]
ROUNDS = 1500          # the platform default, which is the whole point of this run
N_STREAMS = 6
N_PAIRS = 10
N_SURROGATES = 200
OUT = _HERE / "results"
LINE = chr(10)


def _streams(seed: int, decay: float) -> tuple[list[np.ndarray], list]:
    """One action series per *separate* match, so any pair of them is independent by construction."""
    series, rosters = [], []
    for k in range(N_STREAMS):
        game = CongestionGame(n_agents=3, num_positions=3, memory_episodes=0)
        roster = [QLearningAgent(name=f"Q{i}", n_states=game.n_states, n_actions=game.n_actions,
                                 alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.02,
                                 epsilon_decay=decay, seed=(seed * 1000 + k) * 100 + i)
                  for i in range(3)]
        rec = run_match(game, roster, rounds=ROUNDS, seed=seed * 1000 + k)
        series.append(rec["actions"][:, 0])
        rosters.append(roster)
    return series, rosters


def _p_value(x: np.ndarray, y: np.ndarray, rng, *, blockwise: bool, burn_in: int = 0) -> float:
    """Surrogate p for T(x -> y). `x` is the source, and the source is what gets permuted."""
    if burn_in:
        x, y = x[burn_in:], y[burn_in:]
    x_prev, y_prev, y_t = x[:-1], y[:-1], y[1:]
    observed = _te_bits(x_prev, y_prev, y_t)

    null = np.empty(N_SURROGATES)
    if blockwise:
        segments = _block_segments(len(x_prev), N_BLOCKS)
        x_s = np.array(x_prev, dtype=np.int64)
        for s in range(N_SURROGATES):
            for seg in segments:
                if len(seg) > 1:
                    x_s[seg] = x_prev[rng.permutation(seg)]
            null[s] = _te_bits(x_s, y_prev, y_t)
    else:
        for s in range(N_SURROGATES):
            null[s] = _te_bits(rng.permutation(x_prev), y_prev, y_t)
    return float((1 + np.sum(null >= observed)) / (1 + N_SURROGATES))


CELLS = ("whole", "blockwise", "whole+burn_in")


def _one_schedule(label: str, decay: float) -> dict:
    hits = {c: 0 for c in CELLS}
    total = 0
    zs: list[float] = []
    raw: list[float] = []
    burn_in = None

    for seed in SEEDS:
        series, rosters = _streams(seed, decay)
        if burn_in is None:
            burn_in = suggested_burn_in(rosters[0], ROUNDS)[0]

        pre = blockwise_precondition({"actions": np.stack(series, axis=1)}, N_BLOCKS)
        zs.append(pre["worst_drift_z"])
        raw.append(pre["worst_raw_shift"])

        rng = np.random.default_rng(seed + 4242)
        pairs = [(i, j) for i in range(N_STREAMS) for j in range(N_STREAMS) if i != j]
        rng.shuffle(pairs)

        for (i, j) in pairs[:N_PAIRS]:
            total += 1
            x, y = series[i], series[j]
            for cell in CELLS:
                # The trimmed cell is only defined while a window survives the cut. Reporting a
                # rate computed on an empty series would be a number naming a property nobody
                # measured, which is the failure this whole module exists to stop.
                if cell.endswith("burn_in") and burn_in >= ROUNDS - 1:
                    continue
                r = np.random.default_rng(seed * 977 + i * 31 + j)
                p = _p_value(x, y, r, blockwise=(cell == "blockwise"),
                             burn_in=burn_in if cell.endswith("burn_in") else 0)
                hits[cell] += int(p < ALPHA)

    trimmed_defined = burn_in < ROUNDS - 1
    rates: dict[str, float | None] = {}
    for c in CELLS:
        rates[c] = None if (c.endswith("burn_in") and not trimmed_defined) \
            else 100.0 * hits[c] / total

    return {
        "schedule": label, "epsilon_decay": decay, "rounds": ROUNDS, "seeds": len(SEEDS),
        "pairs_tested": total, "n_blocks": N_BLOCKS, "rounds_per_block": ROUNDS / N_BLOCKS,
        "derived_burn_in": int(burn_in), "trimmed_cell_defined": trimmed_defined,
        "drift_z_mean": float(np.mean(zs)), "drift_z_max": float(np.max(zs)),
        "raw_shift_mean": float(np.mean(raw)),
        "false_positive_rate_percent": rates, "nominal_percent": 100.0 * ALPHA,
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    runs = [_one_schedule(label, decay) for label, decay in SCHEDULES]
    (OUT / "te_surrogate_null.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")

    for r in runs:
        print(f"{LINE}schedule {r['schedule']!r}, epsilon_decay {r['epsilon_decay']}")
        print(f"  {r['pairs_tested']} independent pairs, {r['seeds']} seeds, {r['rounds']} rounds, "
              f"{N_BLOCKS} blocks of {r['rounds_per_block']:.0f}")
        print(f"  derived burn-in {r['derived_burn_in']:,}"
              + ("" if r["trimmed_cell_defined"] else "  (exceeds the run: no window survives it)"))
        print(f"  within-block drift: {r['drift_z_mean']:.1f} SE mean, {r['drift_z_max']:.1f} max"
              f"   (raw shift {r['raw_shift_mean']:.3f}, which is mostly noise at this length)")
        for cell in CELLS:
            v = r["false_positive_rate_percent"][cell]
            if v is None:
                print(f"    {cell:<16}     n/a   not computable, the cut leaves no series")
                continue
            flag = "  <-- nominal" if abs(v - 100 * ALPHA) <= 2.5 else ""
            print(f"    {cell:<16} {v:6.2f}%{flag}")


if __name__ == "__main__":
    main()
