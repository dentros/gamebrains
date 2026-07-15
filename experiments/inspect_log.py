"""
Inspect a saved GameBrains event-log (.jsonl) after the fact — the "inner workings" viewer.

The live console only samples a handful of rounds and shows one brain snapshot at a time as a
run happens; this script reads the *entire* stored log (every round, every recorded brain
snapshot) and gives a full retrospective view: the complete cooperation-rate time series as a
plot, and every brain snapshot that was captured during the run.

Usage (from the "gametheoretic platform" directory):
    python -m gamebrains.experiments.inspect_log gamebrains/results/pgg_mixed_n5_....jsonl
    python -m gamebrains.experiments.inspect_log <path.jsonl> --out cooperation.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # match engine.console on Windows
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_events(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def print_header(meta: dict) -> None:
    game = meta.get("game", {})
    print("=" * 74)
    print(f"  GameBrains event-log — {game.get('name', '?')}")
    print(f"  game: {game.get('name')}  n_agents={game.get('n_agents')}  "
          f"rounds={meta.get('rounds')}  seed={meta.get('seed')}")
    print("-" * 74)
    print("  roster:")
    for r in meta.get("roster", []):
        print(f"    [{r['idx']}] {r['name']:<22} kind={r['kind']:<12} mode={r['training_mode']}")
    print("=" * 74)


def plot_cooperation(events: list[dict], out_path: Path) -> None:
    rounds_data = [e for e in events if e.get("type") == "round"]
    if not rounds_data:
        print("  (no round events found — nothing to plot)")
        return

    n_agents = len(rounds_data[0]["actions"])
    coop_fraction = np.array([e["cooperators"] / n_agents for e in rounds_data])
    x = np.arange(len(coop_fraction))

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(x, coop_fraction, linewidth=0.8, alpha=0.5, label="per-round")
    window = max(1, len(x) // 100)
    if window > 1:
        smoothed = np.convolve(coop_fraction, np.ones(window) / window, mode="valid")
        ax.plot(x[window - 1:], smoothed, linewidth=2.0, label=f"rolling mean (w={window})")
    ax.set_xlabel("round")
    ax.set_ylabel("cooperation fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Cooperation over the full match")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  cooperation plot saved: {out_path}")


def print_brain_snapshots(events: list[dict]) -> None:
    snapshots = [e for e in events if e.get("type") == "brain_snapshot"]
    if not snapshots:
        print("  (no brain_snapshot events recorded)")
        return
    print(f"  {len(snapshots)} brain snapshot(s) recorded:")
    for snap in snapshots:
        brain = snap["brain"]
        kind = brain.get("kind", "?")
        extra = ""
        if kind in ("qlearning", "dqn"):
            extra = f"epsilon={brain.get('epsilon', 0):.3f}"
        elif kind == "fep":
            extra = f"E[others]={brain.get('expected_others', 0):.2f}"
        print(f"    round {snap['round']:>6}  agent[{snap['agent']}] {snap['name']:<20} "
              f"kind={kind:<10} {extra}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect a GameBrains event-log")
    ap.add_argument("log_path", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="output path for the cooperation plot (default: alongside the log)")
    args = ap.parse_args()

    events = load_events(args.log_path)
    meta = next((e for e in events if e.get("type") == "meta"), {})

    print_header(meta)
    print_brain_snapshots(events)
    print("-" * 74)

    out_path = args.out or args.log_path.with_suffix(".png")
    plot_cooperation(events, out_path)


if __name__ == "__main__":
    main()
