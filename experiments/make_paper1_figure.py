"""
One-off script: render the cooperation-collapse figure for Paper 1 (SMC 2025 GAMEBRAINS) from a
saved run_pgg.py event-log. Not part of the platform proper -- a paper-figure utility, kept here
rather than in paper_figures/ (which doesn't exist yet) since this is currently the only one.

Usage: "./gamebrains/.venv/Scripts/python.exe" -m gamebrains.experiments.make_paper1_figure <log.jsonl> <out.png>
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    log_path, out_path = sys.argv[1], sys.argv[2]
    coop_fraction = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") == "round":
                n = len(rec["actions"])
                coop_fraction.append(rec["cooperators"] / n)
    coop = np.array(coop_fraction)

    window = 50
    kernel = np.ones(window) / window
    smoothed = np.convolve(coop, kernel, mode="valid")

    plt.figure(figsize=(4.2, 2.6))
    plt.plot(coop, color="#b7c9e2", linewidth=0.5, label="per-round")
    plt.plot(np.arange(window - 1, len(coop)), smoothed, color="#1f5c99", linewidth=1.8,
             label=f"{window}-round rolling mean")
    plt.xlabel("round")
    plt.ylabel("cooperation rate")
    plt.ylim(-0.02, 1.02)
    plt.legend(loc="upper right", fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
