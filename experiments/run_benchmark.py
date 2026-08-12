"""
Deterministic cost measurements: adapter overhead, per-architecture work, and metric placement.

Regenerates the cost table in the platform paper. These are properties of the software rather than of
the agents, which is why they live apart from the behavioural experiments.

    python -m gamebrains.experiments.run_benchmark
    python -m gamebrains.experiments.run_benchmark --rounds 500 --json out.json

**Why this script counts instead of timing.** The first version of it measured wall-clock seconds. On
an ordinary contended desktop that turned out to measure the machine, not the software: the same
native operation came out 56% apart in two sections of a single run, and the adapter overhead swung
from +30% to -2% depending on run length, because a torch-heavy section earlier in the process leaves
the CPU in a different thermal and frequency state. Reporting a few-percent effect measured on a noise
floor of tens of percent would have been indefensible.

So the quantities here are deterministic counts rather than durations:

* **Python-level calls**, from `cProfile`'s own call counter. Exactly reproducible on any machine, and
  the natural unit for "what does this indirection cost", since adapter overhead is precisely a
  question of how many extra calls sit between a caller and a game step.
* **Peak allocation**, from `tracemalloc`. Reproducible to within allocator details.
* **Table sizes**, computed from the state and action counts. Exact arithmetic, no measurement at all.

A reader who wants seconds can multiply by their own machine's per-call cost. A reader who wants to
check our numbers can rerun this and get the same integers, which is not true of any timing we could
have published.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import tracemalloc
from typing import Any, Callable, Optional

import numpy as np

from ..agents.classic import AllD
from ..agents.dqn import DQNAgent
from ..agents.fep import FEPAgent
from ..agents.markov_brain import MarkovBrainAgent
from ..agents.qlearning import QLearningAgent
from ..engine.runner import run_match
from ..games.public_goods import PublicGoodsGame
from ..metrics import social

N_AGENTS = 4          # a power of 2, so the Markov brain's motor encoding is valid
MPCR = 0.5


def _game(rounds: int) -> PublicGoodsGame:
    return PublicGoodsGame(n_agents=N_AGENTS, rounds=rounds, mpcr=MPCR)


def _roster(kind: str, game: PublicGoodsGame, seed: int = 0) -> list:
    if kind == "classic":
        return [AllD(f"A{i}") for i in range(N_AGENTS)]
    if kind == "qlearning":
        return [QLearningAgent(f"Q{i}", n_states=game.n_states, n_actions=game.n_actions,
                               seed=seed + i) for i in range(N_AGENTS)]
    if kind == "dqn":
        return [DQNAgent(f"D{i}", n_states=game.n_states, n_actions=game.n_actions,
                         seed=seed + i) for i in range(N_AGENTS)]
    if kind == "fep":
        return [FEPAgent(f"F{i}", n_agents=N_AGENTS, mpcr=MPCR, seed=seed + i,
                         start_state=game.start_state) for i in range(N_AGENTS)]
    if kind == "markov_brain":
        return [MarkovBrainAgent.random(f"M{i}", game.n_states, game.n_actions,
                                        seed=seed + i, start_state=game.start_state)
                for i in range(N_AGENTS)]
    raise ValueError(kind)


def call_count(fn: Callable[[], Any]) -> int:
    """Total Python-level calls made by `fn`, from cProfile's deterministic counter.

    The profiler slows execution down, which does not matter: we read `total_calls`, not the clock.
    Re-running this on different hardware gives the same integer.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    fn()
    profiler.disable()
    return pstats.Stats(profiler).total_calls


def peak_kib(fn: Callable[[], Any]) -> float:
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024.0


# --- 1. adapter overhead, in calls per round -----------------------------------------------------

def bench_adapters(rounds: int) -> dict:
    """Adapter plumbing cost with the work held constant.

    Every path performs identical work: all four Q-learning agents choose an action and take an update
    every round. The only difference is whether one seat is driven through an adapter or by the runner.
    For the adapter paths the caller owns one agent and steps it by hand, exactly as an external
    library would.

    An earlier version drove the adapters with externally supplied random actions and made them look
    50% faster than the native runner. That was not a speedup: that version had no agents inside the
    adapter path at all, so it compared four Q-table updates against none.
    """
    from ..interop.gym_adapter import GameBrainsGymEnv
    from ..interop.pettingzoo_adapter import GameBrainsParallelEnv

    def native():
        game = _game(rounds)
        run_match(game, _roster("qlearning", game), rounds=rounds, seed=0)

    def pettingzoo_one_seat():
        game = _game(rounds)
        agents = _roster("qlearning", game)
        env = GameBrainsParallelEnv(game, [None] + agents[1:], controlled_agent_ids=[0])
        mine = agents[0]
        mine.on_match_start(game)
        obs, _ = env.reset(seed=0)
        aid = "agent_0"
        for _ in range(rounds):
            prev = obs[aid]
            action = mine.act(prev)
            obs, rew, _, trunc, _ = env.step({aid: action})
            mine.update(prev, action, rew[aid], obs[aid], trunc[aid])
            if trunc[aid]:
                obs, _ = env.reset()

    def gymnasium_one_seat():
        game = _game(rounds)
        agents = _roster("qlearning", game)
        env = GameBrainsGymEnv(game, [None] + agents[1:], controlled_agent_id=0)
        mine = agents[0]
        mine.on_match_start(game)
        obs, _ = env.reset(seed=0)
        for _ in range(rounds):
            prev = obs
            action = mine.act(prev)
            obs, rew, _, trunc, _ = env.step(action)
            mine.update(prev, action, rew, obs, trunc)
            if trunc:
                obs, _ = env.reset()

    paths = {"native_runner": native, "pettingzoo_adapter": pettingzoo_one_seat,
             "gymnasium_adapter": gymnasium_one_seat}
    counts = {name: call_count(fn) for name, fn in paths.items()}
    base = counts["native_runner"]
    return {name: {"calls": c, "calls_per_round": c / rounds,
                   "extra_calls_per_round": (c - base) / rounds,
                   "overhead_vs_native": (c - base) / base}
            for name, c in counts.items()}


# --- 2. per-architecture work --------------------------------------------------------------------

def bench_architectures(rounds: int) -> dict:
    out = {}
    for kind in ("classic", "qlearning", "markov_brain", "fep", "dqn"):
        def one(kind=kind):
            game = _game(rounds)
            run_match(game, _roster(kind, game), rounds=rounds, seed=0)
        calls = call_count(one)
        out[kind] = {"calls": calls,
                     "calls_per_agent_step": calls / (rounds * N_AGENTS),
                     "peak_kib": peak_kib(one)}
    base = out["classic"]["calls_per_agent_step"]
    for r in out.values():
        r["relative_to_classic"] = r["calls_per_agent_step"] / base
    return out


# --- 3. where the metric cost falls --------------------------------------------------------------

def bench_metrics(rounds: int) -> dict:
    """Metrics are post-hoc by design, so their cost is separable. This shows the separation."""
    game = _game(rounds)
    records = run_match(game, _roster("qlearning", game), rounds=rounds, seed=0)
    max_welfare = game.max_welfare_per_round()

    def match_only():
        g = _game(rounds)
        run_match(g, _roster("qlearning", g), rounds=rounds, seed=0)

    m = call_count(match_only)
    a = call_count(lambda: social.compute_all(records, max_welfare))
    return {"match_calls": m, "analysis_calls": a, "analysis_as_fraction_of_match": a / m,
            "analysis_calls_per_round": a / rounds}


# --- 4. exact table sizes, no measurement at all -------------------------------------------------

def table_sizes() -> dict:
    """Q-table footprint is arithmetic, not measurement: n_states x n_actions x 8 bytes per agent."""
    rows = []
    for n in (3, 5, 10):
        for m in (0, 1):
            states = 3 ** n * 2 ** (n * m)          # congestion family, 3 positions
            rows.append({"agents": n, "memory_episodes": m, "states": states,
                         "kib_per_agent": states * 2 * 8 / 1024.0,
                         "accepted": states <= 5_000_000})
    return {"congestion_q_table": rows}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    print(f"Deterministic counts, {N_AGENTS} agents, {args.rounds} rounds. "
          f"Python {sys.version.split()[0]}.")
    print("These integers are reproducible on any machine; no wall-clock is reported.\n")

    arch = bench_architectures(args.rounds)
    print(f"  {'architecture':16s} {'calls/agent-step':>18} {'vs classic':>12} {'peak KiB':>10}")
    print("  " + "-" * 60)
    for kind, r in sorted(arch.items(), key=lambda kv: kv[1]["calls_per_agent_step"]):
        print(f"  {kind:16s} {r['calls_per_agent_step']:18,.1f} "
              f"{r['relative_to_classic']:11,.0f}x {r['peak_kib']:10,.0f}")

    ad = bench_adapters(args.rounds)
    print(f"\n  {'drive path':22s} {'calls/round':>13} {'extra/round':>13} {'overhead':>10}")
    print("  " + "-" * 62)
    for name, r in ad.items():
        if name == "native_runner":
            print(f"  {name:22s} {r['calls_per_round']:13,.1f} {'baseline':>13} {'':>10}")
        else:
            print(f"  {name:22s} {r['calls_per_round']:13,.1f} "
                  f"{r['extra_calls_per_round']:+13,.1f} {r['overhead_vs_native']:+9.1%}")

    me = bench_metrics(args.rounds)
    print(f"\n  match calls        : {me['match_calls']:,}")
    print(f"  metric suite calls : {me['analysis_calls']:,} "
          f"({me['analysis_as_fraction_of_match']:.2%} of the match, paid after it)")

    ts = table_sizes()
    print(f"\n  {'agents':>7} {'memory':>7} {'states':>12} {'KiB/agent':>11} {'accepted':>9}")
    print("  " + "-" * 52)
    for row in ts["congestion_q_table"]:
        print(f"  {row['agents']:7d} {row['memory_episodes']:7d} {row['states']:12,} "
              f"{row['kib_per_agent']:11,.0f} {'yes' if row['accepted'] else 'no':>9}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as h:
            json.dump({"config": {"n_agents": N_AGENTS, "rounds": args.rounds,
                                  "python": sys.version.split()[0]},
                       "architectures": arch, "adapters": ad, "metrics": me,
                       "table_sizes": ts}, h, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
