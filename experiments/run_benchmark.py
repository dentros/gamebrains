"""
Deterministic cost measurements: per-architecture work, adapter overhead, the metric suite, and the
tabular state ceiling.

Regenerates the cost table and the state-count table in the platform paper. These are properties of
the software rather than of the agents, which is why they live apart from the behavioural experiments.

    python -m gamebrains.experiments.run_benchmark
    python -m gamebrains.experiments.run_benchmark --repeats 3 --json out.json
    python -m gamebrains.experiments.run_benchmark --states --states-max-n 8

**Why this script counts instead of timing.** The first version of it measured wall-clock seconds. On
an ordinary contended desktop that turned out to measure the machine, not the software: the same
native operation came out 56% apart in two sections of a single run, and the adapter overhead swung
from +30% to -2% depending on run length, because a torch-heavy section earlier in the process leaves
the CPU in a different thermal and frequency state. So the quantities here are counts:

* **Python-level calls**, from `cProfile`'s own call counter.
* **Peak allocation**, from `tracemalloc`.
* **Table sizes**, computed from the state and action counts. Exact arithmetic, no measurement.

**What an earlier version got wrong, found by an external review that reran it.**

1. The per-architecture count included building the agents, and ran with no warm-up, so the first
   architecture measured also paid for lazy imports. Tabular Q therefore read 13.1 calls per
   agent-step, 52.4 per round, while the adapter section measured the *same* match at 44.9 per round.
   Construction is now counted separately, every path is run once before anything is counted, and
   the script checks that the native runner and the Q-learning row agree, since they are one match.
2. "The full metric suite" was `social.compute_all` alone. It is now every measure the results page
   computes after a match: social, information-theoretic, graph, and on the congestion family the
   alternation family, each reported separately.
3. The counts were said to be reproducible "to the integer", and a rerun of the DQN row came out 0.4
   calls per agent-step different. Every measurement is now repeated (`--repeats`, default 2) and the
   spread is reported next to it, with garbage collection held off during counting so that finaliser
   timing cannot add calls. A row whose spread is not zero is printed as such rather than rounded away.
4. The library versions behind the counts were not recorded. They are now, since a different torch
   or numpy is a different call graph.
5. The state-count table had no script. `--states` regenerates it.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import json
import os
import platform
import pstats
import sys
import tracemalloc
from typing import Any, Callable, Optional

import numpy as np

from ..agents.classic import AllD
from ..agents.fep import FEPAgent
from ..agents.markov_brain import MarkovBrainAgent
from ..agents.qlearning import QLearningAgent
from ..engine.runner import run_match
from ..engine.semantics import bind_agents
from ..games.congestion import from_preset as congestion_from_preset
from ..games.public_goods import PublicGoodsGame
from ..metrics import graph, information, social, social_alt

N_AGENTS = 4          # a power of 2, so the Markov brain's motor encoding is valid
MPCR = 0.5
KINDS = ("classic", "qlearning", "markov_brain", "fep", "dqn")
#: Long enough for every architecture to reach its steady path. At 30 rounds a DQN's replay buffer
#: never reaches its batch of 64, so its optimizer had not run once before the first counted match,
#: and that match paid 152 calls of first-use setup the second did not. Measured, both ways.
WARM_UP_ROUNDS = 100


def _game(rounds: int) -> PublicGoodsGame:
    return PublicGoodsGame(n_agents=N_AGENTS, rounds=rounds, mpcr=MPCR)


def _roster(kind: str, game: Any, seed: int = 0) -> list:
    if kind == "classic":
        return [AllD(f"A{i}") for i in range(N_AGENTS)]
    if kind == "qlearning":
        return [QLearningAgent(f"Q{i}", n_states=game.n_states, n_actions=game.n_actions,
                               seed=seed + i) for i in range(N_AGENTS)]
    if kind == "dqn":
        from ..agents.dqn import DQNAgent
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

    Garbage collection is run beforehand and held off during the count. A collection that happens to
    fire mid-measurement runs Python-level finalisers, and when it fires depends on allocation history
    rather than on the work being measured.
    """
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        fn()
        profiler.disable()
    finally:
        if was_enabled:
            gc.enable()
    return pstats.Stats(profiler).total_calls


def repeated(prepare: Callable[[], Callable[[], Any]], repeats: int) -> dict:
    """Count `repeats` fresh instances of one measurement. `prepare` builds whatever the measured
    callable needs (a game, a roster) outside the profiler and returns the callable itself."""
    runs = [call_count(prepare()) for _ in range(repeats)]
    return {"runs": runs, "mean": float(np.mean(runs)), "spread": int(max(runs) - min(runs))}


def peak_kib(fn: Callable[[], Any]) -> float:
    gc.collect()
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1024.0


def versions() -> dict:
    out = {"python": sys.version.split()[0], "numpy": np.__version__,
           "platform": platform.platform(), "machine": platform.machine(),
           "processor": platform.processor(), "cpu_count": os.cpu_count()}
    for name in ("torch", "networkx", "scipy", "statsmodels"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = None
    try:
        from ..repository.record import DEFAULT_CODE_VERSION
        out["code_version"] = DEFAULT_CODE_VERSION
    except Exception:  # the cryptography dependency is optional for a cost measurement
        out["code_version"] = "unknown"
    return out


def _metric_suite(records: dict, game: Any, roster: list) -> dict[str, Callable[[], Any]]:
    """The measures the results page computes after a match, as separately countable callables."""
    te = information.compute_all(records, seed=0, roster=roster)["transfer_entropy_detail"]
    suite = {
        "social": lambda: social.compute_all(records, game.max_welfare_per_round()),
        "information": lambda: information.compute_all(records, seed=0, roster=roster),
        "graph": lambda: graph.compute_all(records, te),
    }
    episodes = records.get("episodes", [])
    if len(episodes) > 1:
        suite["alternation"] = lambda: social_alt.compute_all(
            episodes, n_agents=N_AGENTS, full_reward=getattr(game, "full_reward", 100.0))
    return suite


def warm_up() -> None:
    """Run every measured path once, briefly and uncounted, so that no count pays for a first-time
    import or a cache fill that later calls do not repeat."""
    for kind in KINDS:
        game = _game(WARM_UP_ROUNDS)
        _roster(kind, game)
        run_match(game, _roster(kind, game), rounds=WARM_UP_ROUNDS, seed=0)
    for make in (lambda: _game(WARM_UP_ROUNDS), lambda: congestion_from_preset("hjg", N_AGENTS)):
        game = make()
        roster = _roster("qlearning", game)
        records = run_match(game, roster, rounds=WARM_UP_ROUNDS * 4, seed=0)
        for fn in _metric_suite(records, game, roster).values():
            fn()
    for drive in _adapter_paths(WARM_UP_ROUNDS).values():
        drive()()


# --- 1. per-architecture work --------------------------------------------------------------------

def bench_architectures(rounds: int, repeats: int) -> dict:
    out = {}
    for kind in KINDS:
        def prepare_match(kind=kind):
            game = _game(rounds)
            roster = _roster(kind, game)
            return lambda: run_match(game, roster, rounds=rounds, seed=0)

        def prepare_build(kind=kind):
            game = _game(rounds)
            return lambda: _roster(kind, game)

        def build_and_play(kind=kind):
            game = _game(rounds)
            run_match(game, _roster(kind, game), rounds=rounds, seed=0)

        match = repeated(prepare_match, repeats)
        build = repeated(prepare_build, repeats)
        out[kind] = {"match": match, "construction": build,
                     "calls_per_agent_step": match["mean"] / (rounds * N_AGENTS),
                     "construction_calls_per_agent": build["mean"] / N_AGENTS,
                     "peak_kib": peak_kib(build_and_play)}
    base = out["classic"]["calls_per_agent_step"]
    for r in out.values():
        r["relative_to_classic"] = r["calls_per_agent_step"] / base
    return out


# --- 2. adapter overhead, in calls per round -----------------------------------------------------

def _adapter_paths(rounds: int) -> dict[str, Callable[[], Callable[[], Any]]]:
    """Each entry builds its game, agents and environment, then returns the drive loop to count.

    Every path performs identical work: all four Q-learning agents choose an action and take an update
    every round. The only difference is whether one seat is driven through an adapter or by the runner.
    An earlier version drove the adapters with externally supplied random actions and made them look
    50% faster than the native runner, because that version had no agents inside the adapter path.
    """
    from ..interop.gym_adapter import GameBrainsGymEnv
    from ..interop.pettingzoo_adapter import GameBrainsParallelEnv

    def native():
        game = _game(rounds)
        roster = _roster("qlearning", game)
        return lambda: run_match(game, roster, rounds=rounds, seed=0)

    def pettingzoo_one_seat():
        game = _game(rounds)
        agents = _roster("qlearning", game)
        env = GameBrainsParallelEnv(game, [None] + agents[1:], controlled_agent_ids=[0])
        mine, aid = agents[0], "agent_0"

        def drive():
            bind_agents(game, [mine])
            obs, _ = env.reset(seed=0)
            for _ in range(rounds):
                prev = obs[aid]
                action = mine.act(prev)
                obs, rew, _, trunc, _ = env.step({aid: action})
                mine.update(prev, action, rew[aid], obs[aid], trunc[aid])
                if trunc[aid]:
                    obs, _ = env.reset()
        return drive

    def gymnasium_one_seat():
        game = _game(rounds)
        agents = _roster("qlearning", game)
        env = GameBrainsGymEnv(game, [None] + agents[1:], controlled_agent_id=0)
        mine = agents[0]

        def drive():
            bind_agents(game, [mine])
            obs, _ = env.reset(seed=0)
            for _ in range(rounds):
                prev = obs
                action = mine.act(prev)
                obs, rew, _, trunc, _ = env.step(action)
                mine.update(prev, action, rew, obs, trunc)
                if trunc:
                    obs, _ = env.reset()
        return drive

    return {"native_runner": native, "pettingzoo_adapter": pettingzoo_one_seat,
            "gymnasium_adapter": gymnasium_one_seat}


def bench_adapters(rounds: int, repeats: int) -> dict:
    counts = {name: repeated(prepare, repeats) for name, prepare in _adapter_paths(rounds).items()}
    base = counts["native_runner"]["mean"]
    return {name: {**c, "calls_per_round": c["mean"] / rounds,
                   "extra_calls_per_round": (c["mean"] - base) / rounds,
                   "overhead_vs_native": (c["mean"] - base) / base}
            for name, c in counts.items()}


# --- 3. where the metric cost falls --------------------------------------------------------------

def bench_metrics(rounds: int, repeats: int) -> dict:
    """Metrics are post-hoc by design, so their cost is separable. Measured on a Q-learning roster in
    both game families, since the alternation measures exist only for the congestion one."""
    out = {}
    for label, make in (("public_goods", lambda: _game(rounds)),
                        ("congestion_hjg", lambda: congestion_from_preset("hjg", N_AGENTS))):
        def prepare_match(make=make):
            game = make()
            roster = _roster("qlearning", game)
            return lambda: run_match(game, roster, rounds=rounds, seed=0)

        game = make()
        roster = _roster("qlearning", game)
        records = run_match(game, roster, rounds=rounds, seed=0)
        suite = _metric_suite(records, game, roster)
        match = repeated(prepare_match, repeats)
        families = {name: repeated(lambda fn=fn: fn, repeats) for name, fn in suite.items()}
        total = sum(f["mean"] for f in families.values())
        out[label] = {"match": match, "families": families, "analysis_calls": total,
                      "analysis_as_fraction_of_match": total / match["mean"],
                      "episodes": len(records.get("episodes", []))}
    return out


# --- 4. exact table sizes, no measurement at all -------------------------------------------------

def table_sizes() -> dict:
    """Q-table footprint is arithmetic, not measurement: n_states x n_actions x 8 bytes per agent."""
    rows = []
    for n in (3, 5, 10):
        for m in (0, 1):
            states = 3 ** n * 2 ** (n * m)          # congestion family, 3 positions
            rows.append({"agents": n, "memory_episodes": m, "states": states,
                         "kib_per_agent": states * 2 * 8 / 1024.0,
                         "kib_per_roster": n * states * 2 * 8 / 1024.0,
                         "accepted": states <= 5_000_000})
    return {"congestion_q_table": rows}


# --- 5. the state ceiling, run rather than projected ---------------------------------------------

def bench_states(max_n: int, rounds: int = 3000) -> dict:
    """The paper's state-count table: n tabular Q-learners in the congestion family with one remembered
    episode, a `rounds`-round match each, peak allocation traced over construction and play.

    n = 8 traces about a gigabyte, so the default stops at 7 and the row is opt-in. Configurations
    above the platform's five-million-state limit are refused by the game's constructor, and are
    reported as refused without allocating anything.
    """
    rows = []
    for n in range(3, 11):
        try:
            game = congestion_from_preset("hjg", n, memory_episodes=1)
        except ValueError as exc:
            rows.append({"agents": n, "states": 3 ** n * 2 ** n, "outcome": "refused",
                         "reason": str(exc)[:160]})
            continue
        q_mib = n * game.n_states * game.n_actions * 8 / 2 ** 20
        if n > max_n:
            rows.append({"agents": n, "states": game.n_states, "q_mib": q_mib,
                         "outcome": "not run", "reason": f"above --states-max-n {max_n}"})
            continue
        del game

        def one(n=n):
            g = congestion_from_preset("hjg", n, memory_episodes=1)
            roster = [QLearningAgent(f"Q{i}", n_states=g.n_states, n_actions=g.n_actions, seed=i)
                      for i in range(n)]
            run_match(g, roster, rounds=rounds, seed=0)

        rows.append({"agents": n, "states": 3 ** n * 2 ** n, "q_mib": q_mib,
                     "peak_mib": peak_kib(one) / 1024.0, "outcome": "ran"})
    return {"rounds": rounds, "rows": rows}


# --- output --------------------------------------------------------------------------------------

def _spread_note(entry: dict) -> str:
    return "exact" if entry["spread"] == 0 else f"spread {entry['spread']:,} calls"


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--states", action="store_true", help="also regenerate the state table")
    parser.add_argument("--states-max-n", type=int, default=7)
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)
    if args.repeats < 2:
        parser.error("--repeats must be at least 2, or the spread it reports means nothing")

    env = versions()
    print(f"Deterministic counts, {N_AGENTS} agents, {args.rounds} rounds, {args.repeats} repeats.")
    print("  " + ", ".join(f"{k} {v}" for k, v in env.items() if k in
                           ("python", "numpy", "torch", "code_version")))
    warm_up()

    arch = bench_architectures(args.rounds, args.repeats)
    print(f"\n  {'architecture':14s} {'calls/agent-step':>17} {'vs classic':>11} "
          f"{'build/agent':>12} {'peak KiB':>9}  repeats")
    print("  " + "-" * 84)
    for kind, r in sorted(arch.items(), key=lambda kv: kv[1]["calls_per_agent_step"]):
        print(f"  {kind:14s} {r['calls_per_agent_step']:17,.2f} "
              f"{r['relative_to_classic']:10,.1f}x {r['construction_calls_per_agent']:12,.0f} "
              f"{r['peak_kib']:9,.0f}  {_spread_note(r['match'])}")

    ad = bench_adapters(args.rounds, args.repeats)
    print(f"\n  {'drive path':20s} {'calls/round':>12} {'extra/round':>12} {'overhead':>9}  repeats")
    print("  " + "-" * 72)
    for name, r in ad.items():
        extra = "baseline" if name == "native_runner" else f"{r['extra_calls_per_round']:+,.2f}"
        over = "" if name == "native_runner" else f"{r['overhead_vs_native']:+.1%}"
        print(f"  {name:20s} {r['calls_per_round']:12,.2f} {extra:>12} {over:>9}  "
              f"{_spread_note(r)}")
    same = ad["native_runner"]["mean"] == arch["qlearning"]["match"]["mean"]
    print(f"\n  native runner and the Q-learning row are one match: "
          f"{'they agree' if same else 'THEY DISAGREE, so one of the two counts is not what it says'}")

    me = bench_metrics(args.rounds, args.repeats)
    for label, r in me.items():
        print(f"\n  {label} ({r['episodes']} episodes): match {r['match']['mean']:,.0f} calls")
        for name, f in r["families"].items():
            print(f"    {name:12s} {f['mean']:12,.0f} calls  {f['mean'] / r['match']['mean']:8.1%} "
                  f"of the match  {_spread_note(f)}")
        print(f"    {'all':12s} {r['analysis_calls']:12,.0f} calls  "
              f"{r['analysis_as_fraction_of_match']:8.1%} of the match, paid after it")

    ts = table_sizes()
    states = bench_states(args.states_max_n) if args.states else None
    if states:
        print(f"\n  {'n':>3} {'states':>12} {'Q MiB':>8} {'peak MiB':>9}  outcome")
        print("  " + "-" * 50)
        for row in states["rows"]:
            q = f"{row['q_mib']:8.1f}" if "q_mib" in row else f"{'n/a':>8}"
            peak = f"{row['peak_mib']:9.1f}" if "peak_mib" in row else f"{'n/a':>9}"
            print(f"  {row['agents']:3d} {row['states']:12,} {q} {peak}  {row['outcome']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as h:
            json.dump({"config": {"n_agents": N_AGENTS, "rounds": args.rounds,
                                  "repeats": args.repeats, "versions": env},
                       "architectures": arch, "adapters": ad, "metrics": me,
                       "table_sizes": ts, "states": states}, h, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
