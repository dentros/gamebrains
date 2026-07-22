"""
Vertical-slice demo: a mixed roster plays the N-player Public Goods Game.

Run from the "gametheoretic platform" directory:

    python -m gamebrains.experiments.run_pgg
    python -m gamebrains.experiments.run_pgg --all qlearning --agents 4 --rounds 4000
    python -m gamebrains.experiments.run_pgg --all classic --classic-strategy MajorityTFT

The default roster mixes learning and fixed brains so you can watch the social dynamics live in the
console; `--all qlearning`/`--all dqn` reproduce the classic free-riding result (cooperation
collapses toward universal defection). `--all classic` fills the roster with n copies of one fixed
strategy (see `--classic-strategy`); `--all fep` with n active-inference agents.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from ..agents.classic import AllC, AllD, MajorityTFT, RandomAgent
from ..agents.qlearning import QLearningAgent
from ..engine.console import LiveConsole
from ..engine.eventlog import EventLog
from ..engine.runner import run_match
from ..games.public_goods import PublicGoodsGame
from ..metrics import graph, information, social
from ..repository.record import record_experiment


def _classic_agent(strategy: str, i: int, n: int, seed: int):
    if strategy == "AllC":
        return AllC(f"AllC {i}")
    if strategy == "AllD":
        return AllD(f"AllD {i}")
    if strategy == "Random":
        return RandomAgent(f"Random {i}", seed=seed + 400 + i)
    if strategy == "MajorityTFT":
        return MajorityTFT(f"TFT {i}", n_agents=n)
    raise ValueError(f"unknown classic strategy {strategy}")


def build_roster(game: PublicGoodsGame, seed: int, all_kind: str | None = None,
                 dqn_count: int = 0, fep_count: int = 0, reciprocity: float = 0.0,
                 classic_strategy: str = "MajorityTFT"):
    n = game.n_agents
    labels = game.state_labels()
    actions = game.action_names

    def ql(i: int, name: str) -> QLearningAgent:
        return QLearningAgent(
            name=name, n_states=game.n_states, n_actions=game.n_actions,
            alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.9995,
            seed=seed + 100 + i, state_labels=labels, action_labels=actions,
        )

    def dqn(i: int, name: str):
        from ..agents.dqn import DQNAgent   # lazy: only import torch when a DQN is requested
        return DQNAgent(
            name=name, n_states=game.n_states, n_actions=game.n_actions,
            gamma=0.95, epsilon=1.0, epsilon_min=0.02, epsilon_decay=0.9995,
            seed=seed + 200 + i, state_labels=labels, action_labels=actions,
        )

    def fep(i: int, name: str):
        from ..agents.fep import FEPAgent
        return FEPAgent(
            name=name, n_agents=n, mpcr=game.mpcr, cost=game.cost,
            reciprocity=reciprocity, seed=seed + 300 + i, start_state=game.start_state,
        )

    if all_kind == "qlearning":
        return [ql(i, f"Q-learner {i}") for i in range(n)]
    if all_kind == "dqn":
        return [dqn(i, f"DeepQ {i}") for i in range(n)]
    if all_kind == "fep":
        return [fep(i, f"FEP {i}") for i in range(n)]
    if all_kind == "classic":
        return [_classic_agent(classic_strategy, i, n, seed) for i in range(n)]

    # Mixed roster: overlay DQN then FEP onto Q-learners, then a few fixed characters at the end.
    roster = [ql(i, f"Q-learner {i}") for i in range(n)]
    idx = 0
    for _ in range(min(dqn_count, n)):
        roster[idx] = dqn(idx, f"DeepQ {idx}"); idx += 1
    for _ in range(min(fep_count, n - idx)):
        roster[idx] = fep(idx, f"FEP {idx}"); idx += 1
    specialists = idx
    if n - specialists >= 1:
        roster[-1] = AllD("Defector")
    if n - specialists >= 2:
        roster[-2] = MajorityTFT("Majority-TFT", n_agents=n)
    if n - specialists >= 3:
        roster[-3] = AllC("Altruist")
    return roster


def main() -> None:
    ap = argparse.ArgumentParser(description="GameBrains — Public Goods Game demo")
    ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--mpcr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", choices=["qlearning", "dqn", "fep", "classic"], default=None,
                    help="make every agent the same kind (qlearning/dqn reproduce free-riding)")
    ap.add_argument("--classic-strategy", choices=["AllC", "AllD", "Random", "MajorityTFT"],
                    default="MajorityTFT", help="which fixed strategy to use with --all classic")
    ap.add_argument("--dqn", type=int, default=0,
                    help="number of DQN agents in the mixed roster (rest Q-learners)")
    ap.add_argument("--fep", type=int, default=0,
                    help="number of FEP / active-inference agents in the mixed roster")
    ap.add_argument("--reciprocity", type=float, default=0.0,
                    help="FEP prosocial preference (>0 -> belief-driven conditional cooperator)")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--repo", type=str, default=None,
                    help="repository-lite root dir to record this run into (default: gamebrains/repo_store; "
                         "pass '' to skip recording)")
    args = ap.parse_args()

    game = PublicGoodsGame(n_agents=args.agents, rounds=args.rounds, mpcr=args.mpcr)
    roster = build_roster(game, seed=args.seed, all_kind=args.all,
                          dqn_count=args.dqn, fep_count=args.fep,
                          reciprocity=args.reciprocity, classic_strategy=args.classic_strategy)

    results_dir = Path(__file__).resolve().parents[1] / "results"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.all if args.all else "mixed"
    log_path = results_dir / f"pgg_{tag}_n{args.agents}_{stamp}.jsonl"

    console = LiveConsole()
    # snapshot the first Q-learner if there is one
    snap_idx = next((i for i, a in enumerate(roster) if a.kind == "qlearning"), 0)

    with EventLog(path=log_path) as log:
        records = run_match(
            game, roster, rounds=args.rounds, seed=args.seed,
            eventlog=log, console=console,
            log_every=args.log_every,
            snapshot_every=max(args.rounds // 3, 1),
            snapshot_agent=snap_idx,
        )

    metrics = social.compute_all(records, game.max_welfare_per_round())
    metrics.update(information.compute_all(records, seed=args.seed))
    metrics.update(graph.compute_all(records, metrics["transfer_entropy_detail"]))
    print()
    # Show one final brain per interesting kind present (Q-table / network / beliefs).
    shown: set[str] = set()
    for ag in roster:
        if ag.kind in ("qlearning", "dqn", "fep") and ag.kind not in shown:
            console.brain_snapshot(ag, title="final brain")
            shown.add(ag.kind)
    print()
    console.summary(metrics)
    print()
    console.leaderboard(roster, metrics["cumulative_payoffs"])
    print(f"\n  event-log: {log_path}")

    coop = metrics["cooperation_series"]
    first100 = float(coop[:100].mean()) if len(coop) >= 1 else 0.0
    last100 = float(coop[-100:].mean()) if len(coop) >= 1 else 0.0
    print(f"  cooperation: first 100 rounds {first100:.3f}  ->  last 100 rounds {last100:.3f}")

    if args.repo != "":
        repo_root = Path(args.repo) if args.repo else Path(__file__).resolve().parents[1] / "repo_store"
        record = record_experiment(repo_root, game, roster, log_path, rounds=args.rounds,
                                   seed=args.seed, metrics=metrics)
        lineage_bits = [k for k, v in record["lineage"].items() if v]
        lineage_txt = ", ".join(lineage_bits) if lineage_bits else "none (first of its kind)"
        print(f"  repository: config_hash={record['config_hash'][:12]}...  "
             f"content_cid={record['content_cid'][:12]}...  lineage: {lineage_txt}")


if __name__ == "__main__":
    main()
