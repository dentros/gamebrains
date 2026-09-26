"""Run a stored experiment again from its own declaration, and append the result.

A record carries everything needed to rebuild the run that produced it: the game and its
parameters, the roster with each agent's construction-time parameters, the round budget and the
master seed. That is deliberate -- it is the same property `experiments/sweep.py` relies on when a
worker computes a cell and the parent records it -- and it means a stored experiment can be
repeated without anyone reconstructing a command line from memory.

**Nothing is edited.** A signed, hash-chained record is not correctable in place, and a tool that
rewrote one would defeat the ledger it lives in. The rerun appends a *new* record. The old one
stays exactly where it is, and lineage detection marks the new one as a replication, so the pair
is legible as what it is: the same declared experiment, run twice.

Written for a concrete need. Nine congestion records carry a cooperation rate computed before
`metrics/social.py` learned to read the game's own declaration (CLAUDE.md section 9n), so their
stored number is the complement of the truth. Their event logs are intact and correct, so the
right answer was always recoverable; this produces it as a first-class record rather than as a
footnote somebody has to remember.

    py -m gamebrains.experiments.rerun_record --game congestion
    py -m gamebrains.experiments.rerun_record --hash 7146ec18691e22a6
    py -m gamebrains.experiments.rerun_record --game congestion --dry-run

`--dry-run` rebuilds and replays each run and prints what the record would say, without writing,
which is the honest way to see whether a rerun reproduces before committing it to the chain.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..agents import factory
from ..engine.eventlog import EventLog
from ..engine.runner import run_match
from ..games.congestion import CongestionGame
from ..games.public_goods import PublicGoodsGame
from ..metrics import social
from ..repository.ledger import Ledger

RESULTS_DIR = Path(__file__).resolve().parent / "results"
from ..repository.record import protocol_from_run, record_experiment

#: Parameters `CongestionGame` takes, as `describe()` reports them. Listed rather than inferred
#: so that a game gaining a parameter fails loudly here instead of being silently dropped from a
#: rebuilt run, which would make the rerun a different experiment wearing the same name.
_CONGESTION_ARGS = ("num_positions", "reward_rule", "full_reward", "collapse_at_full",
                    "memory_episodes", "episode_max_rounds")
_PGG_ARGS = ("mpcr", "cost")


def rebuild_game(desc: dict[str, Any]):
    """The game a record describes, with every parameter it declared."""
    name = desc.get("name")
    n_agents = int(desc["n_agents"])
    if name == "public_goods":
        kwargs = {k: desc[k] for k in _PGG_ARGS if desc.get(k) is not None}
        return PublicGoodsGame(n_agents=n_agents, **kwargs)
    if name == "congestion":
        kwargs = {k: desc[k] for k in _CONGESTION_ARGS if desc.get(k) is not None}
        # A custom rule is stored resolved, as a base and an exponent, so it rebuilds exactly
        # rather than through the name it happened to be given.
        if desc.get("reward_rule") == "custom":
            kwargs["reward_base"] = desc.get("reward_base")
            kwargs["reward_exponent"] = desc.get("reward_exponent")
        return CongestionGame(n_agents=n_agents, **kwargs)
    raise ValueError(f"no rebuild path for game {name!r}; add one rather than guessing")


def rebuild_roster(roster_desc: list[dict[str, Any]], game, seed: int) -> list:
    """The agents a record describes, through the one factory every batch tool uses.

    Going through `agents/factory.build` matters more than it looks: the seed offsets it applies
    per seat are part of the reproducibility contract, so a roster rebuilt any other way would
    produce a different match from the same declaration.
    """
    agents = []
    for index, entry in enumerate(roster_desc):
        agents.append(factory.build(entry["kind"], index=index, game=game, seed=seed,
                                    overrides=dict(entry.get("params") or {})))
    return agents


def rerun(record: dict[str, Any], repo_root: Path, write: bool = True) -> dict[str, Any]:
    """Replay one record's declared experiment. Returns what the new record says."""
    game = rebuild_game(record["game"])
    seed = int((record.get("seeds") or {}).get("master", 0))
    rounds = int(record["horizon"]["rounds"])
    roster = rebuild_roster(record["roster"], game, seed)

    # The event log goes to a file because that is what `record_experiment` stores as the run's
    # evidence, and the evidence is the whole point here: a rerun that produced numbers without a
    # log would be exactly as uncheckable as the record it is replacing.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_path = RESULTS_DIR / f"rerun_{stamp}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with EventLog(path=log_path) as log:
        result = run_match(game, roster, rounds=rounds, seed=seed, eventlog=log)
    metrics = social.compute_all(result, game.max_welfare_per_round())

    summary = {
        "cooperation_rate": metrics["cooperation_rate"],
        "efficiency": metrics["efficiency"],
        "payoff_gini": metrics["payoff_gini"],
        "action_entropy_bits": metrics["action_entropy_bits"],
    }
    if write:
        summary["record"] = record_experiment(
            repo_root, game, roster, log_path, rounds=rounds, seed=seed, metrics=metrics,
            protocol=protocol_from_run(result))
    else:
        log_path.unlink(missing_ok=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1] / "repo_store"))
    parser.add_argument("--game", help="rerun every stored record of this game")
    parser.add_argument("--hash", dest="hashes", action="append", default=[],
                        help="rerun the record whose config_hash starts with this; repeatable")
    parser.add_argument("--dry-run", action="store_true",
                        help="replay and print, without appending anything to the chain")
    args = parser.parse_args()

    repo_root = Path(args.repo)
    records = Ledger(repo_root).load_all()
    chosen = [r for r in records
              if (not args.game or (r.get("game") or {}).get("name") == args.game)
              and (not args.hashes or any(str(r.get("config_hash", "")).startswith(h)
                                          for h in args.hashes))]
    if not chosen:
        print("no stored record matches that selection")
        return 1

    print(f"{len(chosen)} record(s) to replay"
          f"{' (dry run, nothing will be written)' if args.dry_run else ''}\n")
    for record in chosen:
        game_desc = record["game"]
        stored = (record.get("metrics_summary") or {}).get("cooperation_rate")
        fresh = rerun(record, repo_root, write=not args.dry_run)
        moved = "" if stored is None else f"   was {stored:.4f}"
        print(f"  {game_desc['name']:11} n={game_desc['n_agents']} "
              f"rounds={record['horizon']['rounds']:<6} "
              f"seed={(record.get('seeds') or {}).get('master')}   "
              f"cooperation {fresh['cooperation_rate']:.4f}{moved}")
    print("\ndone" + ("" if args.dry_run else ", appended to the chain as replications"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
