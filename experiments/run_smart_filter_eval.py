"""Does the Smart Filter's similarity lookup actually retrieve similar experiments?

The lookup used to rank by unweighted Euclidean distance over raw feature vectors, where `rounds`
(10^2 to 10^5) outweighs every other dimension by orders of magnitude. It returned results and
nobody checked whether they were the right ones. This script checks, and it is the evidence behind
the change in `repository/smart_filter.py`.

Two phases, deliberately different in kind, because they answer different questions and only one of
them can be answered with invented data.

**Phase 1, correctness, on real runs.** A grid of genuine matches is played and recorded through
the normal pipeline: real games, real agents, real metrics, real ledger records. Then every record
is used as a leave-one-out query and we ask what fraction of its k nearest neighbours share its
*design* rather than merely its length. Fabricated feature vectors could not support this claim,
because the thing being tested is whether the recorded representation of a real experiment is
informative, and a synthetic vector assumes the answer.

**Phase 2, scale, on synthetic records.** How long a lookup takes against a large ledger is a
property of the arithmetic, not of the experiments, so synthetic vectors are appropriate here and
real ones would be an absurd way to spend a week of compute. The sizes go far beyond anything the
correctness phase could reach, which is the point of separating them.

Run:
    python -m gamebrains.experiments.run_smart_filter_eval
    python -m gamebrains.experiments.run_smart_filter_eval --sizes 10000,100000,1000000
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from ..agents.classic import AllD
from ..agents.dqn import DQNAgent
from ..agents.qlearning import QLearningAgent
from ..engine.eventlog import EventLog
from ..engine.runner import run_match
from ..games.public_goods import PublicGoodsGame
from ..metrics import social
from ..repository import record as record_mod
from ..repository import smart_filter
from ..repository.ledger import Ledger

# A grid chosen so that design and scale vary independently. That independence is the whole
# experiment: if every long run also had a distinctive design, a length-dominated distance would
# score well by accident and tell us nothing.
GRID_N = (3, 5)
GRID_MPCR = (0.4, 0.6)
GRID_ROUNDS = (200, 1000, 5000)
GRID_MIX = ("all-q", "mixed")


def _roster(mix: str, game: PublicGoodsGame, seed: int) -> list:
    if mix == "all-q":
        return [QLearningAgent(f"Q{i}", game.n_states, game.n_actions, seed=seed * 10 + i)
                for i in range(game.n_agents)]
    roster = [QLearningAgent("Q0", game.n_states, game.n_actions, seed=seed * 10)]
    roster.append(DQNAgent("D1", game.n_states, game.n_actions, seed=seed * 10 + 1))
    roster.extend(AllD(f"AllD{i}") for i in range(2, game.n_agents))
    return roster


def _design_of(rec: dict) -> tuple:
    """What makes two recorded runs the same experiment, ignoring how long each was observed."""
    fv = rec["feature_vector"]
    return (fv[0], fv[1], fv[2], tuple(fv[4:]))     # n, mpcr, cost, roster composition


def phase1_real_runs(repo_root: Path, k: int) -> None:
    print("Phase 1: playing a real grid and recording it through the normal pipeline")
    ledger = Ledger(repo_root)
    played = 0
    t0 = time.time()

    for n in GRID_N:
        for mpcr in GRID_MPCR:
            if not (1.0 / n < mpcr < 1.0):          # the social-dilemma condition
                continue
            for rounds in GRID_ROUNDS:
                for mix in GRID_MIX:
                    seed = played + 1
                    game = PublicGoodsGame(n_agents=n, rounds=rounds, mpcr=mpcr)
                    roster = _roster(mix, game, seed)
                    log = repo_root / f"run_{played:03d}.jsonl"

                    # A real event log, because record_experiment packages the log itself and a
                    # record without one would not be the artefact the pipeline actually produces.
                    eventlog = EventLog(log)
                    result = run_match(game, roster, rounds=rounds, seed=seed, eventlog=eventlog)
                    eventlog.close()
                    metrics = social.compute_all(result, game.max_welfare_per_round())
                    record_mod.record_experiment(
                        repo_root, game, roster, log, rounds, seed, metrics,
                        protocol=record_mod.protocol_from_run(result))
                    played += 1
                    print(f"  {played:3d}. n={n} mpcr={mpcr} rounds={rounds:>5} mix={mix}")

    print(f"  {played} real matches recorded in {time.time() - t0:.1f}s\n")

    records = ledger.load_all()
    print(f"Phase 1 evaluation: leave-one-out over {len(records)} records, k={k}")

    def raw_nearest(pool, query, kk):
        """The old behaviour, kept here rather than in the module: unweighted Euclidean on raw
        features. Reproduced so the comparison is against what actually shipped."""
        q = np.asarray(query, dtype=float)
        scored = [(r, float(np.linalg.norm(np.asarray(r["feature_vector"], float) - q)))
                  for r in pool]
        scored.sort(key=lambda p: p[1])
        return scored[:kk]

    results = {}
    for label, fn in (("unweighted raw (old)", raw_nearest),
                      ("standardised weighted (new)", smart_filter.nearest)):
        design_hits = 0
        length_only = 0
        total = 0
        for held_out in records:
            pool = [r for r in records if r["content_cid"] != held_out["content_cid"]]
            neighbours = fn(pool, held_out["feature_vector"], k)
            want = _design_of(held_out)
            for neighbour, _distance in neighbours:
                total += 1
                same_design = _design_of(neighbour) == want
                same_length = neighbour["feature_vector"][3] == held_out["feature_vector"][3]
                design_hits += same_design
                length_only += (same_length and not same_design)
        results[label] = (design_hits / total, length_only / total)
        print(f"  {label:30s} same design {design_hits / total:6.1%}   "
              f"same length but different design {length_only / total:6.1%}")

    # A rate is uninterpretable without its ceiling. Each design in this grid is present at three
    # round counts, so any query has exactly two same-design siblings available and a top-5 list
    # cannot contain more than two of them however good the metric is. Reporting 40% without this
    # line would read as a middling score when it is in fact the maximum.
    ceiling = float(np.mean([
        min(k, sum(1 for r in records
                   if r["content_cid"] != q["content_cid"] and _design_of(r) == _design_of(q)))
        for q in records
    ])) / k

    old = results["unweighted raw (old)"][0]
    new = results["standardised weighted (new)"][0]
    print(f"\n  Ceiling for this grid at k={k}: {ceiling:.1%} "
          f"(only so many same-design records exist to be found)")
    print(f"  Design-relevant retrieval went from {old:.1%} to {new:.1%}, "
          f"which is {new / ceiling:.0%} of what is attainable.")
    if new <= old:
        print("  The change did not help on this grid. Do not ship it on the strength of the "
              "argument alone; the argument was what produced the old version too.")


def phase2_scale(sizes: tuple[int, ...], k: int) -> None:
    print("\nPhase 2: lookup cost against synthetic ledgers, orders of magnitude beyond phase 1")
    rng = np.random.default_rng(0)

    print(f"  {'records':>10} {'build s':>9} {'lookup ms':>11} {'per record us':>15}")
    for size in sizes:
        n = rng.integers(2, 11, size=size).astype(float)
        mpcr = rng.uniform(0.2, 0.9, size=size)
        cost = np.ones(size)
        rounds = rng.choice([200, 1000, 5000, 20000, 100000], size=size).astype(float)
        kinds = rng.integers(0, 5, size=(size, 5)).astype(float)
        matrix = np.column_stack([n, mpcr, cost, rounds, kinds])

        t = time.time()
        pool = [{"feature_vector": row.tolist(), "content_cid": f"synthetic-{i}"}
                for i, row in enumerate(matrix)]
        build = time.time() - t

        query = matrix[0].tolist()
        t = time.time()
        out = smart_filter.nearest(pool, query, k)
        lookup_ms = (time.time() - t) * 1000

        assert len(out) == k, f"expected {k} neighbours, got {len(out)}"
        print(f"  {size:>10,} {build:>9.2f} {lookup_ms:>11.1f} {lookup_ms * 1000 / size:>15.2f}")

    print("  Linear in ledger size, as a full scan must be. A ledger large enough for this to "
          "matter needs an index, which is future work and not a distance-metric question.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="neighbours per lookup (default 5)")
    parser.add_argument("--sizes", default="1000,10000,100000",
                        help="synthetic ledger sizes for phase 2")
    parser.add_argument("--keep", action="store_true",
                        help="keep the temporary repository instead of deleting it")
    args = parser.parse_args()

    repo_root = Path(tempfile.mkdtemp(prefix="gb_sf_eval_"))
    try:
        phase1_real_runs(repo_root, args.k)
        phase2_scale(tuple(int(x) for x in args.sizes.split(",") if x.strip()), args.k)
    finally:
        if args.keep:
            print(f"\nrepository kept at {repo_root}")
        else:
            shutil.rmtree(repo_root, ignore_errors=True)


if __name__ == "__main__":
    main()
