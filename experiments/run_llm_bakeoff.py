"""How does a language model do against the other architectures, on the same board?

The honest answer needs care, because the comparison is rigged by cost before anyone plays. A
reinforcement learner needs thousands of rounds to become anything, and a served model answers in
seconds, so a match long enough to train one is unaffordable for the other. Putting them in one
match untrained would compare a considered policy against noise.

So this reuses the bake-off construction the platform already has (`run_bakeoff.pretrain`): every
other architecture is trained separately in self-play, then frozen, and the evaluation match is
short enough for a model to sit in. What that measures is the *final policies* playing each other,
which is the only comparison the two cost structures allow. It is one match, one model and a few
dozen rounds, so it is a picture rather than a ranking, and the script prints that caveat with the
numbers rather than leaving it to a reader.

    py -m gamebrains.experiments.run_llm_bakeoff --rounds 60
    py -m gamebrains.experiments.run_llm_bakeoff --rounds 60 --model qwen2.5:1.5b

Writes `experiments/results/llm_bakeoff_<host>.json`.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Any

import numpy as np

from ..agents.llm import LLMAgent
from ..agents.llm_ollama import DEFAULT_MODEL, OllamaBackend
from ..engine.runner import run_match
from ..games.public_goods import PublicGoodsGame
from ..metrics import social
from . import run_bakeoff
from .run_bakeoff import MPCR, PRETRAIN_ROUNDS, pretrain
from .run_llm_contract import RESULTS_DIR

#: Four opponents and the model make five seats, which is what the genetic algorithm's population
#: of twenty divides by. `classic` is the one left out, since AllD is a reference point rather than
#: an architecture and the evaluation is already crowded; pass `--opponents` to change the mix.
OPPONENTS = ["qlearning", "dqn", "fep", "markov_brain"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=60,
                        help="evaluation rounds. Every one of them is a model call, so this is "
                             "minutes rather than the thousands the other architectures train for")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--opponents", nargs="*", default=OPPONENTS)
    args = parser.parse_args(argv)

    backend = OllamaBackend(model=args.model)
    if not backend.is_available():
        print(f"no Ollama model {args.model!r}: `ollama serve`, then `ollama pull {args.model}`")
        return 1

    # A tabular policy is a table over the observation space, and in this game that space is the
    # population size plus two, so a policy trained at five seats cannot play at six: the index it
    # would need does not exist. Training therefore happens at the size the evaluation match will
    # have, which is the opponents plus the model. Found by running it, as an IndexError.
    seats = len(args.opponents) + 1
    run_bakeoff.N_AGENTS = seats

    print(f"pretraining {len(args.opponents)} architectures for {PRETRAIN_ROUNDS} rounds each, "
          f"at the {seats} seats they will be evaluated in")
    started = time.time()
    roster: list[Any] = [pretrain(kind, args.seed, quiet=True) for kind in args.opponents]
    roster.append(LLMAgent(f"LLM ({args.model})", backend=backend))
    print(f"  done in {time.time() - started:.0f}s\n")

    game = PublicGoodsGame(n_agents=seats, rounds=args.rounds, mpcr=MPCR)
    print(f"evaluation match: {len(roster)} seats, {args.rounds} rounds, every round one model call")
    started = time.time()
    result = run_match(game, roster, rounds=args.rounds, seed=args.seed)
    elapsed = time.time() - started

    actions = np.asarray(result["actions"])
    rewards = np.asarray(result["rewards"])
    metrics = social.compute_all(result, game.max_welfare_per_round())

    rows = []
    for i, agent in enumerate(roster):
        rows.append({
            "name": agent.name,
            "kind": agent.kind,
            "information": getattr(agent, "information", "?"),
            "payoff": float(rewards[:, i].sum()),
            "cooperation": float((actions[:, i] == game.action_roles["cooperate"]).mean()),
        })
    rows.sort(key=lambda r: -r["payoff"])

    print(f"\n{'agent':22} {'kind':13} {'sees':22} {'payoff':>8} {'cooperated':>11}")
    for row in rows:
        print(f"{row['name'][:22]:22} {row['kind']:13} {row['information']:22} "
              f"{row['payoff']:8.1f} {row['cooperation']:10.0%}")

    print(f"\nmatch cooperation {metrics['cooperation_rate']:.2f}, "
          f"efficiency {metrics['efficiency']:.2f}, {elapsed:.0f}s "
          f"({elapsed / max(1, args.rounds):.1f}s per round)")
    print("One match, one model, and policies frozen after separate training. A payoff ordering "
          "here is a picture of these policies meeting once, not a ranking of architectures.")

    # What the model said while the numbers happened. This is the transparency panel's content, and
    # it is the only place in the platform where an agent's own account of its move can be read
    # against what the move cost, which is worth printing beside the payoffs rather than filing
    # away. It is testimony: nothing checks it against the computation that produced the action.
    journal = roster[-1].journal()
    samples = [(i, journal[i]) for i in dict.fromkeys(
        [0, len(journal) // 2, len(journal) - 1]) if 0 <= i < len(journal)]
    if samples:
        print("\nwhat the model said, in its own words:")
        for index, entry in samples:
            data = entry["data"]
            print(f"  round {index + 1:3}  {data.get('action', '?'):8} "
                  f"confidence {data.get('confidence')}  {str(data.get('rationale'))[:88]}")

    out = RESULTS_DIR / f"llm_bakeoff_{socket.gethostname().lower()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "rounds": args.rounds, "seed": args.seed,
        "pretrain_rounds": PRETRAIN_ROUNDS, "mpcr": MPCR, "seconds": round(elapsed, 1),
        "agents": rows,
        "match": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        "rationales": [{"round": i + 1, **e["data"]} for i, e in samples],
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
