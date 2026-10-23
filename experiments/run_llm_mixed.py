"""Several model families in one match, and the same match again on the other side of r = n.

Every language-model measurement this platform has taken so far seated **one** model against the
other architectures (`run_llm_bakeoff.py` takes a single `--model`). That answers how a model does
against trained policies. It cannot answer the question a collaborator asked, which is whether
models from different families coordinate with each other or merely arrive at similar totals for
unrelated reasons -- because that question needs them in the same match, reading each other's
behaviour through the public signal.

The second half is the multiplier. `r = mpcr * n` below `n` makes defection dominant and above it
makes contribution dominant, so sweeping across that crossing changes what a payoff-maximiser
should do while changing nothing else: same seats, same seed, same prompt profile. A model that is
tracking the incentive moves with it. A model that is reciting what public-goods games usually look
like does not. Those two are indistinguishable at a single multiplier, which is the whole reason
this script runs a list.

**What it reports and why that shape.** Per model per multiplier: cooperation rate, payoff, and
every rationale it gave. The rationales matter here more than anywhere else in the platform,
because two models agreeing on an action while giving incompatible reasons is exactly the finding
the design is built to catch, and no scalar carries it. Nothing checks a rationale against the
computation that produced the action: it is testimony.

**Cost, before you start it.** One model call per seat per round, so three families over 40 rounds
at two multipliers is 240 calls. Measured on this project's hardware that is 2.5-10 seconds each
depending on model and machine, so plan in tens of minutes and run it on the machine with the
models, not the laptop.

    py -m gamebrains.experiments.run_llm_mixed --models llama3.2:1b qwen2.5:1.5b
    py -m gamebrains.experiments.run_llm_mixed --models llama3.2:1b qwen2.5:1.5b gemma2:2b \\
        --multipliers 1.5 3.0 4.5 --rounds 40

Writes `experiments/results/llm_mixed_<host>_<profile>.json`.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Any

import numpy as np

from ..agents.llm import LLMAgent
from ..agents.llm_ollama import OllamaBackend
from ..agents.llm_prompt import DEFAULT_PROFILE, PROFILES
from ..engine.runner import run_match
from ..games.public_goods import PublicGoodsGame
from ..metrics import social
from . import run_bakeoff
from .run_bakeoff import PRETRAIN_ROUNDS, pretrain
from .run_llm_contract import RESULTS_DIR

#: Two multipliers either side of the crossing, at the default three seats: r=1.5 is the social
#: dilemma every existing measurement was taken in, r=4.5 is where contributing dominates. The
#: crossing itself (r=n) is deliberately not a default, since every profile is an equilibrium
#: there and a reading taken at it describes the configuration rather than the agents.
DEFAULT_MULTIPLIERS = [1.5, 4.5]


def _scalars(metrics: dict[str, Any]) -> dict[str, Any]:
    """The headline numbers only, because the per-round series are already stored per seat.

    `social.compute_all` returns numpy arrays alongside its scalars (the cooperation series, for
    one), and json refuses them. Dropping them here rather than converting them is deliberate:
    this file holds one row per seat per round under `actions` already, so writing the series a
    second time would store the same data twice in two shapes that could later disagree.
    """
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (bool, str)):
            out[key] = value
        elif isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = float(value)
    return out


def _regime_line(game: PublicGoodsGame) -> str:
    """One sentence a reader of the log can check the arithmetic of, rather than a label."""
    if game.regime == "social_dilemma":
        return (f"r={game.multiplier:.2f} < n={game.n_agents}: a contribution returns "
                f"{game.mpcr:.2f} of its cost to the contributor, so defecting dominates")
    if game.regime == "cooperation_dominant":
        return (f"r={game.multiplier:.2f} > n={game.n_agents}: a contribution returns "
                f"{game.mpcr:.2f} of its cost to the contributor, so contributing dominates")
    return (f"r={game.multiplier:.2f} = n={game.n_agents}: a contribution returns exactly its own "
            f"cost, so no player is pushed either way and every profile is an equilibrium")


def _play(models: list[str], baselines: list[str], multiplier: float, rounds: int, seed: int,
          profile: str, backends: dict[str, OllamaBackend]) -> dict[str, Any]:
    """One match at one multiplier. The roster is rebuilt per multiplier, deliberately.

    A language-model seat carries its own decision journal and a trained baseline carries a policy,
    so reusing either across multipliers would let the first match's history leak into the second
    and the comparison would no longer be between multipliers alone.
    """
    seats = len(models) + len(baselines)
    mpcr = multiplier / seats

    # Tabular policies are tables over this game's observation space, which is the population size
    # plus two, so they have to be trained at the number of seats they will play at.
    roster: list[Any] = []
    if baselines:
        run_bakeoff.N_AGENTS = seats
        print(f"  pretraining {len(baselines)} baseline(s) for {PRETRAIN_ROUNDS} rounds at "
              f"{seats} seats")
        # Trained at this match's multiplier, not at the bake-off default. A frozen policy
        # learned in the dilemma and dropped in above the crossing would fail to track the
        # incentive for a reason that has nothing to do with the architecture.
        roster.extend(pretrain(kind, seed, quiet=True, mpcr=mpcr, allow_non_dilemma=True)
                      for kind in baselines)
    roster.extend(LLMAgent(f"LLM ({m})", backend=backends[m], profile=profile) for m in models)

    game = PublicGoodsGame(n_agents=seats, rounds=rounds, mpcr=mpcr, allow_non_dilemma=True)
    print(f"  {_regime_line(game)}")

    started = time.time()
    result = run_match(game, roster, rounds=rounds, seed=seed)
    elapsed = time.time() - started

    actions = np.asarray(result["actions"])
    rewards = np.asarray(result["rewards"])
    cooperate = game.action_roles["cooperate"]

    seats_out = []
    for i, agent in enumerate(roster):
        entry: dict[str, Any] = {
            "name": agent.name,
            "kind": agent.kind,
            "model": models[i - len(baselines)] if i >= len(baselines) else None,
            "payoff": float(rewards[:, i].sum()),
            "cooperation": float((actions[:, i] == cooperate).mean()),
            "actions": [int(a) for a in actions[:, i]],
        }
        if hasattr(agent, "journal"):
            # Kept whole rather than sampled. The question the design asks is whether two models
            # that act alike reason alike, and a sample of three rounds cannot answer it.
            entry["rationales"] = [
                {"round": k, "action": e["data"].get("action"),
                 "confidence": e["data"].get("confidence"),
                 "rationale": e["data"].get("rationale")}
                for k, e in enumerate(agent.journal())]
        seats_out.append(entry)

    return {
        "multiplier": multiplier, "mpcr": mpcr, "regime": game.regime, "seats": seats,
        "rounds": rounds, "seed": seed, "seconds": round(elapsed, 1),
        "metrics": _scalars(social.compute_all(result, game.max_welfare_per_round())),
        "agents": seats_out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True,
                        help="Ollama model tags, exactly as pulled. One seat each, so two tags "
                             "from one family measure that family twice and tell you nothing "
                             "about agreement across families")
    parser.add_argument("--multipliers", nargs="+", type=float, default=DEFAULT_MULTIPLIERS,
                        help="r values to run, one match each. mpcr is derived as r/seats, so the "
                             "crossing is wherever r equals the seat count")
    parser.add_argument("--baselines", nargs="*", default=[],
                        help="architectures trained and frozen before the match, e.g. qlearning "
                             "dqn. Each adds a seat and so moves the crossing")
    parser.add_argument("--rounds", type=int, default=40,
                        help="every round is one call per model seat, so this is the cost dial")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", default=DEFAULT_PROFILE,
                        help=f"what each model is told, by name: {', '.join(sorted(PROFILES))}. "
                             f"All seats get the same profile, since a profile is an information "
                             f"set and varying it per seat would confound the comparison")
    args = parser.parse_args(argv)

    if len(set(args.models)) != len(args.models):
        print("the same model tag twice is two seats of one family, which this script exists to "
              "avoid; pass distinct tags")
        return 1

    # Every backend is checked before anything is trained or played. A match that dies on round
    # four hundred because the third model was never pulled wastes everything before it, and the
    # message names what to run.
    backends: dict[str, OllamaBackend] = {}
    for model in args.models:
        backend = OllamaBackend(model=model)
        if not backend.is_available():
            print(f"no Ollama model {model!r}: `ollama serve`, then `ollama pull {model}`")
            return 1
        backends[model] = backend

    seats = len(args.models) + len(args.baselines)
    calls = len(args.models) * args.rounds * len(args.multipliers)
    print(f"{seats} seats: {len(args.models)} model(s) {', '.join(args.models)}"
          + (f" and baselines {', '.join(args.baselines)}" if args.baselines else "")
          + f"\n{len(args.multipliers)} multiplier(s) {args.multipliers}, crossing at r={seats}"
          f"\n{calls} model calls at {args.rounds} rounds each, prompt profile "
          f"{args.profile!r}\n")

    runs = []
    for multiplier in args.multipliers:
        print(f"r = {multiplier}")
        run = _play(args.models, args.baselines, multiplier, args.rounds, args.seed,
                    args.profile, backends)
        runs.append(run)
        for seat in sorted(run["agents"], key=lambda s: -s["payoff"]):
            print(f"    {seat['name'][:28]:28} payoff {seat['payoff']:8.1f}  "
                  f"contributed {seat['cooperation']:5.0%}")
        print(f"    match cooperation {run['metrics']['cooperation_rate']:.2f}, "
              f"{run['seconds']:.0f}s\n")

    # The comparison the whole script exists for, printed rather than left to be derived: did each
    # seat move when the incentive moved, and did the seats move together?
    if len(runs) > 1:
        print(f"{'seat':30}" + "".join(f"  r={r['multiplier']:<8.2f}" for r in runs) + "   shift")
        for i in range(len(runs[0]["agents"])):
            rates = [r["agents"][i]["cooperation"] for r in runs]
            print(f"{runs[0]['agents'][i]['name'][:30]:30}"
                  + "".join(f"  {rate:>9.0%}" for rate in rates)
                  + f"  {rates[-1] - rates[0]:+8.0%}")
        print("\nA seat whose contribution rate barely moves across the crossing is not reading "
              "the incentive. Equal rates on both sides is agreement about an outcome, not "
              "evidence of agreement about a reason: read the rationales for that.")

    tag = args.profile.replace(":", "-").replace("/", "-")
    out = RESULTS_DIR / f"llm_mixed_{socket.gethostname().lower()}_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "host": socket.gethostname().lower(),
        "models": args.models, "baselines": args.baselines, "profile": args.profile,
        "rounds": args.rounds, "seed": args.seed, "crossing_at": seats,
        "runs": runs,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
