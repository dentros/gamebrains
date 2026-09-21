"""What asking a language model for a structured decision actually costs, measured.

Three questions, each answered on this platform's own prompts rather than on a benchmark's:

  **E1, the schema.** How often does a model answer in the requested schema on the first
  attempt, how many repairs does it take when it does not, and how much of that does
  constrained decoding buy? Every cell answers the *same* prompts, because comparing
  constrained against unconstrained on different prompts would measure the prompts.

  **E2, determinism.** The same prompt, repeated, at temperature 0. This is the measurement
  behind the platform's fourth refusal (`repository/record.reproducibility_of`): a roster
  holding a model cannot support the byte-identical reproduction claim, and that should be a
  number rather than an assumption.

  **E3, role binding.** One agent, two games whose action indices mean opposite things, and a
  third that declares neither role. Cheap, deterministic, and it belongs with the others
  because it is the part a reader would otherwise have to take on faith.

The prompts come from real matches. `collect_prompts` plays Q-learners through the game and
builds each prompt with the agent's own `build_prompt`, so what is measured is the text the
platform really sends, not a hand-written approximation of it.

    py -m gamebrains.experiments.run_llm_contract --decisions 60 --repeats 4
    py -m gamebrains.experiments.run_llm_contract --quick        # a few minutes, for a smoke test

Results are written to `experiments/results/llm_contract_<host>.json`, keyed by host so a run on
a second machine sits beside this one rather than overwriting it. That is the point of the file
layout: E2 asks whether a model's answers hold still, and the harder version of that question is
whether they hold still across machines.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from ..agents.llm import LLMAgent
from ..agents.llm_prompt import DEFAULT_PROFILE, decision_schema
from ..agents.llm_backends import BackendUnavailable, SchemaViolation
from ..agents.llm_ollama import OllamaBackend
from ..agents.qlearning import QLearningAgent
from ..engine.game import Game
from ..games.congestion import CongestionGame
from ..games.public_goods import PublicGoodsGame

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_MODELS = ("llama3.2:1b", "qwen2.5:1.5b")


# --- prompts, from real play -------------------------------------------------------------

def _make_game(kind: str) -> Game:
    if kind == "public_goods":
        return PublicGoodsGame(n_agents=4, rounds=400)
    if kind == "congestion":
        return CongestionGame(n_agents=4)
    raise ValueError(f"unknown game {kind!r}")


def collect_prompts(kind: str, n_prompts: int, seed: int = 0,
                    profile: str = DEFAULT_PROFILE) -> list[str]:
    """Play the game with cheap agents and keep the prompts an LLM seat would have been sent.

    No model is involved, so this is deterministic and free. One seat is a bound `LLMAgent` whose
    action is supplied by a Q-learner rather than by a backend: it never decides anything, it only
    accumulates the history that shapes the prompt.

    The profile is an argument because the prompts are the study's material: a reliability rate
    measured on one profile's prompts says nothing about another's, and the measurements already
    published were taken on `minimal-v1`, which is why it stays the default.
    """
    game = _make_game(kind)
    n = game.n_agents
    learners = [QLearningAgent(f"Q{i}", n_states=game.n_states, n_actions=game.n_actions,
                               seed=seed + i) for i in range(n)]
    watcher = LLMAgent("prompt source", backend=None, profile=profile)  # type: ignore[arg-type]
    watcher.on_match_start(game)

    observations = game.reset()
    for agent in learners:
        agent.on_match_start(game)

    prompts: list[str] = []
    while len(prompts) < n_prompts:
        prompts.append(watcher.build_prompt(observations[0]))
        actions = [learners[i].act(observations[i]) for i in range(n)]
        step = game.step(actions)
        for i, agent in enumerate(learners):
            agent.update(observations[i], actions[i], step.rewards[i],
                         step.observations[i], step.done)
        watcher.update(observations[0], actions[0], step.rewards[0], step.observations[0],
                       step.done)
        observations = game.reset() if step.done else step.observations
    return prompts


# --- E1: the schema ----------------------------------------------------------------------

def _complete_with_retry(backend: OllamaBackend, prompt: str, schema: dict[str, Any],
                         tries: int = 3, pause: float = 20.0):
    """Retry a request that failed because the *server* went away, never one the model failed.

    The distinction is the whole point of `BackendUnavailable` (see `agents/llm_backends.py`). A
    laptop that sleeps, or an Ollama that unloads a model under memory pressure, produces an
    infrastructure gap in the middle of a two-hour run, and counting that as a schema failure
    would publish an installation problem as a property of the model. A `SchemaViolation`
    propagates untouched.
    """
    for attempt in range(tries):
        try:
            return backend.complete(prompt, schema)
        except BackendUnavailable:
            if attempt == tries - 1:
                raise
            print(f"      server unreachable, waiting {pause:.0f}s and retrying "
                  f"({attempt + 1}/{tries - 1})")
            time.sleep(pause)
    raise AssertionError("unreachable")


def measure_schema(model: str, constrained: bool, prompts: list[str],
                   roles: tuple[str, ...] = ("concede", "claim")) -> dict[str, Any]:
    backend = OllamaBackend(model=model, constrained=constrained)
    schema = decision_schema(roles)

    first_ok = 0
    attempts_used: list[int] = []
    seconds: list[float] = []
    eval_tokens: list[int] = []
    prompt_tokens: list[int] = []
    gave_up = 0
    errors: dict[str, int] = {}
    chosen: dict[str, int] = {}

    for prompt in prompts:
        try:
            response = _complete_with_retry(backend, prompt, schema)
        except SchemaViolation as exc:
            gave_up += 1
            errors[_error_class(str(exc))] = errors.get(_error_class(str(exc)), 0) + 1
            continue

        attempts_used.append(response.n_attempts)
        seconds.append(response.seconds)
        if response.attempts[0].ok:
            first_ok += 1
        else:
            for attempt in response.attempts:
                if not attempt.ok:
                    errors[_error_class(attempt.error)] = errors.get(_error_class(attempt.error), 0) + 1
        for attempt in response.attempts:
            if attempt.eval_tokens is not None:
                eval_tokens.append(attempt.eval_tokens)
            if attempt.prompt_tokens is not None:
                prompt_tokens.append(attempt.prompt_tokens)
        action = str(response.data.get("action"))
        chosen[action] = chosen.get(action, 0) + 1

    answered = len(attempts_used)
    return {
        "model": model,
        "constrained": constrained,
        "prompts": len(prompts),
        "answered": answered,
        "gave_up": gave_up,
        "first_attempt_ok": first_ok,
        "first_attempt_rate": round(first_ok / len(prompts), 4) if prompts else None,
        "mean_attempts": round(statistics.fmean(attempts_used), 3) if attempts_used else None,
        "median_seconds": round(statistics.median(seconds), 3) if seconds else None,
        "mean_eval_tokens": round(statistics.fmean(eval_tokens), 1) if eval_tokens else None,
        "mean_prompt_tokens": round(statistics.fmean(prompt_tokens), 1) if prompt_tokens else None,
        "errors": errors,
        "actions": chosen,
    }


def _error_class(message: str) -> str:
    """Group failures by what the model did wrong, since the raw strings carry values."""
    text = message.lower()
    if "no json object" in text:
        return "no JSON at all"
    if "not valid json" in text:
        return "malformed JSON"
    if "must be one of" in text:
        return "action outside the enum"
    if "is missing" in text:
        return "a required field missing"
    if "must be a" in text or "at most" in text or "at least" in text:
        return "a field of the wrong type or range"
    return "other"


# --- E2: determinism ---------------------------------------------------------------------

def measure_determinism(model: str, prompts: list[str], repeats: int) -> dict[str, Any]:
    """Ask the same question several times at temperature 0 and see whether the answer holds.

    Two rates, because they are different claims. Byte-identical raw output is what the event log
    would need in order to be reproducible. An identical *decision* is the weaker property that a
    behavioural result would need, and a model can fail the first while keeping the second.
    """
    backend = OllamaBackend(model=model, constrained=True)
    schema = decision_schema(("concede", "claim"))

    identical_text = 0
    identical_action = 0
    usable = 0
    for prompt in prompts:
        raws: list[str] = []
        actions: list[str] = []
        for _ in range(repeats):
            try:
                response = _complete_with_retry(backend, prompt, schema)
            except SchemaViolation:
                break
            raws.append(response.attempts[-1].raw)
            actions.append(str(response.data.get("action")))
        if len(raws) < repeats:
            continue
        usable += 1
        identical_text += int(len(set(raws)) == 1)
        identical_action += int(len(set(actions)) == 1)

    return {
        "model": model,
        "prompts": usable,
        "repeats": repeats,
        "byte_identical_rate": round(identical_text / usable, 4) if usable else None,
        "same_decision_rate": round(identical_action / usable, 4) if usable else None,
    }


# --- E3: role binding --------------------------------------------------------------------

def check_role_binding() -> dict[str, Any]:
    """No model needed. One agent, two games, opposite indices, and a refusal where neither role
    is declared."""
    pgg, hjg = _make_game("public_goods"), _make_game("congestion")

    class _Coordination(Game):
        n_agents, n_actions, n_states, name = 2, 2, 2, "coordination_stub"
        action_names = ["Option A", "Option B"]
        action_roles = {"option_a": 0, "option_b": 1}

        def reset(self):
            return [0, 0]

        def step(self, actions):
            raise NotImplementedError

    in_pgg = LLMAgent("a", backend=None)                  # type: ignore[arg-type]
    in_pgg.on_match_start(pgg)
    in_hjg = LLMAgent("b", backend=None)                  # type: ignore[arg-type]
    in_hjg.on_match_start(hjg)

    refused = ""
    try:
        LLMAgent("c", backend=None).on_match_start(_Coordination())  # type: ignore[arg-type]
    except ValueError as exc:
        refused = str(exc)

    return {
        "claim_in_public_goods": pgg.action_roles["claim"],
        "claim_in_congestion": hjg.action_roles["claim"],
        "indices_differ": pgg.action_roles["claim"] != hjg.action_roles["claim"],
        "coordination_game_refused": bool(refused),
        "refusal_names_declared_roles": "option_a" in refused,
    }


# --- the run -----------------------------------------------------------------------------

def _host_key() -> str:
    return socket.gethostname().lower().replace(" ", "-")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--decisions", type=int, default=60,
                        help="prompts per game per cell in E1")
    parser.add_argument("--determinism-prompts", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--games", nargs="*", default=["public_goods", "congestion"])
    parser.add_argument("--quick", action="store_true",
                        help="a few minutes: 8 decisions, 4 determinism prompts, 3 repeats")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore an existing results file instead of resuming it")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    if args.quick:
        args.decisions, args.determinism_prompts, args.repeats = 8, 4, 3

    available = []
    for model in args.models:
        backend = OllamaBackend(model=model)
        if backend.is_available():
            available.append(model)
        else:
            print(f"SKIP {model}: not installed. `ollama pull {model}`")
    if not available:
        print("no model available; start `ollama serve` and pull at least one model")
        return 1

    out_path = Path(args.out) if args.out else RESULTS_DIR / f"llm_contract_{_host_key()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume by default. A full run is hours of model calls, and the two ways it ends early are a
    # sleeping laptop and a shared machine's queue. Re-measuring a cell that already has an answer
    # would cost as much as the interruption did, so finished cells are kept and skipped. A cell
    # is identified by what defines it, (model, game, constrained), never by its position.
    previous: dict[str, Any] = {}
    if out_path.exists() and not args.fresh:
        try:
            previous = json.loads(out_path.read_text(encoding="utf-8"))
            done = len(previous.get("schema", [])) + len(previous.get("determinism", []))
            if done:
                print(f"resuming {out_path.name}: {done} cell(s) already measured "
                      f"(--fresh to start over)")
        except json.JSONDecodeError:
            print(f"{out_path.name} is not readable JSON, starting fresh")

    def already(bucket: str, **keys: Any) -> bool:
        return any(all(cell.get(k) == v for k, v in keys.items())
                   for cell in previous.get(bucket, []))

    started = time.time()
    results: dict[str, Any] = {
        "host": _host_key(),
        "platform": f"{platform.system()} {platform.machine()}",
        "python": sys.version.split()[0],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": {"decisions": args.decisions, "repeats": args.repeats,
                     "determinism_prompts": args.determinism_prompts, "games": args.games},
        "role_binding": check_role_binding(),
        "schema": list(previous.get("schema", [])),
        "determinism": list(previous.get("determinism", [])),
    }

    def save() -> None:
        results["elapsed_seconds"] = round(time.time() - started, 1)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    prompts = {kind: collect_prompts(kind, args.decisions) for kind in args.games}
    determinism_prompts = {kind: pool[:args.determinism_prompts] for kind, pool in prompts.items()}
    print(f"prompts built from real matches: "
          + ", ".join(f"{k}={len(v)}" for k, v in prompts.items()))

    try:
        for model in available:
            for kind, pool in prompts.items():
                for constrained in (True, False):
                    label = f"{model} {kind} {'constrained' if constrained else 'prompt-only'}"
                    if already("schema", model=model, game=kind, constrained=constrained):
                        print(f"\n[E1] {label}: already measured, skipping")
                        continue
                    print(f"\n[E1] {label}: {len(pool)} decisions")
                    cell = measure_schema(model, constrained, pool)
                    cell["game"] = kind
                    results["schema"].append(cell)
                    save()
                    print(f"      first attempt {cell['first_attempt_rate']}, "
                          f"mean attempts {cell['mean_attempts']}, "
                          f"median {cell['median_seconds']}s, gave up {cell['gave_up']}")

            for kind, pool in determinism_prompts.items():
                if already("determinism", model=model, game=kind):
                    print(f"\n[E2] {model} {kind}: already measured, skipping")
                    continue
                print(f"\n[E2] {model} {kind}: {len(pool)} prompts x {args.repeats}")
                cell = measure_determinism(model, pool, args.repeats)
                cell["game"] = kind
                results["determinism"].append(cell)
                save()
                print(f"      byte-identical {cell['byte_identical_rate']}, "
                      f"same decision {cell['same_decision_rate']}")
    except BackendUnavailable as exc:
        print(f"\nstopped: {exc}")
        save()
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; partial results kept")
        save()
        return 130

    save()
    print(f"\nwrote {out_path}")
    print(f"total {results['elapsed_seconds'] / 60:.1f} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
