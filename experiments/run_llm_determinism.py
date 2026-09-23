"""Why a language model in a seat cannot promise a byte-identical event log.

`run_llm_contract.py` measured how often repeating one request returns the same text and got
between 6.7% and 53.3% across four model and game pairs. Re-measured later the same day the same
cell gave 92% and 100%, which means the rate is not a stable property of the model and must not be
quoted as one. This script exists to say what it *is*.

Two parts.

**Negative controls.** Each arm changes one suspect and leaves everything else alone, so an arm
that comes back stable removes a candidate explanation rather than suggesting one:

    seed          a fixed `options.seed`, since sampling with a pinned seed would be reproducible
    reload        `keep_alive: 0`, so every repeat is answered by freshly loaded weights
    load          every core but one kept busy, since thread scheduling changes reduction order
    interleaved   a different prompt forced between repeats, to defeat the server's prompt cache

**Sessions.** The same prompts and repeats, measured several times over, because a single session
reports one draw from whatever this actually is. The spread between sessions is the finding.

    py -m gamebrains.experiments.run_llm_determinism --sessions 3 --prompts 10

Writes `experiments/results/llm_determinism_<host>.json`. Everything here is on CPU. A GPU changes
the arithmetic and would need its own run before any of it is repeated for that case.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import socket
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from ..agents.llm import decision_schema
from ..agents.llm_ollama import OllamaBackend
from .run_llm_contract import RESULTS_DIR, collect_prompts

SCHEMA = decision_schema(("concede", "claim"))


def _burn(stop_at: float) -> None:
    total = 0.0
    while time.time() < stop_at:
        total += sum(i * i for i in range(4000))


def _repeat(backend: OllamaBackend, prompt: str, repeats: int,
            keep_alive: Optional[int] = None, seed: Optional[int] = None,
            between: Optional[str] = None, threads: Optional[int] = None) -> list[str]:
    answers = []
    for _ in range(repeats):
        if between is not None:
            backend.complete(between, SCHEMA)            # evict the cached prefix
        answers.append(_ask(backend, prompt, keep_alive, seed, threads))
    return answers


def _ask(backend: OllamaBackend, prompt: str, keep_alive: Optional[int],
         seed: Optional[int], threads: Optional[int] = None) -> str:
    """One request, with the two fields the ordinary backend never sets.

    Sent here rather than added to `OllamaBackend`, because neither belongs in the shipped agent:
    a seed would imply a reproducibility the measurements below refuse, and unloading after every
    call is a way to provoke the failure rather than a way to run.
    """
    import urllib.request

    body: dict[str, Any] = {
        "model": backend.model, "prompt": prompt, "stream": False, "format": SCHEMA,
        "options": {"temperature": backend.temperature},
    }
    if seed is not None:
        body["options"]["seed"] = seed
    if threads is not None:
        body["options"]["num_thread"] = threads
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    request = urllib.request.Request(
        f"{backend.host}/api/generate", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=backend.timeout) as fh:
        return json.loads(fh.read().decode("utf-8")).get("response", "")


def _rate(backend, prompts, repeats, **kwargs) -> dict[str, Any]:
    """The share of prompts whose repeats agree, and **which** prompts they were.

    The index list is the useful part. Two independent passes over the same pool disagreed on the
    same four prompts of ten, which says the variation sits at particular decisions rather than
    arriving at random, and a rate alone would have hidden that.
    """
    stable, shapes, action_stable = 0, [], 0
    unstable: list[int] = []
    for position, prompt in enumerate(prompts):
        answers = _repeat(backend, prompt, repeats, **kwargs)
        counts = Counter(answers)
        stable += int(len(counts) == 1)
        if len(counts) > 1:
            shapes.append(sorted(counts.values(), reverse=True))
            unstable.append(position)
        actions = set()
        for answer in answers:
            try:
                actions.add(json.loads(answer).get("action"))
            except json.JSONDecodeError:
                actions.add(answer[:20])
        action_stable += int(len(actions) == 1)
    return {
        "prompts": len(prompts), "repeats": repeats,
        "byte_identical": round(stable / len(prompts), 4),
        "same_decision": round(action_stable / len(prompts), 4),
        "splits": shapes,
        "unstable_prompts": unstable,
    }


def paired_cache_test(backend, prompts, repeats: int, other: str) -> dict[str, Any]:
    """Does evicting the server's cached prefix between repeats change how often it repeats itself?

    The one control that moved the rate was the interleaved arm, 70% against a baseline of 100% in
    the same session, and that would explain every measurement taken so far: the contract run swept
    thirty different prompts, so the prefix was evicted constantly and it reported 50%, while the
    re-checks repeated one prompt back to back with the prefix warm and reported 90% and 100%.

    Ten prompts cannot separate 70% from 100%, so this measures both arms **on the same prompt,
    back to back**, which removes the session drift that makes unpaired arms hard to read. The
    answer is the count of prompts where one arm repeated itself and the other did not.
    """
    both, warm_only, evicted_only, neither = 0, 0, 0, 0
    for prompt in prompts:
        warm = len(set(_repeat(backend, prompt, repeats))) == 1
        evicted = len(set(_repeat(backend, prompt, repeats, between=other))) == 1
        both += int(warm and evicted)
        warm_only += int(warm and not evicted)
        evicted_only += int(evicted and not warm)
        neither += int(not warm and not evicted)
    return {
        "prompts": len(prompts), "repeats": repeats,
        "warm_stable": round((both + warm_only) / len(prompts), 4),
        "evicted_stable": round((both + evicted_only) / len(prompts), 4),
        "stable_only_when_warm": warm_only,
        "stable_only_when_evicted": evicted_only,
        "stable_in_both": both,
        "stable_in_neither": neither,
    }


def paired_reload_test(backend, prompts, repeats: int) -> dict[str, Any]:
    """Does unloading the weights between repeats change how often the model repeats itself?

    The unpaired arm said yes on both machines this has been run on: a forced reload was the only
    control that reached 100% with no prompt disagreeing, while everything else, including the
    baseline run twice, left the same handful of prompts unstable. That is suggestive and it is not
    a result. Ten prompts cannot separate 80% from 100%, which is exactly the objection this study
    raises against its own unpaired arms, so it is raised here too.

    Both conditions are therefore measured **on the same prompt, back to back**, as for the cached
    prefix. The count that matters is the disagreement: prompts stable only when the weights were
    reloaded, against prompts stable only when they stayed resident. A real cause shows up as a
    one-sided disagreement, and the interleaved arm is the reminder of why this matters, since its
    unpaired version had the sign backwards.
    """
    both, resident_only, reloaded_only, neither = 0, 0, 0, 0
    for prompt in prompts:
        resident = len(set(_repeat(backend, prompt, repeats))) == 1
        reloaded = len(set(_repeat(backend, prompt, repeats, keep_alive=0))) == 1
        both += int(resident and reloaded)
        resident_only += int(resident and not reloaded)
        reloaded_only += int(reloaded and not resident)
        neither += int(not resident and not reloaded)
    return {
        "prompts": len(prompts), "repeats": repeats,
        "resident_stable": round((both + resident_only) / len(prompts), 4),
        "reloaded_stable": round((both + reloaded_only) / len(prompts), 4),
        "stable_only_when_resident": resident_only,
        "stable_only_when_reloaded": reloaded_only,
        "stable_in_both": both,
        "stable_in_neither": neither,
    }


def _save(out: Path, results: dict[str, Any]) -> None:
    """Write, merging with whatever is on disk rather than replacing it.

    Two runs of this script can be in flight at once, which happened here: a run that was thought
    to be stopped kept going and the two processes took turns overwriting each other's cells with
    their own in-memory state. Re-reading before every write does not make concurrent runs safe in
    general, and it does make the common case, one model measured while another finishes, keep both.
    """
    merged = dict(results)
    if out.exists():
        try:
            disk = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            disk = {}
        sessions = dict(disk.get("sessions", {}))
        sessions.update(results.get("sessions", {}))
        controls = dict(disk.get("controls", {}))
        controls.update(results.get("controls", {}))
        merged["sessions"], merged["controls"] = sessions, controls
        for key in ("cache_pairs", "reload_pairs"):
            if key not in merged and key in disk:
                merged[key] = disk[key]
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=["llama3.2:1b", "qwen2.5:1.5b"])
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--prompts", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--game", default="public_goods")
    parser.add_argument("--controls", action="store_true",
                        help="also run the four negative controls, on the first model only")
    parser.add_argument("--cache-pairs", type=int, default=0,
                        help="paired warm-against-evicted test on this many prompts, which is the "
                             "one comparison the unpaired controls could not settle")
    parser.add_argument("--reload-pairs", type=int, default=0,
                        help="paired resident-against-reloaded test on this many prompts. The "
                             "unpaired reload arm reached 100 percent on two machines, which ten "
                             "prompts cannot establish, so it gets the same treatment")
    args = parser.parse_args(argv)

    pool = [p for p in collect_prompts(args.game, args.prompts + 8) if len(p) > 700][:args.prompts]
    other = collect_prompts("congestion", 3)[2]
    host = socket.gethostname().lower()
    out = RESULTS_DIR / f"llm_determinism_{host}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Sessions accumulate across invocations rather than replacing what is there. The spread
    # between sessions is the finding, and a run measuring one model would otherwise delete the
    # sessions measured for another, which is how the first version of this script nearly lost
    # three sessions of llama data to a two-session qwen run.
    results: dict[str, Any] = {
        "host": host,
        "platform": f"{platform.system()} {platform.machine()}",
        "game": args.game, "prompts": len(pool), "repeats": args.repeats,
        "sessions": {}, "controls": {},
    }
    if out.exists():
        try:
            previous = json.loads(out.read_text(encoding="utf-8"))
            results["sessions"] = previous.get("sessions", {})
            results["controls"] = previous.get("controls", {})
            kept = sum(len(v) for v in results["sessions"].values())
            if kept:
                print(f"keeping {kept} session(s) already in {out.name}")
        except json.JSONDecodeError:
            print(f"{out.name} is not readable JSON, starting fresh")

    for model in args.models:
        backend = OllamaBackend(model=model, constrained=True)
        if not backend.is_available():
            print(f"SKIP {model}: not installed")
            continue
        backend.complete(pool[0], SCHEMA)

        runs = list(results["sessions"].get(model, []))
        for session in range(args.sessions):
            measured = _rate(backend, pool, args.repeats)
            runs.append(measured)
            print(f"{model} session {session + 1}: byte-identical "
                  f"{measured['byte_identical']:.0%}, same decision "
                  f"{measured['same_decision']:.0%}")
            results["sessions"][model] = runs
            _save(out, results)

        rates = [r["byte_identical"] for r in runs]
        print(f"{model}: {min(rates):.0%} to {max(rates):.0%} across {len(rates)} sessions\n")

    if args.controls:
        model = args.models[0]
        backend = OllamaBackend(model=model, constrained=True)
        controls = {
            "baseline": {},
            "baseline_again": {},          # the same arm twice, so a difference has a scale to
                                           # be read against before any other arm is believed
            "seed": {"seed": 7},
            "reload": {"keep_alive": 0},
            "one_thread": {"threads": 1},
            "interleaved": {"between": other},
        }
        for name, kwargs in controls.items():
            measured = _rate(backend, pool, args.repeats, **kwargs)
            results["controls"][name] = measured
            print(f"control {name:12} byte-identical {measured['byte_identical']:.0%}")
            _save(out, results)

        workers = max(1, multiprocessing.cpu_count() - 1)
        processes = [multiprocessing.Process(target=_burn, args=(time.time() + 1800,))
                     for _ in range(workers)]
        for process in processes:
            process.start()
        time.sleep(3)
        try:
            measured = _rate(backend, pool, args.repeats)
        finally:
            for process in processes:
                process.terminate()
            for process in processes:
                process.join()
        measured["busy_cores"] = workers
        results["controls"]["load"] = measured
        print(f"control {'load':12} byte-identical {measured['byte_identical']:.0%} "
              f"with {workers} cores busy")

    if args.cache_pairs:
        model = args.models[0]
        backend = OllamaBackend(model=model, constrained=True)
        wide = [p for p in collect_prompts(args.game, args.cache_pairs + 10)
                if len(p) > 700][:args.cache_pairs]
        backend.complete(wide[0], SCHEMA)
        paired = paired_cache_test(backend, wide, args.repeats, other)
        paired["model"] = model
        results["cache_pairs"] = paired
        print(f"\npaired cache test on {paired['prompts']} prompts, {args.repeats} repeats each:")
        print(f"  warm prefix    {paired['warm_stable']:.0%} repeated identically")
        print(f"  evicted prefix {paired['evicted_stable']:.0%} repeated identically")
        print(f"  disagreed on {paired['stable_only_when_warm']} prompts one way and "
              f"{paired['stable_only_when_evicted']} the other")

    if args.reload_pairs:
        model = args.models[0]
        backend = OllamaBackend(model=model, constrained=True)
        wide = [p for p in collect_prompts(args.game, args.reload_pairs + 10)
                if len(p) > 700][:args.reload_pairs]
        backend.complete(wide[0], SCHEMA)
        paired = paired_reload_test(backend, wide, args.repeats)
        paired["model"] = model
        results["reload_pairs"] = paired
        print(f"\npaired reload test on {paired['prompts']} prompts, "
              f"{args.repeats} repeats each:")
        print(f"  weights resident {paired['resident_stable']:.0%} repeated identically")
        print(f"  weights reloaded {paired['reloaded_stable']:.0%} repeated identically")
        print(f"  disagreed on {paired['stable_only_when_resident']} prompts one way and "
              f"{paired['stable_only_when_reloaded']} the other")

    _save(out, results)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
