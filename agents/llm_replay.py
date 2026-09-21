"""Replaying a language model's decisions instead of asking it again.

A served model does not repeat itself reliably (Section on the LLM seat in the paper, and
`experiments/run_llm_determinism.py` for what that does and does not depend on), so a run that
contains one cannot be rerun into the same event log. Six candidate causes were measured and none
of them is a setting a caller can turn: temperature is already zero, a pinned seed changes nothing,
so do reloading the weights, pinning the server to one thread, loading the machine, and defeating
the prompt cache.

What can be made exact is the *run*. Every decision an `LLMAgent` makes is journalled with the
prompt that produced it, that journal is stored in the run's package alongside its event log, and
this backend answers from the journal rather than from a model. The claim it supports is therefore
precise, and narrower than it looks: **the recording reproduces, the model does not**. That is why
`Agent.reproducibility()` returns `replay` here rather than `byte`, and why the stored record says
which of the two a reader is holding.

Two properties are deliberate.

**A prompt that was never recorded is an error, not a fallback.** If a replayed run diverges, for
any reason, some agent asks something the recording does not contain, and the replay stops there
rather than quietly inventing an answer. A replay that silently filled gaps would be a different
experiment wearing the old one's identity.

**A repeated prompt replays in order.** The same board state can recur in a long match, and two
decisions made at the same prompt need not be the same decision, so lookups consume entries rather
than reading them.

    from gamebrains.agents.llm import LLMAgent
    from gamebrains.agents.llm_replay import ReplayBackend

    agent = LLMAgent("LLM 0", ReplayBackend(journal))
"""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from typing import Any, Iterable

from .llm_backends import DEFAULT_MAX_ATTEMPTS, Attempt, Capabilities, Response, SchemaViolation


class NotRecorded(Exception):
    """The replay was asked something the recording does not contain.

    Raised rather than answered, because the only honest thing a replay can do at that point is
    stop: the run has left the path the recording describes, and every decision after this one
    would be about a different experiment.
    """


def prompt_key(prompt: str) -> str:
    """Prompts are long and repeat, so the journal is indexed by digest."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class ReplayBackend:
    """A backend that answers from a recorded journal.

    Args:
        journal: what `LLMAgent.journal()` returned on the recorded run, or the same list read
            back out of a stored package.
        model: what to report as the model. Defaults to whatever the journal recorded, so a
            replayed record still names the model whose decisions it carries.
    """

    def __init__(self, journal: Iterable[dict[str, Any]], model: str = "") -> None:
        self.entries = [dict(entry) for entry in journal]
        self.model = model or next((e.get("model", "") for e in self.entries if e.get("model")), "")
        self._by_prompt: dict[str, deque] = defaultdict(deque)
        for entry in self.entries:
            self._by_prompt[prompt_key(entry["prompt"])].append(entry)
        self._served = 0

    def capabilities(self) -> Capabilities:
        return Capabilities(
            name="replay",
            model=self.model,
            native_schema=False,
            activations=False,
            deterministic=True,
            note=("Decisions are read back from the recording of an earlier run, so the rerun is "
                  "exact. What reproduces is that recording rather than the model, which does not "
                  "repeat itself reliably even at temperature zero."),
        )

    def complete(self, prompt: str, schema: dict[str, Any],
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> Response:
        queue = self._by_prompt.get(prompt_key(prompt))
        if not queue:
            raise NotRecorded(
                f"this run asked something the recording does not contain, after {self._served} "
                f"replayed decision(s). The rerun has diverged from the recorded one, so there is "
                f"nothing truthful to answer with.\n\nThe prompt was:\n{prompt[:400]}")
        entry = queue.popleft()
        self._served += 1
        # The recorded attempt count travels with the decision, so a replayed run still reports
        # what the original cost rather than flattering it to one attempt.
        attempts = [Attempt(len(prompt), "", ok=False, error="replayed")
                    for _ in range(max(0, int(entry.get("attempts", 1)) - 1))]
        attempts.append(Attempt(len(prompt), "", ok=True))
        return Response(data=dict(entry["data"]), attempts=attempts, model=self.model)

    def remaining(self) -> int:
        """Decisions in the journal that no lookup has consumed.

        A replay that finishes with entries left over ran a shorter match than the recording, which
        is legitimate and worth being able to see.
        """
        return sum(len(q) for q in self._by_prompt.values())


def journal_from_package(package: dict[str, Any], agent: str = "") -> list[dict[str, Any]]:
    """The decisions a stored run's package holds, optionally for one named agent.

    `repository/record.py` writes them under `llm_decisions` as a mapping from agent name to that
    agent's journal, because a match may seat more than one model and replaying the wrong one would
    be worse than failing.
    """
    decisions = package.get("llm_decisions") or {}
    if agent:
        if agent not in decisions:
            raise NotRecorded(f"the package has no decisions for {agent!r}. "
                              f"It holds: {sorted(decisions) or 'none'}")
        return list(decisions[agent])
    if len(decisions) != 1:
        raise NotRecorded(
            f"the package holds decisions for {len(decisions)} agents, so one has to be named. "
            f"It holds: {sorted(decisions) or 'none'}")
    return list(next(iter(decisions.values())))
