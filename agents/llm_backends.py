"""What an LLM backend must provide, and what it must admit it cannot.

`agents/llm.py` asks a language model for a decision. How that model is served changes what can be
observed about the decision, and the two ways of serving one locally differ in exactly the property
this platform cares about most:

    Ollama            an HTTP server over quantised GGUF weights. Structured output is native and
                      the model is fast, but there is no PyTorch module to attach to, so hidden
                      activations do not exist as far as a caller is concerned.

    transformers      the model loaded in-process as a torch module. Forward hooks could record
                      any layer's activations, which is what interpretability tooling such as
                      TransformerLens and nnsight is built on, at roughly an order of magnitude
                      more compute and memory. **Not implemented.** It is described here because
                      it is what the capability declaration exists to accommodate, and describing
                      it is not the same as shipping it.

What the platform actually ships is the Ollama backend and the replay backend of
`agents/llm_replay.py`. **Neither serving strategy is a superset of the other**, so the contract
is written so that a backend declares its own capabilities rather than failing quietly at the
point of use. That is the
same decision the rest of this codebase makes about games declaring their action semantics, and it
exists for the same reason: a caller that asks for something a component cannot provide should be
told so, not handed an empty structure that looks like a measurement.

The immediate case is `render_brain`. A panel that shows an empty activation plot for one backend
and a populated one for another, with nothing saying why, is worse than a panel that says the
backend does not expose activations.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

#: How many times a backend may be asked to repair its own output before the caller gives up.
#: Not a tuning knob so much as a budget: past a handful of attempts the model is not going to
#: produce the schema, and the interesting number is how often that happens rather than whether a
#: larger budget would eventually hide it.
DEFAULT_MAX_ATTEMPTS = 4


class SchemaViolation(Exception):
    """The model's output did not satisfy the requested schema.

    Carries the text so the caller can feed the failure back to the model, which is the whole
    repair strategy, and so a failed attempt can be recorded rather than only counted.
    """

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


class BackendUnavailable(Exception):
    """The backend cannot be reached or its optional dependency is absent.

    Distinct from `SchemaViolation` on purpose. One says the model answered badly, the other says
    there was no model to answer, and a study that conflates them reports an interesting failure
    rate that is really an installation problem.
    """


@dataclass
class Attempt:
    """One request and what came back, kept whether or not it succeeded.

    The failed attempts are the point. A backend that quietly retried until something parsed would
    report a clean success rate and hide the cost that makes structured output worth measuring.
    """

    prompt_chars: int
    raw: str
    ok: bool
    error: str = ""
    seconds: float = 0.0

    #: Tokens in and out, when the server reports them. Wall-clock on a contended laptop is not a
    #: quantity that transfers between machines (see `experiments/run_benchmark.py`, where the same
    #: native operation measured 56% apart within one run), while a token count is the same number
    #: wherever the model runs. None where the backend does not report them.
    prompt_tokens: Optional[int] = None
    eval_tokens: Optional[int] = None


@dataclass
class Response:
    """A parsed decision plus everything the platform wants to be able to say about it."""

    data: dict[str, Any]
    attempts: list[Attempt] = field(default_factory=list)
    activations: Optional[dict[str, Any]] = None
    model: str = ""

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def repaired(self) -> bool:
        """True when the first attempt failed and a later one succeeded."""
        return self.n_attempts > 1

    @property
    def seconds(self) -> float:
        return sum(a.seconds for a in self.attempts)


@dataclass(frozen=True)
class Capabilities:
    """What a backend can report about its own decisions.

    Every field is a claim the backend makes about itself, and `render_brain` and the metrics layer
    read these instead of guessing from the backend's class name. A field that is False is not a
    defect to be worked around: it is the reason the caller should not build a display or a
    measurement that depends on it.
    """

    name: str
    model: str

    native_schema: bool
    """The server constrains generation to the schema itself. Where this is False the schema is a
    request in the prompt, which is the case worth measuring: a repair loop only earns its keep
    when the model can disobey."""

    activations: bool
    """Hidden states can be recorded per decision."""

    deterministic: bool
    """Two identical requests give byte-identical output, holding the model fixed. **This is False
    for every LLM backend here**, and it matters beyond this file: the platform's reproducibility
    claim is that one configuration gives one byte-identical event log, and a roster containing an
    LLM cannot support that claim. It is recorded rather than hidden so the claim can be withheld
    for such a roster instead of quietly weakening for every roster."""

    note: str = ""
    """Why a False above is False, in a sentence a results page can show."""


class Backend(Protocol):
    """The contract `agents/llm.py` programs against."""

    def capabilities(self) -> Capabilities:
        ...

    def complete(self, prompt: str, schema: dict[str, Any],
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> Response:
        """Return a parsed object satisfying `schema`, repairing at most `max_attempts` times.

        Raises `SchemaViolation` if the budget is exhausted, carrying the last raw output.
        Raises `BackendUnavailable` if there was nothing to ask.
        """
        ...


# --- schema checking ------------------------------------------------------------------------
#
# Deliberately a small hand-written checker rather than the `jsonschema` package, for two reasons.
# The schemas here are one flat object with typed fields and an enum, which is the shape both
# backends can send to a server, and the error strings are fed back to the model as repair
# instructions, so they have to read as instructions rather than as validator diagnostics.


def validate(data: Any, schema: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems, empty when the data satisfies the schema."""
    problems: list[str] = []
    if not isinstance(data, dict):
        return [f"expected a JSON object, got {type(data).__name__}"]

    properties = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in data:
            problems.append(f"the field {key!r} is missing")

    for key, value in data.items():
        spec = properties.get(key)
        if spec is None:
            if not schema.get("additionalProperties", True):
                problems.append(f"the field {key!r} is not part of the schema")
            continue

        expected = spec.get("type")
        if expected == "string" and not isinstance(value, str):
            problems.append(f"{key!r} must be a string, got {type(value).__name__}")
        elif expected == "integer" and not isinstance(value, int):
            problems.append(f"{key!r} must be an integer, got {type(value).__name__}")
        elif expected == "number" and not isinstance(value, (int, float)):
            problems.append(f"{key!r} must be a number, got {type(value).__name__}")
        elif expected == "boolean" and not isinstance(value, bool):
            problems.append(f"{key!r} must be true or false, got {type(value).__name__}")

        if "enum" in spec and value not in spec["enum"]:
            problems.append(f"{key!r} must be one of {spec['enum']}, got {value!r}")
        if "minimum" in spec and isinstance(value, (int, float)) and value < spec["minimum"]:
            problems.append(f"{key!r} must be at least {spec['minimum']}, got {value}")
        if "maximum" in spec and isinstance(value, (int, float)) and value > spec["maximum"]:
            problems.append(f"{key!r} must be at most {spec['maximum']}, got {value}")

    return problems


def extract_json(text: str) -> Any:
    """Parse the first JSON object in `text`, tolerating what models put around it.

    Models wrap objects in prose and in fenced code blocks even when told not to. Being strict
    here would report a schema failure for output that satisfies the schema perfectly well, which
    would inflate exactly the rate this module exists to measure.
    """
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1] if "```" in body[3:] else body[3:]
        if body.lstrip().lower().startswith("json"):
            body = body.lstrip()[4:]
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise SchemaViolation("no JSON object found in the output", raw=text)
    try:
        return json.loads(body[start:end + 1])
    except json.JSONDecodeError as exc:
        raise SchemaViolation(f"the output is not valid JSON: {exc}", raw=text) from exc


def repair_prompt(original: str, raw: str, problems: list[str]) -> str:
    """Build the follow-up that tells the model what was wrong with its own answer.

    Kept as one function so the repair wording is identical across backends. If each backend
    phrased its own, a difference in failure rates between them would partly be a difference in
    how clearly they complained, and the comparison would measure the wrong thing.
    """
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        f"{original}\n\n"
        f"Your previous answer was:\n{raw}\n\n"
        f"It could not be used:\n{listed}\n\n"
        f"Reply with the corrected JSON object and nothing else."
    )


def run_with_repair(ask, prompt: str, schema: dict[str, Any],
                    max_attempts: int) -> tuple[dict[str, Any], list[Attempt]]:
    """Shared repair loop. `ask(prompt) -> str` is whatever the backend does to reach its model.

    Returns the parsed object and every attempt, successful or not. Both backends use this so the
    measured cost of structured output is a property of the model rather than of two separately
    written loops.
    """
    attempts: list[Attempt] = []
    current = prompt

    for _ in range(max_attempts):
        started = time.perf_counter()
        raw = ask(current)
        elapsed = time.perf_counter() - started

        try:
            data = extract_json(raw)
            problems = validate(data, schema)
        except SchemaViolation as exc:
            attempts.append(Attempt(len(current), raw, ok=False, error=str(exc), seconds=elapsed))
            current = repair_prompt(prompt, raw, [str(exc)])
            continue

        if not problems:
            attempts.append(Attempt(len(current), raw, ok=True, seconds=elapsed))
            return data, attempts

        attempts.append(Attempt(len(current), raw, ok=False,
                                error="; ".join(problems), seconds=elapsed))
        current = repair_prompt(prompt, raw, problems)

    last = attempts[-1] if attempts else Attempt(len(prompt), "", ok=False)
    raise SchemaViolation(
        f"no schema-valid answer after {max_attempts} attempts: {last.error}", raw=last.raw)
