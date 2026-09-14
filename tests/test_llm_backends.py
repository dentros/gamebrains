"""The backend contract and the repair loop, tested without a model.

Everything here runs on a stub that returns scripted strings, which is the only way to test a
repair loop honestly: with a real model the interesting cases, malformed JSON and a value outside
an enum, appear when they feel like it and never on the run where you need them.

The Ollama-specific tests SKIP when no server is reachable, following `tests/test_interop_*.py`.
That is the pattern for optional infrastructure in this project: skip loudly with an install hint,
never pass silently.

Run: python -m gamebrains.tests.test_llm_backends
"""

from __future__ import annotations

from ..agents.llm_backends import (
    Attempt, BackendUnavailable, Capabilities, Response, SchemaViolation, extract_json,
    repair_prompt, run_with_repair, validate,
)
from ..agents.llm_ollama import OllamaBackend

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["concede", "claim"]},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["action", "reasoning"],
    "additionalProperties": False,
}

GOOD = '{"action": "claim", "reasoning": "nobody else moved", "confidence": 0.7}'


class Scripted:
    """Returns prepared answers in order, so a failure sequence can be constructed exactly."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)]


def test_validate_names_each_problem_rather_than_just_failing() -> None:
    """The messages are fed back to the model as repair instructions, so they have to be readable.

    A validator that returned only True or False would make the repair loop impossible: there
    would be nothing to tell the model except that it was wrong.
    """
    assert validate({"action": "claim", "reasoning": "x"}, SCHEMA) == []

    problems = validate({"action": "wait", "reasoning": 3, "confidence": 2.0}, SCHEMA)
    joined = " ".join(problems)
    assert "'action'" in joined and "concede" in joined, joined
    assert "'reasoning' must be a string" in joined, joined
    assert "at most 1.0" in joined, joined

    missing = validate({"confidence": 0.5}, SCHEMA)
    assert any("'action' is missing" in p for p in missing), missing
    assert any("'reasoning' is missing" in p for p in missing), missing

    extra = validate({"action": "claim", "reasoning": "x", "mood": "bold"}, SCHEMA)
    assert any("'mood' is not part of the schema" in p for p in extra), extra
    print("OK: validation names every problem in words a model can act on")


def test_extract_json_tolerates_what_models_actually_emit() -> None:
    """Strictness here would inflate the very failure rate this module exists to measure.

    Models wrap objects in prose and in fenced blocks even when told not to. That is a formatting
    habit, not a schema failure, and counting it as one would report a worse rate than the truth.
    """
    assert extract_json(GOOD)["action"] == "claim"
    assert extract_json(f"Sure, here you go:\n{GOOD}\nHope that helps.")["action"] == "claim"
    assert extract_json(f"```json\n{GOOD}\n```")["action"] == "claim"
    assert extract_json(f"```\n{GOOD}\n```")["action"] == "claim"

    for bad in ("I would claim the jar.", "{not json at all}"):
        try:
            extract_json(bad)
        except SchemaViolation as exc:
            assert exc.raw == bad
        else:
            raise AssertionError(f"expected a SchemaViolation for {bad!r}")
    print("OK: prose and code fences are parsed, genuine non-JSON is refused")


def test_repair_loop_succeeds_on_a_later_attempt_and_records_the_failures() -> None:
    """The failed attempts are the measurement. A loop that hid them would report a clean rate."""
    ask = Scripted("I think claiming is best.", '{"action": "wait", "reasoning": "hmm"}', GOOD)
    data, attempts = run_with_repair(ask, "Your move.", SCHEMA, max_attempts=4)

    assert data["action"] == "claim"
    assert len(attempts) == 3, attempts
    assert [a.ok for a in attempts] == [False, False, True]
    assert "no JSON object" in attempts[0].error
    assert "concede" in attempts[1].error

    # Each retry has to carry the failure back, otherwise the model is being asked the same
    # question again and any improvement is luck.
    assert "could not be used" in ask.prompts[1]
    assert "I think claiming is best." in ask.prompts[1]
    print(f"OK: repaired after {len(attempts)} attempts, and every failure was kept")


def test_repair_gives_up_and_says_what_was_wrong() -> None:
    ask = Scripted('{"action": "wait", "reasoning": "no"}')
    try:
        run_with_repair(ask, "Your move.", SCHEMA, max_attempts=3)
    except SchemaViolation as exc:
        assert "after 3 attempts" in str(exc)
        assert "concede" in str(exc), "the reason for giving up must name the last problem"
        assert exc.raw, "the last raw output must travel with the failure"
    else:
        raise AssertionError("a model that never satisfies the schema must raise")
    print("OK: exhausting the budget raises, carrying the reason and the last output")


def test_repair_prompt_is_shared_so_backends_are_comparable() -> None:
    """If each backend phrased its own complaint, a difference in their failure rates would partly
    be a difference in how clearly they complained, and the comparison would measure the wrong
    thing."""
    text = repair_prompt("Your move.", '{"action": "wait"}', ["'action' must be one of [...]"])
    assert text.startswith("Your move.")
    assert '{"action": "wait"}' in text
    assert "corrected JSON object" in text
    print("OK: one repair wording, so two backends can be compared on the model rather than on it")


def test_capabilities_are_declared_rather_than_inferred() -> None:
    """A backend says what it cannot do. `render_brain` reads this instead of guessing from a
    class name, so a panel can say why a plot is absent rather than drawing an empty one."""
    caps = OllamaBackend().capabilities()
    assert isinstance(caps, Capabilities)
    assert caps.activations is False, "Ollama serves over HTTP, there is no module to hook"
    assert caps.deterministic is False, (
        "no LLM backend may claim determinism: the platform's byte-identical reproduction claim "
        "has to be withheld for a roster containing one, and that decision reads this field")
    assert caps.note, "a False capability must carry the reason a results page can show"
    print(f"OK: capabilities declared, activations={caps.activations} "
          f"deterministic={caps.deterministic}")


def test_a_response_reports_its_own_cost() -> None:
    response = Response(
        data={"action": "claim", "reasoning": "x"},
        attempts=[Attempt(10, "bad", ok=False, error="e", seconds=0.2),
                  Attempt(40, GOOD, ok=True, seconds=0.3)],
        model="test")
    assert response.n_attempts == 2
    assert response.repaired is True
    assert abs(response.seconds - 0.5) < 1e-9
    assert Response(data={}, attempts=[Attempt(1, GOOD, ok=True)]).repaired is False
    print("OK: a response carries its attempt count and its latency")


def test_missing_server_is_reported_as_unavailable_not_as_a_bad_model() -> None:
    """Separating these matters for the study. A backend that reported an unreachable server as a
    schema failure would publish an installation problem as an interesting failure rate."""
    backend = OllamaBackend(host="http://127.0.0.1:1")     # nothing listens there
    assert backend.is_available() is False

    try:
        backend.installed_models()
    except BackendUnavailable as exc:
        assert "ollama" in str(exc).lower()
        assert "pull" in str(exc), "the error must tell a reader how to fix it"
    else:
        raise AssertionError("an unreachable server must raise BackendUnavailable")
    print("OK: an absent server is unavailable, which is not the same as a model answering badly")


def test_against_a_real_model_if_one_is_running() -> None:
    """The only test here that needs Ollama. SKIPs with a hint when absent, as the interop tests do."""
    backend = OllamaBackend()
    if not backend.is_available():
        print(f"SKIP: no Ollama model {backend.model!r} available "
              f"(install ollama, then `ollama serve` and `ollama pull {backend.model}`)")
        return

    prompt = ("You are playing a game. You may either concede the resource or claim it. "
              "Nobody else has moved yet. Reply with a JSON object containing action, "
              "reasoning and confidence.")
    response = backend.complete(prompt, SCHEMA)
    assert response.data["action"] in ("concede", "claim")
    assert isinstance(response.data["reasoning"], str) and response.data["reasoning"]
    print(f"OK: {backend.model} answered in {response.n_attempts} attempt(s), "
          f"{response.seconds:.2f}s, action={response.data['action']!r}")


if __name__ == "__main__":
    test_validate_names_each_problem_rather_than_just_failing()
    test_extract_json_tolerates_what_models_actually_emit()
    test_repair_loop_succeeds_on_a_later_attempt_and_records_the_failures()
    test_repair_gives_up_and_says_what_was_wrong()
    test_repair_prompt_is_shared_so_backends_are_comparable()
    test_capabilities_are_declared_rather_than_inferred()
    test_a_response_reports_its_own_cost()
    test_missing_server_is_reported_as_unavailable_not_as_a_bad_model()
    test_against_a_real_model_if_one_is_running()
    print("\nall LLM backend tests passed")
