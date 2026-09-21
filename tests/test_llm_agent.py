"""The LLM seat, tested against a scripted backend rather than a model.

A language model cannot be asked to produce the interesting cases on demand. It answers well on
the run where you wanted a malformed reply and badly on the run where you wanted a clean one, so
every case here drives a stub that returns prepared strings. The one thing a real model is needed
for, that it answers this platform's own prompt at all, is covered in `test_llm_backends.py` and
skips when no server is running.

Four of these pin down behaviour the rest of the codebase depends on:

  * the same agent plays two games whose action indices mean opposite things, by name
  * a game declaring neither role is refused rather than guessed at
  * exhausting the repair budget stops the match instead of substituting a move
  * a roster holding one of these withholds the platform's reproduction claim, and the Smart
    Filter stops offering the run as a finished answer

Run: python -m gamebrains.tests.test_llm_agent
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..agents.llm import LLMAgent, decision_schema
from ..agents.llm_replay import NotRecorded, ReplayBackend
from ..agents.llm_backends import (
    Attempt, Capabilities, Response, SchemaViolation, run_with_repair,
)
from ..agents.qlearning import QLearningAgent
from ..engine.game import Game
from ..engine.semantics import bind_agents
from ..games.congestion import CongestionGame
from ..games.public_goods import PublicGoodsGame
from ..repository.ledger import Ledger
from ..repository.record import record_experiment, reproducibility_of
from ..repository.smart_filter import lookup


class ScriptedBackend:
    """A backend whose model always says what the test wants, in order.

    Implements the same contract as `OllamaBackend` and runs the real repair loop, so a test that
    passes here exercises the code path a real backend takes rather than a simplified one.
    """

    def __init__(self, *answers: str, deterministic: bool = False,
                 native_schema: bool = False) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []
        self._deterministic = deterministic
        self._native_schema = native_schema

    def capabilities(self) -> Capabilities:
        return Capabilities(name="scripted", model="scripted-1b",
                            native_schema=self._native_schema, activations=False,
                            deterministic=self._deterministic,
                            note="" if self._deterministic else "a scripted stub, sampling anyway")

    def _ask(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers[min(len(self.prompts) - 1, len(self.answers) - 1)]

    def complete(self, prompt: str, schema: dict[str, Any], max_attempts: int = 4) -> Response:
        data, attempts = run_with_repair(self._ask, prompt, schema, max_attempts)
        return Response(data=data, attempts=attempts, model="scripted-1b")


def _answer(action: str, rationale: str = "because", confidence: float = 0.6) -> str:
    return json.dumps({"action": action, "rationale": rationale, "confidence": confidence})


class _CoordinationGame(Game):
    """A game with no concede axis at all, which is the case the refusal path exists for."""

    n_agents, n_actions, n_states, name = 2, 2, 2, "coordination_stub"
    action_names = ["Option A", "Option B"]
    action_roles = {"option_a": 0, "option_b": 1}

    def reset(self) -> list[int]:
        return [0, 0]

    def step(self, actions: list[int]):
        raise NotImplementedError


def test_one_agent_plays_two_games_whose_indices_mean_opposite_things() -> None:
    """The defect this whole declaration mechanism exists for, in the one agent that is asked for
    meaning in words. `claim` is action 0 in the Public Goods Game (defecting) and action 1 in the
    congestion family (rushing the jar). An agent that answered with an index would be right in
    one of them and confidently backwards in the other."""
    pgg = PublicGoodsGame(n_agents=3, rounds=10)
    hjg = CongestionGame(n_agents=3)

    in_pgg = LLMAgent("LLM pgg", ScriptedBackend(_answer("claim")))
    in_pgg.on_match_start(pgg)
    in_hjg = LLMAgent("LLM hjg", ScriptedBackend(_answer("claim")))
    in_hjg.on_match_start(hjg)

    assert in_pgg.act(3) == pgg.action_roles["claim"]
    assert in_hjg.act(0) == hjg.action_roles["claim"]
    assert in_pgg.act(3) != in_hjg.act(0), (
        "the two games put 'claim' on different indices, so one answer must map to two actions")
    print(f"OK: 'claim' resolved to {in_pgg.act(3)} in the Public Goods Game and "
          f"{in_hjg.act(0)} in the congestion family, from one unchanged agent")


def test_a_game_with_no_concede_axis_is_refused_by_name() -> None:
    agent = LLMAgent("LLM", ScriptedBackend(_answer("claim")))
    try:
        agent.on_match_start(_CoordinationGame())
    except ValueError as exc:
        assert "concede" in str(exc)
        assert "option_a" in str(exc), "the refusal must name what the game does declare"
    else:
        raise AssertionError("a game without the roles must be refused, not guessed at")
    print("OK: a pure coordination game is declined, naming the roles it does have")


def test_the_runner_accepts_it_as_role_bound_and_it_cannot_act_unbound() -> None:
    """`engine.semantics.bind_agents` refuses an agent that declares a dependence on meaning and
    then asks the game nothing. This agent has to pass that check for real, through the runner's
    own function rather than a test's imitation of it."""
    game = PublicGoodsGame(n_agents=2, rounds=5)
    agent = LLMAgent("LLM", ScriptedBackend(_answer("concede")))

    unbound = LLMAgent("LLM unbound", ScriptedBackend(_answer("concede")))
    try:
        unbound.act(0)
    except RuntimeError as exc:
        assert "on_match_start" in str(exc)
    else:
        raise AssertionError("acting before binding must fail loudly")

    bind_agents(game, [agent])
    assert agent.act(3) in (0, 1)
    print("OK: binds through the runner's own check, and refuses to act before it")


def test_a_repaired_decision_is_visible_in_the_panel_rather_than_smoothed_over() -> None:
    """The repair is the cost of using a model. A panel that showed only the final answer would
    report a clean decision that actually took three requests."""
    backend = ScriptedBackend("I would claim it.", _answer("wait"), _answer("claim", "nobody moved"))
    agent = LLMAgent("LLM", backend)
    agent.on_match_start(PublicGoodsGame(n_agents=3, rounds=10))

    action = agent.act(4)
    panel = agent.render_brain()

    assert action == PublicGoodsGame.action_roles["claim"]
    assert panel["attempts"] == 3 and panel["repaired"] is True
    assert panel["rationale"] == "nobody moved"
    assert panel["role"] == "claim"
    assert panel["activations_available"] is False
    assert panel["deterministic"] is False and panel["note"]
    assert "could not be used" in backend.prompts[1], "the retry must carry the failure back"
    print(f"OK: the panel reports {panel['attempts']} attempts and says why activations are absent")


def test_exhausting_the_budget_stops_the_match_instead_of_substituting_a_move() -> None:
    """There is no fallback action on purpose. A substituted move would be written to the event log
    indistinguishable from a decision the model made, and every metric downstream would read it as
    one."""
    agent = LLMAgent("LLM", ScriptedBackend(_answer("wait")), max_attempts=3)
    agent.on_match_start(PublicGoodsGame(n_agents=3, rounds=10))
    try:
        agent.act(4)
    except SchemaViolation as exc:
        assert "after 3 attempts" in str(exc)
    else:
        raise AssertionError("a model that never answers in schema must stop the match")
    print("OK: the budget runs out and the match stops, with no invented action")


def test_the_prompt_carries_the_roles_the_observation_gloss_and_the_history() -> None:
    """`build_prompt` is what the experiment script measures, so what it contains is part of the
    result rather than an implementation detail."""
    agent = LLMAgent("LLM", ScriptedBackend(_answer("concede")))
    agent.on_match_start(PublicGoodsGame(n_agents=3, rounds=10))

    first = agent.build_prompt(4)
    assert "concede" in first and "claim" in first
    assert "how many players took the conceding action" in first, (
        "the prompt must say what the observation means, in the game's own declared terms")
    assert "No rounds have been played yet" in first

    action = agent.act(4)
    agent.update(4, action, 1.5, 2, False)
    later = agent.build_prompt(2)
    assert "you played concede and received 1.50" in later
    print("OK: roles, the declared observation meaning and the last rounds all reach the model")


def test_a_roster_with_a_model_withholds_the_reproduction_claim() -> None:
    """The fourth refusal. The claim is that one configuration gives one byte-identical event log,
    and it cannot cover a roster containing a language model, so the run says so."""
    game = PublicGoodsGame(n_agents=2, rounds=5)
    q = QLearningAgent("Q0", n_states=game.n_states, n_actions=game.n_actions)
    llm = LLMAgent("LLM 1", ScriptedBackend(_answer("concede")))

    assert reproducibility_of([q]) == {"tier": "byte", "byte_identical_claimed": True,
                                       "nondeterministic_agents": []}
    mixed = reproducibility_of([q, llm])
    assert mixed["tier"] == "none" and mixed["byte_identical_claimed"] is False
    assert mixed["nondeterministic_agents"][0]["name"] == "LLM 1"
    assert mixed["nondeterministic_agents"][0]["reason"], "the caveat must carry its reason"

    honest = LLMAgent("LLM det", ScriptedBackend(_answer("concede"), deterministic=True))
    assert reproducibility_of([honest])["tier"] == "byte", (
        "the tier is read from the backend, so a backend that could genuinely promise it would "
        "be believed rather than refused for being an LLM")
    print("OK: the run records the weakest tier in its roster, and names the agent that set it")


def test_a_replayed_decision_is_the_recorded_one_and_says_so() -> None:
    """The exactness a served model cannot offer, bought from the recording instead.

    The tier matters as much as the decisions: a replayed run reproduces, and what reproduces is
    the recording rather than the model, so it is `replay` and not `byte`.
    """
    game = PublicGoodsGame(n_agents=3, rounds=10)
    live = LLMAgent("LLM 0", ScriptedBackend(
        _answer("claim", "first"), _answer("concede", "second"), _answer("claim", "third")))
    live.on_match_start(game)
    # Interleaved exactly as a match drives it, because the prompt carries the history and a
    # recording made in a different order would describe a different sequence of questions.
    actions = []
    for obs in (4, 2, 1):
        actions.append(live.act(obs))
        live.update(obs, actions[-1], 1.0, obs, False)

    journal = live.journal()
    assert len(journal) == 3 and journal[0]["data"]["rationale"] == "first"

    replayed = LLMAgent("LLM 0", ReplayBackend(journal))
    replayed.on_match_start(game)
    again = []
    for obs, action in zip((4, 2, 1), actions):
        again.append(replayed.act(obs))
        replayed.update(obs, again[-1], 1.0, obs, False)

    assert again == actions, "a replay has to give the decisions the recording holds"
    assert replayed.render_brain()["rationale"] == "third"
    assert replayed.reproducibility() == "replay"
    assert reproducibility_of([replayed])["tier"] == "replay"
    assert reproducibility_of([replayed])["byte_identical_claimed"] is False, (
        "a replayed run is exact and is still not a substitute for running the thing")
    print("OK: replay returns the recorded decisions and calls itself a replay")


def test_a_replay_that_wanders_off_the_recording_stops() -> None:
    """The alternative is a replay that fills the gap, which would be a different experiment
    wearing the recorded one's identity."""
    game = PublicGoodsGame(n_agents=3, rounds=10)
    live = LLMAgent("LLM 0", ScriptedBackend(_answer("claim")))
    live.on_match_start(game)
    live.act(4)

    replayed = LLMAgent("LLM 0", ReplayBackend(live.journal()))
    replayed.on_match_start(game)
    assert replayed.act(4) == PublicGoodsGame.action_roles["claim"]

    try:
        replayed.act(1)                      # a state the recorded run never reached
    except NotRecorded as exc:
        assert "diverged" in str(exc) and "1 replayed decision" in str(exc)
    else:
        raise AssertionError("a prompt the recording does not hold must stop the replay")
    print("OK: a divergent replay stops and says where")


def test_a_repeated_prompt_replays_in_order() -> None:
    """The same board state recurs in a long match, and two decisions taken at the same prompt need
    not agree, so lookups consume entries rather than reading them."""
    game = PublicGoodsGame(n_agents=3, rounds=10)
    live = LLMAgent("LLM 0", ScriptedBackend(_answer("claim"), _answer("concede")))
    live.on_match_start(game)
    first, second = live.act(4), live.act(4)      # identical prompts, no history recorded between
    assert first != second

    replayed = LLMAgent("LLM 0", ReplayBackend(live.journal()))
    replayed.on_match_start(game)
    assert [replayed.act(4), replayed.act(4)] == [first, second]
    assert ReplayBackend(live.journal()).remaining() == 2
    print("OK: repeated prompts replay in the order they were recorded")


def test_the_smart_filter_stops_offering_an_unreproducible_run_as_a_finished_answer() -> None:
    """Reusing such a record would hand back one sample of a random process in place of the run
    that was asked for. It stays visible as similar prior work, which answers a different question.
    """
    tmp = Path(tempfile.mkdtemp(prefix="gamebrains_llm_record_"))
    try:
        log_path = tmp / "log.jsonl"
        log_path.write_text('{"type": "meta", "seed": 0}\n', encoding="utf-8")

        game = PublicGoodsGame(n_agents=2, rounds=20)
        roster = [LLMAgent("LLM 0", ScriptedBackend(_answer("concede"))),
                  QLearningAgent("Q1", n_states=game.n_states, n_actions=game.n_actions)]
        record = record_experiment(tmp, game, roster, log_path, rounds=20, seed=0, metrics={})

        assert record["reproducibility"]["byte_identical_claimed"] is False
        assert Ledger(tmp).verify_chain(), "the caveat is signed with the rest of the record"

        # The stored description, not `game.describe()`: the recorder strips `rounds` before
        # hashing (see record._config_game_desc), so querying with the raw description would miss
        # for that reason instead of for the one under test, and this assertion would pass while
        # checking nothing.
        ledger = Ledger(tmp)
        hits = lookup(ledger, record["game"], record["roster"], record["code_version"],
                      rounds=20, master_seed=0, feature_vector=record["feature_vector"])
        assert hits["exact"] is None, "an unreproducible run must not be offered as exact reuse"
        assert any(r["content_cid"] == record["content_cid"] for r, _ in hits["similar"]), (
            "it must still be findable as prior work")

        deterministic = [QLearningAgent(f"Q{i}", n_states=game.n_states,
                                        n_actions=game.n_actions) for i in range(2)]
        det_record = record_experiment(tmp, game, deterministic, log_path, rounds=20, seed=0,
                                       metrics={})
        found = Ledger(tmp).find_exact(det_record["game"], det_record["roster"],
                                       det_record["code_version"], 20, 0)
        assert found is not None, "a reproducible run is still offered, so the guard is not blanket"
        print("OK: unreproducible runs are excluded from reuse and kept as prior work")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_schema_asks_for_more_than_an_enum() -> None:
    """A single enum is the case a constrained server makes trivial, so a study built on one would
    measure nothing. The free-text and bounded-number fields are where models actually fail."""
    schema = decision_schema(("concede", "claim"))
    assert schema["properties"]["action"]["enum"] == ["concede", "claim"]
    assert set(schema["required"]) == {"action", "rationale", "confidence"}
    assert schema["properties"]["confidence"]["maximum"] == 1.0
    print("OK: the decision schema has an enum, free text and a bounded number")


def test_a_response_without_attempts_is_not_reported_as_repaired() -> None:
    assert Response(data={}, attempts=[Attempt(1, "{}", ok=True)]).repaired is False


if __name__ == "__main__":
    test_one_agent_plays_two_games_whose_indices_mean_opposite_things()
    test_a_game_with_no_concede_axis_is_refused_by_name()
    test_the_runner_accepts_it_as_role_bound_and_it_cannot_act_unbound()
    test_a_repaired_decision_is_visible_in_the_panel_rather_than_smoothed_over()
    test_exhausting_the_budget_stops_the_match_instead_of_substituting_a_move()
    test_the_prompt_carries_the_roles_the_observation_gloss_and_the_history()
    test_a_roster_with_a_model_withholds_the_reproduction_claim()
    test_a_replayed_decision_is_the_recorded_one_and_says_so()
    test_a_replay_that_wanders_off_the_recording_stops()
    test_a_repeated_prompt_replays_in_order()
    test_the_smart_filter_stops_offering_an_unreproducible_run_as_a_finished_answer()
    test_the_schema_asks_for_more_than_an_enum()
    test_a_response_without_attempts_is_not_reported_as_repaired()
    print("\nall LLM agent tests passed")
