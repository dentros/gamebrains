"""The two-player matrix game, the scripted opponent, and the prompt profiles that tell a model a
repeated matrix game with its whole history.

Run: python -m gamebrains.tests.test_matrix_game
"""

from __future__ import annotations

import io
import json
import urllib.request
from typing import Any

from ..agents import factory
from ..agents.classic import AllC, Scripted
from ..agents.llm import LLMAgent
from ..agents.llm_backends import (Response, SchemaViolation, parse_move_line, repair_move_prompt,
                                   run_with_repair)
from ..agents.llm_ollama import OllamaBackend
from ..engine.runner import run_match
from ..games.matrix_game import COOPERATE, DEFECT, MatrixGame, from_preset
from ..games.public_goods import PublicGoodsGame
from ..repository.normalize import build_config, hash_config
from ..repository.record import (DEFAULT_CODE_VERSION, _agent_params, _config_game_desc,
                                 _roster_description)


def test_the_four_payoffs_are_paid_as_written():
    game = from_preset("pd", rounds=4)
    game.reset()
    cases = {(COOPERATE, COOPERATE): [3, 3], (COOPERATE, DEFECT): [0, 5],
             (DEFECT, COOPERATE): [5, 0], (DEFECT, DEFECT): [1, 1]}
    for actions, rewards in cases.items():
        result = game.step(list(actions))
        assert result.rewards == rewards, (actions, result.rewards)
        assert result.observations == [actions.count(COOPERATE)] * 2
    assert result.done and game.max_welfare_per_round() == 6


def test_every_agent_that_can_read_a_count_plays_it():
    game = from_preset("pd", rounds=30)
    for kind in ("qlearning", "dqn", "dac", "markov_brain", "markov_gates"):
        roster = [factory.build(kind, 0, game, 1), factory.build("classic", 1, game, 1,
                                                                classic_strategy="MajorityTFT")]
        records = run_match(game, roster, rounds=30, seed=1)
        assert len(records["actions"]) == 30, kind
    try:
        factory.build("fep", 0, game, 1)
    except ValueError as exc:
        assert "declares neither" in str(exc)
    else:
        raise AssertionError("the FEP agent has no payoff model for this game and must say so")


def test_the_payoffs_are_the_design():
    def h(game):
        return hash_config(build_config(_config_game_desc(game), [], DEFAULT_CODE_VERSION, {}))
    assert h(from_preset("pd")) != h(from_preset("pd", S=-1.0))
    assert h(from_preset("pd", rounds=14)) == h(from_preset("pd", rounds=24)), \
        "rounds is scale, not design"


def test_bad_games_are_refused():
    for kwargs in ({"T": "5"}, {"R": float("nan")}, {"P": True}):
        try:
            MatrixGame(**{"T": 5, "R": 3, "P": 1, "S": 0, **kwargs})
        except ValueError:
            continue
        raise AssertionError(f"accepted {kwargs}")
    try:
        MatrixGame(5, 3, 1, 0, n_agents=3)
    except ValueError as exc:
        assert "two-player" in str(exc)
    else:
        raise AssertionError("a three-player matrix game was accepted")


def test_a_scripted_agent_plays_its_plan_by_role():
    plan = ["concede"] * 3 + ["claim"] + ["concede"] * 2
    for game in (from_preset("pd", rounds=6), PublicGoodsGame(n_agents=2, rounds=6)):
        agent = Scripted("S", plan)
        records = run_match(game, [agent, AllC("c")], rounds=6, seed=0)
        played = [a[0] for a in records["actions"]]
        assert played == [game.action_for(r) for r in plan], game.name
    assert from_preset("pd").action_for("concede") != PublicGoodsGame(2).action_for("concede"), \
        "the two games must number the roles differently, or this case tests nothing"
    assert _agent_params(Scripted("S", plan))["plan"] == tuple(plan)
    for bad in ([], ["cooperate"]):
        try:
            Scripted("S", bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted the plan {bad}")
    try:
        run_match(from_preset("pd", rounds=7), [Scripted("S", plan), AllC("c")], rounds=7, seed=0)
    except RuntimeError as exc:
        assert "round 7" in str(exc)
    else:
        raise AssertionError("a match longer than the plan was played on")


class PDStub:
    """A backend that answers in order and keeps what it was sent, system message included."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.systems: list[Any] = []
        self.temperature = 0.7

    def capabilities(self):
        from ..agents.llm_backends import Capabilities
        return Capabilities(name="stub", model="stub-1b", native_schema=False, activations=False,
                            deterministic=False, note="stub")

    def complete(self, prompt, schema, max_attempts=4, system=None, reply="json"):
        self.systems.append(system)

        def ask(text):
            self.prompts.append(text)
            return self.answers.pop(0)
        parse = parse_move_line(schema["properties"]["action"]["enum"]) if reply == "move" else None
        data, attempts = run_with_repair(ask, prompt, schema, max_attempts, parse=parse,
                                         repair=repair_move_prompt if reply == "move" else None)
        return Response(data=data, attempts=attempts, model="stub-1b")


def test_the_history_profile_shows_both_moves_and_the_table():
    plan = ["concede"] * 2 + ["claim"] + ["concede"]
    agent = LLMAgent("LLM", PDStub(*["Fine.\nMOVE: Cooperate"] * 4), profile="pd-history-move-v1")
    run_match(from_preset("pd", rounds=4), [agent, Scripted("S", plan)], rounds=4, seed=0)
    system = agent.backend.systems[0]
    assert "a 4-round game against the same opponent" in system
    assert "You Cooperate and they Defect: you 0, they 5." in system
    assert "'MOVE: Cooperate' or 'MOVE: Defect'" in system
    assert agent.backend.prompts[0] == "Round 1 of 4. Your move."
    last = agent.backend.prompts[3]
    assert last.splitlines()[1:4] == ["round 1: you=Cooperate, opponent=Cooperate",
                                      "round 2: you=Cooperate, opponent=Cooperate",
                                      "round 3: you=Cooperate, opponent=Defect"], last
    assert agent.journal()[0]["system"] == system


def test_the_move_line_is_read_strictly():
    parse = parse_move_line(["Cooperate", "Defect"])
    assert parse("They defected, so I answer in kind.\nMOVE: Defect")["action"] == "Defect"
    assert parse("ok\nmove: cooperate.")["action"] == "Cooperate"
    for bad in ("I will cooperate.", "MOVE: Cooperate or Defect", "MOVE: pass", ""):
        try:
            parse(bad)
        except SchemaViolation:
            continue
        raise AssertionError(f"read {bad!r} as a move")
    # A reply the parser refuses is asked again, never read as one of the two.
    agent = LLMAgent("LLM", PDStub("I cooperate.", "Sure.\nMOVE: Defect"),
                     profile="pd-history-move-v1")
    run_match(from_preset("pd", rounds=1), [agent, AllC("c")], rounds=1, seed=0)
    assert agent.journal()[0]["attempts"] == 2 and agent._log[0]["role"] == "claim"
    assert "MOVE: <option>" in agent.backend.prompts[1]


def test_the_json_variant_and_its_refusal_outside_a_matrix_game():
    answer = json.dumps({"action": "Defect", "rationale": "r", "confidence": 0.5})
    agent = LLMAgent("LLM", PDStub(answer), profile="pd-history-v1")
    run_match(from_preset("pd", rounds=1), [agent, AllC("c")], rounds=1, seed=0)
    assert agent._log[0]["role"] == "claim" and "JSON object" in agent.backend.systems[0]
    try:
        run_match(PublicGoodsGame(n_agents=4, rounds=2),
                  [LLMAgent("LLM", PDStub(), profile="pd-history-v1")] +
                  [AllC(f"c{i}") for i in range(3)], rounds=2, seed=0)
    except ValueError as exc:
        assert "two-player matrix game" in str(exc)
    else:
        raise AssertionError("a pd_history profile played a four-player game")
    hashes = {hash_config(build_config({}, _roster_description([LLMAgent("x", PDStub(), profile=p)]),
                                       DEFAULT_CODE_VERSION, {}))
              for p in ("pd-history-v1", "pd-history-move-v1")}
    assert len(hashes) == 2, "the reply format is part of the design"


class _Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_ollama_sends_the_system_message_and_no_schema_for_a_move_reply():
    bodies = []

    def server(request, timeout=None):
        bodies.append(json.loads(request.data.decode("utf-8")))
        return _Reply(json.dumps({"response": "They cooperate.\nMOVE: Cooperate"}).encode())
    original = urllib.request.urlopen
    urllib.request.urlopen = server
    try:
        agent = LLMAgent("LLM", OllamaBackend(temperature=0.7, seed=5),
                         profile="pd-history-move-v1")
        run_match(from_preset("pd", rounds=2), [agent, AllC("c")], rounds=2, seed=0)
    finally:
        urllib.request.urlopen = original
    assert all("format" not in b for b in bodies), "a schema would force the reply into JSON"
    assert all(b["system"].startswith("You are playing a 2-round game") for b in bodies)
    assert [e["role"] for e in agent._log] == ["concede", "concede"]


if __name__ == "__main__":
    test_the_four_payoffs_are_paid_as_written()
    print("OK: the four payoffs are paid as written, and the count is published")
    test_every_agent_that_can_read_a_count_plays_it()
    print("OK: every agent that reads a count plays it; the FEP agent says why it cannot")
    test_the_payoffs_are_the_design()
    print("OK: the payoffs are the design and the rounds are not")
    test_bad_games_are_refused()
    print("OK: payoffs that are not finite numbers, and a third player, are refused")
    test_a_scripted_agent_plays_its_plan_by_role()
    print("OK: a scripted agent plays its plan by role, records it, and stops where it ends")
    test_the_history_profile_shows_both_moves_and_the_table()
    print("OK: the history profile shows the payoff table and both moves of every round")
    test_the_move_line_is_read_strictly()
    print("OK: the MOVE line is read strictly, and a reply it refuses is asked again")
    test_the_json_variant_and_its_refusal_outside_a_matrix_game()
    print("OK: the JSON variant works, and either variant refuses a game that is not 2x2")
    test_ollama_sends_the_system_message_and_no_schema_for_a_move_reply()
    print("OK: Ollama gets the system message, and no schema when the reply is a MOVE line")
