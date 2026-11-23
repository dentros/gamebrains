"""The `neutral-v1` prompt profile, the per-request sampling seed, and the recorded temperature.

`neutral-v1` tells the model what `informed-v1` tells it in words that name no side, with the order
of the two options drawn from the agent's seed. These cases pin what that promises: no side is
named, the order really varies with the seed and nowhere else, an answer maps back to the role it
labelled, and a return that changes during the match is announced on the round it changes.

The Ollama cases replace the HTTP call, so they run without a server.

Run: python -m gamebrains.tests.test_neutral_profile
"""

from __future__ import annotations

import io
import json
import re
import urllib.request

from ..agents import llm_prompt
from ..agents.classic import AllD
from ..agents.llm import LLMAgent
from ..agents.llm_ollama import OllamaBackend
from ..engine.runner import run_match
from ..games.congestion import CongestionGame
from ..games.public_goods import PublicGoodsGame
from ..repository.normalize import build_config, hash_config
from ..repository.record import (DEFAULT_CODE_VERSION, _agent_params, _config_game_desc,
                                 _roster_description)
from .test_llm_agent import ScriptedBackend, _answer

#: Words that name a side or the game. "stay" and "move" are the congestion family's action names
#: and are matched as whole words, since saying that an option "moves you" is what the rule
#: sentence is for.
SIDE_WORDS = re.compile(r"concede|claim|cooperat|defect|public.goods|contribut|\bstay\b|\bmove\b")


def _neutral(seed: int, *answers: str) -> LLMAgent:
    return LLMAgent("LLM", ScriptedBackend(*answers), profile="neutral-v1", seed=seed)


def _switching_game(rounds: int = 6) -> PublicGoodsGame:
    return PublicGoodsGame(n_agents=4, rounds=rounds, mpcr=0.5, allow_non_dilemma=True,
                           mpcr_schedule=[(3, 1.25)])


def test_the_prompt_names_no_side():
    for game in (_switching_game(), CongestionGame(n_agents=3)):
        agent = _neutral(1, *[_answer("Option A")] * 8)
        run_match(game, [agent] + [AllD(f"d{i}") for i in range(game.n_agents - 1)], rounds=4,
                  seed=0)
        for prompt in agent.backend.prompts:
            assert "Option A" in prompt and "Option B" in prompt
            leaked = SIDE_WORDS.findall(prompt.lower())
            assert not leaked, f"{game.name}: the prompt names a side: {leaked}\n{prompt}"


def test_the_option_order_comes_from_the_seed_and_only_from_it():
    maps = [llm_prompt.neutral_label_map(("concede", "claim"), seed) for seed in range(40)]
    a_is_concede = sum(1 for m in maps if m["concede"] == "Option A")
    assert 10 < a_is_concede < 30, f"Option A meant concede in {a_is_concede} of 40 seeds"
    assert maps[7] == llm_prompt.neutral_label_map(("concede", "claim"), 7)


def test_an_answer_maps_back_to_the_role_it_labelled():
    for seed in range(6):
        agent = _neutral(seed, _answer("Option A"))
        game = _switching_game()
        run_match(game, [agent, AllD("d1"), AllD("d2"), AllD("d3")], rounds=1, seed=0)
        role = {label: r for r, label in agent._labels.items()}["Option A"]
        played = agent.journal()[0]
        assert played["role"] == role and played["labels"] == agent._labels
        assert agent._log[0]["action"] == game.action_for(role)


def test_a_changed_return_is_announced_on_the_round_it_changes():
    agent = _neutral(2, *[_answer("Option B")] * 6)
    run_match(_switching_game(), [agent, AllD("d1"), AllD("d2"), AllD("d3")], rounds=5, seed=0)
    prompts = agent.backend.prompts
    notice = "From this round on the rule has changed."
    assert "multiplied by 2 " in prompts[1] and notice not in prompts[1]
    assert notice in prompts[2] and "multiplied by 5 " in prompts[2]
    assert notice not in prompts[3] and "multiplied by 5 " in prompts[3]


def test_the_rule_sentence_gives_the_share_the_game_pays():
    """Read literally, a pool multiplied by r and split n ways pays mpcr per unit put in, which is
    what the game pays."""
    game = PublicGoodsGame(n_agents=5, rounds=3, mpcr=0.4)
    sentence = llm_prompt.neutral_rules(game, {"concede": "Option A", "claim": "Option B"})[0][1]
    r = float(re.search(r"multiplied by ([0-9.]+)", sentence).group(1))
    n = int(re.search(r"among all (\d+) players", sentence).group(1))
    assert abs(r / n - game.mpcr) < 1e-12


def test_the_congestion_prompt_says_what_each_option_does():
    agent = _neutral(3, *[_answer("Option A")] * 3)
    game = CongestionGame(n_agents=3)
    run_match(game, [agent, AllD("d1"), AllD("d2")], rounds=2, seed=0)
    first = agent.backend.prompts[0]
    moves = agent._labels["claim"]
    assert f"{moves} moves you one cell forward" in first
    assert "The position is" in first
    assert "your cell is digit" not in first, "without a seat the prompt cannot say which digit"
    seated = LLMAgent("LLM", ScriptedBackend(_answer("Option A")), profile="neutral-v1", seed=3,
                      seat=2)
    run_match(CongestionGame(n_agents=3), [AllD("d0"), AllD("d1"), seated], rounds=1, seed=0)
    assert "You are player 3, so your cell is digit 3 from the left." in seated.backend.prompts[0]


def test_the_temperature_is_part_of_the_design():
    cold = LLMAgent("a", OllamaBackend(temperature=0.0), profile="neutral-v1")
    warm = LLMAgent("a", OllamaBackend(temperature=0.7), profile="neutral-v1")
    assert _agent_params(warm)["temperature"] == 0.7
    game = PublicGoodsGame(n_agents=2, rounds=3)
    hashes = {hash_config(build_config(_config_game_desc(game), _roster_description([a]),
                                       DEFAULT_CODE_VERSION, {})) for a in (cold, warm)}
    assert len(hashes) == 2, "two temperatures must be two designs"
    stub = LLMAgent("s", ScriptedBackend(), profile="neutral-v1")
    assert "temperature" not in _agent_params(stub), "a backend with none records nothing"


class _FakeServer:
    """Stands in for urlopen: answers every generate request with a valid decision, and keeps the
    request bodies."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def __call__(self, request, timeout=None):
        body = json.loads(request.data.decode("utf-8"))
        self.bodies.append(body)
        reply = {"response": _answer("Option A"), "prompt_eval_count": 10, "eval_count": 5}
        return _Reply(json.dumps(reply).encode("utf-8"))


class _Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _seeds_sent(seed):
    server = _FakeServer()
    original = urllib.request.urlopen
    urllib.request.urlopen = server
    try:
        agent = LLMAgent("LLM", OllamaBackend(temperature=0.7, seed=seed), profile="neutral-v1",
                         seed=0)
        run_match(_switching_game(), [agent, AllD("d1"), AllD("d2"), AllD("d3")], rounds=4,
                  seed=0)
    finally:
        urllib.request.urlopen = original
    return [b["options"].get("seed") for b in server.bodies], agent


def test_every_request_carries_a_recorded_seed():
    seeds, agent = _seeds_sent(11)
    assert None not in seeds and len(set(seeds)) == len(seeds), seeds
    assert [e["sampling_seeds"][0] for e in agent.journal()] == seeds
    again, _ = _seeds_sent(11)
    other, _ = _seeds_sent(12)
    assert again == seeds and other != seeds
    unseeded, plain = _seeds_sent(None)
    assert unseeded == [None] * len(unseeded)
    assert all("sampling_seeds" not in e for e in plain.journal())


if __name__ == "__main__":
    test_the_prompt_names_no_side()
    print("OK: the neutral prompt names no side, in either game")
    test_the_option_order_comes_from_the_seed_and_only_from_it()
    print("OK: the option order comes from the seed and only from it")
    test_an_answer_maps_back_to_the_role_it_labelled()
    print("OK: an answer maps back to the role its label stood for")
    test_a_changed_return_is_announced_on_the_round_it_changes()
    print("OK: a changed return is announced on the round it changes")
    test_the_rule_sentence_gives_the_share_the_game_pays()
    print("OK: the rule sentence, read literally, gives the share the game pays")
    test_the_congestion_prompt_says_what_each_option_does()
    print("OK: in the congestion family the prompt says what each option does")
    test_the_temperature_is_part_of_the_design()
    print("OK: the sampling temperature is part of the recorded design")
    test_every_request_carries_a_recorded_seed()
    print("OK: every request carries its own seed, recorded, and the same base repeats them")
