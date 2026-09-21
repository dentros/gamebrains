"""What the model is told, pinned down as a recorded choice rather than a string in a function.

The prompt is the LLM seat's information set, so these tests guard two different things. One is
arithmetic: the first bake-off's model read an observation of 1 out of 5 as evidence that the
others had cooperated, because the published count includes the agent itself, and `informed-v1`
exists to do that subtraction where it can be checked. The other is bookkeeping: the profile name
has to reach `config_hash`, or two runs told different things would be offered to a reader as one
another's answer, which is the failure the whole repository is built to prevent.

`minimal-v1` is also frozen here on purpose. Every measurement in the paper was taken under it,
and a later edit to make it "better" would silently redefine results already recorded.

Run: python -m gamebrains.tests.test_llm_prompt
"""

from __future__ import annotations

from typing import Any

from ..agents import llm_prompt
from ..agents.llm import LLMAgent
from ..agents.llm_backends import Capabilities
from ..games.congestion import from_preset
from ..games.public_goods import PublicGoodsGame
from ..repository.record import _agent_params, _roster_description


class SilentBackend:
    """Never asked to answer anything: these tests read prompts, they do not play."""

    def capabilities(self) -> Capabilities:
        return Capabilities(name="scripted", model="scripted-1b", native_schema=False,
                            activations=False, deterministic=False, note="stub")

    def complete(self, prompt: str, schema: dict, max_attempts: int = 1) -> Any:
        raise AssertionError("no decision should be requested in a prompt test")


def _pgg(profile: str, n_agents: int = 5) -> LLMAgent:
    agent = LLMAgent("LLM", SilentBackend(), profile=profile)
    agent.on_match_start(PublicGoodsGame(n_agents=n_agents, rounds=40, mpcr=0.4))
    return agent


def test_an_unknown_profile_fails_when_the_roster_is_built() -> None:
    """Not forty rounds in, and not by quietly falling back to the default."""
    try:
        LLMAgent("LLM", SilentBackend(), profile="chain-of-thought-v9")
    except ValueError as exc:
        assert "minimal-v1" in str(exc) and "informed-v1" in str(exc), str(exc)
    else:
        raise AssertionError("an unknown profile was accepted")
    print("OK: an unknown profile is refused, and the message lists the ones that exist")


def test_minimal_v1_still_says_exactly_what_it_said(  ) -> None:
    """The profile the paper's numbers were measured under, held still."""
    agent = _pgg("minimal-v1")
    agent._log = [{"role": "concede", "reward": 0.4}]
    agent._earned = 0.4
    prompt = agent.build_prompt(1)

    assert "how many players took the conceding action" in prompt
    assert "you played concede and received 0.40" in prompt
    for absent in ("This is round", "Your total so far", "not counting you",
                   "multiplied by", "The position is"):
        assert absent not in prompt, f"minimal-v1 has grown a line: {absent!r}"
    assert agent.information == "observation+history"
    print("OK: minimal-v1 is unchanged, so results recorded under it still mean what they meant")


def test_informed_v1_takes_the_agent_out_of_the_count_it_reports() -> None:
    """The off-by-one the profile exists to remove, in both directions."""
    agent = _pgg("informed-v1")

    agent._log = [{"role": "concede", "reward": 0.4}]
    assert "In the previous round 2 of the 4 other players conceded" in agent.build_prompt(3)

    agent._log = [{"role": "claim", "reward": 1.4}]
    assert "In the previous round 3 of the 4 other players conceded" in agent.build_prompt(3)
    print("OK: the count the model is given excludes the model's own move")


def test_nothing_is_claimed_about_a_round_that_has_not_happened() -> None:
    """The start marker is n_agents + 1, and no subtraction on it would mean anything."""
    agent = _pgg("informed-v1")
    assert "not counting you" not in agent.build_prompt(6)
    assert llm_prompt.others_conceded(6, 5, None) is None
    print("OK: before the first round the model is told nothing about a previous round")


def test_a_count_is_not_dressed_up_as_a_decoded_position() -> None:
    """The Public Goods Game renders state 1 as `k=1`, which says less than the sentence."""
    prompt = _pgg("informed-v1").build_prompt(1)
    assert "The position is" not in prompt and "k=1" not in prompt
    print("OK: a game that publishes a count is not decoded into a shorter label")


def test_a_decoded_board_arrives_with_the_key_to_reading_it() -> None:
    """A label without its convention is a number with more digits."""
    agent = LLMAgent("LLM", SilentBackend(), profile="informed-v1")
    game = from_preset("hjg", 3)
    agent.on_match_start(game)
    prompt = agent.build_prompt(1)

    assert "The position is 001." in prompt
    assert "in seat order" in prompt
    assert llm_prompt.state_legend(game) in prompt
    assert "an encoded index" not in prompt
    print("OK: a decoded position is followed by the game's own account of how to read it")


def test_an_enormous_state_space_is_not_enumerated_into_the_prompt() -> None:
    """The decode is skipped rather than held, and the gloss takes over unremarked."""
    agent = LLMAgent("LLM", SilentBackend(), profile="informed-v1")
    agent.on_match_start(from_preset("hjg", 9))
    assert agent._state_labels == []
    assert "an encoded index" in agent.build_prompt(1)
    print("OK: a state space too large to enumerate falls back to the gloss")


def test_the_profile_is_recorded_so_two_profiles_are_two_experiments() -> None:
    """The point of the whole module: the choice reaches config_hash and the filters."""
    minimal, informed = _pgg("minimal-v1"), _pgg("informed-v1")

    params = _agent_params(minimal)
    assert params["profile"] == "minimal-v1"
    assert _agent_params(informed)["profile"] == "informed-v1"

    assert _roster_description([minimal]) != _roster_description([informed])
    assert minimal.information != informed.information
    print("OK: the profile name is part of what identifies the experiment")


def test_a_profile_cannot_be_changed_out_from_under_a_recorded_run() -> None:
    """Frozen dataclass, read-only property: the recorded name stays true."""
    agent = _pgg("minimal-v1")
    for attempt in (lambda: setattr(agent, "profile", "informed-v1"),
                    lambda: setattr(agent._profile, "history", 99)):
        try:
            attempt()
        except (AttributeError, TypeError):
            continue
        raise AssertionError("a recorded profile was edited at runtime")
    print("OK: neither the name nor its contents can be swapped mid-run")


if __name__ == "__main__":
    test_an_unknown_profile_fails_when_the_roster_is_built()
    test_minimal_v1_still_says_exactly_what_it_said()
    test_informed_v1_takes_the_agent_out_of_the_count_it_reports()
    test_nothing_is_claimed_about_a_round_that_has_not_happened()
    test_a_count_is_not_dressed_up_as_a_decoded_position()
    test_a_decoded_board_arrives_with_the_key_to_reading_it()
    test_an_enormous_state_space_is_not_enumerated_into_the_prompt()
    test_the_profile_is_recorded_so_two_profiles_are_two_experiments()
    test_a_profile_cannot_be_changed_out_from_under_a_recorded_run()
    print("\nall prompt profile tests passed")
