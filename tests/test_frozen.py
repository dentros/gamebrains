"""A frozen agent must be recorded as the agent it is, not as a generic wrapper.

The bake-off wrapper inherited `semantics` and nothing else, so the record of every bake-off said
that a frozen FEP agent and a frozen Markov brain conditioned on the current observation alone,
that a frozen AllD looked at anything at all, and, because the recorder reads hyperparameters off
the live agent, that every frozen learner had none. Two frozen Q-learners trained at different
learning rates therefore shared one `config_hash`. Play was right throughout; the description of
what played was not.

The checks compare a frozen agent's roster description against the unwrapped agent's, field by
field, so they hold for any kind added later without being edited.

Run:  "$GB/gamebrains/.venv/Scripts/python.exe" -m gamebrains.tests.test_frozen
"""

from __future__ import annotations

import sys

from ..agents.classic import AllD
from ..agents.fep import FEPAgent
from ..agents.frozen import Frozen
from ..agents.markov_brain import MarkovBrainAgent
from ..agents.qlearning import QLearningAgent
from ..engine.agent import Agent
from ..games.public_goods import PublicGoodsGame
from ..repository.record import _roster_description, reproducibility_of


def _zoo() -> list[Agent]:
    game = PublicGoodsGame(n_agents=3)
    return [
        AllD("d"),
        FEPAgent("f", n_agents=3, mpcr=game.mpcr, seed=1),
        MarkovBrainAgent("m", n_states=game.n_states, n_actions=game.n_actions, seed=1),
        QLearningAgent("q", game.n_states, game.n_actions, seed=1, alpha=0.3),
    ]


def test_a_frozen_agent_declares_what_the_agent_declares() -> None:
    for agent in _zoo():
        frozen = Frozen(agent, f"{agent.name} (frozen)")
        assert frozen.semantics == agent.semantics, agent.kind
        assert frozen.information == agent.information, (
            f"{agent.kind}: the wrapper declares {frozen.information!r} and the agent "
            f"{agent.information!r}, so the record would say the agent saw something it did not")
    print("OK: semantics and information match the wrapped agent for four kinds")


def test_the_record_describes_a_frozen_agent_as_the_agent() -> None:
    """Everything but `training_mode`, which is genuinely `fixed` for a frozen learner."""
    for agent in _zoo():
        plain = _roster_description([agent])[0]
        frozen = _roster_description([Frozen(agent, "x")])[0]
        assert frozen["training_mode"] == "fixed"
        for field in ("kind", "information", "params"):
            assert frozen[field] == plain[field], (
                f"{agent.kind}: the record of the frozen agent has {field}={frozen[field]!r} "
                f"where the agent has {plain[field]!r}")
    print("OK: kind, information and hyperparameters reach the record through the wrapper")


def test_two_differently_trained_learners_stay_two_designs() -> None:
    game = PublicGoodsGame(n_agents=3)
    slow = Frozen(QLearningAgent("a", game.n_states, game.n_actions, seed=1, alpha=0.05), "a")
    fast = Frozen(QLearningAgent("b", game.n_states, game.n_actions, seed=1, alpha=0.5), "b")
    assert _roster_description([slow]) != _roster_description([fast]), (
        "two frozen Q-learners trained at different learning rates have the same roster "
        "description, so their runs would share a config_hash")
    print("OK: frozen learners trained differently are recorded as different designs")


class _Unrepeatable(AllD):
    def reproducibility(self) -> str:
        return "none"

    def reproducibility_note(self) -> str:
        return "a stand-in for a component that cannot repeat itself"


def test_the_reproducibility_tier_comes_through() -> None:
    tier = reproducibility_of([Frozen(_Unrepeatable("u"), "u")])["tier"]
    assert tier == "none", f"a frozen agent that cannot repeat itself was recorded as {tier!r}"
    print("OK: the wrapped agent's reproducibility tier is the one recorded")


def test_a_frozen_learner_reports_the_epsilon_it_plays_with() -> None:
    game = PublicGoodsGame(n_agents=3)
    q = QLearningAgent("q", game.n_states, game.n_actions, seed=1)
    q.epsilon = 0.7
    frozen = Frozen(q, "q")
    assert frozen.epsilon == 0.0
    frozen.act(0)
    assert q.epsilon == 0.7, "acting through the wrapper changed the agent's own epsilon"
    print("OK: a frozen learner reads epsilon 0 and leaves the agent's own value alone")


def test_both_bake_offs_use_this_wrapper() -> None:
    """Two copies of the class is how one was fixed and the other was not."""
    from ..experiments import run_bakeoff
    from ..webui import app
    assert run_bakeoff.Frozen is Frozen and app._Frozen is Frozen
    print("OK: the script and the web interface share one wrapper")


if __name__ == "__main__":
    failures = 0
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {fn_name}\n     {exc}")
            except Exception as exc:
                failures += 1
                print(f"ERROR {fn_name}\n     {type(exc).__name__}: {exc}")
    print(f"\n{'all frozen-wrapper tests passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
