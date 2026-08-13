"""The declaration mechanism is enforced, not advised.

`tests/test_action_roles.py` already checks that the mechanism works when an agent uses it. These
cases check the other half, which is the half that failed in practice: what happens when an agent
does not. Every one of them asserts a *refusal*, and a refusal that stops the match rather than
logging a warning, because a misbound match produces numbers indistinguishable from valid ones.

Run: python -m gamebrains.tests.test_semantics
"""

from __future__ import annotations

from ..agents.classic import AllC, MajorityTFT
from ..agents.qlearning import QLearningAgent
from ..engine.agent import Agent
from ..engine.runner import run_match
from ..engine.semantics import SemanticBindingError, bind_agents
from ..games.congestion import CongestionGame
from ..games.public_goods import PublicGoodsGame


def _pgg(n: int = 3) -> PublicGoodsGame:
    return PublicGoodsGame(n_agents=n, rounds=20)


class _Undeclared(Agent):
    """An agent written without knowing the contract exists. The common case, and the one the
    platform used to accept silently."""

    kind = "undeclared"

    def __init__(self, name: str = "undeclared") -> None:
        self.name = name

    def act(self, observation: int) -> int:
        return 1


class _LyingRoleBound(Agent):
    """Declares that its behaviour depends on meaning, then hardcodes an index anyway.

    This is the original defect reproduced exactly: `1` is the conceding action in the Public
    Goods Game and the claiming one in a congestion game, so this agent is cooperative in one and
    maximally aggressive in the other while calling itself the same thing in both.
    """

    kind = "lying"
    semantics = "role-bound"

    def __init__(self, name: str = "lying") -> None:
        self.name = name

    def act(self, observation: int) -> int:
        return 1


def test_undeclared_agent_is_refused() -> None:
    game = _pgg()
    try:
        bind_agents(game, [_Undeclared()])
    except SemanticBindingError as exc:
        assert "does not declare" in str(exc)
        assert "index-agnostic" in str(exc) and "role-bound" in str(exc)
        print("PASS undeclared agent refused, with both options named")
        return
    raise AssertionError("an agent with no `semantics` declaration was accepted")


def test_role_bound_agent_that_never_asks_is_refused() -> None:
    """The headline case. The agent declares dependence on meaning and then ignores the game."""
    game = _pgg()
    try:
        bind_agents(game, [_LyingRoleBound()])
    except SemanticBindingError as exc:
        assert "asked" in str(exc)
        print("PASS role-bound agent that resolved nothing was refused")
        return
    raise AssertionError("a role-bound agent that never queried the game was accepted")


def test_refusal_happens_before_any_round_is_played() -> None:
    """Refusing at binding time and refusing after 10,000 rounds are not the same guarantee."""
    game = _pgg()
    try:
        run_match(game, [_LyingRoleBound(f"lying{i}") for i in range(3)], rounds=50, seed=0)
    except SemanticBindingError:
        # The game must be untouched: reset() has not run, so no observation was ever handed out.
        assert not hasattr(game, "round") or getattr(game, "round", 0) == 0
        print("PASS run_match refused the roster before playing a round")
        return
    raise AssertionError("run_match played a match with a misbound roster")


def test_declared_agents_still_play() -> None:
    """The guard must not be satisfiable by refusing everything."""
    game = _pgg()
    roster = [AllC("allc"), MajorityTFT("tft", n_agents=3), QLearningAgent("q", game.n_states, game.n_actions, seed=1)]
    result = run_match(game, roster, rounds=50, seed=0)
    assert result["actions"].shape == (50, 3)
    print("PASS a correctly declared mixed roster plays as before")


def test_index_agnostic_agent_needs_no_binding() -> None:
    """A learner that assigns no meaning to indices is accepted without asking anything, in both
    games. If this failed, the contract would be taxing agents it has nothing to say about."""
    for game in (_pgg(), CongestionGame(n_agents=3)):
        q = QLearningAgent("q", game.n_states, game.n_actions, seed=1)
        bind_agents(game, [q])
    print("PASS index-agnostic learner accepted in both game families, no queries required")


def test_wrappers_inherit_the_declaration() -> None:
    """A wrapper that answered for itself would launder a role-bound agent into an accepted one."""
    from ..experiments.run_bakeoff import Frozen

    inner = AllC("allc")
    wrapped = Frozen(inner, "frozen-allc")
    assert wrapped.semantics == inner.semantics == "role-bound"

    # And it must still bind through the wrapper, not merely claim to.
    game = _pgg()
    bind_agents(game, [wrapped])
    assert inner._action == game.action_for("concede")
    print("PASS bake-off wrapper inherits the declaration and forwards the binding")


def test_adapter_enforces_the_same_contract() -> None:
    """The runner is not the only entry point. A guarantee that depends on which door you came
    through is not a guarantee."""
    try:
        from ..interop.pettingzoo_adapter import GameBrainsParallelEnv
    except ImportError:
        print("SKIP pettingzoo not installed")
        return

    game = _pgg()
    env = GameBrainsParallelEnv(game, [None, _LyingRoleBound("l1"), _LyingRoleBound("l2")],
                                controlled_agent_ids=[0])
    try:
        env.reset(seed=0)
    except SemanticBindingError:
        print("PASS PettingZoo adapter refused a misbound background roster")
        return
    raise AssertionError("the adapter accepted background agents the runner would refuse")


if __name__ == "__main__":
    test_undeclared_agent_is_refused()
    test_role_bound_agent_that_never_asks_is_refused()
    test_refusal_happens_before_any_round_is_played()
    test_declared_agents_still_play()
    test_index_agnostic_agent_needs_no_binding()
    test_wrappers_inherit_the_declaration()
    test_adapter_enforces_the_same_contract()
    print("\nall semantic-enforcement tests passed")
