"""Lesion tests: the internal state an agent displays must be what produces its behaviour.

A transparency panel shows a belief, a memory or a value table, and invites the reader to explain
the agent's moves by it. That reading is only licensed if the state actually mediates the moves.
These tests replace or remove the state and require the behaviour to change, in the spirit of the
perturbation tests of Verschure (2026, Phil. Trans. R. Soc. B 381:20250156): behaviour that
survives the removal of the mediator it is attributed to is mimicry of that mediator.

The FEP agent failed this for as long as its default was `reciprocity=0`. Its two expected values
then differ by `mpcr*cost - cost` whatever the posterior says, so it conceded with probability
0.119 in every round while its panel showed a belief being updated. An external review found it
from the arithmetic. The checks below would have found it from the behaviour.

Run:  "$GB/gamebrains/.venv/Scripts/python.exe" -m gamebrains.tests.test_ablation
"""

from __future__ import annotations

import sys

import numpy as np

from ..agents.classic import AllC, AllD
from ..agents.fep import _I_CONCEDE, FEPAgent
from ..engine.runner import run_match
from ..games.public_goods import PublicGoodsGame

N = 4


def _concede_probability(agent: FEPAgent, game: PublicGoodsGame, believed_conceders: int) -> float:
    """P(concede) with the posterior replaced by certainty that `believed_conceders` others concede.

    Acting on the start observation skips perception, so the replaced posterior is exactly the one
    the decision uses."""
    agent.qs = np.zeros(agent.n_states)
    agent.qs[believed_conceders] = 1.0
    agent.act(game.start_state)
    return float(agent.last_action_probs[_I_CONCEDE])


def _bound_fep(**kwargs) -> tuple[FEPAgent, PublicGoodsGame]:
    game = PublicGoodsGame(n_agents=N)
    agent = FEPAgent("F", n_agents=N, mpcr=game.mpcr, seed=0, start_state=game.start_state,
                     **kwargs)
    agent.on_match_start(game)
    return agent, game


def test_the_fep_belief_reaches_the_action() -> None:
    agent, game = _bound_fep()
    none = _concede_probability(agent, game, 0)
    every = _concede_probability(agent, game, agent.n_others)
    assert every - none >= 0.5, (
        f"replacing the belief 'nobody else concedes' with 'everybody else does' moved P(concede) "
        f"from {none:.3f} to {every:.3f}; a belief that barely changes the move is not what the "
        f"move is made from")
    print(f"OK: at the defaults the belief drives the move, P(concede) {none:.3f} -> {every:.3f}")


def test_the_lesion_detects_an_inert_belief() -> None:
    """Guards the guard, and records the old default: at reciprocity 0 the belief is inert, and
    the check above must be able to tell."""
    agent, game = _bound_fep(reciprocity=0.0)
    none = _concede_probability(agent, game, 0)
    every = _concede_probability(agent, game, agent.n_others)
    assert abs(every - none) < 1e-12, (
        "at reciprocity 0 the belief now moves the action, so this pair no longer documents the "
        "case the lesion test exists for")
    expected = 1.0 / (1.0 + np.exp(agent.precision * agent.cost * (1.0 - agent.mpcr)))
    assert abs(none - expected) < 1e-12, f"{none} is not the closed-form {expected}"
    print(f"OK: at reciprocity 0 the move is a fixed coin of {none:.3f}, and the lesion sees it")


def _fep_rate_among(opponent: type) -> float:
    game = PublicGoodsGame(n_agents=N)
    fep = FEPAgent("F", n_agents=N, mpcr=game.mpcr, seed=3, start_state=game.start_state)
    roster = [fep] + [opponent(f"O{i}") for i in range(N - 1)]
    records = run_match(game, roster, rounds=300, seed=0)
    return float(np.asarray(records["cooperated"])[:, 0].mean())


def test_the_fep_agent_conditions_on_its_company() -> None:
    """The same agent, unchanged, among cooperators and among defectors. A belief-driven
    conditional cooperator must behave differently in the two; a fixed coin cannot."""
    among_cooperators = _fep_rate_among(AllC)
    among_defectors = _fep_rate_among(AllD)
    assert among_cooperators > 0.7 and among_defectors < 0.2, (
        f"cooperated {among_cooperators:.3f} among cooperators and {among_defectors:.3f} among "
        f"defectors")
    print(f"OK: the FEP agent cooperates {among_cooperators:.3f} among cooperators and "
          f"{among_defectors:.3f} among defectors")


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
    print(f"\n{'all ablation tests passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
