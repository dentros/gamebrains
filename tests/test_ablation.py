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


# --- The DAC-style agent: what removing a layer changes ------------------------------------------


def _dac_pgg(**kwargs) -> tuple[np.ndarray, list]:
    from ..agents.dac import DACAgent
    game = PublicGoodsGame(n_agents=N)
    roster = [DACAgent(f"D{i}", seed=i, **kwargs) for i in range(N)]
    return np.asarray(run_match(game, roster, rounds=400, seed=0)["actions"]), roster


def test_the_contextual_layer_changes_what_the_dac_agent_does() -> None:
    full, roster = _dac_pgg()
    lesioned, _ = _dac_pgg(layers=("reactive", "adaptive"))
    share = roster[0].layer_counts["contextual"] / 400
    assert share > 0.5, f"the contextual layer decided only {share:.2f} of the rounds"
    assert not np.array_equal(full, lesioned), (
        "removing a layer that decided most rounds left every move unchanged, so the moves were "
        "not made by it")
    print(f"OK: the contextual layer decides {share:.2f} of rounds, and removing it changes the play")


def test_a_layer_that_never_decides_can_be_removed_without_trace() -> None:
    """Guards the guard: if the contextual layer never takes control, removing it must change
    nothing at all, so the difference above is the layer acting and not a side effect of removing
    its bookkeeping."""
    never, roster = _dac_pgg(trust_threshold=2.0)
    lesioned, _ = _dac_pgg(layers=("reactive", "adaptive"), trust_threshold=2.0)
    assert roster[0].layer_counts["contextual"] == 0
    assert np.array_equal(never, lesioned), (
        "a contextual layer that never decided still changed the play when removed")
    print("OK: a contextual layer that never decides leaves the play identical when removed")


def test_turn_taking_in_the_honey_jar_game_comes_from_the_reactive_layer() -> None:
    """The finding the paper reports about this agent, pinned so that a change of defaults has to
    face it. With all three layers the agent's coordination collapses as a tabular learner's does;
    with the reflexes alone, a deficit that rises with every contest lost and a withdrawal after
    every fruitless one, three agents take turns. This runs against Freire et al. (2020), where in
    a two-agent continuous version of the game it was the complete reactive plus adaptive agent
    that formed turn-taking conventions and each layer alone was less efficient. The difference
    between the two settings is a question this result raises, and it is not a replication."""
    from ..agents.dac import DACAgent
    from ..games.congestion import from_preset
    from ..metrics import social_alt

    def calt(**kwargs) -> float:
        game = from_preset("hjg", 3)
        roster = [DACAgent(f"D{i}", seed=i, **kwargs) for i in range(3)]
        records = run_match(game, roster, rounds=3000, seed=0)
        return social_alt.compute_all(records["episodes"], n_agents=3,
                                      full_reward=game.full_reward)["CALT"]

    full, reflexes = calt(), calt(layers=("reactive",))
    assert reflexes > 0.5 and full < 0.3, (
        f"CALT {full:.3f} with all layers and {reflexes:.3f} with the reflexes alone")
    print(f"OK: CALT {full:.3f} with all three layers, {reflexes:.3f} with the reactive layer alone")


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
