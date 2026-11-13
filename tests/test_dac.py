"""The DAC-style agent: each layer does what its description says, and the agent is one agent in
both game families.

The behavioural claims about the layers taken together (what removing one changes) live in
`test_ablation.py`; these check each mechanism on its own, with the state set by hand so that the
expected outcome follows from the description rather than from a run.

Run:  "$GB/gamebrains/.venv/Scripts/python.exe" -m gamebrains.tests.test_dac
"""

from __future__ import annotations

import json
import sys

import numpy as np

from ..agents.dac import _I_CLAIM, _I_CONCEDE, DACAgent, _Sequence
from ..engine.game import Game, StepResult
from ..engine.runner import run_match
from ..games.congestion import CongestionGame, from_preset
from ..games.public_goods import PublicGoodsGame
from ..repository.record import _roster_description


class _Roleless(Game):
    name = "roleless"
    n_agents = 2
    n_actions = 2
    action_names = ["A", "B"]
    n_states = 1

    def reset(self) -> list[int]:
        return [0, 0]

    def step(self, actions: list[int]) -> StepResult:
        return StepResult(observations=[0, 0], rewards=[0.0, 0.0], done=False,
                          info={"cooperators": 0})


def _bound(game: Game | None = None, **kwargs) -> DACAgent:
    agent = DACAgent("dac", seed=0, **kwargs)
    agent.on_match_start(game or PublicGoodsGame(n_agents=3))
    return agent


def test_it_binds_to_each_game_by_role_and_refuses_a_game_without_roles() -> None:
    pgg, hjg = PublicGoodsGame(n_agents=3), CongestionGame(n_agents=3)
    for game in (pgg, hjg):
        agent = _bound(game)
        assert agent._concede == game.action_for("concede")
        assert agent._claim == game.action_for("claim")
    assert _bound(pgg)._concede != _bound(hjg)._concede, (
        "the two games number conceding the same way, so this checks nothing")
    try:
        _bound(_Roleless())
    except ValueError:
        print("OK: binds by role in both families, opposite indices, and refuses a roleless game")
        return
    raise AssertionError("a game declaring no roles was accepted")


def test_one_unchanged_agent_plays_both_families() -> None:
    for game, rounds in ((PublicGoodsGame(n_agents=3), 200), (from_preset("hjg", 3), 300)):
        roster = [DACAgent(f"D{i}", seed=i) for i in range(3)]
        records = run_match(game, roster, rounds=rounds, seed=0)
        played = set(np.asarray(records["actions"]).flatten().tolist())
        allowed = {game.action_for("concede"), game.action_for("claim")}
        assert played <= allowed, f"{game.name}: played {played}, roles are {allowed}"
        for agent in roster:
            assert sum(agent.layer_counts.values()) == rounds, agent.layer_counts
    print("OK: the same agent plays the Public Goods Game and the Honey-Jar Game")


def _trusted_memory_of_conceding(agent: DACAgent, obs: int) -> None:
    agent._ltm = [_Sequence(uid=0, couplets=[(obs, _I_CONCEDE)], value=5.0)]
    agent.trust = 1.0


def test_urgency_overrides_every_other_layer() -> None:
    agent = _bound()
    _trusted_memory_of_conceding(agent, obs=2)
    agent.deficit = 1.0
    for _ in range(20):
        assert agent.act(2) == agent._claim and agent.last_layer == "reactive-urgent"
    agent.deficit = 0.0
    assert agent.act(2) == agent._concede and agent.last_layer == "contextual", (
        "with the deficit relieved the trusted memory should decide again")
    print("OK: an urgent deficit claims over a trusted memory, and the memory decides once it is not")


def test_after_a_fruitless_contest_the_agent_withdraws() -> None:
    game = PublicGoodsGame(n_agents=3)
    conceded = 0
    for seed in range(200):
        agent = DACAgent("dac", seed=seed, layers=("reactive",))
        agent.on_match_start(game)
        agent.deficit = 0.5
        agent.act(0)
        agent.update(0, agent._claim, 0.0, 0, True)       # claimed, gained nothing, contest over
        if agent.act(0) == agent._concede:
            conceded += 1
    # The reflex concedes with probability `withdraw` (0.8); otherwise the fallback reflex claims
    # with probability equal to the deficit, 0.65 after the loss, so about 0.87 in all.
    rate = conceded / 200
    assert 0.8 <= rate <= 0.98, f"conceded {rate:.2f} of the time after a fruitless contest"
    print(f"OK: after claiming and gaining nothing the agent concedes {rate:.2f} of the time")


def test_memory_is_consolidated_only_after_a_gain_above_baseline() -> None:
    agent = _bound(layers=("contextual",))
    agent.baseline = 1.0
    agent.act(3)
    agent.update(3, agent._concede, 0.5, 3, False)      # below the baseline
    assert agent._ltm == [], "a gain below the agent's baseline was stored as a success"
    before = agent.baseline
    agent.act(3)
    agent.update(3, agent._concede, 4.0, 3, False)
    assert len(agent._ltm) == 1, "a gain above the baseline was not stored"
    assert abs(agent._ltm[0].value - (4.0 - before)) < 1e-12, "stored value is not the excess"
    assert agent._ltm[0].couplets[-1] == (3, _I_CONCEDE)
    print("OK: memory is stored only after a gain above baseline, valued at the excess")


def test_trust_is_lost_faster_than_it_is_gained() -> None:
    agent = _bound()
    _trusted_memory_of_conceding(agent, obs=2)
    agent.trust, agent.trust_threshold = 0.5, 2.0       # tested but never in control
    agent.act(2)
    agent.update(2, agent._concede, 1.0, 2, False)      # positive TD error from a zero table
    gained = agent.trust - 0.5
    assert abs(gained - agent.trust_up * 0.5) < 1e-12, gained
    agent.trust = 0.5
    agent.act(2)
    agent.update(2, agent._concede, -5.0, 2, False)     # negative TD error
    lost = 0.5 - agent.trust
    assert abs(lost - agent.trust_down * 0.5) < 1e-12, lost
    assert lost > gained, f"trust rose by {gained} and fell by {lost} at the same level"
    print(f"OK: at trust 0.5 a confirmation adds {gained:.3f} and a violation removes {lost:.3f}")


def test_same_seed_same_play_and_different_seeds_differ() -> None:
    def play(seed: int) -> list:
        game = from_preset("hjg", 3)
        roster = [DACAgent(f"D{i}", seed=seed + i) for i in range(3)]
        return np.asarray(run_match(game, roster, rounds=400, seed=0)["actions"]).tolist()
    assert play(1) == play(1), "the same seeds produced different play"
    assert play(1) != play(50), "different seeds produced identical play, so seeding does nothing"
    print("OK: byte-identical play under one seed, different play under another")


def test_the_record_carries_the_layers_and_the_hyperparameters() -> None:
    full = _roster_description([DACAgent("a", seed=0)])[0]
    lesioned = _roster_description([DACAgent("b", seed=0, layers=("adaptive", "reactive"))])[0]
    assert full["information"] == "observation+memory"
    assert full["params"]["layers"] == ["reactive", "adaptive", "contextual"] or \
        tuple(full["params"]["layers"]) == ("reactive", "adaptive", "contextual")
    assert full != lesioned, "a lesioned agent is recorded as the full one"
    assert tuple(lesioned["params"]["layers"]) == ("reactive", "adaptive"), (
        "the layer order should be canonical, so one lesion is one design")
    print("OK: layers and hyperparameters reach the record, and a lesion is a different design")


def test_the_brain_payload_serialises_and_names_a_real_layer() -> None:
    game = from_preset("hjg", 3)
    roster = [DACAgent(f"D{i}", seed=i) for i in range(3)]
    run_match(game, roster, rounds=120, seed=0)
    payload = roster[0].render_brain()
    json.dumps(payload)
    assert payload["layer"] in roster[0].layer_counts
    for key in ("deficit", "trust", "ltm_size", "q_values", "contextual_vote", "role_labels"):
        assert key in payload, key
    print("OK: the brain payload serialises and reports the layer in control")


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
    print(f"\n{'all DAC tests passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
