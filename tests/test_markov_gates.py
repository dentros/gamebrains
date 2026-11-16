"""The gate-based Markov brain: each gate computes what its table says, gates combine by OR, the
transition matrix is the brain's own dynamics, and the genome survives the GA.

Most cases set the genome by hand, so the expected outcome follows from the definition in
`agents/markov_gates.py` rather than from a run. The sigmoid brain (`agents/markov_brain.py`) is a
different architecture and is checked here only for being recorded as one.

Run:  "$GB/gamebrains/.venv/Scripts/python.exe" -m gamebrains.tests.test_markov_gates
"""

from __future__ import annotations

import json
import random
import sys

import numpy as np

from ..agents import factory
from ..agents.markov_brain import MarkovBrainAgent
from ..agents.markov_gates import (
    MAX_GATE_INPUTS, MAX_GATE_OUTPUTS, MAX_GATES, MIN_GATES, NO_START_OBSERVATION, Gate,
    MarkovGatesAgent, crossover, spawn,
)
from ..engine.evolution import EvolutionConfig, evolve
from ..engine.runner import run_match
from ..games.congestion import from_preset
from ..games.public_goods import PublicGoodsGame
from ..repository.record import _KIND_ORDER, _feature_vector, _roster_description
from ..repository.smart_filter import FEATURE_WEIGHTS


def _table(columns: list[int], n_out: int) -> np.ndarray:
    """A one-hot table writing output pattern `columns[r]` for input pattern `r`."""
    table = np.zeros((len(columns), 2 ** n_out))
    table[np.arange(len(columns)), columns] = 1.0
    return table


def _brain(gates: list[Gate], n_states: int = 4, n_hidden: int = 1,
           deterministic: bool = True, seed: int = 0) -> MarkovGatesAgent:
    """Nodes for n_states=4, n_hidden=1: s0=0, s1=1, h0=2, m0=3. No start observation, so a
    hand-set state is never cleared by the observation that follows it."""
    return MarkovGatesAgent("g", n_states=n_states, n_actions=2, n_hidden=n_hidden,
                            deterministic=deterministic, gates=gates, seed=seed,
                            start_state=NO_START_OBSERVATION)


def _check_genome(agent: MarkovGatesAgent) -> None:
    """Every validity condition, stated here rather than trusted to the constructor's check."""
    assert MIN_GATES <= agent.n_gates <= MAX_GATES, agent.n_gates
    for gate in agent.gates:
        assert 1 <= gate.n_in <= MAX_GATE_INPUTS and 1 <= gate.n_out <= MAX_GATE_OUTPUTS
        assert len(set(gate.inputs)) == gate.n_in and len(set(gate.outputs)) == gate.n_out
        assert all(0 <= i < agent.n_nodes for i in gate.inputs), gate.inputs
        assert all(agent.n_sensor <= o < agent.n_nodes for o in gate.outputs), gate.outputs
        assert gate.table.shape == (2 ** gate.n_in, 2 ** gate.n_out), gate.table.shape
        assert np.all(gate.table >= 0) and np.allclose(gate.table.sum(axis=1), 1.0)
        if agent.deterministic:
            assert np.all((gate.table == 0) | (gate.table == 1))


def test_deterministic_gates_compute_and_and_xor() -> None:
    # Rows are little-endian over the inputs (s0, s1): (0,0), (1,0), (0,1), (1,1).
    and_gate = Gate((0, 1), (3,), _table([0, 0, 0, 1], 1))     # m0 = s0 AND s1
    xor_gate = Gate((0, 1), (2,), _table([0, 1, 1, 0], 1))     # h0 = s0 XOR s1
    agent = _brain([and_gate, xor_gate])
    for obs in range(4):
        s0, s1 = obs & 1, (obs >> 1) & 1
        assert agent.act(obs) == (s0 & s1), f"AND of {s0},{s1} played {agent.state[3]}"
        assert agent.state[2] == (s0 ^ s1), f"XOR of {s0},{s1} wrote {agent.state[2]}"

    # Every gate reads the state at t, never a node another gate has just written: m0 copies h0,
    # so it must show the XOR of the previous round's sensors, not of this round's.
    delayed = _brain([xor_gate, Gate((2,), (3,), _table([0, 1], 1))])
    previous = None
    for obs in (1, 3, 2, 0, 1, 1, 2):
        action = delayed.act(obs)
        if previous is not None:
            assert action == previous, f"m0 read h0 after it was written: {action} != {previous}"
        previous = (obs & 1) ^ ((obs >> 1) & 1)
    print("OK: AND and XOR gates compute their truth tables, and every gate reads the state at t")


def test_two_gates_writing_one_node_are_ored() -> None:
    copy_s0 = Gate((0,), (3,), _table([0, 1], 1))
    copy_s1 = Gate((1,), (3,), _table([0, 1], 1))
    agent = _brain([copy_s0, copy_s1])
    for obs in range(4):
        s0, s1 = obs & 1, (obs >> 1) & 1
        assert agent.act(obs) == (s0 | s1), f"inputs {s0},{s1}: m0 = {agent.state[3]}, not OR"
    # A node no gate writes starts from zero at t+1, whatever it held at t.
    agent.state[2] = 1
    agent.act(0)
    assert agent.state[2] == 0, "an unwritten hidden node kept its value instead of resetting to 0"
    print("OK: two gates writing m0 combine by OR, and an unwritten node returns to zero")


def test_probabilistic_gate_samples_its_table() -> None:
    # One input (s0), two outputs (h0, m0): column c writes h0 = bit 0, m0 = bit 1.
    row_on = np.array([0.1, 0.2, 0.3, 0.4])
    gate = Gate((0,), (1, 2), np.vstack([[1.0, 0.0, 0.0, 0.0], row_on]))
    agent = MarkovGatesAgent("p", n_states=2, n_actions=2, n_hidden=1, deterministic=False,
                             gates=[gate], seed=7, start_state=NO_START_OBSERVATION)
    n = 20000
    counts = np.zeros(4)
    for _ in range(n):
        agent.act(1)
        counts[agent.state[1] + 2 * agent.state[2]] += 1
    freq = counts / n
    tolerance = 4 * np.sqrt(row_on * (1 - row_on) / n)
    assert np.all(np.abs(freq - row_on) <= tolerance), f"sampled {freq}, table {row_on}"
    for _ in range(200):
        agent.act(0)
        assert agent.state[1] == 0 and agent.state[2] == 0, "a probability-1 row was not obeyed"

    # The per-node TPM keeps each marginal but not the joint: P(h0)=0.6, P(m0)=0.7, and
    # 0.6 * 0.7 = 0.42 is not the table's 0.4, which tpm_is_exact must admit.
    tpm = agent.full_tpm()
    on = 1                                      # s0 = 1, h0 = m0 = 0
    assert np.isclose(tpm[on, 1], 0.6) and np.isclose(tpm[on, 2], 0.7), tpm[on]
    assert not agent.tpm_is_exact(), "a gate drawing correlated outputs was reported as exact"
    print(f"OK: a probabilistic gate samples {np.round(freq, 3)} against {row_on}, and the "
          f"TPM's loss of their correlation is reported")


def test_same_seed_same_play_and_different_seeds_differ() -> None:
    def play(seed: int, deterministic: bool, global_seed: int = 0) -> list:
        game = PublicGoodsGame(n_agents=4)
        roster = [MarkovGatesAgent(f"G{i}", game.n_states, game.n_actions,
                                   deterministic=deterministic, seed=seed + i,
                                   start_state=game.start_state) for i in range(4)]
        np.random.seed(global_seed)                # the global generators must not matter
        random.seed(global_seed)
        return np.asarray(run_match(game, roster, rounds=300, seed=0)["actions"]).tolist()

    for deterministic in (True, False):
        first = play(1, deterministic)
        assert first == play(1, deterministic, global_seed=99), (
            f"deterministic={deterministic}: the same seeds played differently")
        # A deterministic brain's seed reaches play only through its genome, and OR'ed random
        # gates often settle into constant play, so two rosters may coincide. Across four seeds
        # they must not all coincide, or seeding does nothing.
        plays = [first] + [play(s, deterministic) for s in (50, 100, 150)]
        distinct = {json.dumps(p) for p in plays}
        assert len(distinct) > (1 if deterministic else 3), (
            f"deterministic={deterministic}: {len(distinct)} distinct plays from four seeds")
    print("OK: identical play under one seed, different play under others, for both gate kinds")


def test_ga_operators_keep_the_genome_valid() -> None:
    game = PublicGoodsGame(n_agents=4)
    rng = np.random.default_rng(11)
    for deterministic in (True, False):
        agent = MarkovGatesAgent.random("A", game.n_states, game.n_actions, n_hidden=2, n_gates=4,
                                        deterministic=deterministic, seed=3)
        counts, tables = set(), 0
        for step in range(1500):
            child = agent.mutate(rng, rate=0.4, scale=0.5, name="A", seed=step)
            _check_genome(child)
            counts.add(child.n_gates)
            tables += sum(
                1 for g in child.gates for h in agent.gates
                if (g.inputs, g.outputs) == (h.inputs, h.outputs)
                and g.table.shape == h.table.shape and not np.array_equal(g.table, h.table))
            mate = MarkovGatesAgent.random("B", game.n_states, game.n_actions, n_hidden=2,
                                           n_gates=int(rng.integers(1, 9)),
                                           deterministic=deterministic, seed=step + 10_000)
            agent = crossover(child, mate, rng, name="A", seed=step)
            _check_genome(agent)
        assert min(counts) <= 2 and max(counts) >= 8, f"gate counts stayed within {sorted(counts)}"
        assert tables > 0, "no table entry ever changed under point mutation"

        # Rewiring on its own: it moves one input or one output and never touches the table.
        # Nodes s0..s2 = 0..2, h0, h1 = 3, 4, m0 = 5: free nodes on both sides of the gate.
        probe = MarkovGatesAgent("R", game.n_states, game.n_actions, n_hidden=2,
                                 deterministic=deterministic, seed=4,
                                 gates=[Gate((0, 3), (4,), _table([0, 1, 1, 0], 1))])
        gate, before = probe.gates[0], probe.gates[0].copy()
        moved_in = moved_out = False
        for _ in range(300):
            ins, outs = gate.inputs, gate.outputs
            probe._rewire(gate, rng)
            moved_in |= gate.inputs != ins
            moved_out |= gate.outputs != outs
            probe.validate()
        assert moved_in and moved_out, "rewiring never moved an input, or never an output"
        assert np.array_equal(gate.table, before.table), "rewiring changed the table"

    base = MarkovGatesAgent.random("C", game.n_states, game.n_actions, seed=5)
    same = base.mutate(np.random.default_rng(0), rate=0.0, name="C", seed=1)
    assert [(g.inputs, g.outputs) for g in same.gates] == [(g.inputs, g.outputs) for g in base.gates]
    assert all(np.array_equal(a.table, b.table) for a, b in zip(same.gates, base.gates)), (
        "rate 0 changed the genome")
    twice = [base.mutate(np.random.default_rng(9), rate=0.5, name="C", seed=1) for _ in range(2)]
    assert [g.table.tolist() for g in twice[0].gates] == [g.table.tolist() for g in twice[1].gates]
    clone = base.clone(name="D", seed=4)
    assert clone.gates is not base.gates and all(
        np.array_equal(a.table, b.table) and a.table is not b.table
        for a, b in zip(clone.gates, base.gates)), "a clone must copy the genome, not share it"

    other = MarkovGatesAgent.random("E", game.n_states, game.n_actions, n_hidden=3, seed=6)
    try:
        crossover(base, other, rng, name="F", seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError("crossover accepted parents with different node layouts")
    try:
        MarkovGatesAgent("bad", 4, 2, gates=[Gate((0, 9), (3,), _table([0, 0, 0, 1], 1))])
    except ValueError:
        pass
    else:
        raise AssertionError("a gate reading a node the brain does not have was accepted")
    print(f"OK: 1500 mutations and crossovers per gate kind left every genome valid, gate counts "
          f"ranged over {min(counts)}..{max(counts)}")


def test_evolve_runs_on_this_genome_and_reproduces() -> None:
    def game() -> PublicGoodsGame:
        return PublicGoodsGame(n_agents=4, rounds=20)

    spawn_fn, crossover_fn = factory.evolvable("markov_gates")
    assert spawn_fn is spawn and crossover_fn is crossover
    assert factory.EVOLVABLE_KINDS[0] == "markov_brain", "the GA's default genome must stay first"

    config = EvolutionConfig(population_size=8, generations=3, match_rounds=20,
                             matches_per_generation=1, seed=5, name_prefix="MG")
    first = evolve(game, config, spawn=spawn_fn, crossover=crossover_fn)
    second = evolve(game, config, spawn=spawn_fn, crossover=crossover_fn)
    assert len(first.fitness) == 8 and len(first.fitness_history) == 3
    assert isinstance(first.best, MarkovGatesAgent)
    assert np.array_equal(first.fitness, second.fitness), "two runs at one GA seed disagree"

    knobs = EvolutionConfig(population_size=4, generations=1, match_rounds=10, seed=1,
                            genome={"n_gates": 7, "deterministic": False})
    seeded = spawn("MG0", game(), seed=2, config=knobs)
    assert seeded.n_gates == 7 and not seeded.deterministic, "genome knobs did not reach the spawn"
    try:
        spawn("MG0", game(), seed=2, config=EvolutionConfig(genome={"hidden": 64}))
    except ValueError:
        pass
    else:
        raise AssertionError("a genome knob this brain does not have was accepted and ignored")
    print(f"OK: evolve() runs on the gate genome and reproduces (best fitness "
          f"{first.fitness.max():.2f})")


def test_full_tpm_is_the_brain_stepped_from_every_state() -> None:
    for seed in range(6):
        agent = MarkovGatesAgent("T", n_states=4, n_actions=2, n_hidden=2, n_gates=6, seed=seed,
                                 start_state=NO_START_OBSERVATION)
        tpm = agent.full_tpm()
        n, ns = agent.n_nodes, agent.n_sensor
        assert set(np.unique(tpm)) <= {0.0, 1.0}, "a deterministic brain's TPM is not 0/1"
        assert agent.tpm_is_exact()
        for s in range(2 ** n):
            state = [(s >> k) & 1 for k in range(n)]
            agent.state = np.array(state, dtype=int)
            observation = sum(bit << k for k, bit in enumerate(state[:ns]))
            agent.act(observation)
            assert list(tpm[s, :ns]) == state[:ns], "sensor columns are not the identity"
            assert agent.state[ns:].tolist() == tpm[s, ns:].astype(int).tolist(), (
                f"seed {seed}, state {state}: stepping gave {agent.state[ns:].tolist()}, the TPM "
                f"says {tpm[s, ns:].tolist()}")
    print("OK: for six random deterministic brains the TPM equals one step from all 32 states")


def test_one_unchanged_agent_plays_both_families() -> None:
    for game, rounds in ((PublicGoodsGame(n_agents=3), 200), (from_preset("hjg", 3), 300)):
        roster = [factory.build("markov_gates", i, game, seed=0) for i in range(3)]
        assert all(type(a) is MarkovGatesAgent for a in roster)
        records = run_match(game, roster, rounds=rounds, seed=0)
        played = set(np.asarray(records["actions"]).flatten().tolist())
        assert played <= set(range(game.n_actions)), f"{game.name}: played {played}"
    hjg_agent = factory.build("markov_gates", 0, from_preset("hjg", 3), seed=0)
    assert hjg_agent.start_state == NO_START_OBSERVATION, (
        "the congestion family declares no start observation, so no board position may clear "
        "the brain's memory")
    print("OK: the same agent, built by the factory, plays the Public Goods and Honey-Jar Games")


def test_the_record_names_it_as_its_own_kind() -> None:
    game = PublicGoodsGame(n_agents=4)
    gates = factory.build("markov_gates", 0, game, seed=0)
    sigmoid = factory.build("markov_brain", 0, game, seed=0)
    described = _roster_description([gates])[0]
    assert described == {"kind": "markov_gates", "training_mode": "evolutionary",
                         "information": "observation+memory",
                         "params": {"n_hidden": 2, "n_gates": 4, "deterministic": True}}, described
    # The sigmoid brain's description, pinned so this kind's arrival cannot move it.
    assert _roster_description([sigmoid])[0] == {
        "kind": "markov_brain", "training_mode": "evolutionary",
        "information": "observation+memory", "params": {"n_hidden": 2}}
    probabilistic = factory.build("markov_gates", 0, game, seed=0,
                                  overrides={"deterministic": "0"})
    assert _roster_description([probabilistic])[0]["params"]["deterministic"] is False, (
        "the string '0' was read as true")
    assert _roster_description([probabilistic]) != _roster_description([gates])

    vector = _feature_vector(game, [gates, gates, sigmoid], rounds=100)
    assert len(vector) == 4 + len(_KIND_ORDER) == len(FEATURE_WEIGHTS)
    assert vector[4 + _KIND_ORDER.index("markov_gates")] == 2
    assert vector[4 + _KIND_ORDER.index("markov_brain")] == 1
    try:
        factory.build("markov_gates", 0, game, seed=0, overrides={"hidden": 3})
    except ValueError:
        pass
    else:
        raise AssertionError("the factory dropped an override this kind does not take")
    print("OK: recorded as markov_gates with its own parameters, apart from markov_brain")


def test_phi_and_causal_autonomy_on_a_small_deterministic_brain() -> None:
    from ..metrics.phi_autonomy import causal_autonomy, compute_phi

    # Nodes: s0=0, h0=1, m0=2. Each hidden/motor node copies the other: a two-node loop.
    loop = [Gate((2,), (1,), _table([0, 1], 1)), Gate((1,), (2,), _table([0, 1], 1))]
    agent = MarkovGatesAgent("phi", n_states=2, n_actions=2, n_hidden=1, gates=loop, seed=0,
                             start_state=NO_START_OBSERVATION)
    agent.state = np.array([0, 1, 0])
    agent.act(1)                               # reachable: (s0=1, h0=0, m0=1)
    result = compute_phi(agent)
    assert result["complex_nodes"] == [1, 2] and result["phi"] >= 0.0, result

    # Hidden and motor nodes always written 0: one state after the first step, so 0 bits.
    silent = MarkovGatesAgent("zero", n_states=2, n_actions=2, n_hidden=1, seed=0,
                              gates=[Gate((0,), (1, 2), _table([0, 0], 2))],
                              start_state=NO_START_OBSERVATION)
    assert abs(causal_autonomy(silent, n_t=2)) < 1e-12
    # Both nodes copy the sensor: two equally likely states after the last reading, so 1 bit.
    follower = MarkovGatesAgent("copy", n_states=2, n_actions=2, n_hidden=1, seed=0,
                                gates=[Gate((0,), (1, 2), _table([0, 3], 2))],
                                start_state=NO_START_OBSERVATION)
    assert abs(causal_autonomy(follower, n_t=2) - 1.0) < 1e-12
    print(f"OK: Phi = {result['phi']:.4f} on a two-node copy loop; causal autonomy 0 and 1 bit "
          f"where the definition gives exactly those")


def test_the_brain_payload_serialises_and_the_card_renders() -> None:
    from ..webui.app import KIND_META, _analytics_field_options, _markov_gates_html, render_creature

    and_gate = Gate((0, 1), (3,), _table([0, 0, 0, 1], 1))
    agent = _brain([and_gate, Gate((0, 1), (2,), _table([0, 1, 1, 0], 1))])
    agent.act(3)
    payload = agent.render_brain()
    json.dumps(payload)
    json.dumps(agent.inspect())
    assert [g["function"] for g in payload["gates"]] == ["AND", "XOR"], payload["gates"]
    assert payload["gates"][0]["truth_table"][3] == ["11", "1"]
    assert payload["node_labels"] == ["s0", "s1", "h0", "m0"] and payload["state"] == [1, 1, 0, 1]
    assert "tables" in agent.inspect() and "tables" not in payload

    probabilistic = MarkovGatesAgent("p", 4, 2, deterministic=False, seed=3)
    rows = probabilistic.render_brain()["gates"][0]["summary"]
    assert all(0.0 <= r["p"] <= 1.0 and r["entropy_bits"] >= 0.0 for r in rows)

    card = render_creature(agent)
    assert card["kind"] == "markov_gates" and "AND" in card["body"] and card["mascot"]
    assert "approximation" in _markov_gates_html(probabilistic.render_brain()) or \
        probabilistic.tpm_is_exact()
    assert "markov_gates" in KIND_META
    assert any(key.startswith("roster:markov_gates:") for key, _ in _analytics_field_options())

    # The vendored EMD refuses some cuts of deterministic gate brains (CLAUDE.md section 9q). The
    # interface must say so and still report causal autonomy, not fail the whole page.
    from ..metrics import phi_autonomy
    from ..webui import app as webapp

    def refuse(_agent):
        raise RuntimeError("emd: LP solve failed (infeasible)")

    saved = phi_autonomy.compute_phi
    phi_autonomy.compute_phi = refuse
    try:
        reported = webapp._gate_phi(_brain([Gate((0,), (2, 3), _table([0, 3], 2))]), n_t=2)
    finally:
        phi_autonomy.compute_phi = saved
    assert reported["phi"] is None and "not computed" in reported["note"], reported
    assert reported["autonomy"] is not None
    print("OK: the payload serialises, names its gates, the interface draws the card, and a "
          "refused Phi is reported as refused")


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
    print(f"\n{'all gate-brain tests passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
