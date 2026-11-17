"""
Tests for MarkovBrainAgent and its Phi computation.

Causal autonomy has its own module, `tests/test_causal_autonomy.py`, since 2026-11-18. The two
cases that used to be here asserted the earlier implementation's value for a fair-coin genome
(n_hm bits), which the published definition puts at 0.
"""

import numpy as np

from gamebrains.agents.markov_brain import MarkovBrainAgent, crossover
from gamebrains.metrics.phi_autonomy import compute_phi


def test_observation_encoding_and_action_decoding_roundtrip():
    agent = MarkovBrainAgent("MB", n_states=8, n_actions=2, n_hidden=2, seed=0)
    assert agent.n_sensor == 3   # ceil(log2(8))
    assert agent.n_motor == 1
    assert agent.n_nodes == 3 + 2 + 1

    for obs in range(8):
        bits = agent._encode_observation(obs)
        recovered = sum(int(b) << k for k, b in enumerate(bits))
        assert recovered == obs


def test_act_is_reproducible_given_seed():
    a1 = MarkovBrainAgent("A", n_states=8, n_actions=2, n_hidden=2, seed=42)
    a2 = MarkovBrainAgent("A", n_states=8, n_actions=2, n_hidden=2, seed=42)
    obs_sequence = [7, 3, 5, 2, 6, 0, 1, 4]
    actions1 = [a1.act(o) for o in obs_sequence]
    actions2 = [a2.act(o) for o in obs_sequence]
    assert actions1 == actions2


def test_full_tpm_sensor_rows_are_identity():
    agent = MarkovBrainAgent("MB", n_states=8, n_actions=2, n_hidden=2, seed=0)
    tpm = agent.full_tpm()
    for s in range(2 ** agent.n_nodes):
        state = [(s >> k) & 1 for k in range(agent.n_nodes)]
        for k in range(agent.n_sensor):
            assert tpm[s, k] == state[k]


def test_clone_mutate_crossover_produce_valid_agents():
    base = MarkovBrainAgent.random("Base", n_states=8, n_actions=2, n_hidden=2, seed=1)

    clone = base.clone(seed=2)
    assert np.array_equal(clone.W, base.W) and np.array_equal(clone.bias, base.bias)
    assert clone is not base

    rng = np.random.default_rng(0)
    mutant = base.mutate(rng, rate=1.0, scale=1.0, seed=3)  # rate=1.0 -> every gene perturbed
    assert not np.array_equal(mutant.W, base.W)
    assert mutant.W.shape == base.W.shape

    other = MarkovBrainAgent.random("Other", n_states=8, n_actions=2, n_hidden=2, seed=4)
    child = crossover(base, other, rng, name="Child", seed=5)
    assert child.W.shape == base.W.shape
    # every child weight came from one parent or the other
    from_a = np.isclose(child.W, base.W)
    from_b = np.isclose(child.W, other.W)
    assert np.all(from_a | from_b)


def test_factory_seats_a_sigmoid_brain_in_the_congestion_family():
    """The congestion family declares no start observation. The factory read `game.start_state`
    directly and raised AttributeError there, so no sigmoid brain could be seated in a Honey-Jar
    match from the interface or a sweep."""
    from gamebrains.agents import factory
    from gamebrains.agents.markov_brain import spawn
    from gamebrains.engine.game import NO_START_OBSERVATION
    from gamebrains.engine.runner import run_match
    from gamebrains.games.congestion import from_preset
    from gamebrains.games.public_goods import PublicGoodsGame

    game = from_preset("hjg", 3)
    roster = [factory.build("markov_brain", i, game, seed=0) for i in range(3)]
    assert all(a.start_state == NO_START_OBSERVATION for a in roster), (
        "a board position may not clear the brain: the congestion family has no start observation")
    records = run_match(game, roster, rounds=200, seed=0)
    assert set(np.asarray(records["actions"]).flatten().tolist()) <= set(range(game.n_actions))
    assert spawn("MB", game, 0, None).start_state == NO_START_OBSERVATION

    pgg = PublicGoodsGame(n_agents=3)
    assert factory.build("markov_brain", 0, pgg, seed=0).start_state == pgg.start_state, (
        "the Public Goods Game declares its start observation and the brain must still use it")


def test_compute_phi_runs_and_returns_nonnegative():
    agent = MarkovBrainAgent.random("MB", n_states=8, n_actions=2, n_hidden=2, seed=0)
    for _ in range(5):
        agent.act(agent.rng.integers(8))
    result = compute_phi(agent)
    assert result["phi"] >= 0.0


if __name__ == "__main__":
    test_observation_encoding_and_action_decoding_roundtrip()
    print("OK: observation encoding round-trips")
    test_act_is_reproducible_given_seed()
    print("OK: act() is reproducible given seed")
    test_full_tpm_sensor_rows_are_identity()
    print("OK: full_tpm sensor rows are identity")
    test_clone_mutate_crossover_produce_valid_agents()
    print("OK: clone/mutate/crossover produce valid agents")
    test_factory_seats_a_sigmoid_brain_in_the_congestion_family()
    print("OK: the factory seats a sigmoid brain in the congestion family")
    test_compute_phi_runs_and_returns_nonnegative()
    print("OK: compute_phi runs and returns Phi >= 0")
