"""
Tests for MarkovBrainAgent and metrics.phi_autonomy.

Validated against two analytically known degenerate cases (not just "it runs without erroring"):
  - all-zero genome -> every hidden/motor node is an independent fair coin each step -> the
    hidden+motor joint state is exactly uniform over 2^n_hm outcomes regardless of any sensor
    history, so causal_autonomy must equal exactly n_hm bits.
  - strongly-negative-bias genome -> every hidden/motor node is (near-)always 0 -> the joint
    state is (near-)deterministic, so causal_autonomy must be close to 0 bits.
"""

import numpy as np

from gamebrains.agents.markov_brain import MarkovBrainAgent, crossover
from gamebrains.metrics.phi_autonomy import causal_autonomy, compute_phi


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


def test_causal_autonomy_fair_coin_equals_n_hm_bits():
    n_hidden = 2
    agent = MarkovBrainAgent(
        "FairCoin", n_states=8, n_actions=2, n_hidden=n_hidden,
        W=np.zeros((n_hidden + 1, 3 + n_hidden + 1)), bias=np.zeros(n_hidden + 1), seed=0,
    )
    result = causal_autonomy(agent, n_t=3)
    assert abs(result - (n_hidden + 1)) < 1e-9  # n_hm = n_hidden + n_motor(=1) bits, exactly


def test_causal_autonomy_near_deterministic_is_near_zero():
    n_hidden = 2
    n_controlled = n_hidden + 1
    agent = MarkovBrainAgent(
        "Deterministic", n_states=8, n_actions=2, n_hidden=n_hidden,
        W=np.zeros((n_controlled, 3 + n_controlled)), bias=np.full(n_controlled, -30.0), seed=0,
    )
    result = causal_autonomy(agent, n_t=3)
    assert result < 1e-6


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
    test_causal_autonomy_fair_coin_equals_n_hm_bits()
    print("OK: causal_autonomy == n_hm bits for the fair-coin genome")
    test_causal_autonomy_near_deterministic_is_near_zero()
    print("OK: causal_autonomy ~= 0 for the near-deterministic genome")
    test_compute_phi_runs_and_returns_nonnegative()
    print("OK: compute_phi runs and returns Phi >= 0")
