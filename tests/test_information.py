"""
Tests for metrics/information.py -- mutual information and transfer entropy.

Both metrics are validated the same way: construct synthetic `records` dicts (no need to run a
real match, exactly like test_equilibrium.py operates directly on its metrics module) where the
true dependency structure is known by construction, and check the estimator recovers it --
near-zero for genuine independence, near-maximal for a deterministic/perfect-copy relationship.
For transfer entropy specifically, also check the surrogate test correctly passes the genuine
link and rejects the non-existent reverse link.
"""

import numpy as np

from gamebrains.metrics.information import (
    compute_all,
    mutual_information_per_agent,
    predictive_information_per_agent,
    transfer_entropy_pairwise,
)


def test_mutual_information_distinguishes_independent_from_deterministic():
    rng = np.random.default_rng(0)
    rounds = 4000
    # Synthetic binary "cooperators" series -- only its shape/role in reconstructing the shared
    # lag-1 observation matters here, not PGG semantics.
    cooperators = rng.integers(0, 2, size=rounds)
    actions = np.zeros((rounds, 2), dtype=int)
    actions[:, 0] = rng.integers(0, 2, size=rounds)  # independent of the observation

    obs = np.empty(rounds, dtype=int)
    obs[0] = 3  # n_agents(2) + 1 sentinel, matching _reconstruct_observations
    obs[1:] = cooperators[:-1]
    actions[:, 1] = obs % 2  # deterministic function of the observation

    records = {"actions": actions, "cooperators": cooperators}
    result = mutual_information_per_agent(records)

    assert result["bits_by_agent"][0] < 0.03, "independent action should have near-zero MI"
    assert result["bits_by_agent"][1] > 0.9, "deterministic function of obs should have near-maximal MI"
    assert result["n_samples"] == rounds


def test_transfer_entropy_detects_lagged_copying_and_rejects_the_reverse_direction():
    rng = np.random.default_rng(1)
    rounds = 3000
    actions = np.zeros((rounds, 2), dtype=int)
    actions[:, 0] = rng.integers(0, 2, size=rounds)  # source: unrelated random binary process
    actions[0, 1] = rng.integers(0, 2)
    actions[1:, 1] = actions[:-1, 0]  # target copies source with lag 1 -- a genuine T(0->1) link

    records = {"actions": actions, "cooperators": actions.sum(axis=1)}
    result = transfer_entropy_pairwise(records, seed=42, n_surrogates=200)

    forward = result["by_pair"][(0, 1)]
    assert forward["bits"] > 0.9, "perfect lag-1 copy should show near-maximal TE"
    assert forward["p_value"] < 0.01, "a genuine causal link should beat almost all surrogates"

    backward = result["by_pair"][(1, 0)]
    assert backward["bits"] < 0.1, "no reverse causal influence should show near-zero TE"
    assert backward["p_value"] >= 0.05, "no reverse link should not pass the surrogate test"


def test_predictive_information_distinguishes_alternator_from_iid_and_constant():
    rng = np.random.default_rng(3)
    rounds = 3000
    actions = np.zeros((rounds, 3), dtype=int)
    actions[:, 0] = np.arange(rounds) % 2          # deterministic alternator: past fixes future
    actions[:, 1] = rng.integers(0, 2, size=rounds)  # iid random: past says nothing
    actions[:, 2] = 1                                # constant (AllC-like): no variation, no info

    records = {"actions": actions, "cooperators": actions.sum(axis=1)}
    result = predictive_information_per_agent(records)

    assert result["bits_by_agent"][0] > 0.95, "alternator should carry ~1 bit of self-prediction"
    assert result["bits_by_agent"][1] < 0.03, "iid agent should carry near-zero self-prediction"
    assert abs(result["bits_by_agent"][2]) < 0.01, "constant agent has no information to carry"
    assert result["n_samples"] == rounds - 1


def test_compute_all_shape_and_headline_scalars():
    """Shape, plus the transfer-entropy guardrail's contract.

    Mutual and predictive information are computed over whatever window they are handed, since
    neither is a significance test and neither is confounded by a shared training trend. Transfer
    entropy is, so its headline number is only published when the window can be justified, and
    without a roster there is no exploration schedule to derive a burn-in from. `None` there is the
    designed answer rather than a missing value, and `transfer_entropy_bits_status` says so. See
    tests/test_information_guardrail.py for the null control that motivates it.
    """
    rng = np.random.default_rng(2)
    rounds = 500
    actions = rng.integers(0, 2, size=(rounds, 3))
    records = {"actions": actions, "cooperators": actions.sum(axis=1)}

    result = compute_all(records, seed=7, n_surrogates=50)

    assert isinstance(result["mutual_information_bits"], float)
    assert isinstance(result["predictive_information_bits"], float)
    assert result["transfer_entropy_bits"] is None
    assert "no roster supplied" in result["transfer_entropy_bits_status"]

    assert len(result["mutual_information_detail"]["bits_by_agent"]) == 3
    assert len(result["predictive_information_detail"]["bits_by_agent"]) == 3
    assert result["transfer_entropy_detail"]["n_pairs"] == 3 * 2  # every ordered pair, i != j
    # The per-pair detail is always present. Only the aggregate is ever withheld.
    assert len(result["transfer_entropy_detail"]["by_pair"]) == 3 * 2


if __name__ == "__main__":
    test_mutual_information_distinguishes_independent_from_deterministic()
    print("OK: mutual information distinguishes independence from deterministic dependence")
    test_transfer_entropy_detects_lagged_copying_and_rejects_the_reverse_direction()
    print("OK: transfer entropy detects a genuine lag-1 link and rejects the reverse direction")
    test_predictive_information_distinguishes_alternator_from_iid_and_constant()
    print("OK: predictive information separates alternator / iid / constant agents")
    test_compute_all_shape_and_headline_scalars()
    print("OK: compute_all returns the expected shape and headline scalars")
