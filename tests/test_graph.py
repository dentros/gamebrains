"""
Tests for metrics/graph.py -- co-cooperation and TE-influence interaction graphs.

Same validation style as test_information.py: synthetic `records` (and a hand-built
`te_detail` dict, since the influence graph deliberately consumes an already-computed
transfer_entropy_pairwise result rather than recomputing TE) where the correct graph
structure is known by construction.
"""

import numpy as np

from gamebrains.metrics.graph import co_cooperation_graph, compute_all, influence_graph


def test_co_cooperation_weights_match_hand_computed_probabilities():
    # 4 rounds, 3 agents. Agent 0 and 1 both cooperate in rounds 0 and 2 (weight 0.5);
    # agent 2 never cooperates (weight 0 with everyone).
    actions = np.array([
        [1, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ])
    g = co_cooperation_graph(actions)
    assert g[0][1]["weight"] == 0.5
    assert g[0][2]["weight"] == 0.0
    assert g[1][2]["weight"] == 0.0


def test_influence_graph_keeps_only_significant_pairs():
    te_detail = {"by_pair": {
        (0, 1): {"bits": 0.30, "p_value": 0.004},   # significant -> edge
        (1, 0): {"bits": 0.02, "p_value": 0.61},    # not significant -> no edge
        (0, 2): {"bits": 0.10, "p_value": 0.049},   # just under alpha -> edge
        (2, 0): {"bits": 0.50, "p_value": 0.050},   # exactly alpha -> excluded (strict <)
    }}
    g = influence_graph(te_detail, n_agents=3)
    assert g.has_edge(0, 1) and g[0][1]["bits"] == 0.30
    assert not g.has_edge(1, 0)
    assert g.has_edge(0, 2)
    assert not g.has_edge(2, 0)
    assert g.number_of_edges() == 2


def test_compute_all_headline_scalars_and_detail_shape():
    rounds, n = 100, 3
    actions = np.ones((rounds, n), dtype=int)  # everyone always cooperates
    records = {"actions": actions}
    te_detail = {"by_pair": {(i, j): {"bits": 0.0, "p_value": 1.0}
                             for i in range(n) for j in range(n) if i != j}}

    result = compute_all(records, te_detail)

    assert result["coop_graph_weight"] == 1.0, "all-cooperate roster: every pair weight is 1"
    assert result["coop_graph_clustering"] == 1.0, "complete weight-1 graph: clustering 1"
    assert result["influence_graph_density"] == 0.0, "no significant TE pairs: density 0"
    assert len(result["graph_detail"]["coop_edges"]) == n * (n - 1) // 2
    assert result["graph_detail"]["influence_edges"] == []
    assert result["graph_detail"]["n_agents"] == n


if __name__ == "__main__":
    test_co_cooperation_weights_match_hand_computed_probabilities()
    print("OK: co-cooperation weights match hand-computed probabilities")
    test_influence_graph_keeps_only_significant_pairs()
    print("OK: influence graph keeps only surrogate-significant TE pairs")
    test_compute_all_headline_scalars_and_detail_shape()
    print("OK: compute_all headline scalars and detail shape are correct")
