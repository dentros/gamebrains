"""
Graph-theoretic metrics over a match's interaction structure (CLAUDE.md section 5's third metric
family: "metrics on the interaction / who-cooperates-with-whom network via networkx; correlate
these with the social metrics").

Two graphs are built per match, both directly from data the platform already records -- no new
instrumentation, consistent with the "metrics are pure functions of the log" principle:

- The **co-cooperation graph** (undirected, weighted): nodes are agents; the weight of edge (i, j)
  is the fraction of rounds in which BOTH i and j cooperated. This is the literal
  "who-cooperates-with-whom" network. Built from `records["actions"]` alone.
- The **influence graph** (directed, unweighted edges annotated with TE): nodes are agents; a
  directed edge i -> j exists iff the transfer entropy T(i -> j) passed its surrogate significance
  test (p < 0.05) in metrics/information.py. This deliberately reuses the already-computed,
  already-validated TE results rather than recomputing anything -- the graph layer only adds the
  topological reading (density, who influences whom) on top of the per-pair numbers.

Headline scalars (what `_METRIC_META` displays): the co-cooperation graph's mean edge weight and
weighted clustering coefficient, and the influence graph's density. Full node/edge breakdowns are
in the `*_detail` dicts for the results page.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np


def co_cooperation_graph(actions: np.ndarray) -> nx.Graph:
    """Undirected weighted graph; weight (i, j) = P(both i and j cooperate in the same round)."""
    rounds, n_agents = actions.shape
    g = nx.Graph()
    g.add_nodes_from(range(n_agents))
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            w = float(np.mean(actions[:, i] * actions[:, j]))
            g.add_edge(i, j, weight=w)
    return g


def influence_graph(te_detail: dict[str, Any], n_agents: int, alpha: float = 0.05) -> nx.DiGraph:
    """Directed graph of surrogate-significant TE links, from an already-computed
    `transfer_entropy_pairwise` result (its `by_pair` dict). Only pairs with p_value < alpha
    become edges; the raw TE value is kept as the edge's `bits` attribute."""
    g = nx.DiGraph()
    g.add_nodes_from(range(n_agents))
    for (i, j), d in te_detail.get("by_pair", {}).items():
        if d["p_value"] < alpha:
            g.add_edge(i, j, bits=d["bits"], p_value=d["p_value"])
    return g


def compute_all(records: dict[str, np.ndarray], te_detail: dict[str, Any]) -> dict[str, Any]:
    """Entry point mirroring metrics/social.py's and metrics/information.py's `compute_all`.
    Headline scalars: `coop_graph_weight` (mean co-cooperation edge weight -- 1.0 means every pair
    cooperated together every round, 0.0 means no pair ever did), `coop_graph_clustering`
    (weighted average clustering: do cooperating pairs form cooperating triangles, or only
    isolated pairs?), and `influence_graph_density` (fraction of possible directed influence links
    that are surrogate-significant -- the graph reading of the TE panel's `n_significant_pairs`).
    """
    actions = records["actions"]
    n_agents = actions.shape[1]

    coop_g = co_cooperation_graph(actions)
    weights = [d["weight"] for _, _, d in coop_g.edges(data=True)]
    mean_weight = float(np.mean(weights)) if weights else 0.0
    # nx.average_clustering(weight=...) needs at least one triangle to be meaningful; for n=2 it
    # is 0 by construction (no triangles exist), which is correct, not an error.
    clustering = float(nx.average_clustering(coop_g, weight="weight")) if n_agents >= 2 else 0.0

    infl_g = influence_graph(te_detail, n_agents)
    possible = n_agents * (n_agents - 1)
    density = float(infl_g.number_of_edges() / possible) if possible else 0.0

    weighted_degree = dict(coop_g.degree(weight="weight"))
    return {
        "coop_graph_weight": mean_weight,
        "coop_graph_clustering": clustering,
        "influence_graph_density": density,
        "graph_detail": {
            "coop_edges": [
                {"i": i, "j": j, "weight": d["weight"]} for i, j, d in coop_g.edges(data=True)
            ],
            "coop_weighted_degree": {int(k): float(v) for k, v in weighted_degree.items()},
            "influence_edges": [
                {"source": u, "target": v, "bits": d["bits"], "p_value": d["p_value"]}
                for u, v, d in infl_g.edges(data=True)
            ],
            "n_agents": int(n_agents),
        },
    }
