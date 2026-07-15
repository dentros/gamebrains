"""
Pure NumPy/SciPy replacement for pyemd's ``emd`` function.

**GameBrains fork note.** PyPhi 1.2.0 (unmaintained since ~2020) depends on the compiled
``pyemd`` C-extension for Earth Mover's Distance. On a modern stack this dependency is
broken two different ways: the only prebuilt Windows wheel (pyemd 1.0.0) was compiled
against the pre-2.0 NumPy C-ABI and segfaults/raises under NumPy>=2.0 (the "great NumPy 2.0
ABI break"), while the ABI-fixed release (pyemd 1.1.0) ships no Windows wheel and requires a
full C toolchain with a linker to build from source.

Since PyPhi only ever calls ``emd`` on the state space of a small subsystem/purview (tiny
Markov-brain animats, on the order of a handful of nodes), an *exact* LP-based
optimal-transport solve is both correct and fast enough here — no compiled extension
needed. This module solves the same transportation-problem formulation pyemd solves for
histograms of equal total mass (always true in PyPhi's usage: both inputs are probability
distributions summing to 1), so results are numerically equivalent.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


def emd(first_histogram: np.ndarray, second_histogram: np.ndarray,
        distance_matrix: np.ndarray) -> float:
    """Exact Earth Mover's Distance via linear programming (drop-in for pyemd.emd).

    Args:
        first_histogram, second_histogram: 1-D arrays of equal length and equal total mass.
        distance_matrix: square ground-distance matrix between histogram bins.

    Returns:
        The minimum-cost transportation plan's cost (a.k.a. Wasserstein-1 distance).
    """
    u = np.asarray(first_histogram, dtype=float)
    v = np.asarray(second_histogram, dtype=float)
    D = np.asarray(distance_matrix, dtype=float)
    m = u.size

    if u.shape != v.shape:
        raise ValueError("emd: histograms must have the same shape")
    if D.shape != (m, m):
        raise ValueError("emd: distance_matrix must be square with side == len(histograms)")

    total = u.sum()
    if total == 0:
        return 0.0
    if not np.isclose(total, v.sum()):
        raise ValueError("emd: histograms must have equal total mass")

    # Transportation LP: minimize sum_ij flow[i,j] * D[i,j]
    #   s.t. sum_j flow[i,j] = u[i]  (row sums)
    #        sum_i flow[i,j] = v[j]  (col sums)
    #        flow[i,j] >= 0
    # flow is flattened row-major: variable index = i*m + j.
    c = D.reshape(-1)

    n_vars = m * m
    idx = np.arange(n_vars)
    row_of = idx // m
    col_of = idx % m

    # Row-sum constraints (m rows) then column-sum constraints (m rows): 2m constraints total.
    # Each variable participates in exactly one row constraint and one column constraint,
    # so A_eq has exactly 2 nonzeros per column -> build it sparse.
    data = np.ones(2 * n_vars)
    rows = np.concatenate([row_of, m + col_of])
    cols = np.concatenate([idx, idx])
    A_eq = csr_matrix((data, (rows, cols)), shape=(2 * m, n_vars))
    b_eq = np.concatenate([u, v])

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(f"emd: LP solve failed ({result.message})")
    return float(result.fun)
