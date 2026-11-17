"""Causal autonomy against cases whose value follows from the published definition by hand.

The measure is Albantakis (2021), Entropy 23:1415, Eq. 12, for a system V = {O, M} of hidden and
motor nodes driven by sensors S:

    A_m = H(O_t, M_t | S^_{t-1}, ..., S^_{t-m}) - H(O_t, M_t | V^_{t-1})

with every intervened variable (the hat) set to the uniform distribution and the hidden and motor
state uniform before the sensor history starts. Each case below has a value that follows from
that formula without running anything, and each pins a different way an implementation can be
wrong:

  - fair coins: both terms are n_hm bits, so the measure is 0. A system that is pure noise has no
    autonomy. An implementation that reports only the first term reports n_hm here.
  - a node that copies a sensor: given the sensor history the node is known, so the measure is 0.
    An implementation that takes the entropy of the distribution averaged over sensor histories,
    without conditioning on them, reports 1 bit here, because it counts the sensors' own
    variation as the system's.
  - a node that copies itself: the sensor history says nothing about it and its own past fixes it,
    so the measure is 1 bit per such node.
  - a node that is always off: 0.

The platform computed the averaged form until 2026-11-18, and the first two cases fail on it.

Run:  "$GB/gamebrains/.venv/Scripts/python.exe" -m gamebrains.tests.test_causal_autonomy
"""

from __future__ import annotations

import sys

import numpy as np

from ..agents.markov_brain import MarkovBrainAgent
from ..agents.markov_gates import NO_START_OBSERVATION, Gate, MarkovGatesAgent
from ..metrics.phi_autonomy import causal_autonomy

#: A sigmoid unit with weight 40 and bias -20 is on with probability 1 - 2e-9 when its input is
#: on and 2e-9 when it is off, so the sigmoid cases hold to about 1e-6 bits rather than exactly.
NEAR = 1e-6


def _sigmoid_brain(W: np.ndarray, bias: np.ndarray, n_hidden: int = 2) -> MarkovBrainAgent:
    """n_states=8, n_actions=2: sensors s0..s2 are nodes 0..2, hidden h0..h{n-1}, then one motor."""
    return MarkovBrainAgent("A", n_states=8, n_actions=2, n_hidden=n_hidden, W=W, bias=bias, seed=0,
                            start_state=NO_START_OBSERVATION)


def _gate_brain(gates: list[Gate]) -> MarkovGatesAgent:
    """n_states=2, n_actions=2, n_hidden=1: s0 is node 0, h0 node 1, m0 node 2."""
    return MarkovGatesAgent("G", n_states=2, n_actions=2, n_hidden=1, deterministic=True,
                            gates=gates, seed=0, start_state=NO_START_OBSERVATION)


COPY = np.array([[1.0, 0.0], [0.0, 1.0]])


def test_fair_coins_have_no_autonomy() -> None:
    n_hidden = 2
    n_hm = n_hidden + 1
    agent = _sigmoid_brain(np.zeros((n_hm, 3 + n_hm)), np.zeros(n_hm), n_hidden)
    value = causal_autonomy(agent, n_t=3)
    assert abs(value) < 1e-9, (
        f"independent fair coins read {value:.6f} bits; both terms are {n_hm} bits, so the "
        f"measure is 0")
    print("OK: hidden and motor nodes that are fair coins have zero autonomy")


def test_a_node_copying_a_sensor_has_no_autonomy() -> None:
    n_hidden = 2
    n_hm = n_hidden + 1
    W = np.zeros((n_hm, 3 + n_hm))
    W[:, 0] = 40.0                                   # every hidden and motor node copies s0
    agent = _sigmoid_brain(W, np.full(n_hm, -20.0), n_hidden)
    value = causal_autonomy(agent, n_t=3)
    assert abs(value) < NEAR, (
        f"nodes that copy a sensor read {value:.6f} bits; the sensor history fixes them, so the "
        f"measure is 0. Reading 1 bit means the sensors' variation was counted as the system's")
    print("OK: nodes that copy a sensor have zero autonomy")


def test_a_node_copying_itself_has_one_bit_each() -> None:
    n_hidden = 2
    n_hm = n_hidden + 1
    W = np.zeros((n_hm, 3 + n_hm))
    for k in range(n_hm):
        W[k, 3 + k] = 40.0                           # each hidden and motor node copies itself
    agent = _sigmoid_brain(W, np.full(n_hm, -20.0), n_hidden)
    value = causal_autonomy(agent, n_t=3)
    assert abs(value - n_hm) < NEAR, (
        f"{n_hm} self-copying nodes read {value:.6f} bits, and each holds one bit the sensors "
        f"cannot explain")
    print(f"OK: {n_hm} self-copying nodes have {n_hm} bits of autonomy")


def test_nodes_that_are_always_off_have_no_autonomy() -> None:
    n_hm = 3
    agent = _sigmoid_brain(np.zeros((n_hm, 3 + n_hm)), np.full(n_hm, -30.0))
    value = causal_autonomy(agent, n_t=3)
    assert abs(value) < NEAR, f"constant nodes read {value:.6f} bits"
    print("OK: nodes that are always off have zero autonomy")


def test_gate_brain_cases_are_exact() -> None:
    """Deterministic gates draw nothing, so the gate cases hold exactly, not to a tolerance."""
    memory = _gate_brain([Gate((1,), (1,), COPY)])     # h0 <- h0, m0 never written
    follower = _gate_brain([Gate((0,), (1,), COPY)])   # h0 <- s0
    assert causal_autonomy(memory, n_t=3) == 1.0
    assert causal_autonomy(follower, n_t=3) == 0.0
    print("OK: a gate that copies a hidden node gives exactly 1 bit, one that copies a sensor 0")


def test_the_terms_are_reported_and_differ_by_the_measure() -> None:
    from ..metrics.phi_autonomy import causal_autonomy_terms
    n_hm = 3
    agent = _sigmoid_brain(np.zeros((n_hm, 3 + n_hm)), np.zeros(n_hm))
    terms = causal_autonomy_terms(agent, n_t=3)
    assert abs(terms["given_sensors"] - n_hm) < 1e-9
    assert abs(terms["given_previous_state"] - n_hm) < 1e-9
    assert terms["autonomy"] == terms["given_sensors"] - terms["given_previous_state"]
    assert terms["n_t"] == 3
    print("OK: both terms are reported, and the measure is their difference")


def test_a_history_too_long_to_enumerate_is_refused() -> None:
    from ..metrics.phi_autonomy import causal_autonomy_terms
    agent = _sigmoid_brain(np.zeros((3, 6)), np.zeros(3))
    try:
        causal_autonomy_terms(agent, n_t=40)
    except ValueError as exc:
        assert "sensor histories" in str(exc)
        print("OK: a sensor history too long to enumerate is refused with the count")
        return
    raise AssertionError("8^40 sensor histories were accepted")


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
    print(f"\n{'all causal autonomy tests passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
