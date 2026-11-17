"""
Φ (integrated information) and causal autonomy for the two Markov brains.

Φ uses the GameBrains PyPhi fork (see vendor/pyphi/FORK_NOTES.md).

Causal autonomy is computed from its published definition, Albantakis (2021), "Quantifying the
autonomy of structurally diverse automata", Entropy 23:1415, Eq. 12, for the system V = {O, M} of
hidden and motor nodes and its sensors S:

    A_m = H(O_t, M_t | S^_{t-1}, ..., S^_{t-m}) - H(O_t, M_t | V^_{t-1})

A hat is an intervention that sets the variable to the uniform distribution, and the hidden and
motor state is uniform before the sensor history begins. The first term is what remains uncertain
about the system's state once the sensor history is known, so what the system carries from its own
past. The second is the noise of one step from a uniformly chosen state, and is zero for
deterministic dynamics. Both terms are computed here, so the measure is complete for stochastic
brains as well. The code works on the brain's own transition table with numpy alone; it shares no
code with the author's `autonomy` toolbox and does not use PyPhi.

**What changed on 2026-11-18.** The function this replaces was described as a reimplementation of
that toolbox, and it took the entropy of the hidden and motor distribution *averaged* over sensor
histories. That entropy does not condition on the history, so it counted the sensors' own variation
as the system's: a node that only copied a sensor read 1 bit, and nodes that were pure noise read
their full width. Under the definition above both read 0. It also computed the first term only.
`tests/test_causal_autonomy.py` holds the cases that separate the two.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYPHI_WELCOME_OFF", "yes")  # silence PyPhi's banner (incl. in worker procs)

_VENDOR = Path(__file__).resolve().parents[1] / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import numpy as np
import pyphi

from ..engine import procguard

pyphi.config.PROGRESS_BARS = False


#: Below this many nodes, spawning workers costs more than the work they do, so $\Phi$ is computed
#: serially even where spawning is perfectly safe. Measured on this project's own animats, medians
#: of three repetitions, $\Phi$ identical to six decimal places in every condition. Two independent
#: sweeps on the same machine:
#:
#:                 serial          parallel        verdict
#:     6 nodes     4.57s / 3.79s   15.87s / 28.35s serial wins, by 3.5x and by 7.5x
#:     7 nodes     85.12s / 84.56s 46.33s / 76.30s parallel wins, by 1.8x and by 1.1x
#:
#: Read those two columns as one finding and one caveat. The finding is that below the crossover
#: serial is decisively faster, in the same direction by a large margin both times. The caveat is
#: that at the crossover itself the margin (1.8x, then 1.1x) is inside this machine's known
#: wall-clock noise, so 7 is where the sign flips rather than where a reliable gain begins.
#:
#: The shapes explain it: worker start-up on Windows is a fixed cost of seconds paid per cut, while
#: cut evaluation grows with the state space. Re-measure before changing this -- it is a property
#: of the machine as much as of the code, and `experiments/run_phi_scaling.py` regenerates the
#: table. Anyone porting small-animat IIT work to a spawn-based platform should expect a crossover
#: of their own rather than inheriting this number.
PARALLEL_MIN_NODES = 7

_parallelism_settled: int | None = None


def _settle_parallelism(n_nodes: int) -> None:
    """Decide whether to use PyPhi's parallel cut evaluation, by trying it rather than assuming.

    PyPhi's own default is True and it is worth having: parallel cut evaluation is most of the
    difference between a $\\Phi$ computation taking seconds and one taking minutes. But its workers
    are spawned, and spawning fails in contexts this project met for real, where the failure is not
    a clean exception but leaked processes and an eventual MemoryError from the LP solver.

    Both of the project's earlier answers were guesses in opposite directions, and each was wrong
    somewhere: forcing parallelism off cost it everywhere it was fine, leaving it on leaked
    processes where it was not. `engine.procguard` decides by starting one worker and waiting for
    it.

    **Called at computation time, never at import time, and that distinction is not cosmetic.** A
    spawned child re-imports the parent's `__main__`, so a probe reachable from any module body
    runs again inside the child and tries to spawn during its bootstrap, which raises. The first
    version of this code probed at import and reproduced that failure exactly. Deferring the
    decision to the first $\\Phi$ computation means the probe only ever runs somewhere a process
    can legitimately be started.

    Two independent questions, and conflating them is what made both earlier answers wrong. *Can*
    we spawn here is about the environment, and `procguard` probes it. *Should* we is about the
    workload, and below `PARALLEL_MIN_NODES` the answer is no even in a perfectly healthy process,
    because worker start-up costs more than the cuts save. The project previously had no answer to
    the second question at all, and paid up to 3.5x for it on its smallest animats.
    """
    global _parallelism_settled
    if _parallelism_settled == n_nodes:
        return
    _parallelism_settled = n_nodes

    big_enough = n_nodes >= PARALLEL_MIN_NODES
    ok = big_enough and procguard.spawn_is_safe()
    pyphi.config.PARALLEL_CUT_EVALUATION = ok
    if not ok:
        why = (f"{n_nodes} nodes is below the {PARALLEL_MIN_NODES}-node threshold where workers "
               f"start paying for themselves" if not big_enough
               else procguard.spawn_verdict_reason())
        print(f"[phi] computing serially: {why}. The value is identical either way.")


def compute_phi(agent) -> dict:
    """Φ of the animat's hidden+motor subsystem at its current state, via PyPhi's `sia`.

    We deliberately compute `sia` directly on the fixed hidden+motor subsystem rather than
    `pyphi.compute.major_complex` (which searches over every candidate subsystem of the network):
    empirically, on these animats the maximal-Φ complex always turns out to be exactly the
    hidden+motor nodes (sensors are clamped, not autonomous, so they contribute little intrinsic
    cause-effect power) -- and skipping that outer search over all other candidate subsystems is
    the difference between a computation taking on the order of a minute rather than ~20 minutes
    for a network of just 5-6 nodes on this machine. This trades a small amount of generality
    (we no longer detect the rare case where sensors *do* end up part of the main complex) for
    practicality; see FORK_NOTES.md-style honesty here in the docstring rather than silently
    hiding the shortcut.
    """
    _settle_parallelism(agent.n_nodes)
    tpm = agent.full_tpm()
    state = tuple(int(x) for x in agent.state)
    hm_idx = tuple(range(agent.n_sensor, agent.n_nodes))
    network = pyphi.Network(tpm)
    subsystem = pyphi.Subsystem(network, state, nodes=hm_idx)
    sia = pyphi.compute.sia(subsystem)
    return {"phi": float(sia.phi), "complex_nodes": list(hm_idx)}


#: The largest (number of sensor histories) x (number of hidden and motor states) the first term
#: will enumerate, about half a gigabyte of float64. Past it the computation is refused with the
#: count, which is more useful than a machine that stops answering.
MAX_AUTONOMY_CELLS = 2 ** 26


def _hidden_motor_transitions(agent) -> np.ndarray:
    """T[s, i, j] = P(hidden and motor state j at t+1 | sensors s and hidden and motor state i at t).

    Read off `agent.full_tpm()`, the state-by-node table both Markov brains provide. Its row index
    is the whole state, little-endian, with the sensors as the low `n_sensor` bits, so row
    `s + i * 2**n_sensor` holds sensors `s` and hidden and motor state `i`. Its columns are each
    node's probability of being on next.

    The next state's probability is the product of its nodes' probabilities, which is exact when
    the nodes are drawn independently given the current state: always for the sigmoid brain and
    for deterministic gates, and for probabilistic gates only where `tpm_is_exact()` says so. Φ
    rests on the same table and the same caveat.
    """
    tpm = np.asarray(agent.full_tpm(), dtype=float)
    n_s = int(agent.n_sensor)
    n_hm = int(agent.n_nodes) - n_s
    on = tpm[:, n_s:]                                                   # (2^n, n_hm)
    bits = (np.arange(2 ** n_hm)[:, None] >> np.arange(n_hm)[None, :]) & 1
    joint = np.where(bits[None, :, :] == 1, on[:, None, :], 1.0 - on[:, None, :]).prod(axis=2)
    return joint.reshape(2 ** n_hm, 2 ** n_s, 2 ** n_hm).transpose(1, 0, 2)


def _entropy_bits(rows: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits of each row, with 0 log 0 taken as 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(rows > 0.0, rows * np.log2(rows), 0.0)
    return -terms.sum(axis=-1)


def causal_autonomy_terms(agent, n_t: int = 4) -> dict:
    """Both terms of Albantakis (2021) Eq. 12 and their difference, for a sensor history of `n_t`.

    `given_sensors` is H(O_t, M_t | S^_{t-1..t-n_t}): every sensor history of length `n_t` equally
    likely, the hidden and motor state uniform before it, and the entropy of the state at t
    averaged over histories. Averaged after taking the entropy, so a history that fixes the state
    contributes nothing. `given_previous_state` is H(O_t, M_t | V^_{t-1}): the entropy of one step
    averaged over every whole state at t-1, sensors included.

    `autonomy` is the first minus the second. It is at most the number of hidden and motor nodes.
    For a stochastic brain it can fall slightly below zero, because the two terms average over
    different distributions of the previous state; it is reported as computed, not clipped.
    The published evaluation used `n_t = 4`.
    """
    if int(n_t) < 1:
        raise ValueError(f"the sensor history must be at least one step long, got n_t={n_t}")
    n_t = int(n_t)
    T = _hidden_motor_transitions(agent)
    n_sensor_states, n_hm_states = T.shape[0], T.shape[1]
    histories = n_sensor_states ** n_t
    if histories * n_hm_states > MAX_AUTONOMY_CELLS:
        raise ValueError(
            f"{n_sensor_states}^{n_t} = {histories:,} sensor histories over {n_hm_states} hidden "
            f"and motor states is past the {MAX_AUTONOMY_CELLS:,}-cell limit this enumeration "
            f"allows; use a shorter history")

    dist = np.full((1, n_hm_states), 1.0 / n_hm_states)
    for _ in range(n_t):
        dist = np.einsum("hi,sij->hsj", dist, T).reshape(-1, n_hm_states)
    given_sensors = float(_entropy_bits(dist).mean())
    given_previous_state = float(_entropy_bits(T.reshape(-1, n_hm_states)).mean())
    return {"autonomy": given_sensors - given_previous_state, "given_sensors": given_sensors,
            "given_previous_state": given_previous_state, "n_t": n_t}


def causal_autonomy(agent, n_t: int = 4) -> float:
    """Causal autonomy in bits, Albantakis (2021) Eq. 12. See `causal_autonomy_terms`."""
    return causal_autonomy_terms(agent, n_t)["autonomy"]
