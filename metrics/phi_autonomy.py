"""
Φ (integrated information) and causal autonomy for MarkovBrainAgent animats.

Uses the GameBrains PyPhi fork (see vendor/pyphi/FORK_NOTES.md) rather than Albantakis et al.'s
`autonomy` companion package (github.com/Albantakis/autonomy), because that package is officially
Windows-unsupported and requires the same unpatched PyPhi our fork exists to fix. The causal
autonomy measure below is a direct reimplementation of that package's `A_m_sensors_causal`
(causal_agent_analysis.py), built from the same PyPhi TPM utilities it uses internally.

Honesty note, carried over verbatim from that source's own comment: the full causal-autonomy
measure is a *difference* of two conditional entropies, H(M_t|S_hist) - H(M_t|M_{t-1},S_hist);
the second term is exactly zero for deterministic transition dynamics, which is why the reference
implementation computes only the first term, with an explicit "TODO: add the part for
non-deterministic agents" left in the source. Our MarkovBrainAgent has stochastic (sigmoid)
dynamics, so -- exactly as in the published implementation -- what we compute here is that first
term only: a first-order approximation, not the complete measure, for non-deterministic systems.
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

os.environ.setdefault("PYPHI_WELCOME_OFF", "yes")  # silence PyPhi's banner (incl. in worker procs)

_VENDOR = Path(__file__).resolve().parents[1] / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import numpy as np
import pyphi
from pyphi.convert import state_by_node2state_by_state, to_multidimensional
from pyphi.tpm import condition_tpm
from scipy.stats import entropy as _entropy

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


def _conditioned_hidden_motor_transition(tpm_multi: np.ndarray, n_nodes: int, n_sensor: int,
                                         hm_idx: tuple[int, ...], sensor_state: int) -> np.ndarray:
    """The (2^n_hm, 2^n_hm) state-by-state transition matrix over hidden+motor nodes, given the
    sensor nodes are fixed to `sensor_state`. See module docstring; shapes verified empirically
    against pyphi.tpm.condition_tpm (a plain to_2dimensional does not apply here, since it assumes
    the conditioned array's free dimensions still equal the full node count -- they don't once
    some dimensions have been collapsed to singletons by conditioning).
    """
    n_hm = len(hm_idx)
    sensor_bits = tuple((sensor_state >> k) & 1 for k in range(n_sensor))
    full_state = sensor_bits + tuple([0] * (n_nodes - n_sensor))

    fixed_indices = tuple(range(n_sensor))
    cond = condition_tpm(tpm_multi, fixed_indices, full_state)
    squeezed = cond.squeeze(axis=fixed_indices)          # shape (2,)*n_hm + (n_nodes,)
    flat = squeezed.reshape([2 ** n_hm, n_nodes], order="F")
    state_by_node = flat[:, hm_idx]                       # (2^n_hm, n_hm)
    return state_by_node2state_by_state(state_by_node)    # (2^n_hm, 2^n_hm)


def causal_autonomy(agent, n_t: int = 5) -> float:
    """A_m_sensors_causal, reimplemented from Albantakis et al.'s `autonomy` toolbox: the entropy
    of the hidden+motor state at time t given the preceding n_t sensor readings, assuming every
    length-n_t sensor sequence is equally likely and starting from a maximum-entropy prior over
    the hidden+motor state before the sequence. See the module docstring for the documented scope
    limitation on non-deterministic systems (which applies to our stochastic animats).
    """
    tpm = agent.full_tpm()
    n = agent.n_nodes
    n_sensor = agent.n_sensor
    hm_idx = tuple(range(n_sensor, n))
    n_hm = len(hm_idx)

    tpm_multi = to_multidimensional(tpm)
    tpm_cond_input = [
        _conditioned_hidden_motor_transition(tpm_multi, n, n_sensor, hm_idx, si)
        for si in range(2 ** n_sensor)
    ]

    maxent = np.full(2 ** n_hm, 1.0 / (2 ** n_hm))
    reps = []
    for seq in itertools.product(range(2 ** n_sensor), repeat=n_t):
        rep = maxent
        for si in seq:
            rep = np.average(tpm_cond_input[si], weights=rep, axis=0)
        reps.append(rep)

    return float(_entropy(np.mean(reps, axis=0), base=2))
