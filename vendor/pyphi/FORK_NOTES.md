# GameBrains fork of PyPhi 1.2.0

**Base:** PyPhi 1.2.0 (PyPI), the last release of the toolbox introduced in
Mayner et al. (2018), *PyPhi: A toolbox for integrated information theory*, PLOS Computational
Biology 14(7): e1006343. Unmaintained since ~2020; targets Python ≤3.8.

## Why this fork exists

Getting PyPhi running on a modern stack (Python 3.11, NumPy ≥2.0, Windows) hits three
independent, cumulative breakages:

1. **`collections.Iterable`/`Sequence`/`Mapping` moved to `collections.abc`** in Python 3.10,
   breaking 6 call sites across `db.py`, `models/cmp.py`, `labels.py`,
   `models/actual_causation.py`, `models/cuts.py`, `models/subsystem.py`, `registry.py`.
2. **The `pyemd` C-extension dependency is broken both ways** on a current stack:
   - `pyemd==1.0.0` (the only version with a prebuilt Windows wheel) was compiled against the
     pre-2.0 NumPy C ABI and raises `ValueError: numpy.dtype size changed...` under NumPy ≥2.0
     (the NumPy 2.0 ABI break).
   - `pyemd==1.1.0` (ABI-fixed) ships no Windows wheel and fails to build from source without a
     complete C toolchain (missing linker on a typical Windows dev machine).
3. A Windows-specific antivirus interaction (Defender's `Sabsik.FL.B!ml` heuristic quarantining a
   freshly-compiled `.pyd` mid-build) can additionally break *any* local compile attempt — a known,
   common false positive for freshly built, unsigned native extensions, not a real threat.

None of these are fixable by pinning versions alone. This fork applies the **minimum surgical
patch**, leaving PyPhi's actual IIT computation (cause-effect repertoires, MIP search, concept
structures) completely untouched:

## What changed

| File | Change |
|---|---|
| `db.py`, `models/cmp.py`, `labels.py`, `models/actual_causation.py`, `models/cuts.py`, `models/subsystem.py`, `registry.py` | `collections.X` → `collections.abc.X` (7 call sites, 2 import styles) |
| `distance.py` | `from pyemd import emd` → `from ._emd_numpy import emd` |
| `_emd_numpy.py` **(new)** | Pure NumPy/SciPy Earth Mover's Distance: exact optimal-transport LP via `scipy.optimize.linprog` (sparse constraints), matching `pyemd.emd`'s signature and semantics for equal-mass histograms (always true for PyPhi's use: both inputs are probability distributions). No compiled extension required. |

## Validation

`gamebrains/tests/test_pyphi_fork.py` reproduces PyPhi's own documented canonical example
(`pyphi.examples.basic_subsystem()`) and asserts **Φ = 2.3125**, matching the published reference
value exactly — i.e. the NumPy/SciPy EMD replacement is numerically equivalent to `pyemd`'s
compiled implementation, not merely "a value that runs."

## Known cost

The LP-based EMD is exact but scales as O((2^N)^2) variables for an N-node purview; fine for the
tiny Markov-brain animats GameBrains targets (Φ is only ever computed on small systems — Φ search
itself is exponential in system size well before EMD becomes the bottleneck), not intended as a
general replacement for large-scale PyPhi usage.

## Upstream

This fix has not yet been proposed upstream (to `wmayner/pyphi` and/or the `pyemd` maintainers).
Given how generic the breakage is (any current Python/NumPy/Windows user hits it), doing so is a
worthwhile, low-effort open-source contribution independent of GameBrains.
