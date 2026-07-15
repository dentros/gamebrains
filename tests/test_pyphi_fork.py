"""
Regression test for the vendored PyPhi fork (see gamebrains/vendor/pyphi/FORK_NOTES.md).

Confirms the fork (a) imports on Python 3.11+/NumPy 2.x, (b) uses our pure NumPy/SciPy `emd`
in place of the broken `pyemd` C-extension, and (c) reproduces PyPhi's own documented
reference Phi value exactly — i.e. the replacement EMD is numerically equivalent, not just
"a number that runs."
"""

import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parents[1] / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

import pyphi  # noqa: E402

pyphi.config.PARALLEL_CUT_EVALUATION = False
pyphi.config.PARALLEL_CONCEPT_EVALUATION = False
pyphi.config.PARALLEL_COMPLEX_EVALUATION = False
pyphi.config.PROGRESS_BARS = False


def test_uses_numpy_emd_backend():
    assert pyphi.distance.emd.__module__ == "pyphi._emd_numpy"


def test_matches_documented_reference_phi():
    """PyPhi's own canonical tutorial network; documented Phi = 2.3125."""
    subsystem = pyphi.examples.basic_subsystem()
    sia = pyphi.compute.sia(subsystem)
    assert abs(sia.phi - 2.3125) < 1e-4


if __name__ == "__main__":
    test_uses_numpy_emd_backend()
    test_matches_documented_reference_phi()
    print("OK: pyphi fork matches documented reference Phi = 2.3125")
