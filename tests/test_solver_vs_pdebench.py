"""Independent validation: our spectral solver vs. stored PDEBench trajectories.

This is what fixes the PDE convention (D = nu/pi) and the time-index mapping
(tensor[k] <-> t = 0.01 k). Skipped if the PDEBench subset has not been downloaded.
"""

import glob

import h5py
import numpy as np
import pytest

from pde_service.config import resolve
from pde_service.solver.burgers import solve_burgers

FILES = sorted(glob.glob(str(resolve("data/pdebench/burgers_nu0.1_n*_stride10.h5"))))
pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(not FILES, reason="PDEBench nu=0.1 subset not downloaded"),
]


def test_solver_reproduces_pdebench_nu01():
    with h5py.File(FILES[0]) as f:
        idx = np.flatnonzero(f["done"][:])[:3]  # only fully downloaded samples
        u, t, nu = f["u"][idx], f["t"][:], float(f.attrs["nu"])
    assert len(idx) > 0
    for i in range(len(idx)):
        s = solve_burgers(u[i, 0], t[1:], nu)
        rel = np.linalg.norm(s - u[i, 1:], axis=1) / np.linalg.norm(u[i, 1:], axis=1)
        assert rel.max() < 5e-3, rel
