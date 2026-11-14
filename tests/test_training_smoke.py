"""SYNTHETIC SMOKE TEST of the full data -> train -> evaluate pipeline.

Initial conditions are random sine sums solved with our own reference solver and
written in the same HDF5 layout as the downloaded PDEBench subset. The resulting
numbers say nothing about model quality; they only prove the pipeline runs.
"""

import copy
import json

import h5py
import numpy as np

from pde_service.config import load_config
from pde_service.solver.burgers import solve_burgers
from pde_service.training.train import train


def write_synthetic(path, n_samples=24, n_x=128, nu=0.1):
    rng = np.random.default_rng(0)
    x = (np.arange(n_x) + 0.5) / n_x
    t = np.arange(0, 2.01, 0.5)
    u = np.empty((n_samples, len(t), n_x), dtype=np.float32)
    for i in range(n_samples):
        u0 = sum(rng.uniform(-0.5, 0.5) * np.sin(2 * np.pi * k * x + rng.uniform(0, 6.3)) for k in (1, 2, 3))
        u[i, 0] = u0
        u[i, 1:] = solve_burgers(u0, t[1:], nu)
    with h5py.File(path, "w") as f:
        f["u"], f["t"], f["x"] = u, t, x
        f["done"] = np.ones(n_samples, bool)
        f.attrs["nu"] = nu


def test_pipeline_runs_end_to_end(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["data"].update(local_dir=str(tmp_path), n_samples=24, time_stride=50, output_stride=1,
                       split={"train": 16, "val": 4, "test": 4, "seed": 0})
    cfg["model"].update(grid_size=64, modes=8, width=16)
    cfg["training"].update(batch_size=8, run_dir=str(tmp_path / "runs"), artifact_dir=str(tmp_path / "art"))
    write_synthetic(tmp_path / "burgers_nu0.1_n24_stride50.h5")

    report = train("0.1", epochs=3, cfg=cfg, label="SYNTHETIC SMOKE TEST")

    assert report["label"] == "SYNTHETIC SMOKE TEST"
    assert report["split_sizes"] == {"train": 16, "val": 4, "test": 4}
    assert len(report["test"]["per_time"]) == 4
    assert report["split_overlap"] == {"train&val": 0, "train&test": 0, "val&test": 0}
    assert "mean_baseline" in report["test"]
    assert np.isfinite(report["test"]["overall"]["rel_l2"])
    saved = json.loads((tmp_path / "art" / "metrics_nu0.1.json").read_text())
    assert saved["label"] == "SYNTHETIC SMOKE TEST"
    assert (tmp_path / "art" / "fno_burgers_nu0.1.pt").exists()
