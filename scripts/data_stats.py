"""Statistics of the TRAINING initial conditions, used to set the API input limits
(api.max_abs_u0, api.ood_k_cut, api.ood_energy_fraction). Test data is not used.

Usage:  python scripts/data_stats.py --file data/pdebench/burgers_nu0.1_n1200_stride10.h5
"""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np

from pde_service.config import load_config
from pde_service.data.dataset import split_indices
from pde_service.inference.initial_conditions import spectral_energy_fraction_above


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    args = p.parse_args()
    cfg = load_config()
    s, grid = cfg["data"]["split"], cfg["model"]["grid_size"]
    with h5py.File(args.file) as f:
        u0_full = f["u"][:, 0]
    train_idx, _, _ = split_indices(len(u0_full), s["train"], s["val"], s["test"], s["seed"])
    u0 = u0_full[train_idx, :: u0_full.shape[1] // grid]

    peak = np.abs(u0).max(axis=1)
    spec = np.abs(np.fft.rfft(u0 - u0.mean(axis=1, keepdims=True), axis=1)) ** 2
    dominant_k = spec.argmax(axis=1)
    stats = {
        "n_train": len(u0),
        "grid": grid,
        "max_abs_u0": {"max": float(peak.max()), "p99": float(np.percentile(peak, 99)),
                       "median": float(np.median(peak))},
        "mean_u0": {"min": float(u0.mean(axis=1).min()), "max": float(u0.mean(axis=1).max())},
        "dominant_wavenumber_counts": {int(k): int(c) for k, c in zip(*np.unique(dominant_k, return_counts=True))},
        "energy_fraction_above_k": {
            k: {"max": float(max(spectral_energy_fraction_above(u, k) for u in u0)),
                "p99": float(np.percentile([spectral_energy_fraction_above(u, k) for u in u0], 99))}
            for k in (2, 4, 6, 8, 12, 16, 24, 32)
        },
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
