"""Load the downloaded PDEBench subset and build FNO train/val/test tensors.

Local file (written by data/download.py):
    u : (S, T, 1024) float32, u[:, j] is the solution at t[j]; t[0] = 0 is the initial condition
    t : (T,)  output times,  x : (1024,) cell centres in [0, 1)

FNO sample (s = output_stride):
    input  : (n, 2)   [u(x, 0), x]  on the downsampled grid (n = grid_size)
    target : (n, K)   u(x, t) at t[s], t[2s], ...  (K = (T - 1) // s)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch


@dataclass
class Split:
    inputs: torch.Tensor   # (S, n, 2)
    targets: torch.Tensor  # (S, n, K)
    indices: np.ndarray    # sample indices in the PDEBench file


@dataclass
class BurgersData:
    nu: float
    x: np.ndarray      # (n,)
    t_out: np.ndarray  # (K,)
    train: Split
    val: Split
    test: Split


def split_indices(n_samples: int, n_train: int, n_val: int, n_test: int, seed: int):
    if n_train + n_val + n_test > n_samples:
        raise ValueError(f"split {n_train}+{n_val}+{n_test} exceeds {n_samples} samples")
    perm = np.random.default_rng(seed).permutation(n_samples)
    return (
        np.sort(perm[:n_train]),
        np.sort(perm[n_train : n_train + n_val]),
        np.sort(perm[n_train + n_val : n_train + n_val + n_test]),
    )


def make_inputs(u0: np.ndarray, x: np.ndarray) -> torch.Tensor:
    """(S, n) initial conditions + grid -> (S, n, 2) FNO input."""
    u0 = np.asarray(u0, dtype=np.float32)
    grid = np.broadcast_to(np.asarray(x, dtype=np.float32), u0.shape)
    return torch.from_numpy(np.stack([u0, grid], axis=-1).copy())


def load_burgers(path: Path, grid_size: int, split: dict, output_stride: int = 1) -> BurgersData:
    with h5py.File(path, "r") as f:
        done = f["done"][:]
        if not done.all():
            raise RuntimeError(f"{path}: {int((~done).sum())} samples not downloaded yet")
        u, t, x, nu = f["u"][:], f["t"][:], f["x"][:], float(f.attrs["nu"])
    n_x = u.shape[-1]
    if n_x % grid_size:
        raise ValueError(f"grid_size {grid_size} must divide {n_x}")
    stride = n_x // grid_size
    u, x = u[..., ::stride], x[::stride]
    u0, u_out, t_out = u[:, 0], u[:, output_stride::output_stride], t[output_stride::output_stride]

    def build(idx: np.ndarray) -> Split:
        return Split(
            inputs=make_inputs(u0[idx], x),
            targets=torch.from_numpy(np.ascontiguousarray(u_out[idx].transpose(0, 2, 1))),
            indices=idx,
        )

    tr, va, te = split_indices(len(u), split["train"], split["val"], split["test"], split["seed"])
    return BurgersData(nu=nu, x=x, t_out=t_out, train=build(tr), val=build(va), test=build(te))
