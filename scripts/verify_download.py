"""Verify the downloaded PDEBench subset.

1. shapes, completeness, NaN/Inf counts
2. byte-offset check independent of our own arithmetic: download a few whole samples
   as one contiguous range, place them in the sparse probe file at the offset given by
   our formula, read tensor[i, ::stride, :] through h5py's own layout logic, and
   compare bit-for-bit with the row-wise download
3. physics sanity: spatial mean of u is conserved in time (periodic Burgers)
4. plot a few samples -> runs/verify_samples.png

Usage: python scripts/verify_download.py --file data/pdebench/burgers_nu0.1_n1200_stride10.h5 \
                                         --probe data/probe/probe_268185.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pde_service.config import load_config, resolve  # noqa: E402
from pde_service.data.download import resolve_signed_url  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--check-samples", type=int, nargs="+", default=None)
    args = p.parse_args()
    cfg = load_config()["data"]

    with h5py.File(args.file) as f:
        u, t, x, done = f["u"][:], f["t"][:], f["x"][:], f["done"][:]
        nu, tidx = float(f.attrs["nu"]), list(f.attrs["time_indices"])
    print("== 1. shapes and completeness")
    print(f"u {u.shape} {u.dtype} | t {t.shape} [{t[0]}..{t[-1]}] | x {x.shape} | nu {nu}")
    print(f"done: {int(done.sum())}/{len(done)}  NaN: {int(np.isnan(u).sum())}  Inf: {int(np.isinf(u).sum())}")
    print(f"all-zero rows (would indicate missing data): {int((np.abs(u).max(axis=-1) == 0).sum())}")
    print(f"value range [{u.min():.4f}, {u.max():.4f}]")

    print("== 2. byte-offset check through h5py's own layout")
    n = len(u)
    samples = args.check_samples or [0, n // 2, n - 1]
    row = cfg["n_x"] * 4
    sample_bytes = cfg["n_time_total"] * row
    url = resolve_signed_url(cfg["darus_file_ids"][str(nu)])
    with httpx.Client(timeout=300) as c, open(args.probe, "r+b") as probe:
        for i in samples:
            start = cfg["tensor_offset"] + i * sample_bytes  # our formula
            r = c.get(url, headers={"Range": f"bytes={start}-{start + sample_bytes - 1}"})
            assert r.status_code == 206 and len(r.content) == sample_bytes
            probe.seek(start)
            probe.write(r.content)
    with h5py.File(args.probe) as f:
        for i in samples:
            via_h5py = f["tensor"][i, tidx, :]  # h5py computes the location itself
            equal = np.array_equal(via_h5py, u[i])
            print(f"sample {i:5d}: h5py read == row-wise download: {equal}  "
                  f"(max |diff| {np.abs(via_h5py - u[i]).max():.2e})")
            assert equal

    print("== 3. conservation of the spatial mean (periodic Burgers)")
    drift = np.abs(u.mean(axis=-1) - u[:, :1].mean(axis=-1)).max(axis=1)
    print(f"max |mean(u(t)) - mean(u(0))| over samples: median {np.median(drift):.2e}, max {drift.max():.2e}")

    print("== 4. plots -> runs/verify_samples.png")
    show = [0, 1, n // 2, n - 1]
    fig, axes = plt.subplots(2, len(show), figsize=(4 * len(show), 6.5), constrained_layout=True)
    for col, i in enumerate(show):
        for k in (0, len(t) // 4, len(t) // 2, len(t) - 1):
            axes[0, col].plot(x, u[i, k], lw=1.2, label=f"t={t[k]:.1f}")
        axes[0, col].set_title(f"PDEBench sample {i}")
        axes[0, col].legend(fontsize=7)
        lim = np.abs(u[i]).max()
        im = axes[1, col].pcolormesh(x, t, u[i], cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
        axes[1, col].set_xlabel("x")
        axes[1, col].set_ylabel("t")
        fig.colorbar(im, ax=axes[1, col])
    out = resolve("runs") / "verify_samples.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=80)
    print(out)


if __name__ == "__main__":
    main()
