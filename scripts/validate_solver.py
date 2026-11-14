"""Validate the numerical solver against stored PDEBench trajectories.

The solver may only be called a reference if it reproduces PDEBench. This checks the
equation form (D = nu/pi vs the alternative D = nu as a control), on the full 1024 grid
and on the 256-point grid the API uses. Writes artifacts/solver_validation_nu<nu>.json
and runs/solver_validation.png.

Usage: python scripts/validate_solver.py --file data/pdebench/burgers_nu0.1_n1200_stride10.h5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pde_service.config import resolve  # noqa: E402
from pde_service.solver.burgers import solve_burgers  # noqa: E402


def rel(a, b):
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()
    with h5py.File(args.file) as f:
        u, t, nu = f["u"][:], f["t"][:], float(f.attrs["nu"])
    idx = np.linspace(0, len(u) - 1, args.n).astype(int)  # spread over the file
    report = {"nu": nu, "n_samples": len(idx), "sample_indices": idx.tolist(), "t": t[1:].tolist(), "cases": {}}

    cases = {"D=nu/pi, N=1024": (nu, 1), "D=nu/pi, N=256": (nu, 4), "CONTROL D=nu, N=1024": (nu * np.pi, 1)}
    curves = {}
    for name, (nu_solver, stride) in cases.items():
        t0 = time.time()
        errs = np.array([rel(solve_burgers(u[i, 0, ::stride], t[1:], nu_solver), u[i, 1:, ::stride]) for i in idx])
        per_sample = errs.mean(axis=1)
        report["cases"][name] = {
            "rel_l2_mean": float(errs.mean()), "rel_l2_max": float(errs.max()),
            "per_sample_worst": float(per_sample.max()), "per_time_mean": errs.mean(axis=0).tolist(),
            "seconds_per_solve": (time.time() - t0) / len(idx),
        }
        curves[name] = errs.mean(axis=0)
        print(f"{name:22s} rel L2 mean {errs.mean():.2e}  max {errs.max():.2e}  "
              f"({(time.time() - t0) / len(idx):.2f} s/solve)")
    out = resolve("artifacts") / f"solver_validation_nu{nu}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for name, c in curves.items():
        ax[0].semilogy(t[1:], c, marker="o", ms=3, label=name)
    ax[0].set(xlabel="t", ylabel="mean rel L2 vs PDEBench", title=f"Solver vs PDEBench, nu={nu}, {len(idx)} samples")
    ax[0].legend(fontsize=8)
    i = idx[len(idx) // 2]
    s = solve_burgers(u[i, 0], t[1:], nu)
    for k in (0, 9, 19):
        ax[1].plot(u[i, k + 1], lw=2, alpha=0.5, label=f"PDEBench t={t[k + 1]:.1f}")
        ax[1].plot(s[k], "k--", lw=0.8)
    ax[1].set(title=f"sample {i}: PDEBench (colour) vs solver (dashed)", xlabel="grid index")
    ax[1].legend(fontsize=8)
    fig.savefig(resolve("runs") / "solver_validation.png", dpi=80)
    print(out)


if __name__ == "__main__":
    main()
