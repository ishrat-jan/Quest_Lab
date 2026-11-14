"""Absolute vs relative error over time on the held-out test split, and ripple levels.

For each output time, over all test simulations (model grid):
  signal : RMS of the PDEBench field u and of its fluctuation u - mean(u0)
  error  : RMSE and pooled fluctuation rel. L2 for FNO vs PDEBench, solver vs PDEBench,
           FNO vs solver
  ripple : RMS of the Fourier components with wavenumber >= k_ripple in PDEBench, FNO and
           solver fields. At late times such components are physically negligible
           (diffusion damps mode k by exp(-D (2 pi k)^2 t)), so their level measures
           numerical ripple.

Usage: python scripts/error_vs_time.py --nu 0.1 [--k-ripple 3]
Writes artifacts/error_vs_time_nu<nu>.json and runs/error_vs_time_nu<nu>.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pde_service.config import resolve  # noqa: E402
from pde_service.data.dataset import make_inputs  # noqa: E402
from pde_service.models.fno import FNO1d  # noqa: E402
from pde_service.solver.burgers import solve_burgers  # noqa: E402


def rms(a: np.ndarray) -> np.ndarray:
    """RMS over samples and space, per time. a: (S, T, N) -> (T,)"""
    return np.sqrt(np.mean(a**2, axis=(0, 2)))


def high_pass(a: np.ndarray, k_min: int) -> np.ndarray:
    a_hat = np.fft.rfft(a, axis=-1)
    a_hat[..., :k_min] = 0
    return np.fft.irfft(a_hat, n=a.shape[-1], axis=-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nu", default="0.1")
    p.add_argument("--k-ripple", type=int, default=3)
    args = p.parse_args()
    art = resolve("artifacts")
    ck = torch.load(art / f"fno_burgers_nu{args.nu}.pt", weights_only=False)
    model = FNO1d(**ck["model"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    d = np.load(art / f"test_samples_nu{args.nu}.npz")
    u, x, t = d["u"].astype(np.float64), d["x"], d["t"][1:]
    u0, ref = u[:, 0], u[:, 1:]                                   # ref: (S, T, N)
    nu = float(ck["nu"])
    with torch.no_grad():
        fno = model(make_inputs(u0, x)).numpy().transpose(0, 2, 1).astype(np.float64)
    sol = np.stack([solve_burgers(a, t, nu) for a in u0])
    mean = u0.mean(axis=1)[:, None, None]

    def fluct_rel(a, b):
        return np.linalg.norm(a - b, axis=(0, 2)) / np.linalg.norm(b - mean, axis=(0, 2))

    res = {
        "nu": nu, "n_test": len(u), "t": t.tolist(), "k_ripple": args.k_ripple,
        "rms_u_pdebench": rms(ref).tolist(),
        "rms_fluct_pdebench": rms(ref - mean).tolist(),
        "rmse": {"fno_vs_pdebench": rms(fno - ref).tolist(), "solver_vs_pdebench": rms(sol - ref).tolist(),
                 "fno_vs_solver": rms(fno - sol).tolist()},
        "fluct_rel_l2": {"fno_vs_pdebench": fluct_rel(fno, ref).tolist(),
                         "solver_vs_pdebench": fluct_rel(sol, ref).tolist(),
                         "fno_vs_solver": fluct_rel(fno, sol).tolist()},
        "ripple_rms": {"pdebench": rms(high_pass(ref, args.k_ripple)).tolist(),
                       "fno": rms(high_pass(fno, args.k_ripple)).tolist(),
                       "solver": rms(high_pass(sol, args.k_ripple)).tolist()},
    }
    (art / f"error_vs_time_nu{args.nu}.json").write_text(json.dumps(res, indent=2))

    print(f"nu={nu}, {len(u)} held-out test simulations, grid {u.shape[-1]}")
    print("   t | RMS u    RMS fluct | RMSE FNO-PDE  SOL-PDE   FNO-SOL | fluct relL2 FNO-PDE SOL-PDE FNO-SOL"
          f" | ripple k>={args.k_ripple}: PDE      FNO      SOL")
    r = res
    for i, tt in enumerate(t):
        print(f" {tt:.1f} | {r['rms_u_pdebench'][i]:.2e} {r['rms_fluct_pdebench'][i]:.2e} | "
              f"{r['rmse']['fno_vs_pdebench'][i]:.2e}  {r['rmse']['solver_vs_pdebench'][i]:.2e}  "
              f"{r['rmse']['fno_vs_solver'][i]:.2e} |  {r['fluct_rel_l2']['fno_vs_pdebench'][i]:6.1%}  "
              f"{r['fluct_rel_l2']['solver_vs_pdebench'][i]:6.1%}  {r['fluct_rel_l2']['fno_vs_solver'][i]:6.1%} | "
              f"           {r['ripple_rms']['pdebench'][i]:.1e}  {r['ripple_rms']['fno'][i]:.1e}  "
              f"{r['ripple_rms']['solver'][i]:.1e}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    ax[0].semilogy(t, r["rms_fluct_pdebench"], "k-", lw=2.5, label="RMS true fluctuation (PDEBench)")
    for k, ls in [("fno_vs_pdebench", "o-"), ("solver_vs_pdebench", "s-"), ("fno_vs_solver", "^--")]:
        ax[0].semilogy(t, r["rmse"][k], ls, ms=4, label=f"RMSE {k.replace('_', ' ')}")
    ax[0].set(xlabel="t", title="signal size vs absolute error")
    ax[0].legend(fontsize=8)
    for k, ls in [("fno_vs_pdebench", "o-"), ("solver_vs_pdebench", "s-"), ("fno_vs_solver", "^--")]:
        ax[1].semilogy(t, r["fluct_rel_l2"][k], ls, ms=4, label=k.replace("_", " "))
    ax[1].set(xlabel="t", title="fluctuation relative L2 = RMSE / RMS fluct")
    ax[1].legend(fontsize=8)
    for k, ls in [("pdebench", "k-"), ("fno", "o-"), ("solver", "s-")]:
        ax[2].semilogy(t, r["ripple_rms"][k], ls, ms=4, label=k)
    ax[2].set(xlabel="t", title=f"ripple: RMS of modes k >= {args.k_ripple}")
    ax[2].legend(fontsize=8)
    fig.savefig(resolve("runs") / f"error_vs_time_nu{args.nu}.png", dpi=80)


if __name__ == "__main__":
    main()
