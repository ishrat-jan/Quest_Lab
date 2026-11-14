"""Does the convection (nonlinear) term matter at nu = 0.1, and does our solver get it right?

Compares PDEBench trajectories with variants of the equation
    u_t + c * u u_x = D u_xx,   D = nu / pi
  full Burgers         c = 1
  no convection        c = 0                 (pure diffusion, exact in Fourier space)
  mean advection only  u_t + mean(u) u_x = D u_xx   (linear: translation, no steepening; exact)
  half / double        c = 0.5, 2            (via scaling: c*u solves standard Burgers)
The error is measured on the fluctuation u - mean(u) (the mean is exact in every variant),
pooled over samples, per snapshot time.

Also reports, from the PDEBench snapshots themselves, the term-size ratio
    R(t) = || (u - mean) u_x || / || D u_xx ||
(nonlinear steepening vs diffusion; the mean-advection part ū u_x is a pure translation).

Usage: python scripts/convection_controls.py --file data/pdebench/burgers_nu0.1_n1200_stride10.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pde_service.config import resolve  # noqa: E402
from pde_service.solver.burgers import solve_burgers  # noqa: E402


def linear_solution(u0: np.ndarray, t: np.ndarray, d: float, speed: float) -> np.ndarray:
    """Exact solution of u_t + speed * u_x = d * u_xx on the periodic unit domain."""
    n = u0.size
    k = 2 * np.pi * np.fft.rfftfreq(n, d=1.0 / n)
    u_hat = np.fft.rfft(u0)
    return np.stack([np.fft.irfft(u_hat * np.exp((-d * k**2 - 1j * speed * k) * tt), n=n) for tt in t])


def spectral_derivatives(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = u.shape[-1]
    k = 2 * np.pi * np.fft.rfftfreq(n, d=1.0 / n)
    u_hat = np.fft.rfft(u, axis=-1)
    return np.fft.irfft(1j * k * u_hat, n=n, axis=-1), np.fft.irfft(-(k**2) * u_hat, n=n, axis=-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()
    with h5py.File(args.file) as f:
        u, t, nu = f["u"][:], f["t"][:], float(f.attrs["nu"])
    d = nu / np.pi
    idx = np.linspace(0, len(u) - 1, args.n).astype(int)
    ref = u[idx, 1:].astype(np.float64)          # (S, T, N) at t[1:]
    mean = u[idx, 0].mean(axis=-1)               # conserved mean, (S,)
    fluct = ref - mean[:, None, None]

    variants = {
        "full Burgers (c=1)": lambda u0, m: solve_burgers(u0, t[1:], nu),
        "no convection (c=0)": lambda u0, m: linear_solution(u0, t[1:], d, 0.0),
        "mean advection only": lambda u0, m: linear_solution(u0, t[1:], d, m),
        "half convection (c=0.5)": lambda u0, m: solve_burgers(0.5 * u0, t[1:], nu) / 0.5,
        "double convection (c=2)": lambda u0, m: solve_burgers(2.0 * u0, t[1:], nu) / 2.0,
    }
    result = {"nu": nu, "D": d, "samples": idx.tolist(), "t": t[1:].tolist(), "fluct_rel_l2": {}}
    for name, fn in variants.items():
        pred = np.stack([fn(u[i, 0].astype(np.float64), m) for i, m in zip(idx, mean)])
        err = np.linalg.norm((pred - ref).reshape(len(idx), len(t) - 1, -1), axis=(0, 2)) / np.linalg.norm(
            fluct.reshape(len(idx), len(t) - 1, -1), axis=(0, 2))
        result["fluct_rel_l2"][name] = err.tolist()
        print(f"{name:26s} " + " ".join(f"{e:8.2e}" for e in err[:6]) + "  ...  t=2.0: " + f"{err[-1]:.2e}")

    # term-size ratio from the PDEBench data itself (full 1024 grid, include t=0)
    all_u = u[idx].astype(np.float64)
    ux, uxx = spectral_derivatives(all_u)
    fl = all_u - mean[:, None, None]
    ratio = np.linalg.norm(fl * ux, axis=(0, 2)) / np.linalg.norm(d * uxx, axis=(0, 2))
    energy = (fl**2).mean(axis=(0, 2))
    result["nonlinear_over_diffusion"] = ratio.tolist()
    result["fluct_energy"] = (energy / energy[0]).tolist()
    print("t                          " + " ".join(f"{tt:8.1f}" for tt in t[1:7]))
    print("R = |u' u_x| / |D u_xx|    " + " ".join(f"{r:8.2f}" for r in ratio[1:7]) + f"   (t=0: {ratio[0]:.2f})")
    print("fluct energy / E(0)        " + " ".join(f"{e:8.2e}" for e in energy[1:7] / energy[0]))

    out = resolve("artifacts") / f"convection_controls_nu{nu}.json"
    out.write_text(json.dumps(result, indent=2))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for name, err in result["fluct_rel_l2"].items():
        ax[0].semilogy(t[1:], err, marker="o", ms=3, label=name)
    ax[0].set(xlabel="t", ylabel="fluctuation rel L2 vs PDEBench", title=f"Equation variants vs PDEBench, nu={nu}")
    ax[0].legend(fontsize=8)
    ax[1].semilogy(t, ratio, marker="o", ms=3, label="|u' u_x| / |D u_xx|")
    ax[1].semilogy(t, energy / energy[0], marker="s", ms=3, label="fluctuation energy / E(0)")
    ax[1].axhline(1, color="gray", lw=0.8)
    ax[1].set(xlabel="t", title="PDEBench: nonlinear vs diffusion term, energy decay")
    ax[1].legend(fontsize=8)
    fig.savefig(resolve("runs") / "convection_controls.png", dpi=80)
    print(out)


if __name__ == "__main__":
    main()
