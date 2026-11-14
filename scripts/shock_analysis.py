"""Why does the nu = 0.01 FNO miss the 5 % window rule? Diagnostics on the held-out test split.

Measured, for nu = 0.1 (served model) and nu = 0.01 (selected `sym` model), per output time:
  * error table: RMS of the true fluctuation, FNO RMSE and fluctuation rel. L2, solver floor
  * front mask: grid points where |du/dx| of PDEBench exceeds `--slope` (dilated by
    `--dilate` cells). Share of the domain vs share of the FNO error energy inside it.
  * front width in grid cells: (jump across the steepest descent) / max|du/dx| / dx
  * front position error: steepest-descent location, FNO vs PDEBench, in cells
  * spectrum: share of the true fluctuation energy above k = 16 and k = 32
Plot: one nu = 0.01 test simulation zoomed on its strongest front.

Usage: python scripts/shock_analysis.py
Writes artifacts/shock_analysis.json and runs/shock_front_nu0.01.png
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

MODELS = {"0.1": "fno_burgers_nu0.1.pt", "0.01": "fno_burgers_nu0.01_sym.pt"}


def load(nu: str):
    art = resolve("artifacts")
    ck = torch.load(art / MODELS[nu], weights_only=False)
    model = FNO1d(**ck["model"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    d = np.load(art / f"test_samples_nu{nu}.npz")
    u0, ref, x = d["u"][:, 0].astype(np.float64), d["u"][:, 1:].astype(np.float64), d["x"]
    with torch.no_grad():
        inp = make_inputs(u0, x)[..., : ck["model"]["in_channels"]]
        fno = model(inp).numpy().transpose(0, 2, 1).astype(np.float64)  # (S, T, N)
    evt = json.loads((art / f"error_vs_time_nu{nu}{'_sym' if nu == '0.01' else ''}.json").read_text())
    return u0, ref, fno, x, d["t"][1:], d["indices"], evt


def ddx(u: np.ndarray) -> np.ndarray:
    """Centred periodic difference, (u[j+1] - u[j-1]) / (2 dx)."""
    n = u.shape[-1]
    return (np.roll(u, -1, -1) - np.roll(u, 1, -1)) * n / 2


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for s in range(1, r + 1):
        out |= np.roll(mask, s, -1) | np.roll(mask, -s, -1)
    return out


def front_stats(ref, fno, slope, r):
    n = ref.shape[-1]
    g = ddx(ref)
    mask = dilate(np.abs(g) > slope, r)
    err2 = (fno - ref) ** 2
    area = mask.mean(axis=(0, 2))
    err_share = (err2 * mask).sum(axis=(0, 2)) / err2.sum(axis=(0, 2))
    # steepest descent per simulation and time: width and FNO position offset
    j = np.argmin(g, axis=-1)                                   # (S, T)
    steep = -np.take_along_axis(g, j[..., None], -1)[..., 0]    # max |du/dx| at the front
    idx = (j[..., None] + np.arange(-6, 7)) % n
    seg = np.take_along_axis(ref, idx, -1)
    jump = seg.max(-1) - seg.min(-1)                            # jump across +-6 cells
    width_cells = jump / np.maximum(steep, 1e-12) * n
    g_f = ddx(fno)
    win = (j[..., None] + np.arange(-10, 11)) % n               # look for the FNO front nearby
    jf = np.take_along_axis(win, np.argmin(np.take_along_axis(g_f, win, -1), -1)[..., None], -1)[..., 0]
    offset = (jf - j + n // 2) % n - n // 2
    strong = steep > slope                                       # only real fronts
    return {
        "front_area_share": area.tolist(),
        "front_error_share": err_share.tolist(),
        "n_fronts": strong.sum(0).tolist(),
        "front_width_cells_median": [float(np.median(width_cells[:, k][strong[:, k]])) if strong[:, k].any()
                                     else None for k in range(ref.shape[1])],
        "front_offset_abs_cells_mean": [float(np.abs(offset[:, k][strong[:, k]]).mean()) if strong[:, k].any()
                                        else None for k in range(ref.shape[1])],
        "front_offset_nonzero_share": [float((offset[:, k][strong[:, k]] != 0).mean()) if strong[:, k].any()
                                       else None for k in range(ref.shape[1])],
    }


def spectrum_share(ref, u0, k_cut):
    f = ref - u0.mean(-1)[:, None, None]
    e = np.abs(np.fft.rfft(f, axis=-1)) ** 2
    return (e[..., k_cut + 1 :].sum(axis=(0, 2)) / e.sum(axis=(0, 2))).tolist()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--slope", type=float, default=5.0, help="|du/dx| threshold for a front")
    p.add_argument("--dilate", type=int, default=3)
    args = p.parse_args()
    out = {"slope_threshold": args.slope, "dilate_cells": args.dilate, "models": MODELS}
    data = {}
    for nu in MODELS:
        u0, ref, fno, x, t, idx, evt = load(nu)
        data[nu] = (u0, ref, fno, x, t, idx)
        out[nu] = {"t": t.tolist(), "rms_fluct": evt["rms_fluct_pdebench"],
                   "fno_rmse": evt["rmse"]["fno_vs_pdebench"],
                   "fno_fluct_rel_l2": evt["fluct_rel_l2"]["fno_vs_pdebench"],
                   "solver_fluct_rel_l2": evt["fluct_rel_l2"]["solver_vs_pdebench"],
                   "energy_above_k16": spectrum_share(ref, u0, 16),
                   "energy_above_k32": spectrum_share(ref, u0, 32),
                   **front_stats(ref, fno, args.slope, args.dilate)}
    (resolve("artifacts") / "shock_analysis.json").write_text(json.dumps(out, indent=2))

    def row(label, key, fmt, nu):
        vals = out[nu][key]
        return f"  {label:34s}" + "".join(fmt(v) if v is not None else "      -" for v in vals)

    pct = lambda v: f"{100 * v:7.1f}"  # noqa: E731
    for nu in MODELS:
        print(f"\nnu = {nu}  (model {MODELS[nu]}, 200 test simulations, 256 grid, front: |du/dx| > {args.slope})")
        print("  " + " " * 34 + "".join(f"{tt:7.1f}" for tt in out[nu]["t"]))
        print(row("RMS true fluctuation", "rms_fluct", lambda v: f"{v:7.3f}", nu))
        print(row("FNO RMSE", "fno_rmse", lambda v: f"{v:7.4f}", nu))
        print(row("FNO fluct rel L2 %", "fno_fluct_rel_l2", pct, nu))
        print(row("solver fluct rel L2 %", "solver_fluct_rel_l2", pct, nu))
        print(row("true energy above k=16 %", "energy_above_k16", pct, nu))
        print(row("true energy above k=32 %", "energy_above_k32", pct, nu))
        print(row("# test sims with a front", "n_fronts", lambda v: f"{v:7d}", nu))
        print(row("front area share %", "front_area_share", pct, nu))
        print(row("FNO error energy at fronts %", "front_error_share", pct, nu))
        print(row("front width, cells (median)", "front_width_cells_median", lambda v: f"{v:7.1f}", nu))
        print(row("|front offset| FNO, cells (mean)", "front_offset_abs_cells_mean", lambda v: f"{v:7.2f}", nu))
        print(row("fronts with offset != 0 %", "front_offset_nonzero_share", pct, nu))

    # plot: nu = 0.01 test simulation with the strongest front at t = 0.6
    u0, ref, fno, x, t, idx = data["0.01"]
    from pde_service.solver.burgers import solve_burgers
    k6 = int(np.argmin(np.abs(t - 0.6)))
    s = int(np.argmax(-ddx(ref)[:, k6].min(-1)))
    sol = solve_burgers(u0[s], t, 0.01)
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.9), constrained_layout=True)
    axes[0].plot(x, u0[s], "gray", lw=1, label="u0")
    for k, c in [(0, "C0"), (k6, "C1"), (int(np.argmin(np.abs(t - 1.2))), "C2")]:
        axes[0].plot(x, ref[s, k], c=c, lw=2, alpha=0.5, label=f"PDEBench t={t[k]:.1f}")
        axes[0].plot(x, fno[s, k], c=c, ls="--", lw=1)
    axes[0].set(title=f"nu=0.01 test sim #{idx[s]} (dashed = FNO)", xlabel="x")
    axes[0].legend(fontsize=7)
    for ax, k in zip(axes[1:], [0, k6, int(np.argmin(np.abs(t - 1.2)))]):
        j = int(np.argmin(ddx(ref[s, k])))
        w = (np.arange(j - 14, j + 15)) % len(x)
        xx = x[j] + (np.arange(-14, 15)) / len(x)
        ax.plot(xx, ref[s, k, w], "o-", c="k", lw=2, ms=3.5, alpha=0.6, label="PDEBench (grid points)")
        ax.plot(xx, sol[k, w], ":", c="C2", lw=2, label="reference solver")
        ax.plot(xx, fno[s, k, w], "s--", c="C3", ms=3, lw=1.2, label="FNO (sym)")
        ax2 = ax.twinx()
        ax2.bar(xx, np.abs(fno[s, k, w] - ref[s, k, w]), width=0.8 / len(x), color="C3", alpha=0.18)
        ax2.set_ylabel("|FNO - PDEBench|", color="C3", fontsize=8)
        ax.set(title=f"zoom on front, t={t[k]:.1f} (+-14 cells)", xlabel="x")
        ax.legend(fontsize=7, loc="lower left")
    fig.savefig(resolve("runs") / "shock_front_nu0.01.png", dpi=80)
    print(f"\nplot: test sim #{idx[s]} -> runs/shock_front_nu0.01.png")


if __name__ == "__main__":
    main()
