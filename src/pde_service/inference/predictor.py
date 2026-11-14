"""Load a trained FNO checkpoint, validate user inputs, and produce predictions.

Kept free of web-framework code so it can be unit-tested directly.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from pde_service.data.dataset import make_inputs
from pde_service.inference.initial_conditions import spectral_energy_fraction_above
from pde_service.metrics import all_metrics
from pde_service.models.fno import FNO1d
from pde_service.solver.burgers import SolverError, solve_burgers

log = logging.getLogger(__name__)


class InputError(ValueError):
    """User input that the model cannot (or must not) be applied to."""


class ModelNotAvailable(RuntimeError):
    pass


class Predictor:
    def __init__(self, nu: float, checkpoint: Path, test_samples: Path | None, limits: dict):
        if not checkpoint.exists():
            raise ModelNotAvailable(f"checkpoint {checkpoint} not found; train the model first")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if abs(ckpt["nu"] - nu) > 1e-12:
            raise ModelNotAvailable(f"{checkpoint} was trained for nu={ckpt['nu']}, not {nu}")
        self.nu = float(ckpt["nu"])
        self.x = np.asarray(ckpt["x"])
        self.t_out = np.asarray(ckpt["t_out"])
        self.n = len(self.x)
        self.model = FNO1d(**ckpt["model"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.meta = {k: ckpt[k] for k in ("epoch", "val_rel_l2", "data_source", "train_u0_abs_max")}
        self.max_abs_u0 = float(limits["max_abs_u0"])
        self.warn_abs_u0 = float(limits["warn_abs_u0"])
        self.mean_range = tuple(limits["mean_range"])
        self.reliable_t_max = float(limits["reliable_t_max"][f"{self.nu:g}"])
        self.k_cut = int(limits["ood_k_cut"])
        self.ood_energy = float(limits["ood_energy_fraction"])
        self.test = None
        if test_samples is not None and test_samples.exists():
            d = np.load(test_samples)
            if d["u"].shape[-1] != self.n:
                raise ModelNotAvailable(f"{test_samples} grid does not match the checkpoint")
            self.test = {"u": d["u"], "indices": d["indices"]}  # u: (S, K+1, n), [:, 0] = u0
        log.info("loaded FNO for nu=%s (n=%d, K=%d, %s test samples)", self.nu, self.n,
                 len(self.t_out), 0 if self.test is None else len(self.test["u"]))

    # ---- validation -------------------------------------------------------
    def validate(self, u0: np.ndarray) -> list[str]:
        """Raise InputError for unusable input; return warnings for usable-but-unusual input."""
        if u0.ndim != 1:
            raise InputError("u0 must be a 1D list of numbers")
        if u0.size != self.n:
            raise InputError(f"u0 must have exactly {self.n} points (uniform periodic grid on [0,1)), got {u0.size}")
        if not np.all(np.isfinite(u0)):
            raise InputError("u0 contains NaN or infinite values")
        peak = float(np.abs(u0).max())
        if peak > self.max_abs_u0:
            raise InputError(
                f"max|u0| = {peak:.3g} exceeds the supported range {self.max_abs_u0:g} "
                "(outside the amplitude range of the training data)"
            )
        warnings = []
        if peak < 1e-8:
            warnings.append("u0 is identically zero; the exact solution is u = 0")
        elif peak > self.warn_abs_u0:
            warnings.append(f"max|u0| = {peak:.3g} is larger than ~99% of training initial conditions "
                            f"(> {self.warn_abs_u0:g}); expect lower accuracy")
        mean = float(u0.mean())
        if not self.mean_range[0] <= mean <= self.mean_range[1]:
            warnings.append(f"mean(u0) = {mean:.3g} is outside the training range "
                            f"[{self.mean_range[0]:g}, {self.mean_range[1]:g}]; the mean sets the advection speed")
        hf = spectral_energy_fraction_above(u0, self.k_cut)
        if hf > self.ood_energy:
            warnings.append(
                f"{100 * hf:.1f}% of the initial energy is in wavenumbers > {self.k_cut}; "
                "training initial conditions are much smoother, so the FNO may be inaccurate. "
                "Compare against the reference solver."
            )
        return warnings

    def check_nu(self, nu: float) -> None:
        if abs(nu - self.nu) > 1e-9:
            raise InputError(f"nu={nu:g} is not supported; this model was trained only for nu={self.nu:g}")

    # ---- prediction -------------------------------------------------------
    def test_sample(self, i: int) -> tuple[np.ndarray, np.ndarray, int]:
        if self.test is None:
            raise InputError("held-out PDEBench test samples are not available on this server")
        if not 0 <= i < len(self.test["u"]):
            raise InputError(f"test sample index must be in [0, {len(self.test['u']) - 1}]")
        return self.test["u"][i, 0], self.test["u"][i, 1:], int(self.test["indices"][i])

    def predict(self, u0: np.ndarray, nu: float, run_solver: bool, reference: np.ndarray | None = None) -> dict:
        self.check_nu(nu)
        u0 = np.asarray(u0, dtype=np.float64)
        warnings = self.validate(u0)

        t0 = time.perf_counter()
        with torch.no_grad():
            pred = self.model(make_inputs(u0[None], self.x))[0].numpy().T  # (K, n)
        fno_ms = 1000 * (time.perf_counter() - t0)
        if not np.all(np.isfinite(pred)):
            raise InputError("the model produced non-finite values for this input; it is outside the supported range")
        out = {
            "nu": self.nu,
            "x": self.x.tolist(),
            "t": [0.0, *self.t_out.tolist()],
            "u0": u0.tolist(),
            "fno": [u0.tolist(), *pred.tolist()],  # row 0 is the given initial condition
            "reliable_t_max": self.reliable_t_max,  # later snapshots: low confidence
            "timing_ms": {"fno": round(fno_ms, 2)},
            "warnings": warnings,
        }
        if reference is not None:
            out["pdebench_reference"] = [u0.tolist(), *np.asarray(reference).tolist()]
            out["metrics_vs_pdebench"] = all_metrics(pred[None], np.asarray(reference)[None])
        if run_solver:
            t0 = time.perf_counter()
            try:
                sol = solve_burgers(u0, self.t_out, self.nu)
            except SolverError as exc:
                warnings.append(f"reference solver failed: {exc}")
            else:
                out["timing_ms"]["solver"] = round(1000 * (time.perf_counter() - t0), 2)
                out["solver"] = [u0.tolist(), *sol.tolist()]
                out["metrics_vs_solver"] = all_metrics(pred[None], sol[None])
        log.info("predict n=%d solver=%s ref=%s warnings=%d fno_ms=%.1f", self.n, run_solver,
                 reference is not None, len(warnings), fno_ms)
        return out
