"""Independent numerical reference solver for the 1D viscous Burgers equation.

    u_t + (u^2 / 2)_x = D * u_xx,   x in [0, 1) periodic,   D = nu / pi

(The D = nu/pi convention follows the PDEBench generator; it is verified against
the dataset in tests/test_solver_vs_pdebench.py, not assumed.)

Method: Fourier pseudo-spectral in space with 2/3-rule dealiasing, integrating-factor
RK4 in time (diffusion integrated exactly), adaptive CFL time step. This is
deliberately a different method from the neural operator so that it can act as an
independent check on user-supplied initial conditions.
"""

from __future__ import annotations

import numpy as np


class SolverError(RuntimeError):
    pass


def solve_burgers(
    u0: np.ndarray,
    t_out: np.ndarray,
    nu: float,
    cfl: float = 0.4,
    max_steps: int = 200_000,
) -> np.ndarray:
    """Integrate from t=0 and return u at each time in `t_out`.

    u0: (N,) initial condition on the uniform grid x_j = j/N (or cell centres; only spacing matters)
    t_out: (K,) strictly increasing, non-negative output times
    returns: (K, N) float64
    """
    u0 = np.asarray(u0, dtype=np.float64)
    t_out = np.asarray(t_out, dtype=np.float64)
    if u0.ndim != 1 or u0.size < 8:
        raise ValueError("u0 must be a 1D array with at least 8 points")
    if not np.all(np.isfinite(u0)):
        raise ValueError("u0 contains NaN or Inf")
    if t_out.ndim != 1 or np.any(t_out < 0) or np.any(np.diff(t_out) <= 0):
        raise ValueError("t_out must be non-negative and strictly increasing")
    if nu <= 0:
        raise ValueError("nu must be positive")

    n = u0.size
    diff = nu / np.pi
    k = 2.0 * np.pi * np.fft.rfftfreq(n, d=1.0 / n)  # angular wavenumbers
    dealias = np.abs(np.fft.rfftfreq(n, d=1.0 / n)) < (n / 3.0)
    k_max = k[dealias].max()

    def nonlinear(v_hat: np.ndarray) -> np.ndarray:
        # -(u^2/2)_x in Fourier space, dealiased
        u = np.fft.irfft(v_hat, n=n)
        return -0.5j * k * np.fft.rfft(u * u) * dealias

    u_hat = np.fft.rfft(u0) * dealias
    out = np.empty((t_out.size, n))
    t = 0.0
    steps = 0
    for i, t_target in enumerate(t_out):
        while t < t_target - 1e-12:
            umax = np.abs(np.fft.irfft(u_hat, n=n)).max()
            dt = cfl / (k_max * max(umax, 1e-8))
            dt = min(dt, t_target - t)
            e_half = np.exp(-diff * k**2 * dt / 2)
            e_full = e_half**2
            # integrating-factor RK4 (Cox & Matthews style, IF form)
            a = nonlinear(u_hat)
            b = nonlinear(e_half * (u_hat + 0.5 * dt * a))
            c = nonlinear(e_half * u_hat + 0.5 * dt * b)
            d = nonlinear(e_full * u_hat + dt * e_half * c)
            u_hat = e_full * u_hat + dt / 6.0 * (e_full * a + 2 * e_half * (b + c) + d)
            t += dt
            steps += 1
            if steps > max_steps:
                raise SolverError("exceeded max_steps; solution may be too steep for this grid")
        u = np.fft.irfft(u_hat, n=n)
        if not np.all(np.isfinite(u)):
            raise SolverError("solver diverged (non-finite values)")
        out[i] = u
    return out
