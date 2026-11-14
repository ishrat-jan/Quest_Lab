"""Initial-condition builders on the periodic grid x in [0, 1).

PDEBench 1D Burgers initial conditions are random sums of sine waves with low
integer wavenumbers (plus random windowing / sign flips); the presets here produce
inputs of a similar kind. All builders return a float64 array of shape (N,).
"""

from __future__ import annotations

import numpy as np


def grid(n: int) -> np.ndarray:
    """Cell centres, matching the PDEBench x-coordinate convention."""
    return (np.arange(n) + 0.5) / n


def sine_sum(x: np.ndarray, terms: list[dict]) -> np.ndarray:
    """sum_j amp_j * sin(2 pi k_j x + phase_j); k_j must be a positive integer (periodicity)."""
    u = np.zeros_like(x)
    for term in terms:
        k = term["k"]
        if int(k) != k or k < 1:
            raise ValueError(f"wavenumber k must be a positive integer for a periodic domain, got {k}")
        u += term["amp"] * np.sin(2 * np.pi * k * x + term.get("phase", 0.0))
    return u


def gaussian(x: np.ndarray, amp: float, center: float, width: float) -> np.ndarray:
    """Periodised Gaussian bump (sum over neighbouring images so u is smooth at x=0/1)."""
    if width <= 0:
        raise ValueError("width must be positive")
    return amp * sum(np.exp(-(((x - center + s) / width) ** 2)) for s in (-1.0, 0.0, 1.0))


def spectral_energy_fraction_above(u: np.ndarray, k_cut: int) -> float:
    """Fraction of the non-mean spectral energy in wavenumbers > k_cut."""
    e = np.abs(np.fft.rfft(u - u.mean())) ** 2
    total = e.sum()
    return 0.0 if total == 0 else float(e[k_cut + 1 :].sum() / total)
