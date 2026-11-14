"""Error metrics between a prediction and a reference.

Convention: axis 0 is the sample axis; everything after it (time, space) is one
sample's field. A 1D array is treated as a single sample.
    MSE    = mean((pred - ref)^2)              over all values
    RMSE   = sqrt(MSE)
    rel L2 = mean_i ||pred_i - ref_i||_2 / ||ref_i||_2   (per sample, then averaged)
"""

from __future__ import annotations

import numpy as np


def mse(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean((np.asarray(pred) - np.asarray(ref)) ** 2))


def rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, ref)))


def rel_l2(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-12) -> float:
    pred, ref = np.asarray(pred), np.asarray(ref)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {ref.shape}")
    p = pred.reshape(1, -1) if pred.ndim == 1 else pred.reshape(pred.shape[0], -1)
    r = ref.reshape(1, -1) if ref.ndim == 1 else ref.reshape(ref.shape[0], -1)
    return float(np.mean(np.linalg.norm(p - r, axis=1) / (np.linalg.norm(r, axis=1) + eps)))


def fluctuation_rel_l2(pred: np.ndarray, ref: np.ndarray, u0: np.ndarray) -> float:
    """||pred - ref|| / ||ref - mean(u0)|| pooled over ALL samples (not per sample).

    Periodic Burgers conserves the spatial mean, so `ref - mean(u0)` is the part of the
    solution that actually evolves. Pooling avoids dividing by near-zero fluctuations of
    individual, almost-constant samples. pred/ref: (S, ..., N); u0: (S, N).
    """
    pred, ref, u0 = np.asarray(pred), np.asarray(ref), np.asarray(u0)
    mean = u0.mean(axis=-1).reshape((-1,) + (1,) * (ref.ndim - 1))
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref - mean))


def all_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    return {"mse": mse(pred, ref), "rmse": rmse(pred, ref), "rel_l2": rel_l2(pred, ref)}
