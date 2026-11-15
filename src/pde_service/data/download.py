"""Download a subset of a PDEBench 1D Burgers HDF5 file using HTTP range requests.

The remote `tensor` dataset is a contiguous little-endian float32 array of shape
(n_samples_total, n_time_total, n_x) starting at `tensor_offset` bytes (verified with
scripts/inspect_remote_hdf5.py). One (sample, time) row is n_x * 4 bytes, so each
requested row is one small range request. Only the rows we need are fetched.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import httpx
import numpy as np

log = logging.getLogger(__name__)

DARUS_ACCESS = "https://darus.uni-stuttgart.de/api/access/datafile/{file_id}"
BYTES_PER_VALUE = 4


def row_byte_range(sample: int, k: int, n_time: int, n_x: int, offset: int) -> tuple[int, int]:
    """Inclusive byte range of tensor[sample, k, :]."""
    row = n_x * BYTES_PER_VALUE
    start = offset + (sample * n_time + k) * row
    return start, start + row - 1


def resolve_signed_url(file_id: int) -> str:
    """DaRUS redirects (303) to a pre-signed S3 URL; resolve it once and reuse it."""
    r = httpx.get(DARUS_ACCESS.format(file_id=file_id), follow_redirects=False, timeout=60)
    if r.status_code not in (302, 303) or "location" not in r.headers:
        raise RuntimeError(f"unexpected DaRUS response {r.status_code}")
    return r.headers["location"]


def download_subset(
    file_id: int,
    out_path: Path,
    n_samples: int,
    time_indices: list[int],
    n_time: int,
    n_x: int,
    offset: int,
    nu: float,
    dt: float,
    workers: int = 16,
) -> Path:
    """Fetch tensor[:n_samples, time_indices, :] into a local HDF5 file (resumable)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_t = len(time_indices)
    with h5py.File(out_path, "a") as f:
        if "u" not in f:
            f.create_dataset("u", shape=(n_samples, n_t, n_x), dtype="<f4")
            f.create_dataset("done", shape=(n_samples,), dtype=bool)
            f.create_dataset("t", data=np.asarray(time_indices, dtype=np.float64) * dt)
            f.create_dataset("x", data=(np.arange(n_x) + 0.5) / n_x)
            f.attrs.update(
                nu=nu,
                source=f"PDEBench 1D Burgers, DaRUS doi:10.18419/darus-2986 file id {file_id}",
                time_indices=time_indices,
            )
        if f["u"].shape != (n_samples, n_t, n_x):
            raise RuntimeError(f"{out_path} exists with a different shape; delete it to restart")
        done = f["done"][:]
    todo = [i for i in range(n_samples) if not done[i]]
    log.info("%d/%d samples already present, fetching %d", n_samples - len(todo), n_samples, len(todo))
    if not todo:
        return out_path

    url = resolve_signed_url(file_id)
    lock = threading.Lock()
    local = threading.local()

    def client() -> httpx.Client:
        if not hasattr(local, "c"):
            local.c = httpx.Client(timeout=60)
        return local.c

    def fetch_sample(i: int) -> tuple[int, np.ndarray]:
        rows = np.empty((n_t, n_x), dtype="<f4")
        for j, k in enumerate(time_indices):
            a, b = row_byte_range(i, k, n_time, n_x, offset)
            for attempt in range(5):
                try:
                    r = client().get(url, headers={"Range": f"bytes={a}-{b}"})
                    if r.status_code == 206 and len(r.content) == b - a + 1:
                        break
                    raise RuntimeError(f"HTTP {r.status_code}, {len(r.content)} bytes")
                except Exception as exc:  # network hiccup: retry
                    log.warning("sample %d row %d attempt %d failed: %s", i, k, attempt, exc)
            else:
                raise RuntimeError(f"giving up on sample {i} row {k}")
            rows[j] = np.frombuffer(r.content, dtype="<f4")
        return i, rows

    with h5py.File(out_path, "a") as f, ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(fetch_sample, i) for i in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            i, rows = fut.result()
            with lock:
                f["u"][i] = rows
                f["done"][i] = True
            if n % 50 == 0 or n == len(todo):
                f.flush()
                log.info("downloaded %d/%d samples", n, len(todo))
    return out_path
