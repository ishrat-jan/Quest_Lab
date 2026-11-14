"""Inspect the layout of a remote PDEBench HDF5 file without downloading it.

Downloads only the first and last `--probe-bytes` bytes via HTTP range requests,
places them in a *sparse* local file of the full size, and lets h5py read the
metadata (names, shapes, dtypes, storage layout, byte offsets, attributes).

Usage:
    python scripts/inspect_remote_hdf5.py --file-id 268185 --out data/probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import httpx

DARUS_ACCESS = "https://darus.uni-stuttgart.de/api/access/datafile/{file_id}"


def fetch_range(client: httpx.Client, url: str, start: int, end: int) -> tuple[bytes, int]:
    r = client.get(url, headers={"Range": f"bytes={start}-{end}"})
    if r.status_code != 206:
        raise RuntimeError(f"expected HTTP 206 Partial Content, got {r.status_code}")
    total = int(r.headers["content-range"].rsplit("/", 1)[1])
    return r.content, total


def build_sparse_probe(file_id: int, out_dir: Path, probe_bytes: int) -> Path:
    url = DARUS_ACCESS.format(file_id=file_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"probe_{file_id}.h5"
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        head, total = fetch_range(client, url, 0, probe_bytes - 1)
        tail, _ = fetch_range(client, url, total - probe_bytes, total - 1)
    with open(path, "wb") as f:
        f.truncate(total)  # sparse: no disk used for the hole
        f.write(head)
        f.seek(total - probe_bytes)
        f.write(tail)
    return path


def describe(path: Path) -> dict:
    info: dict = {"file_size": path.stat().st_size, "attrs": {}, "datasets": {}}
    with h5py.File(path, "r") as f:
        info["attrs"] = {k: repr(v) for k, v in f.attrs.items()}

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                info["datasets"][name] = {
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                    "chunks": obj.chunks,
                    "compression": obj.compression,
                    "byte_offset": obj.id.get_offset(),  # None if chunked
                    "storage_bytes": obj.id.get_storage_size(),
                    "attrs": {k: repr(v) for k, v in obj.attrs.items()},
                }

        f.visititems(visit)
    return info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file-id", type=int, required=True)
    p.add_argument("--out", type=Path, default=Path("data/probe"))
    p.add_argument("--probe-bytes", type=int, default=65536)
    args = p.parse_args()
    path = build_sparse_probe(args.file_id, args.out, args.probe_bytes)
    info = describe(path)
    print(json.dumps(info, indent=2, default=str))
    (args.out / f"layout_{args.file_id}.json").write_text(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
