"""Download the configured PDEBench Burgers subset (resumable).

Usage:  python scripts/download_subset.py --nu 0.1 [--n-samples 1200]
"""

from __future__ import annotations

import argparse
import logging

from pde_service.config import load_config, resolve
from pde_service.data.download import download_subset


def main() -> None:
    cfg = load_config()["data"]
    p = argparse.ArgumentParser()
    p.add_argument("--nu", default="0.1")
    p.add_argument("--n-samples", type=int, default=cfg["n_samples"])
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    time_indices = list(range(0, cfg["n_time_total"], cfg["time_stride"]))
    out = resolve(cfg["local_dir"]) / f"burgers_nu{args.nu}_n{args.n_samples}_stride{cfg['time_stride']}.h5"
    download_subset(
        file_id=cfg["darus_file_ids"][args.nu],
        out_path=out,
        n_samples=args.n_samples,
        time_indices=time_indices,
        n_time=cfg["n_time_total"],
        n_x=cfg["n_x"],
        offset=cfg["tensor_offset"],
        nu=float(args.nu),
        dt=cfg["dt"],
        workers=args.workers,
    )
    print(out)


if __name__ == "__main__":
    main()
