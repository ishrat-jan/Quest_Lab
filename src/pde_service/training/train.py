"""Train the FNO on the PDEBench subset and evaluate once on the held-out test split.

Usage:  python -m pde_service.training.train [--nu 0.1] [--epochs N]
Outputs:
    runs/<name>/history.json          per-epoch train/val loss
    artifacts/fno_burgers_nu<nu>.pt   best-on-validation checkpoint (self-describing)
    artifacts/metrics_nu<nu>.json     held-out test metrics (real PDEBench data)
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np
import torch

from pde_service.config import load_config, resolve
from pde_service.data.dataset import BurgersData, load_burgers
from pde_service.metrics import all_metrics, fluctuation_rel_l2, rel_l2
from pde_service.models.fno import FNO1d, count_parameters

log = logging.getLogger(__name__)


def rel_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = torch.linalg.vector_norm((pred - target).flatten(1), dim=1)
    return (diff / torch.linalg.vector_norm(target.flatten(1), dim=1)).mean()


@torch.no_grad()
def predict(model: torch.nn.Module, inputs: torch.Tensor, batch: int = 64) -> np.ndarray:
    model.eval()
    return torch.cat([model(inputs[i : i + batch]) for i in range(0, len(inputs), batch)]).numpy()


def scored(pred: np.ndarray, ref: np.ndarray, u0: np.ndarray) -> dict:
    """pred/ref (S, n, K) or (S, n); u0 (S, n)."""
    p = pred if pred.ndim == 2 else pred.transpose(0, 2, 1)  # put x last for the fluctuation metric
    r = ref if ref.ndim == 2 else ref.transpose(0, 2, 1)
    return all_metrics(pred, ref) | {"fluct_rel_l2": fluctuation_rel_l2(p, r, u0)}


def evaluate(model: torch.nn.Module, data: BurgersData) -> dict:
    """Metrics on the held-out test split, for the FNO and for the trivial baseline
    u(x, t) = mean(u0) (exact conservation, no dynamics). pred/ref are (S, n, K)."""
    pred = predict(model, data.test.inputs)
    ref = data.test.targets.numpy()
    u0 = data.test.inputs[..., 0].numpy()
    baseline = np.broadcast_to(u0.mean(axis=1)[:, None, None], ref.shape)
    per_time = [
        {"t": float(t), "fno": scored(pred[..., k], ref[..., k], u0),
         "mean_baseline": scored(baseline[..., k], ref[..., k], u0)}
        for k, t in enumerate(data.t_out)
    ]
    per_sample = [rel_l2(pred[i][None], ref[i][None]) for i in range(len(pred))]  # one value per simulation
    return {
        "overall": scored(pred, ref, u0),
        "mean_baseline": scored(baseline, ref, u0),
        "per_time": per_time,
        "per_sample_rel_l2": {
            "median": float(np.median(per_sample)),
            "p90": float(np.percentile(per_sample, 90)),
            "max": float(np.max(per_sample)),
        },
    }


def train(nu: str, epochs: int | None = None, cfg: dict | None = None, label: str | None = None) -> dict:
    cfg = cfg or load_config()
    dcfg, tcfg = cfg["data"], cfg["training"]
    mcfg = cfg["model"] | cfg.get("model_overrides", {}).get(nu, {})  # per-viscosity architecture
    epochs = epochs or tcfg["epochs"]
    torch.manual_seed(tcfg["seed"])
    torch.set_num_threads(tcfg["threads"])

    pattern = str(resolve(dcfg["local_dir"]) / f"burgers_nu{nu}_n{dcfg['n_samples']}_stride{dcfg['time_stride']}.h5")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"{pattern} not found; run scripts/download_subset.py --nu {nu}")
    data = load_burgers(Path(files[0]), mcfg["grid_size"], dcfg["split"], dcfg.get("output_stride", 1))
    k_out = len(data.t_out)
    log.info("train %s val %s test %s | n=%d K=%d", tuple(data.train.inputs.shape),
             tuple(data.val.inputs.shape), tuple(data.test.inputs.shape), len(data.x), k_out)

    overlap = {
        "train&val": len(set(data.train.indices) & set(data.val.indices)),
        "train&test": len(set(data.train.indices) & set(data.test.indices)),
        "val&test": len(set(data.val.indices) & set(data.test.indices)),
    }
    log.info("split by whole simulation (seed %d): overlap %s", dcfg["split"]["seed"], overlap)
    if any(overlap.values()):
        raise RuntimeError(f"data leakage: split overlap {overlap}")

    model_kwargs = {"modes": mcfg["modes"], "width": mcfg["width"], "n_layers": mcfg["n_layers"],
                    "in_channels": 2, "out_channels": k_out, "conserve_mean": mcfg.get("conserve_mean", False)}
    model = FNO1d(**model_kwargs)
    loss_fn = {"mse": torch.nn.functional.mse_loss, "rel_l2": rel_l2_loss}[tcfg.get("loss", "rel_l2")]
    log.info("FNO parameters: %d", count_parameters(model))
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, tcfg["scheduler_step"], tcfg["scheduler_gamma"])

    run_dir = resolve(tcfg["run_dir"]) / f"fno_nu{nu}"
    art_dir = resolve(tcfg["artifact_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = art_dir / f"fno_burgers_nu{nu}.pt"

    x_tr, y_tr = data.train.inputs, data.train.targets
    gen = torch.Generator().manual_seed(tcfg["seed"])
    history, best_val, t_start = [], float("inf"), time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_tr), generator=gen)
        total = 0.0
        for i in range(0, len(x_tr), tcfg["batch_size"]):
            b = perm[i : i + tcfg["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(x_tr[b]), y_tr[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        val = rel_l2(predict(model, data.val.inputs), data.val.targets.numpy())
        history.append({"epoch": epoch, "train_loss": total / len(x_tr), "val_rel_l2": val})
        if val < best_val:
            best_val = val
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model": model_kwargs,
                    "nu": data.nu,
                    "x": data.x.tolist(),
                    "t_out": data.t_out.tolist(),
                    "epoch": epoch,
                    "val_rel_l2": val,
                    "train_u0_abs_max": float(x_tr[..., 0].abs().max()),
                    "data_source": "PDEBench 1D Burgers (DaRUS doi:10.18419/darus-2986)",
                },
                ckpt_path,
            )
        if epoch % 10 == 0 or epoch == 1:
            log.info("epoch %4d  train loss %.4e  val rel L2 %.4e  best %.4e  (%.0fs)", epoch,
                     history[-1]["train_loss"], val, best_val, time.time() - t_start)
        (run_dir / "history.json").write_text(json.dumps(history))

    # held-out test evaluation, once, with the best-on-validation checkpoint
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    report = {
        "label": label or "REAL RESULT - PDEBench 1D Burgers held-out test split",
        "nu": data.nu,
        "data_file": Path(files[0]).name,
        "split_overlap": overlap,
        "loss": tcfg.get("loss", "rel_l2"),
        "split_sizes": {"train": len(data.train.indices), "val": len(data.val.indices),
                        "test": len(data.test.indices)},
        "test_sample_indices": data.test.indices.tolist(),
        "grid_size": len(data.x),
        "t_out": data.t_out.tolist(),
        "model": ckpt["model"] | {"parameters": count_parameters(model)},
        "best_epoch": ckpt["epoch"],
        "epochs": epochs,
        "best_val_rel_l2": ckpt["val_rel_l2"],
        "train_seconds": round(time.time() - t_start, 1),
        "hardware": f"CPU {platform.processor() or platform.machine()}, torch {torch.__version__}, {tcfg['threads']} threads",
        "test": evaluate(model, data),
    }
    (art_dir / f"metrics_nu{nu}.json").write_text(json.dumps(report, indent=2))
    # held-out test trajectories on the model grid, served by the API for real comparisons
    test_u = np.concatenate(
        [data.test.inputs[..., :1].numpy(), data.test.targets.numpy()], axis=-1
    ).transpose(0, 2, 1)  # (S, K+1, n), [:, 0] = u0
    np.savez_compressed(art_dir / f"test_samples_nu{nu}.npz", u=test_u.astype(np.float32),
                        indices=data.test.indices, t=np.concatenate([[0.0], data.t_out]), x=data.x)
    log.info("TEST overall: %s", report["test"]["overall"])
    return report


def reevaluate(nu: str) -> dict:
    """Recompute the test section of metrics_nu<nu>.json from the saved checkpoint
    (same data, split and checkpoint; training metadata is kept)."""
    cfg = load_config()
    dcfg, art_dir = cfg["data"], resolve(cfg["training"]["artifact_dir"])
    mcfg = cfg["model"] | cfg.get("model_overrides", {}).get(nu, {})
    path = resolve(dcfg["local_dir"]) / f"burgers_nu{nu}_n{dcfg['n_samples']}_stride{dcfg['time_stride']}.h5"
    data = load_burgers(path, mcfg["grid_size"], dcfg["split"], dcfg.get("output_stride", 1))
    ckpt = torch.load(art_dir / f"fno_burgers_nu{nu}.pt", weights_only=False)
    model = FNO1d(**ckpt["model"])
    model.load_state_dict(ckpt["state_dict"])
    metrics_path = art_dir / f"metrics_nu{nu}.json"
    report = json.loads(metrics_path.read_text())
    if report["test_sample_indices"] != data.test.indices.tolist():
        raise RuntimeError("test split differs from the one recorded at training time")
    report["test"] = evaluate(model, data)
    report["test_reevaluated"] = time.strftime("%Y-%m-%d %H:%M:%S") + " (evaluation code fix; same checkpoint)"
    metrics_path.write_text(json.dumps(report, indent=2))
    log.info("TEST overall: %s", report["test"]["overall"])
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nu", default="0.1")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--eval-only", action="store_true", help="re-evaluate the saved checkpoint on the test split")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.eval_only:
        reevaluate(args.nu)
    else:
        train(args.nu, args.epochs)


if __name__ == "__main__":
    main()
