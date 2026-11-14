"""FastAPI application: JSON API under /api, static frontend at /.

Run locally:  uvicorn pde_service.api.app:app --port 8000
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pde_service.api.schemas import ICRequest, PredictRequest
from pde_service.config import load_config, resolve
from pde_service.inference import initial_conditions as ics
from pde_service.inference.predictor import InputError, ModelNotAvailable, Predictor

log = logging.getLogger("pde_service.api")


def _artifact(cfg: dict, name: str) -> Path:
    return resolve(cfg["training"]["artifact_dir"]) / name


def load_predictors(cfg: dict) -> tuple[dict[float, Predictor], dict[float, str]]:
    predictors, errors = {}, {}
    for nu in cfg["api"]["supported_nu"]:
        try:
            predictors[nu] = Predictor(
                nu,
                _artifact(cfg, f"fno_burgers_nu{nu}.pt"),
                _artifact(cfg, f"test_samples_nu{nu}.npz"),
                cfg["api"],
            )
        except ModelNotAvailable as exc:
            log.error("model for nu=%s unavailable: %s", nu, exc)
            errors[nu] = str(exc)
    return predictors, errors


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    logging.basicConfig(level=cfg["api"]["log_level"], format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app.state.cfg = cfg
    app.state.predictors, app.state.load_errors = load_predictors(cfg)
    yield


app = FastAPI(title="PDE Solutions as a Service", version="0.1.0", lifespan=lifespan)


@app.exception_handler(InputError)
async def input_error(_: Request, exc: InputError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    msgs = []
    for e in exc.errors():
        loc = ".".join(str(p) for p in e["loc"] if p != "body")
        msgs.append(f"{loc}: {e['msg']}" if loc else e["msg"])
    return JSONResponse(status_code=422, content={"detail": "; ".join(msgs)})


@app.exception_handler(Exception)
async def unexpected_error(_: Request, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


def get_predictor(request: Request, nu: float) -> Predictor:
    preds: dict[float, Predictor] = request.app.state.predictors
    for supported, p in preds.items():
        if abs(supported - nu) < 1e-9:
            return p
    supported = request.app.state.cfg["api"]["supported_nu"]
    if any(abs(s - nu) < 1e-9 for s in supported):
        raise ModelNotAvailable(f"model for nu={nu:g} failed to load")
    raise InputError(f"nu={nu:g} is not supported; supported values: {', '.join(f'{s:g}' for s in supported)}")


@app.exception_handler(ModelNotAvailable)
async def model_unavailable(_: Request, exc: ModelNotAvailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/api/health")
def health(request: Request):
    preds = request.app.state.predictors
    return {
        "status": "ok" if preds else "degraded",
        "models_loaded": sorted(preds),
        "load_errors": {str(k): v for k, v in request.app.state.load_errors.items()},
    }


@app.get("/api/info")
def info(request: Request):
    cfg = request.app.state.cfg
    models = {}
    for nu, p in request.app.state.predictors.items():
        metrics_path = _artifact(cfg, f"metrics_nu{nu}.json")
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        models[str(nu)] = {
            "grid_size": p.n,
            "t_out": [0.0, *p.t_out.tolist()],
            "n_test_samples": 0 if p.test is None else len(p.test["u"]),
            "max_abs_u0": p.max_abs_u0,
            "reliable_t_max": p.reliable_t_max,
            "checkpoint": p.meta,
            "test_metrics": None if metrics is None else {
                "label": metrics["label"],
                "overall": metrics["test"]["overall"],
                "per_sample_rel_l2": metrics["test"]["per_sample_rel_l2"],
                "split_sizes": metrics["split_sizes"],
            },
        }
    return {
        "pde": "u_t + (u^2/2)_x = (nu/pi) u_xx, x in [0,1) periodic, t in [0,2]",
        "supported_nu": cfg["api"]["supported_nu"],
        "models": models,
    }


@app.post("/api/initial-condition")
def initial_condition(req: ICRequest, request: Request):
    p = get_predictor(request, req.nu)
    x = p.x
    ic = req.ic
    if ic.type == "sines":
        u0 = ics.sine_sum(x, [t.model_dump() for t in ic.terms])
    elif ic.type == "gaussian":
        u0 = ics.gaussian(x, ic.amp, ic.center, ic.width)
    else:
        u0, _, src = p.test_sample(ic.index)
        return {"x": x.tolist(), "u0": u0.tolist(), "test_index": ic.index, "pdebench_sample": src}
    return {"x": x.tolist(), "u0": np.asarray(u0).tolist()}


@app.post("/api/predict")
def predict(req: PredictRequest, request: Request):
    p = get_predictor(request, req.nu)
    if req.test_index is not None:
        u0, ref, src = p.test_sample(req.test_index)
        out = p.predict(u0, req.nu, req.run_solver, reference=ref)
        out["source"] = {"type": "pdebench_test", "test_index": req.test_index, "pdebench_sample": src}
    else:
        out = p.predict(np.asarray(req.u0, dtype=np.float64), req.nu, req.run_solver)
        out["source"] = {"type": "user"}
    return out


_frontend = resolve("frontend")
if _frontend.exists():
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")
