# PDE Solutions as a Service: 1D Burgers + Fourier Neural Operator

A web application that predicts the solution of the 1D viscous Burgers equation from a
user-supplied initial condition, using a Fourier Neural Operator (FNO) trained on the
official **PDEBench** dataset, and checks the prediction against an independent,
validated numerical solver.

> **Deployment status:** not deployed by me. The repository contains a Dockerfile
> intended for Render (it honours `$PORT`). It has not been built locally (no Docker
> on the development machine), so the first real image build will be on the host.

## The problem

```
u_t + (u^2 / 2)_x = (nu / pi) u_xx,    x in [0, 1) periodic,   t in [0, 2]
```

This is the PDEBench convention (diffusion coefficient `nu/pi`). It was **verified, not
assumed**: our own solver reproduces stored PDEBench trajectories only with `D = nu/pi`
(rel. L2 ~1e-3), while `D = nu` is ~130x worse (`scripts/validate_solver.py`).

## What the model does

| | |
|---|---|
| Input  | `(B, 256, 2)` = `[u(x, 0), x]` on the grid `x_j = (j + 0.5) / 256` |
| Output | `(B, 256, 10)` = `u(x, t)` at `t = 0.2, 0.4, ..., 2.0`; **supported window t <= 1.0** (later times low confidence, see Results) |
| Viscosity | **one model per nu**; only `nu = 0.1` is trained. Other values are rejected (HTTP 422). |
| Architecture | FNO1d: lift -> 4 x (spectral conv on 16 lowest Fourier modes + pointwise conv) -> projection; 551k real parameters |
| Physics constraint | the spatial mean of each output is set to mean(u0) (exact conservation law of periodic Burgers) |
| Loss | MSE (with the mean fixed, this acts on the evolving fluctuation) |

Why these choices:
* **Direct map u0 -> all snapshots** (not autoregressive): one forward pass gives the
  space-time field, no error accumulation.
* **256 grid** (stride 4 of PDEBench's 1024): the nu = 0.1 solutions are smooth; the
  reference solver gives the same accuracy at 256 and 1024 points.
* **MSE, not per-sample relative L2.** At nu = 0.1 the field decays quickly to its
  (conserved) mean, so a full-field relative loss rewards predicting the mean. A
  per-sample relative loss on the fluctuation would divide by near-zero fluctuations and
  fit numerical ripples in the data.

## Data

* Source: PDEBench 1D Burgers, DaRUS `doi:10.18419/darus-2986`, file
  `1D_Burgers_Sols_Nu0.1.hdf5` (8.23 GB, 10000 simulations x 201 times x 1024 points).
* The HDF5 layout was inspected **without downloading the file**
  (`scripts/inspect_remote_hdf5.py`: the first/last 64 KB placed in a sparse local file and
  read with h5py). `tensor` is contiguous little-endian float32 at byte offset 2048.
* Only what is needed is downloaded with HTTP range requests
  (`scripts/download_subset.py`): the first **1200 simulations**, time indices
  `0, 10, ..., 200` (t = 0, 0.1, ..., 2.0): ~100 MB instead of 8.2 GB.
* Verified (`scripts/verify_download.py`): shapes, 0 NaN/Inf, and a byte-offset check
  through h5py's own layout logic (whole samples 0/600/1199 bit-identical to the
  row-wise download); conservation of the mean.
* **Split by whole simulation**, seed 0: 900 train / 100 validation / 200 test; overlap
  0/0/0 (checked at training time; training aborts on any overlap). The test split is
  evaluated once, with the best-on-validation checkpoint.

## Independent validation: the reference solver

`src/pde_service/solver/burgers.py`: Fourier pseudo-spectral, 2/3 dealiasing,
integrating-factor RK4 (diffusion integrated exactly), adaptive CFL step.

| Check | Result |
|---|---|
| Exact Cole-Hopf solution | rel. L2 ~1e-12 |
| 30 PDEBench trajectories, 1024 and 256 points | mean rel. L2 1.0e-3, max abs. error <= 5e-3 |
| Control: `D = nu` | rel. L2 1.3e-1 (rejected) |
| Control: no convection / mean-advection only / c = 0.5 / c = 2 | fluctuation error 0.26-0.72 at t = 0.1 vs **3.4e-4** for full Burgers |

The convection controls (`scripts/convection_controls.py`) show the nonlinear term is
essential at nu = 0.1: in the PDEBench data `|u' u_x| / |D u_xx| ~ 0.55` for t = 0.1-0.4,
and dropping convection gives 72 % error already at t = 0.1. Energy is dissipated only by
diffusion, but convection (steepening) sets the rate.

**Known limitation of the comparison:** ~96 % of the fluctuation energy is gone by t = 0.5.
The stored PDEBench fields also contain small (~5e-4) ripples that do not decay, most
likely an artefact of PDEBench's own numerical scheme (observed, cause not verified). As a
result, relative errors on the late-time fluctuation are a few percent even for the
reference solver (5.4 % at t = 2 on the 200 test simulations; 6.9 % on the 30-sample check).
The excess ripple in PDEBench is small: RMS of modes k >= 3 is 1.1e-4 at t = 2 vs 4.9e-5 in
the solver.

## Results (real PDEBench, held-out test split)

From `artifacts/metrics_nu0.1.json` (200 held-out simulations, never used for training or
model selection; best-on-validation checkpoint, epoch 274/300; 55 min on a 2-core i3 CPU).

| Full space-time field (t = 0.2 ... 2.0) | **FNO** | Mean-only baseline u = mean(u0) |
|---|---|---|
| MSE | **9.06e-6** | 1.08e-2 |
| RMSE | **3.01e-3** | 1.04e-1 |
| Relative L2 | **0.46 %** | 25.4 % |
| Fluctuation rel. L2 (pooled, `‖pred−ref‖ / ‖ref−mean‖`) | **2.9 %** | 100 % |

Per simulation, relative L2: median 0.28 %, p90 1.11 %, max 2.33 %.

Fluctuation relative L2 per output time, FNO vs the reference solver's own agreement with
PDEBench (`artifacts/convection_controls_nu0.1.json`):

| t | 0.2 | 0.4 | 0.6 | 1.0 | 1.6 | 2.0 |
|---|---|---|---|---|---|---|
| FNO vs PDEBench | 2.1 % | 2.5 % | 2.5 % | 4.3 % | 9.0 % | 11.9 % |
| reference solver vs PDEBench | 0.04 % | 0.20 % | 0.44 % | 1.0 % | 3.2 % | 6.9 % |

Interpretation:
* The FNO clearly learned the dynamics: ~55x lower relative L2 than the mean-only baseline.
* It is ~50x less accurate than the numerical solver at early times; it is ~8x faster
  (16 ms vs 135 ms per prediction on the 256 grid). That is the surrogate trade-off.
* The late-time growth of the relative error is mostly the signal fading, not the error
  growing; but the late-time error is the FNO's own, not the data floor (see below).
* Generalisation gap: final train MSE ~1.7e-6 vs test MSE 9.1e-6 (mild overfitting).
* Out-of-distribution example: a k = 24 initial condition triggers the API warning and
  the FNO is 38 % off the solver. The warning is warranted.

### Supported time window (nu = 0.1): t <= 1.0

`scripts/error_vs_time.py` (200 test simulations, `artifacts/error_vs_time_nu0.1.json`):

| t | 0.2 | 0.6 | 1.0 | 1.4 | 2.0 |
|---|---|---|---|---|---|
| RMS of true fluctuation (PDEBench) | 0.235 | 0.112 | 0.062 | 0.036 | 0.017 |
| RMSE FNO vs PDEBench | 4.9e-3 | 2.8e-3 | 2.7e-3 | 2.7e-3 | 2.0e-3 |
| RMSE FNO vs solver | 4.9e-3 | 2.9e-3 | 2.7e-3 | 2.8e-3 | 2.1e-3 |
| RMSE solver vs PDEBench | 1.2e-4 | 5.0e-4 | 6.2e-4 | 6.0e-4 | 9.0e-4 |
| Fluct. rel. L2 FNO vs PDEBench | 2.1 % | 2.5 % | 4.3 % | 7.5 % | 11.9 % |
| Ripple RMS (modes k >= 3): PDEBench / FNO / solver | 3.6e-2 / 3.6e-2 / 3.6e-2 | 8.9e-3 / 8.7e-3 / 8.8e-3 | 2.1e-3 / 2.0e-3 / 2.1e-3 | 4.7e-4 / 7.6e-4 / 4.7e-4 | 1.1e-4 / 6.2e-4 / 4.9e-5 |

* The FNO's **absolute** error is roughly flat (2-5e-3) while the true fluctuation decays
  14x, so the **relative** error grows from 2 % to 12 %. Most of that growth is the
  denominator, but the error stops decreasing after t ~ 0.6 while the signal keeps decaying.
* The late-time error belongs to the FNO: the FNO is equally far from PDEBench and from
  the solver, while the solver and PDEBench agree 2-40x more closely.
* About 90 % of the FNO's late error energy is in the large-scale modes k = 1-2; ~10 % is
  a non-decaying ripple (k >= 3, ~6e-4 RMS, 13x the physical level at t = 2).
* k >= 3 content is *not* unphysical up to t ~ 1.2: nonlinear harmonics keep it alive, and
  PDEBench, FNO and solver agree there.
* **Window rule:** the largest output time with fluctuation rel. L2 <= 5 % vs PDEBench, i.e.
  **t <= 1.0**. Later snapshots are still returned, with `reliable_t_max` in the API
  response; the UI draws the boundary on the heatmaps and labels later slices as low
  confidence.

History notes, for transparency: (1) the per-simulation statistics were first computed
with a bug (a single field passed without a sample axis); fixed and recomputed from the
same checkpoint (`--eval-only`), regression test added; overall metrics were unaffected.
(2) Commit `094b560` accidentally contains an intermediate checkpoint (~epoch 20) because
it was made while training was running; no reported number comes from it, and it is
replaced by commit `1d49fac`.

## Application

* `GET /api/health`, `GET /api/info` (supported nu, grid, real test metrics)
* `POST /api/initial-condition`: presets (sine sum, periodic Gaussian, held-out PDEBench
  test sample)
* `POST /api/predict`: `{nu, u0 | test_index, run_solver}` -> FNO field, optional
  reference-solver field, PDEBench reference for test samples, MSE/RMSE/rel. L2, warnings
* Frontend at `/`: Plotly heatmaps (FNO, reference, |error|), time-slice plot, metrics,
  loading/error/warning states, JSON/CSV download, explanation of the PDE and the limits.

Input handling (thresholds from `scripts/data_stats.py` on the **training** ICs):

| Condition | Response |
|---|---|
| nu not trained, wrong grid length, NaN/Inf, malformed JSON | 422 with a readable message |
| max\|u0\| > 3.2 (training max 3.14) | 422 |
| max\|u0\| > 2.4 (training p99), mean(u0) outside [-1.6, 1.8], > 6 % energy above k = 16 | accepted **with a warning** |
| model produces non-finite output | 422 |

## Setup

Requires Python >= 3.10 (developed on 3.14; the Docker image uses 3.12).

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt && pip install -e .

# data (~100 MB via range requests) and checks
python scripts/download_subset.py --nu 0.1
python scripts/verify_download.py --file data/pdebench/burgers_nu0.1_n1200_stride10.h5 \
       --probe data/probe/probe_268185.h5        # probe from scripts/inspect_remote_hdf5.py --file-id 268185
python scripts/validate_solver.py --file data/pdebench/burgers_nu0.1_n1200_stride10.h5

# train (~20 min on a 2-core laptop CPU) -> artifacts/
python -m pde_service.training.train --nu 0.1

# run
uvicorn pde_service.api.app:app --port 8000      # open http://localhost:8000
pytest
```

The trained checkpoint, held-out test samples and metrics in `artifacts/` are committed,
so the app runs without downloading data or training.

### Docker / Render

```bash
docker build -t burgers-fno . && docker run -p 8000:8000 -e PORT=8000 burgers-fno
```

On Render: new Web Service from this repo, environment "Docker". Render sets `PORT`.

## Tests

* Unit: FNO shapes, resolution independence, mean conservation; solver vs exact
  Cole-Hopf, mean conservation, invalid inputs; byte-offset arithmetic; metrics; split
  disjointness.
* Data (skipped if data not downloaded): solver vs PDEBench.
* Integration: full API on an **untrained** fixture checkpoint (validation, status codes,
  shapes; no accuracy claims).
* **Synthetic smoke test** of the training pipeline (solver-generated data, labelled
  `SYNTHETIC SMOKE TEST`; says nothing about accuracy).

## Layout

```
configs/default.yaml        all settings (data, model, training, API limits)
src/pde_service/
  data/        download.py (range requests), dataset.py (split, tensors)
  models/      fno.py
  solver/      burgers.py (reference solver)
  training/    train.py (train + one-shot test evaluation)
  inference/   predictor.py (validation + prediction), initial_conditions.py
  api/         app.py (FastAPI), schemas.py (pydantic)
  metrics.py
frontend/                   index.html, style.css, app.js (Plotly via CDN)
scripts/                    inspection, download, verification, solver validation, controls
artifacts/                  checkpoint, test samples, metrics, validation reports
tests/
```

## References

* Takamoto et al., *PDEBench: An Extensive Benchmark for Scientific Machine Learning*, NeurIPS 2022 Datasets & Benchmarks.
* Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021.
