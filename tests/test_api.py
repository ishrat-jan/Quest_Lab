"""API integration tests against an UNTRAINED, randomly initialised FNO checkpoint and
synthetic test samples created in a temp dir. They test request handling, validation
and response structure only - never model accuracy."""

import copy

import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from pde_service.config import load_config
from pde_service.models.fno import FNO1d

N, K = 64, 4


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    tmp = tmp_path_factory.mktemp("api")
    art = tmp / "artifacts"
    art.mkdir()
    model_cfg = {"modes": 8, "width": 8, "n_layers": 2, "in_channels": 2, "out_channels": K}
    x = (np.arange(N) + 0.5) / N
    t_out = [0.5, 1.0, 1.5, 2.0]
    torch.save({"state_dict": FNO1d(**model_cfg).state_dict(), "model": model_cfg, "nu": 0.1,
                "x": x.tolist(), "t_out": t_out, "epoch": 0, "val_rel_l2": 1.0,
                "train_u0_abs_max": 2.0, "data_source": "SYNTHETIC TEST FIXTURE"},
               art / "fno_burgers_nu0.1.pt")
    u = np.zeros((3, K + 1, N), np.float32)
    u[:, 0] = np.sin(2 * np.pi * x)
    np.savez(art / "test_samples_nu0.1.npz", u=u, indices=np.array([5, 7, 9]))

    cfg = copy.deepcopy(load_config())
    cfg["training"]["artifact_dir"] = str(art)
    cfg["api"].update(supported_nu=[0.1], max_abs_u0=2.5, ood_k_cut=8, ood_energy_fraction=0.05)
    cfg_path = tmp / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch_module.setenv("PDE_SERVICE_CONFIG", str(cfg_path))

    from pde_service.api.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def smooth_u0():
    x = (np.arange(N) + 0.5) / N
    return np.sin(2 * np.pi * x).tolist()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["models_loaded"] == [0.1]


def test_info(client):
    info = client.get("/api/info").json()
    assert info["supported_nu"] == [0.1]
    assert info["models"]["0.1"]["grid_size"] == N
    assert info["models"]["0.1"]["n_test_samples"] == 3


def test_predict_user_ic_with_solver(client):
    r = client.post("/api/predict", json={"nu": 0.1, "u0": smooth_u0(), "run_solver": True})
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["t"]) == K + 1 and len(d["x"]) == N
    assert np.array(d["fno"]).shape == (K + 1, N)
    assert np.array(d["solver"]).shape == (K + 1, N)
    assert set(d["metrics_vs_solver"]) == {"mse", "rmse", "rel_l2"}
    assert d["source"] == {"type": "user"}
    np.testing.assert_allclose(d["fno"][0], smooth_u0())  # row 0 is the given IC


def test_predict_without_solver(client):
    d = client.post("/api/predict", json={"nu": 0.1, "u0": smooth_u0(), "run_solver": False}).json()
    assert "solver" not in d and "metrics_vs_solver" not in d


def test_predict_test_sample_has_pdebench_reference(client):
    d = client.post("/api/predict", json={"nu": 0.1, "test_index": 1, "run_solver": False}).json()
    assert d["source"]["pdebench_sample"] == 7
    assert np.array(d["pdebench_reference"]).shape == (K + 1, N)
    assert "metrics_vs_pdebench" in d


def test_initial_condition_presets(client):
    r = client.post("/api/initial-condition", json={"nu": 0.1, "ic": {"type": "sines", "terms": [{"amp": 1, "k": 2}]}})
    assert r.status_code == 200 and len(r.json()["u0"]) == N
    r = client.post("/api/initial-condition", json={"nu": 0.1, "ic": {"type": "gaussian", "amp": 1, "center": 0.5, "width": 0.1}})
    assert r.status_code == 200
    r = client.post("/api/initial-condition", json={"nu": 0.1, "ic": {"type": "pdebench_test", "index": 2}})
    assert r.json()["pdebench_sample"] == 9


@pytest.mark.parametrize(
    "body,fragment",
    [
        ({"nu": 0.5, "u0": smooth_u0()}, "not supported"),
        ({"nu": 0.1, "u0": [0.0] * (N - 1)}, f"exactly {N} points"),
        ({"nu": 0.1, "u0": [5.0] * N}, "exceeds the supported range"),
        ({"nu": 0.1}, "exactly one of"),
        ({"nu": 0.1, "u0": smooth_u0(), "test_index": 0}, "exactly one of"),
        ({"nu": 0.1, "test_index": 99}, "test sample index"),
        ({"nu": "abc", "u0": smooth_u0()}, "nu"),
        ({"nu": 0.1, "u0": ["a"] * N}, "u0"),
    ],
)
def test_invalid_requests_return_422_with_message(client, body, fragment):
    r = client.post("/api/predict", json=body)
    assert r.status_code == 422
    assert fragment in r.json()["detail"]


def test_nan_rejected(client):
    u0 = smooth_u0()
    u0[3] = float("nan")
    r = client.post("/api/predict", content=f'{{"nu": 0.1, "u0": {str(u0).replace("nan", "NaN")}}}',
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_high_frequency_input_warns(client):
    x = (np.arange(N) + 0.5) / N
    u0 = (0.5 * np.sin(2 * np.pi * 20 * x)).tolist()
    d = client.post("/api/predict", json={"nu": 0.1, "u0": u0, "run_solver": False}).json()
    assert any("wavenumbers" in w for w in d["warnings"])


def test_invalid_preset_parameters(client):
    r = client.post("/api/initial-condition", json={"nu": 0.1, "ic": {"type": "sines", "terms": [{"amp": 1, "k": 0}]}})
    assert r.status_code == 422


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "PDE Solutions as a Service" in r.text
