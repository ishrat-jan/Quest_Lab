import numpy as np
import pytest

from pde_service.data.dataset import make_inputs, split_indices
from pde_service.metrics import all_metrics, rel_l2


def test_metrics_known_values():
    ref = np.array([[3.0, 4.0]])          # ||ref|| = 5
    pred = ref + np.array([[0.0, 1.0]])   # error norm 1
    m = all_metrics(pred, ref)
    assert m["mse"] == pytest.approx(0.5)
    assert m["rmse"] == pytest.approx(np.sqrt(0.5))
    assert m["rel_l2"] == pytest.approx(0.2)


def test_rel_l2_is_per_sample_mean():
    ref = np.array([[1.0, 0.0], [10.0, 0.0]])
    pred = np.array([[1.1, 0.0], [11.0, 0.0]])  # both 10 % off
    assert rel_l2(pred, ref) == pytest.approx(0.1)


def test_rel_l2_single_2d_field_needs_sample_axis():
    # regression: a single (n, K) field must be passed as (1, n, K); passed bare, axis 0 is
    # taken as the sample axis and the result is a different quantity
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(16, 4))
    pred = ref + rng.normal(scale=0.1, size=(16, 4)) * np.linspace(0, 2, 16)[:, None]
    whole = np.linalg.norm(pred - ref) / np.linalg.norm(ref)
    assert rel_l2(pred[None], ref[None]) == pytest.approx(whole)
    assert rel_l2(pred, ref) != pytest.approx(whole)


def test_rel_l2_shape_mismatch():
    with pytest.raises(ValueError):
        rel_l2(np.zeros(3), np.zeros(4))


def test_split_is_disjoint_and_deterministic():
    a = split_indices(100, 70, 10, 20, seed=0)
    b = split_indices(100, 70, 10, 20, seed=0)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    tr, va, te = map(set, a)
    assert not (tr & va or tr & te or va & te)
    assert len(tr | va | te) == 100


def test_split_too_large():
    with pytest.raises(ValueError):
        split_indices(10, 8, 2, 1, seed=0)


def test_make_inputs_shape():
    x = np.linspace(0, 1, 16, endpoint=False)
    inp = make_inputs(np.zeros((3, 16)), x)
    assert inp.shape == (3, 16, 2)
    np.testing.assert_allclose(inp[1, :, 1].numpy(), x, atol=1e-7)


def test_reflection_matches_cell_centred_grid():
    import torch

    from pde_service.training.train import reflect

    n = 32
    x = (np.arange(n) + 0.5) / n
    u = torch.from_numpy(np.sin(2 * np.pi * x) + 0.3 * np.cos(6 * np.pi * x)).reshape(1, n, 1)
    expected = -(np.sin(2 * np.pi * (1 - x)) + 0.3 * np.cos(6 * np.pi * (1 - x)))
    np.testing.assert_allclose(reflect(u)[0, :, 0].numpy(), expected, atol=1e-12)
