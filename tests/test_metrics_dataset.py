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
