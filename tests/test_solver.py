import numpy as np
import pytest

from pde_service.solver.burgers import solve_burgers


def cole_hopf(x, t, nu, a):
    """Exact periodic Burgers solution via phi = 1 + a exp(-D (2pi)^2 t) cos(2 pi x)."""
    d = nu / np.pi
    e = a * np.exp(-d * (2 * np.pi) ** 2 * t)
    return 4 * np.pi * d * e * np.sin(2 * np.pi * x) / (1 + e * np.cos(2 * np.pi * x))


@pytest.mark.parametrize("nu,a", [(0.1, 0.9), (0.01, 0.99)])
def test_matches_exact_cole_hopf_solution(nu, a):
    x = np.arange(512) / 512
    t = np.array([0.1, 1.0, 2.0])
    u = solve_burgers(cole_hopf(x, 0.0, nu, a), t, nu)
    ref = np.stack([cole_hopf(x, tt, nu, a) for tt in t])
    rel = np.linalg.norm(u - ref, axis=1) / np.linalg.norm(ref, axis=1)
    assert rel.max() < 1e-8


def test_mean_is_conserved():
    # Burgers in conservation form on a periodic domain conserves the integral of u
    x = np.arange(256) / 256
    u0 = np.sin(2 * np.pi * x) + 0.3
    u = solve_burgers(u0, np.array([0.5, 2.0]), 0.1)
    np.testing.assert_allclose(u.mean(axis=1), u0.mean(), atol=1e-10)


@pytest.mark.parametrize(
    "u0,t,nu",
    [
        (np.array([1.0, np.nan] * 8), [0.1], 0.1),
        (np.zeros(4), [0.1], 0.1),
        (np.zeros(16), [0.2, 0.1], 0.1),
        (np.zeros(16), [0.1], -1.0),
    ],
)
def test_rejects_invalid_inputs(u0, t, nu):
    with pytest.raises(ValueError):
        solve_burgers(u0, np.array(t), nu)
