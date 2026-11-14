import torch

from pde_service.models.fno import FNO1d, SpectralConv1d


def test_output_shape():
    m = FNO1d(modes=8, width=16, n_layers=2, in_channels=2, out_channels=20)
    assert m(torch.randn(3, 64, 2)).shape == (3, 64, 20)


def test_resolution_independent_weights():
    m = FNO1d(modes=8, width=16, n_layers=2, out_channels=5)
    for n in (32, 128, 256):
        assert m(torch.randn(2, n, 2)).shape == (2, n, 5)


def test_spectral_conv_ignores_high_frequencies():
    torch.manual_seed(0)
    conv = SpectralConv1d(1, 1, modes=4)
    x = torch.arange(64) / 64
    high = torch.sin(2 * torch.pi * 20 * x).reshape(1, 1, -1)  # mode 20 >> 4
    assert conv(high).abs().max() < 1e-5


def test_gradients_flow():
    m = FNO1d(modes=4, width=8, n_layers=2, out_channels=3)
    m(torch.randn(2, 32, 2)).sum().backward()
    assert all(p.grad is not None for p in m.parameters())


def test_conserve_mean_matches_input_mean_exactly():
    m = FNO1d(modes=4, width=8, n_layers=2, out_channels=3, conserve_mean=True)
    x = torch.randn(2, 32, 2)
    out = m(x)
    torch.testing.assert_close(out.mean(dim=1), x[..., 0].mean(dim=1, keepdim=True).expand(2, 3))


def test_no_grid_fno_is_exactly_shift_equivariant():
    # without the x-coordinate channel every layer commutes with periodic grid shifts
    torch.manual_seed(0)
    m = FNO1d(modes=8, width=16, n_layers=3, in_channels=1, out_channels=4, conserve_mean=True)
    u = torch.randn(2, 64, 1)
    torch.testing.assert_close(m(torch.roll(u, 11, dims=1)), torch.roll(m(u), 11, dims=1), atol=1e-5, rtol=1e-5)


def test_grid_channel_breaks_shift_equivariance():
    torch.manual_seed(0)
    m = FNO1d(modes=8, width=16, n_layers=3, in_channels=2, out_channels=4)
    x = torch.linspace(0, 1, 64).reshape(1, -1, 1).expand(2, -1, -1)
    u = torch.randn(2, 64, 1)
    a = m(torch.cat([torch.roll(u, 11, dims=1), x], -1))
    b = torch.roll(m(torch.cat([u, x], -1)), 11, dims=1)
    assert (a - b).abs().max() > 1e-3
