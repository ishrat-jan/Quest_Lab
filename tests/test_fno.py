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
