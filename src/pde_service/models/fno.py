"""1D Fourier Neural Operator (Li et al., 2021, arXiv:2010.08895).

Shapes (B = batch, N = spatial grid points, K = number of output time snapshots):
    input  : (B, N, in_channels)   default in_channels=2 -> [u(x, t=0), x]
    output : (B, N, K)             u(x, t_k) for the K configured snapshot times

The model is a learned map from the initial condition to the solution at fixed
times, for ONE fixed viscosity (the one it was trained on). It assumes a
periodic, uniform grid on [0, 1), as in PDEBench 1D Burgers.

conserve_mean=True enforces the exact conservation law of periodic Burgers,
mean_x u(x, t) = mean_x u(x, 0), by shifting each output snapshot's mean to the
input's mean (input channel 0 must be u0).
"""

from __future__ import annotations

import torch
from torch import nn


class SpectralConv1d(nn.Module):
    """Global convolution done in Fourier space on the lowest `modes` frequencies."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, N)
        n = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, N//2 + 1)
        modes = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            x.shape[0], self.weight.shape[1], x_ft.shape[-1], dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :modes] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :modes], self.weight[:, :, :modes]
        )
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FNO1d(nn.Module):
    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        in_channels: int = 2,
        out_channels: int = 1,
        conserve_mean: bool = False,
    ):
        super().__init__()
        self.conserve_mean = conserve_mean
        self.lift = nn.Linear(in_channels, width)
        self.spectral = nn.ModuleList(SpectralConv1d(width, width, modes) for _ in range(n_layers))
        self.pointwise = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(n_layers))
        self.project = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, out_channels))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, N, in_channels)
        h = self.lift(x).permute(0, 2, 1)  # (B, width, N)
        last = len(self.spectral) - 1
        for i, (spec, pw) in enumerate(zip(self.spectral, self.pointwise)):
            h = spec(h) + pw(h)
            if i != last:
                h = self.act(h)
        out = self.project(h.permute(0, 2, 1))  # (B, N, out_channels)
        if self.conserve_mean:
            u0_mean = x[..., :1].mean(dim=1, keepdim=True)  # (B, 1, 1)
            out = out - out.mean(dim=1, keepdim=True) + u0_mean
        return out


def count_parameters(model: nn.Module) -> int:
    # complex weights count as two real parameters
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
