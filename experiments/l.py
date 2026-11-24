"""MArFE sine + Fourier-reparameterized channel projection; see experiments/README.md."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


class FourierLinear(nn.Module):
    def __init__(self, width=72, low=8, high=8, phases=4):
        super().__init__()
        # Match the upstream frequency/phase grid on the input-channel axis.
        grid = torch.linspace(-math.pi * low, math.pi * low, width, dtype=torch.float64)
        freq = torch.cat((torch.arange(1, low + 1) / low, torch.arange(1, high + 1)))
        phase = torch.arange(phases, dtype=grid.dtype) * (2 * math.pi / phases)
        angles = freq[:, None, None] * grid + phase[None, :, None]
        basis = (0.05 * angles.cos()).flatten(0, 1).float()
        self.register_buffer('basis', basis)
        self.coeff = nn.Parameter(torch.empty(width, len(basis)))
        self.bias = nn.Parameter(torch.zeros(width))
        with torch.no_grad():
            self.coeff.uniform_(-1, 1).mul_(math.sqrt(6 / len(basis)) / basis.norm(dim=1))

    def forward(self, x):
        weights = self.coeff @ self.basis
        return F.conv2d(x, weights[:, :, None, None], self.bias)


class FourierCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.project = nn.Conv2d(72, 72, 1)
        self.fourier = FourierLinear()
        self.gain = nn.Parameter(torch.zeros(()))

    def decode_hv_level2(self, hv, intensity):
        features = super().decode_hv_level2(hv, intensity)
        return features + self.gain * self.fourier(self.project(features).sin())


if __name__ == '__main__':
    main('l', FourierCIDNet, standard_backward)
