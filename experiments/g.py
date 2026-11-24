"""Subband-specific residuals on the decoder's HV features."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


def haar(a, b, c, d):
    """Orthonormal 2D Haar transform; the same matrix inverts it."""
    return ((a + b + c + d) * 0.5, (a - b + c - d) * 0.5,
            (a + b - c - d) * 0.5, (a - b - c + d) * 0.5)


class SubbandCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        # Each group maps one 72-channel band through its own 12-channel bottleneck.
        self.band_residual = nn.Sequential(
            nn.Conv2d(4 * 72, 4 * 12, 1, groups=4, bias=False),
            nn.SiLU(),
            nn.Conv2d(4 * 12, 4 * 72, 1, groups=4, bias=False),
        )
        nn.init.zeros_(self.band_residual[-1].weight)

    def decode_hv_level2(self, hv, intensity):
        features = super().decode_hv_level2(hv, intensity)
        # Runner inputs are padded or cropped to multiples of eight.
        pixels = F.pixel_unshuffle(features, 2).unflatten(1, (-1, 4)).unbind(2)
        bands = torch.cat(haar(*pixels), dim=1)
        corrections = self.band_residual(bands).chunk(4, dim=1)
        pixels = torch.stack(haar(*corrections), dim=2).flatten(1, 2)
        return features + F.pixel_shuffle(pixels, 2)


if __name__ == '__main__':
    main('g', SubbandCIDNet, standard_backward)
