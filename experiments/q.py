"""Multinex's six analytic illumination measurements at the CIDNet intensity stem."""
import torch
from torch import nn

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


def illumination_maps(rgb, peak):
    red, green, blue = rgb.unbind(1)
    peak = peak.squeeze(1)
    mean = rgb.mean(1)
    rec709 = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    lightness = (peak + rgb.amin(1)) * 0.5
    ycgco = 0.25 * (red + 2 * green + blue)
    length = (rgb.square().sum(1) + 1e-6).sqrt()
    return torch.stack((mean, rec709, peak, lightness, ycgco, length), dim=1)


class PhotometricCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.illumination_projection = nn.Conv2d(6, 36, 1, bias=False)
        nn.init.zeros_(self.illumination_projection.weight)

    def encode_intensity_input(self, rgb, intensity):
        stem = super().encode_intensity_input(rgb, intensity)
        return stem + self.illumination_projection(illumination_maps(rgb, intensity))


if __name__ == '__main__':
    main('q', PhotometricCIDNet, standard_backward)
