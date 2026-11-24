"""D²R²OSR AdaptiveFusion on same-scale intensity and HV decoder features."""
import torch
from torch import nn

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


class AdaptiveFusion(nn.Module):
    def __init__(self, width=36):
        super().__init__()
        self.project = nn.Conv2d(width, width, 1)
        self.spatial = nn.Sequential(nn.Conv2d(2 * width, width, 3, padding=1),
                                     nn.LeakyReLU(0.2), nn.Conv2d(width, 1, 1), nn.Sigmoid())
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(width, width // 4, 1),
                                     nn.ReLU(), nn.Conv2d(width // 4, width, 1), nn.Sigmoid())

    def forward(self, target, guide, gain=1.0):
        guide = self.project(guide)
        spatial = self.spatial(torch.cat((target, guide), dim=1))
        return target + gain * (spatial * self.channel(target) * guide)


class FusionCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.fusion = AdaptiveFusion()
        self.gain = nn.Parameter(torch.zeros(()))

    def refine_decoder_intensity(self, intensity, hv):
        return self.fusion(intensity, hv, self.gain)


if __name__ == '__main__':
    main('o', FusionCIDNet, standard_backward)
