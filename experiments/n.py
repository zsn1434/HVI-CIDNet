"""DRWKV's trainable Scharr branch on the half-resolution intensity decoder."""
import torch
from torch import nn

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


class ScharrFeatures(nn.Module):
    def __init__(self, channels=36):
        super().__init__()
        self.dx = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dy = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        kernel = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]])
        with torch.no_grad():
            self.dx.weight.copy_(kernel.expand_as(self.dx.weight))
            self.dy.weight.copy_(kernel.T.expand_as(self.dy.weight))
        self.normalize = nn.BatchNorm2d(channels)
        self.mix = nn.Sequential(nn.Conv2d(channels, 16, 1), nn.SiLU(),
                                 nn.Conv2d(16, channels, 3, padding=1))

    def forward(self, x):
        magnitude = self.dx(x).abs() + self.dy(x).abs()
        return self.mix(x + self.normalize(magnitude))


class EdgeCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.edge = ScharrFeatures()
        self.gain = nn.Parameter(torch.zeros(()))

    def refine_decoder_intensity(self, intensity, hv):
        return intensity + self.gain * self.edge(intensity)


if __name__ == '__main__':
    main('n', EdgeCIDNet, standard_backward)
