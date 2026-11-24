"""Local exposure curve conditioned on input and predicted intensity."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


class ExposureCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.curve = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1, padding_mode='replicate'),
            nn.SiLU(),
            nn.Conv2d(16, 1, 1),
        )
        nn.init.zeros_(self.curve[-1].weight)
        nn.init.zeros_(self.curve[-1].bias)

    def forward(self, low):
        rgb = super().forward(low)
        value = rgb.amax(dim=1, keepdim=True)
        stats = torch.cat((low.amax(dim=1, keepdim=True), value), dim=1)
        tiles = F.avg_pool2d(stats, 16, 16, ceil_mode=True, count_include_pad=False)
        strength = F.interpolate(self.curve(tiles).tanh(), size=value.shape[-2:],
                                 mode='bilinear', align_corners=False)
        # v' = v + strength * v * (1 - v), applied with a common RGB gain.
        return rgb * (1 + strength * (1 - value))


if __name__ == '__main__':
    main('f', ExposureCIDNet, standard_backward)
