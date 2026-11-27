"""FiDeSR-inspired coarse and fine enhancement residual correction."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


class DenseBlock(nn.Module):
    def __init__(self, width, growth):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv2d(width + i * growth, growth, 3, padding=1) for i in range(4))
        self.reduce = nn.Conv2d(width + 4 * growth, width, 3, padding=1)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(F.silu(layer(torch.cat(features, 1))))
        return x + 0.2 * self.reduce(torch.cat(features, 1))


def correction(width, growth, depth):
    output = nn.Conv2d(width, 3, 1)
    nn.init.zeros_(output.weight)
    nn.init.zeros_(output.bias)
    return nn.Sequential(nn.Conv2d(6, width, 1), *(DenseBlock(width, growth) for _ in range(depth)), output)


class ResidualCIDNet(CIDNet):
    coarse_width = 32

    def __init__(self):
        super().__init__()
        self.coarse = correction(self.coarse_width, self.coarse_width // 2, 2)
        self.fine = correction(16, 8, 1)
        self.use_input = True

    def refine_residual(self, hvi, residual):
        guide = hvi if self.use_input else torch.zeros_like(hvi)
        coarse = self.coarse(F.avg_pool2d(torch.cat((guide, residual), 1), 2))
        updated = residual + F.interpolate(coarse, size=hvi.shape[-2:], mode='bilinear', align_corners=False)
        return updated + self.fine(torch.cat((guide, updated), 1))


def run(name, cls):
    def options(parser):
        parser.add_argument('--residual-only', action='store_true')

    def configure(model, config):
        model.use_input = not config['residual_only']

    main(name, cls, standard_backward, options, configure=configure, resume_keys=('residual_only',))


if __name__ == '__main__':
    run('af', ResidualCIDNet)
