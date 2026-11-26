"""X: condition C's local expert routing on R's learned Gaussian scale."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.c import RegionCIDNet
from experiments.r import GDN, backward, gaussian_bits
from experiments.runner import main


def shuffle_tiles(grid, training):
    # A random cycle moves every tile when there are at least two tiles.
    flat = grid.flatten(2)
    generator = None if training else torch.Generator(device=grid.device).manual_seed(42)
    order = torch.rand(flat.shape, device=grid.device, generator=generator).argsort(-1)
    shuffled = torch.empty_like(flat).scatter(-1, order, flat.gather(-1, order.roll(1, dims=-1)))
    return shuffled.reshape_as(grid)


class ScaleRouterCIDNet(RegionCIDNet):
    def __init__(self):
        super().__init__(router_inputs=5)
        self.prior = 'learned'
        self.light_in = nn.Conv2d(72, 16, 1, bias=False)
        self.mean = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), GDN(16))
        self.scale = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), GDN(16))

    def encode_hv_skip(self, hv, intensity, hvi, context, auxiliary):
        stats = self.region_statistics(hvi)
        if auxiliary is not None or self.prior in ('learned', 'shuffled'):
            light = self.light_in(intensity)
            scale = F.softplus(self.scale(light)) + 1e-3
            if auxiliary is not None:
                auxiliary.append(gaussian_bits(light, self.mean(light), scale))
        if self.prior == 'zero':
            grid = torch.zeros_like(stats[:, :1])
        elif self.prior == 'gradient':
            value = hvi[:, 2:]
            dx = F.pad(value.diff(dim=-1).abs(), (0, 1))
            dy = F.pad(value.diff(dim=-2).abs(), (0, 0, 0, 1))
            grid = F.avg_pool2d(dx + dy, 32, 32, ceil_mode=True)
        else:
            grid = F.avg_pool2d(scale.log().mean(1, keepdim=True), 8, 8, ceil_mode=True)
            if self.prior == 'shuffled':
                grid = shuffle_tiles(grid, self.training)
        logits = self.router(torch.cat((stats, grid), dim=1))
        return self.mix_experts(self.HV_LCA2(hv, intensity), logits)


def options(parser):
    parser.add_argument('--prior', choices=('learned', 'gradient', 'shuffled', 'zero'), default='learned')


def configure(model, config):
    model.prior = config['prior']


if __name__ == '__main__':
    main('x', ScaleRouterCIDNet, backward, options, configure=configure, resume_keys=('prior',))
