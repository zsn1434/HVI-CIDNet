"""AB: normalized anisotropic Gaussian splatting at the second HV decoder upsample."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


def gaussian_upsample(values, geometry, mode):
    # For output (2i, 2j) through (2i+1, 2j+1), the nominal source center is (2i+.5, 2j+.5).
    offset_x, offset_y, width, height, angle = geometry.unbind(1)
    centers_move = mode == 'offset'
    coord = torch.arange(4, dtype=geometry.dtype, device=geometry.device) - 1.5
    yy, xx = torch.meshgrid(coord, coord, indexing='ij')
    dx = xx.reshape(1, 16, 1, 1) - (0.5 * offset_x.tanh()[:, None] if centers_move else 0)
    dy = yy.reshape(1, 16, 1, 1) - (0.5 * offset_y.tanh()[:, None] if centers_move else 0)
    sx = 0.5 + 1.5 * width.sigmoid()
    sy = sx if mode == 'circular' else 0.5 + 1.5 * height.sigmoid()
    theta = (math.pi / 2) * angle.tanh()
    c, s = theta.cos()[:, None], theta.sin()[:, None]
    u = (c * dx + s * dy) / sx[:, None]
    v = (c * dy - s * dx) / sy[:, None]
    weights = (-0.5 * (u.square() + v.square())).exp()
    shape = (2 * values.size(2), 2 * values.size(3))

    def accumulate(patches):
        return F.fold(patches.flatten(-2), shape, kernel_size=4, padding=1, stride=2)

    mass = accumulate(weights).clamp_min(1e-8)
    # Bound the size of each temporary patch tensor on full-resolution LOL inputs.
    output = []
    for chunk in values.split(8, dim=1):
        weighted = (chunk[:, :, None] * weights[:, None]).flatten(1, 2)
        output.append(accumulate(weighted) / mass)
    return torch.cat(output, dim=1)


class GaussianCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.geometry = nn.Sequential(nn.Conv2d(73, 16, 1), nn.SiLU(), nn.Conv2d(16, 5, 1))
        self.gain = nn.Parameter(torch.zeros(()))
        self.kernel_mode = 'offset'

    def upsample_decoder_hv(self, hv, skip, intensity, hvi):
        block = self.HVD_block2
        if self.kernel_mode == 'bilinear':
            return block(hv, skip)
        values = block.up_scale[0](hv)
        aligned = block.up_scale[1](values)
        light = F.adaptive_avg_pool2d(hvi[:, 2:], values.shape[-2:])
        light = F.avg_pool2d(F.pad(light, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
        dx = F.pad(light.diff(dim=-1), (0, 1))
        dy = F.pad(light.diff(dim=-2), (0, 0, 0, 1))
        geometry = self.geometry(torch.cat((intensity, dx.abs() + dy.abs()), dim=1))
        splat = gaussian_upsample(values, geometry, self.kernel_mode)
        return block.fuse_skip(aligned + self.gain * (splat - aligned), skip)


def options(parser):
    parser.add_argument('--kernel-mode', choices=('bilinear', 'circular', 'ellipse', 'offset'), default='offset')


def configure(model, config):
    model.kernel_mode = config['kernel_mode']


if __name__ == '__main__':
    main('ab', GaussianCIDNet, standard_backward, options, configure=configure, resume_keys=('kernel_mode',))
