"""CROWn-inspired phase aggregation for the first two HV downsamplers."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


class PhaseDown(nn.Module):
    def __init__(self, original, channels, width):
        super().__init__()
        self.original = original
        self.groups = channels // 6
        self.weights = nn.Sequential(nn.Conv2d(4 * channels, width, 1, bias=False), nn.GELU(),
                                     nn.Conv2d(width, 4 * self.groups, 1, bias=False))
        self.detail = nn.Sequential(nn.Conv2d(4 * channels, width, 1, bias=False), nn.GELU(),
                                    nn.Conv2d(width, width, 3, padding=1, bias=False), nn.GELU(),
                                    nn.Conv2d(width, channels, 1, bias=False))
        kernel = torch.tensor([1., 2., 1.])
        self.register_buffer('kernel', (kernel.outer(kernel) / 16).expand(channels, 1, 3, 3).contiguous())
        self.mix = nn.Parameter(torch.zeros(2))
        self.use_detail = True

    def forward(self, x):
        base = self.original(x)
        smooth = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), self.kernel, groups=x.size(1))
        phases = F.pixel_unshuffle(smooth, 2)
        b, _, h, w = phases.shape
        weights = self.weights(phases).reshape(b, self.groups, 1, 4, h, w).softmax(3)
        pooled = (phases.reshape(b, self.groups, 6, 4, h, w) * weights).sum(3).reshape_as(base)
        result = base + self.mix[0] * (pooled - base)
        if self.use_detail:
            result = result + self.mix[1] * self.detail(F.pixel_unshuffle(x - smooth, 2))
        return result


class PhaseCIDNet(CIDNet):
    widths = (48, 96)

    def __init__(self):
        super().__init__()
        for block, channels, width in zip((self.HVE_block1, self.HVE_block2), (36, 72), self.widths):
            block.down[1] = PhaseDown(block.down[1], channels, width)


def run(name, cls):
    def options(parser):
        parser.add_argument('--no-detail', action='store_true')

    def configure(model, config):
        for block in (model.HVE_block1, model.HVE_block2):
            block.down[1].use_detail = not config['no_detail']

    main(name, cls, standard_backward, options, configure=configure, resume_keys=('no_detail',))


if __name__ == '__main__':
    run('ae', PhaseCIDNet)
