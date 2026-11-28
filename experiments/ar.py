"""Fuse T's broad intensity update with O's local cross-branch update."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.o import AdaptiveFusion
from experiments.runner import main, standard_backward
from experiments.t import LargeKernelAttention
from net.CIDNet import CIDNet


class ScaleSplitCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.fusion = AdaptiveFusion()
        self.gain = nn.Parameter(torch.zeros(()))
        self.reduce = nn.Conv2d(36, 16, 1, bias=False)
        self.attention = LargeKernelAttention()
        self.expand = nn.Conv2d(16, 36, 1, bias=False)
        self.large_gain = nn.Parameter(torch.zeros(()))
        self.split = nn.Sequential(nn.Conv2d(72, 48, 3, padding=1, bias=False),
                                   nn.GELU(), nn.Conv2d(48, 3, 1, bias=False))
        self.split_mode = 'split'

    @staticmethod
    def smooth(update, weights):
        filtered = [F.avg_pool2d(F.pad(update, (k // 2,) * 4, mode='replicate'), k, stride=1)
                    for k in (3, 5, 9)]
        return sum(weight * value for weight, value in zip(weights.split(1, dim=1), filtered))

    def refine_decoder_intensity(self, intensity, hv):
        # O's gates depend on the original intensity, including its channel statistics.
        local = self.fusion(intensity, hv, self.gain) - intensity
        broad = self.large_gain * self.expand(self.attention(self.reduce(intensity)))
        if self.split_mode == 'sum':
            return intensity + broad + local
        if self.split_mode == 'swapped':
            broad, local = local, broad
        weights = self.split(torch.cat((intensity, hv), dim=1)).softmax(dim=1)
        return intensity + self.smooth(broad, weights) + (local - self.smooth(local, weights))


def options(parser):
    parser.add_argument('--split-mode', choices=('split', 'sum', 'swapped'), default='split')


def configure(model, config):
    model.split_mode = config['split_mode']


if __name__ == '__main__':
    main('ar', ScaleSplitCIDNet, standard_backward, options, configure=configure, resume_keys=('split_mode',))
