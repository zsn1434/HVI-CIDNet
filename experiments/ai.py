"""Two stages of differentiable pairwise logic for half-resolution HV features."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


def gradient(x):
    return F.pad(x.diff(dim=-1).abs(), (0, 1)) + F.pad(x.diff(dim=-2).abs(), (0, 0, 0, 1))


class LogicCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.color = nn.Conv2d(36, 32, 1)
        self.light = nn.Conv2d(36, 32, 1)
        self.context = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
                                     nn.Conv2d(32, 32, 1, bias=False))
        self.rules = nn.Parameter(torch.randn(2, 32, 16) * 0.1)
        table = ((torch.arange(16)[:, None] >> torch.arange(4)) & 1).float()
        self.register_buffer('truth', table)
        self.out = nn.Conv2d(32, 36, 1, bias=False)
        nn.init.zeros_(self.out.weight)
        self.bilinear = False

    def combine(self, a, b, stage):
        rule = self.rules[stage]
        if self.bilinear:
            bias, left, right, joint = rule[:, :4].T[:, None, :, None, None].unbind()
            return bias + left * a + right * b + joint * a * b
        # Expected truth value at the four corners under independent Bernoulli inputs.
        corners = (rule.softmax(-1) @ self.truth).T[:, None, :, None, None]
        c00, c01, c10, c11 = corners.unbind()
        return (1-a) * ((1-b)*c00 + b*c01) + a * ((1-b)*c10 + b*c11)

    def refine_half(self, hv, intensity):
        edge = self.combine(self.color(gradient(hv)).sigmoid(), self.light(gradient(intensity)).sigmoid(), 0)
        return hv + self.out(self.combine(edge, self.context(edge).sigmoid(), 1))


def options(parser):
    parser.add_argument('--bilinear', action='store_true')


def configure(model, config):
    model.bilinear = config['bilinear']


if __name__ == '__main__':
    main('ai', LogicCIDNet, standard_backward, options, configure=configure, resume_keys=('bilinear',))
