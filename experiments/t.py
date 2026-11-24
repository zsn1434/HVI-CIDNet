"""UCAN large-kernel spatial attention at the intensity decoder."""
import torch
from torch import nn

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


class LargeKernelAttention(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.horizontal = nn.Conv2d(width, width, (1, 5), padding=(0, 2), groups=width)
        self.vertical = nn.Conv2d(width, width, (5, 1), padding=(2, 0), groups=width)
        self.dilated_h = nn.Conv2d(width, width, (1, 5), padding=(0, 4),
                                   dilation=(1, 2), groups=width)
        self.dilated_v = nn.Conv2d(width, width, (5, 1), padding=(4, 0),
                                   dilation=(2, 1), groups=width)
        self.local = nn.Sequential(nn.Conv2d(width, 4, 1), nn.GELU(),
                                   nn.Conv2d(4, 4, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(4, width, 1))
        self.channel = nn.Conv2d(width, width, 1)

    def forward(self, x, large_kernel=True):
        response = self.local(x)
        if large_kernel:
            response = response + self.dilated_v(self.dilated_h(self.vertical(self.horizontal(x))))
        return self.channel(x) * response


class LargeKernelCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv2d(36, 16, 1, bias=False)
        self.attention = LargeKernelAttention()
        self.expand = nn.Conv2d(16, 36, 1, bias=False)
        self.gain = nn.Parameter(torch.zeros(()))
        self.large_kernel = True

    def refine_decoder_intensity(self, intensity, hv):
        return intensity + self.gain * self.expand(self.attention(self.reduce(intensity), self.large_kernel))


def options(parser):
    parser.add_argument("--local-only", action="store_true", help="Use the local LKSA branch alone")


def configure(model, config):
    model.large_kernel = not config["local_only"]


if __name__ == "__main__":
    main("t", LargeKernelCIDNet, standard_backward, options, configure=configure,
         resume_keys=("local_only",))
