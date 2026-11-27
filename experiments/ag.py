"""Bidirectional axial feature retrieval inspired by UCMNet."""
import torch
from torch import nn

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


class Retrieval(nn.Module):
    def __init__(self, channels, width, span):
        super().__init__()
        self.q = nn.Conv2d(channels, width, 1, bias=False)
        self.k = nn.Conv2d(channels, width, 1, bias=False)
        self.v = nn.Conv2d(channels, width, 1, bias=False)
        self.out = nn.Conv2d(width, channels, 1, bias=False)
        self.mix = nn.Parameter(torch.zeros(()))
        self.span = span

    def along_rows(self, q, k, v):
        # B,H,W,D; splitting the last partial stripe avoids padding tokens.
        length = self.span or q.size(2)
        parts = []
        for query, key, value in zip(*(t.split(length, dim=2) for t in (q, k, v))):
            scores = query @ key.transpose(-1, -2) * (q.size(-1) ** -0.5)
            parts.append(scores.softmax(-1) @ value)
        return torch.cat(parts, dim=2)

    def forward(self, x, guide):
        q, k, v = (t.permute(0, 2, 3, 1) for t in (self.q(x), self.k(guide), self.v(x)))
        row = self.along_rows(q, k, v)
        col = self.along_rows(*(t.transpose(1, 2) for t in (q, k, v))).transpose(1, 2)
        return x + self.mix * self.out(((row + col) * 0.5).permute(0, 3, 1, 2))


class AxialCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.paths = nn.ModuleDict({str(level): nn.ModuleList(Retrieval(c, d, span) for _ in range(2))
                                    for level, c, d, span in ((2, 72, 48, 16), (3, 144, 96, None))})
        self.cross = True

    def exchange(self, hv, intensity, level):
        color, light = self.paths[str(level)]
        return color(hv, intensity if self.cross else hv), light(intensity, hv if self.cross else intensity)


def options(parser):
    parser.add_argument('--self-only', action='store_true')


def configure(model, config):
    model.cross = not config['self_only']


if __name__ == '__main__':
    main('ag', AxialCIDNet, standard_backward, options, configure=configure, resume_keys=('self_only',))
