"""TPCNet CG_MSA_V port. Upstream source and HVI adaptations: experiments/README.md."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


def diagonal_pairs(x):
    cells = F.pixel_unshuffle(x, 2).unflatten(1, (-1, 4))
    tl, tr, bl, br = cells.unbind(2)
    return (tr + bl) * 0.5, (tl + br) * 0.5


class PairedAttention(nn.Module):
    def __init__(self, width=24, heads=4):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        channels = 2 * width

        def project_pair():
            return nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            )

        self.query_value = project_pair()
        self.key_value = project_pair()
        self.merge_values = nn.Conv2d(channels, width, 3, padding=1)
        self.upsample = nn.Sequential(nn.Conv2d(width, 4 * width, 1, bias=False), nn.PixelShuffle(2))
        self.skip = nn.Conv2d(width, width, 1, bias=False)

    def forward(self, x, guide):
        x1, x2 = diagonal_pairs(x)
        y1, y2 = diagonal_pairs(guide)
        query, first_value = self.query_value(torch.cat((x1, y2), 1)).chunk(2, 1)
        key, second_value = self.key_value(torch.cat((x2, y1), 1)).chunk(2, 1)
        value = self.merge_values(torch.cat((second_value, first_value), 1))
        q, k, v = (t.flatten(2).unflatten(1, (self.heads, -1)) for t in (query, key, value))
        scores = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        mixed = ((scores * self.temperature).softmax(-1) @ v).reshape_as(value)
        return self.upsample(mixed) + self.skip(x)


class PairedCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.chroma_in = nn.Conv2d(72, 24, 1, bias=False)
        self.light_in = nn.Conv2d(72, 24, 1, bias=False)
        self.paired = PairedAttention()
        self.chroma_out = nn.Conv2d(24, 72, 1, bias=False)
        nn.init.zeros_(self.chroma_out.weight)

    def decode_hv_level2(self, hv, intensity):
        features = super().decode_hv_level2(hv, intensity)
        update = self.paired(self.chroma_in(features), self.light_in(intensity))
        return features + self.chroma_out(update)


if __name__ == '__main__':
    main('j', PairedCIDNet, standard_backward)
