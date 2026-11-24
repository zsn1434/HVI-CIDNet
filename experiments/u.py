"""EIC-LIE-inspired illumination-conditioned sampling of HV features."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


class GuidedFilter(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.vertical = nn.Conv2d(width, 5, 3, padding=1, bias=False)
        self.horizontal = nn.Conv2d(width, 5, 3, padding=1, bias=False)
        self.taps = nn.Conv2d(width, 25, 3, padding=1, bias=False)
        self.offset_x = nn.Conv2d(width, 25, 3, padding=1, bias=False)
        self.offset_y = nn.Conv2d(width, 25, 3, padding=1, bias=False)
        self.bias = nn.Conv2d(width, width, 3, padding=1, bias=False)
        nn.init.zeros_(self.offset_x.weight)
        nn.init.zeros_(self.offset_y.weight)

    def forward(self, chroma, intensity, adaptive=True):
        batch, channels, height, width = chroma.shape
        vertical, horizontal = self.vertical(intensity), self.horizontal(intensity)
        logits = self.taps(chroma) + (vertical[:, :, None] + horizontal[:, None]).flatten(1, 2)
        weights = logits.softmax(dim=1).unflatten(1, (5, 5))
        dx = self.offset_x(chroma).tanh().unflatten(1, (5, 5)) if adaptive else None
        dy = self.offset_y(chroma).tanh().unflatten(1, (5, 5)) if adaptive else None

        x = torch.arange(width, device=chroma.device, dtype=chroma.dtype)[None, None, None, :]
        y = torch.arange(height, device=chroma.device, dtype=chroma.dtype)[None, None, :, None]
        taps = torch.arange(-2, 3, device=chroma.device, dtype=chroma.dtype)[None, :, None, None]
        update = self.bias(chroma)
        for row in range(5):
            sample_x = x + taps
            sample_y = y + row - 2
            if adaptive:
                sample_x = sample_x + dx[:, row]
                sample_y = sample_y + dy[:, row]
            gx = (sample_x + 0.5) * (2 / width) - 1
            gy = (sample_y + 0.5) * (2 / height) - 1
            grid = torch.stack(torch.broadcast_tensors(gx, gy), dim=-1)
            grid = grid.expand(batch, 5, height, width, 2).permute(0, 2, 3, 1, 4)
            sampled = F.grid_sample(chroma, grid.reshape(batch, height, width * 5, 2),
                                    padding_mode="border", align_corners=False)
            sampled = sampled.reshape(batch, channels, height, width, 5)
            update = update + (sampled * weights[:, row].permute(0, 2, 3, 1)[:, None]).sum(-1)
        return update


class GuidedCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.light_in = nn.Conv2d(72, 16, 1, bias=False)
        self.hv_in = nn.Conv2d(72, 16, 1, bias=False)
        self.filter = GuidedFilter()
        self.out = nn.Conv2d(16, 72, 1, bias=False)
        self.gain = nn.Parameter(torch.zeros(()))
        self.adaptive = True

    def decode_hv_level2(self, hv, intensity):
        features = super().decode_hv_level2(hv, intensity)
        filtered = self.filter(self.hv_in(features), self.light_in(intensity), self.adaptive)
        return features + self.gain * self.out(filtered)


def options(parser):
    parser.add_argument("--fixed-grid", action="store_true", help="Keep offsets at zero")


def configure(model, config):
    model.adaptive = not config["fixed_grid"]


if __name__ == "__main__":
    main("u", GuidedCIDNet, standard_backward, options, configure=configure,
         resume_keys=("fixed_grid",))
