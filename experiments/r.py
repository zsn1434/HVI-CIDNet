"""IAFMNet-inspired Gaussian information density and affine recalibration."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.runner import main
from net.CIDNet import CIDNet


class GDN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(channels))
        self.gamma = nn.Parameter(0.1 * torch.eye(channels))

    def forward(self, x):
        denominator = F.conv2d(x.square(), self.gamma.clamp_min(0)[:, :, None, None],
                               self.beta.clamp_min(1e-4))
        return x * denominator.rsqrt()


def gaussian_bits(light, mean, scale):
    # Detach the coding target so the loss does not reward a collapsed I projection.
    sample = light.detach() + torch.rand_like(light) - 0.5
    centered = sample - mean
    width = scale * math.sqrt(2)
    probability = 0.5 * (((centered + 0.5) / width).erf() - ((centered - 0.5) / width).erf())
    return -probability.clamp_min(1e-8).log2().mean()


class InformationCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.prior = "learned"
        self.light_in = nn.Conv2d(72, 16, 1, bias=False)
        self.hv_in = nn.Conv2d(72, 16, 1, bias=False)
        self.mean = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), GDN(16))
        self.scale = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), GDN(16))
        self.expand = nn.Conv2d(16, 32, 1)
        self.modulation = nn.Conv2d(32, 16, 1)
        self.local = nn.Conv2d(16, 16, 3, padding=1, groups=16)
        self.out = nn.Conv2d(16, 72, 1, bias=False)
        nn.init.zeros_(self.out.weight)

    def information_scale(self, intensity, auxiliary):
        light = self.light_in(intensity)
        scale = F.softplus(self.scale(light)) + 1e-3
        if auxiliary is not None:
            auxiliary.append(gaussian_bits(light, self.mean(light), scale))
        return scale

    def refine_encoder_hv(self, hv, intensity, hvi, auxiliary):
        scale = self.information_scale(intensity, auxiliary)
        return self.apply_information(hv, hvi, scale)

    def apply_information(self, hv, hvi, scale):
        if self.prior == "constant":
            prior = torch.ones_like(scale)
        elif self.prior == "gradient":
            brightness = F.adaptive_avg_pool2d(hvi[:, 2:], scale.shape[-2:])
            dx = F.pad(brightness.diff(dim=-1).abs(), (0, 1))
            dy = F.pad(brightness.diff(dim=-2).abs(), (0, 0, 0, 1))
            prior = (dx + dy).expand_as(scale)
        else:
            prior = scale
        gate, local = self.expand(self.hv_in(hv)).chunk(2, 1)
        update = self.modulation(torch.cat((gate, prior), 1)) * self.local(local)
        return hv + self.out(update)


def options(parser):
    parser.add_argument("--prior", choices=["learned", "constant", "gradient"], default="learned")


def configure(model, config):
    model.prior = config["prior"]


def backward(model, low, target, objective, args):
    auxiliary = []
    output = model.enhance(*model.input_features(low), auxiliary=auxiliary)
    reconstruction, perceptual = objective(model, output, target)
    loss = reconstruction + perceptual + 1e-4 * auxiliary[0]
    loss.backward()
    return loss.detach()


if __name__ == "__main__":
    main("r", InformationCIDNet, backward, options, configure=configure, resume_keys=("prior",))
