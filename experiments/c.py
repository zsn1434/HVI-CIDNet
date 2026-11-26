"""C: softly route local regions between two bottleneck experts."""
import torch
from torch import nn
from torch.nn import functional as F

from net.CIDNet import CIDNet
from experiments.runner import main, standard_backward


class RegionCIDNet(CIDNet):
    def __init__(self, router_inputs=4):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Conv2d(72, 8, 1, bias=False), nn.SiLU(), nn.Conv2d(8, 72, 1, bias=False))
            for _ in range(2)
        ])
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
        self.router = nn.Sequential(nn.Conv2d(router_inputs, 8, 1), nn.SiLU(), nn.Conv2d(8, 2, 1))

    @staticmethod
    def region_statistics(hvi):
        light = hvi[:, 2:]
        radius = torch.linalg.vector_norm(hvi[:, :2], dim=1, keepdim=True)
        # Tiles in the input RGB coordinate system. Upsample soft mixtures to avoid seams.
        mean = F.avg_pool2d(light, 32, 32, ceil_mode=True)
        peak = F.max_pool2d(light, 32, 32, ceil_mode=True)
        chroma = F.avg_pool2d(radius, 32, 32, ceil_mode=True)
        texture = (light - F.avg_pool2d(light, 3, 1, 1, count_include_pad=False)).abs()
        texture = F.avg_pool2d(texture, 32, 32, ceil_mode=True)
        return torch.cat((mean, peak, chroma, texture), dim=1)

    def router_hidden(self, statistics):
        return self.router[1](self.router[0](statistics))

    def mix_experts(self, features, logits):
        weights = F.interpolate(logits.softmax(1), size=features.shape[-2:], mode='bilinear', align_corners=False)
        return features + weights[:, :1] * self.experts[0](features) + weights[:, 1:] * self.experts[1](features)

    def encode_hv_level2(self, hv, intensity, hvi):
        features = super().encode_hv_level2(hv, intensity, hvi)
        return self.mix_experts(features, self.router(self.region_statistics(hvi)))


if __name__ == '__main__':
    main('c', RegionCIDNet, standard_backward)
