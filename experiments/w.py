"""W: share C's regional representation between the HV experts and the I decoder."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.c import RegionCIDNet
from experiments.runner import main, standard_backward


class RegionGateCIDNet(RegionCIDNet):
    def __init__(self):
        super().__init__()
        self.gate_mode = 'region'
        self.gate = nn.Sequential(nn.Conv2d(10, 8, 1), nn.SiLU(), nn.Conv2d(8, 1, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def routing_context(self, hvi):
        return self.router_hidden(self.region_statistics(hvi))

    def encode_hv_skip(self, hv, intensity, hvi, context, auxiliary):
        return self.mix_experts(self.HV_LCA2(hv, intensity), self.router[2](context))

    def decode_intensity_with_context(self, intensity, attended, context):
        delta = attended - intensity
        if self.gate_mode == 'scalar':
            return intensity + self.gate[-1].bias * delta
        regions = F.interpolate(context, size=intensity.shape[-2:], mode='bilinear', align_corners=False)
        if self.gate_mode == 'decoder':
            regions = torch.zeros_like(regions)
        signals = torch.cat((regions, intensity.abs().mean(1, keepdim=True),
                             delta.abs().mean(1, keepdim=True)), dim=1)
        return intensity + self.gate(signals) * delta


def options(parser):
    parser.add_argument('--gate-mode', choices=('region', 'decoder', 'scalar'), default='region')


def configure(model, config):
    model.gate_mode = config['gate_mode']


if __name__ == '__main__':
    main('w', RegionGateCIDNet, standard_backward, options, configure=configure, resume_keys=('gate_mode',))
