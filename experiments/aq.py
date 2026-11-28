"""Use R's spatial scale to guide the detail band of Z's decoder update."""
import torch
from torch import nn

from experiments.r import InformationCIDNet, backward
from experiments.runner import main
from experiments.z import smooth_update


class ScaleGuidedBands(InformationCIDNet):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(()))
        self.high_gain = nn.Parameter(torch.zeros(()))
        self.intensity_projection = nn.Conv2d(72, 32, 1, bias=False)
        self.detail_projection = nn.Conv2d(72, 32, 1, bias=False)
        self.detail_gate = nn.Sequential(nn.Conv2d(80, 96, 3, padding=1, bias=False),
                                         nn.GELU(), nn.Conv2d(96, 8, 1, bias=False))
        nn.init.zeros_(self.detail_gate[-1].weight)
        self.detail_mode = 'spatial'

    def quarter_context(self, intensity, context, auxiliary):
        return context, self.information_scale(intensity, auxiliary)

    def encode_hv_skip(self, hv, intensity, hvi, context, auxiliary):
        features = super().encode_hv_level2(hv, intensity, hvi)
        return self.apply_information(features, hvi, context[1])

    def band_update(self, intensity, delta, scale):
        smooth = smooth_update(delta)
        detail = delta - smooth
        updated = self.gain * smooth + self.high_gain * detail
        if self.detail_mode == 'independent':
            return updated
        scale = scale.log()
        if self.detail_mode == 'mean':
            scale = scale.mean(dim=(-2, -1), keepdim=True).expand_as(scale)
        signals = torch.cat((scale, self.intensity_projection(intensity), self.detail_projection(detail)), 1)
        gate = self.detail_gate(signals).tanh().repeat_interleave(9, dim=1)
        correction = gate * detail
        return updated + (correction - smooth_update(correction))

    def decode_intensity_with_context(self, intensity, attended, context):
        return intensity + self.band_update(intensity, attended - intensity, context[1])


def options(parser):
    parser.add_argument('--detail-mode', choices=('spatial', 'mean', 'independent'), default='spatial')


def configure(model, config):
    model.detail_mode = config['detail_mode']


if __name__ == '__main__':
    main('aq', ScaleGuidedBands, backward, options, configure=configure, resume_keys=('detail_mode',))
