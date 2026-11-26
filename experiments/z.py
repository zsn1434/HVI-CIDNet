"""Z: learn separate gains for smooth and detail components of P's decoder update."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.p import IntensitySkipCIDNet
from experiments.runner import main, standard_backward


class FrequencySkipCIDNet(IntensitySkipCIDNet):
    def __init__(self):
        super().__init__()
        self.high_gain = nn.Parameter(torch.zeros(()))
        self.band_mode = 'split'

    def select_decoder_intensity(self, intensity, attended):
        if self.band_mode == 'tied':
            return super().select_decoder_intensity(intensity, attended)
        delta = attended - intensity
        smooth = F.avg_pool2d(F.pad(delta, (2, 2, 2, 2), mode='replicate'), 5, stride=1)
        detail = delta - smooth
        if self.band_mode == 'low':
            return intensity + self.gain * smooth
        if self.band_mode == 'high':
            return intensity + self.high_gain * detail
        return intensity + self.gain * smooth + self.high_gain * detail


def options(parser):
    parser.add_argument('--band-mode', choices=('split', 'tied', 'low', 'high'), default='split')


def configure(model, config):
    model.band_mode = config['band_mode']


if __name__ == '__main__':
    main('z', FrequencySkipCIDNet, standard_backward, options, configure=configure, resume_keys=('band_mode',))
