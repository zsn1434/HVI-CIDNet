"""AC: intensity-indexed bilateral grid of local affine maps on the output HV plane."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


class BilateralColorCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.coefficients = nn.Conv2d(72, 7 * 8, 1)
        # The grid predicts a residual affine transform. Zero residual equals identity.
        nn.init.zeros_(self.coefficients.weight)
        nn.init.zeros_(self.coefficients.bias)
        self.guide = 'intensity'
        self.diagonal = False

    def correct_chroma(self, hvi, features):
        b, _, h, w = hvi.shape
        grid_h, grid_w = min(16, features.size(2)), min(16, features.size(3))
        features = F.adaptive_avg_pool2d(features, (grid_h, grid_w))
        grid = self.coefficients(features).reshape(b, 7, 8, grid_h, grid_w)
        mass = F.softplus(grid[:, 6:]) + 1e-4
        weighted_grid = torch.cat((grid[:, :6] * mass, mass), dim=1)

        xs = (torch.arange(w, dtype=hvi.dtype, device=hvi.device) + 0.5) * (2 / w) - 1
        ys = (torch.arange(h, dtype=hvi.dtype, device=hvi.device) + 0.5) * (2 / h) - 1
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        z = hvi[:, 2].clamp(0, 1) * 2 - 1
        if self.guide == 'fixed':
            z = torch.zeros_like(z)
        query = torch.stack((xx.expand_as(z), yy.expand_as(z), z), dim=-1).unsqueeze(1)
        sampled = F.grid_sample(weighted_grid, query, mode='bilinear', padding_mode='border',
                                align_corners=False).squeeze(2)
        a, b12, b21, d, bh, bv = (sampled[:, :6] / sampled[:, 6:]).unbind(1)
        hue_x, hue_y, light = hvi.unbind(1)
        dx, dy = a * hue_x + bh, d * hue_y + bv
        if not self.diagonal:
            dx, dy = dx + b12 * hue_y, dy + b21 * hue_x
        return torch.stack((hue_x + dx, hue_y + dy, light), dim=1)

    def output_color(self, output_hvi, hv_features):
        return self.trans.PHVIT(self.correct_chroma(output_hvi, hv_features))


def options(parser):
    parser.add_argument('--guide', choices=('intensity', 'fixed'), default='intensity')
    parser.add_argument('--diagonal', action='store_true', help='Disable both HV cross terms')


def configure(model, config):
    model.guide = config['guide']
    model.diagonal = config['diagonal']


if __name__ == '__main__':
    main('ac', BilateralColorCIDNet, standard_backward, options, configure=configure, resume_keys=('guide', 'diagonal'))
