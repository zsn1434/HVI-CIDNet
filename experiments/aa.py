"""AA: predict full-image exposure statistics and condition the aligned I decoder crop."""
import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from experiments.runner import Pairs, main
from net.CIDNet import CIDNet


def thumbnail(image):
    return F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False, antialias=True)


class ExposurePairs(Pairs):
    def __init__(self, args):
        # Load paired full frames before selecting a training window.
        super().__init__(args.train_dir)
        self.window = args.crop
        self.local = args.context_mode == 'crop'

    def __getitem__(self, index):
        low, target = super().__getitem__(index)
        h, w = low.shape[-2:]
        size = self.window
        if min(h, w) < size:
            raise ValueError(f'{self.pairs[index][0]} smaller than --crop {size}')
        x, y = random.randint(0, w - size), random.randint(0, h - size)
        if random.random() < 0.5:
            low, target = low.flip(-1), target.flip(-1)
            x = w - x - size
        if random.random() < 0.5:
            low, target = low.flip(-2), target.flip(-2)
            y = h - y - size
        low_crop = low[:, y:y + size, x:x + size]
        gt_crop = target[:, y:y + size, x:x + size]
        box = (x / w, y / h, (x + size) / w, (y + size) / h)
        if self.local:
            low, target = low_crop, gt_crop
            box = (0., 0., 1., 1.)
        stats = torch.cat((target.mean(0, keepdim=True), target.amax(0, keepdim=True)))
        return low_crop, thumbnail(low[None])[0], thumbnail(stats[None])[0], low.new_tensor(box), gt_crop


def sample_exposure(exposure, boxes, output_shape):
    h, w = output_shape
    left, top, right, bottom = boxes[:, :, None, None].unbind(1)
    xs = (torch.arange(w, device=boxes.device, dtype=boxes.dtype) + 0.5) / w
    ys = (torch.arange(h, device=boxes.device, dtype=boxes.dtype) + 0.5) / h
    x = left + (right - left) * xs
    y = top + (bottom - top) * ys[:, None]
    grid = torch.stack(torch.broadcast_tensors(x, y), dim=-1) * 2 - 1
    return F.grid_sample(exposure, grid, padding_mode='border', align_corners=False)


class ExposureCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.context_mode = 'aligned'
        self.predictor = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.SiLU(),
                                       nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, 2, 1))
        self.inject = nn.Conv2d(2, 36, 1, bias=False)
        nn.init.zeros_(self.inject.weight)

    def condition_decoder_intensity(self, intensity, hv, context):
        exposure, boxes = context
        if self.context_mode == 'shifted':
            # Displace only the conditioning map. Supervision retains true image coordinates.
            exposure = exposure.roll((32, 32), dims=(-2, -1))
        return intensity + self.inject(sample_exposure(exposure, boxes, intensity.shape[-2:]))

    def forward(self, rgb):
        boxes = rgb.new_tensor([0, 0, 1, 1]).expand(len(rgb), -1)
        context = (self.predictor(thumbnail(rgb)), boxes)
        return self.enhance(*self.input_features(rgb), context=context)


def backward(model, low, small, target_stats, boxes, target, objective, args):
    exposure = model.predictor(small)
    output = model.enhance(*model.input_features(low), context=(exposure, boxes))
    reconstruction, perceptual = objective(model, output, target)
    loss = reconstruction + perceptual + args.exposure_weight * F.l1_loss(exposure, target_stats)
    loss.backward()
    return loss.detach()


def options(parser):
    parser.add_argument('--context-mode', choices=('aligned', 'crop', 'shifted'), default='aligned')
    parser.add_argument('--exposure-weight', type=float, default=0.1)


def configure(model, config):
    if not math.isfinite(config['exposure_weight']) or config['exposure_weight'] < 0:
        raise ValueError('--exposure-weight must be finite and nonnegative')
    model.context_mode = config['context_mode']


if __name__ == '__main__':
    main('aa', ExposureCIDNet, backward, options, configure=configure, train_dataset=ExposurePairs,
         resume_keys=('context_mode', 'exposure_weight'))
