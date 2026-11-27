"""AD: train a shared HV stem on disjoint next-scale RGB samples from the low image."""
import argparse
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from experiments.runner import load_model, main, save_checkpoint
from net.CIDNet import CIDNet


def next_scale_pair(low, visible=False):
    # Discard only the incomplete right/bottom tiles of the auxiliary input.
    tiles = low.unfold(2, 5, 5).unfold(3, 5, 5).flatten(-2)
    batch, channels, rows, cols, _ = tiles.shape
    shape = (batch, 1, rows, cols, 1)
    y = torch.randint(4, shape, device=low.device)
    x = torch.randint(4, shape, device=low.device)
    target_ids = y * 5 + x + low.new_tensor([0, 1, 5, 6], dtype=torch.long)
    scores = torch.rand(batch, 1, rows, cols, 25, device=low.device)
    input_id = scores.scatter_(-1, target_ids, -1.).argmax(-1, keepdim=True)
    sample = tiles.gather(-1, input_id.expand(-1, channels, -1, -1, -1)).squeeze(-1)
    if visible:
        target = sample.repeat_interleave(2, -2).repeat_interleave(2, -1)
    else:
        patch = tiles.gather(-1, target_ids.expand(-1, channels, -1, -1, -1))
        target = patch.reshape(batch, channels, rows, cols, 2, 2).permute(0, 1, 2, 4, 3, 5)
        target = target.reshape(batch, channels, rows * 2, cols * 2)
    return sample, target


class NextScaleCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.next_scale = nn.Sequential(nn.Conv2d(36, 12, 1), nn.PixelShuffle(2))

    def predict_next(self, sample):
        return self.next_scale(self.HVE_block0(self.HVIT(sample)))


def options(parser):
    parser.add_argument('--aux-target', choices=('heldout', 'visible'), default='heldout')
    parser.add_argument('--nsp-weight', type=float, default=0.1)


def configure(model, config):
    if not math.isfinite(config['nsp_weight']) or config['nsp_weight'] < 0:
        raise ValueError('--nsp-weight must be finite and nonnegative')


def backward(model, low, target, objective, args):
    reconstruction, perceptual = objective(model, model(low), target)
    loss = reconstruction + perceptual
    if args.nsp_weight:
        sample, next_scale = next_scale_pair(low, visible=args.aux_target == 'visible')
        loss = loss + args.nsp_weight * F.l1_loss(model.predict_next(sample), next_scale)
    loss.backward()
    return loss.detach()


def export():
    parser = argparse.ArgumentParser(description='Export AD as a plain CIDNet state_dict')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(sys.argv[2:])
    if args.out.exists() or args.out.suffix == '.tmp':
        parser.error('Use a new output filename with a suffix other than .tmp')
    model = NextScaleCIDNet()
    load_model(args.checkpoint, model, 'ad')
    del model.next_scale
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model.state_dict(), args.out)


if __name__ == '__main__':
    if sys.argv[1:2] == ['export']:
        export()
    else:
        main('ad', NextScaleCIDNet, backward, options, configure=configure, resume_keys=('aux_target', 'nsp_weight'))
