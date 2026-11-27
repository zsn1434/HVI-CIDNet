"""Intervention training on quarter-resolution I/HV features, inspired by ISFuse."""
import torch
from torch.nn import functional as F

from experiments.runner import main
from net.CIDNet import CIDNet


class InterventionCIDNet(CIDNet):
    def decode_quarter(self, hv, intensity, hvi, context, auxiliary, skips):
        clean = super().decode_quarter(hv, intensity, hvi, context, auxiliary, skips)
        if auxiliary is None:
            return clean
        # Split a shuffled coarse tile grid into equally sized disjoint sets.
        b, _, h, w = hv.shape
        rows, cols = min(4, h), min(4, w)
        ranks = torch.rand(b, rows * cols, device=hv.device).argsort(1).reshape(b, 1, rows, cols)
        count = max(1, rows * cols // 8)
        def expand(mask):
            return F.interpolate(mask.float(), size=(h, w), mode='nearest')
        color_mask = expand(ranks < count)
        light_mask = expand((ranks >= count) & (ranks < 2 * count))
        if auxiliary == 'dropout':
            # Independent pixel dropout at the same expected masking rate.
            color_mask, light_mask = [(torch.rand_like(color_mask) < count / (rows * cols)).float()
                                      for _ in range(2)]
        perturbed = super().decode_quarter(hv * (1 - color_mask), intensity * (1 - light_mask),
                                           hvi, context, None, skips)
        mask = F.interpolate(torch.maximum(color_mask, light_mask), size=clean.shape[-2:], mode='nearest')
        return clean, perturbed, mask


def backward(model, low, target, objective, args):
    if torch.rand(()).item() < args.intervention_prob:
        clean, perturbed, mask = model.enhance(*model.input_features(low), auxiliary=args.mask_mode)
        extra = ((perturbed - target).abs() * mask).sum() / (3 * mask.sum()).clamp_min(1)
    else:
        clean, extra = model(low), 0
    reconstruction, perceptual = objective(model, clean, target)
    loss = reconstruction + perceptual + args.mask_weight * extra
    loss.backward()
    return loss.detach()


def options(parser):
    parser.add_argument('--mask-mode', choices=('complementary', 'dropout'), default='complementary')
    parser.add_argument('--intervention-prob', type=float, default=0.5)
    parser.add_argument('--mask-weight', type=float, default=0.1)


def configure(model, config):
    import math
    if not 0 <= config['intervention_prob'] <= 1 or not 0 <= config['mask_weight'] < math.inf:
        raise ValueError('Invalid intervention probability or mask weight')


if __name__ == '__main__':
    main('ah', InterventionCIDNet, backward, options, configure=configure,
         resume_keys=('mask_mode', 'intervention_prob', 'mask_weight'))
