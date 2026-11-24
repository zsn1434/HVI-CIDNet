"""B: brightness-stratified chroma supervision with optional noise calibration."""
import torch
from torch.nn import functional as F

from net.CIDNet import CIDNet
from experiments.runner import main, noisy_views


def options(parser):
    parser.add_argument('--bands', type=int, choices=range(2, 17), default=5)
    parser.add_argument('--band-mode', choices=['uniform', 'calibrated'], default='calibrated')
    parser.add_argument('--noise-std', type=float, default=0.015)
    parser.add_argument('--band-weight', type=float, default=0.1)


def backward(model, low, target, objective, args):
    output = model(low)
    reconstruction, perceptual = objective(model, output, target)
    error = (model.HVIT(output)[:, :2] - model.HVIT(target)[:, :2]).square().mean(1, keepdim=True)
    with torch.no_grad():
        intensity = F.avg_pool2d(low.amax(1, keepdim=True), 7, 1, 3, count_include_pad=False)
        centers = torch.linspace(0, 1, args.bands, device=low.device).view(1, -1, 1, 1)
        memberships = (-2 * ((intensity - centers) * (args.bands - 1)).square()).softmax(1)
        mass = memberships.sum((0, 2, 3))
        present = mass >= 1
        memberships = memberships[:, present] / mass[present].view(1, -1, 1, 1)
    risk = (memberships * error).sum((0, 2, 3))
    if args.band_mode == 'calibrated':
        # Same-exposure synthetic replicates estimate sensitivity to added noise.
        with torch.no_grad():
            a, b = noisy_views(low, args.noise_std)
            delta = model.HVIT(a)[:, :2] - model.HVIT(b)[:, :2]
            variance = (memberships * (0.5 * delta.square().mean(1, keepdim=True))).sum((0, 2, 3))
            scale = variance.clamp_min(1e-6)
            scale = (scale.mean() / scale).clamp(0.1, 10)
        risk = risk * scale
        weights = (risk.detach() / risk.detach().mean().clamp_min(1e-6)).softmax(0)
        band_loss = (weights * risk).sum()
    else:
        band_loss = risk.mean()
    loss = reconstruction + perceptual + args.band_weight * band_loss
    loss.backward()
    return loss.detach()


if __name__ == '__main__':
    main('b', CIDNet, backward, options)
