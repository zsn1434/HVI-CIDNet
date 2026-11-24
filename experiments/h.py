"""Supervise local RGB chromaticity with target-brightness weighting."""
from torch.nn import functional as F

from experiments.runner import main
from net.CIDNet import CIDNet


def options(parser):
    parser.add_argument('--chroma-weight', type=float, default=0.1)


def backward(model, low, target, objective, args):
    output = model(low)
    reconstruction, perceptual = objective(model, output, target)
    pred = F.avg_pool2d(output, 4)
    gt = F.avg_pool2d(target, 4)
    # Floor the denominator near black where RGB ratios are poorly defined.
    pred_ratio = pred / pred.sum(1, keepdim=True).clamp_min(0.05)
    gt_ratio = gt / gt.sum(1, keepdim=True).clamp_min(0.05)
    confidence = gt.mean(1, keepdim=True)
    error = (pred_ratio - gt_ratio).abs().mean(1, keepdim=True)
    chroma = (confidence * error).sum() / confidence.sum().clamp_min(1e-6)
    loss = reconstruction + perceptual + args.chroma_weight * chroma
    loss.backward()
    return loss.detach()


if __name__ == '__main__':
    main('h', CIDNet, backward, options)
