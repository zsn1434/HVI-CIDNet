"""Mine high-MSE patches independently within each training image."""
import math

from torch.nn import functional as F

from experiments.runner import main
from net.CIDNet import CIDNet


def options(parser):
    parser.add_argument('--hard-weight', type=float, default=0.5)
    parser.add_argument('--hard-fraction', type=float, default=0.25)


def backward(model, low, target, objective, args):
    output = model(low)
    reconstruction, perceptual = objective(model, output, target)
    squared_error = (output - target).square().mean(1, keepdim=True)
    patch_mse = F.avg_pool2d(squared_error, 8).flatten(1)
    count = math.ceil(args.hard_fraction * patch_mse.size(1))
    hard_loss = patch_mse.topk(count, dim=1, sorted=False).values.mean()
    loss = reconstruction + perceptual + args.hard_weight * hard_loss
    loss.backward()
    return loss.detach()


if __name__ == '__main__':
    main('i', CIDNet, backward, options)
