"""A: local reliability gate and independent-noise chroma consistency."""
import torch
from torch import nn
from torch.nn import functional as F

from net.CIDNet import CIDNet
from experiments.runner import main, noisy_views


class ReliabilityCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(3, 8, 1), nn.SiLU(), nn.Conv2d(8, 2, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def encode_hv_level2(self, hv, intensity, hvi):
        chroma = hvi[:, :2]
        local_mean = F.avg_pool2d(chroma, 5, 1, 2, count_include_pad=False)
        radius = torch.linalg.vector_norm(chroma, dim=1, keepdim=True)
        deviation = (chroma - local_mean).abs().mean(dim=1, keepdim=True)
        stats = torch.cat((hvi[:, 2:], radius, deviation), dim=1)
        stats = F.adaptive_avg_pool2d(stats, hv.shape[-2:])
        # Equal initial logits recover the pretrained residual exactly.
        keep, cross = (2 * self.gate(stats).softmax(dim=1)).chunk(2, dim=1)
        block = self.HV_LCA2
        residual = block.ffn(block.norm(hv), block.norm(intensity))
        return block.gdfn(block.norm(keep * hv + cross * residual))


def options(parser):
    parser.add_argument('--noise-std', type=float, default=0.015)
    parser.add_argument('--consistency-weight', type=float, default=0.1)


def backward(model, low, target, objective, args):
    view1, view2 = noisy_views(low, args.noise_std)
    output = model(torch.cat((view1, view2)))
    reconstruction, perceptual = objective(model, output, torch.cat((target, target)))
    first, second = model.HVIT(output)[:, :2].chunk(2)
    loss = reconstruction + perceptual + args.consistency_weight * F.l1_loss(first, second)
    loss.backward()
    return loss.detach()


if __name__ == '__main__':
    main('a', ReliabilityCIDNet, backward, options)
