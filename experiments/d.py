"""D: fidelity/detail adapters with output-subspace gradient projection."""
import torch
from torch import nn

from net.CIDNet import CIDNet
from experiments.runner import main


class DualAdapterCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.fidelity = self.adapter()
        self.detail = self.adapter()

    @staticmethod
    def adapter():
        layers = nn.Sequential(nn.Conv2d(72, 8, 1, bias=False), nn.Conv2d(8, 72, 1, bias=False))
        nn.init.zeros_(layers[-1].weight)
        return layers

    def decode_hv_level2(self, hv, intensity):
        features = super().decode_hv_level2(hv, intensity)
        return features + self.fidelity(features) + self.detail(features)


def options(parser):
    parser.add_argument('--disable-projection', action='store_true', help='Control with identical adapters')


def backward(model, low, target, objective, args):
    reconstruction, perceptual = objective(model, model(low), target)
    if not perceptual.requires_grad:
        raise ValueError('D requires a positive --perceptual-weight')
    fidelity = list(model.fidelity.parameters())
    detail = list(model.detail.parameters())
    fidelity_grad = torch.autograd.grad(reconstruction, fidelity, retain_graph=True)
    detail_grad = torch.autograd.grad(perceptual, detail, retain_graph=True)
    loss = reconstruction + perceptual
    loss.backward()
    # Backbone gets both losses; each adapter gets only its designated objective.
    for parameter, gradient in zip(fidelity + detail, fidelity_grad + detail_grad):
        parameter.grad = gradient
    if not args.disable_projection:
        with torch.no_grad():
            up = model.fidelity[-1].weight.flatten(1)
            u, s, _ = torch.linalg.svd(up, full_matrices=False)
            basis = u[:, s > s.max() * 1e-5]
            grad = model.detail[-1].weight.grad.flatten(1)
            grad.sub_(basis @ (basis.T @ grad))
    return loss.detach()


if __name__ == '__main__':
    main('d', DualAdapterCIDNet, backward, options)
