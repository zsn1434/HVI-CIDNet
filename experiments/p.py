"""Test the existing I_LCA5 output as a gated residual before ID_block2."""
import torch
from torch import nn

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


class IntensitySkipCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(()))

    def select_decoder_intensity(self, intensity, attended):
        return intensity + self.gain * (attended - intensity)


if __name__ == '__main__':
    main('p', IntensitySkipCIDNet, standard_backward)
