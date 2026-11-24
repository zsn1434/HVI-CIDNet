"""InterLight's restoration dictionary and channel modulation on the HV encoder skip."""
import torch
from torch import nn
from torch.nn import functional as F

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


class RestorationDictionary(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=3), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, stride=3), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 1), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 32, 1),
        )
        self.logits = nn.Linear(32, 16)
        self.dictionary = nn.Parameter(torch.randn(16, 32))

    def forward(self, hvi):
        # Two valid stride-3 convolutions require at least nine input pixels.
        h, w = hvi.shape[-2:]
        if h < 9 or w < 9:
            hvi = F.pad(hvi, (0, max(0, 9 - w), 0, max(0, 9 - h)), mode='replicate')
        descriptor = self.condition(hvi).mean(dim=(-2, -1))
        weights = self.logits(descriptor).softmax(dim=1)
        return F.gelu(weights @ self.dictionary)


class DictionaryCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.prior = RestorationDictionary()
        self.affine = nn.Linear(32, 2 * 72)
        self.gain = nn.Parameter(torch.zeros(()))

    def encode_hv_level2(self, hv, intensity, hvi):
        features = super().encode_hv_level2(hv, intensity, hvi)
        scale, shift = self.affine(self.prior(hvi))[:, :, None, None].chunk(2, dim=1)
        return features + self.gain * (features * scale.sigmoid() + shift)


if __name__ == '__main__':
    main('m', DictionaryCIDNet, standard_backward)
