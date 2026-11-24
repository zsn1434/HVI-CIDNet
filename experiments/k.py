"""LAformer global LinearAttention, reimplemented for CIDNet; see experiments/README.md."""
from torch import nn
from torch.nn import functional as F

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


class RankEnhancedAttention(nn.Module):
    def __init__(self, width=72, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Conv2d(width, 3 * width, 1)
        self.local = nn.Conv2d(width, width, 5, padding=2, groups=width)
        self.output = nn.Conv2d(width, width, 1)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        local = self.local(v)

        def tokens(t):
            return t.flatten(2).unflatten(1, (self.heads, -1)).transpose(-2, -1)

        q, k, v = tokens(F.elu(q) + 1), tokens(F.elu(k) + 1), tokens(v)
        scale = q.shape[-2] ** -0.5
        context = (k.transpose(-2, -1) * scale) @ (v * scale)
        normalizer = q @ k.mean(-2, keepdim=True).transpose(-2, -1)
        mixed = (q @ context) / normalizer.clamp_min(1e-6)
        return self.output(mixed.transpose(-2, -1).reshape_as(x) + local)


class LinearAttentionCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.attention = RankEnhancedAttention()
        nn.init.zeros_(self.attention.output.weight)
        nn.init.zeros_(self.attention.output.bias)

    def decode_hv_level2(self, hv, intensity):
        features = super().decode_hv_level2(hv, intensity)
        return features + self.attention(features)


if __name__ == '__main__':
    main('k', LinearAttentionCIDNet, standard_backward)
