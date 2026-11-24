"""C2SSM cluster routing and selective recurrence in plain PyTorch."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from experiments.runner import main, standard_backward
from net.CIDNet import CIDNet


class ClusterScan(nn.Module):
    def __init__(self):
        super().__init__()
        width, inner, rank, state = 36, 36, 3, 8
        self.heads = 2
        self.in_proj = nn.Linear(width, 2 * inner, bias=False)
        self.local = nn.Conv2d(inner, inner, 3, padding=1, groups=inner)
        self.keys = nn.Conv2d(inner, inner * self.heads, 1)
        self.values = nn.Conv2d(inner, inner * self.heads, 1)
        self.merge = nn.Conv2d(inner * self.heads, inner, 1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.scan_inputs = nn.Linear(inner, rank + 2 * state, bias=False)
        self.scan_mix = nn.Conv1d(rank + 2 * state, rank + 2 * state, 7, padding=3,
                                  groups=rank + 2 * state)
        self.dt = nn.Linear(rank, inner)
        self.log_a = nn.Parameter(torch.arange(1, state + 1).float().log().repeat(inner, 1))
        self.direct = nn.Parameter(torch.ones(inner))
        self.norm = nn.LayerNorm(inner)
        self.out = nn.Linear(inner, width, bias=False)
        with torch.no_grad():
            self.dt.weight.uniform_(-rank ** -0.5, rank ** -0.5)
            dt = torch.empty(inner).uniform_(math.log(0.001), math.log(0.1)).exp()
            self.dt.bias.copy_(dt + (-dt).expm1().neg().log())

    def scan(self, centers):
        projected = self.scan_mix(self.scan_inputs(centers).transpose(1, 2)).transpose(1, 2)
        dt, b, c = projected.split((3, 8, 8), dim=-1)
        dt = F.softplus(self.dt(dt)).float()
        a = -self.log_a.float().exp()
        hidden = centers.new_zeros((centers.shape[0], a.shape[0], a.shape[1]), dtype=torch.float32)
        outputs = []
        for step in range(centers.shape[1]):
            delta = dt[:, step, :, None]
            token = centers[:, step, :].float()
            hidden = (delta * a).exp() * hidden + delta * b[:, step, None, :].float() * token[..., None]
            outputs.append((hidden * c[:, step, None, :].float()).sum(-1) + self.direct * token)
        return torch.stack(outputs, dim=1).to(centers.dtype)

    def forward(self, x, scan=True):
        features, gate = self.in_proj(x.permute(0, 2, 3, 1)).chunk(2, dim=-1)
        features = F.silu(self.local(features.permute(0, 3, 1, 2)))
        batch, channels, height, width = features.shape
        keys = self.keys(features).reshape(-1, channels, height, width)
        values = self.values(features).reshape_as(keys)
        pooled_keys = F.adaptive_avg_pool2d(keys, (2, 2)).flatten(2).transpose(1, 2)
        pooled_values = F.adaptive_avg_pool2d(values, (2, 2)).flatten(2).transpose(1, 2)
        scores = (self.beta + self.alpha * (F.normalize(pooled_keys, dim=-1)
                                            @ F.normalize(keys.flatten(2), dim=1))).sigmoid()
        strength, index = scores.max(dim=1, keepdim=True)
        assignment = torch.zeros_like(scores).scatter(1, index, strength)
        centers = (assignment @ values.flatten(2).transpose(1, 2) + pooled_values)
        centers = centers / (1 + assignment.sum(dim=-1, keepdim=True))
        if scan:
            centers = self.scan(centers)
        context = (centers.transpose(1, 2) @ assignment).reshape(batch, -1, height, width)
        context = self.merge(context).permute(0, 2, 3, 1)
        return self.out(self.norm(context) * F.silu(gate)).permute(0, 3, 1, 2)


class ClusterCIDNet(CIDNet):
    def __init__(self):
        super().__init__()
        self.cluster = ClusterScan()
        self.use_scan = True
        self.gain = nn.Parameter(torch.zeros(()))

    def refine_decoder_intensity(self, intensity, hv):
        return intensity + self.gain * self.cluster(intensity, self.use_scan)


def options(parser):
    parser.add_argument("--cluster-only", action="store_true", help="Aggregate and dispatch without SSM")


def configure(model, config):
    model.use_scan = not config["cluster_only"]


if __name__ == "__main__":
    main("s", ClusterCIDNet, standard_backward, options, configure=configure,
         resume_keys=("cluster_only",))
