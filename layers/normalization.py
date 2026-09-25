"""Model-independent normalization layers."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, size, eps, *, device, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(x.dtype)
