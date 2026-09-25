"""Model-independent feed-forward layers with checkpoint-friendly parameters."""

from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, *, bias=False, device, dtype):
        super().__init__()
        args = {"device": device, "dtype": dtype, "bias": bias}
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, **args)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, **args)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, **args)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
