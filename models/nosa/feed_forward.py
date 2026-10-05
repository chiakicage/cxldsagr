"""Model-independent feed-forward layers with a merged gate/up projection."""

from torch import nn
from torch.nn import functional as F

from operators import flashinfer


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, *, bias=False, device, dtype):
        super().__init__()
        args = {"device": device, "dtype": dtype, "bias": bias}
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, **args)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, **args)

    def forward(self, x):
        width = self.down_proj.in_features
        gate_up = self.gate_up_proj(x)
        if flashinfer.can_use_flashinfer(x) and width * x.element_size() % 16 == 0:
            activated = flashinfer.silu_and_mul(gate_up.reshape(-1, 2 * width)).view(
                *x.shape[:-1], width
            )
            return self.down_proj(activated)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
