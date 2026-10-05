"""Model-independent normalization layers."""

import torch
from torch import nn

from operators import flashinfer


class RMSNorm(nn.Module):
    """Inference RMSNorm, optionally consuming a pending residual addition.

    With residual, return (normalized, sum). The FlashInfer path overwrites both
    inputs, so callers must own distinct activation buffers and discard x after
    this call. CPU/FP32/autograd retain the eager reference equations.
    """

    def __init__(self, size, eps, *, device, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x, residual=None):
        # The generic layer also accepts shapes/layouts outside FlashInfer's
        # contract. Restrict the fast path to owned, aligned dense activations.
        use_kernel = (
            flashinfer.can_use_flashinfer(x)
            and x.ndim in (2, 3)
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
            and self.weight.dtype == x.dtype
            and self.weight.device == x.device
            and self.weight.is_contiguous()
            and self.weight.data_ptr() % 16 == 0
            and (
                residual is None
                or (
                    x.ndim == 2
                    and residual.shape == x.shape
                    and residual.dtype == x.dtype
                    and residual.device == x.device
                    and residual.is_contiguous()
                    and residual.data_ptr() % 16 == 0
                )
            )
        )
        if use_kernel:
            if residual is None:
                return flashinfer.rmsnorm(x, self.weight, self.eps)
            return flashinfer.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        if residual is not None:
            x = x + residual
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        output = self.weight * normalized.to(x.dtype)
        return output if residual is None else (output, x)
