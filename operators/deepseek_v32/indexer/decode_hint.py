"""The official ECHO decode threshold, separate from the prefill mean hint."""

from functools import cache

import torch

DECODE_HINT_INDEX = 1


@cache
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def update_decode_ema(Values, Offset, LAST: tl.constexpr):
        previous = tl.load(Offset + 1)
        kth = tl.load(Values + LAST)
        tl.store(Offset + 1, previous * 0.5 + kth * 0.5)

    return update_decode_ema


def update_decode_hint(values, offset):
    """Apply official EMA decay 0.5 to Q1's exact kth score in offset[1].

    This one-request ABI reserves offset[0] for the existing prefill mean.
    Both hints are owned and restored by the session's existing offset slab.
    """
    if (
        values.ndim != 2
        or values.shape[0] != 1
        or values.shape[1] < 1
        or values.dtype != torch.float32
        or values.stride(1) != 1
        or offset.shape != (16,)
        or offset.dtype != torch.float32
        or not offset.is_contiguous()
        or offset.device != values.device
    ):
        raise ValueError("expected FP32 Q1 sorted values[1,K] and offset[16]")
    if values.is_cuda:
        _kernel()[(1,)](
            values, offset, LAST=values.shape[1] - 1, num_warps=4, enable_fp_fusion=False
        )
    else:
        offset[1] = offset[1] * 0.5 + values[0, -1] * 0.5
