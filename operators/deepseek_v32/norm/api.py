"""BF16 norm I/O with the official Float32 CuTe arithmetic on Hopper.

CuTe and vendor kernel modules are loaded only by an eligible CUDA invocation.
Other model inputs retain the ordinary nonmatrix adapter paths.
"""

from functools import cache

import torch


@cache
def _is_sm90(device):
    return torch.cuda.get_device_capability(device) == (9, 0)


def _supported_layout(value):
    width = value.shape[1]
    return value.data_ptr() % 128 == 0 and (
        value.is_contiguous() or value.stride() == (width + 64, 1)
    )


def can_use_typed_norm(x, weight, residual=None):
    """Select measured BF16 widths/layouts; leave other model inputs unchanged."""
    widths = (512, 1536, 7168) if residual is None else (7168,)
    if (
        torch.is_grad_enabled()
        or x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or not x.shape[0]
        or x.shape[1] not in widths
        or weight.device != x.device
        or weight.dtype != torch.float32
        or weight.shape != (x.shape[1],)
        or not weight.is_contiguous()
        or weight.data_ptr() % 16
        or not _supported_layout(x)
    ):
        return False
    if residual is not None and (
        residual.shape != x.shape
        or residual.dtype != x.dtype
        or residual.device != x.device
        or not _supported_layout(residual)
    ):
        return False
    return _is_sm90(x.device)


def rms_norm(x, weight, eps):
    """Return one owned BF16 tensor with FP32 weights/reduction."""
    if not can_use_typed_norm(x, weight):
        raise ValueError("Typed RMSNorm requires supported BF16 SM90 inference inputs")
    import tvm_ffi

    from operators.deepseek_v32.norm._compile import get_kernel

    source = x.contiguous()
    output = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    with torch.cuda.device(x.device), tvm_ffi.use_torch_stream():
        compiled = get_kernel("plain", x.shape[1], x.device.index)
        compiled(source, weight, output, len(source), eps)
    return output


def residual_rms_norm(x, residual, weight, eps):
    """Normalize the unrounded FP32 sum; separately store an owned BF16 sum."""
    if residual is None or not can_use_typed_norm(x, weight, residual):
        raise ValueError("Typed fused RMSNorm requires supported BF16 SM90 inference inputs")
    import tvm_ffi

    from operators.deepseek_v32.norm._compile import get_kernel

    source, saved_input = x.contiguous(), residual.contiguous()
    output, saved = (torch.empty(x.shape, device=x.device, dtype=torch.bfloat16) for _ in range(2))
    with torch.cuda.device(x.device), tvm_ffi.use_torch_stream():
        compiled = get_kernel("fused", x.shape[1], x.device.index)
        compiled(source, saved_input, weight, output, saved, len(source), eps)
    return output, saved
