"""DeepSeek ordinary operations with the checkpoint's FP32 arithmetic contract.

GPU inference uses FlashInfer and model-specific typed norm I/O. Norm weights
and pending residual sums stay FP32. Eligible BF16 Hopper inputs avoid full
FP32 activation copies while preserving the official CuTe computation; other
inputs retain the ordinary adapter. CPU and autograd use reference equations.
"""

import torch
from torch.nn import functional as F

from operators import flashinfer
from operators.deepseek_v32.norm import api as typed_norm


def _gpu_inference(x):
    return x.device.type == "cuda" and not torch.is_grad_enabled()


def _reference_rms_norm(x, weight, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def rms_norm(x, weight, eps):
    """Normalize with FP32 weights and accumulators, then restore input dtype."""
    if not _gpu_inference(x) or not x.numel():
        return _reference_rms_norm(x, weight, eps)
    if typed_norm.can_use_typed_norm(x, weight):
        return typed_norm.rms_norm(x, weight, eps)
    activation = x.float().reshape(-1, x.shape[-1]).contiguous()
    output = flashinfer.rmsnorm(activation, weight.float().contiguous(), eps)
    return output.reshape(x.shape).to(x.dtype)


def residual_rms_norm(x, residual, weight, eps):
    """Return normalized unrounded FP32 sum and a separately rounded residual.

    Neither input is overwritten. The ordinary official fused API uses owned
    FP32 copies; the eligible typed path widens in registers. Both returned
    tensors have the input dtype, matching the separate DeepSeek residual stream.
    """
    if residual is None:
        return rms_norm(x, weight, eps), x
    if residual.shape != x.shape or residual.dtype != x.dtype or residual.device != x.device:
        raise ValueError("Residual must match input shape, dtype, and device")
    if not _gpu_inference(x) or not x.numel():
        summed = x.float() + residual.float()
        return _reference_rms_norm(summed, weight, eps).to(x.dtype), summed.to(x.dtype)
    if typed_norm.can_use_typed_norm(x, weight, residual):
        return typed_norm.residual_rms_norm(x, residual, weight, eps)
    activation = x.to(dtype=torch.float32, copy=True).reshape(-1, x.shape[-1]).contiguous()
    saved = residual.to(dtype=torch.float32, copy=True).reshape(activation.shape).contiguous()
    normalized, summed = flashinfer.fused_add_rmsnorm(
        activation, saved, weight.float().contiguous(), eps
    )
    return normalized.reshape(x.shape).to(x.dtype), summed.reshape(x.shape).to(x.dtype)


def silu_mul(gate, up):
    """FP32 SiLU and multiplication with a single final BF16 conversion.

    Checkpoint projections are separate tensors, so packing them into the
    official [gate | up] layout is part of this call and its measured cost.
    """
    if gate.shape != up.shape or gate.dtype != up.dtype or gate.device != up.device:
        raise ValueError("Gate and up projections must match shape, dtype, and device")
    if (
        flashinfer.can_use_flashinfer(gate)
        and gate.dtype == torch.bfloat16
        and gate.numel()
        and gate.shape[-1] * gate.element_size() % 16 == 0
    ):
        packed = torch.cat((gate, up), dim=-1)
        return flashinfer.silu_and_mul(packed).bfloat16()
    return (F.silu(gate.float()) * up.float()).bfloat16()


def silu_mul_packed(packed):
    """Consume an existing [gate | up] allocation with the same FP32 SiLU math."""
    if packed.ndim < 1 or not packed.shape[-1] or packed.shape[-1] % 2:
        raise ValueError("Packed SiLU requires a positive even final dimension")
    width = packed.shape[-1] // 2
    if (
        flashinfer.can_use_flashinfer(packed)
        and packed.dtype == torch.bfloat16
        and packed.ndim == 2
        and packed.numel()
        and packed.is_contiguous()
        and packed.data_ptr() % 16 == 0
        and width * packed.element_size() % 16 == 0
    ):
        return flashinfer.silu_and_mul(packed).bfloat16()
    return silu_mul(*packed.split(width, dim=-1))
