"""Thin adapters for FlashInfer inference kernels."""

import torch


def can_use_flashinfer(x: torch.Tensor) -> bool:
    """Ordinary layers keep the eager path for CPU, FP32 and autograd."""
    return (
        x.device.type == "cuda"
        and x.dtype in (torch.bfloat16, torch.float16)
        and not torch.is_grad_enabled()
    )


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    from flashinfer.norm import rmsnorm as kernel

    with torch.cuda.device(x.device):
        return kernel(x, weight, eps=eps)


def fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Overwrite x with normalized output and residual with their sum."""
    from flashinfer.norm import fused_add_rmsnorm as kernel

    with torch.cuda.device(x.device):
        kernel(x, residual, weight, eps=eps)
    return x, residual


def silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """Consume contiguous [..., gate | up] projections without a packing copy."""
    from flashinfer.activation import silu_and_mul as kernel

    with torch.cuda.device(gate_up.device):
        return kernel(gate_up)


def apply_rope_with_cos_sin_cache(q, k, positions, cos_sin_cache):
    """Rotate NHD Q/K in place, preserving packed QKV token-row strides."""
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

    head_size = q.shape[-1]
    with torch.cuda.device(q.device):
        # Flatten only adjacent head/feature axes. These views preserve token
        # strides and storage offsets; FlashInfer forwards both to the kernel.
        apply_rope_with_cos_sin_cache_inplace(
            positions,
            q.view(q.shape[0], -1),
            k.view(k.shape[0], -1),
            head_size,
            cos_sin_cache,
            is_neox=True,
        )
    return q, k


def rotary_pair(q, k, positions, cos_sin_cache, *, interleaved=False):
    """Out-of-place Q/K RoPE with explicit model-supplied FP32 trig values.

    The cache width determines the rotary prefix; any remaining head features
    pass through unchanged. Reshaping may pack sliced head dimensions, and that
    copy is part of the adapter cost.
    """
    from flashinfer.rope import apply_rope_with_cos_sin_cache as kernel

    with torch.cuda.device(q.device):
        query, key = kernel(
            positions,
            q.reshape(q.shape[0], -1),
            k.reshape(k.shape[0], -1),
            q.shape[-1],
            cos_sin_cache,
            is_neox=not interleaved,
        )
    return query.reshape(q.shape), key.reshape(k.shape)


def _rotary_vector_aligned(tensor):
    """The D64 vendor kernel loads and stores one 16-byte vector per lane."""
    return (
        tensor.stride(-1) == 1
        and tensor.data_ptr() % 16 == 0
        and all(stride % 8 == 0 for stride in tensor.stride()[:-1])
    )


def rotary_pair_into(q, k, positions, cos_sin_cache, q_out, *, interleaved=False):
    """Return Q in its supplied destination and an independently owned K.

    Aligned interleaved D64 inputs use the same vendor RoPE kernel with its
    independent input/output strides. Other layouts keep the ordinary adapter
    and copy its Q result into the destination. The destination cannot share
    storage with either input.
    """
    if (
        q.ndim != 3
        or k.ndim != 2
        or k.shape != (q.shape[0], q.shape[-1])
        or q.dtype != k.dtype
        or q.device != k.device
        or q_out.shape != q.shape
        or q_out.dtype != q.dtype
        or q_out.device != q.device
    ):
        raise ValueError("rotary destination requires matching NHD Q and ND K tensors")
    if (
        positions.shape != (q.shape[0],)
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.device != q.device
        or not positions.is_contiguous()
        or cos_sin_cache.ndim != 2
        or cos_sin_cache.dtype != torch.float32
        or cos_sin_cache.device != q.device
        or not cos_sin_cache.is_contiguous()
    ):
        raise ValueError("rotary positions and FP32 trig cache must match the input device")
    if q_out.untyped_storage().data_ptr() in {
        q.untyped_storage().data_ptr(),
        k.untyped_storage().data_ptr(),
    }:
        raise ValueError("rotary Q destination must not share storage with Q or K inputs")
    direct = (
        can_use_flashinfer(q)
        and interleaved is True
        and q.shape[-1] == cos_sin_cache.shape[-1] == 64
        and all(_rotary_vector_aligned(tensor) for tensor in (q, k, q_out))
        and q_out.stride(1) >= 64
        and q_out.stride(0) >= q.shape[1] * q_out.stride(1)
    )
    if not direct:
        query, key = rotary_pair(q, k, positions, cos_sin_cache, interleaved=interleaved)
        q_out.copy_(query)
        return q_out, key

    from flashinfer.rope import _apply_rope_pos_ids_cos_sin_cache

    with torch.cuda.device(q.device):
        key = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        _apply_rope_pos_ids_cos_sin_cache(
            q, k.unsqueeze(1), q_out, key.unsqueeze(1), cos_sin_cache, positions, True
        )
    return q_out, key


class FlashInferFullAttention:
    """Single-request GQA over NHD K/V, already rotated by model LongRoPE."""

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if q.device.type != "cuda" or q.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("FlashInfer attention requires CUDA and bfloat16 or float16 tensors")
        import flashinfer

        with torch.cuda.device(q.device):
            if q.shape[0] == 1:
                return flashinfer.single_decode_with_kv_cache(
                    q[0],
                    k,
                    v,
                    kv_layout="NHD",
                    pos_encoding_mode="NONE",
                    use_tensor_cores=True,
                    sm_scale=q.shape[-1] ** -0.5,
                ).unsqueeze(0)
            # FlashInfer's causal mask aligns Q to the end of K. This also
            # handles a prefill chunk appended to an existing prefix cache.
            return flashinfer.single_prefill_with_kv_cache(
                q,
                k,
                v,
                causal=True,
                kv_layout="NHD",
                pos_encoding_mode="NONE",
                sm_scale=q.shape[-1] ** -0.5,
                backend="auto",
            )
