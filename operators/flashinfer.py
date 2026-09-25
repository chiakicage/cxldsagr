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
