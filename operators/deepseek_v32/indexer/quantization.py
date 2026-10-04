"""Exact indexer FP8 quantization, with a fused BF16 D128 Hopper path.

The eager reference remains independently usable on CPU and for other input
layouts. Importing this module does not load Triton or a native extension.
"""

from functools import lru_cache

import torch


def reference_quantize_index(x: torch.Tensor, scale_fmt: str | None = "ue8m0"):
    """Preserve the model's FP32 reduction and log2/ceil UE8M0 convention."""
    if scale_fmt not in ("ue8m0", None):
        raise ValueError("Unsupported indexer quantization scale format")
    amax = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = amax / 448.0
    if scale_fmt == "ue8m0":
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    quantized = (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized.contiguous(), scale.contiguous()


@lru_cache(maxsize=1)
def _quantization_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def maximum_with_nan(a, b):
        return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)

    @triton.jit
    def quantize_kernel(X, Y, S, ROWS: tl.constexpr, UE8M0: tl.constexpr):
        rows = tl.program_id(0) * 16 + tl.arange(0, 16)
        columns = tl.arange(0, 128)
        offset = rows[:, None] * 128 + columns[None, :]
        values = tl.load(X + offset, rows[:, None] < ROWS, 0).to(tl.float32)
        maximum = tl.maximum(
            tl.reduce(tl.abs(values), 1, maximum_with_nan),
            1e-4,
            propagate_nan=tl.PropagateNan.ALL,
        )
        # PyTorch's CUDA scalar divide multiplies by the FP32 reciprocal.
        scale = maximum * (1.0 / 448.0)
        if UE8M0:
            # On the BF16 amax domain this equals exp2(ceil(log2(scale))).
            # Preserve the reference's infinity and canonical NaN behavior.
            bits = scale.to(tl.int32, bitcast=True)
            rounded = ((bits + 0x007FFFFF) & 0x7F800000).to(tl.float32, bitcast=True)
            scale = tl.where(bits > 0x7F800000, scale, rounded)
            inverse_scale = tl.div_rn(1.0, scale)
            normalized = values * inverse_scale[:, None]
        else:
            normalized = tl.div_rn(values, scale[:, None])
        normalized = tl.clamp(normalized, -448.0, 448.0, propagate_nan=tl.PropagateNan.ALL)
        tl.store(Y + offset, normalized, rows[:, None] < ROWS)
        tl.store(S + rows, scale, rows < ROWS)

    return quantize_kernel


def quantize_index(x: torch.Tensor, scale_fmt: str | None = "ue8m0"):
    """Quantize each last-dimension row, returning contiguous FP8 and FP32.

    Contiguous BF16 D128 CUDA inputs use one kernel on SM90. Other inputs use
    the original eager arithmetic, including non-contiguous views and CPU.
    """
    if scale_fmt not in ("ue8m0", None):
        raise ValueError("Unsupported indexer quantization scale format")
    if (
        x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or x.ndim == 0
        or x.shape[-1] != 128
        or not x.is_contiguous()
        or torch.cuda.get_device_capability(x.device) != (9, 0)
    ):
        return reference_quantize_index(x, scale_fmt)
    data = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((*x.shape[:-1], 1), dtype=torch.float32, device=x.device)
    rows = x.numel() // 128
    if rows:
        with torch.cuda.device(x.device):
            _quantization_kernel()[((rows + 15) // 16,)](
                x, data, scales, rows, scale_fmt == "ue8m0", num_warps=4
            )
    return data, scales
