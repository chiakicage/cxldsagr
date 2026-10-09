"""Direct SM90 activation quantization with lazy Triton compilation."""

from pathlib import Path

import torch

from ._quantization_identity import Identity

POLICY = {
    "revision": "deepseek_linear_triton_t1_consumer_scale_layout_01",
    "target": "sm_90",
    "group_size": 128,
    "groups_per_cta": 32,
    "num_warps": 2,
    "num_stages": 1,
    "enable_fp_fusion": False,
    "input": "bf16_fp16_fp32_nonnegative_strides_partial_k",
    "output": "owned_contiguous_e4m3fn_default_row_major_fp32_opt_in_stride_1_aligned_rows",
    "scale_layout": "default_contiguous_or_explicit_sm90_consumer_layout",
    "arithmetic": "nan_propagating_max_ue8m0_div_full_satfinite",
}
_identity = Identity(
    __file__, Path(__file__).resolve().with_name("_quantization_kernel.py"), POLICY
)


def build_info():
    return _identity.build_info()


def runtime_info():
    return _identity.runtime_info()


def _launch(x, data, scales, *, consumer_scale_layout=False):
    _identity.capture()
    from ._quantization_kernel import activation_quantization

    rows, columns = x.shape
    if rows:
        with torch.cuda.device(x.device):
            compiled = activation_quantization[((rows * ((columns + 127) // 128) + 31) // 32,)](
                x,
                data,
                scales,
                rows,
                columns,
                x.stride(0),
                x.stride(1),
                x.stride() == (columns, 1) and columns % 128 == 0,
                32,
                scales.stride(1) if consumer_scale_layout else 0,
                num_warps=2,
                num_stages=1,
                enable_fp_fusion=False,
            )
        _identity.observe(compiled)


def quantize(x, *, consumer_scale_layout=False):
    if type(consumer_scale_layout) is not bool:
        raise TypeError("consumer_scale_layout must be a boolean")
    if x.ndim != 2 or not x.shape[1]:
        raise ValueError("Expected [rows, positive input channels] activation")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Activations must be BF16, FP16, or FP32")
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) != (9, 0):
        raise NotImplementedError("Direct Triton quantization requires SM90/Hopper")
    rows, columns = x.shape
    data = torch.empty((rows, columns), dtype=torch.float8_e4m3fn, device=x.device)
    scale_shape = (rows, (columns + 127) // 128)
    if consumer_scale_layout:
        scales = torch.empty_strided(
            scale_shape, (1, ((rows + 3) // 4) * 4), dtype=torch.float32, device=x.device
        )
    else:
        scales = torch.empty(scale_shape, dtype=torch.float32, device=x.device)
    _launch(x, data, scales, consumer_scale_layout=consumer_scale_layout)
    return data, scales
