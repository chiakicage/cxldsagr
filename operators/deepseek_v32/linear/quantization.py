"""Direct SM90 activation quantization with lazy Triton compilation."""

from pathlib import Path

import torch

from ._quantization_identity import Identity

POLICY = {
    "revision": "deepseek_linear_triton_t1_groups32_warps2_original_division",
    "target": "sm_90",
    "group_size": 128,
    "groups_per_cta": 32,
    "num_warps": 2,
    "num_stages": 1,
    "enable_fp_fusion": False,
    "input": "bf16_fp16_fp32_nonnegative_strides_partial_k",
    "output": "independent_contiguous_e4m3fn_and_row_major_fp32_scales",
    "arithmetic": "nan_propagating_max_ue8m0_div_full_satfinite",
}
_identity = Identity(
    __file__, Path(__file__).resolve().with_name("_quantization_kernel.py"), POLICY
)


def build_info():
    return _identity.build_info()


def runtime_info():
    return _identity.runtime_info()


def _launch(x, data, scales):
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
                num_warps=2,
                num_stages=1,
                enable_fp_fusion=False,
            )
        _identity.observe(compiled)


def quantize(x):
    if x.ndim != 2 or not x.shape[1]:
        raise ValueError("Expected [rows, positive input channels] activation")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Activations must be BF16, FP16, or FP32")
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) != (9, 0):
        raise NotImplementedError("Direct Triton quantization requires SM90/Hopper")
    rows, columns = x.shape
    data = torch.empty((rows, columns), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((rows, (columns + 127) // 128), dtype=torch.float32, device=x.device)
    _launch(x, data, scales)
    return data, scales
