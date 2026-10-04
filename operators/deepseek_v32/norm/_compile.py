"""Compile the typed norm kernels with a source identity in the CuTe module."""

from functools import cache

import cutlass
import torch
from cutlass import cute
from cutlass._mlir import ir
from flashinfer.utils import device_support_pdl

from operators.deepseek_v32.norm._fingerprint import build_info
from operators.deepseek_v32.norm._fused import LocalFusedRMSNorm
from operators.deepseek_v32.norm._plain import LocalPlainRMSNorm


def _identity_hook(fingerprint):
    def attach_identity(_owner, module, _function_name):
        # The hook provides IR provenance and precedes hashing when the DSL
        # cache is enabled. Direct cute.compile currently sets no_cache=True.
        module.operation.attributes["cxldsagr.norm_source"] = ir.StringAttr.get(fingerprint)

    return attach_identity


def get_kernel(kind, width, device_index):
    device = torch.device("cuda", device_index)
    return _get_compiled_typed_norm_kernel(
        kind,
        width,
        device_index,
        device_support_pdl(device),
        build_info()["fingerprint"],
        "bf16_fp32_bf16_align128_weight16_sm90",
        "--enable-tvm-ffi",
    )


@cache
def _get_compiled_typed_norm_kernel(
    kind, width, device_index, enable_pdl, fingerprint, signature, options
):
    if kind not in ("plain", "fused") or width not in (512, 1536, 7168):
        raise ValueError("Unsupported typed norm specialization")
    if kind == "fused" and width != 7168:
        raise ValueError("Typed fused norm supports width 7168")
    device = torch.device("cuda", device_index)
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise ValueError("Typed norm compilation requires SM90/Hopper")
    with torch.cuda.device(device):
        instance = (LocalPlainRMSNorm if kind == "plain" else LocalFusedRMSNorm)(
            cutlass.Float32, width, 0.0, sm_version=90
        )
        expected_shared = {512: 8208, 1536: 24592, 7168: 28688}[width]
        if kind == "fused":
            expected_shared = 57360
        if (
            instance.cluster_n != 1
            or instance.vec_size != 4
            or instance.copy_bits != 128
            or not instance.use_async_copy
            or instance._smem_size_in_bytes() != expected_shared
        ):
            raise RuntimeError(
                "Official Float32 norm launch geometry differs from validated geometry"
            )
        rows = cute.sym_int(64)

        def matrix():
            return cute.runtime.make_fake_compact_tensor(
                cutlass.BFloat16, (rows, width), stride_order=(1, 0), assumed_align=128
            )

        weight = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (width,), assumed_align=16)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        operands = (
            (matrix(), weight, matrix())
            if kind == "plain"
            else (matrix(), matrix(), weight, matrix(), matrix())
        )
        return cute.compile(
            instance,
            *operands,
            cutlass.Int64(1),
            cutlass.Float32(1e-6),
            enable_pdl,
            stream,
            options=options,
            trace_finalize_hooks=_identity_hook(fingerprint),
        )
