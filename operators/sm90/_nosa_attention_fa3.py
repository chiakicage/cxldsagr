"""Resident sparse Hopper attention using the installed FlashInfer FA3 mainloop."""

import hashlib
import json
from functools import cache
from importlib.metadata import distribution
from pathlib import Path

import torch
import tvm_ffi

from operators.sm90 import _native

_GROUP = 8
_SOURCE = Path(__file__).resolve().parent / "csrc"
_ROOT = Path(_native.__file__).resolve().parents[2]
_FLAGS = (
    *_native._CUDA_FLAGS,
    "-use_fast_math",
    "-DFA3_GROUP=8",
    "-DFA3_KV=128",
    "-DFA3_TMA=1",
    "-DFA3_STAGES=2",
)


@cache
def build_info():
    """Pin the template interface and fingerprint the headers used by this build."""
    package = distribution("flashinfer-python")
    if package.version != "0.6.18":
        raise RuntimeError("NOSA FA3 attention requires flashinfer-python==0.6.18")
    include = Path(package.locate_file("flashinfer/data/include"))
    entry = include / "flashinfer/attention/hopper/prefill_sm90.cuh"
    if not entry.is_file():
        raise RuntimeError(f"FlashInfer Hopper headers are missing: {entry}")
    headers = hashlib.sha256()
    for path in sorted(include.rglob("*")):
        if path.is_file():
            headers.update(str(path.relative_to(include)).encode())
            headers.update(path.read_bytes())
    sources = [
        Path(__file__),
        *(
            _SOURCE / name
            for name in ("nosa_attention_fa3.cu", "nosa_attention.cu", "nosa_attention_grouped.cuh")
        ),
    ]
    return {
        "toolchain": _native._toolchain(),
        "torch": torch.__version__,
        "flashinfer": package.version,
        "flashinfer_include": str(include),
        "flashinfer_headers_sha256": headers.hexdigest(),
        "cuda_flags": list(_FLAGS),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "group_queries": _GROUP,
        "minimum_fa3_queries": 1,
        "cta_order": "descending_union_tiles",
        "cta_order_work_items": 256,
        "cta_order_ties": "ascending_logical_batch",
        "q_transfer": "direct_strided_tma",
        "output_store": "direct",
        "numerical_repair": "nonfinite_output_postcheck",
        "native_pv_accumulation": "bf16_power_of_two_scale_finite_output_guard",
        "kv_tile_tokens": 128,
        "stages": 2,
    }


@cache
def _module():
    import tvm_ffi.cpp

    info = build_info()
    digest = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[:16]
    return tvm_ffi.cpp.load(
        name=f"cxldsagr_nosa_fa3_{digest}",
        sources=[str(_SOURCE / "nosa_attention_fa3.cu")],
        extra_include_paths=[
            str(_SOURCE),
            str(_ROOT / "3rdparty/cutlass/include"),
            str(_ROOT / "3rdparty/cutlass/tools/util/include"),
            info["flashinfer_include"],
        ],
        extra_cuda_cflags=list(_FLAGS),
    )


def launch_nosa_fa3_attention(q, keys, values, selection, query_start, cis_bias):
    """Run the complete operator, including metadata and native numerical repair."""
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if not len(q):
        return output
    groups = (len(q) + _GROUP - 1) // _GROUP
    heads = keys.shape[1]
    empty = torch.empty(0, dtype=torch.bool, device=q.device)
    fallback = torch.empty(((len(q) + 3) // 4, heads), dtype=torch.int32, device=q.device)
    mask = selection.valid_mask if selection.valid_mask is not None else empty
    bias = cis_bias if cis_bias is not None else empty
    module = _module()
    # Native kernels preserve FP16 and independent K/V layout support.
    if q.dtype != torch.bfloat16 or keys.stride() != values.stride():
        nonfinite = torch.empty(((len(keys) + 63) // 64, heads), dtype=torch.int32, device=q.device)
        with torch.cuda.device(q.device), tvm_ffi.use_torch_stream():
            module.forward(
                q,
                keys,
                values,
                selection.block_ids,
                mask,
                bias,
                output,
                fallback,
                nonfinite,
                query_start,
            )
        return output
    pages = torch.empty((groups * heads, _GROUP * 64), dtype=torch.int32, device=q.device)
    members = torch.empty_like(pages)
    batches = groups * heads
    counts = torch.empty(
        (batches * (2 if batches == 256 else 1),), dtype=torch.int32, device=q.device
    )
    with torch.cuda.device(q.device), tvm_ffi.use_torch_stream():
        module.fa3_forward(
            q,
            keys,
            values,
            selection.block_ids,
            mask,
            bias,
            output,
            fallback,
            pages,
            members,
            counts,
            query_start,
        )
    return output
