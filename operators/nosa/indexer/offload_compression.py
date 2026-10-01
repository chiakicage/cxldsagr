"""Append offloaded NOSA records with the resident native reduction arithmetic."""

import hashlib
from functools import cache
from pathlib import Path

import torch


@cache
def _module():
    import tvm_ffi.cpp

    source = Path(__file__).with_name("csrc")
    digest = hashlib.sha256()
    for name in ("nosa_offload_compression.cu", "nosa_prepare.cu"):
        digest.update((source / name).read_bytes())
    return tvm_ffi.cpp.load(
        name=f"cxldsagr_nosa_offload_compression_{digest.hexdigest()[:16]}",
        sources=[str(source / "nosa_offload_compression.cu")],
        extra_include_paths=[str(source)],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_90,code=sm_90",
            "-lineinfo",
        ],
    )


def append_compressed(
    keys,
    cis,
    compressed_keys,
    compressed_cis,
    pooled_cis,
    *,
    compressed_start,
    pooled_start,
):
    """Append BF16 D128 records from a boundary-plus-suffix K view.

    K starts at logical token ``16 * compressed_start``; CIS and destinations
    retain their original logical indexing. The owner has already checked the
    newly appended K/CIS for finite inputs and publishes metadata separately.
    Derived overflow follows the native reduction and NaN-ignoring pool rules.
    """
    import tvm_ffi

    if keys.ndim != 3 or keys.shape[-1] != 128 or cis.ndim != 2:
        raise ValueError("Offload compression requires K [tokens,heads,128] and CIS [tokens,heads]")
    heads = keys.shape[1]
    count = max(0, len(cis) // 16 - 1)
    stable = max(0, (len(cis) - 16) // 64)
    if heads <= 0 or heads > 65535 or cis.shape[1] != heads or len(cis) > 262144:
        raise ValueError("Unsupported offload compression head count or token length")
    for name, start, end in (
        ("compressed_start", compressed_start, count),
        ("pooled_start", pooled_start, stable),
    ):
        if type(start) is not int or not 0 <= start <= end:
            raise ValueError(f"{name} must be within the complete visible prefix")
    if compressed_start < count and len(keys) != len(cis) - compressed_start * 16:
        raise ValueError("Offload compression K must begin at the first new window")
    tensors = (keys, cis, compressed_keys, compressed_cis, pooled_cis)
    if any(
        not tensor.is_cuda
        or tensor.device != keys.device
        or tensor.dtype != torch.bfloat16
        or tensor.requires_grad
        for tensor in tensors
    ):
        raise ValueError("Offload compression requires CUDA BF16 inference tensors on one device")
    if torch.cuda.get_device_capability(keys.device) != (9, 0):
        raise RuntimeError("Offload compression requires SM90/Hopper")
    if (
        keys.stride(-1) != 1
        or keys.data_ptr() % 16
        or any(stride <= 0 or stride % 8 for stride in keys.stride()[:2])
        or any(stride < 0 for stride in cis.stride())
    ):
        raise ValueError("Offload compression requires aligned K rows and nonnegative CIS strides")
    for tensor, shape in (
        (compressed_keys, (count, heads, 128)),
        (compressed_cis, (count, heads)),
        (pooled_cis, (stable, heads)),
    ):
        if (
            tensor.ndim != len(shape)
            or tensor.shape[1:] != shape[1:]
            or len(tensor) < shape[0]
            or not tensor.is_contiguous()
            or (tensor.numel() and tensor.data_ptr() % 16)
        ):
            raise ValueError("Offload compressed output shape, capacity or alignment is invalid")
    if compressed_start == count and pooled_start == stable:
        return
    with torch.cuda.device(keys.device), tvm_ffi.use_torch_stream():
        _module().append_compressed(
            keys,
            cis,
            compressed_keys,
            compressed_cis,
            pooled_cis,
            compressed_start,
            pooled_start,
        )
