"""Explicit native validation/compression, without publishing cache metadata."""

from dataclasses import dataclass

import torch

from operators.sm90._native import load_module


@dataclass(frozen=True)
class PreparationScratch:
    """Reusable per-stream validation storage; callers serialize reuse."""

    partial: torch.Tensor
    finite: torch.Tensor
    host_finite: torch.Tensor | None = None

    @classmethod
    def allocate(cls, device, *, include_host=False):
        return cls(
            torch.empty(3072, dtype=torch.uint8, device=device),
            torch.empty((), dtype=torch.bool, device=device),
            torch.empty((), dtype=torch.bool, device="cpu", pin_memory=True)
            if include_host
            else None,
        )


def supports(query, keys, cis):
    """Strided/broadcast scans are supported; K requires aligned row vectors.

    The checked compression layouts are D64/128/256, matching floating-point
    dtypes, and positive shapes. Other shapes retain the existing preparation
    path. This predicate neither reads tensor values nor reserves cache state.
    """
    return (
        query.ndim in (3, 4)
        and keys.ndim == 3
        and cis.shape == keys.shape[:2]
        and 0 < len(query) <= 262144
        and 0 <= len(keys) <= 262144
        and all(size > 0 for size in query.shape[1:])
        and keys.shape[1] > 0
        and keys.shape[2] in (64, 128, 256)
        and query.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and keys.dtype == cis.dtype == query.dtype
        and keys.stride(-1) == 1
        and keys.data_ptr() % 16 == 0
        and all(stride % (16 // keys.element_size()) == 0 for stride in keys.stride()[:2])
        and all(
            tensor.is_cuda
            and tensor.device == query.device
            and not tensor.requires_grad
            and all(stride >= 0 for stride in tensor.stride())
            for tensor in (query, keys, cis)
        )
        and torch.cuda.get_device_capability(query.device)[0] == 9
    )


def _validate(
    query,
    keys,
    cis,
    compressed_keys,
    compressed_cis,
    pooled_cis,
    scratch,
    validated_start,
    compressed_start,
    pooled_start,
):
    if not supports(query, keys, cis):
        raise ValueError("Unsupported native NOSA preparation shape, dtype, device, or stride")
    length, heads, dim = keys.shape
    count, stable = max(0, length // 16 - 1), max(0, (length - 16) // 64)
    for name, value, end in (
        ("validated_start", validated_start, length),
        ("compressed_start", compressed_start, count),
        ("pooled_start", pooled_start, stable),
    ):
        if type(value) is not int or not 0 <= value <= end:
            raise ValueError(f"{name} must be within the visible prefix")
    for tensor, shape in (
        (compressed_keys, (count, heads, dim)),
        (compressed_cis, (count, heads)),
        (pooled_cis, (stable, heads)),
    ):
        if (
            tensor.ndim != len(shape)
            or tensor.shape[1:] != shape[1:]
            or len(tensor) < shape[0]
            or tensor.device != query.device
            or tensor.dtype != query.dtype
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise ValueError("Invalid native NOSA preparation output capacity, dtype or storage")
    if compressed_keys.data_ptr() % 16:
        raise ValueError("Native NOSA preparation requires 16-byte aligned compressed K")
    partials = sum(
        min(1024, (elements + 4095) // 4096)
        for elements in (
            query.numel(),
            (length - validated_start) * heads * dim,
            (length - validated_start) * heads,
        )
    )
    if (
        not isinstance(scratch, PreparationScratch)
        or scratch.partial.ndim != 1
        or scratch.partial.numel() < partials
        or scratch.partial.dtype != torch.uint8
        or not scratch.partial.is_contiguous()
        or scratch.partial.device != query.device
        or scratch.finite.shape != ()
        or scratch.finite.dtype != torch.bool
        or scratch.finite.device != query.device
    ):
        raise ValueError("Native NOSA preparation requires reusable CUDA validation scratch")
    return count, stable


def prepare_out(
    query,
    keys,
    cis,
    compressed_keys,
    compressed_cis,
    pooled_cis,
    *,
    validated_start,
    compressed_start,
    pooled_start,
    scratch,
):
    """Return a CUDA bool; failed validation leaves all derived buffers intact.

    Only newly complete windows and stable pools are written. The owner must
    test the returned flag before publishing a reservation; ``prepare_checked``
    performs that host check. This function itself is CUDA Graph compatible.
    """
    import tvm_ffi

    _validate(
        query,
        keys,
        cis,
        compressed_keys,
        compressed_cis,
        pooled_cis,
        scratch,
        validated_start,
        compressed_start,
        pooled_start,
    )
    module = load_module("nosa_prepare")
    with torch.cuda.device(query.device), tvm_ffi.use_torch_stream():
        module.prepare_out(
            query,
            keys,
            cis,
            compressed_keys,
            compressed_cis,
            pooled_cis,
            scratch.partial,
            scratch.finite,
            validated_start,
            compressed_start,
            pooled_start,
        )
    return scratch.finite


def prepare_checked(*args, **kwargs):
    """Validate before returning; the caller can now finish its reservation."""
    if not prepare_out(*args, **kwargs):
        raise ValueError("NOSA indexer requires finite Q, K and CIS scores")
