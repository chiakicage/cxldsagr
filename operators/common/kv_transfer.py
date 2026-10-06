"""Pinned-host record transfers using sparse gathers or contiguous CUDA copies."""

from functools import cache
from pathlib import Path

import torch


@cache
def _module():
    import tvm_ffi.cpp

    source = Path(__file__).with_name("csrc") / "kv_transfer.cu"
    return tvm_ffi.cpp.load(
        name="cxldsagr_kv_transfer",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo"],
    )


def gather_host_records(host, device, host_ids, device_ids, *, max_ctas=None, valid_count=None):
    """Copy complete records on the current CUDA stream; IDs stay on GPU.

    The caller owns both allocations until stream completion. Maps are published
    by the cache only after this launch on the same stream.
    Contiguous views with unaligned storage offsets use byte copies.
    ``max_ctas`` bounds the grid for copies that overlap compute on another
    stream. ``valid_count`` is an optional device int64/uint32 scalar selecting the
    initialized prefix of fixed-capacity ID vectors without a CPU round trip.
    """
    if max_ctas is not None and (type(max_ctas) is not int or not 0 < max_ctas < 2**31):
        raise ValueError("max_ctas must be a positive CUDA grid size")
    if host.device.type != "cpu" or not host.is_pinned():
        raise ValueError("host records must use pinned CPU memory")
    if device.device.type != "cuda" or host.dtype != device.dtype:
        raise ValueError("device records must be CUDA and match host dtype")
    if host.ndim != 2 or device.ndim != 2 or host.shape[1] != device.shape[1]:
        raise ValueError("host/device records must have the same row width")
    if not host.is_contiguous() or not device.is_contiguous():
        raise ValueError("records must be contiguous")
    if host_ids.dtype != torch.int64 or device_ids.dtype != torch.int64:
        raise ValueError("record IDs must be int64")
    if host_ids.device != device.device or device_ids.device != device.device:
        raise ValueError("record IDs must be on the cache GPU")
    if host_ids.ndim != 1 or host_ids.shape != device_ids.shape:
        raise ValueError("host/device IDs must be equal length vectors")
    if not host_ids.is_contiguous() or not device_ids.is_contiguous():
        raise ValueError("record IDs must be contiguous")
    if valid_count is not None and (
        valid_count.device != device.device
        or valid_count.dtype not in (torch.int64, torch.uint32)
        or valid_count.numel() != 1
        or not valid_count.is_contiguous()
    ):
        raise ValueError("valid_count must be a contiguous int64/uint32 scalar on the cache GPU")
    if host_ids.numel():
        import tvm_ffi

        with torch.cuda.device(device.device), tvm_ffi.use_torch_stream():
            if valid_count is None:
                _module().gather(host, device, host_ids, device_ids, max_ctas or 0)
            else:
                counted = (
                    _module().gather_counted_u32
                    if valid_count.dtype == torch.uint32
                    else _module().gather_counted
                )
                counted(host, device, host_ids, device_ids, valid_count, max_ctas or 0)


def copy_host_records_async(host, device, *, host_start, device_start, count):
    """Copy one contiguous record span with cudaMemcpyAsync on the current stream.

    The row offsets and count are CPU integers. Both tensors must be contiguous,
    have matching dtypes and row widths, and the source must be pinned. No SM
    kernel, staging allocation, ID gather, or synchronization is performed. The
    caller retains both allocations and orders map publication and consumers
    after the copy; returning from this function does not imply completion.
    """
    for name, value in (
        ("host_start", host_start),
        ("device_start", device_start),
        ("count", count),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if host.device.type != "cpu" or not host.is_pinned():
        raise ValueError("host records must use pinned CPU memory")
    if device.device.type != "cuda" or host.dtype != device.dtype:
        raise ValueError("device records must be CUDA and match host dtype")
    if host.ndim != 2 or device.ndim != 2 or host.shape[1] != device.shape[1] or host.shape[1] <= 0:
        raise ValueError("host/device records must have the same positive row width")
    if not host.is_contiguous() or not device.is_contiguous():
        raise ValueError("records must be contiguous")
    if host_start > host.shape[0] or count > host.shape[0] - host_start:
        raise ValueError("host record span exceeds its allocation")
    if device_start > device.shape[0] or count > device.shape[0] - device_start:
        raise ValueError("device record span exceeds its allocation")
    if count:
        import tvm_ffi

        with torch.cuda.device(device.device), tvm_ffi.use_torch_stream():
            _module().copy_contiguous(host, device, host_start, device_start, count)
