"""GPU initiated gathers from pinned host token records (no CPU ID round trip)."""

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


def gather_host_records(host, device, host_ids, device_ids):
    """Copy complete records on the current CUDA stream; IDs stay on GPU.

    The caller owns both allocations until stream completion. Maps are published
    by the cache only after this launch on the same stream.
    Contiguous views with unaligned storage offsets use byte copies.
    """
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
    if host_ids.numel():
        import tvm_ffi

        with torch.cuda.device(device.device), tvm_ffi.use_torch_stream():
            _module().gather(host, device, host_ids, device_ids)
