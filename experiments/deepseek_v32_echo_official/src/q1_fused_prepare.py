"""Private stateless ECHO Q1 adaptation candidate; production dispatch is untouched."""

from __future__ import annotations

import hashlib
import os
from functools import cache
from pathlib import Path

import torch

from operators.deepseek_v32.indexer import _native_cache, echo, official_decode, official_prefetch

SOURCE = Path(__file__).with_suffix(".cu")
POLICY = "q1-fused-page-and-stage-prepare-v1"
FLAGS = ["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"]


def build_info():
    return {
        "policy": POLICY,
        "flags": FLAGS,
        "source_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), SOURCE, Path(_native_cache.__file__))
        },
    }


@cache
def module():
    info = build_info()
    digest = hashlib.sha256(repr(info).encode()).hexdigest()[:16]
    name = "cxldsagr_q1_fused_prepare_" + digest
    previous = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "9.0a"
    try:
        loaded = _native_cache.load(
            name=name,
            sources=[str(SOURCE)],
            extra_include_paths=[],
            extra_cuda_cflags=FLAGS,
            extra_ldflags=[],
            source_identity=info,
        )
    finally:
        if previous is None:
            os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
        else:
            os.environ["TVM_FFI_CUDA_ARCH_LIST"] = previous
    return loaded, name


def native_info():
    return _native_cache.native_info(module()[1])


def prepare(k, scales, prefetch, context_lens):
    """Build current packed keys and both tables, owning original cleanup first.

    This follows the production ``official_prefetch._prepare`` lease contract.
    Its provider and state class remain the original ones, so promotion, clear
    and any owner-driven failure cleanup are identical. The caller must hold an
    exclusive lease through consumption and cleanup, including on exceptions.
    """
    device = prefetch["device"].device
    history = prefetch.get("history_length")
    columns = len(k)
    if type(history) is not int or history < 0:
        raise ValueError("official Q1 prefetch requires explicit initialized history length")
    if not history < columns < 2**31 - 1:
        raise ValueError("official Q1 context must contain the single pending suffix token")
    echo._validate_prefetch(prefetch, 1, history + 1, device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
        raise ValueError("official Q1 prefetch promotion requires SM90")
    echo._check_tensor(k, (columns, 128), torch.float8_e4m3fn, device, "k")
    echo._check_tensor(scales, (columns,), torch.float32, device, "scales")
    echo._check_tensor(context_lens, (1,), torch.int32, device, "context_lens")
    # The production packer's uint32 dtype view imposes the same alignment.
    if k.storage_offset() % 4:
        raise ValueError("page64 keys require a 4-byte-aligned storage offset")
    if prefetch["max_prefetch"] < official_prefetch.PREFETCH_CAP:
        raise ValueError("official Q1 prefetch requires 64 prepared legal slots")
    if len(prefetch["device"]) + official_prefetch.PREFETCH_CAP >= 2**31 - 1:
        raise ValueError("official staging tags must fit below the MISSING sentinel")
    if official_prefetch.CLEANUP_KEY in prefetch or official_prefetch.STATE_KEY in prefetch:
        raise ValueError("official staging already belongs to this prefetch lease")
    original = official_prefetch._module()
    candidate, _ = module()
    state = official_prefetch._Staging(
        original,
        torch.empty((1, columns), dtype=torch.int32, device=device),
        torch.empty((1, 64), dtype=torch.int32, device=device),
        torch.empty((1, 64, 576), dtype=torch.bfloat16, device=device),
        prefetch["host_to_device"],
        prefetch["counter"],
        len(prefetch["device"]),
    )
    packed = torch.empty(((columns + 63) // 64, 8448), dtype=torch.uint8, device=device)
    blocks = torch.empty((1, len(packed)), dtype=torch.int32, device=device)
    prepared = prefetch.get("_prepared")
    if type(prepared) is echo._BoundedPreparedPrefetch:
        prepared.consume(prefetch, 1, consumer_limit=64)
    elif prepared is not None:
        prepared.consume(prefetch, 1)
    else:
        prefetch["allocation_log"].fill_(2**31 - 1)
        prefetch["prefetch_stats"].zero_()
    prefetch[official_prefetch.STATE_KEY] = state
    prefetch[official_prefetch.CLEANUP_KEY] = state.clear
    prefetch["prefetch_policy"] = official_prefetch.POLICY
    prefetch["official_prefetch_cap"] = 64
    official_prefetch._call(
        candidate,
        "q1_fused_prepare",
        device,
        k,
        scales,
        prefetch["page_table"],
        context_lens,
        packed,
        blocks,
        state.page_table,
        state.host_ids,
        state.counter,
        history,
        len(prefetch["host"]),
    )
    return packed, blocks, state


def logits(q, k, weights, kscale, query_start, prefetch, *, _bounds=None, _pad_to_stride=False):
    """Complete official Q1 indexer API with fused per-call preparation."""
    if q.ndim != 3 or q.shape[1:] != (64, 128) or not q.is_cuda or len(q) != 1:
        raise ValueError("Private candidate requires CUDA Q[1,64,128]")
    if k.ndim != 2 or k.shape[1] != 128 or len(k) < 1:
        raise ValueError("Private candidate requires K[token,128]")
    columns, device = len(k), q.device
    if type(_pad_to_stride) is not bool:
        raise ValueError("_pad_to_stride must be boolean")
    if type(query_start) is not int or not 0 <= query_start <= columns - 1:
        raise ValueError("query_start must lie inside the visible sequence")
    if not echo._uses_official_q1(prefetch, 1, columns, query_start):
        raise ValueError("Private candidate supports only the existing official Q1 predicate")
    for tensor, shape, dtype, name in (
        (q, (1, 64, 128), torch.float8_e4m3fn, "q"),
        (k, (columns, 128), torch.float8_e4m3fn, "k"),
        (weights, (1, 64), torch.float32, "weights"),
        (kscale, (columns,), torch.float32, "kscale"),
    ):
        echo._check_tensor(tensor, shape, dtype, device, name)
    if _bounds is None:
        # Match both baseline bounds allocations/launches in the measured API.
        starts = torch.zeros(1, dtype=torch.int32, device=device)
        ends = torch.arange(query_start + 1, query_start + 2, dtype=torch.int32, device=device)
    else:
        starts, ends = _bounds
        echo._check_tensor(starts, (1,), torch.int32, device, "starts")
        echo._check_tensor(ends, (1,), torch.int32, device, "ends")
    with torch.cuda.device(device):
        packed, blocks, state = prepare(k, kscale, prefetch, ends)
        schedule = official_decode.metadata(ends)
        scores = official_decode.logits(
            q.unsqueeze(0),
            packed,
            weights,
            ends,
            blocks,
            schedule,
            columns,
            state.page_table,
            prefetch["device"],
            prefetch["host"],
            state.host_ids,
            state.records,
            state.host_to_device,
            state.counter,
            prefetch["offset"][1:2],
        )
        official_prefetch._promote(state, prefetch)
    return scores.as_strided((1, scores.stride(0)), scores.stride()) if _pad_to_stride else scores
