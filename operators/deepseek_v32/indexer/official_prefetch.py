"""Promote official Q1 prefetch staging into an exclusive local pool lease.

The unchanged official kernel writes temporary forward-map tags. This adapter
publishes exact record bytes into the prepared local FIFO slots before normal
cache finalization. A registered cleanup hook owns staging until finalization
and removes any tags left by an interrupted call.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch

from operators.deepseek_v32.indexer import official_decode

PREFETCH_CAP = 64
POLICY = "official-q1-predictive-staging-promotion-v1"
PREPARATION = "official-q1-current-page64-and-staging-v1"
_MODULE_NAME = None
_FUSED_ENTRY_OBSERVED = False
CLEANUP_KEY = "_pending_prefetch_cleanup"
STATE_KEY = "_official_prefetch"
_SOURCE = Path(__file__).with_name("csrc") / "official_prefetch.cu"
_FLAGS = ["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"]


def build_info():
    """Bind this adapter independently of the official score-kernel build."""
    root = Path(__file__).resolve().parents[3]
    return {
        "policy": POLICY,
        "preparation": PREPARATION,
        "cuda_flags": _FLAGS,
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("_native_cache.py"), _SOURCE)
        },
    }


@cache
def _module():
    global _MODULE_NAME

    from operators.deepseek_v32.indexer import _native_cache

    info = build_info()
    digest = hashlib.sha256(repr(info).encode()).hexdigest()[:16]
    name = f"cxldsagr_official_prefetch_{digest}"
    previous_arch = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "9.0a"
    try:
        module = _native_cache.load(
            name=name,
            sources=[str(_SOURCE)],
            extra_include_paths=[],
            extra_cuda_cflags=_FLAGS,
            extra_ldflags=[],
            source_identity=info,
        )
        _MODULE_NAME = name
        return module
    finally:
        if previous_arch is None:
            os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
        else:
            os.environ["TVM_FFI_CUDA_ARCH_LIST"] = previous_arch


def runtime_info():
    """Report a normal current-key entry and its actual immutable native build."""
    if not _FUSED_ENTRY_OBSERVED:
        return None
    from operators.deepseek_v32.indexer import _native_cache

    return {
        "preparation": PREPARATION,
        "entry_point": "logits_from_keys",
        "native": _native_cache.native_info(_MODULE_NAME),
    }


def _call(module, name, device, *args):
    import tvm_ffi

    with torch.cuda.device(device), tvm_ffi.use_torch_stream():
        return getattr(module, name)(*args)


@dataclass
class _Staging:
    module: object
    page_table: torch.Tensor
    host_ids: torch.Tensor
    records: torch.Tensor
    host_to_device: torch.Tensor
    counter: torch.Tensor
    pool_rows: int

    def clear(self):
        """Clear only our remaining tags; published real mappings survive."""
        _call(
            self.module,
            "official_prefetch_clear",
            self.host_to_device.device,
            self.host_ids,
            self.host_to_device,
            self.counter,
            self.pool_rows,
        )


def _new_staging(prefetch, context_lens, max_context_len):
    from operators.deepseek_v32.indexer.echo import (
        _check_tensor,
        _validate_prefetch,
    )

    device = prefetch["device"].device
    history = prefetch.get("history_length")
    if type(history) is not int or history < 0:
        raise ValueError("official Q1 prefetch requires explicit initialized history length")
    if type(max_context_len) is not int or not history < max_context_len < 2**31 - 1:
        raise ValueError("official Q1 context must contain the single pending suffix token")
    _validate_prefetch(prefetch, 1, history + 1, device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
        raise ValueError("official Q1 prefetch promotion requires SM90")
    _check_tensor(context_lens, (1,), torch.int32, device, "context_lens")
    if prefetch["max_prefetch"] < PREFETCH_CAP:
        raise ValueError("official Q1 prefetch requires 64 prepared legal slots")
    if len(prefetch["device"]) + PREFETCH_CAP >= 2**31 - 1:
        raise ValueError("official staging tags must fit below the MISSING sentinel")
    if CLEANUP_KEY in prefetch or STATE_KEY in prefetch:
        raise ValueError("official staging already belongs to this prefetch lease")
    module = _module()
    state = _Staging(
        module,
        torch.empty((1, max_context_len), dtype=torch.int32, device=device),
        torch.empty((1, PREFETCH_CAP), dtype=torch.int32, device=device),
        torch.empty((1, PREFETCH_CAP, 576), dtype=torch.bfloat16, device=device),
        prefetch["host_to_device"],
        prefetch["counter"],
        len(prefetch["device"]),
    )
    return state


def _register_staging(state, prefetch):
    from operators.deepseek_v32.indexer.echo import _BoundedPreparedPrefetch

    prepared = prefetch.get("_prepared")
    if type(prepared) is _BoundedPreparedPrefetch:
        prepared.consume(prefetch, 1, consumer_limit=PREFETCH_CAP)
    elif prepared is not None:
        prepared.consume(prefetch, 1)
    else:
        prefetch["allocation_log"].fill_(2**31 - 1)
        prefetch["prefetch_stats"].zero_()
    # Register before any launch that can publish temporary tags. The owner
    # must call this hook before finalization and retain it if cleanup fails.
    prefetch[STATE_KEY] = state
    prefetch[CLEANUP_KEY] = state.clear
    prefetch["prefetch_policy"] = POLICY
    prefetch["official_prefetch_cap"] = PREFETCH_CAP


def _prepare(prefetch, context_lens, max_context_len):
    state = _new_staging(prefetch, context_lens, max_context_len)
    _register_staging(state, prefetch)
    _call(
        state.module,
        "official_prefetch_prepare",
        prefetch["device"].device,
        prefetch["page_table"],
        state.page_table,
        state.host_ids,
        state.counter,
        context_lens,
        prefetch["history_length"],
        len(prefetch["host"]),
    )
    return state


def _prepare_keys(k, scales, prefetch, context_lens):
    from operators.deepseek_v32.indexer.echo import _check_tensor

    columns, device = len(k), prefetch["device"].device
    _check_tensor(k, (columns, 128), torch.float8_e4m3fn, device, "k")
    _check_tensor(scales, (columns,), torch.float32, device, "scales")
    if k.storage_offset() % 4:
        raise ValueError("page64 keys require a 4-byte-aligned storage offset")
    state = _new_staging(prefetch, context_lens, columns)
    packed = torch.empty(((columns + 63) // 64, 8448), dtype=torch.uint8, device=device)
    blocks = torch.empty((1, len(packed)), dtype=torch.int32, device=device)
    # Finish every allocation before registering cleanup over reset-dependent
    # stage buffers. No launch can publish a tag before ownership is registered.
    _register_staging(state, prefetch)
    _call(
        state.module,
        "official_prefetch_prepare_keys",
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
        prefetch["history_length"],
        len(prefetch["host"]),
    )
    return packed, blocks, state


def _promote(state, prefetch):
    _call(
        state.module,
        "official_prefetch_promote",
        prefetch["device"].device,
        state.host_ids,
        state.records,
        state.counter,
        prefetch["device"],
        prefetch["host_to_device"],
        prefetch["device_to_host"],
        prefetch["free_slots"],
        prefetch["allocation_log"],
        prefetch["prefetch_stats"],
    )


def logits(
    q,
    packed,
    weights,
    context_lens,
    block_table,
    schedule_meta,
    max_context_len,
    prefetch,
):
    """Return official causal scores and publish every successful stage record.

    The lease must remain exclusive through finalize, exact recall and attention.
    Its owner invokes ``_pending_prefetch_cleanup`` before normal finalization,
    including rollback after a failed call. This function never retries or
    synchronizes; CUDA completion and poisoning remain the owner's contract.
    The pending query is excluded from host prefetch; ``offset[1]`` holds a
    separate predictive decode threshold and is not modified here.

    Extra live storage is ``4 * max_context_len + 64 * (576 * 2 + 4)`` bytes.
    The existing lease counter/journal/statistics are reused. Every official
    stage copy counts as H2D, including records outside the exact top-k.
    """
    state = _prepare(prefetch, context_lens, max_context_len)
    return _logits_prepared(
        q,
        packed,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        prefetch,
        state,
    )


def _logits_prepared(
    q,
    packed,
    weights,
    context_lens,
    block_table,
    schedule_meta,
    max_context_len,
    prefetch,
    state,
):
    scores = official_decode.logits(
        q,
        packed,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        state.page_table,
        prefetch["device"],
        prefetch["host"],
        state.host_ids,
        state.records,
        state.host_to_device,
        state.counter,
        prefetch["offset"][1:2],
    )
    _promote(state, prefetch)
    return scores


def logits_from_keys(q, k, weights, scales, context_lens, prefetch):
    """Run official Q1 with fresh fused key/table/stage preparation.

    The current keys and scales are packed on every call and graph replay. The
    original strict predictor, promotion and owner-driven cleanup are retained.
    Storage sizes match the explicit packed-input path; no history is retained.
    """
    global _FUSED_ENTRY_OBSERVED

    packed, blocks, state = _prepare_keys(k, scales, prefetch, context_lens)
    schedule = official_decode.metadata(context_lens)
    scores = _logits_prepared(
        q, packed, weights, context_lens, blocks, schedule, len(k), prefetch, state
    )
    # Stable participation evidence, collected outside measurement. Graph
    # capture records this entry; later trace audits verify the actual nodes.
    _FUSED_ENTRY_OBSERVED = True
    return scores
