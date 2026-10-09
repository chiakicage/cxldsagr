"""Official resident indexer and ECHO inter-query prefetch on Hopper.

The index keys remain in HBM. Main-attention BF16 latent/RoPE records live in
mapped pinned host memory. Q1 uses the official predictive paged decode path
when its staging contract is supported. Prefill prefetches a provable subset
of earlier rows' exact top-2048. Both paths retain complete exact recall.
"""

import hashlib
import os
import subprocess
from functools import cache
from pathlib import Path

import torch

UPSTREAM_REVISION = "bc1b75c1000010d0ac6f032ebaac283255c050b1"
PAGED_Q1_MIN_CONTEXT_TOKENS = 32768
_ROOT = Path(__file__).resolve().parents[3]
_SOURCE = Path(__file__).resolve().parent / "csrc"
_FLAGS = [
    "-O3",
    "-std=c++20",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-gencode=arch=compute_90a,code=sm_90a",
    "-lineinfo",
]


def build_info():
    """Record source and shared-header versions without initializing CUDA."""
    files = [
        Path(__file__),
        Path(__file__).with_name("cache_ops.py"),
        Path(__file__).with_name("page64.py"),
        *_SOURCE.glob("echo_*.cu*"),
    ]
    include = _ROOT / "3rdparty/cutlass/include"
    headers = sorted(path for path in include.rglob("*") if path.is_file())
    return {
        "upstream": "https://github.com/sjtu-zhao-lab/ECHO",
        "upstream_revision": UPSTREAM_REVISION,
        "cutlass_revision": subprocess.check_output(
            ["git", "-C", str(_ROOT / "3rdparty/cutlass"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "cuda_flags": _FLAGS,
        "shared_header_sha256": {
            str(path.relative_to(_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in headers
        },
        "source_sha256": {
            str(path.relative_to(_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)
        },
    }


@cache
def _module():
    import tvm_ffi.cpp

    include = _ROOT / "3rdparty/cutlass/include"
    if not (include / "cute/tensor.hpp").is_file():
        raise RuntimeError("Prepare shared CUTLASS with python scripts/prepare_3rdparty.py --init")
    info = build_info()
    digest = hashlib.sha256(repr(info).encode()).hexdigest()[:16]
    # TVM FFI otherwise adds a generic sm_90 target, which rejects WGMMA PTX.
    previous_arch = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = "9.0a"
    try:
        return tvm_ffi.cpp.load(
            name=f"cxldsagr_echo_indexer_{digest}",
            sources=[str(_SOURCE / "echo_indexer.cu")],
            extra_include_paths=[str(include), str(_SOURCE)],
            extra_cuda_cflags=_FLAGS,
            extra_ldflags=["-lcuda"],
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
        else:
            os.environ["TVM_FFI_CUDA_ARCH_LIST"] = previous_arch


def _check_tensor(tensor, shape, dtype, device, name):
    if (
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != tuple(shape)
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
        or tensor.requires_grad
    ):
        raise ValueError(f"{name} must be contiguous {dtype} {tuple(shape)} on {device}")


class _PreparedPrefetch:
    """One-use proof of journal reset and request metadata in borrowed scratch."""

    def __init__(self, rows, free_slots, allocations, counter, stats, scratch):
        self.rows = rows
        self.tensors = (free_slots, allocations, counter, stats)
        metadata = scratch.view(torch.int32)
        self.metadata = (metadata[:1], metadata[1 : rows + 1])
        self.used = False

    def invalidate(self):
        self.used = True

    def consume(self, lease, rows):
        actual = tuple(
            lease[name] for name in ("free_slots", "allocation_log", "counter", "prefetch_stats")
        )
        if (
            self.used
            or rows != self.rows
            or any(lhs is not rhs for lhs, rhs in zip(actual, self.tensors, strict=True))
        ):
            raise ValueError("prepared prefetch lease was consumed, replaced, or resized")
        self.used = True
        return self.metadata


def _bounded_prepared_storage_identity(tensor):
    storage = tensor.untyped_storage()
    return (
        tensor.data_ptr(),
        storage.data_ptr(),
        storage.nbytes(),
        storage._cdata,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.dtype,
        tensor.device,
    )


class _BoundedPreparedPrefetch:
    """Q1-only preparation proof; generic prefill rejects this distinct type."""

    prepared_limit = 64

    def __init__(self, rows, free_slots, allocations, counter, stats, scratch):
        if type(rows) is not int or rows != 1 or free_slots.numel() < self.prepared_limit:
            raise ValueError("bounded preparation requires one query and 64 slots")
        self.rows = rows
        self.tensors = (free_slots, allocations, counter, stats)
        self.scratch = scratch
        self.identities = tuple(
            _bounded_prepared_storage_identity(t) for t in (*self.tensors, scratch)
        )
        self.stream = (
            torch.cuda.current_stream(free_slots.device).cuda_stream if free_slots.is_cuda else None
        )
        metadata = scratch.view(torch.int32)
        self.metadata = (metadata[:1], metadata[1:2])
        self.used = False

    def invalidate(self):
        self.used = True

    def consume(self, lease, rows, *, consumer_limit):
        if type(consumer_limit) is not int or not 0 <= consumer_limit <= self.prepared_limit:
            raise ValueError("consumer exceeds the explicitly prepared slot limit")
        actual = tuple(
            lease[name] for name in ("free_slots", "allocation_log", "counter", "prefetch_stats")
        )
        stream = (
            torch.cuda.current_stream(actual[0].device).cuda_stream if actual[0].is_cuda else None
        )
        if (
            self.used
            or rows != self.rows
            or any(lhs is not rhs for lhs, rhs in zip(actual, self.tensors, strict=True))
            or tuple(_bounded_prepared_storage_identity(t) for t in (*actual, self.scratch))
            != self.identities
            or stream != self.stream
        ):
            raise ValueError(
                "bounded prefetch lease was consumed, replaced, resized, or moved to another stream"
            )
        self.used = True
        return self.metadata


def _validate_prefetch(prefetch, rows, columns, device):
    """Validate lease tensor ABI without synchronizing tensor values to CPU."""
    host, pool = prefetch["host"], prefetch["device"]
    if host.ndim != 2 or host.shape[1] != 576 or len(host) % 64:
        raise ValueError("host must contain whole 64-token pages of 576-element MLA records")
    if pool.ndim != 2 or pool.shape[1] != 576 or len(pool) <= rows:
        raise ValueError("device pool needs Q <= P allocatable slots plus sentinel slot zero")
    _check_tensor(host, host.shape, torch.bfloat16, torch.device("cpu"), "host")
    if not host.is_pinned():
        raise ValueError("host MLA records must use pinned, CUDA-mapped storage")
    _check_tensor(pool, pool.shape, torch.bfloat16, device, "device")
    if len(host) >= 2**31 - 1:
        raise ValueError("global host IDs must fit signed int32 below the MISSING sentinel")
    page_table = prefetch["page_table"]
    history = prefetch.get("history_length", columns)
    if type(history) is not int or not 0 <= history <= columns:
        raise ValueError("prefetch history must lie within the indexer-visible context")
    if page_table.ndim != 1 or page_table.numel() < (history + 63) // 64:
        raise ValueError("page_table must cover initialized host history pages")
    for key, shape, dtype in (
        ("host_to_device", (len(host),), torch.int32),
        ("device_to_host", (len(pool),), torch.int64),
        ("free_slots", (len(pool) - 1,), torch.int32),
        ("allocation_log", (len(pool),), torch.int64),
        ("prefetch_stats", (3,), torch.int64),
        ("counter", (1,), torch.uint32),
        ("offset", (16,), torch.float32),
        ("page_table", tuple(page_table.shape), torch.int32),
    ):
        _check_tensor(prefetch[key], shape, dtype, device, key)
    headroom = len(pool) - 1 - (0 if prefetch.get("transient_suffix", False) else rows)
    limit = prefetch.get("max_prefetch", min(8192, headroom))
    if type(limit) is not int or not 0 <= limit <= min(8192, headroom):
        raise ValueError("max_prefetch must satisfy min(8192, P-Q), excluding sentinel slot zero")


@cache
def _resident_tail_mask_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _resident_causal_tail_mask(
        Scores,
        Ends,
        rows,
        tail_columns,
        query_start,
        stride_row,
        stride_column,
        BLOCK: tl.constexpr,
    ):
        position = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        row = position // tail_columns
        column = position % tail_columns + query_start
        inside = position < tl.cast(rows, tl.int64) * tail_columns
        end = tl.load(Ends + row, inside, 0).to(tl.int64)
        address = row * stride_row + column * stride_column
        # Do not read scores or rewrite valid values or unexposed storage.
        tl.store(Scores + address, -float("inf"), inside & (column >= end))

    return _resident_causal_tail_mask


def _pack_q1_keys(k, scales):
    """Build current page64 bytes, preserving key and FP32 scale bit patterns."""
    from operators.deepseek_v32.indexer.page64 import pack_q1_keys

    return pack_q1_keys(k, scales)


def _uses_official_q1(prefetch, rows, columns, query_start):
    return (
        prefetch is not None
        and rows == 1
        and columns == query_start + 1
        and columns >= PAGED_Q1_MIN_CONTEXT_TOKENS
        and prefetch.get("history_length") == query_start
        and type(prefetch.get("max_prefetch")) is int
        and prefetch.get("max_prefetch", 0) >= 64
    )


def _paged_q1_logits(q, k, weights, scales, ends, score_columns):
    """Pack current FP8 bytes/scales and run official SM90 split-KV logits.

    All temporary storage and metadata are rebuilt on the caller stream. No
    cache identity, causal length, or data is retained across invocations.
    The page ABI stores all 64 keys before all 64 FP32 scales in each page.
    """
    import deep_gemm

    packed = _pack_q1_keys(k, scales)
    pages = len(packed)
    block_table = torch.arange(pages, dtype=torch.int32, device=k.device).unsqueeze(0)
    context_lens = ends.reshape(1, 1)
    metadata = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, 64, deep_gemm.get_num_sms(), indices=None
    )
    return deep_gemm.fp8_fp4_paged_mqa_logits(
        q=(q.unsqueeze(0), None),
        kv_cache=packed.view(pages, 64, 1, 132),
        weights=weights,
        context_lens=context_lens,
        block_table=block_table,
        schedule_meta=metadata,
        max_context_len=score_columns,
        indices=None,
    )


def logits(
    q, k, weights, kscale, query_start, prefetch=None, *, _bounds=None, _pad_to_stride=False
):
    """Return causal FP32 ``[Q,N]`` logits with a padded physical row stride.

    Q/K are E4M3 indexer values, ``weights`` includes the query quantization
    scales and model head weights, and ``kscale`` is the per-key FP32 scale.
    This model-specific implementation accepts 64 index heads of dimension 128.
    Resident Q1 with at least 32768 causally visible tokens uses official paged
    MQA with per-call packing and metadata. Other resident shapes use nonpaged MQA;
    only the offload path loads the local fused ECHO prefetch extension.

    A full-history Q1 offload call with context >=32768 and at least 64 legal
    prepared slots invokes the unchanged official paged fused decode kernel.
    It predicts using score > offset[1], stages at most 64 records, then promotes
    them into the local pool before ordinary finalization and exact recall.
    Predictions may fall outside exact top-k; every staged transfer is counted.
    The cache owner must run the registered cleanup hook on failure as well.

    The private ``_pad_to_stride`` option exposes the resident or official Q1
    256-column-aligned backing as a contiguous score matrix. Coarse prefill
    retains its logical output shape. Every exposed extra
    column is -inf. Callers must preserve logical-N selection capacity and
    pass only the logical columns to reductions such as the prefetch hint.

    ``prefetch`` is an exclusive shared-pool lease. It supplies pinned host
    records, physical ``device[P+1,576]`` (padding slot 0), live global maps,
    ``page_table`` of global 64-token page numbers, a priority-sorted
    ``free_slots[P]`` permutation, reusable ``allocation_log[P+1]`` int64,
    ``counter[1]`` uint32, ``prefetch_stats[3]`` int64, and ``offset[16]`` FP32.
    Log entries are global host IDs indexed by physical slot; MISSING is INT_MAX.

    Preparation must wait for all host writes read by this invocation. The
    initialized ``history_length`` excludes the pending main-KV suffix.
    ``max_prefetch`` is capped at ``min(8192,P-Q)``; Q>P is rejected. Allocation
    claims evict only actual victims. Stats contain actual copied records,
    actual evictions and rejected reservations; the attempt counter may
    overshoot. Finalize and all consumers must follow kernel completion on
    the owning stream. The caller supplies and accounts for reusable scratch.
    """
    if q.ndim != 3 or q.shape[1:] != (64, 128) or not q.is_cuda or len(q) < 1:
        raise ValueError("ECHO requires nonempty CUDA Q[query,64,128]")
    if k.ndim != 2 or k.shape[1] != 128 or len(k) < 1:
        raise ValueError("ECHO requires nonempty K[token,128]")
    rows, columns, device = len(q), len(k), q.device
    if type(_pad_to_stride) is not bool:
        raise ValueError("_pad_to_stride must be boolean")
    if type(query_start) is not int or not 0 <= query_start <= columns - rows:
        raise ValueError("query_start must place all query rows inside the visible KV sequence")
    if torch.cuda.get_device_capability(device)[0] != 9:
        raise ValueError("ECHO's WGMMA prefetch kernel requires Hopper SM90")
    for tensor, shape, dtype, name in (
        (q, (rows, 64, 128), torch.float8_e4m3fn, "q"),
        (k, (columns, 128), torch.float8_e4m3fn, "k"),
        (weights, (rows, 64), torch.float32, "weights"),
        (kscale, (columns,), torch.float32, "kscale"),
    ):
        _check_tensor(tensor, shape, dtype, device, name)
    if _bounds is None:
        starts = torch.zeros(rows, dtype=torch.int32, device=device)
        ends = torch.arange(
            query_start + 1, query_start + rows + 1, dtype=torch.int32, device=device
        )
    else:
        # Private compute-graph metadata: values are generated from the same
        # validated dynamic position used by RoPE. Arbitrary external bounds
        # are outside this interface; retain ABI checks without a host read.
        starts, ends = _bounds
        _check_tensor(starts, (rows,), torch.int32, device, "starts")
        _check_tensor(ends, (rows,), torch.int32, device, "ends")
    use_paged_q1 = prefetch is None and rows == 1 and query_start + 1 >= PAGED_Q1_MIN_CONTEXT_TOKENS
    use_official_q1 = _uses_official_q1(prefetch, rows, columns, query_start)
    if columns % 4 and not use_paged_q1 and not use_official_q1:
        scales = torch.nn.functional.pad(kscale, (0, 4 - columns % 4))
    else:
        scales = kscale
    if prefetch is None:
        import deep_gemm

        # K/scales keep logical N; only the output view may expose its already
        # allocated physical stride. The SM90 TMA scale descriptor rounds N
        # to four floats, so retain padded backing storage.
        score_columns = (columns + 255) // 256 * 256 if _pad_to_stride else columns
        with torch.cuda.device(device):
            if use_paged_q1:
                result = _paged_q1_logits(q, k, weights, kscale, ends, score_columns)
            else:
                result = deep_gemm.fp8_fp4_mqa_logits(
                    (q, None),
                    (k, scales[:columns]),
                    weights,
                    starts,
                    ends,
                    max_seqlen_k=score_columns,
                )
            # Mainline exposes compressed logits without initialized tails.
            # All row starts are zero, so these are logical token positions;
            # explicitly mask each exclusive endpoint before exact top-k.
            if query_start + 1 < score_columns:
                tail_columns = score_columns - query_start
                _resident_tail_mask_kernel()[((rows * tail_columns + 255) // 256,)](
                    result,
                    ends,
                    rows,
                    tail_columns,
                    query_start,
                    result.stride(0),
                    result.stride(1),
                    BLOCK=256,
                    num_warps=4,
                )
            return result

    _validate_prefetch(prefetch, rows, columns, device)
    if use_official_q1:
        from operators.deepseek_v32.indexer import official_prefetch

        with torch.cuda.device(device):
            result = official_prefetch.logits_from_keys(
                q.unsqueeze(0), k, weights, kscale, ends.reshape(1), prefetch
            )
            # The official clean kernel initializes the complete physical tail.
            # This view lets FlashInfer vectorize without copying or rescoring.
            return (
                result.as_strided((rows, result.stride(0)), result.stride())
                if _pad_to_stride
                else result
            )
    module = _module()
    output = torch.empty(
        ((rows + 1) // 2 * 2, (columns + 127) // 128 * 128),
        device=device,
        dtype=torch.float32,
    )
    host, pool = prefetch["host"], prefetch["device"]
    host_to_device, device_to_host = prefetch["host_to_device"], prefetch["device_to_host"]
    slots, counter, offsets = prefetch["free_slots"], prefetch["counter"], prefetch["offset"]
    allocations, stats = prefetch["allocation_log"], prefetch["prefetch_stats"]
    headroom = len(pool) - 1 - (0 if prefetch.get("transient_suffix", False) else rows)
    limit = prefetch.get("max_prefetch", min(8192, headroom))
    history_length = prefetch.get("history_length", query_start)
    if type(history_length) is not int or not 0 <= history_length <= query_start:
        raise ValueError("history_length must cover only initialized history before query_start")
    prepared = prefetch.get("_prepared")
    if prepared is None:
        # External callers receive the full initialization contract. Internal
        # native preparation already resets these tensors on the owning stream.
        counter.zero_()
        allocations.fill_(2**31 - 1)
        stats.zero_()
        extend_lengths = torch.tensor([rows], dtype=torch.int32, device=device)
        query_requests = torch.zeros(rows, dtype=torch.int32, device=device)
    else:
        if type(prepared) is not _PreparedPrefetch:
            raise ValueError("invalid private prefetch preparation token")
        extend_lengths, query_requests = prepared.consume(prefetch, rows)
    page_table = prefetch["page_table"]
    import tvm_ffi

    with torch.cuda.device(device), tvm_ffi.use_torch_stream():
        module.echo_logits(
            q.view(torch.uint8),
            k.view(torch.uint8),
            weights,
            scales,
            starts,
            ends,
            output,
            page_table,
            extend_lengths,
            query_requests,
            host,
            pool,
            host_to_device,
            device_to_host,
            slots,
            allocations,
            counter,
            stats,
            offsets,
            query_start,
            history_length,
            limit,
        )
    return output[:rows, :columns]
