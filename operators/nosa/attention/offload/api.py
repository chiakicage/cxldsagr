"""NOSA sparse host fetch and native attention in one cooperative CUDA kernel.

One request shares a logically addressed, one-layer K/V staging allocation.
Only selected historical blocks are transferred, once per layer invocation;
Spare producer warps read each selected host record once into HBM. Persistent
FA3 consumers reuse that staging through TMA after per-page publication. The
native numerical repair follows main-kernel completion. This is not a bounded
slot cache policy.
"""

import hashlib
from contextlib import nullcontext
from functools import cache
from pathlib import Path

import torch

from layers.attention import BlockSelection
from operators.nosa.attention.workspace import NosaAttentionWorkspace


@cache
def _module():
    import tvm_ffi.cpp

    source = Path(__file__).with_name("csrc") / "nosa_offload.cu"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    return tvm_ffi.cpp.load(
        name=f"cxldsagr_nosa_offload_{digest}",
        sources=[str(source)],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_90,code=sm_90",
            "-lineinfo",
        ],
    )


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value < 2**31:
        raise ValueError(f"{name} must be a positive int32 integer")


def _validate_cache_tags(tags, owner, keys, history_tokens):
    if type(owner) is not int or not 0 < owner < 2**63:
        raise ValueError("cache_owner must be a positive int64 session identity")
    if (
        not isinstance(tags, torch.Tensor)
        or tags.dtype != torch.int64
        or tags.device != keys.device
        or tags.ndim != 2
        or tags.numel() == 0
        or tags.shape[1] != keys.shape[1]
        or not (history_tokens + 63) // 64 <= tags.shape[0] <= (len(keys) + 63) // 64
        or not tags.is_contiguous()
        or tags.requires_grad
    ):
        raise ValueError("cache_tags must be contiguous GPU int64 [history pages, KV heads]")


@torch.inference_mode()
def prefetch_cached_history(
    keys, values, host_keys, host_values, cache_tags, cache_owner, history_tokens, *, out=None
):
    """Copy historical page misses into a logically addressed, persistent pool.

    All complete pages are tagged only after their writers finish. Partial pages
    are copied on every invocation. The caller owns the exclusive pool lease and
    must fence consumers, backing mutation and storage release against this
    stream. Returned bytes count K/V payload read from host, not link traffic.
    ``out`` permits a backend to reuse its preallocated int64 counter.
    """
    if type(history_tokens) is not int or not 0 <= history_tokens <= len(keys):
        raise ValueError("history_tokens must fit the pool")
    if (
        keys.device.type != "cuda"
        or keys.ndim != 3
        or keys.shape[-1] != 128
        or keys.dtype != torch.bfloat16
        or values.shape != keys.shape
        or values.device != keys.device
        or values.dtype != keys.dtype
        or any(not t.is_contiguous() or t.requires_grad for t in (keys, values))
        or any(t.data_ptr() % 16 for t in (keys, values))
    ):
        raise ValueError("Cached prefetch requires contiguous CUDA BF16 [capacity,heads,128]")
    _validate_cache_tags(cache_tags, cache_owner, keys, history_tokens)
    for host in (host_keys, host_values):
        if (
            host.device.type != "cpu"
            or not host.is_pinned()
            or host.dtype != keys.dtype
            or host.ndim != 3
            or len(host) < history_tokens
            or host.shape[1:] != keys.shape[1:]
            or not host.is_contiguous()
            or host.data_ptr() % 16
            or host.requires_grad
        ):
            raise ValueError("Cached prefetch requires contiguous pinned historical K/V")
    if out is None:
        out = torch.empty((), dtype=torch.int64, device=keys.device)
    if out.shape != () or out.dtype != torch.int64 or out.device != keys.device:
        raise ValueError("Cached prefetch counter must be a GPU int64 scalar")
    tensors = (keys, values, cache_tags, out)
    if len({t.untyped_storage().data_ptr() for t in tensors}) != len(tensors):
        raise ValueError("Cached prefetch writable storages must be independent")
    import tvm_ffi

    with torch.cuda.device(keys.device), tvm_ffi.use_torch_stream():
        if torch.cuda.is_current_stream_capturing():
            raise NotImplementedError("Cached prefetch does not support CUDA Graph capture")
        stream = torch.cuda.current_stream(keys.device)
        for tensor in tensors:
            tensor.record_stream(stream)
        _module().prefetch_cached(
            keys, values, host_keys, host_values, cache_tags, out, cache_owner, history_tokens
        )
    return out


def estimate_trace_rows(max_seq_len, max_queries, kv_heads):
    """Pure upper bound: page envelopes, math rows and eight copy stripes/page."""
    for name, value in (
        ("max_seq_len", max_seq_len),
        ("max_queries", max_queries),
        ("kv_heads", kv_heads),
    ):
        _positive_integer(name, value)
    if max_queries > max_seq_len:
        raise ValueError("max_queries exceeds sequence capacity")
    return ((max_seq_len + 63) // 64) * kv_heads * 9 + ((max_queries + 7) // 8) * kv_heads * 64


class NosaFetchWorkspace:
    """Request-shared staging and exact sparse fetch scheduling for NOSA-8B.

    The initial implementation supports BF16, D128 and GQA16 with the native
    SM90 FA3 backend. Its FP16 fallback scans all V, which is incompatible with
    concurrently writing not-yet-consumed blocks, so that path is rejected.

    Inputs and pinned backing must remain valid until the caller stream has
    completed, or until :meth:`synchronize` returns. Calls may reuse this object
    sequentially on different streams; concurrent Python calls are unsupported.
    The result is ready in the calling stream without a host synchronization.
    ``overlap=False`` fetches the complete sparse union before the original
    full-query FA3 attention. ``query_tile_size`` groups first-use byte counters;
    neither path splits attention into query tiles. ``fetch_ctas`` caps how many
    CTAs contribute spare producer warps to unique host reads from a compact
    selected-page queue. All CTAs retain their attention consumers; the cap is
    clamped to the cooperative grid size.

    ``bounded=True`` requires ``max_queries`` and reserves all scratch at
    construction. ``trace_capacity`` is an explicit number of four-int64 rows;
    neither queries nor tracing can grow that reservation. CPU construction is
    allowed only in bounded mode for allocation tests, and cannot execute fetch.
    The default retains lazy standalone scratch and diagnostic allocation.
    """

    def __init__(
        self,
        max_seq_len: int,
        kv_heads: int,
        head_dim: int,
        *,
        device,
        dtype,
        query_tile_size: int = 128,
        overlap: bool = True,
        fetch_ctas: int = 96,
        max_queries: int | None = None,
        bounded: bool = False,
        trace_capacity: int = 0,
    ):
        _positive_integer("max_seq_len", max_seq_len)
        _positive_integer("kv_heads", kv_heads)
        _positive_integer("query_tile_size", query_tile_size)
        _positive_integer("fetch_ctas", fetch_ctas)
        if not isinstance(bounded, bool):
            raise TypeError("bounded must be boolean")
        self.device = torch.device(device)
        allocation_only = bounded and self.device.type == "cpu"
        if not allocation_only and (
            head_dim != 128 or isinstance(head_dim, bool) or dtype != torch.bfloat16
        ):
            raise ValueError("NOSA offload currently requires BF16 and head_dim=128")
        # The pure layout contract also validates allocation-only CPU shapes.
        self.estimate_capacity_bytes(
            max_seq_len,
            kv_heads,
            head_dim,
            dtype=dtype,
            query_tile_size=query_tile_size,
            max_queries=max_queries,
            trace_capacity=trace_capacity,
        )
        if bounded and max_queries is None:
            raise ValueError("Bounded NOSA fetch workspace requires max_queries")
        if kv_heads > 65535:
            raise ValueError("kv_heads exceeds the CUDA grid limit")
        page_capacity = ((max_seq_len + 63) // 64) * kv_heads
        _positive_integer("logical page capacity", page_capacity)
        if not isinstance(overlap, bool):
            raise TypeError("overlap must be boolean")
        if self.device.type != "cuda" and not allocation_only:
            raise NotImplementedError("NOSA fetch/attention overlap requires SM90/Hopper")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if not allocation_only and torch.cuda.get_device_capability(self.device) != (9, 0):
            raise NotImplementedError("NOSA fetch/attention overlap requires SM90/Hopper")
        if bounded and not allocation_only:
            from operators.nosa._native import native_enabled

            if not native_enabled():
                raise NotImplementedError("Shared NOSA offload requires the native FA3 backend")
        with nullcontext() if allocation_only else torch.cuda.device(self.device):
            if not allocation_only and torch.cuda.is_current_stream_capturing():
                raise NotImplementedError("NOSA offload does not support CUDA Graph capture")
            self.max_seq_len = max_seq_len
            self.kv_heads = kv_heads
            self.head_dim = head_dim
            self.dtype = dtype
            self.query_tile_size = query_tile_size
            self.overlap = overlap
            self.fetch_ctas = fetch_ctas
            self.bounded = False
            self.max_queries = max_queries
            self._attention_scratch = None
            shape = (max_seq_len, kv_heads, head_dim)
            self.keys = torch.empty(shape, dtype=dtype, device=self.device)
            self.values = torch.empty_like(self.keys)
            self._first_use = torch.empty(
                ((max_seq_len + 63) // 64, kv_heads), dtype=torch.int32, device=self.device
            )
            self._tile_bytes = torch.empty(
                ((max_seq_len + query_tile_size - 1) // query_tile_size,),
                dtype=torch.int64,
                device=self.device,
            )
            self.last_transfer_bytes = torch.zeros((), dtype=torch.int64, device=self.device)
            self.last_tile_transfer_bytes = self._tile_bytes[:0]
            self._empty_mask = torch.empty(0, dtype=torch.bool, device=self.device)
            self._ready_blocks = torch.empty_like(self._first_use)
            # Capacity-sized and independent of query scratch: [count, cursor,
            # unique logical page slots]. Compaction resets the header per run.
            self._fetch_queue = torch.empty(
                (page_capacity + 2,), dtype=torch.int32, device=self.device
            )
            self._scratch_capacity = 0
            self._scratch = {}
            self.profile_work_intervals = False
            self._trace_storage = torch.empty((0, 4), dtype=torch.int64, device=self.device)
            self.last_work_intervals = self._trace_storage
            self._last_done = None if allocation_only else torch.cuda.Event()
            if self._last_done is not None:
                self._last_done.record(torch.cuda.current_stream(self.device))
            if bounded:
                self._attention_scratch = NosaAttentionWorkspace(
                    max_queries, kv_heads, head_dim, device=self.device, dtype=dtype
                )
                self._scratch = self._attention_scratch._storage
                self._scratch_capacity = max_queries
            elif max_queries is not None:
                self._ensure_scratch(max_queries)
            if trace_capacity:
                self._trace_storage = torch.empty(
                    (trace_capacity, 4), dtype=torch.int64, device=self.device
                )
                self.last_work_intervals = self._trace_storage[:0]
            self.bounded = bounded

    @staticmethod
    def allocation_sizes(
        max_seq_len,
        kv_heads,
        head_dim,
        *,
        dtype,
        query_tile_size=128,
        max_queries=None,
        trace_capacity=0,
    ):
        """Pure independent payload sizes in ``tensors()`` order.

        Scratch aliases are counted once. Empty mask/trace buffers contribute
        explicit zero entries; the caller supplies any allocator rounding.
        """
        for name, value in (
            ("max_seq_len", max_seq_len),
            ("kv_heads", kv_heads),
            ("head_dim", head_dim),
            ("query_tile_size", query_tile_size),
        ):
            _positive_integer(name, value)
        if kv_heads > 65535:
            raise ValueError("kv_heads exceeds the CUDA grid limit")
        sizes = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}
        if dtype not in sizes:
            raise ValueError("NOSA workspace requires a floating point dtype")
        if (
            isinstance(trace_capacity, bool)
            or not isinstance(trace_capacity, int)
            or trace_capacity < 0
        ):
            raise ValueError("trace_capacity must be a nonnegative row count")
        pages = ((max_seq_len + 63) // 64) * kv_heads
        _positive_integer("logical page capacity", pages)
        kv_bytes = max_seq_len * kv_heads * head_dim * sizes[dtype]
        scratch_sizes = ()
        if max_queries is not None:
            _positive_integer("max_queries", max_queries)
            if max_queries > max_seq_len:
                raise ValueError("max_queries exceeds sequence capacity")
            # The fetch workspace exposes its own empty mask. The nested
            # attention mask has no storage and is not a second allocation.
            scratch_sizes = NosaAttentionWorkspace.allocation_sizes(
                max_queries, kv_heads, head_dim, dtype=dtype
            )[:-1]
        return (
            kv_bytes,
            kv_bytes,
            pages * 4,
            pages * 4,
            (pages + 2) * 4,
            ((max_seq_len + query_tile_size - 1) // query_tile_size) * 8,
            8,
            trace_capacity * 4 * 8,
            *scratch_sizes,
            0,
        )

    @staticmethod
    def estimate_capacity_bytes(
        max_seq_len,
        kv_heads,
        head_dim,
        *,
        dtype,
        query_tile_size=128,
        max_queries=None,
        trace_capacity=0,
    ):
        """Pure byte contract for staging, metadata, explicit scratch and trace."""
        return sum(
            NosaFetchWorkspace.allocation_sizes(
                max_seq_len,
                kv_heads,
                head_dim,
                dtype=dtype,
                query_tile_size=query_tile_size,
                max_queries=max_queries,
                trace_capacity=trace_capacity,
            )
        )

    estimate_trace_rows = staticmethod(estimate_trace_rows)

    def reserve(self, queries, *, trace_rows=0):
        """Reserve explicit standalone capacity; bounded instances never grow."""
        _positive_integer("queries", queries)
        if queries > self.max_seq_len:
            raise ValueError("queries exceeds sequence capacity")
        if isinstance(trace_rows, bool) or not isinstance(trace_rows, int) or trace_rows < 0:
            raise ValueError("trace_rows must be a nonnegative row count")
        if self.bounded and (
            queries > self._scratch_capacity or trace_rows > len(self._trace_storage)
        ):
            raise ValueError("Bounded NOSA workspace cannot grow beyond its reservation")
        self.synchronize()
        self._ensure_scratch(queries)
        if trace_rows > len(self._trace_storage):
            self._trace_storage = torch.empty(
                (trace_rows, 4), dtype=torch.int64, device=self.device
            )
            self.last_work_intervals = self._trace_storage[:0]

    def _validate(
        self,
        q,
        selection,
        host_keys,
        host_values,
        suffix_keys,
        suffix_values,
        cis_bias,
        query_start,
    ):
        if self.device.type != "cuda":
            raise NotImplementedError("CPU NOSA fetch workspace is allocation-only")
        from operators.nosa._native import native_enabled

        if not native_enabled():
            raise NotImplementedError("NOSA offload requires the native FA3 attention backend")
        if isinstance(query_start, bool) or not isinstance(query_start, int) or query_start < 0:
            raise ValueError("query_start must be a nonnegative integer")
        if q.ndim != 3 or q.shape[1:] != (self.kv_heads * 16, self.head_dim):
            raise ValueError("NOSA offload Q must be [queries,KV heads * 16,128]")
        queries = q.shape[0]
        if self.bounded:
            if queries > self.max_queries:
                raise ValueError("NOSA offload query exceeds the reserved query capacity")
            if self.profile_work_intervals:
                rows = ((query_start + 63) // 64) * self.kv_heads * 9
                rows += ((queries + 7) // 8) * self.kv_heads * 64
                if rows > len(self._trace_storage):
                    raise ValueError("NOSA offload trace exceeds its explicit reservation")
        end = query_start + queries
        if end > self.max_seq_len:
            raise ValueError("Query end exceeds the workspace capacity")
        if not isinstance(selection, BlockSelection) or selection.block_size != 64:
            raise ValueError("NOSA offload requires a 64-token BlockSelection")
        ids = selection.block_ids
        if (
            ids.ndim != 3
            or ids.shape[0] not in (1, queries)
            or ids.shape[1] not in (1, self.kv_heads)
            or not 1 <= ids.shape[2] <= 64
            or ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Block IDs must be integral [queries or 1,KV heads or 1,1..64]")
        mask = selection.valid_mask
        if mask is not None and (mask.shape != ids.shape or mask.dtype != torch.bool):
            raise ValueError("Selection validity must be boolean and match the block IDs")
        expected_suffix = (queries, self.kv_heads, self.head_dim)
        if suffix_keys.shape != expected_suffix or suffix_values.shape != expected_suffix:
            raise ValueError("Suffix K/V must contain exactly the current queries in NHD layout")
        gpu_tensors = [q, suffix_keys, suffix_values, ids]
        if mask is not None:
            gpu_tensors.append(mask)
        if cis_bias is not None:
            if cis_bias.shape != (end, self.kv_heads) or cis_bias.dtype not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ):
                raise ValueError("CIS bias must be floating point [prefix + queries,KV heads]")
            gpu_tensors.append(cis_bias)
        if any(t.device != self.device for t in gpu_tensors):
            raise ValueError("NOSA offload GPU tensors must be on the workspace device")
        if any(t.dtype != self.dtype for t in (q, suffix_keys, suffix_values)):
            raise ValueError("Q and suffix K/V must use the workspace BF16 dtype")
        if any(t.stride(-1) != 1 for t in (q, suffix_keys, suffix_values)):
            raise ValueError("Q and suffix K/V require a contiguous innermost dimension")
        if q.numel() and (
            q.data_ptr() % 16 != 0
            or q.stride(0) <= 0
            or q.stride(1) <= 0
            or q.stride(0) % 8 != 0
            or q.stride(1) % 8 != 0
        ):
            raise ValueError("Q pointer and row/head strides must satisfy native FA3 alignment")
        if any(any(stride < 0 for stride in t.stride()) for t in gpu_tensors):
            raise ValueError("NOSA offload does not support negative tensor strides")
        writable_storages = {
            tensor.untyped_storage().data_ptr()
            for tensor in (
                self.keys,
                self.values,
                self._first_use,
                self._ready_blocks,
                self._fetch_queue,
                self._tile_bytes,
                self.last_transfer_bytes,
                self._trace_storage,
                *self._scratch.values(),
            )
            if tensor.numel()
        }
        if any(t.untyped_storage().data_ptr() in writable_storages for t in gpu_tensors):
            raise ValueError("NOSA offload inputs must not alias writable workspace storage")
        for host in (host_keys, host_values):
            if (
                host.device.type != "cpu"
                or not host.is_pinned()
                or host.dtype != self.dtype
                or host.ndim != 3
                or host.shape[0] < query_start
                or host.shape[1:] != (self.kv_heads, self.head_dim)
                or not host.is_contiguous()
                or (host.numel() and host.data_ptr() % 16 != 0)
            ):
                raise ValueError("Historical K/V require contiguous, pinned BF16 CPU NHD backing")
        if host_keys.shape != host_values.shape:
            raise ValueError("Historical K/V backing shapes must match")
        if any(t.requires_grad for t in (*gpu_tensors, host_keys, host_values)):
            raise ValueError("NOSA offload is inference-only")

    @torch.inference_mode()
    def run(
        self,
        q: torch.Tensor,
        selection: BlockSelection,
        host_keys: torch.Tensor,
        host_values: torch.Tensor,
        suffix_keys: torch.Tensor,
        suffix_values: torch.Tensor,
        cis_bias: torch.Tensor | None,
        query_start: int,
        *,
        cache_tags: torch.Tensor | None = None,
        cache_owner: int | None = None,
    ) -> torch.Tensor:
        """Return exact selected attention while producers fetch historical KV.

        ``last_transfer_bytes`` and ``last_tile_transfer_bytes`` are GPU int64
        tensors, reset on every call. They count K + V bytes read from pinned
        historical storage, including only the historical part of a tail block;
        they exclude the already-resident suffix. Per-tile values group pages by
        their first requesting query, not by physical transfer phases. Inspect
        them after completion. Profiling may enable ``profile_work_intervals``
        to record device-globaltimer fetch and softmax intervals outside waits.
        """
        self._validate(
            q, selection, host_keys, host_values, suffix_keys, suffix_values, cis_bias, query_start
        )
        if cache_tags is not None:
            _validate_cache_tags(cache_tags, cache_owner, self.keys, query_start)
            if (
                self.keys.shape != (self.max_seq_len, self.kv_heads, self.head_dim)
                or self.values.shape != self.keys.shape
                or self.keys.untyped_storage().data_ptr()
                == self.values.untyped_storage().data_ptr()
                or any(
                    t.dtype != self.dtype or t.device != self.device or not t.is_contiguous()
                    for t in (self.keys, self.values)
                )
            ):
                raise ValueError("Persistent cache storage must match workspace geometry")
            if cache_tags.untyped_storage().data_ptr() in {
                tensor.untyped_storage().data_ptr()
                for tensor in (
                    *self.tensors(),
                    q,
                    suffix_keys,
                    suffix_values,
                    selection.block_ids,
                    *((cis_bias,) if cis_bias is not None else ()),
                    *((selection.valid_mask,) if selection.valid_mask is not None else ()),
                )
            }:
                raise ValueError("Persistent cache tags must not alias workspace or inputs")
        elif cache_owner is not None:
            raise ValueError("cache_owner requires cache_tags")
        import tvm_ffi

        from operators.nosa.attention.offload._fused import _module as fused_module

        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise NotImplementedError("NOSA offload does not support CUDA Graph capture")
            module = _module()
            fused = fused_module()
            current = torch.cuda.current_stream(self.device)
            current.wait_event(self._last_done)
            try:
                queries = q.shape[0]
                tiles = (queries + self.query_tile_size - 1) // self.query_tile_size
                self.last_tile_transfer_bytes = self._tile_bytes[:tiles]
                self._ensure_scratch(queries)
                scratch = (
                    self._attention_scratch.slices(queries)
                    if self._attention_scratch is not None
                    else self._scratch
                )
                trace_rows = ((query_start + 63) // 64) * self.kv_heads
                trace_rows += ((queries + 7) // 8) * self.kv_heads * 64
                if self.profile_work_intervals:
                    from operators.nosa.attention.offload._fused import build_info

                    trace_rows += (
                        ((query_start + 63) // 64) * self.kv_heads * build_info()["fetch_stripes"]
                    )
                    if len(self._trace_storage) < trace_rows:
                        if self.bounded:
                            raise ValueError("NOSA offload trace exceeds its explicit reservation")
                        self._trace_storage = torch.empty(
                            (trace_rows, 4), dtype=torch.int64, device=self.device
                        )
                    self.last_work_intervals = self._trace_storage[:trace_rows]
                else:
                    self.last_work_intervals = self._trace_storage[:0]
                mask = (
                    selection.valid_mask if selection.valid_mask is not None else self._empty_mask
                )
                # Register lifetimes before the first asynchronous mutation,
                # including empty calls and failures before entering the FFI.
                for tensor in (
                    q,
                    suffix_keys,
                    suffix_values,
                    selection.block_ids,
                    mask,
                    self.keys,
                    self.values,
                    self._first_use,
                    self._ready_blocks,
                    self._fetch_queue,
                    self._tile_bytes,
                    self.last_transfer_bytes,
                    self.last_work_intervals,
                    *self._scratch.values(),
                ):
                    tensor.record_stream(current)
                if cis_bias is not None:
                    cis_bias.record_stream(current)
                if cache_tags is not None:
                    cache_tags.record_stream(current)
                if self.profile_work_intervals:
                    self.last_work_intervals.zero_()
                # Native preparation clears the full reusable metadata capacity,
                # defines historical page-zero padding and stages strided suffix K/V.
                end = query_start + queries
                output = torch.empty_like(q, memory_format=torch.contiguous_format)
                with tvm_ffi.use_torch_stream():
                    prepare = module.prepare if cache_tags is None else module.prepare_cached
                    prepare(
                        self.keys,
                        self.values,
                        suffix_keys,
                        suffix_values,
                        self._first_use,
                        self._ready_blocks,
                        self._fetch_queue,
                        self._tile_bytes,
                        self.last_transfer_bytes,
                        selection.block_ids,
                        mask,
                        query_start,
                        self.query_tile_size,
                        *((cache_tags, cache_owner) if cache_tags is not None else ()),
                    )
                    if not queries:
                        if cache_tags is not None:
                            module.publish_cache(
                                cache_tags, self._first_use, cache_owner, query_start, end
                            )
                        return output
                    fused.forward(
                        q,
                        self.keys[:end],
                        self.values[:end],
                        selection.block_ids,
                        mask,
                        cis_bias if cis_bias is not None else self._empty_mask,
                        output,
                        scratch["fallback"],
                        scratch["pages"],
                        scratch["members"],
                        scratch["counts"],
                        host_keys,
                        host_values,
                        self._first_use,
                        self._ready_blocks,
                        self.last_tile_transfer_bytes,
                        self.last_transfer_bytes,
                        self.last_work_intervals,
                        self._fetch_queue,
                        query_start,
                        self.query_tile_size,
                        self.fetch_ctas,
                        self.overlap,
                    )
                    if cache_tags is not None:
                        module.publish_cache(
                            cache_tags, self._first_use, cache_owner, query_start, end
                        )
                return output
            except BaseException:
                # Preparation may already have overwritten another owner's page
                # zero or suffix. No stale tag may survive a failed submission.
                if cache_tags is not None:
                    cache_tags.zero_()
                raise
            finally:
                # All producer, consumer and repair work belongs to the caller
                # stream, including already-enqueued work if a later launch fails.
                self._last_done.record(current)

    __call__ = run

    def _ensure_scratch(self, queries):
        if queries <= self._scratch_capacity:
            return
        if self.bounded:
            raise ValueError("NOSA offload query exceeds the reserved query capacity")
        batches = ((queries + 7) // 8) * self.kv_heads
        options = {"device": self.device, "dtype": torch.int32}
        self._scratch = {
            "fallback": torch.empty(((queries + 3) // 4, self.kv_heads), **options),
            "pages": torch.empty((batches, 8 * 64), **options),
            "members": torch.empty((batches, 8 * 64), **options),
            # The 256-work-item native sort uses a second counts-sized region.
            "counts": torch.empty((2 * batches,), **options),
        }
        self._scratch_capacity = queries

    def work_intervals(self):
        """Export optional device-clock evidence after completion, outside timing.

        Fetch rows identify a unique historical (block, head). Math rows cover
        softmax updates only; they exclude ready waits and do not represent the
        full attention execution window. Absolute clocks are not mixed with nsys.
        """
        return [
            {"row": index, "start_ns": row[0], "end_ns": row[1], "bytes": row[2], "kind": row[3]}
            for index, row in enumerate(self.last_work_intervals.detach().cpu().tolist())
            if row[3] in (1, 2)
        ]

    def stripe_work_intervals(self):
        """Export nonempty stripe copy windows for independent overlap audit."""
        return [
            {"row": index, "start_ns": row[0], "end_ns": row[1], "bytes": row[2], "kind": row[3]}
            for index, row in enumerate(self.last_work_intervals.detach().cpu().tolist())
            if row[3] == 3
        ]

    @property
    def capacity_bytes(self) -> int:
        """Allocated HBM tensor bytes, counting statistics views only once."""
        return sum(tensor.untyped_storage().nbytes() for tensor in self.tensors())

    def tensors(self):
        """Enumerate every complete tensor storage, never statistics views."""
        return (
            self.keys,
            self.values,
            self._first_use,
            self._ready_blocks,
            self._fetch_queue,
            self._tile_bytes,
            self.last_transfer_bytes,
            self._trace_storage,
            *self._scratch.values(),
            self._empty_mask,
        )

    def synchronize(self):
        """Wait on the host before cache abort, backing mutation or release."""
        if self._last_done is not None:
            self._last_done.synchronize()
