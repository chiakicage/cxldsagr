"""Intrusive allocator-event audit, separate from all formal latency samples.

Wrap the existing model scopes and projection methods only during this audit.
Allocation ownership follows CPU launch scopes; deallocation is tracked globally,
including after a scope exits. Projected KV storage is charged from its original
allocation, not from the later append. CUDA events contain allocator block sizes;
CPU events contain PyTorch's reported pageable allocation sizes, not malloc
usable sizes. Pinned CPU blocks require a separate fixed-storage inventory.

This observes PyTorch allocator allocations. Direct library cudaMalloc/malloc,
CUDA allocator segments/fragmentation and non-PyTorch memory are not proved by
this audit. Process peak allocated/reserved memory must be reported separately.
"""

from __future__ import annotations

import argparse
import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

SCOPE_PREFIX = "echo_cache_memory.scope|"
SOURCE_PREFIX = "echo_cache_memory.source|"
ORDINARY_PREFIX = "echo_cache_memory.ordinary|"
SCALAR_PREFIX = "echo_cache_memory.scalar|"
SCALAR_RESULT_PREFIX = "echo_cache_memory.scalar_result|"
CACHE_CATEGORIES = frozenset(("indexer", "cache_helpers", "copy_source", "diagnostics"))
INDEXER_SCOPES = frozenset(("indexer", "indexer_prefetch", "exact_topk"))
HELPER_SCOPES = frozenset(
    (
        "index_cache_write",
        "offload_prepare",
        "offload_finalize",
        "cache_write",
        "offload_exact_recall",
        "offload_source_reservation",
        "cache_maintenance",
    )
)


def scope_category(name):
    if name in INDEXER_SCOPES:
        return "indexer"
    if name in HELPER_SCOPES:
        return "cache_helpers"
    if name.startswith("cache_diagnostic_"):
        return "diagnostics"
    return "ordinary"


@dataclass(frozen=True)
class WorkspaceAllowance:
    """Additive admission terms; helpers can consume selection and metadata terms.

    Scope categories are attribution, not independently enforceable sub-budgets.
    For example, exact recall's Q*K union uses the selection allowance, although
    it executes in a helper scope. Only their simultaneous total is compared.
    """

    logits: int
    selection: int
    hint: int
    metadata: int
    copy_source: int
    paged_q1_extra: int = 0
    attention_q1_extra: int = 0

    @property
    def total(self):
        return (
            self.logits
            + self.selection
            + self.hint
            + self.metadata
            + self.copy_source
            + self.paged_q1_extra
            + self.attention_q1_extra
        )

    @classmethod
    def echo(cls, *, queries, context, topk, width, host_tokens, pool_tokens, inflight=2):
        values = (queries, context, topk, width, host_tokens, pool_tokens, inflight)
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("workspace dimensions must be positive integers")
        columns = (context + 127) // 128 * 128
        logits = 16 * queries * columns
        selection = 64 * queries * min(topk, context)
        hint = 32 * columns
        paged_q1_extra = 0
        # Independent SM90 audit contract: resident Q1 uses paged logits from
        # N=32768. A larger maximum query size may still execute a Q1 tail.
        # Keep this calculation independent of the production reservation.
        if context >= 32768:
            pages = (context + 63) // 64
            paged_columns = (context + 255) // 256 * 256
            paged_q1 = (
                48 * paged_columns
                + 64 * min(topk, context)
                + pages * 64 * (128 + 4)
                + pages * 4
                + (132 + 1) * 2 * 4  # Two int32 schedule values for up to 132 SM90 SMs.
                + 4 * context
                + 64 * (width * 2 + 4)  # Official token table and 64-record stage.
            )
            paged_q1_extra = max(0, paged_q1 - (logits + selection + hint))
        attention_q1_extra = 0
        if min(topk, context) >= 2048:
            # Independent H128, 16-split BF16 decode contract. Existing model
            # KV is aligned/contiguous; include prepared/repeated Q, partials,
            # both FP32 statistics, merged output and selected-ID temporaries.
            decode = 128 * (16 * (width * 2 + 512 * 2 + 8) + width * 2 + 512 * 2) + 32 * 2048
            attention_q1_extra = max(0, decode - (logits + selection + hint + paged_q1_extra))
        return cls(
            logits,
            selection,
            hint,
            64 * (pool_tokens + 1) + 64 * host_tokens,
            inflight * queries * width * 2,
            paged_q1_extra,
            attention_q1_extra,
        )


def storage_inventory(tensors, *, cuda_snapshot=None):
    """Inventory fixed-cache tensors, including CUDA padding and pinned host bins.

    This is a baseline ledger, not a temporary peak measurement. Callers must
    enumerate cache-owned tensors only (including aliases, which are deduplicated).
    Subtract its allocator bytes from the full cache reservation to obtain the
    ``limits`` passed to ``audit_trace``. Pinned storage uses PyTorch's installed
    CachingHostAllocator power-of-two contract, independently of runtime helpers.
    CPU malloc overhead remains unobserved. This does not include cached pinned
    blocks that are no longer owned by any supplied tensor.
    A snapshot can be supplied for independent testing or a coordinated capture.
    """
    tensors = tuple(tensors)
    if cuda_snapshot is None and any(tensor.device.type == "cuda" for tensor in tensors):
        cuda_snapshot = torch.cuda.memory_snapshot()
    blocks = {}
    for segment in cuda_snapshot or ():
        address = segment["address"]
        for block in segment["blocks"]:
            start = block.get("address", address)
            if block["state"] == "active_allocated":
                blocks[(f"cuda:{segment['device']}", start)] = block["size"]
            address = start + block["size"]
    rows, seen, devices = [], set(), {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        device, address, size = str(tensor.device), storage.data_ptr(), storage.nbytes()
        key = (device, address)
        if not size or key in seen:
            continue
        seen.add(key)
        pinned = tensor.device.type == "cpu" and tensor.is_pinned()
        allocated = (
            blocks.get(key)
            if tensor.device.type == "cuda"
            else 1 << (size - 1).bit_length()
            if pinned
            else size
        )
        if allocated is None or allocated < size:
            raise ValueError(f"fixed CUDA storage has no covering active allocator block: {key}")
        row = {
            "device": device,
            "address": address,
            "storage_bytes": size,
            "allocator_bytes": allocated,
            "pinned": pinned,
        }
        rows.append(row)
        totals = devices.setdefault(
            device, {"storage_bytes": 0, "allocator_bytes": 0, "padding_bytes": 0}
        )
        totals["storage_bytes"] += size
        totals["allocator_bytes"] += allocated
        totals["padding_bytes"] += allocated - size
    return {
        "devices": devices,
        "storages": rows,
        "cpu_malloc_overhead_observed": False,
        "pinned_capacity_basis": "independent next power of two of owned storage bytes",
    }


def host_allocator_counters():
    """Global counters, never object ownership or a precise concurrent peak.

    Installed torch 2.12.1 has a no-stream free accounting defect: active.current
    increases on cached-block reuse without a matching decrease. Even without
    that defect its documented peak sums bucket peaks. Cumulative handouts can
    still reveal pinned allocations absent from profiler memory events.
    """
    values = torch.cuda.memory.host_memory_stats()
    return {
        name: int(values[name])
        for name in (
            "allocated_bytes.current",
            "allocated_bytes.peak",
            "active_bytes.allocated",
        )
    }


class CacheMemoryAudit:
    """One trace, with no latency claim and no persistent runtime modification.

    Warm compilation and allocate fixed pools before entering. Supply every CUDA
    device used by the call. Use ``audit.scopes`` as the model's ``scope`` argument
    and wrap its CheckpointAttention objects with ``track_projected_kv``. Include
    commit/drain in the context so retained copy sources have observable releases.
    An outer ``audit.scopes('cache_maintenance')`` can cover lifecycle helpers.
    """

    def __init__(
        self,
        trace_path,
        *,
        devices=(),
        history_max_entries=2_000_000,
        record_shapes=True,
        with_stack=True,
    ):
        self.trace_path = Path(trace_path)
        self.devices = tuple(torch.device(device) for device in devices)
        if any(device.type != "cuda" or device.index is None for device in self.devices):
            raise ValueError("devices must contain explicit CUDA indices, such as cuda:0")
        self.profiler = None
        self.history_path = self.trace_path.with_suffix(".cuda-memory.json")
        self.history_max_entries = history_max_entries
        self.record_shapes, self.with_stack = record_shapes, with_stack
        self.scalar_sequence = 0
        if type(history_max_entries) is not int or history_max_entries < 1:
            raise ValueError("history_max_entries must be a positive integer")

    @contextmanager
    def scopes(self, name):
        category = scope_category(name)
        with torch.profiler.record_function(f"{SCOPE_PREFIX}{category}|{name}"):
            yield

    def mark_ordinary_result(self, tensor):
        """Demote a returned hidden/logit storage allocated outside a named stage.

        In particular, serving's final concatenation of all prefix hidden states
        is an ordinary output even though it occurs after the last named stage.
        No tensor copy or additional retained reference is introduced.
        """
        storage = tensor.untyped_storage()
        marker = f"{tensor.device}|{storage.data_ptr()}|{storage.nbytes()}"
        with torch.profiler.record_function(ORDINARY_PREFIX + marker):
            pass

    @contextmanager
    def track_projected_kv(self, attentions):
        """Identify full KV storage without copying it or keeping extra references."""
        originals = []
        try:
            for attention in attentions:
                original = attention.project
                # Preserve whether a bound method was originally shadowed locally.
                had_local = "project" in vars(attention)
                local = vars(attention).get("project")

                def project(*args, _original=original, **kwargs):
                    result = _original(*args, **kwargs)
                    storage = result.kv.untyped_storage()
                    # Kineto does not JSON-escape record_function names. Keep
                    # the pointer marker quote-free so its Chrome trace is valid.
                    marker = f"{result.kv.device}|{storage.data_ptr()}|{storage.nbytes()}"
                    with torch.profiler.record_function(SOURCE_PREFIX + marker):
                        pass
                    return result

                attention.project = project
                originals.append((attention, had_local, local))
            yield
        finally:
            for attention, had_local, local in reversed(originals):
                if had_local:
                    attention.project = local
                else:
                    del attention.project

    @contextmanager
    def track_scalar_transfers(self):
        """Observe Python CUDA scalar conversions absent from profiler memory events.

        Tensor int/bool/float conversions dispatch to the same synchronizing CUDA
        scalar extraction as item(). Each wrapper records the input dtype width
        and actual pinned handout counter delta. The entire synchronous call is
        a conservative lifetime for its scalar buffer. Internal scalar calls
        bypassing these Python entry points remain detectable by the phase-wide
        handout reconciliation and fail closed when unaccounted for.
        """
        originals = {}
        try:
            for name in ("item", "__int__", "__float__", "__bool__"):
                original = getattr(torch.Tensor, name)

                def conversion(tensor, *args, _original=original, **kwargs):
                    if tensor.device.type != "cuda":
                        return _original(tensor, *args, **kwargs)
                    sequence = self.scalar_sequence
                    self.scalar_sequence += 1
                    size = tensor.element_size()
                    capacity = 1 << (size - 1).bit_length()
                    label = f"{SCALAR_PREFIX}{sequence}|{tensor.device}|{capacity}"
                    before = host_allocator_counters()["active_bytes.allocated"]
                    with torch.profiler.record_function(label):
                        result = _original(tensor, *args, **kwargs)
                        delta = host_allocator_counters()["active_bytes.allocated"] - before
                        with torch.profiler.record_function(
                            f"{SCALAR_RESULT_PREFIX}{sequence}|{delta}"
                        ):
                            pass
                    return result

                originals[name] = original
                setattr(torch.Tensor, name, conversion)
            yield
        finally:
            for name, original in originals.items():
                setattr(torch.Tensor, name, original)

    def __enter__(self):
        if self.profiler is not None:
            raise RuntimeError("a memory audit cannot be entered twice")
        if self.trace_path.exists() or self.devices and self.history_path.exists():
            raise FileExistsError(self.trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        for device in self.devices:
            torch.cuda.synchronize(device)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.devices:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            # A separate intrusive process should own this global recorder.
            # Stack capture is unnecessary here: profiler scopes supply ownership.
            torch.cuda.memory._record_memory_history(
                enabled="all", context=None, max_entries=self.history_max_entries
            )
        self.profiler = torch.profiler.profile(
            activities=activities,
            profile_memory=True,
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
        )
        try:
            self.profiler.__enter__()
        except BaseException:
            if self.devices:
                torch.cuda.memory._record_memory_history(enabled=None)
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            for device in self.devices:
                torch.cuda.synchronize(device)
        finally:
            try:
                self.profiler.__exit__(exc_type, exc, traceback)
                self.profiler.export_chrome_trace(str(self.trace_path))
                if self.devices:
                    snapshot = torch.cuda.memory._snapshot()
                    traces = snapshot["device_traces"]
                    history = {
                        "events_by_device": {
                            f"cuda:{device.index}": traces[device.index] for device in self.devices
                        },
                        "possibly_truncated": any(
                            len(traces[device.index]) >= self.history_max_entries
                            for device in self.devices
                        ),
                        "max_entries_per_device": self.history_max_entries,
                    }
                    self.history_path.write_text(json.dumps(history) + "\n")
            finally:
                if self.devices:
                    torch.cuda.memory._record_memory_history(enabled=None)

    def result(self, *, limits, allow_empty_memory=False):
        return audit_trace(
            self.trace_path,
            limits=limits,
            cuda_history=self.history_path if self.devices else None,
            allow_empty_memory=allow_empty_memory,
        )


def _device(args):
    kind = args["Device Type"]
    if kind == 0:
        return "cpu"
    if kind == 1:
        return f"cuda:{args['Device Id']}"
    raise ValueError(f"unsupported profiler memory device type: {kind}")


def _peak(allocations, *, active=False):
    deltas, category_live, category_peak = [], defaultdict(int), defaultdict(int)
    start, end = ("active_start_us", "active_end_us") if active else ("start_us", "end_us")
    for item in allocations:
        # Allocations precede frees at equal timestamps, a conservative tie.
        deltas.append((item[start], 0, item["bytes"], item["category"]))
        if item[end] is not None:
            deltas.append((item[end], 1, -item["bytes"], item["category"]))
    current, peak, peak_at, peak_breakdown = 0, 0, None, {}
    for timestamp, _, size, category in sorted(deltas):
        current += size
        category_live[category] += size
        category_peak[category] = max(category_peak[category], category_live[category])
        if current > peak:
            peak, peak_at, peak_breakdown = current, timestamp, dict(category_live)
    return {
        "bytes": peak,
        "timestamp_us": peak_at,
        "categories_at_peak_bytes": peak_breakdown,
        "independent_category_peak_bytes": dict(category_peak),
        "bytes_live_at_trace_end": current,
    }


def _apply_cuda_history(allocations, history, *, base_us, errors):
    """Join complete per-address generation sequences, including ordinary work.

    Profiler and allocator clocks can drift beyond a short allocation's entire
    lifetime. Pairing by cross-clock containment is therefore unsound. An address
    cannot be reused before its previous completed free: equal complete counts
    give a unique ordinal pairing, checked against bytes and requested-free state.
    All active lifetimes use only the history clock; pending frees stay live.
    """
    if history.get("possibly_truncated", True):
        errors.append("CUDA allocation history may be truncated")
    generations = defaultdict(list)
    for device, events in history["events_by_device"].items():
        live = {}
        for event in events:
            action = event["action"]
            if action not in ("alloc", "free_requested", "free_completed"):
                continue
            address, timestamp = event["addr"], event["time_us"] - base_us
            if action == "alloc":
                previous = generations[(device, address)]
                if (
                    previous
                    and previous[-1]["completed"] is not None
                    and timestamp < previous[-1]["completed"]
                ):
                    errors.append(f"CUDA history allocation order is invalid: {device}/{address}")
                if address in live:
                    errors.append(
                        f"CUDA history address reused before completed free: {device}/{address}"
                    )
                size = event.get("size")
                if type(size) is not int or size <= 0:
                    errors.append(
                        f"CUDA history allocation lacks requested bytes: {device}/{address}"
                    )
                item = {"start": timestamp, "requested": None, "completed": None, "bytes": size}
                live[address] = item
                generations[(device, address)].append(item)
            elif address in live:
                item = live[address]
                field = "requested" if action == "free_requested" else "completed"
                if item[field] is not None:
                    errors.append(f"duplicate CUDA history {action}: {device}/{address}")
                if timestamp < item["start"] or (
                    action == "free_completed"
                    and (item["requested"] is None or timestamp < item["requested"])
                ):
                    errors.append(f"CUDA history free order is invalid: {device}/{address}")
                item[field] = timestamp
                if action == "free_completed":
                    live.pop(address)
            elif generations[(device, address)]:
                errors.append(f"CUDA history free lacks a live generation: {device}/{address}")
    profiler_generations = defaultdict(list)
    for item in allocations:
        profiler_generations[(item["device"], item["address"])].append(item)
    for key in generations.keys() | profiler_generations.keys():
        device, address = key
        recorded, profiled = generations[key], profiler_generations[key]
        if len(recorded) != len(profiled):
            errors.append(
                f"CUDA history/profiler generation count differs: {device}/{address} "
                f"({len(recorded)} versus {len(profiled)})"
            )
            continue
        for item, match in zip(profiled, recorded, strict=True):
            if type(match["bytes"]) is not int or not 0 < match["bytes"] <= item["bytes"]:
                errors.append(
                    f"CUDA history requested bytes exceed profiler block: {device}/{address}"
                )
                continue
            if (item["end_us"] is None) != (match["requested"] is None):
                errors.append(f"CUDA/profiler free evidence disagrees: {device}/{address}")
                continue
            item["active_start_us"] = match["start"]
            item["active_end_us"] = match["completed"]


def _internal_pinned_intervals(runtime, starts, copies, scalar_scopes, errors):
    """Recover native nonzero/scalar scratch from correlated pinned DtoH copies.

    PyTorch nonzero copies its pinned count array synchronously. Native scalar
    extraction (including unique's internal count.item()) similarly owns exactly
    the scalar bytes copied. Both release their CPU tensor before return. GPU
    memcpy events provide actual bytes; no fixed dtype or tensor shape is guessed.
    """
    python_scopes = defaultdict(list)
    for _, event in scalar_scopes.values():
        python_scopes[(event.get("pid"), event.get("tid"))].append(event)
    for events in python_scopes.values():
        events.sort(key=lambda event: event["ts"])
    python_starts = {
        thread: [event["ts"] for event in events] for thread, events in python_scopes.items()
    }
    groups = {}
    for thread, events in runtime.items():
        active = []
        for event in events:
            active = [item for item in active if item["ts"] + item["dur"] >= event["ts"]]
            if event["name"] in ("aten::nonzero", "aten::_local_scalar_dense"):
                active.append(event)
                continue
            if event["name"] != "cudaMemcpyAsync" or not active:
                continue
            transfer = copies.get(event.get("args", {}).get("correlation"))
            if transfer is None:
                continue
            owner = max(active, key=lambda item: (item["ts"], -item["dur"]))
            position = bisect_right(python_starts.get(thread, []), owner["ts"]) - 1
            if position >= 0:
                wrapper = python_scopes[thread][position]
                if owner["ts"] + owner["dur"] <= wrapper["ts"] + wrapper["dur"]:
                    continue  # Already charged with per-call handout evidence.
            group = groups.setdefault(id(owner), {"event": owner, "transfers": []})
            group["transfers"].append(transfer)
    rows = []
    for group in groups.values():
        event, transfers = group["event"], group["transfers"]
        thread = (event.get("pid"), event.get("tid"))
        nested = runtime[thread][
            bisect_left(starts[thread], event["ts"]) : bisect_right(
                starts[thread], event["ts"] + event["dur"]
            )
        ]
        syncs = [item for item in nested if item["name"] == "cudaStreamSynchronize"]
        if (
            len(transfers) != 1
            or len(syncs) != 1
            or (syncs[0]["ts"] + syncs[0]["dur"] > event["ts"] + event["dur"])
        ):
            errors.append(
                f"native pinned buffer lacks one contained synchronous copy: {event['name']}"
            )
            continue
        copied = transfers[0]["args"]["bytes"]
        if type(copied) is not int or copied < 1:
            errors.append("native pinned copy has invalid byte count")
            continue
        rows.append(
            {
                "origin": event["name"],
                "copy_bytes": copied,
                "bytes": 1 << (copied - 1).bit_length(),
                "start_us": event["ts"],
                "end_us": event["ts"] + event["dur"],
                "thread": list(thread),
            }
        )
    return rows


def audit_trace(trace, *, limits, cuda_history=None, allow_empty_memory=False):
    """Reconstruct rounded live allocations, failing closed on malformed evidence.

    ``limits`` maps actual physical devices (``cpu`` / ``cuda:0``) to separately
    reserved temporary bytes. Do not put a CPU-reference device-role HBM number
    under a CUDA key. Fixed pools allocated before capture are excluded naturally;
    they remain the responsibility of the fixed/shared/session storage ledger.
    Positive events inside tracked scopes own the entire allocation until free,
    even if later operations run outside those scopes. Source markers promote the
    matching allocation generation retroactively to its real allocation time.
    """
    if not limits or any(type(size) is not int or size < 0 for size in limits.values()):
        raise ValueError("provide nonnegative integer temporary byte limits per device")
    if isinstance(trace, (str, Path)):
        trace = json.loads(Path(trace).read_text())
    if isinstance(cuda_history, (str, Path)):
        cuda_history = json.loads(Path(cuda_history).read_text())
    timeline, scope_count, memory_count = [], 0, 0
    scalar_scopes, scalar_results, scalar_errors = {}, {}, []
    scalar_runtime = defaultdict(list)
    pinned_copies = {}
    for sequence, event in enumerate(trace["traceEvents"]):
        name = event.get("name", "")
        if name in (
            "aten::_local_scalar_dense",
            "aten::nonzero",
            "cudaMemcpyAsync",
            "cudaStreamSynchronize",
        ):
            scalar_runtime[(event.get("pid"), event.get("tid"))].append(event)
        if event.get("cat") == "gpu_memcpy" and name == "Memcpy DtoH (Device -> Pinned)":
            correlation = event["args"]["correlation"]
            if correlation in pinned_copies:
                scalar_errors.append(f"ambiguous pinned memcpy correlation: {correlation}")
            pinned_copies[correlation] = event
        if name == "[memory]":
            timeline.append((event["ts"], 1, sequence, "memory", event))
            memory_count += 1
        # Kineto also emits GPU user annotations. Ownership is the CPU launch
        # interval, never the delayed or overlapping GPU annotation interval.
        elif event.get("cat") == "user_annotation" and event.get("ph") == "X":
            if name.startswith(SCOPE_PREFIX):
                category, label = name.removeprefix(SCOPE_PREFIX).split("|", 1)
                if category not in CACHE_CATEGORIES | {"ordinary"}:
                    raise ValueError(f"unknown memory scope category: {category}")
                if event["dur"] < 0:
                    raise ValueError("negative memory scope duration")
                item = (category, label, sequence, event)
                timeline.append((event["ts"], 0, -event["dur"], "start", item))
                timeline.append((event["ts"] + event["dur"], 3, sequence, "end", item))
                scope_count += category in CACHE_CATEGORIES
            elif name.startswith(SOURCE_PREFIX):
                timeline.append((event["ts"], 2, sequence, "source", event))
            elif name.startswith(ORDINARY_PREFIX):
                timeline.append((event["ts"], 2, sequence, "ordinary", event))
            elif name.startswith(SCALAR_RESULT_PREFIX):
                scalar_id, handout = map(int, name.removeprefix(SCALAR_RESULT_PREFIX).split("|"))
                if scalar_id in scalar_results:
                    scalar_errors.append(f"duplicate pinned scalar result: {scalar_id}")
                scalar_results[scalar_id] = handout
            elif name.startswith(SCALAR_PREFIX):
                scalar_id, device, capacity = name.removeprefix(SCALAR_PREFIX).split("|")
                scalar_id, capacity = int(scalar_id), int(capacity)
                if scalar_id in scalar_scopes or not device.startswith("cuda:") or capacity < 1:
                    scalar_errors.append(f"malformed pinned scalar scope: {scalar_id}")
                scalar_scopes[scalar_id] = (capacity, event)
    for scalar_id in scalar_results.keys() - scalar_scopes.keys():
        scalar_errors.append(f"pinned scalar result lacks its call scope: {scalar_id}")
    for events in scalar_runtime.values():
        events.sort(key=lambda event: event["ts"])
    scalar_runtime_starts = {
        thread: [event["ts"] for event in events] for thread, events in scalar_runtime.items()
    }
    scalar_rows = []
    for scalar_id, (capacity, event) in scalar_scopes.items():
        handout = scalar_results.get(scalar_id)
        if handout != capacity:
            scalar_errors.append(
                f"pinned scalar handout differs from input dtype bin: {scalar_id}/{handout}/{capacity}"
            )
            continue
        thread = (event.get("pid"), event.get("tid"))
        starts = scalar_runtime_starts.get(thread, [])
        nested = scalar_runtime[thread][
            bisect_left(starts, event["ts"]) : bisect_right(starts, event["ts"] + event["dur"])
        ]
        required = {"aten::_local_scalar_dense", "cudaMemcpyAsync", "cudaStreamSynchronize"}
        if {item["name"] for item in nested} != required or any(
            item["ts"] + item["dur"] > event["ts"] + event["dur"] for item in nested
        ):
            scalar_errors.append(
                f"pinned scalar lacks contained synchronous copy evidence: {scalar_id}"
            )
            continue
        scalar_rows.append(
            {
                "id": scalar_id,
                "bytes": handout,
                "start_us": event["ts"],
                "end_us": event["ts"] + event["dur"],
                "thread": [event.get("pid"), event.get("tid")],
            }
        )
        # Negative synthetic addresses cannot collide with real allocator pointers.
        # Charging from wrapper entry to exit covers the complete synchronous copy.
        for timestamp, size in ((event["ts"], handout), (event["ts"] + event["dur"], -handout)):
            synthetic = {
                "ts": timestamp,
                "pid": event.get("pid"),
                "tid": event.get("tid"),
                "args": {"Addr": -scalar_id - 1, "Bytes": size, "Device Type": 0, "Device Id": -1},
            }
            timeline.append((timestamp, 1, -scalar_id - 1, "memory", synthetic))
    internal_pinned = _internal_pinned_intervals(
        scalar_runtime, scalar_runtime_starts, pinned_copies, scalar_scopes, scalar_errors
    )
    first_internal = -max(scalar_scopes, default=0) - 2
    for sequence, item in enumerate(internal_pinned):
        address = first_internal - sequence
        for timestamp, size in (
            (item["start_us"], item["bytes"]),
            (item["end_us"], -item["bytes"]),
        ):
            synthetic = {
                "ts": timestamp,
                "pid": item["thread"][0],
                "tid": item["thread"][1],
                "args": {"Addr": address, "Bytes": size, "Device Type": 0, "Device Id": -1},
            }
            timeline.append((timestamp, 1, address, "memory", synthetic))
    if (
        not scope_count
        or not memory_count
        and not scalar_rows
        and not internal_pinned
        and not allow_empty_memory
    ):
        raise ValueError("trace lacks cache scopes or allocator memory events")
    if (
        not memory_count
        and not scalar_rows
        and not internal_pinned
        and any(device.startswith("cuda:") for device in limits)
    ):
        if cuda_history is None:
            scalar_errors.append("zero-allocation CUDA phase requires allocator history")
        elif any(
            event["action"] == "alloc"
            for events in cuda_history["events_by_device"].values()
            for event in events
        ):
            scalar_errors.append("empty profiler memory events disagree with CUDA allocations")
    # event priority and a stable insertion index handle equal timestamps without
    # comparing dictionaries. End is inclusive, conservatively covering ties.
    timeline = sorted(enumerate(timeline), key=lambda item: (*item[1][:3], item[0]))
    active_scopes, live, seen = defaultdict(dict), {}, set()
    owned, errors, baseline_frees, source_markers, ordinary_markers = [], scalar_errors, 0, 0, 0
    cuda_allocations = []
    for _, (timestamp, _, _, kind, payload) in timeline:
        if kind in ("start", "end"):
            category, label, sequence, event = payload
            thread = (event.get("pid"), event.get("tid"))
            if kind == "start":
                active_scopes[thread][sequence] = (event["ts"], -event["dur"], category, label)
            else:
                active_scopes[thread].pop(sequence, None)
            continue
        event = payload
        if kind in ("source", "ordinary"):
            source_markers += kind == "source"
            ordinary_markers += kind == "ordinary"
            prefix = SOURCE_PREFIX if kind == "source" else ORDINARY_PREFIX
            device, address, storage_bytes = event["name"].removeprefix(prefix).split("|")
            storage_bytes = int(storage_bytes)
            key = (device, int(address))
            allocation = live.get(key)
            if allocation is None:
                errors.append(f"{kind} storage marker has no live allocation: {key}")
            elif not 0 < storage_bytes <= allocation["bytes"]:
                errors.append(f"{kind} storage marker size exceeds allocator block: {key}")
            else:
                if kind == "source" and not allocation["tracked"]:
                    owned.append(allocation)
                    allocation["tracked"] = True
                allocation["category"] = "copy_source" if kind == "source" else "ordinary"
                if kind == "source":
                    allocation["source_storage_bytes"] = storage_bytes
            continue
        args = event["args"]
        device, address, size = _device(args), args["Addr"], args["Bytes"]
        key = (device, address)
        if type(size) is not int:
            raise ValueError("allocator event byte delta must be an integer")
        if size > 0:
            if key in live:
                errors.append(f"allocation reuses an address before free: {key}")
                continue
            scopes = active_scopes[(event.get("pid"), event.get("tid"))].values()
            inner = max(scopes, default=(0, 0, "ordinary", "unscoped"))
            allocation = {
                "device": device,
                "address": address,
                "bytes": size,
                "start_us": timestamp,
                "end_us": None,
                "category": inner[2],
                "scope": inner[3],
                "tracked": inner[2] in CACHE_CATEGORIES,
            }
            live[key] = allocation
            if device.startswith("cuda:"):
                cuda_allocations.append(allocation)
            seen.add(key)
            if allocation["category"] in CACHE_CATEGORIES:
                owned.append(allocation)
        elif size < 0:
            allocation = live.pop(key, None)
            if allocation is None:
                if key in seen:
                    errors.append(f"duplicate free or missing allocation event: {key}")
                else:
                    baseline_frees += 1
            else:
                if allocation["bytes"] != -size:
                    errors.append(f"allocator free size mismatch: {key}")
                allocation["end_us"] = timestamp
    owned = [item for item in owned if item["category"] in CACHE_CATEGORIES]
    if cuda_history is not None:
        if "baseTimeNanoseconds" not in trace:
            raise ValueError("CUDA history alignment requires the profiler base timestamp")
        _apply_cuda_history(
            cuda_allocations,
            cuda_history,
            base_us=trace["baseTimeNanoseconds"] / 1000,
            errors=errors,
        )
    elif any(item["device"].startswith("cuda:") for item in owned):
        errors.append("CUDA cache allocations require allocator history through free_completed")
    devices = {}
    for device in sorted(set(limits) | {item["device"] for item in owned}):
        allocations = [item for item in owned if item["device"] == device]
        observed = _peak(allocations)
        active = (
            _peak(allocations, active=True)
            if device.startswith("cuda:") and all("active_start_us" in item for item in allocations)
            else observed
            if device == "cpu"
            else None
        )
        limit = limits.get(device)
        if limit is None:
            errors.append(f"cache allocation has no temporary reservation for {device}")
        devices[device] = {
            "temporary_limit_bytes": limit,
            "observed_cache_peak_bytes": observed["bytes"],
            "observed_allocator_active_peak_bytes": active["bytes"] if active else None,
            "fits_temporary_limit": limit is not None
            and active is not None
            and active["bytes"] <= limit,
            "allocator_live": observed,
            "allocator_active_through_completed_free": active,
            "allocation_count": len(allocations),
            "projected_source_allocator_padding_bytes": sum(
                item["bytes"] - item["source_storage_bytes"]
                for item in allocations
                if "source_storage_bytes" in item
            ),
        }
    return {
        "schema": "echo-cache-allocator-audit-v1",
        "passed_observed_temporary_bound": not errors
        and all(item["fits_temporary_limit"] for item in devices.values()),
        "evidence_errors": errors,
        "devices": devices,
        "cache_scope_count": scope_count,
        "memory_event_count": memory_count,
        "projected_source_markers": source_markers,
        "ordinary_result_markers": ordinary_markers,
        "pinned_scalar_calls": scalar_rows,
        "pinned_scalar_handout_bytes": sum(scalar_results.values()),
        "pinned_internal_calls": internal_pinned,
        "pinned_internal_handout_bytes": sum(item["bytes"] for item in internal_pinned),
        "preexisting_allocation_frees": baseline_frees,
        "coverage": {
            "latency_measurement": False,
            "fixed_pool_storage_included": False,
            "pytorch_allocator_allocations": True,
            "cuda_pending_free_retention": cuda_history is not None,
            "cuda_generation_matching": "complete per-address ordinal pairing including ordinary allocations",
            "cuda_bytes": "allocator block bytes including observed rounding",
            "cpu_bytes": "PyTorch reported pageable allocation bytes; malloc overhead unobserved",
            "source_lifetime": "original allocation through allocator free, not append scope",
            "unobserved": [
                "direct library CUDA/CPU allocations outside PyTorch allocators",
                "CUDA reserved segments and fragmentation",
                "weights, ordinary activations and fixed caches allocated before capture",
                "unannotated cache allocations outside supplied launch scopes",
                "pinned host allocations outside validated scalar/nonzero paths",
            ],
            "profiler_intrusive": "shape/stack capture can perturb lifetimes and latency",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--limits", type=Path, required=True, help='JSON {"cuda:0": bytes, "cpu": bytes}'
    )
    parser.add_argument("--cuda-history", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_trace(
        args.trace, limits=json.loads(args.limits.read_text()), cuda_history=args.cuda_history
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["passed_observed_temporary_bound"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
