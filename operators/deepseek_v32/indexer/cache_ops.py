"""Stream-ordered metadata operations for an exclusive DeepSeek ECHO pool lease.

These operations own no tensors or allocator state. Callers reserve all scratch,
serialize a layer pool, and wait for host-write completion before prefetching.
The native helper uses the same claim/transport function as the fused indexer.
"""

from functools import cache

import torch

from operators.deepseek_v32.indexer.echo import (
    _check_tensor,
    _module,
    _PreparedPrefetch,
    _validate_prefetch,
)


def initialize(device):
    """Prepare the combined native recall entry before pool/session allocation."""
    from operators.deepseek_v32.indexer import recall_dispatch

    recall_dispatch.initialize(device)


def _call(name, device, *args):
    import tvm_ffi

    with torch.cuda.device(device), tvm_ffi.use_torch_stream():
        return getattr(_module(), name)(*args)


def _ids(ids, device):
    _check_tensor(ids, (ids.numel(),), torch.int64, device, "global_ids")


def prefetch_ids(global_ids, lease):
    """Prefetch explicit global IDs; duplicates/invalid IDs are ignored.

    Useful for metadata/transport validation independent of score prediction.
    Like fused prefetch it may evict live records, so it is not guaranteed
    recall for a protected consumer selection.
    """
    device = lease["device"].device
    _validate_prefetch(lease, 0, lease.get("history_length", 0), device)
    _ids(global_ids, device)
    if lease.get("_prepared") is not None:
        lease["_prepared"].invalidate()
    lease["counter"].zero_()
    lease["allocation_log"].fill_(2**31 - 1)
    lease["prefetch_stats"].zero_()
    _call(
        "echo_prefetch_ids",
        device,
        global_ids,
        *(
            lease[key]
            for key in (
                "host",
                "device",
                "host_to_device",
                "device_to_host",
                "free_slots",
                "allocation_log",
                "counter",
                "prefetch_stats",
            )
        ),
        lease.get("max_prefetch", min(8192, len(lease["device"]) - 1)),
    )


def finalize_prefetch(priority, free_bitmap, allocation_log, clock):
    """Post-allocate logged slots; always advance the official event clock."""
    device = priority.device
    for tensor, dtype, name in (
        (priority, torch.int64, "priority"),
        (free_bitmap, torch.bool, "free_bitmap"),
        (allocation_log, torch.int64, "allocation_log"),
    ):
        _check_tensor(tensor, (priority.numel(),), dtype, device, name)
    _check_tensor(clock, (1,), torch.int64, device, "clock")
    _call("echo_finalize_prefetch", device, priority, free_bitmap, allocation_log, clock)


def protect(global_ids, host_to_device, priority, clock):
    """Refresh existing selected slots, including empty/all-hit event calls."""
    device = priority.device
    _ids(global_ids, device)
    _check_tensor(host_to_device, (host_to_device.numel(),), torch.int32, device, "host_to_device")
    _check_tensor(priority, (priority.numel(),), torch.int64, device, "priority")
    _check_tensor(clock, (1,), torch.int64, device, "clock")
    _call("echo_protect", device, global_ids, host_to_device, priority, clock)


def empty_event(priority, clock, *, timestamp):
    """Advance an empty FIFO event without materializing a host scalar copy."""
    device = priority.device
    if (
        device.type != "cuda"
        or priority.numel() < 2
        or type(timestamp) is not int
        or not 0 <= timestamp < 2**63 - 1
    ):
        raise ValueError("empty FIFO event requires CUDA priority storage and a valid timestamp")
    _check_tensor(priority, (priority.numel(),), torch.int64, device, "priority")
    _check_tensor(clock, (1,), torch.int64, device, "clock")
    _call("echo_empty_event", device, priority, clock, timestamp)


def release_ids(global_ids, host_to_device, device_to_host, priority, free_bitmap):
    """Release only the supplied global IDs, without changing the event clock."""
    device = priority.device
    _ids(global_ids, device)
    _check_tensor(host_to_device, (host_to_device.numel(),), torch.int32, device, "host_to_device")
    for tensor, dtype, name in (
        (device_to_host, torch.int64, "device_to_host"),
        (priority, torch.int64, "priority"),
        (free_bitmap, torch.bool, "free_bitmap"),
    ):
        _check_tensor(tensor, (priority.numel(),), dtype, device, name)
    _call(
        "echo_release_ids",
        device,
        global_ids,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
    )


def mark_misses(global_ids, host_to_device, output):
    """Write each actual global miss or MISSING into caller-provided scratch."""
    device = host_to_device.device
    _ids(global_ids, device)
    _check_tensor(host_to_device, (host_to_device.numel(),), torch.int32, device, "host_to_device")
    _check_tensor(output, global_ids.shape, torch.int64, device, "output")
    _call("echo_mark_misses", device, global_ids, host_to_device, output)


def _resident_maps(page_table, host_to_device, priority, clock, history, timestamp):
    device = priority.device
    if device.type != "cuda":
        raise ValueError("native resident metadata requires CUDA")
    if type(history) is not int or not 0 <= history <= priority.numel() - 1:
        raise ValueError("resident history must fit the physical pool")
    if type(timestamp) is not int or not 0 <= timestamp < 2**63 - 2:
        raise ValueError("timestamp is outside the native clock domain")
    if page_table.ndim != 1 or page_table.numel() * 64 < history:
        raise ValueError("page table does not cover the resident history")
    for tensor, dtype, name in (
        (page_table, torch.int32, "page_table"),
        (host_to_device, torch.int32, "host_to_device"),
        (priority, torch.int64, "priority"),
    ):
        _check_tensor(tensor, (tensor.numel(),), dtype, device, name)
    _check_tensor(clock, (1,), torch.int64, device, "clock")
    return device


def resident_selection(
    indices,
    physical,
    page_table,
    host_to_device,
    priority,
    clock,
    union_bitmap,
    union_count,
    selection_totals,
    *,
    written,
    history,
    transient,
    candidate_slots,
    timestamp,
):
    """Map certified exact-top-k IDs and count their exact union on the GPU."""
    device = _resident_maps(page_table, host_to_device, priority, clock, history, timestamp)
    slots = priority.numel() - 1
    if (
        type(written) is not int
        or type(candidate_slots) is not int
        or candidate_slots < 0
        or not history <= written <= history + (candidate_slots if transient else 0)
        or indices.ndim != 2
    ):
        raise ValueError("invalid trusted-selection geometry")
    _check_tensor(indices, indices.shape, torch.int32, device, "indices")
    _check_tensor(physical, indices.shape, torch.int32, device, "physical")
    _check_tensor(
        union_bitmap,
        ((slots + 1 + candidate_slots + 31) // 32,),
        torch.uint32,
        device,
        "union_bitmap",
    )
    _check_tensor(union_count, (1,), torch.uint32, device, "union_count")
    _check_tensor(selection_totals, (3,), torch.int64, device, "selection_totals")
    _call(
        "echo_resident_selection",
        device,
        indices,
        physical,
        page_table,
        host_to_device,
        priority,
        clock,
        union_bitmap,
        union_count,
        selection_totals,
        written,
        history,
        bool(transient),
        candidate_slots,
        timestamp,
    )


def planned_append(
    source,
    records,
    page_table,
    chosen,
    host_to_device,
    device_to_host,
    priority,
    free_bitmap,
    clock,
    evictions,
    *,
    start,
    timestamp,
):
    """Copy opaque records into unique planned slots and publish metadata."""
    if source.ndim != 2 or records.ndim != 2 or source.shape[1] != records.shape[1]:
        raise ValueError("planned append requires matching two-dimensional records")
    if type(start) is not int or start < 0 or not len(source):
        raise ValueError("planned append requires a nonempty logical range")
    device = _resident_maps(
        page_table, host_to_device, priority, clock, start + len(source), timestamp
    )
    if records.shape[0] < priority.numel():
        raise ValueError("record storage does not cover the physical pool")
    _check_tensor(source, source.shape, records.dtype, device, "source")
    _check_tensor(records, records.shape, records.dtype, device, "records")
    _check_tensor(chosen, (len(source),), torch.int64, device, "chosen")
    _check_tensor(device_to_host, priority.shape, torch.int64, device, "device_to_host")
    _check_tensor(free_bitmap, priority.shape, torch.bool, device, "free_bitmap")
    _check_tensor(evictions, (1,), torch.int64, device, "evictions")
    _call(
        "echo_planned_append",
        device,
        source,
        records,
        page_table,
        chosen,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        clock,
        evictions,
        start,
        source.shape[1] * source.element_size(),
        timestamp,
    )


def protect_resident_history(page_table, host_to_device, priority, clock, *, history, timestamp):
    """Refresh a certified full history and retain its empty allocation event."""
    device = _resident_maps(page_table, host_to_device, priority, clock, history, timestamp)
    _call(
        "echo_protect_resident_history",
        device,
        page_table,
        host_to_device,
        priority,
        clock,
        history,
        timestamp,
    )


def _sparse_selection_geometry(
    indices, page_table, host_to_device, priority, clock, written, history, candidate_slots
):
    device = _resident_maps(page_table, host_to_device, priority, clock, history, 0)
    if (
        type(written) is not int
        or type(candidate_slots) is not int
        or candidate_slots < 0
        or not history <= written <= history + candidate_slots
        or indices.ndim != 2
    ):
        raise ValueError("invalid exact-recall selection geometry")
    _check_tensor(indices, indices.shape, torch.int32, device, "indices")
    return device


def sparse_selection_classify(
    indices,
    page_table,
    host_to_device,
    device_to_host,
    priority,
    clock,
    union_bitmap,
    union_count,
    miss_count,
    flags,
    selection_totals,
    *,
    written,
    history,
    candidate_slots,
    timestamp,
):
    """Build a logical union, protect exact hits and count actual history misses."""
    device = _sparse_selection_geometry(
        indices, page_table, host_to_device, priority, clock, written, history, candidate_slots
    )
    _resident_maps(page_table, host_to_device, priority, clock, history, timestamp)
    slots = priority.numel() - 1
    _check_tensor(device_to_host, priority.shape, torch.int64, device, "device_to_host")
    _check_tensor(
        union_bitmap,
        ((slots + 1 + candidate_slots + 31) // 32,),
        torch.uint32,
        device,
        "union_bitmap",
    )
    _check_tensor(union_count, (1,), torch.uint32, device, "union_count")
    _check_tensor(miss_count, (1,), torch.uint32, device, "miss_count")
    _check_tensor(flags, (slots,), torch.int32, device, "miss_flags")
    _check_tensor(selection_totals, (3,), torch.int64, device, "selection_totals")
    _call(
        "echo_sparse_selection_classify",
        device,
        indices,
        page_table,
        host_to_device,
        device_to_host,
        priority,
        clock,
        union_bitmap,
        union_count,
        miss_count,
        flags,
        selection_totals,
        history,
        written,
        timestamp,
    )


def _compact_geometry(
    prefix,
    page_table,
    host_to_device,
    device_to_host,
    priority,
    free_bitmap,
    evictions,
    misses,
    chosen,
    *,
    history,
    timestamp,
):
    """Compact logical-order misses and tombstone FIFO victims before copying."""
    device = priority.device
    slots = priority.numel() - 1
    if type(history) is not int or not 0 <= history <= slots:
        raise ValueError("recall history must fit the physical pool")
    if type(timestamp) is not int or not 0 <= timestamp < 2**63 - 2:
        raise ValueError("timestamp is outside the native clock domain")
    if not 0 < misses.numel() <= slots or page_table.numel() * 64 < history:
        raise ValueError("invalid compacted recall geometry")
    if len({tensor.untyped_storage().data_ptr() for tensor in (prefix, misses, chosen)}) != 3:
        raise ValueError("miss prefix, compact IDs and chosen slots must use distinct storage")
    for tensor, shape, dtype, name in (
        (prefix, (slots,), torch.int64, "miss_prefix"),
        (page_table, (page_table.numel(),), torch.int32, "page_table"),
        (host_to_device, (host_to_device.numel(),), torch.int32, "host_to_device"),
        (device_to_host, (slots + 1,), torch.int64, "device_to_host"),
        (priority, (slots + 1,), torch.int64, "priority"),
        (free_bitmap, (slots + 1,), torch.bool, "free_bitmap"),
        (evictions, (1,), torch.int64, "evictions"),
        (misses, (misses.numel(),), torch.int64, "misses"),
        (chosen, (misses.numel(),), torch.int64, "chosen"),
    ):
        _check_tensor(tensor, shape, dtype, device, name)
    return device


def sparse_selection_compact(
    prefix,
    page_table,
    host_to_device,
    device_to_host,
    priority,
    free_bitmap,
    evictions,
    misses,
    chosen,
    *,
    history,
    timestamp,
):
    """Compact logical-order misses and tombstone FIFO victims before copying."""
    device = _compact_geometry(
        prefix,
        page_table,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        evictions,
        misses,
        chosen,
        history=history,
        timestamp=timestamp,
    )
    _call(
        "echo_sparse_selection_compact",
        device,
        prefix,
        page_table,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        evictions,
        misses,
        chosen,
        history,
        timestamp,
    )


@cache
def _selection_workspace_bytes(slots, device):
    """CUB's storage query uses only geometry/device metadata, with no GPU read."""
    return int(_call("echo_sparse_selection_workspace", device, slots, device.index))


def supports_free_q1_prepare(*, rows, columns, query_start, history_length, limit):
    """Share the actual official Q1 predicate with cache preparation dispatch."""
    from operators.deepseek_v32.indexer.echo import _uses_official_q1

    return _uses_official_q1(
        {"history_length": history_length, "max_prefetch": limit},
        rows,
        columns,
        query_start,
    )


def prepare_prefetch_free(
    priority,
    free_bitmap,
    device_to_host,
    free_slots,
    allocation_log,
    counter,
    prefetch_stats,
    keys,
    *,
    timestamp,
):
    """Prepare exactly 64 existing free slots; the caller proves their capacity."""
    from operators.deepseek_v32.indexer.echo import _BoundedPreparedPrefetch

    device, slots = priority.device, priority.numel() - 1
    if (
        device.type != "cuda"
        or torch.cuda.get_device_capability(device) != (9, 0)
        or not 64 <= slots < 2**31 - 1
        or type(timestamp) is not int
        or not 0 <= timestamp < 2**31 - 2
    ):
        raise ValueError("bounded prefetch preparation requires SM90 pool geometry")
    for tensor, shape, dtype, name in (
        (priority, (slots + 1,), torch.int64, "priority"),
        (free_bitmap, (slots + 1,), torch.bool, "free bitmap"),
        (device_to_host, (slots + 1,), torch.int64, "reverse map"),
        (free_slots, (slots,), torch.int32, "free slots"),
        (allocation_log, (slots + 1,), torch.int64, "journal"),
        (counter, (1,), torch.uint32, "counter"),
        (prefetch_stats, (3,), torch.int64, "statistics"),
        (keys, (slots,), torch.int64, "mask scratch"),
    ):
        _check_tensor(tensor, shape, dtype, device, name)
    tensors = (
        priority,
        free_bitmap,
        device_to_host,
        free_slots,
        allocation_log,
        counter,
        prefetch_stats,
        keys,
    )
    if len({tensor.untyped_storage().data_ptr() for tensor in tensors}) != len(tensors):
        raise ValueError("bounded preparation inputs and scratch cannot alias")
    _call("echo_prepare_prefetch_free", device, *tensors, timestamp)
    return _BoundedPreparedPrefetch(1, free_slots, allocation_log, counter, prefetch_stats, keys)


def prepare_prefetch(
    priority,
    free_slots,
    allocation_log,
    counter,
    prefetch_stats,
    keys,
    *,
    timestamp,
    host_capacity,
    query_count=0,
):
    """Prepare stable FIFO slots and clear the next prefetch journal together.

    This changes no record, map, priority or event clock. The allocation log
    holds sorted keys only between kernels on the exclusive caller stream.
    With a positive query_count, the final kernel reuses the now-dead sort
    input for single-request metadata and returns its one-use internal token.
    """
    device, slots = priority.device, priority.numel() - 1
    if (
        device.type != "cuda"
        or not 0 < slots < 2**31 - 1
        or type(timestamp) is not int
        or not 0 <= timestamp < 2**31 - 2
        or type(host_capacity) is not int
        or not 0 < host_capacity < 2**31 - 1
        or type(query_count) is not int
        or not 0 <= query_count <= slots
    ):
        raise ValueError("native prefetch preparation requires bounded CUDA pool geometry")
    if keys.untyped_storage().data_ptr() == allocation_log.untyped_storage().data_ptr():
        raise ValueError("prefetch sort inputs and outputs require distinct storage")
    if query_count and any(
        keys.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        for tensor in (free_slots, counter, prefetch_stats)
    ):
        raise ValueError("prepared request metadata cannot alias live prefetch tensors")
    for tensor, shape, dtype, name in (
        (priority, (slots + 1,), torch.int64, "priority"),
        (free_slots, (slots,), torch.int32, "free_slots"),
        (allocation_log, (slots + 1,), torch.int64, "allocation_log"),
        (counter, (1,), torch.uint32, "counter"),
        (prefetch_stats, (3,), torch.int64, "prefetch_stats"),
        (keys, (slots,), torch.int64, "keys"),
    ):
        _check_tensor(tensor, shape, dtype, device, name)
    workspace_bytes = _selection_workspace_bytes(slots, device)
    if workspace_bytes > 64 * (slots + 1 + host_capacity):
        raise RuntimeError("native prefetch workspace exceeds the pool execution reservation")
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    _call(
        "echo_prepare_prefetch",
        device,
        priority,
        free_slots,
        allocation_log,
        counter,
        prefetch_stats,
        keys,
        workspace,
        timestamp,
        query_count,
    )
    if query_count:
        return _PreparedPrefetch(
            query_count, free_slots, allocation_log, counter, prefetch_stats, keys
        )
    return None


def sparse_append(
    source,
    records,
    page_table,
    host_to_device,
    device_to_host,
    priority,
    free_bitmap,
    clock,
    evictions,
    keys,
    sorted_keys,
    *,
    start,
    timestamp,
):
    """Append new logical rows using stable FIFO without host-visible counts.

    The new logical range is unmapped. The caller's exclusive lease permits
    reusing its two P-entry scratch arrays after any prefetch has finalized.
    Logical length may exceed P; only this append batch must fit the pool.
    """
    if source.ndim != 2 or records.ndim != 2 or source.shape[1] != records.shape[1]:
        raise ValueError("sparse append requires matching two-dimensional records")
    slots = priority.numel() - 1
    if (
        type(start) is not int
        or start < 0
        or not 0 < len(source) <= slots
        or start + len(source) > page_table.numel() * 64
        or start + len(source) >= 2**31 - 1
    ):
        raise ValueError("sparse append range exceeds its page table or bounded slot capacity")
    device = _resident_maps(page_table, host_to_device, priority, clock, 0, timestamp)
    if timestamp >= 2**31 - 2 or records.shape[0] < priority.numel():
        raise ValueError("sparse append requires a normalized clock and complete record pool")
    if keys.untyped_storage().data_ptr() == sorted_keys.untyped_storage().data_ptr():
        raise ValueError("sparse append sort inputs and outputs require distinct storage")
    for tensor, shape, dtype, name in (
        (source, source.shape, records.dtype, "source"),
        (records, records.shape, records.dtype, "records"),
        (device_to_host, priority.shape, torch.int64, "device_to_host"),
        (free_bitmap, priority.shape, torch.bool, "free_bitmap"),
        (evictions, (1,), torch.int64, "evictions"),
        (keys, (slots,), torch.int64, "keys"),
        (sorted_keys, (slots,), torch.int64, "sorted_keys"),
    ):
        _check_tensor(tensor, shape, dtype, device, name)
    workspace_bytes = _selection_workspace_bytes(slots, device)
    if workspace_bytes > 64 * (slots + 1 + host_to_device.numel()):
        raise RuntimeError("native append workspace exceeds the pool execution reservation")
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    _call(
        "echo_sparse_append",
        device,
        source,
        records,
        page_table,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        clock,
        evictions,
        keys,
        sorted_keys,
        workspace,
        start,
        source.shape[1] * source.element_size(),
        timestamp,
    )


def sparse_append_free(
    source,
    records,
    page_table,
    host_to_device,
    device_to_host,
    priority,
    free_bitmap,
    clock,
    evictions,
    keys,
    sorted_keys,
    *,
    start,
    timestamp,
):
    """Append a new bounded suffix into GPU-validated free slots.

    Caller holds the sole active session's operation lease, has finalized
    prefetch, and appends a persistent unmapped range ending at or below P.
    This is a capacity proof only; no CPU residency certificate is created.
    The existing keys scratch holds one free-bit mask per 32 pool slots.
    """
    if source.ndim != 2 or records.ndim != 2 or source.shape[1] != records.shape[1]:
        raise ValueError("sparse append requires matching two-dimensional records")
    slots = priority.numel() - 1
    if (
        type(start) is not int
        or start < 0
        or not 0 < len(source) <= slots
        or start + len(source) > slots
        or start + len(source) > page_table.numel() * 64
        or start + len(source) >= 2**31 - 1
    ):
        raise ValueError("sparse append range exceeds its page table or bounded slot capacity")
    device = _resident_maps(page_table, host_to_device, priority, clock, 0, timestamp)
    if timestamp >= 2**31 - 2 or records.shape[0] < priority.numel():
        raise ValueError("sparse append requires a normalized clock and complete record pool")
    if keys.untyped_storage().data_ptr() == sorted_keys.untyped_storage().data_ptr():
        raise ValueError("sparse append sort inputs and outputs require distinct storage")
    for tensor, shape, dtype, name in (
        (source, source.shape, records.dtype, "source"),
        (records, records.shape, records.dtype, "records"),
        (device_to_host, priority.shape, torch.int64, "device_to_host"),
        (free_bitmap, priority.shape, torch.bool, "free_bitmap"),
        (evictions, (1,), torch.int64, "evictions"),
        (keys, (slots,), torch.int64, "keys"),
        (sorted_keys, (slots,), torch.int64, "sorted_keys"),
    ):
        _check_tensor(tensor, shape, dtype, device, name)
    _call(
        "echo_sparse_append_free",
        device,
        source,
        records,
        page_table,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        clock,
        evictions,
        keys,
        sorted_keys,
        start,
        source.shape[1] * source.element_size(),
        timestamp,
    )


def _prepare_sparse_selection_allocate(
    flags,
    prefix,
    page_table,
    host_to_device,
    device_to_host,
    priority,
    free_bitmap,
    evictions,
    misses,
    chosen,
    *,
    history,
    timestamp,
):
    """Validate either supported native allocation path and reserve its scratch.

    Scratch is transient and fits the pool's existing 64*(P+1+NH) execution
    reservation, including the caller's chosen[P] int64 output. Both paths
    preserve the caller-owned metadata ABI and event boundaries.
    """
    device = _compact_geometry(
        prefix,
        page_table,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        evictions,
        misses,
        chosen,
        history=history,
        timestamp=timestamp,
    )
    slots = priority.numel() - 1
    if timestamp >= 2**31 - 2 or misses.numel() != slots:
        raise ValueError("native allocation requires bounded slots and a normalized clock")
    _check_tensor(flags, (slots,), torch.int32, device, "miss_flags")
    if flags.untyped_storage().data_ptr() in {
        tensor.untyped_storage().data_ptr() for tensor in (prefix, misses, chosen)
    }:
        raise ValueError("miss flags must use distinct storage")
    workspace_bytes = _selection_workspace_bytes(slots, device)
    reserved_bytes = 64 * (slots + 1 + host_to_device.numel())
    if workspace_bytes + chosen.numel() * chosen.element_size() > reserved_bytes:
        raise RuntimeError("native recall workspace exceeds the pool execution reservation")
    return device, torch.empty(workspace_bytes, dtype=torch.uint8, device=device)


def _sparse_selection_allocate(
    operation,
    flags,
    prefix,
    page_table,
    host_to_device,
    device_to_host,
    priority,
    free_bitmap,
    evictions,
    misses,
    chosen,
    *,
    history,
    timestamp,
):
    device, workspace = _prepare_sparse_selection_allocate(
        flags,
        prefix,
        page_table,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        evictions,
        misses,
        chosen,
        history=history,
        timestamp=timestamp,
    )
    _call(
        operation,
        device,
        flags,
        prefix,
        page_table,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        evictions,
        misses,
        chosen,
        workspace,
        history,
        timestamp,
    )


def sparse_selection_allocate(
    flags,
    prefix,
    page_table,
    host_to_device,
    device_to_host,
    priority,
    free_bitmap,
    evictions,
    misses,
    chosen,
    *,
    history,
    timestamp,
):
    """Allocate exact misses using FIFO sorting across all pool sessions."""
    return _sparse_selection_allocate(
        "echo_sparse_selection_allocate",
        flags,
        prefix,
        page_table,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        evictions,
        misses,
        chosen,
        history=history,
        timestamp=timestamp,
    )


def sparse_selection_allocate_free(
    flags,
    prefix,
    page_table,
    host_to_device,
    device_to_host,
    priority,
    free_bitmap,
    evictions,
    misses,
    chosen,
    *,
    history,
    timestamp,
):
    """Allocate only free slots under the caller's sole-session H<=P proof.

    The GPU verifies enough free slots before compacting misses and slots.
    A violated proof fails; this entry never retries through another allocator.
    """
    return _sparse_selection_allocate(
        "echo_sparse_selection_allocate_free",
        flags,
        prefix,
        page_table,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        evictions,
        misses,
        chosen,
        history=history,
        timestamp=timestamp,
    )


def sparse_selection_complete(
    flags,
    prefix,
    page_table,
    host_to_device,
    device_to_host,
    priority,
    free_bitmap,
    evictions,
    misses,
    chosen,
    host,
    records,
    valid_count,
    clock,
    recalled_totals,
    *,
    history,
    timestamp,
    free_only,
):
    """Allocate, gather and publish through prepared original native Functions.

    Allocation checks remain shared with the separate entries. The host bridge
    validates copy metadata after allocation and publication metadata after
    gather; it propagates every original native exception without retrying.
    """
    import tvm_ffi

    from operators.deepseek_v32.indexer import recall_dispatch

    if type(free_only) is not bool:
        raise TypeError("free_only must be a boolean supported by the caller's pool proof")
    device, workspace = _prepare_sparse_selection_allocate(
        flags,
        prefix,
        page_table,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        evictions,
        misses,
        chosen,
        history=history,
        timestamp=timestamp,
    )
    function = recall_dispatch.prepared_call(device, free_only)
    with torch.cuda.device(device), tvm_ffi.use_torch_stream():
        function(
            flags,
            prefix,
            page_table,
            host_to_device,
            device_to_host,
            priority,
            free_bitmap,
            evictions,
            misses,
            chosen,
            workspace,
            history,
            timestamp,
            host,
            records,
            valid_count,
            clock,
            recalled_totals,
        )


def sparse_selection_publish(
    misses,
    chosen,
    host_to_device,
    device_to_host,
    priority,
    free_bitmap,
    clock,
    *,
    timestamp,
    valid_count=None,
    recalled_totals=None,
):
    """Publish copied records and advance allocation, including a zero-miss event."""
    device = priority.device
    if type(timestamp) is not int or not 0 <= timestamp < 2**63 - 2:
        raise ValueError("timestamp is outside the native clock domain")
    _check_tensor(clock, (1,), torch.int64, device, "clock")
    _ids(misses, device)
    _check_tensor(chosen, misses.shape, torch.int64, device, "chosen")
    _check_tensor(host_to_device, (host_to_device.numel(),), torch.int32, device, "host_to_device")
    for tensor, dtype, name in (
        (device_to_host, torch.int64, "device_to_host"),
        (priority, torch.int64, "priority"),
        (free_bitmap, torch.bool, "free_bitmap"),
    ):
        _check_tensor(tensor, priority.shape, dtype, device, name)
    bounded = valid_count is not None
    if bounded != (recalled_totals is not None):
        raise ValueError("bounded publication requires a count and recalled total together")
    if bounded:
        _check_tensor(valid_count, (1,), torch.uint32, device, "valid_count")
        _check_tensor(recalled_totals, (1,), torch.int64, device, "recalled_totals")
        if misses.numel() != priority.numel() - 1:
            raise ValueError("bounded publication IDs must cover every physical slot")
    _call(
        "echo_sparse_selection_publish_bounded" if bounded else "echo_sparse_selection_publish",
        device,
        misses,
        chosen,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        clock,
        timestamp,
        *((valid_count, recalled_totals) if bounded else ()),
    )


def sparse_selection_map(
    indices,
    physical,
    page_table,
    host_to_device,
    priority,
    clock,
    *,
    history,
    written,
    candidate_slots,
):
    """Map the original selection after both FIFO events have committed."""
    device = _sparse_selection_geometry(
        indices, page_table, host_to_device, priority, clock, written, history, candidate_slots
    )
    _check_tensor(physical, indices.shape, torch.int32, device, "physical")
    _call(
        "echo_sparse_selection_map",
        device,
        indices,
        physical,
        page_table,
        host_to_device,
        priority,
        history,
        written,
    )


def _dense_history_maps(
    page_table,
    host_to_device,
    device_to_host,
    priority,
    free_bitmap,
    clock,
    history,
    host_start,
    timestamp,
):
    device = _resident_maps(page_table, host_to_device, priority, clock, history, timestamp)
    if (
        type(host_start) is not int
        or host_start < 0
        or host_start > host_to_device.numel()
        or history > host_to_device.numel() - host_start
    ):
        raise ValueError("dense host span exceeds its arena")
    _check_tensor(device_to_host, priority.shape, torch.int64, device, "device_to_host")
    _check_tensor(free_bitmap, priority.shape, torch.bool, device, "free_bitmap")
    return device


def dense_history_clear(
    page_table,
    host_to_device,
    device_to_host,
    priority,
    free_bitmap,
    clock,
    evictions,
    *,
    history,
    host_start,
    timestamp,
):
    """Clear displaced and relocated records before a contiguous history DMA.

    Only slots 1..H and old copies of the incoming host span are invalidated.
    This operation must finish before copying; publication is a separate call
    after the copy, preventing races between old-map clears and new-map writes.
    """
    device = _dense_history_maps(
        page_table,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        clock,
        history,
        host_start,
        timestamp,
    )
    _check_tensor(evictions, (1,), torch.int64, device, "evictions")
    _call(
        "echo_dense_history_clear",
        device,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        evictions,
        history,
        host_start,
    )


def dense_history_publish(
    page_table,
    host_to_device,
    device_to_host,
    priority,
    free_bitmap,
    clock,
    *,
    history,
    host_start,
    timestamp,
):
    """Publish logical i at slot i+1 after its DMA on the same CUDA stream."""
    device = _dense_history_maps(
        page_table,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        clock,
        history,
        host_start,
        timestamp,
    )
    _call(
        "echo_dense_history_publish",
        device,
        page_table,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        clock,
        history,
        host_start,
        timestamp,
    )
