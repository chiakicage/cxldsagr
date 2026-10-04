"""Stream-ordered metadata operations for an exclusive DeepSeek ECHO pool lease.

These operations own no tensors or allocator state. Callers reserve all scratch,
serialize a layer pool, and wait for host-write completion before prefetching.
The native helper uses the same claim/transport function as the fused indexer.
"""

import torch

from operators.deepseek_v32.indexer.echo import _check_tensor, _module, _validate_prefetch


def _call(name, device, *args):
    import tvm_ffi

    with torch.cuda.device(device), tvm_ffi.use_torch_stream():
        getattr(_module(), name)(*args)


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


def sparse_selection_publish(
    misses, chosen, host_to_device, device_to_host, priority, free_bitmap, clock, *, timestamp
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
    _call(
        "echo_sparse_selection_publish",
        device,
        misses,
        chosen,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        clock,
        timestamp,
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


def dense_history_classify(
    page_table,
    host_to_device,
    device_to_host,
    priority,
    clock,
    flags,
    miss_count,
    *,
    history,
    timestamp,
):
    """Protect complete initialized history without changing selection counters."""
    device = _resident_maps(page_table, host_to_device, priority, clock, history, timestamp)
    _check_tensor(device_to_host, priority.shape, torch.int64, device, "device_to_host")
    _check_tensor(flags, (priority.numel() - 1,), torch.int32, device, "miss_flags")
    _check_tensor(miss_count, (1,), torch.uint32, device, "miss_count")
    _call(
        "echo_dense_history_classify",
        device,
        page_table,
        host_to_device,
        device_to_host,
        priority,
        clock,
        flags,
        miss_count,
        history,
        timestamp,
    )


def dense_history_reserve(
    prefix,
    victims,
    page_table,
    host_to_device,
    device_to_host,
    priority,
    free_bitmap,
    clock,
    evictions,
    misses,
    chosen,
    *,
    history,
    timestamp,
):
    """Reserve dense tickets before copy; outputs must own private ID storage.

    The target layer cannot be used until its ticket's copy event is joined.
    This ordering differs from synchronous sparse recall's post-copy publication.
    """
    device = _resident_maps(page_table, host_to_device, priority, clock, history, timestamp)
    slots = priority.numel() - 1
    count = misses.numel()
    if not 0 <= count <= history:
        raise ValueError("dense miss count must fit initialized history")
    if count:
        buffers = (prefix, victims, misses, chosen)
        if len({tensor.untyped_storage().data_ptr() for tensor in buffers}) != len(buffers):
            raise ValueError("dense prefix, victims and ticket IDs must use distinct storage")
        for tensor in (misses, chosen):
            if tensor.storage_offset() or tensor.untyped_storage().nbytes() != count * 8:
                raise ValueError("dense ticket IDs must own exactly their declared storage")
    for tensor, shape, dtype, name in (
        (prefix, (slots,), torch.int64, "miss_prefix"),
        (victims, (slots if count else 0,), torch.int64, "sorted_victims"),
        (device_to_host, priority.shape, torch.int64, "device_to_host"),
        (free_bitmap, priority.shape, torch.bool, "free_bitmap"),
        (evictions, (1,), torch.int64, "evictions"),
        (misses, (count,), torch.int64, "ticket_host_ids"),
        (chosen, (count,), torch.int64, "ticket_slots"),
    ):
        _check_tensor(tensor, shape, dtype, device, name)
    _call(
        "echo_dense_history_reserve",
        device,
        prefix,
        victims,
        page_table,
        host_to_device,
        device_to_host,
        priority,
        free_bitmap,
        clock,
        evictions,
        misses,
        chosen,
        history,
        timestamp,
    )
