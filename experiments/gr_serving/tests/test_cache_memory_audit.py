from types import SimpleNamespace

import pytest
import torch

from experiments.gr_serving.src.cache_memory_audit import (
    ORDINARY_PREFIX,
    SCALAR_PREFIX,
    SCALAR_RESULT_PREFIX,
    SCOPE_PREFIX,
    SOURCE_PREFIX,
    CacheMemoryAudit,
    WorkspaceAllowance,
    audit_trace,
    host_allocator_counters,
    storage_inventory,
)


def scope(name, start, end, *, category="cache_helpers", tid=1, gpu=False):
    return {
        "name": f"{SCOPE_PREFIX}{category}|{name}",
        "cat": "gpu_user_annotation" if gpu else "user_annotation",
        "ph": "X",
        "ts": start,
        "dur": end - start,
        "pid": 1,
        "tid": tid,
    }


def memory(address, size, timestamp, *, tid=1, cuda=False):
    return {
        "name": "[memory]",
        "ph": "i",
        "ts": timestamp,
        "pid": 1,
        "tid": tid,
        "args": {
            "Addr": address,
            "Bytes": size,
            "Device Type": int(cuda),
            "Device Id": 0 if cuda else -1,
        },
    }


def source(address, size, timestamp, *, cuda=False):
    return {
        "name": f"{SOURCE_PREFIX}{'cuda:0' if cuda else 'cpu'}|{address}|{size}",
        "cat": "user_annotation",
        "ph": "X",
        "ts": timestamp,
        "dur": 0.1,
        "pid": 1,
        "tid": 1,
    }


def trace(*events):
    return {"traceEvents": list(events), "baseTimeNanoseconds": 0}


def test_concurrent_lifetimes_outlive_scopes_and_budget_checks_real_peak():
    events = trace(
        scope("indexer", 0, 3, category="indexer"),
        memory(1, 100, 1),
        scope("offload_exact_recall", 4, 7),
        memory(2, 200, 5),
        memory(1, -100, 8),  # Outside both allocating scopes.
        memory(2, -200, 9),
    )
    result = audit_trace(events, limits={"cpu": 299})
    assert not result["passed_observed_temporary_bound"]
    assert result["evidence_errors"] == []
    device = result["devices"]["cpu"]
    assert device["observed_cache_peak_bytes"] == 300
    assert device["allocator_live"]["bytes_live_at_trace_end"] == 0
    assert device["allocator_live"]["categories_at_peak_bytes"] == {
        "indexer": 100,
        "cache_helpers": 200,
    }
    assert audit_trace(events, limits={"cpu": 300})["passed_observed_temporary_bound"]


def test_source_is_charged_before_marker_but_only_in_matching_pointer_generation():
    events = trace(
        scope("indexer", 0, 3, category="indexer"),
        memory(1, 100, 1),
        scope("projection", 3, 7, category="ordinary"),
        memory(2, 200, 4),
        memory(1, -100, 5),
        source(2, 180, 6),
        memory(2, -200, 8),
        memory(2, 999, 9),  # Later ordinary reuse must not inherit source ownership.
        memory(2, -999, 10),
    )
    result = audit_trace(events, limits={"cpu": 300})
    assert result["passed_observed_temporary_bound"]
    device = result["devices"]["cpu"]
    assert device["observed_cache_peak_bytes"] == 300  # At t=4, before source marker.
    assert device["projected_source_allocator_padding_bytes"] == 20
    assert device["allocation_count"] == 2


def test_thread_and_nested_scope_ownership_ignores_gpu_intervals():
    result = audit_trace(
        trace(
            scope("root", 0, 20),
            scope("projection", 2, 8, category="ordinary"),
            memory(1, 9000, 3),
            memory(2, 8000, 4, tid=2),  # A different host launch thread.
            scope("delayed_gpu_indexer", 8, 19, category="indexer", gpu=True),
            memory(3, 100, 9),
            memory(1, -9000, 11),
            memory(2, -8000, 12, tid=2),
            memory(3, -100, 13),
        ),
        limits={"cpu": 100},
    )
    assert result["passed_observed_temporary_bound"]
    assert result["devices"]["cpu"]["allocation_count"] == 1
    assert result["devices"]["cpu"]["observed_cache_peak_bytes"] == 100


def test_returned_hidden_storage_is_retroactively_excluded_from_cache_peak():
    marker = source(2, 5000, 5)
    marker["name"] = marker["name"].replace(SOURCE_PREFIX, ORDINARY_PREFIX)
    result = audit_trace(
        trace(
            scope("root", 0, 10),
            memory(1, 100, 1),
            memory(2, 5000, 2),
            memory(1, -100, 3),
            marker,
            memory(2, -5000, 6),
        ),
        limits={"cpu": 100},
    )
    assert result["passed_observed_temporary_bound"]
    assert result["devices"]["cpu"]["observed_cache_peak_bytes"] == 100
    assert result["ordinary_result_markers"] == 1


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ([memory(1, 100, 1), memory(1, 100, 2)], "before free"),
        ([memory(1, 100, 1), memory(1, -90, 2)], "size mismatch"),
        ([memory(1, 100, 1), memory(1, -100, 2), memory(1, -100, 3)], "duplicate free"),
        ([source(1, 100, 1), memory(2, 100, 2)], "no live allocation"),
        ([memory(1, 100, 1), source(1, 101, 2)], "exceeds allocator block"),
    ],
)
def test_broken_allocation_evidence_fails_closed(events, message):
    result = audit_trace(trace(scope("cache_write", 0, 10), *events), limits={"cpu": 1000})
    assert not result["passed_observed_temporary_bound"]
    assert any(message in error for error in result["evidence_errors"])


def test_frees_of_preexisting_fixed_storage_are_not_negative_scratch():
    result = audit_trace(
        trace(scope("cache_write", 0, 10), memory(1, -4096, 1), memory(2, 100, 2)),
        limits={"cpu": 100},
    )
    assert result["passed_observed_temporary_bound"]
    assert result["preexisting_allocation_frees"] == 1
    assert result["devices"]["cpu"]["allocator_live"]["bytes_live_at_trace_end"] == 100


def history(*events, truncated=False):
    return {"events_by_device": {"cuda:0": list(events)}, "possibly_truncated": truncated}


def action(kind, address, time, *, size=1):
    return {"action": kind, "addr": address, "time_us": time, "size": size}


def test_cuda_pending_stream_free_counts_until_completed_and_includes_rounding():
    events = trace(
        scope("cache_write", 0, 20),
        memory(1, 512, 1.5, cuda=True),
        memory(1, -512, 3.5, cuda=True),
        memory(2, 512, 5.5, cuda=True),
        memory(2, -512, 8.5, cuda=True),
    )
    device_history = history(
        action("alloc", 1, 1),
        action("free_requested", 1, 3),
        action("alloc", 2, 5),
        action("free_completed", 1, 6),
        action("free_requested", 2, 8),
        action("free_completed", 2, 8),
    )
    result = audit_trace(events, limits={"cuda:0": 1000}, cuda_history=device_history)
    assert not result["passed_observed_temporary_bound"]
    assert result["evidence_errors"] == []
    device = result["devices"]["cuda:0"]
    assert device["observed_cache_peak_bytes"] == 512
    assert device["observed_allocator_active_peak_bytes"] == 1024
    assert audit_trace(events, limits={"cuda:0": 1024}, cuda_history=device_history)[
        "passed_observed_temporary_bound"
    ]


@pytest.mark.parametrize("missing", [True, False])
def test_cuda_requires_complete_matching_allocator_history(missing):
    events = trace(scope("cache_write", 0, 20), memory(1, 512, 1.5, cuda=True))
    result = audit_trace(
        events,
        limits={"cuda:0": 1024},
        cuda_history=None if missing else history(action("alloc", 2, 1)),
    )
    assert not result["passed_observed_temporary_bound"]
    assert result["evidence_errors"]


def test_cuda_history_address_generations_and_truncation():
    events = trace(
        scope("cache_write", 0, 20),
        memory(1, 512, 1.5, cuda=True),
        memory(1, -512, 3.5, cuda=True),
        memory(1, 1024, 5.5, cuda=True),
    )
    device_history = history(
        action("alloc", 1, 1),
        action("free_requested", 1, 3),
        action("free_completed", 1, 3),
        action("alloc", 1, 5),
    )
    result = audit_trace(events, limits={"cuda:0": 1024}, cuda_history=device_history)
    assert result["passed_observed_temporary_bound"]
    device_history["possibly_truncated"] = True
    result = audit_trace(events, limits={"cuda:0": 1024}, cuda_history=device_history)
    assert not result["passed_observed_temporary_bound"]
    assert "CUDA allocation history may be truncated" in result["evidence_errors"]


def test_cuda_integer_history_timestamp_can_round_past_profiler_allocation():
    events = trace(scope("cache_write", 0, 20), memory(1, 512, 4.97, cuda=True))
    result = audit_trace(
        events, limits={"cuda:0": 512}, cuda_history=history(action("alloc", 1, 5))
    )
    assert result["passed_observed_temporary_bound"]


def test_cuda_generation_join_includes_ordinary_allocations_despite_clock_drift():
    events = trace(
        scope("model", 0, 10, category="ordinary"),
        memory(1, 512, 1, cuda=True),
        memory(1, -512, 3, cuda=True),
        scope("cache_write", 10, 30),
        memory(1, 1024, 20, cuda=True),
        memory(1, -1024, 22, cuda=True),
    )
    device_history = history(
        action("alloc", 1, 101, size=8),
        action("free_requested", 1, 103),
        action("free_completed", 1, 104),
        action("alloc", 1, 110, size=700),
        action("free_requested", 1, 111),
        action("free_completed", 1, 121),
    )
    result = audit_trace(events, limits={"cuda:0": 1024}, cuda_history=device_history)
    assert result["passed_observed_temporary_bound"]
    assert result["evidence_errors"] == []
    active = result["devices"]["cuda:0"]["allocator_active_through_completed_free"]
    assert active["bytes"] == 1024
    assert active["timestamp_us"] == 110


@pytest.mark.parametrize(
    "defect",
    [
        "missing_generation",
        "extra_generation",
        "missing_requested_free",
        "reuse_before_completed_free",
        "completed_before_requested",
        "backwards_generation_time",
        "oversized_request",
        "missing_size",
        "duplicate_completed_free",
        "unclosed_profiler_lifetime",
    ],
)
def test_cuda_generation_join_rejects_incomplete_or_inconsistent_evidence(defect):
    events = trace(
        scope("cache_write", 0, 30),
        memory(1, 512, 1, cuda=True),
        memory(1, -512, 5, cuda=True),
        memory(1, 1024, 10, cuda=True),
        memory(1, -1024, 15, cuda=True),
    )
    records = [
        action("alloc", 1, 1, size=400),
        action("free_requested", 1, 4),
        action("free_completed", 1, 6),
        action("alloc", 1, 9, size=900),
        action("free_requested", 1, 14),
        action("free_completed", 1, 16),
    ]
    if defect == "missing_generation":
        records.pop(0)
    elif defect == "extra_generation":
        records.append(action("alloc", 2, 20))
    elif defect == "missing_requested_free":
        records.pop(1)
    elif defect == "reuse_before_completed_free":
        records.pop(2)
    elif defect == "completed_before_requested":
        records[2]["time_us"] = 3
    elif defect == "backwards_generation_time":
        records[3]["time_us"] = 5
    elif defect == "oversized_request":
        records[0]["size"] = 513
    elif defect == "missing_size":
        records[0].pop("size")
    elif defect == "duplicate_completed_free":
        records.append(action("free_completed", 1, 17))
    elif defect == "unclosed_profiler_lifetime":
        events["traceEvents"].pop()
    result = audit_trace(events, limits={"cuda:0": 2048}, cuda_history=history(*records))
    assert not result["passed_observed_temporary_bound"]
    assert result["evidence_errors"]


def test_cuda_requested_free_without_completed_event_remains_active():
    events = trace(
        scope("cache_write", 0, 20),
        memory(1, 512, 1, cuda=True),
        memory(1, -512, 3, cuda=True),
        memory(2, 512, 5, cuda=True),
        memory(2, -512, 8, cuda=True),
    )
    device_history = history(
        action("alloc", 1, 1),
        action("free_requested", 1, 3),
        action("alloc", 2, 5),
        action("free_requested", 2, 8),
        action("free_completed", 2, 9),
    )
    result = audit_trace(events, limits={"cuda:0": 512}, cuda_history=device_history)
    assert result["evidence_errors"] == []
    assert not result["passed_observed_temporary_bound"]
    active = result["devices"]["cuda:0"]["allocator_active_through_completed_free"]
    assert active["bytes"] == 1024
    assert active["bytes_live_at_trace_end"] == 512


def test_no_scope_or_allocator_trace_is_not_evidence():
    for events in (trace(memory(1, 100, 1)), trace(scope("cache_write", 0, 2))):
        with pytest.raises(ValueError, match="lacks cache scopes"):
            audit_trace(events, limits={"cpu": 100})


def test_explicit_empty_metadata_phase_still_requires_ownership_scope():
    result = audit_trace(
        trace(scope("cache_maintenance", 0, 2)), limits={"cpu": 0}, allow_empty_memory=True
    )
    assert result["passed_observed_temporary_bound"]
    assert result["memory_event_count"] == 0
    with pytest.raises(ValueError, match="lacks cache scopes"):
        audit_trace(trace(), limits={"cpu": 0}, allow_empty_memory=True)


def test_empty_cuda_phase_requires_history_and_rejects_hidden_allocations():
    events = trace(scope("cache_maintenance", 0, 2))
    for cuda_history in (None, history(action("alloc", 100, 1))):
        result = audit_trace(
            events, limits={"cuda:0": 0}, cuda_history=cuda_history, allow_empty_memory=True
        )
        assert not result["passed_observed_temporary_bound"]
        assert result["evidence_errors"]
    assert audit_trace(
        events, limits={"cuda:0": 0}, cuda_history=history(), allow_empty_memory=True
    )["passed_observed_temporary_bound"]


def test_pinned_inventory_charges_full_owned_bin_and_deduplicates_views():
    pinned = SimpleNamespace(
        device=torch.device("cpu"),
        is_pinned=lambda: True,
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 1000, nbytes=lambda: 4097),
    )
    result = storage_inventory([pinned, pinned])
    assert result["devices"]["cpu"] == {
        "storage_bytes": 4097,
        "allocator_bytes": 8192,
        "padding_bytes": 4095,
    }
    assert len(result["storages"]) == 1
    assert result["storages"][0]["pinned"]


def scalar_events(sequence, start, end, *, handout=8, missing_sync=False):
    marker = scope("scalar", start, end)
    marker["name"] = f"{SCALAR_PREFIX}{sequence}|cuda:0|8"
    result = scope("result", end - 0.1, end - 0.05)
    result["name"] = f"{SCALAR_RESULT_PREFIX}{sequence}|{handout}"
    events = [marker, result]
    names = ("aten::_local_scalar_dense", "cudaMemcpyAsync", "cudaStreamSynchronize")
    for name in names[:-1] if missing_sync else names:
        event = scope(name, start + 0.1, end - 0.2)
        event.update(name=name, cat="cpu_op" if name.startswith("aten") else "cuda_runtime")
        events.append(event)
    return events


def test_serial_pinned_scalar_reuse_counts_live_interval_not_handout_sum():
    result = audit_trace(
        trace(scope("cache_maintenance", 0, 10), *scalar_events(0, 1, 3), *scalar_events(1, 5, 7)),
        limits={"cpu": 8},
    )
    assert result["passed_observed_temporary_bound"]
    assert result["pinned_scalar_handout_bytes"] == 16
    assert result["devices"]["cpu"]["observed_cache_peak_bytes"] == 8
    # A pageable temporary that actually overlaps the scalar adds to the bound.
    result = audit_trace(
        trace(scope("cache_maintenance", 0, 10), *scalar_events(0, 1, 3), memory(123, 24, 2)),
        limits={"cpu": 31},
    )
    assert not result["passed_observed_temporary_bound"]
    assert result["devices"]["cpu"]["observed_cache_peak_bytes"] == 32


@pytest.mark.parametrize("options", [{"handout": 16}, {"missing_sync": True}])
def test_scalar_buffer_requires_dtype_capacity_and_completed_sync(options):
    result = audit_trace(
        trace(
            scope("cache_maintenance", 0, 10),
            memory(123, 1, 0.5),
            *scalar_events(0, 1, 3, **options),
        ),
        limits={"cpu": 100},
    )
    assert not result["passed_observed_temporary_bound"]
    assert result["evidence_errors"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA scalar transfer requires GPU")
def test_real_cuda_scalar_conversions_are_observed_and_methods_restored(tmp_path):
    tensor = torch.tensor([1, 2, 3], dtype=torch.int64, device="cuda:0")
    originals = {
        name: getattr(torch.Tensor, name) for name in ("item", "__int__", "__float__", "__bool__")
    }
    audit = CacheMemoryAudit(
        tmp_path / "scalar.json", devices=["cuda:0"], record_shapes=False, with_stack=False
    )
    with audit, audit.track_scalar_transfers(), audit.scopes("cache_maintenance"):
        assert tensor[0].item() == 1
        assert int(tensor[1]) == 2
        assert float(tensor[2]) == 3.0
        assert bool(tensor[0])
    result = audit.result(limits={"cpu": 8, "cuda:0": 0})
    assert result["passed_observed_temporary_bound"], result["evidence_errors"]
    assert result["pinned_scalar_handout_bytes"] == 32
    assert result["devices"]["cpu"]["observed_cache_peak_bytes"] == 8
    assert all(getattr(torch.Tensor, name) is original for name, original in originals.items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA native count transfer requires GPU")
def test_native_nonzero_and_unique_pinned_lifetimes_reconcile_all_handouts(tmp_path):
    tensor = torch.tensor([3, 1, 3, 2, 0], dtype=torch.int64, device="cuda:0")
    mask = tensor > 0
    audit = CacheMemoryAudit(
        tmp_path / "native.json", devices=["cuda:0"], record_shapes=False, with_stack=False
    )
    with audit, audit.track_scalar_transfers(), audit.scopes("cache_maintenance"):
        before = host_allocator_counters()
        outputs = (
            torch.nonzero(mask),
            tensor[mask],
            torch.unique(tensor),
            torch.unique_consecutive(tensor),
        )
        assert int(tensor[0]) == 3
        after = host_allocator_counters()
    result = audit.result(limits={"cpu": 8, "cuda:0": 16 * 2**20})
    assert result["passed_observed_temporary_bound"], result["evidence_errors"]
    assert result["pinned_scalar_handout_bytes"] == 8
    assert result["pinned_internal_handout_bytes"] == 24
    assert after["active_bytes.allocated"] - before["active_bytes.allocated"] == 32
    assert [row["copy_bytes"] for row in result["pinned_internal_calls"]] == [4, 4, 8, 8]
    assert result["devices"]["cpu"]["observed_cache_peak_bytes"] == 8
    assert len(outputs) == 4


def test_workspace_components_match_current_independent_admission_contract():
    from cache.sparse_token_pool import SharedSparseTokenPool
    from models.deepseek_v32.cache_resources import execution_reservation

    allowance = WorkspaceAllowance.echo(
        queries=1024, context=66560, topk=2048, width=576, host_tokens=131072, pool_tokens=4096
    )
    execution = execution_reservation(1024, 66560, topk=2048, width=576)
    assert (
        allowance.total
        == execution.hbm + SharedSparseTokenPool.estimate_execution_workspace_bytes(131072, 4096)
    )


def test_fixed_storage_inventory_counts_aliases_once_and_includes_cuda_padding():
    fixed = torch.empty(64)
    inventory = storage_inventory([fixed, fixed[2:], fixed[:0]])
    assert inventory["devices"]["cpu"] == {
        "storage_bytes": 256,
        "allocator_bytes": 256,
        "padding_bytes": 0,
    }
    # The inventory only needs the storage/device protocol; no CUDA is touched.
    gpu = SimpleNamespace(
        device=torch.device("cuda:0"),
        untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: 1000, nbytes=lambda: 400),
    )
    snapshot = [
        {"device": 0, "address": 1000, "blocks": [{"state": "active_allocated", "size": 512}]}
    ]
    inventory = storage_inventory([gpu, gpu], cuda_snapshot=snapshot)
    assert inventory["devices"]["cuda:0"]["allocator_bytes"] == 512
    assert inventory["devices"]["cuda:0"]["padding_bytes"] == 112
    with pytest.raises(ValueError, match="no covering active allocator block"):
        storage_inventory([gpu], cuda_snapshot=[])


class Projection:
    def __init__(self, device="cpu"):
        self.device = device

    def project(self):
        return SimpleNamespace(kv=torch.empty(100, device=self.device))


def test_real_cpu_trace_records_temporary_peak_and_restores_projection(tmp_path):
    projection = Projection()
    original = projection.project
    with (
        CacheMemoryAudit(tmp_path / "trace.json") as audit,
        audit.track_projected_kv([projection]),
    ):
        with audit.scopes("attention_projection"):
            projected = projection.project()
        with audit.scopes("exact_topk"):
            temporary = torch.empty(200)
        del temporary, projected
    result = audit.result(limits={"cpu": 1200})
    assert result["passed_observed_temporary_bound"], result
    assert result["devices"]["cpu"]["observed_cache_peak_bytes"] == 1200
    assert result["projected_source_markers"] == 1
    assert projection.project == original
    assert "project" not in vars(projection)
    assert not audit.history_path.exists()


def test_projection_wrapper_restores_local_override_after_failure(tmp_path):
    projection = Projection()
    projection.project = lambda: SimpleNamespace(kv=torch.empty(1))
    original = projection.project
    audit = CacheMemoryAudit(tmp_path / "unused.json")
    with pytest.raises(RuntimeError, match="injected"), audit.track_projected_kv([projection]):
        raise RuntimeError("injected")
    assert projection.project is original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA allocator history requires GPU")
def test_real_cuda_trace_correlates_allocator_history_and_rounded_bytes(tmp_path):
    projection = Projection("cuda:0")
    # Initialize CUDA before the profiler; fixed allocations are not scratch.
    fixed = torch.empty(7, device="cuda:0")
    with (
        CacheMemoryAudit(
            tmp_path / "trace.json", devices=["cuda:0"], record_shapes=False, with_stack=False
        ) as audit,
        audit.track_projected_kv([projection]),
    ):
        with audit.scopes("attention_projection"):
            projected = projection.project()
        with audit.scopes("exact_topk"):
            temporary = torch.empty(200, device="cuda:0")
        del temporary, projected
    result = audit.result(limits={"cuda:0": 1536, "cpu": 0})
    assert result["passed_observed_temporary_bound"], result
    device = result["devices"]["cuda:0"]
    assert device["observed_cache_peak_bytes"] == 1536
    assert device["observed_allocator_active_peak_bytes"] == 1536
    assert device["projected_source_allocator_padding_bytes"] == 112
    assert result["coverage"]["cuda_pending_free_retention"]
    del fixed
