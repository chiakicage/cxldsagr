"""Adversarial evidence checks for the isolated NOSA CUDA generation join."""

from copy import deepcopy

import pytest

from evaluation import cache_memory_audit as generic
from models.nosa.tests.cache_allocation_join import CudaGenerationJoin
from models.nosa.tests.test_cache_allocation_checkpoint import _ordinary_allocations


def memory(index, address, size, timestamp, *, device=0, thread=1):
    return {
        "name": "[memory]",
        "ph": "i",
        "ts": timestamp,
        "pid": 1,
        "tid": thread,
        "args": {
            # This is the actual Kineto field, including its space and case.
            "Ev Idx": index,
            "Device Type": 0 if device is None else 1,
            "Device Id": -1 if device is None else device,
            "Addr": address,
            "Bytes": size,
        },
    }


def scope(category, start=0, end=100):
    return {
        "name": f"{generic.SCOPE_PREFIX}{category}|synthetic",
        "cat": "user_annotation",
        "ph": "X",
        "ts": start,
        "dur": end - start,
        "pid": 1,
        "tid": 1,
    }


def action(name, address, timestamp, *, size=100):
    return {"action": name, "addr": address, "size": size, "time_us": timestamp, "pool_id": [0, 0]}


def evidence(callbacks, events, *, scopes=None):
    return (
        {
            "traceEvents": [*(scopes or [scope("cache_helpers")]), *callbacks],
            "baseTimeNanoseconds": 0,
        },
        {
            "events_by_device": {"cuda:0": events},
            "possibly_truncated": False,
            "max_entries_per_device": 1000,
        },
    )


def one_allocation(*, requested=100, block=512):
    return evidence(
        [memory(20, 4096, block, 1), memory(21, 4096, -block, 2)],
        [
            action("alloc", 4096, 1000, size=requested),
            action("free_requested", 4096, 1010, size=requested),
            action("free_completed", 4096, 1020, size=requested),
        ],
    )


def join(trace, history, *, baseline=()):
    return CudaGenerationJoin(trace, history, baseline={"storages": list(baseline)})


def test_clock_independent_join_preserves_cpu_and_complementary_ordinary_ownership():
    trace, history = evidence(
        [
            memory(100, 4096, 1024, 1),
            memory(101, 40, 24, 2, device=None),
            memory(102, 8192, 512, 11),
            memory(103, 41, 16, 12, device=None),
            memory(104, 8192, -512, 21),
            memory(105, 41, -16, 22, device=None),
            memory(106, 4096, -1024, 24),
            memory(107, 40, -24, 25, device=None),
        ],
        [
            action("alloc", 4096, 1000, size=900),
            action("alloc", 8192, 1010),
            action("free_requested", 8192, 1020),
            action("free_requested", 4096, 1030, size=900),
            action("free_completed", 8192, 1040),
            action("free_completed", 4096, 1050, size=900),
        ],
        scopes=[scope("ordinary"), scope("cache_helpers", 10, 20)],
    )
    untouched = deepcopy((trace, history))
    original = generic._apply_cuda_history
    adapter = join(trace, history)
    cache = adapter.audit(trace, limits={"cuda:0": 512, "cpu": 16})
    ordinary = _ordinary_allocations(trace, history, ("cuda:0", "cpu"), join=adapter)
    assert cache["passed_observed_temporary_bound"]
    assert ordinary["passed_observed_temporary_bound"]
    assert cache["evidence_errors"] == ordinary["evidence_errors"] == []
    for report, cuda_bytes, cpu_bytes in ((cache, 512, 16), (ordinary, 1024, 24)):
        assert report["devices"]["cuda:0"]["observed_allocator_active_peak_bytes"] == cuda_bytes
        assert report["devices"]["cpu"]["observed_allocator_active_peak_bytes"] == cpu_bytes
        assert report["coverage"]["cuda_generation_matching"] == (
            "all-callback stream bijection in single-thread encounter order"
        )
    assert adapter.summary["devices"]["cuda:0"]["matched_callbacks"] == 4
    assert adapter.summary["clock_tolerance_used"] is False
    assert generic._apply_cuda_history is original
    assert (trace, history) == untouched
    # The public parser now also accepts skewed clocks using its own per-address
    # pairing. The stricter NOSA audit must restore that independent entrypoint.
    public = generic.audit_trace(trace, limits={"cuda:0": 512, "cpu": 16}, cuda_history=history)
    assert public["passed_observed_temporary_bound"]
    assert "cuda_generation_join" not in public
    assert (
        public["coverage"]["cuda_generation_matching"]
        != (cache["coverage"]["cuda_generation_matching"])
    )


@pytest.mark.parametrize("pending", (False, True))
def test_completed_free_lifetime_includes_delayed_or_uncompleted_tail(pending):
    events = [
        action("alloc", 4096, 1000),
        action("free_requested", 4096, 1010),
        action("alloc", 8192, 1020),
        action("free_requested", 8192, 1030),
        action("free_completed", 8192, 1040),
    ]
    if not pending:
        events.append(action("free_completed", 4096, 1050))
    trace, history = evidence(
        [
            memory(0, 4096, 512, 1),
            memory(1, 4096, -512, 2),
            memory(2, 8192, 512, 3),
            memory(3, 8192, -512, 4),
        ],
        events,
    )
    adapter = join(trace, history)
    report = adapter.audit(trace, limits={"cuda:0": 1024})
    assert report["passed_observed_temporary_bound"]
    device = report["devices"]["cuda:0"]
    assert device["observed_cache_peak_bytes"] == 512
    assert device["observed_allocator_active_peak_bytes"] == 1024
    assert device["allocator_live"]["bytes_live_at_trace_end"] == 0
    assert device["allocator_active_through_completed_free"]["bytes_live_at_trace_end"] == (
        512 if pending else 0
    )
    assert adapter.summary["devices"]["cuda:0"]["pending_free_at_end"] == int(pending)
    assert not adapter.audit(trace, limits={"cuda:0": 512})["passed_observed_temporary_bound"]


def test_preexisting_free_requires_owner_and_completed_free_before_reuse():
    trace, history = evidence(
        [memory(0, 4096, -512, 1), memory(1, 4096, 512, 3), memory(2, 4096, -512, 4)],
        [
            action("free_requested", 4096, 1000, size=256),
            action("free_completed", 4096, 1010, size=256),
            action("alloc", 4096, 1020),
            action("free_requested", 4096, 1030),
            action("free_completed", 4096, 1040),
        ],
    )
    owner = {"device": "cuda:0", "address": 4096, "storage_bytes": 256, "allocator_bytes": 512}
    adapter = join(trace, history, baseline=[owner])
    report = adapter.audit(trace, limits={"cuda:0": 512})
    assert report["passed_observed_temporary_bound"]
    assert report["preexisting_allocation_frees"] == 1
    assert report["devices"]["cuda:0"]["allocation_count"] == 1
    assert adapter.summary["devices"]["cuda:0"]["baseline_frees"] == 1
    with pytest.raises(ValueError, match="unproven preexisting"):
        join(trace, history)
    history["events_by_device"]["cuda:0"].pop(1)
    with pytest.raises(ValueError, match="before completed free"):
        join(trace, history, baseline=[owner])


@pytest.mark.parametrize("category", ("ordinary", "unscoped"))
def test_unowned_cuda_callbacks_cannot_be_ignored(category):
    trace, history = one_allocation()
    trace["traceEvents"].extend([memory(22, 8192, 512, 110), memory(23, 8192, -512, 120)])
    if category == "ordinary":
        trace["traceEvents"].insert(0, scope("ordinary", 100, 130))
    history["events_by_device"]["cuda:0"].extend(
        [
            action("alloc", 8192, 1100),
            action("free_requested", 8192, 1110),
            action("free_completed", 8192, 1120),
        ]
    )
    report = join(trace, history).audit(trace, limits={"cuda:0": 512})
    assert report["devices"]["cuda:0"]["allocation_count"] == 1
    assert report["cuda_generation_join"]["devices"]["cuda:0"]["matched_callbacks"] == 4
    history["events_by_device"]["cuda:0"][3]["addr"] = 12288
    with pytest.raises(ValueError, match="full callback stream mismatch"):
        join(trace, history)


@pytest.mark.parametrize("change", ("lost", "extra", "duplicate_index", "reordered_index"))
def test_callback_loss_extra_and_invalid_encounter_indices_fail(change):
    trace, history = one_allocation()
    if change == "lost":
        trace["traceEvents"].pop()
    elif change == "extra":
        trace["traceEvents"].append(memory(22, 8192, 512, 3))
    elif change == "duplicate_index":
        trace["traceEvents"][-1]["args"]["Ev Idx"] = 20
    else:
        trace["traceEvents"][-2:] = reversed(trace["traceEvents"][-2:])
    with pytest.raises(ValueError, match="callbacks|Ev Idx order"):
        join(trace, history)


@pytest.mark.parametrize("wrong_key", ("EvIdx", "Ev idx", "Ev Index"))
def test_actual_ev_idx_spelling_is_required(wrong_key):
    trace, history = one_allocation()
    args = trace["traceEvents"][1]["args"]
    args[wrong_key] = args.pop("Ev Idx")
    with pytest.raises(ValueError, match="Ev Idx order"):
        join(trace, history)


def test_swapping_complete_generations_fails_despite_matching_per_address_counts():
    trace, history = evidence(
        [
            memory(0, 4096, 512, 1),
            memory(1, 4096, -512, 2),
            memory(2, 8192, 512, 3),
            memory(3, 8192, -512, 4),
        ],
        [
            action("alloc", 8192, 1000),
            action("free_requested", 8192, 1010),
            action("free_completed", 8192, 1020),
            action("alloc", 4096, 1030),
            action("free_requested", 4096, 1040),
            action("free_completed", 4096, 1050),
        ],
    )
    with pytest.raises(ValueError, match="full callback stream mismatch"):
        join(trace, history)


@pytest.mark.parametrize(
    "change", ("reuse_pending", "completion_first", "duplicate_request", "duplicate_completion")
)
def test_illegal_allocation_lifecycle_fails(change):
    trace, history = one_allocation()
    events = history["events_by_device"]["cuda:0"]
    if change == "reuse_pending":
        events[-1] = action("alloc", 4096, 1020)
        trace["traceEvents"].append(memory(22, 4096, 512, 3))
    elif change == "completion_first":
        events[1]["action"] = "free_completed"
    elif change == "duplicate_request":
        events.insert(2, action("free_requested", 4096, 1015))
        trace["traceEvents"].append(memory(22, 4096, -512, 3))
    else:
        events.append(action("free_completed", 4096, 1030))
    with pytest.raises(
        ValueError,
        match="before completed free|completion before request|duplicate free request|completion without request",
    ):
        join(trace, history)


@pytest.mark.parametrize(
    "requested,block,valid",
    (
        (1, 512, True),
        (513, 1024, True),
        (513, 512, False),
        (1, 513, False),
        (1, 1024, False),
        ((1 << 20) + 1, (2 << 20) + 512, True),
        ((1 << 20) + 1, (2 << 20) + 1024, False),
    ),
)
def test_profiler_block_must_fit_independent_native_requested_size(requested, block, valid):
    trace, history = one_allocation(requested=requested, block=block)
    if valid:
        assert join(trace, history).audit(trace, limits={"cuda:0": block})[
            "passed_observed_temporary_bound"
        ]
    else:
        with pytest.raises(ValueError, match="native allocation bound"):
            join(trace, history)


@pytest.mark.parametrize("change", ("history_request", "history_completion", "profiler_free"))
def test_history_requested_sizes_and_profiler_block_sizes_are_checked_separately(change):
    trace, history = one_allocation()
    if change == "profiler_free":
        trace["traceEvents"][-1]["args"]["Bytes"] = -1024
    else:
        history["events_by_device"]["cuda:0"][1 if change == "history_request" else 2]["size"] = 101
    with pytest.raises(ValueError, match="history free size|profiler free size"):
        join(trace, history)


@pytest.mark.parametrize(
    "change", ("duplicate", "storage_exceeds_block", "unaligned", "excess_tail", "noninteger")
)
def test_malformed_independent_baseline_is_not_accepted(change):
    trace, history = one_allocation()
    owner = {"device": "cuda:0", "address": 8192, "storage_bytes": 100, "allocator_bytes": 512}
    owners = [owner]
    if change == "duplicate":
        owners.append(dict(owner))
    elif change == "storage_exceeds_block":
        owner["storage_bytes"] = 513
    elif change == "unaligned":
        owner["allocator_bytes"] = 513
    elif change == "excess_tail":
        owner["allocator_bytes"] = 1024
    else:
        owner["storage_bytes"] = True
    with pytest.raises(ValueError, match="baseline"):
        join(trace, history, baseline=owners)


@pytest.mark.parametrize("change", ("drop", "size", "order", "capacity"))
def test_history_changes_after_validation_fail_before_audit(change):
    trace, history = one_allocation()
    adapter = join(trace, history)
    events = history["events_by_device"]["cuda:0"]
    if change == "drop":
        events.pop()
    elif change == "size":
        events[0]["size"] = 99
    elif change == "order":
        events[:2] = reversed(events[:2])
    else:
        history["max_entries_per_device"] += 1
    with pytest.raises(ValueError, match="history"):
        adapter.audit(trace, limits={"cuda:0": 512})


@pytest.mark.parametrize("change", ("callback", "base"))
def test_trace_identity_changes_after_validation_are_rejected(change):
    trace, history = one_allocation()
    adapter = join(trace, history)
    if change == "callback":
        trace["traceEvents"][1]["args"]["Bytes"] = 1024
    else:
        trace["baseTimeNanoseconds"] = 1
    with pytest.raises(ValueError, match="callbacks changed|origin changed"):
        adapter.audit(trace, limits={"cuda:0": 512})


def test_cuda_callbacks_on_multiple_cpu_threads_fail_even_when_unowned():
    trace, history = one_allocation()
    trace["traceEvents"][-1]["tid"] = 2
    with pytest.raises(ValueError, match="one CPU thread"):
        join(trace, history)


@pytest.mark.parametrize(
    "change", ("truncated", "recording_cap", "private_pool", "history_clock", "callback_clock")
)
def test_incomplete_or_unsupported_evidence_is_rejected(change):
    trace, history = one_allocation()
    if change == "truncated":
        history["possibly_truncated"] = True
    elif change == "recording_cap":
        history["max_entries_per_device"] = 3
    elif change == "private_pool":
        history["events_by_device"]["cuda:0"][0]["pool_id"] = [0, 1]
    elif change == "history_clock":
        history["events_by_device"]["cuda:0"][1]["time_us"] = 999
    else:
        trace["traceEvents"][-1]["ts"] = 0
    with pytest.raises(ValueError):
        join(trace, history)


def test_scoped_shared_parser_patch_is_restored_when_audit_raises():
    trace, history = one_allocation()
    adapter = join(trace, history)
    original = generic._apply_cuda_history
    with pytest.raises(ValueError, match="nonnegative"):
        adapter.audit(trace, limits={"cuda:0": -1})
    assert generic._apply_cuda_history is original
