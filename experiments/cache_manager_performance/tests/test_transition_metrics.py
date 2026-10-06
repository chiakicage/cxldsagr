"""Synthetic NSYS records exercise asynchronous transition accounting on CPU."""

import copy

import pytest

from experiments.cache_manager_performance.src.transition_metrics import analyze_transition

PROCESS = 17 << 24
THREAD = PROCESS | 3


def make_trace(*, manager=False):
    names = [
        ("indexer_fused", 0, 10),
        ("exact_topk", 20, 30),
        ("prefetch_hint", 35, 50),
        ("cache_write", 55, 70),
        ("offload_exact_recall", 75, 90),
        ("attention_output" if manager else "sparse_mla", 100, 125),
    ]
    scopes = [
        {"id": i, "stage": name, "start": start, "end": end, "thread": THREAD, "label": name}
        for i, (name, start, end) in enumerate(names)
    ]
    apis, activities = [], []

    def add(stage, name, start, end, lane, *, api_start=None, kind="kernel"):
        scope = next(scope for scope in scopes if scope["stage"] == stage)
        call = {
            "id": len(apis),
            "scope": scope,
            "thread": THREAD,
            "process": PROCESS,
            "start": scope["start"] + 1 if api_start is None else api_start,
            "name": "cudaLaunchKernel" if kind == "kernel" else "cudaMemcpyAsync",
            "source": "RUNTIME",
            "correlation": len(apis),
            "bookkeeping": False,
        }
        call["end"] = call["start"] + 2
        apis.append(call)
        row = {
            "id": len(activities),
            "scope": scope,
            "api": call,
            "process": PROCESS,
            "device_id": 0,
            "stream_id": 1,
            "correlation": call["correlation"],
            "start": start,
            "end": end,
            "kind": kind,
            "name": name,
            "lane": lane,
        }
        activities.append(row)
        return row

    add("indexer_fused", "sm90_fp8_mqa_logits_fuse_prefetch", 12, 32, "Compute + IO")
    add("exact_topk", "FilteredTopKUnifiedKernel", 35, 55, "Compute")
    add("prefetch_hint", "hint_reduce", 60, 75, "GPU control")
    add("cache_write", "Device-to-Host", 70, 100, "IO", kind="memcpy")
    add("offload_exact_recall", "sparse_map_kernel", 90, 130, "GPU control")
    if not manager:
        add("sparse_mla", "CatArrayBatchedCopy", 132, 140, "GPU control", api_start=101)
        add("sparse_mla", "sparse_attn_fwd_kernel<sm90>", 150, 180, "Compute", api_start=115)
    return scopes, apis, activities


def analyze(trace, *, manager=False, classifier=None):
    return analyze_transition(
        *trace,
        indexer_stage="indexer_fused",
        topk_stage="exact_topk",
        consumer_stage="attention_output" if manager else "sparse_mla",
        endpoint_kind="manager_ready" if manager else "attention_kernel",
        lane_classifier=classifier or (lambda row: row["lane"]),
    )


def stage(result, name):
    return next(row for row in result["stages"] if row["stage"] == name)


def test_cpu_submission_gpu_execution_and_wrapper_preparation_are_separate():
    trace = make_trace()
    saved = copy.deepcopy(trace)
    result = analyze(trace)
    assert result["cpu_topk_to_consumer"]["wall_ms"] == 70 / 1e6
    assert result["gpu_indexer_to_topk"]["window_ms"] == 3 / 1e6
    post = result["gpu_topk_to_consumer"]
    assert (post["start_ns"], post["end_ns"]) == (55, 150)
    assert post["window_ms"] == 95 / 1e6
    assert post["io_union_ms"] == 30 / 1e6
    assert post["gpu_control_union_ms"] == 63 / 1e6
    assert post["exposed_control_ms"] == 48 / 1e6
    assert post["gpu_idle_ms"] == 17 / 1e6
    assert post["gap_ms"] == 65 / 1e6
    assert post["partition_ms"]["exposed_control"] == 48 / 1e6
    assert sum(post["partition_ms"].values()) == pytest.approx(post["window_ms"])
    wrapper = stage(result, "sparse_mla")["gpu_in_post_topk"]
    assert wrapper["gpu_control_union_ms"] == 8 / 1e6
    assert [row["name"] for row in wrapper["activities"]] == ["CatArrayBatchedCopy"]
    assert result["boundaries"]["attention_activity_id"] == 6
    assert result["boundaries"]["attention_launch_start_minus_topk_end_ms"] == 60 / 1e6
    assert trace == saved


def test_cpu_runs_ahead_and_later_layer_occupancy_keeps_source_scope():
    scopes, apis, activities = make_trace()
    for row in activities:
        row["start"] += 200
        row["end"] += 200
    other_scope = {**scopes[2], "id": 100, "label": "layer_1/prefetch_hint"}
    other_api = {**apis[2], "id": 100, "scope": other_scope, "start": 300, "end": 310}
    apis.append(other_api)
    other = {
        **activities[2],
        "id": 100,
        "scope": other_scope,
        "api": other_api,
        "start": 312,
        "end": 360,
        "name": "other_layer_compute",
        "lane": "Compute",
    }
    activities.append(other)
    result = analyze((scopes, apis, activities))
    assert result["boundaries"]["attention_launch_start_minus_topk_end_ms"] == -140 / 1e6
    post = result["gpu_topk_to_consumer"]
    overlap = next(row for row in post["activities"] if row["id"] == 100)
    assert (overlap["start"], overlap["end"]) == (312, 360)
    assert (overlap["clipped_start_ns"], overlap["clipped_end_ns"]) == (312, 350)
    assert overlap["scope"]["label"] == "layer_1/prefetch_hint"
    assert post["compute_union_ms"] == 38 / 1e6
    assert all(row["id"] != 100 for row in result["cpu_topk_to_consumer"]["apis"])
    assert all(row["id"] != 100 for row in stage(result, "prefetch_hint")["cpu"]["apis"])


def test_nested_runtime_driver_events_and_sync_use_interval_union():
    scopes, apis, activities = make_trace()
    hint = scopes[2]
    base = {"thread": THREAD, "process": PROCESS, "scope": hint, "bookkeeping": False}
    apis.extend(
        [
            {
                **base,
                "id": 100,
                "name": "cudaStreamSynchronize",
                "source": "RUNTIME",
                "start": 35,
                "end": 45,
            },
            {
                **base,
                "id": 101,
                "name": "cuStreamSynchronize",
                "source": "DRIVER",
                "start": 37,
                "end": 43,
            },
            {
                **base,
                "id": 102,
                "name": "cudaEventRecord",
                "source": "RUNTIME",
                "start": 44,
                "end": 49,
                "bookkeeping": True,
            },
        ]
    )
    result = analyze((scopes, apis, activities))
    cpu = stage(result, "prefetch_hint")["cpu"]
    assert cpu["wall_ms"] == 15 / 1e6
    assert cpu["api_union_ms"] == 14 / 1e6
    assert cpu["non_api_residual_ms"] == 1 / 1e6
    assert cpu["api_count"] == 4
    assert cpu["api_class_union_ms"]["synchronization"] == 10 / 1e6
    assert cpu["api_class_union_ms"]["event"] == 5 / 1e6
    assert cpu["api_union_intervals_ns"] == [[35, 49]]


def test_manager_readiness_covers_queued_work_beyond_cpu_interval():
    result = analyze(make_trace(manager=True), manager=True)
    post = result["gpu_topk_to_consumer"]
    assert result["cpu_topk_to_consumer"]["end_ns"] == 100
    assert post["end_ns"] == 130
    assert result["boundaries"]["manager_last_activity_id"] == 4
    assert result["boundaries"]["attention_activity_id"] is None
    assert result["boundaries"]["attention_launch_start_minus_topk_end_ms"] is None
    assert (
        stage(result, "offload_exact_recall")["gpu_in_post_topk"]["gpu_control_union_ms"]
        == 40 / 1e6
    )


def test_manager_readiness_also_waits_for_cpu_consumer_entry():
    scopes, apis, activities = make_trace(manager=True)
    scopes[-1].update(start=200, end=210)
    result = analyze((scopes, apis, activities), manager=True)
    assert result["boundaries"]["endpoint_ns"] == 200
    assert result["gpu_topk_to_consumer"]["gpu_idle_ms"] == 75 / 1e6


def test_zero_transport_is_classified_by_caller_evidence():
    trace = make_trace(manager=True)
    recall = trace[2][4]
    recall.update(name="gather_records<unsigned int>", actual_records=0, lane="IO")

    def classify(row):
        if row.get("actual_records") == 0:
            return "GPU control"
        return row["lane"]

    result = analyze(trace, manager=True, classifier=classify)
    post = result["gpu_topk_to_consumer"]
    assert post["io_union_ms"] == 30 / 1e6
    assert post["exposed_control_ms"] == 40 / 1e6
    record = next(row for row in post["activities"] if "gather_records" in row["name"])
    assert record["actual_records"] == 0
    assert record["lane"] == "GPU control"


def test_overlapping_fused_compute_and_io_preserve_additive_partition():
    scopes, apis, activities = make_trace()
    activities[2]["lane"] = "Compute + IO"
    activities[4]["lane"] = "Compute"
    result = analyze((scopes, apis, activities))
    post = result["gpu_topk_to_consumer"]
    assert post["fused_internal_io_unresolved"] is True
    assert post["partition_ms"]["io_fused_overlap"] == 5 / 1e6
    assert post["partition_ms"]["compute_io_overlap"] == 10 / 1e6
    assert post["partition_ms"]["fused_only"] == 10 / 1e6
    assert sum(post["partition_ms"].values()) == pytest.approx(post["window_ms"])


@pytest.mark.parametrize("missing", ["indexer_fused", "exact_topk", "sparse_mla"])
def test_missing_boundary_scope_is_rejected(missing):
    scopes, apis, activities = make_trace()
    scopes = [scope for scope in scopes if scope["stage"] != missing]
    with pytest.raises(ValueError, match="Missing boundary scope"):
        analyze((scopes, apis, activities))


def test_duplicate_scope_and_split_attention_are_rejected():
    scopes, apis, activities = make_trace()
    scopes.append({**scopes[1], "id": 100})
    with pytest.raises(ValueError, match="Duplicate stage scope"):
        analyze((scopes, apis, activities))
    scopes.pop()
    activities.append({**activities[-1], "id": 100, "start": 180, "end": 210})
    with pytest.raises(ValueError, match="split consumers"):
        analyze((scopes, apis, activities))


def test_attention_requires_actual_compute_and_manager_requires_no_op():
    scopes, apis, activities = make_trace()
    activities.pop()
    with pytest.raises(ValueError, match="actual attention kernel"):
        analyze((scopes, apis, activities))
    scopes[-1]["stage"] = "attention_output"
    with pytest.raises(ValueError, match="no-op consumer"):
        analyze((scopes, apis, activities), manager=True)


@pytest.mark.parametrize("field,value", [("device_id", 1), ("process", 18 << 24)])
def test_mixed_gpu_identity_is_rejected(field, value):
    trace = make_trace()
    trace[2][3][field] = value
    with pytest.raises(ValueError, match="Mixed or missing GPU"):
        analyze(trace)


@pytest.mark.parametrize("change", ["negative_timestamp", "reversed_cpu", "reversed_gpu"])
def test_invalid_or_reversed_intervals_are_rejected(change):
    scopes, apis, activities = make_trace()
    if change == "negative_timestamp":
        activities[1]["start"] = -1
    elif change == "reversed_cpu":
        scopes[1]["end"] = 110
    else:
        activities[1]["start"] = 31
    with pytest.raises(ValueError, match="Invalid|Reversed"):
        analyze((scopes, apis, activities))


def test_zero_width_transition_is_valid_and_has_no_clipped_activity():
    scopes, apis, activities = make_trace(manager=True)
    activities[1]["start"] = 32
    result = analyze((scopes, apis, activities), manager=True)
    gap = result["gpu_indexer_to_topk"]
    assert gap["window_ms"] == 0
    assert gap["activities"] == []
    assert sum(gap["partition_ms"].values()) == 0


def test_idle_submission_lower_bound_includes_actual_attention_endpoint():
    scopes, apis, activities = make_trace()
    # The final idle segment is [140, 150), and MLA launch begins at 146.
    activities[-1]["api"].update(start=146, end=148)
    scopes[-1]["end"] = 149
    result = analyze((scopes, apis, activities))
    submission = result["post_topk_idle_submission"]
    assert submission["idle_before_next_launch_api_ms"] == 6 / 1e6
    assert submission["idle_after_next_launch_api_ms"] == 11 / 1e6
    assert submission["unresolved_idle_ms"] == 0
    last = submission["segments"][-1]
    assert (last["start_ns"], last["end_ns"]) == (140, 150)
    assert last["next_activity_id"] == 6
    assert last["next_launch_start_ns"] == 146
    assert last["next_api"]["scope"]["stage"] == "sparse_mla"


def test_queued_attention_does_not_attribute_idle_to_host_submission():
    result = analyze(make_trace())
    submission = result["post_topk_idle_submission"]
    assert submission["idle_before_next_launch_api_ms"] == 0
    assert submission["idle_after_next_launch_api_ms"] == 17 / 1e6


@pytest.mark.parametrize(
    "stream,reason", [(2, "other_stream_ends_idle"), (1, "non_kernel_ends_idle")]
)
def test_transfer_ending_idle_is_not_attributed_to_future_main_stream_launch(stream, reason):
    scopes, apis, activities = make_trace()
    activities[-1]["api"].update(start=146, end=148)
    scopes[-1]["end"] = 149
    transfer_api = {**apis[3], "id": 100, "correlation": 100}
    apis.append(transfer_api)
    activities.append(
        {
            **activities[3],
            "id": 100,
            "api": transfer_api,
            "correlation": 100,
            "stream_id": stream,
            "start": 142,
            "end": 144,
        }
    )
    result = analyze((scopes, apis, activities))
    submission = result["post_topk_idle_submission"]
    # [140,142) ends at the transfer. Only [144,146) can be bounded by MLA's launch.
    assert submission["idle_before_next_launch_api_ms"] == 2 / 1e6
    assert submission["unresolved_idle_ms"] == 2 / 1e6
    unresolved = next(row for row in submission["segments"] if row["start_ns"] == 140)
    assert unresolved["unresolved_reason"] == reason
    assert unresolved["idle_before_next_launch_api_ms"] is None
    assert unresolved["idle_end_activities"][0]["id"] == 100
    assert sum(
        submission[key]
        for key in (
            "idle_before_next_launch_api_ms",
            "idle_after_next_launch_api_ms",
            "unresolved_idle_ms",
        )
    ) == pytest.approx(result["gpu_topk_to_consumer"]["gpu_idle_ms"])


def test_missing_next_stream_or_launch_correlation_keeps_idle_unresolved():
    scopes, apis, activities = make_trace(manager=True)
    scopes[-1].update(start=200, end=210)
    activities[2]["api"] = None
    result = analyze((scopes, apis, activities), manager=True)
    submission = result["post_topk_idle_submission"]
    assert submission["unresolved_idle_ms"] == 75 / 1e6
    assert {row["unresolved_reason"] for row in submission["segments"]} == {
        "unknown_or_invalid_launch_correlation",
        "missing_next_main_stream_activity",
    }
