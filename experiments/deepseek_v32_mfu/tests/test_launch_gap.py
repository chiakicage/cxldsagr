"""CPU-only interval and semantic checks for the complete-extend gap metric."""

import csv
import json
import sqlite3
from copy import deepcopy

import pytest

from experiments.deepseek_v32_mfu.src import launch_gap
from experiments.deepseek_v32_mfu.src.launch_gap import (
    complement,
    host_scope_gap_totals,
    summarize_window,
)
from experiments.deepseek_v32_mfu.src.timeline import (
    activity_lane,
    computation_segments,
    draw,
    extract,
    select_forward_activities,
    select_layer,
)


def activity(start, end, *, stage="q_a_proj", kind="kernel", name="gemm"):
    return {
        "start": start,
        "end": end,
        "kind": kind,
        "name": name,
        "scope": {"stage": stage},
    }


def test_gap_includes_control_and_idle_but_keeps_compute_overlapped_with_io():
    rows = [
        activity(10, 40),
        activity(30, 70, kind="memcpy", name="Host-to-Device"),
        activity(60, 80, stage="offload_exact_recall", name="sparse_map"),
    ]
    result = summarize_window(rows, 0, 100)
    # Productive union [10,70], all-GPU union [10,80]. IO-only is [40,70].
    assert result["gap_ms"] == pytest.approx(40 / 1e6)
    assert result["gpu_idle_ms"] == pytest.approx(30 / 1e6)
    assert result["control_only_ms"] == pytest.approx(10 / 1e6)
    assert result["pure_io_only_ms"] == pytest.approx(30 / 1e6)
    assert result["gap_no_io_percent_upper_bound"] == pytest.approx(100 * 40 / 70)
    assert not result["conservative_gate_pass"]


def test_fused_compute_io_produces_bounds_without_assuming_an_io_fraction():
    rows = [
        activity(10, 40),
        activity(30, 80, name="sm90_fp8_mqa_logits_fuse_prefetch"),
        activity(75, 90, kind="memcpy", name="Device-to-Host"),
    ]
    result = summarize_window(rows, 0, 100)
    # Gap [0,10]+[90,100]; known IO-only [80,90]; potentially IO [40,90].
    assert result["gap_no_io_percent_lower_bound"] == pytest.approx(100 * 20 / 90)
    assert result["gap_no_io_percent_upper_bound"] == pytest.approx(100 * 20 / 50)
    assert result["fused_io_separately_identifiable"] is False


def test_control_overlapping_compute_does_not_double_count_gap():
    rows = [activity(0, 96), activity(20, 70, stage="cache_write", name="planned_append")]
    result = summarize_window(rows, 0, 100)
    assert result["control_only_ms"] == 0
    assert result["gap_no_io_percent_upper_bound"] == 4
    assert result["conservative_gate_pass"]


def test_threshold_is_strict_and_complete_boundary_includes_leading_trailing_time():
    result = summarize_window([activity(5, 95)], 0, 100)
    assert result["gap_no_io_percent_upper_bound"] == 10
    assert not result["conservative_gate_pass"]
    assert complement([(5, 95)], 0, 100) == [(0, 5), (95, 100)]


def test_all_io_window_does_not_claim_a_measured_non_io_gate():
    result = summarize_window([activity(0, 100, kind="memcpy", name="Host-to-Device")], 0, 100)
    assert result["gap_ms"] == 0
    assert result["gap_no_io_percent_upper_bound"] is None
    assert not result["conservative_gate_pass"]


def test_host_scope_partition_counts_nested_scopes_and_cuda_apis_once():
    gaps = [{"start_ns": 0, "end_ns": 100}]
    scopes = [
        {"start": 0, "end": 100, "stage": "forward_misc"},
        {"start": 20, "end": 60, "stage": "cache_write"},
        {"start": 30, "end": 40, "stage": "metadata"},
    ]
    apis = [{"start": 25, "end": 45}, {"start": 30, "end": 35}]
    rows = {row["stage"]: row for row in host_scope_gap_totals(gaps, scopes, apis)}
    assert rows["forward_misc"]["gap_ms"] == pytest.approx(60 / 1e6)
    assert rows["cache_write"]["gap_ms"] == pytest.approx(30 / 1e6)
    assert rows["metadata"]["gap_ms"] == pytest.approx(10 / 1e6)
    assert sum(row["cuda_api_ms"] for row in rows.values()) == pytest.approx(20 / 1e6)


@pytest.mark.parametrize(
    "stage,name,expected",
    [
        ("embedding", "vectorized_gather_kernel", "Compute"),
        ("lm_head", "nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT", "Compute"),
        ("final_norm_lm_head", "CUDAFunctor_add<float>", "Compute"),
        ("indexer_fused", "echo_native::clean(float *, int, int, int, int)", "Compute"),
        ("indexer_fused", "FillFunctor<unsigned int>", "GPU control"),
        ("exact_topk", "radix_topk_kernel", "Compute"),
        ("exact_topk", "mask_count", "GPU control"),
        ("exact_topk", "publish", "GPU control"),
        ("exact_topk", "void at::native::reduce_kernel<float>", "GPU control"),
        ("prefetch_hint", "prediction_kernel", "GPU control"),
        ("offload_exact_recall", "sparse_union_clear", "GPU control"),
        ("offload_exact_recall", "gather_records_bounded", "IO"),
        ("offload_exact_recall", "resident_selection_kernel", "GPU control"),
        ("cache_write", "planned_append_kernel", "GPU control"),
        ("forward_misc", "reduce_kernel<long>", "GPU control"),
    ],
)
def test_actual_profile_stage_families_are_classified_by_semantics(stage, name, expected):
    assert activity_lane(activity(0, 1, stage=stage, name=name)) == expected


def test_memset_and_device_copies_are_control_even_inside_a_model_operator():
    assert (
        activity_lane(activity(0, 1, stage="lm_head", kind="memset", name="memset"))
        == "GPU control"
    )


@pytest.mark.parametrize(
    "stage,name,expected",
    [
        (
            "exact_topk",
            "void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(float)>",
            "GPU control",
        ),
        (
            "compute_graph_projection_layer_0_q_128",
            "void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(c10::BFloat16)>",
            "GPU control",
        ),
        (
            "q_a_proj",
            "void deep_gemm::transpose_fp32<512, 64>(const float *, float *)",
            "GPU control",
        ),
        (
            "compute_graph_projection_layer_0_q_128",
            "void at::native::CatArrayBatchedCopy_vectorized<OpaqueType<2>>",
            "GPU control",
        ),
        ("indexer_qk", "at::native::arange_cuda_out(...)", "GPU control"),
        (
            "compute_graph_projection_layer_0_q_128",
            "at::native::CUDAFunctorOnSelf_add<long>",
            "GPU control",
        ),
        (
            "compute_graph_projection_layer_0_q_128",
            "unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda::[lambda(int)]>, LoadWithCast<1>, StoreWithCast<1>",
            "GPU control",
        ),
        ("exact_topk", "at::native::FillFunctor<unsigned int>", "GPU control"),
        (
            "compute_graph_projection_layer_0_q_128",
            "direct_copy_kernel_cuda(float), LoadWithCast<1>, StoreWithCast<1>",
            "Compute",
        ),
        ("indexer_qk", "at::native::masked_fill_kernel(float, bool)", "Compute"),
    ],
)
def test_layout_and_metadata_helpers_do_not_become_compute_inside_operators(stage, name, expected):
    row = activity(0, 1, stage=stage, name=name)
    # Matrix graph ownership must not override a node's pure-copy semantics.
    if stage == "q_a_proj":
        row["graph_call"] = 1
    assert activity_lane(row) == expected


def test_compute_labels_group_graph_helpers_without_filling_internal_gaps():
    rows = [
        {
            "lane": "Compute",
            "start_ms": -0.1,
            "end_ms": -0.05,
            "stage": "embedding",
            "scope_label": "compute_graph_projection_layer_0",
        },
        {
            "lane": "Compute",
            "start_ms": 0.0,
            "end_ms": 0.1,
            "stage": "q_a_proj",
            "scope_label": "compute_graph_projection_layer_0",
        },
        {
            "lane": "Compute",
            "start_ms": 0.2,
            "end_ms": 0.3,
            "stage": "apply_rope",
            "scope_label": "compute_graph_projection_layer_0",
        },
        {"lane": "GPU control", "start_ms": 0.3, "end_ms": 0.4, "stage": "cache_write"},
        {"lane": "Compute", "start_ms": 0.5, "end_ms": 0.6, "stage": "exact_topk"},
    ]
    segments = computation_segments(rows)
    assert [row["label"] for row in segments] == ["Embedding", "Projection / RoPE", "Exact top-k"]
    assert segments[1]["kernel_count"] == 2
    assert rows[1]["end_ms"] == 0.1 and rows[2]["start_ms"] == 0.2


def test_same_capture_warmup_and_restore_cannot_erase_measured_edges():
    forward = {"layer": "shared", "stage": "forward_misc", "start": 20, "end": 80}
    compute = activity(35, 60)
    compute["scope"]["start"] = 30
    warmup = activity(0, 10)
    warmup["scope"] = None
    restore = activity(10, 20, kind="memcpy", name="Device-to-Device")
    restore["scope"] = None
    selected, audit = select_forward_activities([forward], [warmup, restore, compute])
    assert selected == [compute]
    assert audit["outside_window_activity_count"] == 2
    result = summarize_window(selected, 20, 80)
    assert result["gap_ms"] == pytest.approx(35 / 1e6)


def test_excluded_work_overlapping_measured_forward_is_rejected():
    forward = {"layer": "shared", "stage": "forward_misc", "start": 20, "end": 80}
    escaped = activity(10, 30)
    escaped["scope"] = None
    with pytest.raises(ValueError, match="overlaps measured"):
        select_forward_activities([forward], [escaped])
    assert (
        activity_lane(activity(0, 1, stage="lm_head", kind="memcpy", name="Device-to-Device"))
        == "GPU control"
    )


def test_primary_gate_requires_full_forward_and_every_layer(tmp_path, monkeypatch):
    labels = [
        f"{method}/{phase}_annotated"
        for method in launch_gap.METHODS
        for phase in ("extend", "prefill")
    ]
    (tmp_path / "result.json").write_text(
        json.dumps({"run_id": "gate_contract", "nsys_capture_order": labels})
    )
    monkeypatch.setattr(launch_gap, "read_calls", lambda _: ([], {}))
    full_pass, layer_pass = True, False

    def method(path):
        return labels[int(path.stem.split("_")[-1]) - 1].split("/")[0]

    def extend(path, *_args, **_kwargs):
        return {
            "method": method(path),
            "full_extend": {"conservative_gate_pass": full_pass},
            "all_layer_gates_pass": layer_pass,
        }

    monkeypatch.setattr(launch_gap, "analyze_capture", extend)
    monkeypatch.setattr(
        launch_gap,
        "analyze_prefill_capture",
        lambda path, *_, **_kwargs: {
            "method": method(path),
            "full_prefill": {
                "gap_ms": 1.0,
                "gap_no_io_percent_upper_bound": 1.0,
                "gate_certifiable": True,
            },
        },
    )
    for full_pass, layer_pass, expected in (
        (True, False, False),
        (False, True, False),
        (True, True, True),
    ):
        result = launch_gap.analyze_run(tmp_path)
        assert result["extend_gate_pass"] is expected
        assert result["all_methods_pass"] is expected
        assert result["all_layers_gate_pass"] is layer_pass


def transport(start, end, *, family="gather", layer=0):
    row = activity(
        start,
        end,
        stage="offload_exact_recall" if family == "gather" else "indexer_fused",
        name="gather_records_bounded"
        if family == "gather"
        else "sm90_fp8_mqa_logits_fuse_prefetch",
    )
    row["scope"].update(layer=f"layer_{layer}", start=start, end=end, label=f"transport_{start}")
    row.update(stream_id=0, device_id=0, correlation=start)
    return row


def transport_counts(recalled=0, prefetched=0):
    return {
        "recalled_records": recalled,
        "prefetched_records": prefetched,
        "record_bytes": 8,
        "host_to_device_bytes": (recalled + prefetched) * 8,
    }


def test_zero_phase_totals_change_summary_details_inventory_and_layer_consistently():
    compute = activity(0, 50)
    compute.update(stream_id=0, device_id=0)
    compute["scope"].update(layer="layer_0", start=0, end=50, label="projection")
    rows = [compute, transport(50, 60), transport(60, 70), transport(70, 90, family="fused")]
    raw = deepcopy(rows)
    evidence = launch_gap.annotate_actual_io(
        rows, [transport_counts()], "echo", "prefill_annotated"
    )
    assert [activity_lane(row) for row in rows] == [
        "Compute",
        "GPU control",
        "GPU control",
        "Compute",
    ]
    assert all(group["status"] == "phase_zero" for group in evidence)
    assert sorted(group["matching_kernel_count"] for group in evidence) == [1, 2]
    assert [{key: value for key, value in row.items() if key != "actual_io"} for row in rows] == raw
    summary = summarize_window(rows, 0, 100)
    assert summary["gap_ms"] == pytest.approx(30 / 1e6)
    assert summary["control_only_ms"] == pytest.approx(20 / 1e6)
    assert summary["io_union_ms"] == summary["fused_union_ms"] == 0
    assert summary["gate_certifiable"]
    details = launch_gap.gap_details(rows, [], 0, 100)
    assert sum(row["duration_ms"] for row in details) == pytest.approx(summary["gap_ms"])
    assert len(details[0]["control_activities"]) == 2
    assert details[0]["control_activities"][0]["actual_io"]["records"] == 0
    inventory = launch_gap.activity_inventory(rows)
    assert sum(row["count"] for row in inventory if row["lane"] == "GPU control") == 2
    scopes = [
        {"layer": "shared", "stage": "forward_misc", "start": 0, "end": 100},
        {"layer": "layer_0", "stage": None, "start": 0, "end": 100},
    ]
    layer, visible = select_layer(scopes, [], rows, layer=0)
    assert layer["end_ns"] == 90
    assert layer["compute_io_gap_ms"] == pytest.approx(20 / 1e6)
    assert [row["lane"] for row in visible if row["kind"] == "kernel"] == [
        activity_lane(row) for row in rows
    ]


def _timeline_activity(start, end, scope, *, name="gemm", kind="kernel"):
    row = activity(start, end, name=name, kind=kind)
    row.update(scope=scope, stream_id=0, device_id=0)
    return row


@pytest.mark.parametrize("phase", ["prefill_annotated", "extend_annotated"])
def test_first_l0_timeline_keeps_startup_embedding_and_overlapped_io(phase):
    forward = {"layer": "shared", "stage": "forward_misc", "start": 0, "end": 140}
    layer = {"layer": "layer_0", "stage": None, "start": 20, "end": 90}
    embedding = {"layer": "shared", "stage": "embedding", "start": 5, "end": 20}
    projection = {"layer": "layer_0", "stage": "q_a_proj", "start": 50, "end": 80}
    scopes = [dict(scope, phase=phase) for scope in (forward, layer, embedding, projection)]
    rows = [
        _timeline_activity(10, 20, scopes[2]),
        _timeline_activity(12, 18, scopes[2], name="Host-to-Device", kind="memcpy"),
        _timeline_activity(60, 100, scopes[3]),
    ]
    summary, visible = select_layer(scopes, [], rows, layer=0)
    assert summary["origin_ns"] == 0
    assert summary["origin_boundary"] == "complete forward start"
    assert summary["end_ns"] == 100
    assert summary["compute_io_gap_ms"] == pytest.approx(50 / 1e6)
    assert summary["compute_io_overlap_ms"] == pytest.approx(6 / 1e6)
    assert "Embedding" in {segment["label"] for segment in computation_segments(visible)}


def test_later_prefill_l0_starts_at_previous_chunk_compute_end():
    forward = {"layer": "shared", "stage": "forward_misc", "start": 0, "end": 400}
    first = {"layer": "layer_0", "stage": None, "start": 10, "end": 80}
    previous = {"layer": "layer_2", "stage": None, "start": 100, "end": 180}
    chosen = {"layer": "layer_0", "stage": None, "start": 190, "end": 290}
    first_compute = dict(first, stage="q_a_proj", start=20, end=70)
    previous_compute = dict(previous, stage="mlp", start=120, end=170)
    projection = dict(chosen, stage="q_a_proj", start=240, end=280)
    embedding = {"layer": "shared", "stage": "embedding", "start": 185, "end": 189}
    rows = [
        _timeline_activity(30, 100, first_compute),
        _timeline_activity(180, 200, previous_compute),
        _timeline_activity(210, 220, embedding),
        _timeline_activity(260, 300, projection),
    ]
    summary, visible = select_layer([forward, first, previous, chosen], [], rows, layer=0)
    assert summary["origin_ns"] == 200
    assert summary["origin_boundary"] == "previous chunk computation end"
    assert summary["end_ns"] == 300
    assert summary["compute_io_gap_ms"] == pytest.approx(50 / 1e6)
    assert all(row["start_ns"] >= 200 for row in visible)


def test_l0_timeline_rejects_missing_complete_forward_boundary():
    layer = {"layer": "layer_0", "stage": None, "start": 20, "end": 90}
    projection = dict(layer, stage="q_a_proj", start=50, end=80)
    with pytest.raises(ValueError, match="complete measured forward boundary"):
        select_layer([layer], [], [_timeline_activity(60, 100, projection)], layer=0)


@pytest.mark.parametrize("phase", ["prefill_annotated", "extend_annotated"])
@pytest.mark.parametrize("recalled", [0, 3])
def test_final_timeline_uses_same_profile_phase_evidence(tmp_path, phase, recalled):
    from experiments.deepseek_v32_mfu.src.report_mfu import save_csv

    path = tmp_path / "capture.sqlite"
    pid, tid = 3 << 24, (3 << 24) + 7
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, globalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, name TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, globalPid INTEGER, correlationId INTEGER, name TEXT);
        """)
        db.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
            [
                (start, end, f"echo/echo/{phase}/{owner}", tid)
                for start, end, owner in [
                    (0, 300, "shared/forward_misc"),
                    (10, 50, "layer_0"),
                    (20, 40, "layer_0/q_a_proj"),
                    (55, 60, "shared/dense_history_prefetch_layer_2"),
                    (90, 210, "layer_1"),
                    (100, 150, "layer_1/indexer_fused"),
                    (170, 210, "layer_1/q_a_proj"),
                ]
            ],
        )
        for correlation, launch, start, end, name in [
            (1, 25, 50, 80, "gemm"),
            (2, 56, 90, 100, "gather_records_bounded"),
            (3, 57, 100, 110, "gather_records_bounded"),
            (4, 110, 120, 170, "sm90_fp8_mqa_logits_fuse_prefetch"),
            (5, 111, 170, 180, "echo_native::clean(float *, int, int, int, int)"),
            (6, 180, 200, 220, "gemm"),
        ]:
            db.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, 'cudaLaunchKernel')",
                (launch, launch + 1, tid, correlation),
            )
            db.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, 1, ?, ?, ?)",
                (start, end, pid, correlation, name),
            )
    phase_key = (
        "prefix_cache_per_layer" if phase == "prefill_annotated" else "extend_cache_per_layer"
    )
    counters = [transport_counts(9, 9), transport_counts(), transport_counts(recalled)]
    measurements = {
        "echo": {
            key: counters if key == phase_key else [transport_counts(9, 9)] * 3
            for key in ("prefix_cache_per_layer", "extend_cache_per_layer")
        }
    }
    summary, rows = extract(path, [], None, measurements=measurements)
    assert (summary["origin_ns"], summary["end_ns"]) == (80, 220)
    assert summary["gpu_gap_ms"] == summary["gpu_idle_ms"] == pytest.approx(40 / 1e6)
    assert summary["compute_io_gap_ms"] == pytest.approx((40 if recalled else 60) / 1e6)
    assert summary["gate_certifiable"] is (recalled == 0)
    assert summary["unresolved_gather_count"] == (2 if recalled else 0)
    assert summary["next_layer_prefetch_overlap_certifiable"] is (recalled == 0)
    assert len(summary["next_layer_prefetch_zero_transfer_activities"]) == (0 if recalled else 2)
    assert summary["next_layer_prefetch_intervals_ns"] == (
        [(90, 100), (100, 110)] if recalled else []
    )
    fused = next(row for row in rows if row["name"] == "sm90_fp8_mqa_logits_fuse_prefetch")
    assert fused["lane"] == "Compute" and fused["actual_io"]["records"] == 0
    assert fused["actual_io"]["counter_path"].endswith(f"{phase_key}[1].prefetched_records")
    assert (fused["raw_start_ns"], fused["raw_end_ns"]) == (120, 170)
    assert "Indexer + prefetch" not in [item["label"] for item in computation_segments(rows)]
    csv_path = tmp_path / "activities.csv"
    save_csv(csv_path, rows)
    with csv_path.open() as stream:
        exported = list(csv.DictReader(stream))
    exported_fused = next(row for row in exported if row["name"] == fused["name"])
    assert json.loads(exported_fused["actual_io"]) == fused["actual_io"]
    draw([summary], rows, tmp_path, phase=summary["phase"])
    svg = (tmp_path / f"timeline_{summary['phase']}.svg").read_text()
    assert "Indexer logits" in svg and "Indexer + prefetch" not in svg
    assert ("unverified gather IO" in svg) is bool(recalled)


def test_unique_positive_transport_total_binds_only_its_matching_family():
    rows = [transport(10, 30), transport(30, 60, family="fused")]
    audit = launch_gap.annotate_actual_io(
        rows, [transport_counts(3, 7)], "echo", "extend_annotated"
    )
    assert {row["status"] for row in audit} == {"unique_positive"}
    assert rows[0]["actual_io"]["records"] == 3
    assert rows[1]["actual_io"]["records"] == 7
    assert [activity_lane(row) for row in rows] == ["IO", "Compute + IO"]
    assert summarize_window(rows, 0, 100)["gate_certifiable"]


@pytest.mark.parametrize("counts", [None, [transport_counts(1)]])
def test_unresolved_gathers_cannot_certify_an_otherwise_passing_gate(counts):
    rows = [activity(0, 99), transport(10, 11), transport(11, 12)]
    launch_gap.annotate_actual_io(rows, counts, "serial_sparse", "extend_annotated")
    summary = summarize_window(rows, 0, 100)
    assert summary["gap_no_io_percent_upper_bound"] == 1
    assert summary["unresolved_gather_count"] == 2
    assert not summary["gate_certifiable"]
    assert not summary["conservative_gate_pass"]
    assert all(row["actual_io"]["records"] is None for row in rows[1:])
    assert summarize_window(rows, 20, 100)["gate_certifiable"]


def test_positive_multiple_fused_calls_keep_their_conservative_bounds():
    rows = [activity(0, 99), transport(10, 11, family="fused"), transport(11, 12, family="fused")]
    launch_gap.annotate_actual_io(
        rows, [transport_counts(prefetched=1)], "echo", "prefill_annotated"
    )
    assert all(activity_lane(row) == "Compute + IO" for row in rows[1:])
    assert all(row["actual_io"]["status"] == "unresolved" for row in rows[1:])
    summary = summarize_window(rows, 0, 100)
    assert summary["gate_certifiable"] and summary["conservative_gate_pass"]


def test_transport_counters_do_not_leak_across_layers_or_capture_warmups():
    rows = [transport(30, 40), transport(40, 50, layer=1)]
    warmup = transport(0, 10)
    warmup["scope"] = None
    scopes = [{"layer": "shared", "stage": "forward_misc", "start": 20, "end": 80}]
    selected, _ = select_forward_activities(scopes, [warmup, *rows])
    launch_gap.annotate_actual_io(
        selected, [transport_counts(), transport_counts(4)], "echo", "extend_annotated"
    )
    assert rows[0]["actual_io"]["status"] == "phase_zero"
    assert rows[1]["actual_io"]["status"] == "unique_positive"
    assert rows[1]["actual_io"]["matching_kernel_count"] == 1
    assert "actual_io" not in warmup


@pytest.mark.parametrize(
    "field,value",
    [
        ("recalled_records", -1),
        ("prefetched_records", True),
        ("host_to_device_bytes", 1),
        ("record_bytes", 0),
    ],
)
def test_transport_annotation_rejects_invalid_counter_evidence(field, value):
    counts = transport_counts()
    counts[field] = value
    with pytest.raises(ValueError, match="counter"):
        launch_gap.annotate_actual_io([transport(0, 10)], [counts], "echo", "extend_annotated")


def test_dense_mixed_transport_total_is_not_assigned_to_one_gather():
    row = transport(0, 10)
    launch_gap.annotate_actual_io(
        [row], [transport_counts(4)], "dense_prefetch", "extend_annotated"
    )
    assert row["actual_io"]["status"] == "unresolved"
    assert not summarize_window([row], 0, 10)["gate_certifiable"]


def test_zero_annotation_requires_matching_transport_family():
    row = activity(0, 10)
    row["actual_io"] = {"status": "phase_zero", "records": 0, "family": "mapped_host_kv_gather"}
    with pytest.raises(ValueError, match="zero IO annotation"):
        activity_lane(row)


def test_unresolved_prefill_cannot_pass_the_primary_gap_gate(tmp_path, monkeypatch):
    labels = [
        f"{method}/{phase}_annotated"
        for method in launch_gap.METHODS
        for phase in ("extend", "prefill")
    ]
    (tmp_path / "result.json").write_text(
        json.dumps({"run_id": "unresolved_prefill", "nsys_capture_order": labels})
    )
    monkeypatch.setattr(launch_gap, "read_calls", lambda _: ([], {}))

    def method(path):
        return labels[int(path.stem.split("_")[-1]) - 1].split("/")[0]

    monkeypatch.setattr(
        launch_gap,
        "analyze_capture",
        lambda path, *_, **_kwargs: {
            "method": method(path),
            "full_extend": {"conservative_gate_pass": True},
            "all_layer_gates_pass": True,
        },
    )
    monkeypatch.setattr(
        launch_gap,
        "analyze_prefill_capture",
        lambda path, *_, **_kwargs: {
            "method": method(path),
            "full_prefill": {
                "gap_ms": 1.0,
                "gap_no_io_percent_upper_bound": 1.0,
                "gate_certifiable": method(path) != "echo",
            },
        },
    )
    result = launch_gap.analyze_run(tmp_path)
    assert result["extend_gate_pass"]
    assert not result["prefill_absolute_gap_gate_pass"]
    assert not result["all_methods_pass"]
    echo = next(row for row in result["prefill_methods"] if row["method"] == "echo")
    assert echo["absolute_gap_over_hbm"] == 1.0
    assert not echo["absolute_gap_at_most_120_percent_hbm"]
