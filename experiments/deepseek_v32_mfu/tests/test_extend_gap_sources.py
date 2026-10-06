"""Offline gap partition checks, including full-graph concurrent activity."""

import pytest

from experiments.deepseek_v32_mfu.src.extend_gap_sources import (
    SUBMISSION_KEYS,
    analyze,
    control_category,
)


def row(start, end, stage, *, lane="Compute", source_stage=None, name="kernel", stream=1):
    return {
        "kind": "kernel",
        "name": name,
        "lane": lane,
        "stage": stage,
        "source_stage": source_stage or stage,
        "raw_start_ns": start,
        "raw_end_ns": end,
        "correlation": 100 + stream,
        "graph_id": 7,
        "stream": stream,
    }


def panel(rows, *, end=40, productive=20, control=5, idle=15):
    return {
        "method": "echo",
        "rows": rows,
        "window": {
            "start_ns": 0,
            "end_ns": end,
            "compute_io_union_ms": productive / 1e6,
            "control_only_ms": control / 1e6,
            "gpu_idle_ms": idle / 1e6,
            "gap_ms": (control + idle) / 1e6,
        },
    }


def accepted(rows, *, api_times=None):
    gaps = []
    for index, activity in enumerate(rows):
        start, end = (-10, -5) if api_times is None else api_times[index]
        gpu = {
            **activity,
            "start": activity["raw_start_ns"],
            "end": activity["raw_end_ns"],
            "api": {
                "name": "cudaGraphLaunch_v10000",
                "start": start,
                "end": end,
                "correlation": activity["correlation"],
            },
        }
        gaps.append({"previous_gpu": None, "next_gpu": gpu, "control_activities": []})
    return {"gaps": gaps}


@pytest.mark.parametrize(
    "stage,source,category",
    [
        ("attention_projection", None, "projection_layout"),
        ("residual_rms_norm", "input_residual_norm", "input_norm_layout"),
        ("residual_rms_norm", "post_attention_residual_norm", "post_norm_layout"),
        ("residual_rms_norm", None, "normalization_layout"),
        ("indexer_qk", "indexer", "indexer_layout"),
        ("indexer_fused", "indexer_prefetch", "indexer_layout"),
        ("exact_topk", None, "topk_layout"),
        ("attention_output", None, "attention_output_layout"),
        ("sparse_mla", None, "attention_layout"),
        ("q_a_proj", None, "projection_layout"),
        ("mlp", "dense_mlp", "mlp_layout"),
        ("offload_prepare", None, "echo_prepare"),
        ("offload_finalize", None, "echo_finalize"),
        ("prefetch_hint", None, "prefetch_hint"),
        ("extend_graph_rollback_backup", None, "rollback_backup"),
        ("compute_graph_projection_layer_0_q_128", None, "projection_layout"),
        ("compute_graph_finish_layer_2_q_128", None, "finish_layout"),
    ],
)
def test_control_categories_keep_capture_source_semantics(stage, source, category):
    assert control_category(row(0, 1, stage, source_stage=source)) == category


def test_scale_transpose_remains_a_specific_source_inside_mlp():
    activity = row(0, 1, "mlp", name="deep_gemm::transpose_fp32<128>")
    assert control_category(activity) == "scale_transpose"


def test_one_graph_api_accounts_for_multiple_idle_intervals_without_new_launches():
    rows = [
        row(0, 10, "attention_projection"),
        row(20, 25, "attention_projection", lane="GPU control", name="CatArrayBatchedCopy"),
        row(30, 40, "mlp"),
    ]
    result = analyze(panel(rows), accepted(rows))
    assert result["gap_ns"] == 20
    assert result["idle_submission_partition_ns"] == {
        key: 15 if key == "after_next_api" else 0 for key in SUBMISSION_KEYS
    }
    assert result["exposed_control_categories_ns"] == {"projection_layout": 5}
    assert result["observed_graph_ids"] == [7]
    assert len(result["idle_intervals"]) == 2
    assert result["idle_intervals"][0]["next_api"] == result["idle_intervals"][1]["next_api"]


def test_concurrent_exposed_controls_form_a_union_bucket():
    rows = [
        row(0, 10, "attention_projection"),
        row(10, 20, "offload_prepare", lane="GPU control"),
        row(15, 25, "indexer_qk", lane="GPU control", stream=2),
        row(25, 30, "mlp"),
    ]
    result = analyze(panel(rows, end=30, productive=15, control=15, idle=0), accepted(rows))
    assert result["control_total_union_ns"] == result["exposed_control_ns"] == 15
    combinations = {
        key: members
        for key, members in result["control_category_members"].items()
        if len(members) > 1
    }
    key, members = next(iter(combinations.items()))
    assert members == ["echo_prepare", "indexer_layout"]
    assert result["exposed_control_categories_ns"][key] == 5
    assert result["concurrent_exposed_control_ns"] == 5
    assert sum(result["exposed_control_categories_ns"].values()) == 15
    assert result["idle_submission_partition_ns"] == dict.fromkeys(SUBMISSION_KEYS, 0)


def test_same_category_concurrency_retains_both_activities_but_counts_union():
    rows = [
        row(0, 10, "offload_prepare", lane="GPU control"),
        row(0, 10, "offload_prepare", lane="GPU control", stream=2),
    ]
    result = analyze(panel(rows, end=10, productive=0, control=10, idle=0), accepted(rows))
    assert result["exposed_control_categories_ns"] == {"echo_prepare": 10}
    assert result["concurrent_exposed_control_ns"] == 10
    assert len(result["concurrent_exposed_control_intervals"][0]["activities"]) == 2


def test_productive_overlap_hides_control_and_is_not_counted_twice():
    rows = [
        row(0, 20, "indexer_fused", lane="Compute + IO"),
        row(10, 25, "offload_prepare", lane="GPU control"),
        row(20, 30, "offload_finalize", lane="GPU control", stream=2),
        row(30, 40, "mlp"),
    ]
    result = analyze(panel(rows, productive=30, control=10, idle=0), accepted(rows))
    assert result["control_total_union_ns"] == 20
    assert result["control_overlapped_by_productive_ns"] == 10
    assert result["gap_ns"] == 10


@pytest.mark.parametrize("same_api", [False, True])
def test_simultaneous_successors_with_agreeing_api_partitions_count_idle_once(same_api):
    rows = [row(0, 10, "attention_projection"), row(20, 30, "mlp"), row(20, 30, "mlp", stream=2)]
    if same_api:
        rows[2]["correlation"] = rows[1]["correlation"]
    result = analyze(panel(rows, end=30, productive=20, control=0, idle=10), accepted(rows))
    interval = result["idle_intervals"][0]
    assert interval["next_gpu"] is None and len(interval["next_gpu_candidates"]) == 2
    assert interval["successor_relation"] == "simultaneous_successors_agree"
    assert (interval["next_api"] is not None) is same_api
    assert result["idle_submission_partition_ns"]["after_next_api"] == 10


def test_simultaneous_successors_with_different_api_timing_remain_ambiguous():
    rows = [row(0, 10, "attention_projection"), row(20, 30, "mlp"), row(20, 30, "mlp", stream=2)]
    result = analyze(
        panel(rows, end=30, productive=20, control=0, idle=10),
        accepted(rows, api_times=[(-10, -5), (0, 5), (18, 22)]),
    )
    assert result["idle_submission_partition_ns"] == {
        key: 10 if key == "ambiguous_next_api" else 0 for key in SUBMISSION_KEYS
    }
    interval = result["idle_intervals"][0]
    assert interval["successor_relation"] == "simultaneous_successors_disagree"
    assert len(interval["candidate_submission_partitions_ns"]) == 2


def test_outside_rows_are_clipped_and_trailing_idle_has_no_invented_successor():
    rows = [row(-20, -10, "mlp"), row(-5, 10, "attention_projection"), row(40, 50, "mlp")]
    result = analyze(panel(rows, end=20, productive=10, control=0, idle=10), accepted(rows))
    assert result["gpu_idle_ns"] == 10
    assert result["idle_submission_partition_ns"]["no_next_gpu"] == 10
    assert result["idle_intervals"][0]["next_gpu_candidates"] == []


def test_exact_accepted_totals_remain_mandatory():
    rows = [row(0, 40, "mlp")]
    with pytest.raises(ValueError, match="Accepted gpu_idle_ms differs"):
        analyze(panel(rows), accepted(rows))
