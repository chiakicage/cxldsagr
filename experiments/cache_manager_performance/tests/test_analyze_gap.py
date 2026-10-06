"""Standalone diagnostics share MFU interval math without asserting its gate."""

import pytest

from experiments.cache_manager_performance.src.analyze import (
    annotate_diagnostic_io,
    category,
    diagnostic_gap_metrics,
    diagnostic_lane,
    interval_metrics,
)
from experiments.deepseek_v32_mfu.src.launch_gap import summarize_window


def activity(name, stage, start=0, end=10, kind="kernel"):
    return {
        "kind": kind,
        "name": name,
        "scope": {"stage": f"window/layer_0/{stage}"},
        "start": start,
        "end": end,
    }


@pytest.mark.parametrize(
    "name",
    [
        "flashinfer::sampling::FilteredTopKUnifiedKernel<float>",
        "flashinfer::sampling::FinalizeTopKIndicesKernel<float>",
        "flashinfer::sampling::StableSortTopKByValueKernel<float>",
        "mask_kernel",
    ],
)
def test_exact_selection_math_is_compute_but_retains_original_selection_category(name):
    row = activity(name, "exact_topk")
    assert category(row) == "selection"
    assert diagnostic_lane(row) == "Compute"


@pytest.mark.parametrize(
    "name",
    [
        "gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda<float>",
        "CatArrayBatchedCopy<float>",
        "deep_gemm::transpose_fp32<float>",
        "FillFunctor<float>",
    ],
)
def test_layout_and_plain_fill_stay_control_inside_exact_selection(name):
    assert diagnostic_lane(activity(name, "exact_topk")) == "GPU control"


def test_cache_fifo_sort_and_hint_stay_control_and_host_io_is_not_compute():
    assert diagnostic_lane(activity("DeviceRadixSortOnesweepKernel", "cache_write")) == (
        "GPU control"
    )
    assert diagnostic_lane(activity("at::native::reduce_kernel", "prefetch_hint")) == (
        "GPU control"
    )
    assert diagnostic_lane(activity("prefetch_ids_kernel", "offload_prepare")) == "IO"
    assert diagnostic_lane(activity("DtoH", "cache_write", kind="memcpy")) == "IO"
    assert diagnostic_lane(activity("DtoD", "exact_topk", kind="memcpy")) == "GPU control"


@pytest.mark.parametrize(
    "name", ["masked_fill_kernel", "CompareFunctor<float>", "_resident_causal_tail_mask"]
)
def test_indexer_causal_mask_is_compute_and_remains_gap_outside_indexer(name):
    mask = activity(name, "indexer", 20, 60)
    assert category(mask) == "indexer_compute"
    assert diagnostic_gap_metrics([mask], 0, 100)["diagnostic_gap_ms"] == 60 / 1e6
    control = activity(name, "cache_write", 20, 60)
    assert diagnostic_lane(control) == "GPU control"
    assert diagnostic_gap_metrics([control], 0, 100)["diagnostic_gap_ms"] == 100 / 1e6


def test_overlap_retains_compute_and_original_outside_indexer_metric():
    rows = [
        activity("sm90_fp8_mqa_logits", "indexer", 0, 40),
        activity("Host-to-Device", "dense_prefetch_0", 20, 60, "memcpy"),
        activity("FilteredTopKUnifiedKernel", "exact_topk", 55, 70),
        activity("CatArrayBatchedCopy<float>", "exact_topk", 50, 80),
        activity("sparse_map_kernel", "offload_exact_recall", 70, 90),
    ]
    metrics = diagnostic_gap_metrics(rows, 0, 100)
    old = interval_metrics([{**row, "category": category(row)} for row in rows], 0, 100)
    assert old["outside_indexer_and_io_ms"] == 40 / 1e6
    assert metrics["diagnostic_gap_ms"] == 30 / 1e6
    assert metrics["diagnostic_gpu_idle_ms"] == 10 / 1e6
    assert metrics["diagnostic_control_only_ms"] == 20 / 1e6
    assert metrics["diagnostic_pure_io_only_ms"] == 15 / 1e6
    assert metrics["diagnostic_non_io_window_lower_ms"] == 85 / 1e6
    assert metrics["diagnostic_gap_no_io_percent_upper_bound"] == pytest.approx(30 / 85 * 100)
    assert old["whole_model_gap_ratio"] is None
    assert all("gate" not in name and "threshold" not in name for name in metrics)


def test_fused_work_reuses_mfu_denominator_without_claiming_separate_io_time():
    rows = [
        activity("sm90_fp8_mqa_logits", "indexer", 0, 30),
        activity("sm90_fp8_mqa_logits_fuse_prefetch", "indexer_prefetch", 20, 60),
        activity("Host-to-Device", "dense_prefetch_0", 50, 70, "memcpy"),
        activity("sparse_map_kernel", "offload_exact_recall", 70, 80),
    ]
    metrics = diagnostic_gap_metrics(rows, 0, 100)
    reference = summarize_window(rows, 0, 100, lane_classifier=diagnostic_lane)
    for name, value in metrics.items():
        key = name.removeprefix("diagnostic_")
        if key in reference:
            assert value == reference[key]
    assert metrics["diagnostic_gap_ms"] == 30 / 1e6
    assert metrics["diagnostic_non_io_window_ms"] == 90 / 1e6
    assert metrics["diagnostic_gap_no_io_percent"] == pytest.approx(100 / 3)
    assert metrics["diagnostic_non_io_window_lower_ms"] == 90 / 1e6
    assert metrics["diagnostic_non_io_window_upper_ms"] == 90 / 1e6
    assert metrics["diagnostic_gap_no_io_percent_lower_bound"] == pytest.approx(100 / 3)
    assert metrics["diagnostic_gap_no_io_percent_upper_bound"] == pytest.approx(100 / 3)
    assert metrics["diagnostic_fused_io_separately_identifiable"] is False


def test_pure_io_zero_denominator_has_no_gate_and_pure_recall_has_no_compute():
    io = activity("Host-to-Device", "dense_prefetch_0", 0, 100, "memcpy")
    metrics = diagnostic_gap_metrics([io], 0, 100)
    assert metrics["diagnostic_gap_no_io_percent_upper_bound"] is None
    recall = activity("sparse_map_kernel", "offload_exact_recall", 10, 80)
    metrics = diagnostic_gap_metrics([recall], 0, 100)
    assert metrics["diagnostic_compute_union_ms"] == 0
    assert metrics["diagnostic_gap_ms"] == 100 / 1e6
    assert metrics["diagnostic_gap_no_io_percent_upper_bound"] == 100
    assert all("gate" not in name and "threshold" not in name for name in metrics)


def test_incomplete_or_empty_window_is_rejected():
    with pytest.raises(ValueError, match="positive duration"):
        diagnostic_gap_metrics([], 3, 3)


def counter_sample(recalled, prefetched=0, scheme="echo"):
    metrics = {
        "recalled_records": recalled,
        "prefetched_records": prefetched,
        "record_bytes": 1152,
        "host_to_device_bytes": (recalled + prefetched) * 1152,
    }
    return {
        "scheme": scheme,
        "metrics": [metrics],
        "checks": [{"layer": 0, "exact_records": True, "metrics": dict(metrics)}],
    }


def test_zero_recall_is_control_in_both_category_and_gap_metrics():
    row = activity("gather_records<unsigned int>", "offload_exact_recall", 10, 80)
    annotate_diagnostic_io([row], counter_sample(0), 7)
    assert row["diagnostic_io_empty"] is True
    assert row["diagnostic_io_records"] == 0
    assert row["diagnostic_io_bytes"] == 0
    assert row["diagnostic_io_evidence"] == "result.json:samples[7].metrics[0].recalled_records"
    assert category(row) == "cache_management"
    assert diagnostic_lane(row) == "GPU control"
    metrics = diagnostic_gap_metrics([row], 0, 100)
    assert metrics["diagnostic_io_union_ms"] == 0
    assert metrics["diagnostic_gap_ms"] == 100 / 1e6
    assert metrics["diagnostic_empty_gather_count"] == 1
    intervals = interval_metrics([{**row, "category": category(row)}], 0, 100)
    assert intervals["known_io_union_ms"] == 0
    assert intervals["outside_indexer_and_io_ms"] == 100 / 1e6
    assert intervals["manager_selection_control_union_ms"] == 70 / 1e6


def test_nonzero_recall_and_zero_fused_io_use_their_distinct_counters():
    gather = activity("gather_records<unsigned int>", "offload_exact_recall", 70, 80)
    fused = activity("sm90_fp8_mqa_logits_fuse_prefetch", "indexer_prefetch", 0, 60)
    annotate_diagnostic_io([gather, fused], counter_sample(9), 0)
    assert diagnostic_lane(gather) == "IO"
    assert gather["diagnostic_io_bytes"] == 9 * 1152
    assert diagnostic_lane(fused) == "Compute"
    assert category(fused) == "indexer_compute"
    metrics = diagnostic_gap_metrics([gather, fused], 0, 100)
    assert metrics["diagnostic_fused_union_ms"] == 0
    assert metrics["diagnostic_compute_union_ms"] == 60 / 1e6
    assert metrics["diagnostic_fused_io_separately_identifiable"] is True
    assert (
        metrics["diagnostic_gap_no_io_percent_lower_bound"]
        == (metrics["diagnostic_gap_no_io_percent_upper_bound"])
    )
    assert metrics["diagnostic_zero_io_fused_count"] == 1
    intervals = interval_metrics(
        [{**row, "category": category(row)} for row in (gather, fused)], 0, 100
    )
    assert intervals["fused_internal_io_unresolved"] is False


def test_zero_recall_with_nonzero_prefetch_retains_fused_bounds():
    gather = activity("gather_records<unsigned int>", "offload_exact_recall", 70, 80)
    fused = activity("sm90_fp8_mqa_logits_fuse_prefetch", "indexer_prefetch", 0, 60)
    annotate_diagnostic_io([gather, fused], counter_sample(0, prefetched=12), 0)
    assert diagnostic_lane(gather) == "GPU control"
    assert diagnostic_lane(fused) == "Compute + IO"
    assert fused["diagnostic_io_records"] == 12


def test_ambiguous_fused_calls_keep_bounds_and_duplicate_recall_needs_per_call_counts():
    fused = activity("sm90_fp8_mqa_logits_fuse_prefetch", "indexer_prefetch")
    rows = [fused, dict(fused)]
    annotate_diagnostic_io(rows, counter_sample(0), 0)
    assert all(row["diagnostic_io_empty"] is None for row in rows)
    assert all(diagnostic_lane(row) == "Compute + IO" for row in rows)
    gather = activity("gather_records<unsigned int>", "offload_exact_recall")
    with pytest.raises(ValueError, match="per-call"):
        annotate_diagnostic_io([gather, dict(gather)], counter_sample(3), 0)


def test_counter_mismatch_and_unmatched_layer_or_transport_path_are_rejected():
    gather = activity("gather_records<unsigned int>", "offload_exact_recall")
    sample = counter_sample(0)
    sample["checks"][0]["metrics"]["recalled_records"] = 1
    with pytest.raises(ValueError, match="differ from the checked"):
        annotate_diagnostic_io([gather], sample, 0)
    gather["scope"] = {"stage": "window/layer_1/offload_exact_recall"}
    with pytest.raises(ValueError, match="matching per-layer"):
        annotate_diagnostic_io([gather], counter_sample(0), 0)
    with pytest.raises(ValueError, match="another transport path"):
        annotate_diagnostic_io([gather], counter_sample(0, scheme="dense_prefetch"), 0)
