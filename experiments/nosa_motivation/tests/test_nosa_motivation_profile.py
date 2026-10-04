"""Protect timeline interval and resident-page exclusion semantics."""

import pytest
import torch

from experiments.nosa_motivation.src.profile import runtime_sources
from experiments.nosa_motivation.src.profile_audit import overlap_gate, recompute_layer
from experiments.nosa_motivation.src.provenance import digest
from experiments.nosa_motivation.src.report import async_speedup_gate
from experiments.nosa_motivation.src.timeline import analyze_trace
from experiments.nosa_motivation.src.work_intervals import expected_misses


def event(category, name, start, elapsed, *, tid=1, external=None):
    return {
        "cat": category,
        "ph": "X",
        "name": name,
        "ts": start,
        "dur": elapsed,
        "tid": tid,
        "args": {} if external is None else {"External id": external},
    }


def test_timeline_excludes_parent_gpu_annotations_and_counts_union_once():
    trace = {
        "traceEvents": [
            event("user_annotation", "root", 0, 100),
            event("user_annotation", "nosa::matrix/test", 0, 80),
            event("cpu_op", "aten::mm", 2, 4, external=7),
            event("kernel", "gemm", 10, 40, tid=9, external=7),
            event("kernel", "gemm", 30, 40, tid=10, external=7),
            event("gpu_user_annotation", "nosa::matrix/test", 10, 60, tid=9),
        ]
    }
    result = analyze_trace(trace, "root")
    assert result["gpu_activity_union_us"] == 60
    assert result["gpu_activity_sum_us"] == 80
    assert result["gpu_idle_inside_request_us"] == 40
    assert result["category_union_us"] == {"matrix": 60}
    assert result["unattributed_kernel_names"] == []


def test_fused_execution_is_not_double_attributed_as_fetch_and_compute():
    trace = {
        "traceEvents": [
            event("user_annotation", "root", 0, 100),
            event("user_annotation", "nosa::sparse_attention", 0, 80),
            event("cpu_op", "launch", 2, 4, external=7),
            event("kernel", "fused_offload", 10, 60, tid=9, external=7),
        ]
    }
    result = analyze_trace(trace, "root")
    assert result["category_union_us"] == {"sparse_attention": 60}
    assert "fetch" not in result["category_union_us"]
    trace["traceEvents"][-1]["dur"] = 200
    with pytest.raises(ValueError, match="escaped"):
        analyze_trace(trace, "root")


def test_miss_provenance_excludes_owned_full_pages_but_reloads_partial_pages():
    ids = torch.tensor([[[0, 1, 1, 2], [0, 1, 2, 2]]], dtype=torch.int32)
    valid = torch.ones_like(ids, dtype=torch.bool)
    tags = torch.tensor([[5, 9], [9, 5], [5, 5]], dtype=torch.int64)
    result = expected_misses(ids, valid, 129, tags, 5, 128, 2)
    assert result == [
        {"row": 1, "bytes": 64 * 128 * 4},
        {"row": 2, "bytes": 64 * 128 * 4},
        {"row": 4, "bytes": 128 * 4},
        {"row": 5, "bytes": 128 * 4},
    ]


def test_profile_requires_identical_runtime_and_orchestration_sources():
    base = {
        "models/nosa/model.py": "model",
        "experiments/nosa_motivation/src/measure.py": "measure",
        "experiments/nosa_motivation/src/profile.py": "profile1",
    }
    changed_profile = {**base, "experiments/nosa_motivation/src/profile.py": "profile2"}
    assert runtime_sources(base) == runtime_sources(changed_profile)
    assert runtime_sources(base) != runtime_sources({**base, "models/nosa/model.py": "new"})


def overlap_records():
    return [
        {
            "method": "async_sparse",
            "mode": "internal_work",
            "request_id": request,
            "sample": sample,
            "layers": [
                {
                    "layer": layer,
                    "page_and_stripe_90pct": None if request == 0 else True,
                    "page_metrics": {"fetch_math_overlap_fraction": 0.95},
                    "stripe_metrics": {"fetch_stripe_math_overlap_fraction": 0.95},
                }
                for layer in range(32)
            ],
        }
        for request in (0, 16)
        for sample in range(3)
    ]


def test_overlap_gate_fails_one_sample_even_with_passing_median():
    records = overlap_records()
    config = {"num_users": 16, "layers": 32}
    assert overlap_gate(records, config, 3)["status"] == "passed"
    layer = records[-1]["layers"][-1]
    layer["stripe_metrics"]["fetch_stripe_math_overlap_fraction"] = 0.89
    layer["page_and_stripe_90pct"] = False
    gate = overlap_gate(records, config, 3)
    assert gate["status"] == "failed"
    assert len(gate["failed_layer_samples"]) == 1
    assert gate["checked_layer_samples"] == 96
    assert gate["no_fetch_layer_samples"] == 96


def test_missing_or_no_fetch_evidence_stays_pending():
    records = overlap_records()
    config = {"num_users": 16, "layers": 32}
    assert overlap_gate(records[:-1], config, 3)["status"] == "pending"
    for record in records:
        for layer in record["layers"]:
            layer["page_and_stripe_90pct"] = None
    assert overlap_gate(records, config, 3)["status"] == "pending"


def test_speedup_uses_uninstrumented_latency_separately_from_overlap():
    rows = [
        {
            "method": method,
            "is_revisit": revisit,
            "latency_ms": 2 if method == "async_sparse" else 1,
        }
        for method in ("sync_sparse", "async_sparse")
        for revisit in (False, True)
    ]
    assert async_speedup_gate(rows)["status"] == "failed"
    assert overlap_gate(overlap_records(), {"num_users": 16, "layers": 32}, 3)["status"] == "passed"


@pytest.fixture
def saved_layer(tmp_path):
    from experiments.nosa_motivation.src.flops import observe_selection
    from experiments.nosa_offload_overlap.src.analyze import (
        stripe_interval_metrics,
        work_interval_metrics,
    )
    from layers.attention import BlockSelection

    ids = torch.full((1, 2, 64), -1, dtype=torch.int32)
    ids[:, :, :2] = torch.tensor([0, 1])
    valid = ids >= 0
    evidence = {
        "block_ids": ids,
        "valid_mask": valid,
        "tags": torch.tensor([[0, 5]], dtype=torch.int64),
        "owner": 5,
        "head_dim": 128,
        "element_size": 2,
        "query_heads": 32,
    }
    path = tmp_path / "selection.pt"
    torch.save(evidence, path)
    geometry = {
        "prefix": 64,
        "queries": 1,
        "kv_heads": 2,
        "expected_fetch_rows": [{"row": 0, "bytes": 32768}],
        "expected_prefix_bytes": 32768,
    }
    record = {
        "mode": "overlap",
        "recorded_transfer_bytes": 32768,
        "selection_file": path.name,
        "selection_sha256": digest(path),
        "selection_observation": observe_selection(
            BlockSelection(ids, 64, valid), query_start=64, query_heads=32
        ),
        "geometry": geometry,
        "fetch_stripes": 8,
        "intervals": [
            {"row": 0, "start_ns": 100, "end_ns": 200, "bytes": 32768, "kind": 1},
            {"row": 2, "start_ns": 100, "end_ns": 200, "bytes": 0, "kind": 2},
        ],
        "stripe_intervals": [
            {
                "row": 130 + index,
                "start_ns": 100 + index * 10,
                "end_ns": 130 + index * 10,
                "bytes": 4096,
                "kind": 3,
            }
            for index in range(8)
        ],
    }
    record["page_metrics"] = work_interval_metrics(geometry, record)
    record["stripe_metrics"] = stripe_interval_metrics(geometry, record, 8)
    record["page_and_stripe_90pct"] = True
    return tmp_path, record, {"candidate_tokens": 1, "history_tokens": 64}


def test_recompute_layer_reopens_selection_and_validates_exact_envelope(saved_layer):
    directory, record, config = saved_layer
    assert recompute_layer(directory, record, config, "async_sparse") is True
    record["intervals"][0]["end_ns"] += 1
    with pytest.raises(ValueError, match="envelope"):
        recompute_layer(directory, record, config, "async_sparse")


def test_recompute_layer_rejects_metric_tampering(saved_layer):
    directory, record, config = saved_layer
    record["stripe_metrics"]["fetch_stripe_math_overlap_fraction"] = 0.0
    with pytest.raises(ValueError, match="stripe metrics"):
        recompute_layer(directory, record, config, "async_sparse")
