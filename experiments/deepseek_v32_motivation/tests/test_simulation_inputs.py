"""Prevent full-phase or unrelated graph work from entering a single-layer estimate."""

import copy
import json

import pytest

from experiments.deepseek_v32_motivation.src.simulation_inputs import (
    COMPUTE_GROUPS,
    MATRIX_STAGES,
    WITHDRAWN_PROFILE_RUN_ID,
    _dense_pipeline_context,
    _group,
    _work,
    build_inputs,
)


@pytest.mark.parametrize(
    "summary,reason",
    [
        ({"profile_run_id": WITHDRAWN_PROFILE_RUN_ID}, "source profile is withdrawn"),
        (
            {"profile_run_id": "new-profile", "schema": "deepseek-full-extend-graph-mfu-v1"},
            "not adapted or validated",
        ),
    ],
)
def test_withdrawn_and_unadapted_sources_fail_before_other_artifacts_are_read(
    tmp_path, summary, reason
):
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match=reason):
        build_inputs(tmp_path, run_id="new-simulation", operator_calls=tmp_path / "absent.json")


def sample(*, graph=False):
    rows, calls = [], []
    for stage in sorted(MATRIX_STAGES):
        scope = f"selected/{stage}"
        precision = "FP8"
        if stage in {"q_absorb", "mla_qk_pv", "v_expand"}:
            precision = "BF16"
        elif stage == "index_weights_proj":
            precision = "FP32"
        call = {
            "nvtx": scope,
            "mode": "hbm",
            "phase": "prefill_annotated",
            "layer": "layer_1",
            "stage": stage,
            "useful_flops": 1000,
            "precision": precision,
            "dimensions": {"M": 1024},
        }
        if stage == "indexer_qk":
            call["dimensions"] = {"Q": 1024, "P": 64512, "KV": 65536}
        elif stage == "mla_qk_pv":
            call.update(query_tokens=1024, query_start=64512, selected_slots=2048)
        calls.append(call)
        rows.append({"scope_label": scope, "lane": "Compute", "kind": "kernel"})
    if graph:
        projection = [call for call in calls if _group(call["stage"]) == "projection"]
        nodes = list(range(1, len(projection) + 1))
        parent = {
            "nvtx": "selected/projection_replay",
            "useful_flops": None,
            "graph_replay": True,
            "graph_id": 12,
            "graph_capture_id": 11,
            "graph_gpu_node_ids": nodes,
        }
        for call, node in zip(projection, nodes, strict=True):
            row = next(row for row in rows if row["scope_label"] == call["nvtx"])
            row.update(scope_label=parent["nvtx"], graph_id="12", graph_node_id=str(node + 100))
            call.update(
                graph_api=True,
                graph_replay_nvtx=parent["nvtx"],
                graph_id=12,
                graph_capture_id=11,
                graph_gpu_node_ids=nodes,
                graph_node_ids=[node],
            )
        calls.append(parent)
    return rows, calls


def extract(rows, calls):
    return _work(
        rows,
        {"phase": "prefill"},
        calls,
        peaks={"FP8": 1979.0, "BF16": 989.5, "FP32": 67.0},
        peak_reference="https://www.nvidia.com/en-us/data-center/h200/",
    )


def test_only_exact_occurrence_contributes_and_mixed_precision_is_normalized():
    rows, calls = sample()
    earlier = copy.deepcopy(next(call for call in calls if call["stage"] == "indexer_qk"))
    earlier.update(nvtx="earlier/indexer_qk", useful_flops=9999999)
    earlier["dimensions"].update(P=0, KV=1024)
    result = extract(rows, [*calls, earlier])
    assert result["total"]["flops_by_precision"] == {"FP8": 10000, "BF16": 3000, "FP32": 1000}
    expected_ns = 10000 / 1979000 + 3000 / 989500 + 1000 / 67000
    assert result["total"]["ideal_compute_ns"] == pytest.approx(expected_ns)
    assert result["audit"]["matrix_api_count"] == 14
    assert result["stages"]["topk"]["useful_flops"] is None
    assert result["stages"]["topk"]["ideal_compute_ns"] is None


def test_missing_matrix_api_and_wrong_chunk_cannot_silently_reduce_or_replace_work():
    rows, calls = sample()
    with pytest.raises(ValueError, match="14 matrix APIs"):
        extract(rows, calls[:-1])
    index = next(call for call in calls if call["stage"] == "indexer_qk")
    index["dimensions"].update(P=0, KV=1024)
    with pytest.raises(ValueError, match="selected query occurrence"):
        extract(rows, calls)


def test_graph_clones_keep_distinct_ids_but_require_all_replay_nodes():
    rows, calls = sample(graph=True)
    result = extract(rows, calls)
    audit = result["audit"]["graph_replays"][0]
    assert audit["captured_gpu_node_ids"] == list(range(1, 8))
    assert audit["observed_gpu_node_ids"] == list(range(101, 108))
    assert result["total"]["useful_flops"] == 14000
    removed = next(row for row in rows if row.get("graph_node_id") == "101")
    rows.remove(removed)
    with pytest.raises(ValueError, match="node coverage"):
        extract(rows, calls)


def test_graph_matrix_apis_cannot_claim_the_same_captured_node():
    rows, calls = sample(graph=True)
    children = [call for call in calls if call.get("graph_api")]
    children[1]["graph_node_ids"] = children[0]["graph_node_ids"]
    with pytest.raises(ValueError, match="overlapping nodes"):
        extract(rows, calls)


def pipeline_sample():
    rows, calls, native, inventory = [], [], [], []
    selected_compute = {}
    for layer in (0, 1):
        stages = MATRIX_STAGES | {
            "exact_topk",
            f"compute_graph_projection_layer_{layer}_q_128",
            f"compute_graph_finish_layer_{layer}_q_128",
        }
        costs = dict.fromkeys(COMPUTE_GROUPS, 0)
        for stage in sorted(stages):
            identity = {
                "mode": "hbm",
                "phase": "extend_annotated",
                "layer": f"layer_{layer}",
                "stage": stage,
            }
            duration = 10 + layer
            rows.append(
                {
                    **identity,
                    "scope_count": "1",
                    "metadata_call_count": "1",
                    "metadata_call_count_matches": "True",
                    "kernel_ns": str(duration),
                    "kernel_count": "1",
                }
            )
            native.append({**identity, "kernel_ns": duration, "kernel_count": 1})
            inventory.append({**identity, "total_ns": duration})
            costs[_group(stage)] += duration
            if stage == "indexer_qk":
                calls.append({**identity, "dimensions": {"Q": 128, "P": 65536, "KV": 65664}})
        if layer == 1:
            selected_compute = {"stages_ns": costs}
    identity = {
        "mode": "dense_prefetch",
        "phase": "extend_annotated",
        "layer": "shared",
        "stage": "dense_history_prefetch_layer_0",
    }
    rows.append(
        {
            **identity,
            "scope_count": "1",
            "metadata_call_count": "1",
            "metadata_call_count_matches": "True",
            "memcpy_count": "1",
            "memcpy_ns": "1000",
        }
    )
    native.append({**identity, "memcpy_count": 1, "memcpy_ns": 1000})
    calls.append(identity)
    return (
        rows,
        {"operators_by_layer": native, "kernel_inventory": inventory},
        calls,
        selected_compute,
    )


def test_dense_pipeline_context_preserves_distinct_adjacent_layer_costs():
    result = _dense_pipeline_context(*pipeline_sample())
    assert result["previous_layer"] == 0
    assert result["compute"]["stages_ns"] == {
        "projection": 80,
        "index": 10,
        "topk": 10,
        "attention": 10,
        "finish": 60,
    }
    assert result["compute"]["total_ns"] == 170
    assert result["h2d"]["duration_ns"] == 1000
    assert result["audit"]["l1_aggregate_matches_selected_timeline"]


@pytest.mark.parametrize("mutation", ("duplicate", "repeated_call", "timeline_drift", "dma_split"))
def test_dense_pipeline_context_rejects_ambiguous_or_mismatched_sources(mutation):
    rows, analysis, calls, compute = pipeline_sample()
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
        message = "duplicate L0"
    elif mutation == "repeated_call":
        rows[0]["scope_count"] = "2"
        message = "does not describe one call"
    elif mutation == "timeline_drift":
        compute["stages_ns"]["attention"] += 1
        message = "aggregate costs differ"
    else:
        rows[-1]["memcpy_count"] = "2"
        message = "not one complete DMA"
    with pytest.raises(ValueError, match=message):
        _dense_pipeline_context(rows, analysis, calls, compute)
