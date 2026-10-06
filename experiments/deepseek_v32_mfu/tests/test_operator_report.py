"""Independent trace/ledger fixtures verify useful MFU and complete attribution."""

import json
import sqlite3
import subprocess
import sys

import pytest

from experiments.deepseek_v32_mfu.src.operator_report import analyze_captures, read_calls

PID = 3 << 24
TID = PID + 9
PEAKS = {"FP8": 2000.0, "BF16": 1000.0, "FP32": 60.0}


def capture(path, *, mode="resident", phase="extend_annotated"):
    prefix = f"echo/{mode}/{phase}/"
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, globalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, globalPid INTEGER, correlationId INTEGER, demangledName INTEGER);
            """
        )
        db.executemany(
            "INSERT INTO StringIds VALUES (?,?)",
            [
                (1, "cudaLaunchKernel"),
                (2, "cuLaunchKernel"),
                (3, "cudaEventRecord"),
                (11, "full_gemm<float, 128>"),
                (12, "selection_helper"),
                (13, "unknown_kernel"),
            ],
        )
        db.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?)",
            [
                (0, 400000, prefix + "layer_0", TID),
                (10000, 200000, prefix + "layer_0/projection", TID),
                (50000, 100000, prefix + "layer_0/linear", TID),
                (300000, 350000, prefix + "layer_0/selection", TID),
                (500000, 750000, prefix + "layer_1/linear", TID),
            ],
        )
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
            [
                (60000, 90000, TID, 1, 1),
                (10000, 11000, TID, 7, 3),
                (310000, 315000, TID, 2, 1),
                (550000, 565000, TID, 3, 1),
            ],
        )
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES (65000,85000,?,1,2)", (TID,))
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
            [
                (100000, 300000, 0, 1, PID, 1, 11),
                (320000, 330000, 0, 1, PID, 2, 12),
                (600000, 700000, 0, 1, PID, 3, 11),
                (800000, 820000, 0, 1, PID, 99, 13),
            ],
        )
    return path


def ledger(*, mode="resident", phase="extend_annotated"):
    return [
        {
            "mode": mode,
            "phase": phase,
            "layer": layer,
            "stage": stage,
            "useful_flops": flops,
            "precision": precision,
        }
        for layer, stage, flops, precision in (
            ("layer_0", "projection", None, None),
            ("layer_0", "linear", 100_000_000_000, "BF16"),
            ("layer_0", "selection", None, None),
            ("layer_1", "linear", 25_000_000_000, "BF16"),
        )
    ]


def test_kernel_mfu_uses_correlated_gpu_duration_and_weighted_cross_layer_ratio(tmp_path):
    path = capture(tmp_path / "trace.sqlite")
    before = path.read_bytes()
    result = analyze_captures([path], ledger(), peaks=PEAKS)
    assert path.read_bytes() == before
    rows = {(row["layer"], row["stage"]): row for row in result["operators_by_layer"]}
    first = rows["layer_0", "linear"]
    assert first["kernel_ms"] == pytest.approx(0.2)
    assert first["scope_host_union_ms"] == pytest.approx(0.05)
    assert first["api_thread_union_ms"] == pytest.approx(0.03)
    assert first["api_count"] == 2  # Runtime and driver calls; time is a union.
    assert first["kernel_launch_count"] == first["kernel_count"] == 1
    assert first["kernel_mfu_percent"] == pytest.approx(50)
    assert rows["layer_1", "linear"]["kernel_mfu_percent"] == pytest.approx(25)
    assert rows["layer_0", "projection"]["kernel_count"] == 0
    assert rows["layer_0", "selection"]["useful_flops"] is None
    assert rows["layer_0", "selection"]["kernel_mfu_percent"] is None
    pooled = {row["stage"]: row for row in result["operators"]}
    assert pooled["linear"]["kernel_mfu_percent"] == pytest.approx(125 / 300 * 100)
    assert pooled["linear"]["kernel_mfu_percent"] != pytest.approx((50 + 25) / 2)


def test_inventory_keeps_unattributed_work_and_conserves_all_gpu_time(tmp_path):
    result = analyze_captures([capture(tmp_path / "trace.sqlite")], ledger(), peaks=PEAKS)
    summary = result["captures"][0]
    audit = summary["audit"]
    assert audit["kernel_count_and_time_conserved"]
    assert audit["metadata_call_counts_match"]
    assert audit["operator_kernel_ns"] == 310000
    assert audit["inventory_kernel_ns"] == 330000
    assert audit["activity_counts_and_ns"]["kernel"]["unattributed_count"] == 1
    assert audit["unattributed_reasons"] == {"no_correlated_api": 1}
    assert audit["excluded_event_api_count"] == 1
    assert summary["devices"][0]["busy_ms"] == pytest.approx(0.33)
    assert summary["devices"][0]["gap_ms"] == pytest.approx(0.39)
    assert sum(row["total_ns"] for row in result["kernel_inventory"]) == 330000
    assert any(row["kernel_name"] == "full_gemm<float, 128>" for row in result["kernel_inventory"])
    assert any(row["stage"] is None for row in result["kernel_inventory"])


def test_layer_misc_indexer_aux_and_multi_kernel_linear_have_exclusive_gpu_attribution(tmp_path):
    path = capture(tmp_path / "nested.sqlite")
    prefix = "echo/resident/extend_annotated/layer_0/"
    with sqlite3.connect(path) as db:
        db.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?)",
            [
                (1, 399999, prefix + "layer_misc", TID),
                (360000, 390000, prefix + "indexer_aux", TID),
                (370000, 375000, prefix + "indexer_qk", TID),
            ],
        )
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)",
            [
                (91000, 92000, TID, 4, 1),  # Second kernel inside the same linear API.
                (250000, 251000, TID, 5, 1),  # Parent layer's inter-stage helper.
                (361000, 362000, TID, 6, 1),  # Aux work outside the inner indexer API.
                (372000, 373000, TID, 8, 1),
            ],
        )
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?)",
            [
                (95000, 100000, 0, 1, PID, 4, 12),
                (301000, 302000, 0, 1, PID, 5, 12),
                (369000, 369500, 0, 1, PID, 6, 12),
                (400000, 420000, 0, 1, PID, 8, 11),
            ],
        )
    calls = ledger()
    for stage, flops, precision in (
        ("layer_misc", None, None),
        ("indexer_aux", None, None),
        ("indexer_qk", 20_000_000_000, "FP8"),
    ):
        calls.append(
            {
                "mode": "resident",
                "phase": "extend_annotated",
                "layer": "layer_0",
                "stage": stage,
                "useful_flops": flops,
                "precision": precision,
            }
        )
    result = analyze_captures([path], calls, peaks=PEAKS)
    rows = {(row["layer"], row["stage"]): row for row in result["operators_by_layer"]}
    assert rows["layer_0", "linear"]["kernel_count"] == 2
    assert rows["layer_0", "linear"]["kernel_ns"] == 205000
    assert rows["layer_0", "layer_misc"]["kernel_ns"] == 1000
    assert rows["layer_0", "indexer_aux"]["kernel_ns"] == 500
    assert rows["layer_0", "indexer_qk"]["kernel_ns"] == 20000
    assert rows["layer_0", "indexer_qk"]["kernel_mfu_percent"] == pytest.approx(50)
    assert result["captures"][0]["audit"]["metadata_call_counts_match"]
    assert result["captures"][0]["audit"]["kernel_count_and_time_conserved"]


def test_missing_metadata_and_call_count_mismatch_never_publish_plausible_mfu(tmp_path):
    calls = ledger()
    del calls[2]  # The actual selection remains visible with unknown semantic work.
    calls.append(dict(calls[1]))  # Incorrect duplicate work would otherwise double the MFU.
    result = analyze_captures([capture(tmp_path / "trace.sqlite")], calls, peaks=PEAKS)
    rows = {(row["layer"], row["stage"]): row for row in result["operators_by_layer"]}
    assert rows["layer_0", "linear"]["kernel_mfu_percent"] is None
    assert (
        rows["layer_0", "linear"]["mfu_unavailable_reason"] == "scope_metadata_call_count_mismatch"
    )
    assert rows["layer_0", "selection"]["useful_flops"] is None
    assert rows["layer_0", "selection"]["mfu_unavailable_reason"] == "missing_call_metadata"
    assert not result["captures"][0]["audit"]["metadata_call_counts_match"]


def test_prefill_and_extend_use_separate_work_ledgers_and_duplicate_capture_is_rejected(tmp_path):
    prefix = capture(tmp_path / "prefix.sqlite", phase="prefill_annotated")
    extend = capture(tmp_path / "extend.sqlite")
    calls = ledger() + ledger(phase="prefill_annotated")
    result = analyze_captures([prefix, extend], calls, peaks=PEAKS)
    assert len(result["captures"]) == 2
    assert result["calls_outside_selected_captures"] == 0
    assert len(result["operators_by_layer"]) == 8
    with pytest.raises(ValueError, match="duplicate mode/phase"):
        analyze_captures([prefix, prefix], calls, peaks=PEAKS)


def test_count_mismatches_in_different_layers_cannot_cancel_in_pooled_mfu(tmp_path):
    calls = ledger()
    calls[-1]["layer"] = "layer_0"
    result = analyze_captures([capture(tmp_path / "trace.sqlite")], calls, peaks=PEAKS)
    pooled = next(row for row in result["operators"] if row["stage"] == "linear")
    assert pooled["scope_count"] == pooled["metadata_call_count"] == 2
    assert not pooled["metadata_call_count_matches"]
    assert pooled["kernel_mfu_percent"] is None


def test_mixed_precision_has_no_misleading_single_peak(tmp_path):
    calls = ledger()
    calls[-1]["precision"] = "FP8"
    result = analyze_captures([capture(tmp_path / "trace.sqlite")], calls, peaks=PEAKS)
    pooled = next(row for row in result["operators"] if row["stage"] == "linear")
    assert pooled["useful_flops"] == 125_000_000_000
    assert pooled["kernel_mfu_percent"] is None
    assert pooled["mfu_unavailable_reason"] == "mixed_or_unspecified_precision"


def test_ledger_validation_and_cli_preserve_input_and_existing_outputs(tmp_path):
    path = capture(tmp_path / "trace.sqlite")
    calls_path = tmp_path / "calls.json"
    calls_path.write_text(json.dumps({"run_id": "fixture", "calls": ledger()}))
    calls, metadata = read_calls(calls_path)
    assert metadata["run_id"] == "fixture" and len(calls) == 4
    output = tmp_path / "analysis"
    command = [
        sys.executable,
        "-m",
        "experiments.deepseek_v32_mfu.src.operator_report",
        "--sqlite",
        str(path),
        "--calls",
        str(calls_path),
        "--output-dir",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert (output / "operators_by_layer.csv").is_file()
    assert (output / "operators.csv").is_file()
    assert (output / "kernel_inventory.csv").is_file()
    before = (output / "analysis.json").read_bytes()
    rejected = subprocess.run(command, check=False, capture_output=True, text=True)
    assert rejected.returncode and "already exist" in rejected.stderr
    assert (output / "analysis.json").read_bytes() == before
    calls[1]["useful_flops"] = -1
    calls_path.write_text(json.dumps(calls))
    with pytest.raises(ValueError, match="nonnegative"):
        read_calls(calls_path)


def full_graph_capture(path):
    """One graph launch owns two matrix APIs and separate cache-control work."""
    prefix = "echo/echo/extend_annotated/shared/"
    forward = prefix + "forward_misc/call_0"
    replay = prefix + "extend_graph_replay_q_128/call_1"
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, globalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, globalPid INTEGER, correlationId INTEGER, demangledName INTEGER, graphId INTEGER, graphNodeId INTEGER);
            CREATE TABLE CUDA_GRAPH_NODE_EVENTS(graphNodeId INTEGER, originalGraphNodeId INTEGER, globalTid INTEGER);
            """
        )
        db.executemany(
            "INSERT INTO StringIds VALUES (?,?)",
            [
                (1, "cudaGraphLaunch_v10000"),
                (11, "quantize_activation"),
                (12, "gemm"),
                (13, "sm90_fp8_mqa_logits_fuse_prefetch"),
                (14, "prepare_cache"),
            ],
        )
        db.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?,?,?,?)",
            [(0, 500000, forward, TID), (40000, 70000, replay, TID)],
        )
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (50000,60000,?,1,1)", (TID,))
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (100000, 105000, 0, 1, PID, 1, 11, 19, 101),
                (110000, 205000, 0, 1, PID, 1, 12, 19, 102),
                (210000, 410000, 0, 1, PID, 1, 13, 19, 103),
                (412000, 422000, 0, 1, PID, 1, 14, 19, 104),
            ],
        )
        db.executemany(
            "INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES (?,?,?)",
            [(100 + node, node, TID) for node in range(1, 5)],
        )
    common = {"mode": "echo", "phase": "extend_annotated", "layer": "shared"}
    calls = [
        {
            **common,
            "stage": "forward_misc",
            "nvtx": forward,
            "useful_flops": None,
            "precision": None,
            "formula": "N/A (no matrix multiply)",
        },
        {
            **common,
            "stage": "extend_graph_replay_q_128",
            "nvtx": replay,
            "graph_replay": True,
            "full_extend_graph": True,
            "graph_id": 19,
            "graph_gpu_node_ids": [1, 2, 3, 4],
            "graph_node_owners": {
                str(node): {"stage": stage, "layer": "layer_0"}
                for node, stage in (
                    (1, "attention_projection"),
                    (2, "attention_projection"),
                    (3, "indexer_fused"),
                    (4, "offload_prepare"),
                )
            },
            "useful_flops": None,
            "precision": None,
            "formula": "N/A (no matrix multiply)",
        },
    ]
    for index, (stage, nodes, precision, flops) in enumerate(
        (
            ("q_a_proj", [1, 2], "BF16", 50_000_000_000),
            ("indexer_fused", [3], "FP8", 200_000_000_000),
        )
    ):
        calls.append(
            {
                **common,
                "layer": "layer_0",
                "stage": stage,
                "nvtx": f"{replay}/graph_api_{index}_{stage}",
                "graph_replay": False,
                "full_extend_graph": False,
                "graph_api": True,
                "graph_replay_nvtx": replay,
                "graph_node_ids": nodes,
                "useful_flops": flops,
                "precision": precision,
            }
        )
    return path, calls


def test_full_graph_matrix_apis_preserve_quantization_and_fused_prefetch_time(tmp_path):
    path, calls = full_graph_capture(tmp_path / "full.sqlite")
    result = analyze_captures([path], calls, peaks=PEAKS, graph_setup_paths=[path])
    rows = {(row["layer"], row["stage"]): row for row in result["operators_by_layer"]}
    linear = rows["layer_0", "q_a_proj"]
    assert linear["kernel_count"] == 2
    assert linear["kernel_ns"] == 100000  # The activation quantization remains in the API.
    assert linear["kernel_mfu_percent"] == pytest.approx(50)
    assert linear["scope_host_union_ms"] == linear["api_count"] == 0
    fused = rows["layer_0", "indexer_fused"]
    assert fused["kernel_ns"] == 200000  # No inferred prefetch duration is subtracted.
    assert fused["kernel_mfu_percent"] == pytest.approx(50)
    control = rows["shared", "extend_graph_replay_q_128"]
    assert control["kernel_ns"] == 10000
    assert control["kernel_mfu_percent"] is None
    summary = result["captures"][0]
    assert summary["graph_attribution"]["replays"] == 1
    assert summary["graph_attribution"]["every_replay_gpu_node_verified"]
    assert summary["audit"]["kernel_count_and_time_conserved"]
    assert summary["audit"]["metadata_call_counts_match"]
    assert summary["audit"]["inventory_kernel_ns"] == 310000


def test_full_graph_gap_only_metadata_cannot_become_an_operator_mfu_report(tmp_path):
    path, calls = full_graph_capture(tmp_path / "full.sqlite")
    with pytest.raises(ValueError, match="capture-time matrix API metadata"):
        analyze_captures([path], calls[:2], peaks=PEAKS, graph_setup_paths=[path])


@pytest.mark.parametrize("mutation", ["missing_node", "extra_launch"])
def test_full_graph_operator_attribution_rejects_incomplete_or_multiple_replay(tmp_path, mutation):
    path, calls = full_graph_capture(tmp_path / "full.sqlite")
    with sqlite3.connect(path) as db:
        if mutation == "missing_node":
            db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId=104")
        else:
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (61000,62000,?,2,1)", (TID,))
    with pytest.raises(ValueError, match="missing or duplicate|exactly one CUDA graph launch"):
        analyze_captures([path], calls, peaks=PEAKS, graph_setup_paths=[path])
