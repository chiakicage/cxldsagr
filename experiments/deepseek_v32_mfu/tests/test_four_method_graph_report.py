import sqlite3

import pytest

from experiments.deepseek_v32_mfu.src.operator_report import analyze_captures
from experiments.deepseek_v32_mfu.src.report_mfu import (
    audit_dense_sql,
    audit_dense_transport,
    audit_graph_shapes,
    observer_acceptance_text,
    other_gpu_process_observations,
)
from experiments.deepseek_v32_mfu.src.run_contract import METHODS
from experiments.deepseek_v32_mfu.src.timeline import select_layer


def test_graph_audit_requires_both_a128_and_history_shapes_per_layer():
    result = {
        "prefix_tokens": 2048,
        "extend_tokens": 128,
        "chunk_size": 1024,
        "extend_chunk_size": 128,
        "num_layers": 3,
    }
    graphs, calls = [], []
    for layer in range(3):
        for part in ("projection", "finish"):
            for size in (128, 1024):
                graph_id = len(graphs) + 1
                template = {
                    "layer": layer,
                    "part": part,
                    "queries": size,
                    "capture_graph_id": graph_id,
                    "executable_graph_id": graph_id + 100,
                    "gpu_node_ids": [graph_id * 1000],
                }
                graphs.append(template)
                for method in METHODS:
                    for _ in range(2 if size == 1024 else 1):
                        calls.append(
                            {
                                "mode": method,
                                "phase": "prefill_annotated"
                                if size == 1024
                                else "extend_annotated",
                                "layer": f"layer_{layer}",
                                "stage": f"compute_graph_{part}_layer_{layer}_q_{size}",
                                "graph_replay": True,
                                "graph_capture_id": graph_id,
                                "graph_id": graph_id + 100,
                                "graph_gpu_node_ids": template["gpu_node_ids"],
                            }
                        )
    audit = audit_graph_shapes(result, calls, {"graphs": graphs})
    assert audit["query_shapes"] == [128, 1024]
    assert audit["capture_template_count"] == 12
    assert sum(row["count"] for row in audit["replay_counts"]) == 72
    with pytest.raises(ValueError, match="replay counts"):
        audit_graph_shapes(result, calls[:-1], {"graphs": graphs})
    with pytest.raises(ValueError, match="captured graph shapes"):
        audit_graph_shapes(result, calls, {"graphs": graphs[:-1]})
    calls[0]["graph_capture_id"] = -1
    with pytest.raises(ValueError, match="capture template"):
        audit_graph_shapes(result, calls, {"graphs": graphs})


def test_observer_report_uses_this_run_outcomes_and_other_device_samples():
    rows = [
        {
            "time_utc": "2026-10-05T16:51:15+00:00",
            "gpu_utilization": "3, selected, 97 %, 45 %\n7, other, 0 %, 0 %\n",
            "stdout": "42, selected, 100 MiB\n73, other, 1 MiB\n",
        }
    ]
    observations = other_gpu_process_observations(rows, "selected")
    assert [(item["pid"], item["gpu_index"]) for item in observations] == [(73, 7)]
    assert observations[0]["sampled_gpu_utilization_percent"] == 0
    accepted = {
        name: {"child_exit_code": 0, "observer_wrapper_exit_code": 0, "reconciliation": None}
        for name in ("check", "bench", "profile")
    }
    text = observer_acceptance_text(accepted)
    assert "No observer reconciliation was required" in text
    assert "wrapper 1" not in text and "GPU1" not in text
    accepted["profile"].update(observer_wrapper_exit_code=1, reconciliation={"accepted": True})
    text = observer_acceptance_text(accepted)
    assert "observer wrapper 1" in text and "reconciliations are retained for profile" in text


def test_graph_nodes_are_attributed_to_capture_api_without_synthetic_cpu_time(tmp_path):
    pid, tid = 3 << 24, (3 << 24) + 7
    trace, setup = tmp_path / "trace.sqlite", tmp_path / "setup.sqlite"
    replay = (
        "echo/dense_prefetch/extend_annotated/layer_0/compute_graph_projection_layer_0_q_8/call_0"
    )
    with sqlite3.connect(trace) as db:
        db.executescript("""
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, globalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, name TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, globalPid INTEGER, correlationId INTEGER, name TEXT, graphId INTEGER, graphNodeId INTEGER);
        """)
        db.execute("INSERT INTO NVTX_EVENTS VALUES (0, 100, ?, ?)", (replay, tid))
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (10, 20, ?, 1, 'cudaGraphLaunch')",
            (tid,),
        )
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, 1, ?, 1, ?, 9, ?)",
            [(200, 300, pid, "gemm", 11), (300, 325, pid, "helper", 12)],
        )
    with sqlite3.connect(setup) as db:
        db.execute(
            "CREATE TABLE CUDA_GRAPH_NODE_EVENTS(graphNodeId INTEGER, originalGraphNodeId INTEGER, globalTid INTEGER)"
        )
        db.executemany(
            "INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES (?, ?, ?)", [(11, 1, tid), (12, 2, tid)]
        )
    parent = {
        "mode": "dense_prefetch",
        "phase": "extend_annotated",
        "layer": "layer_0",
        "stage": "compute_graph_projection_layer_0_q_8",
        "useful_flops": None,
        "precision": None,
        "nvtx": replay,
        "graph_replay": True,
        "graph_id": 9,
        "graph_gpu_node_ids": [1, 2],
    }
    child = {
        **parent,
        "stage": "q_a_proj",
        "useful_flops": 1000,
        "precision": "FP8",
        "nvtx": replay + "/graph_api_0_q_a_proj",
        "graph_replay": False,
        "graph_api": True,
        "graph_replay_nvtx": replay,
        "graph_node_ids": [1],
    }
    result = analyze_captures([trace], [parent, child], graph_setup_paths=[setup])
    rows = {row["stage"]: row for row in result["operators"]}
    assert rows["q_a_proj"]["kernel_ns"] == 100
    assert rows["q_a_proj"]["scope_host_union_ms"] == 0
    assert rows["q_a_proj"]["api_thread_union_ms"] == 0
    assert rows[parent["stage"]]["kernel_ns"] == 25
    capture = result["captures"][0]
    assert capture["audit"]["metadata_call_counts_match"]
    assert capture["audit"]["kernel_count_and_time_conserved"]
    assert capture["graph_attribution"]["every_replay_gpu_node_verified"]
    with sqlite3.connect(trace) as db:
        db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId=12")
    with pytest.raises(ValueError, match="missing, duplicate or unknown"):
        analyze_captures([trace], [parent, child], graph_setup_paths=[setup])


def test_timeline_preserves_launch_gaps_and_prelaunched_next_layer_io():
    def scope(start, end, layer, stage):
        return {"start": start, "end": end, "layer": layer, "stage": stage}

    previous = scope(0, 50, "layer_0", None)
    current = scope(60, 140, "layer_1", None)
    prior_compute = scope(10, 30, "layer_0", "q_a_proj")
    compute = scope(65, 100, "layer_1", "q_a_proj")
    prefetch = scope(50, 55, "shared", "dense_history_prefetch_layer_2")
    control = scope(101, 120, "layer_1", "cache_mapping")

    def activity(start, end, owner, name="gemm", stream=1):
        return {
            "start": start,
            "end": end,
            "scope": owner,
            "kind": "kernel",
            "stream_id": stream,
            "name": name,
            "bytes": None,
        }

    activities = [
        activity(75, 90, prior_compute),
        activity(110, 150, compute),
        activity(170, 190, compute),
        activity(100, 130, prefetch, "gather_records", 2),
        activity(150, 160, control, "mapping_update"),
    ]
    summary, rows = select_layer(
        [previous, current, prior_compute, compute, prefetch, control], [], activities
    )
    assert summary["origin_ns"] == 90 and summary["end_ns"] == 190
    assert summary["gpu_gap_ms"] == pytest.approx(20 / 1e6)
    assert summary["compute_io_overlap_ms"] == pytest.approx(20 / 1e6)
    assert summary["next_layer_prefetch_overlap_ms"] == pytest.approx(20 / 1e6)
    assert next(row for row in rows if row["lane"] == "IO")["layer"] == "layer_2"


def test_device_copy_is_control_and_prefetch_overlap_uses_unclipped_gather():
    from experiments.deepseek_v32_mfu.src.timeline import activity_lane

    assert activity_lane({"kind": "memcpy", "name": "Device-to-Device"}) == "GPU control"
    assert activity_lane({"kind": "memcpy", "name": "Host-to-Device"}) == "IO"
    assert activity_lane({"kind": "memcpy", "name": "copy_kind_0"}) == "GPU control"
    assert (
        activity_lane({"kind": "kernel", "name": "sm90_fp8_mqa_logits_fuse_prefetch"})
        == "Compute + IO"
    )
    previous = {"start": 0, "end": 50, "layer": "layer_0", "stage": None}
    current = {"start": 60, "end": 140, "layer": "layer_1", "stage": None}
    prior_compute = {"start": 10, "end": 30, "layer": "layer_0", "stage": "q_a_proj"}
    compute = {"start": 65, "end": 100, "layer": "layer_1", "stage": "q_a_proj"}
    prefetch = {
        "start": 50,
        "end": 55,
        "layer": "shared",
        "stage": "dense_history_prefetch_layer_2",
    }
    activities = [
        {
            "start": 75,
            "end": 90,
            "scope": prior_compute,
            "kind": "kernel",
            "stream_id": 1,
            "name": "gemm",
        },
        {
            "start": 110,
            "end": 150,
            "scope": compute,
            "kind": "kernel",
            "stream_id": 1,
            "name": "gemm",
        },
        {
            "start": 170,
            "end": 190,
            "scope": compute,
            "kind": "kernel",
            "stream_id": 1,
            "name": "gemm",
        },
        {
            "start": 80,
            "end": 200,
            "scope": prefetch,
            "kind": "kernel",
            "stream_id": 2,
            "name": "gather_records",
        },
    ]
    summary, rows = select_layer(
        [previous, current, prior_compute, compute, prefetch], [], activities
    )
    assert summary["window_ms"] == pytest.approx(100 / 1e6)
    assert summary["next_layer_prefetch_full_ms"] == pytest.approx(120 / 1e6)
    assert summary["next_layer_prefetch_overlap_ms"] == pytest.approx(60 / 1e6)
    assert summary["next_layer_prefetch_overlap_percent"] == 50
    assert summary["next_layer_prefetch_started_before_window"]
    assert summary["next_layer_prefetch_ends_after_window"]
    io = next(row for row in rows if row["lane"] == "IO")
    assert (io["raw_start_ns"], io["raw_end_ns"]) == (80, 200)
    assert (io["start_ns"], io["end_ns"]) == (90, 190)


@pytest.fixture
def dma_timeline(tmp_path):
    previous = {"start": 0, "end": 50, "layer": "layer_0", "stage": None}
    current = {"start": 60, "end": 140, "layer": "layer_1", "stage": None}
    prior_compute = {"start": 10, "end": 30, "layer": "layer_0", "stage": "q_a_proj"}
    compute = {"start": 65, "end": 100, "layer": "layer_1", "stage": "q_a_proj"}
    prefetch = {
        "start": 50,
        "end": 55,
        "layer": "shared",
        "stage": "dense_history_prefetch_layer_2",
        "thread": 7,
        "label": "dense prefetch layer 2",
    }
    activities = []
    for start, end, owner, correlation in (
        (75, 90, prior_compute, 1),
        (110, 150, compute, 2),
        (170, 190, compute, 3),
    ):
        activities.append(
            {
                "kind": "kernel",
                "name": "gemm",
                "start": start,
                "end": end,
                "scope": owner,
                "device_id": 0,
                "stream_id": 1,
                "process": 8,
                "correlation": correlation,
            }
        )
    for start, end, correlation in ((80, 120, 4), (160, 200, 5)):
        activities.append(
            {
                "kind": "memcpy",
                "name": "Host-to-Device",
                "start": start,
                "end": end,
                "scope": prefetch,
                "device_id": 0,
                "stream_id": 2,
                "process": 8,
                "correlation": correlation,
                "bytes": 4096,
                "api": {
                    "source": "RUNTIME",
                    "name": "cudaMemcpyAsync",
                    "start": 48 + correlation,
                    "end": 49 + correlation,
                    "thread": 7,
                },
            }
        )
    summary, rows = select_layer(
        [previous, current, prior_compute, compute, prefetch], [], activities
    )
    summary["phase"] = "extend"
    path = tmp_path / "dma.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, globalPid INTEGER, correlationId INTEGER, name TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER, globalPid INTEGER, correlationId INTEGER, copyKind INTEGER, bytes INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, name TEXT);
            CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER, label TEXT);
            INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (1, 'Host-to-Device');
        """)
        for item in activities:
            values = tuple(
                item[key]
                for key in ("start", "end", "device_id", "stream_id", "process", "correlation")
            )
            if item["kind"] == "kernel":
                db.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, 'gemm')",
                    values,
                )
            else:
                db.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, ?, ?, ?, 1, 4096)",
                    values,
                )
                api = item["api"]
                db.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 7, ?, 'cudaMemcpyAsync')",
                    (api["start"], api["end"], item["correlation"]),
                )
    return path, summary, rows


def test_dma_overlap_preserves_complete_memcpy_intervals_and_audits_native_bytes(dma_timeline):
    path, summary, rows = dma_timeline
    assert summary["next_layer_prefetch_transports"] == ["cuda_memcpy_h2d"]
    assert summary["next_layer_prefetch_h2d_memcpy_bytes"] == 8192
    assert summary["next_layer_prefetch_full_ms"] == pytest.approx(80 / 1e6)
    assert summary["next_layer_prefetch_span_ms"] == pytest.approx(120 / 1e6)
    assert summary["next_layer_prefetch_overlap_ms"] == pytest.approx(30 / 1e6)
    assert summary["next_layer_prefetch_overlap_percent"] == 37.5
    assert "host_scope_outside_cuda_api_ms" not in summary
    audit = audit_dense_sql(path, summary, rows)
    assert audit["prefetch_h2d_memcpy_bytes"] == 8192
    assert audit["prefetch_streams"] == [2]
    assert audit["prefetch_full_span_ns"] == 120
    assert audit["overlap_ns"] == 30
    assert {row["kind"] for row in audit["native_activity_rows"]} == {"kernel", "memcpy"}


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET bytes=4095 WHERE correlationId=4", "byte count"),
        (
            "UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET streamId=3 WHERE correlationId=4",
            "native CUPTI row",
        ),
        (
            "UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET copyKind=2 WHERE correlationId=4",
            "not native H2D",
        ),
        (
            "UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET name='cudaMemcpy' WHERE correlationId=4",
            "asynchronous memcpy API",
        ),
    ],
)
def test_dma_native_audit_rejects_transport_corruption(dma_timeline, mutation, message):
    path, summary, rows = dma_timeline
    with sqlite3.connect(path) as db:
        db.execute(mutation)
    with pytest.raises(ValueError, match=message):
        audit_dense_sql(path, summary, rows)


def test_declared_dma_requires_all_layer_byte_conservation_without_gather():
    result = {
        "num_layers": 3,
        "dense_history_transport": {"dense_prefetch": "cuda_memcpy_async_contiguous"},
    }
    summary = {
        "phase": "extend",
        "phase_dense_prefetch_totals": {
            layer: {"h2d_memcpy_bytes": 8192, "h2d_memcpy_count": 1, "mapped_gather_count": 0}
            for layer in range(3)
        },
    }
    counters = [{"host_to_device_bytes": 8192} for _ in range(3)]
    assert audit_dense_transport(result, summary, counters)[
        "all_layer_phase_bytes_match_cache_counters"
    ]
    summary["phase_dense_prefetch_totals"][1]["h2d_memcpy_bytes"] -= 1
    with pytest.raises(ValueError, match="phase bytes"):
        audit_dense_transport(result, summary, counters)
    summary["phase_dense_prefetch_totals"][1]["mapped_gather_count"] = 1
    with pytest.raises(ValueError, match="mapped-host gather"):
        audit_dense_transport(result, summary, counters)
