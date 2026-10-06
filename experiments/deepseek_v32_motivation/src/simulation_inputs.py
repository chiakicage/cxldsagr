"""Extract single-layer simulation costs from the accepted three-layer MFU profile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

COMPUTE_GROUPS = ("projection", "index", "topk", "attention", "finish")
PROFILE_RUN_ID = "deepseek_mfu_dma_a128_profile_20261006_01"
DEFAULT_RUN_ID = "deepseek_compute_io_simulation_20261006_04"
REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_REPORT = REPOSITORY / "experiments/deepseek_v32_mfu/report/four_methods"
DEFAULT_OPERATOR_CALLS = (
    REPOSITORY / "experiments/deepseek_v32_mfu/output/data" / PROFILE_RUN_ID / "operator_calls.json"
)
MATRIX_STAGES = frozenset(
    {
        "q_a_proj",
        "q_b_proj",
        "q_absorb",
        "kv_a_proj",
        "index_q_proj",
        "index_k_proj",
        "index_weights_proj",
        "indexer_qk",
        "mla_qk_pv",
        "v_expand",
        "o_proj",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
    }
)
SOURCE_FILES = (
    "summary.json",
    "analysis.json",
    "operator_mfu_by_layer.csv",
    "timeline_activities.csv",
    "timeline_summary.json",
    "run_acceptance.json",
    "dense_overlap_sql_audit.json",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _one(rows, predicate, description):
    selected = [row for row in rows if predicate(row)]
    _require(len(selected) == 1, f"expected one {description}, found {len(selected)}")
    return selected[0]


def _timeline(timelines, method, phase):
    return _one(
        timelines,
        lambda row: row["method"] == method and row["phase"] == phase,
        f"{method}/{phase} timeline",
    )


def _layer(row):
    return int(row["layer"].removeprefix("layer_"))


def _group(stage):
    if stage.startswith("compute_graph_projection_layer_") or stage in {
        "q_a_proj",
        "q_b_proj",
        "q_absorb",
        "kv_a_proj",
        "index_q_proj",
        "index_k_proj",
        "index_weights_proj",
    }:
        return "projection"
    if stage == "indexer_qk":
        return "index"
    if stage == "exact_topk":
        return "topk"
    if stage == "mla_qk_pv":
        return "attention"
    if stage.startswith("compute_graph_finish_layer_") or stage in {
        "v_expand",
        "o_proj",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
    }:
        return "finish"
    raise ValueError(f"unclassified standalone compute stage: {stage}")


def _activity(row):
    start, end = int(row["raw_start_ns"]), int(row["raw_end_ns"])
    _require(end > start, "nonpositive source activity duration")
    return {
        "start_ns": start,
        "end_ns": end,
        "duration_ns": end - start,
        "layer": _layer(row),
        "stream_id": int(row["stream"]),
        "kind": row["kind"],
        "lane": row["lane"],
        "stage": row["stage"],
        "scope_label": row["scope_label"],
        "correlation_id": int(row["correlation"]),
        "name": row["name"],
    }


def _work(rows, timeline, calls, *, peaks, peak_reference):
    """Match actual calls, including captured APIs, to this exact layer occurrence."""
    scopes = {
        row["scope_label"] for row in rows if row["lane"] == "Compute" and row["kind"] == "kernel"
    }
    selected = [
        call
        for call in calls
        if call["useful_flops"] is not None
        and (call["nvtx"] in scopes or call.get("graph_replay_nvtx") in scopes)
    ]
    _require(
        Counter(call["stage"] for call in selected) == Counter(MATRIX_STAGES),
        "selected layer does not contain exactly one of each of the 14 matrix APIs",
    )
    _require(
        all(
            (call["mode"], call["phase"], call["layer"])
            == ("hbm", timeline["phase"] + "_annotated", "layer_1")
            for call in selected
        ),
        "matrix work belongs to another layer or phase",
    )
    q, prefix = (1024, 64512) if timeline["phase"] == "prefill" else (128, 65536)
    index = _one(selected, lambda call: call["stage"] == "indexer_qk", "selected indexer work")
    mla = _one(selected, lambda call: call["stage"] == "mla_qk_pv", "selected MLA work")
    _require(
        (index["dimensions"]["Q"], index["dimensions"]["P"], index["dimensions"]["KV"])
        == (q, prefix, prefix + q),
        "indexer work does not describe the selected query occurrence",
    )
    _require(
        (mla["query_tokens"], mla["query_start"], mla["selected_slots"]) == (q, prefix, 2048),
        "MLA work does not describe the selected query occurrence",
    )
    for call in selected:
        if call["stage"] in {"indexer_qk", "mla_qk_pv"}:
            continue
        dimensions = call["dimensions"]
        _require(dimensions["M"] == q, "matrix work query shape differs from selected occurrence")
    graph_checks = []
    for scope in sorted({call["graph_replay_nvtx"] for call in selected if call.get("graph_api")}):
        parent = _one(calls, lambda call, scope=scope: call["nvtx"] == scope, "graph replay parent")
        children = [call for call in selected if call.get("graph_replay_nvtx") == scope]
        activities = [
            row
            for row in rows
            if row["scope_label"] == scope and row["kind"] in {"kernel", "memcpy", "memset"}
        ]
        original_nodes = set(parent["graph_gpu_node_ids"])
        observed_nodes = {int(row["graph_node_id"]) for row in activities}
        _require(
            parent.get("graph_replay")
            and len(original_nodes) == len(parent["graph_gpu_node_ids"])
            and len(observed_nodes) == len(activities) == len(original_nodes),
            "selected graph replay node coverage differs from its ledger",
        )
        _require(
            all(
                int(row["graph_id"]) == parent["graph_id"]
                for row in activities
                if row["kind"] == "kernel"
            ),
            "selected graph replay identity differs from its ledger",
        )
        claimed = []
        for call in children:
            _require(
                call["graph_id"] == parent["graph_id"]
                and call["graph_capture_id"] == parent["graph_capture_id"]
                and set(call["graph_gpu_node_ids"]) == original_nodes
                and set(call["graph_node_ids"]) <= original_nodes
                and call["graph_node_ids"],
                "captured matrix API graph ownership differs from replay parent",
            )
            claimed.extend(call["graph_node_ids"])
        _require(len(set(claimed)) == len(claimed), "captured matrix APIs claim overlapping nodes")
        graph_checks.append(
            {
                "scope_label": scope,
                "graph_id": parent["graph_id"],
                "graph_capture_id": parent["graph_capture_id"],
                "captured_gpu_node_ids": sorted(original_nodes),
                "observed_gpu_node_ids": sorted(observed_nodes),
                "matrix_api_count": len(children),
                "node_count_matches": True,
                "clone_lineage_basis": (
                    "Accepted source analysis verifies every replay node against setup-capture "
                    "clone lineage. Raw observed clone IDs need not equal captured node IDs."
                ),
            }
        )
    stage_work, total_flops = {}, Counter()
    for group in COMPUTE_GROUPS:
        group_calls = [call for call in selected if _group(call["stage"]) == group]
        flops = Counter()
        for call in group_calls:
            precision = call["precision"]
            useful = call["useful_flops"]
            _require(
                precision in peaks and isinstance(useful, int) and useful > 0,
                "matrix work must have positive integer FLOPs and an explicit precision peak",
            )
            flops[precision] += useful
        total_flops.update(flops)
        stage_work[group] = {
            "flops_by_precision": dict(sorted(flops.items())),
            "useful_flops": sum(flops.values()) if group_calls else None,
            "ideal_compute_ns": sum(value / (peaks[key] * 1000) for key, value in flops.items())
            if group_calls
            else None,
            "calls": group_calls,
        }
    return {
        "dense_peaks_tflops": peaks,
        "peak_reference": peak_reference,
        "stages": stage_work,
        "total": {
            "flops_by_precision": dict(sorted(total_flops.items())),
            "useful_flops": sum(total_flops.values()),
            "ideal_compute_ns": sum(
                value / (peaks[key] * 1000) for key, value in total_flops.items()
            ),
        },
        "audit": {
            "matrix_api_count": len(selected),
            "exact_occurrence_and_query_shape_verified": True,
            "graph_replays": graph_checks,
            "basis": (
                "Useful matrix FLOPs from exact NVTX calls and graph_replay_nvtx-linked "
                "captured APIs. No scaling of all-chunk aggregates; FMA counts as two FLOPs. "
                "Padding and nonmatrix work are excluded from FLOPs; nonmatrix compute time "
                "remains in the simulation denominator. Top-k has no matrix MFU."
            ),
        },
    }


def _compute(rows, timeline, calls, *, peaks, peak_reference):
    own = [row for row in rows if row["method"] == "hbm" and row["phase"] == timeline["phase"]]
    selected = [
        {**_activity(row), "group": _group(row["stage"])}
        for row in own
        if row["lane"] == "Compute" and row["kind"] == "kernel"
    ]
    _require(selected and all(row["layer"] == 1 for row in selected), "invalid compute ownership")
    _require(
        all(
            timeline["origin_ns"] <= row["start_ns"] < row["end_ns"] <= timeline["end_ns"]
            for row in selected
        ),
        "selected compute activity is clipped",
    )
    stages_ns = {
        group: sum(row["duration_ns"] for row in selected if row["group"] == group)
        for group in COMPUTE_GROUPS
    }
    _require(all(stages_ns.values()), "missing standalone compute group")
    union_ns, previous_end = 0, 0
    for row in sorted(selected, key=lambda item: item["start_ns"]):
        union_ns += max(0, row["end_ns"] - max(previous_end, row["start_ns"]))
        previous_end = max(previous_end, row["end_ns"])
    excluded_ns, excluded_count = Counter(), Counter()
    for row in own:
        if row["kind"] in {"kernel", "memcpy", "memset"} and row["lane"] != "Compute":
            key = f"{row['lane']}/{row['kind']}"
            excluded_ns[key] += int(row["raw_end_ns"]) - int(row["raw_start_ns"])
            excluded_count[key] += 1
    return {
        "source_scheme": "hbm",
        "stages_ns": stages_ns,
        "total_ns": sum(stages_ns.values()),
        "kernel_count": len(selected),
        "activities": selected,
        "work": _work(own, timeline, calls, peaks=peaks, peak_reference=peak_reference),
        "audit": {
            "all_five_compute_groups_present": True,
            "all_selected_activities_are_kernels": True,
            "selected_kernels_are_unclipped": True,
            "source_compute_union_ns": union_ns,
            "source_kernel_sum_minus_union_ns": sum(stages_ns.values()) - union_ns,
            "excluded_activity_duration_ns": dict(sorted(excluded_ns.items())),
            "excluded_activity_count": dict(sorted(excluded_count.items())),
            "source_window_ns": timeline["end_ns"] - timeline["origin_ns"],
            "boundary": (
                "Standalone HBM projection, index, top-k, attention and finish kernel-duration "
                "sums; no gaps, cache/control, D2D or memset. The simulator serializes these costs, "
                "rather than subtracting overhead from a measured wall time."
            ),
        },
    }


def _dense_pipeline_context(operator_rows, analysis, calls, selected_compute):
    """Bind the preceding L0 costs without treating repeated layers as identical."""
    layer_costs, layer_sources = {}, {}
    for layer in (0, 1):
        label = f"layer_{layer}"
        expected = MATRIX_STAGES | {
            "exact_topk",
            f"compute_graph_projection_layer_{layer}_q_128",
            f"compute_graph_finish_layer_{layer}_q_128",
        }
        selected = [
            row
            for row in operator_rows
            if row["mode"] == "hbm"
            and row["phase"] == "extend_annotated"
            and row["layer"] == label
            and row["stage"] in expected
        ]
        _require(
            len(selected) == len(expected) and {row["stage"] for row in selected} == expected,
            f"incomplete or duplicate L{layer} extend compute operators",
        )
        for row in selected:
            _require(
                int(row["scope_count"]) == int(row["metadata_call_count"]) == 1
                and row["metadata_call_count_matches"] == "True",
                f"L{layer} extend operator does not describe one call",
            )
            native = _one(
                analysis["operators_by_layer"],
                lambda item, row=row: all(
                    item[key] == row[key] for key in ("mode", "phase", "layer", "stage")
                ),
                f"L{layer} {row['stage']} operator analysis",
            )
            _require(
                native["kernel_ns"] == int(row["kernel_ns"])
                and native["kernel_count"] == int(row["kernel_count"]),
                f"L{layer} operator CSV differs from analysis",
            )
        costs = {
            group: sum(int(row["kernel_ns"]) for row in selected if _group(row["stage"]) == group)
            for group in COMPUTE_GROUPS
        }
        inventory = {
            group: sum(
                item["total_ns"]
                for item in analysis["kernel_inventory"]
                if item["mode"] == "hbm"
                and item["phase"] == "extend_annotated"
                and item["layer"] == label
                and item["stage"] in expected
                and _group(item["stage"]) == group
            )
            for group in COMPUTE_GROUPS
        }
        _require(all(costs.values()) and costs == inventory, f"L{layer} kernel inventory differs")
        index_call = _one(
            calls,
            lambda item, label=label: (
                item["mode"] == "hbm"
                and item["phase"] == "extend_annotated"
                and item["layer"] == label
                and item["stage"] == "indexer_qk"
            ),
            f"L{layer} complete-extend indexer call",
        )
        _require(
            tuple(index_call["dimensions"][key] for key in ("Q", "P", "KV")) == (128, 65536, 65664),
            f"L{layer} extend query shape differs",
        )
        layer_costs[layer], layer_sources[layer] = costs, selected
    _require(
        layer_costs[1] == selected_compute["stages_ns"],
        "L1 aggregate costs differ from selected timeline kernels",
    )
    dma = _one(
        operator_rows,
        lambda row: (
            row["mode"] == "dense_prefetch"
            and row["phase"] == "extend_annotated"
            and row["layer"] == "shared"
            and row["stage"] == "dense_history_prefetch_layer_0"
        ),
        "previous-layer complete H2D operator",
    )
    _require(
        int(dma["scope_count"]) == int(dma["metadata_call_count"]) == int(dma["memcpy_count"]) == 1
        and dma["metadata_call_count_matches"] == "True"
        and int(dma["memcpy_ns"]) > 0,
        "previous-layer H2D is not one complete DMA",
    )
    dma_analysis = _one(
        analysis["operators_by_layer"],
        lambda row: all(row[key] == dma[key] for key in ("mode", "phase", "layer", "stage")),
        "previous-layer H2D analysis",
    )
    _require(
        dma_analysis["memcpy_ns"] == int(dma["memcpy_ns"]) and dma_analysis["memcpy_count"] == 1,
        "previous-layer H2D CSV differs from analysis",
    )
    dma_call = _one(
        calls,
        lambda row: all(row[key] == dma[key] for key in ("mode", "phase", "layer", "stage")),
        "previous-layer H2D call ledger entry",
    )
    return {
        "previous_layer": 0,
        "compute": {
            "source_scheme": "hbm",
            "stages_ns": layer_costs[0],
            "total_ns": sum(layer_costs[0].values()),
            "source_rows": layer_sources[0],
        },
        "h2d": {
            "source_scheme": "dense_prefetch",
            "layer": 0,
            "duration_ns": int(dma["memcpy_ns"]),
            "transport": "host_to_device_dma",
            "source_row": dma,
            "operator_call": dma_call,
        },
        "audit": {
            "one_extend_call_per_compute_operator": True,
            "operator_csv_matches_analysis_and_kernel_inventory": True,
            "adjacent_layer_query_shapes_match": True,
            "l1_aggregate_matches_selected_timeline": True,
            "previous_h2d_is_one_complete_memcpy": True,
        },
        "boundary": (
            "Use measured L0 compute and complete L0 DMA to place L1 prefetch immediately after "
            "L0 fetch completion; retain L1 projection start as t=0. L0 supplies scheduling "
            "context only, not a repeated-layer or full-model latency estimate. The H2D scope's "
            "control kernels are excluded. L2 prefetch starts when the L1 fetch completes."
        ),
    }


def _transfer(row, *, records, record_bytes, transport, byte_basis):
    activity = _activity(row)
    byte_count = records * record_bytes
    if row["kind"] == "memcpy":
        _require(
            int(row["bytes"]) == byte_count, "native memcpy bytes differ from expected records"
        )
    return {
        "source_scheme": row["method"],
        "layer": _layer(row),
        "duration_ns": activity["duration_ns"],
        "bytes": byte_count,
        "records": records,
        "byte_basis": byte_basis,
        "transport": transport,
        "activity": activity,
        "duration_boundary": "Complete raw_start_ns/raw_end_ns; never clipped display coordinates.",
    }


def _echo(rows, timeline, calls, *, serial_counters, record_bytes, peaks):
    """Retain the measured indivisible fused indexer and its residual transfer."""
    selected = [
        row
        for row in rows
        if row["method"] == "echo" and row["phase"] == "extend" and row["layer"] == "layer_1"
    ]
    fused_row = _one(
        selected,
        lambda row: row["lane"] == "Compute + IO" and row["kind"] == "kernel",
        "ECHO fused indexer/prefetch kernel",
    )
    cleanup_row = _one(
        selected,
        lambda row: row["kind"] == "kernel" and row["name"].startswith("echo_native::clean("),
        "ECHO causal score-mask kernel",
    )
    recall_row = _one(
        selected,
        lambda row: row["kind"] == "kernel" and row["io_kind"] == "mapped_host_gather",
        "ECHO residual recall kernel",
    )
    fused_call = _one(
        calls,
        lambda call: call["nvtx"] == fused_row["scope_label"],
        "ECHO fused indexer call ledger entry",
    )
    hbm_call = _one(
        calls,
        lambda call: (
            (call["mode"], call["phase"], call["layer"], call["stage"])
            == ("hbm", "extend_annotated", "layer_1", "indexer_qk")
        ),
        "HBM extend indexer call ledger entry",
    )
    _require(
        fused_call["stage"] == "indexer_fused"
        and fused_call["useful_flops"] == hbm_call["useful_flops"]
        and fused_call["precision"] == hbm_call["precision"] == "FP8"
        and (fused_call["query_start"], fused_call["query_tokens"], fused_call["kv_tokens"])
        == (65536, 128, 65664),
        "ECHO fused work differs from selected standalone HBM indexer work",
    )
    counters = timeline["selected_layer_phase_cache_counters"]
    prefetched, recalled = counters["prefetched_records"], counters["recalled_records"]
    _require(
        counters["record_bytes"] == record_bytes
        and counters["written_records"] == 128
        and counters["prefetch_capacity_failures"] == counters["capacity_splits"] == 0
        and prefetched + recalled == serial_counters["recalled_records"]
        and counters["host_to_device_bytes"]
        == (prefetched + recalled) * record_bytes
        == serial_counters["host_to_device_bytes"]
        and counters["resident_selection_records"] - 128 == prefetched
        and counters["selection_records"]
        == counters["resident_selection_records"] + recalled
        == serial_counters["selection_records"],
        "ECHO prefetch/residual history coverage does not reconcile with serial sparse",
    )
    fused = _transfer(
        fused_row,
        records=prefetched,
        record_bytes=record_bytes,
        transport="fused_indexer_mapped_host_prefetch",
        byte_basis="ECHO L1 complete-extend prefetched_records * record_bytes.",
    )
    fused.update(
        useful_flops=fused_call["useful_flops"],
        precision=fused_call["precision"],
        flops_by_precision={fused_call["precision"]: fused_call["useful_flops"]},
        ideal_compute_ns=fused_call["useful_flops"] / (peaks[fused_call["precision"]] * 1000),
        calls=[fused_call],
        boundary=(
            "Whole measured fused kernel; computation, transfer and internal bookkeeping "
            "are inseparable. Payload bytes / whole-kernel duration is not isolated link bandwidth."
        ),
    )
    cleanup = _activity(cleanup_row)
    recall = _transfer(
        recall_row,
        records=recalled,
        record_bytes=record_bytes,
        transport="mapped_host_gather",
        byte_basis="ECHO L1 complete-extend recalled_records * record_bytes.",
    )
    _require(
        fused["activity"]["end_ns"]
        <= cleanup["start_ns"]
        < cleanup["end_ns"]
        <= recall["activity"]["start_ns"],
        "source ECHO fused/causal-mask/recall order differs",
    )
    _require(
        timeline["layer"] == 1
        and timeline["occurrence"] == -1
        and all(
            timeline["origin_ns"] <= activity["start_ns"] < activity["end_ns"] <= timeline["end_ns"]
            for activity in (fused["activity"], cleanup, recall["activity"])
        ),
        "selected ECHO activity is clipped or belongs to another layer occurrence",
    )
    return {
        "fused": fused,
        "cleanup": {
            "duration_ns": cleanup["duration_ns"],
            "activity": cleanup,
            "useful_flops": None,
            "boundary": (
                "Causal score masking after the fused indexer; retained as nonmatrix computation "
                "despite the source CSV generic GPU-control lane. No cache bookkeeping time added."
            ),
        },
        "recall": recall,
        "counters": counters,
        "coverage_fraction": prefetched / (prefetched + recalled),
        "audit": {
            "fused_useful_work_equals_hbm_indexer": True,
            "prefetch_and_residual_bytes_equal_serial_sparse": True,
            "resident_selection_minus_candidate_equals_prefetched_history": True,
            "coverage_basis": "Prefetched history / exact consumed history union; excludes 128 candidate records.",
        },
    }


def build_inputs(
    report_directory=DEFAULT_REPORT, *, run_id=DEFAULT_RUN_ID, operator_calls=DEFAULT_OPERATOR_CALLS
):
    """Read accepted report data and its hash-bound call ledger without requiring CUDA."""
    report = Path(report_directory).resolve(strict=True)
    manifest = json.loads((report / "publication_manifest.json").read_text())
    for name in SOURCE_FILES:
        _require(digest(report / name) == manifest[name], f"published source hash mismatch: {name}")
    summary = json.loads((report / "summary.json").read_text())
    analysis = json.loads((report / "analysis.json").read_text())
    acceptance = json.loads((report / "run_acceptance.json").read_text())
    timelines = json.loads((report / "timeline_summary.json").read_text())
    dense_sql = json.loads((report / "dense_overlap_sql_audit.json").read_text())
    ledger_path = Path(operator_calls).resolve(strict=True)
    _require(
        digest(ledger_path) == summary["input_sha256"]["operator_calls.json"],
        "operator call ledger hash differs from the accepted MFU report",
    )
    ledger = json.loads(ledger_path.read_text())
    _require(ledger["run_id"] == PROFILE_RUN_ID, "operator call ledger run differs")
    _require(
        summary["dense_peaks_tflops"]
        == analysis["dense_peaks_tflops"]
        == {"FP8": 1979.0, "BF16": 989.5, "FP32": 67.0},
        "unexpected dense precision peaks",
    )
    work_arguments = {
        "peaks": analysis["dense_peaks_tflops"],
        "peak_reference": analysis["peak_reference"],
    }
    with (report / "timeline_activities.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (report / "operator_mfu_by_layer.csv").open(newline="") as stream:
        operator_rows = list(csv.DictReader(stream))
    _require(summary["profile_run_id"] == PROFILE_RUN_ID, "unexpected source profile run")
    _require(
        acceptance["run_results"]["profile"]["run_id"] == PROFILE_RUN_ID, "acceptance run differs"
    )
    _require(
        all(acceptance["execution_identity_fields_equal"].values()), "source identities differ"
    )
    _require(
        all(row["bitwise_equal"] for row in acceptance["saved_profile_tensor_checks"]),
        "source output comparison did not pass",
    )
    config = summary["configuration"]
    _require(
        (
            config["prefix_tokens"],
            config["extend_tokens"],
            config["chunk_size"],
            config["extend_residency"],
        )
        == (65536, 128, 1024, "cold"),
        "unsupported source workload",
    )
    hbm = {phase: _timeline(timelines, "hbm", phase) for phase in ("prefill", "extend")}
    for phase, timeline in hbm.items():
        _require(
            timeline["layer"] == 1 and timeline["occurrence"] == -1, "wrong selected layer/chunk"
        )
        coverage = _one(
            summary["query_coverage"],
            lambda row, phase=phase: (
                row["method"] == "hbm"
                and row["phase"] == phase
                and row["layer"] == 1
                and row["operator"] == "indexer"
            ),
            f"{phase} HBM query coverage",
        )
        _require(
            (coverage["queries"], coverage["calls"])
            == ((65536, 64) if phase == "prefill" else (128, 1)),
            "source query coverage mismatch",
        )
    sparse = _timeline(timelines, "serial_sparse", "extend")["selected_layer_phase_cache_counters"]
    dense = _timeline(timelines, "dense_prefetch", "extend")
    record_bytes = sparse["record_bytes"]
    _require(record_bytes == 1152, "unexpected model KV record width")
    d2h_row = _one(
        rows,
        lambda row: (
            row["method"] == "serial_sparse"
            and row["phase"] == "prefill"
            and row["layer"] == "layer_1"
            and row["kind"] == "memcpy"
            and row["io_kind"] == "Device-to-Host"
            and row["bytes"] == str(1024 * record_bytes)
        ),
        "selected prefill main-KV writeback",
    )
    sparse_row = _one(
        rows,
        lambda row: (
            row["method"] == "serial_sparse"
            and row["phase"] == "extend"
            and row["layer"] == "layer_1"
            and row["io_kind"] == "mapped_host_gather"
        ),
        "selected sparse history gather",
    )
    _require(
        sparse["host_to_device_bytes"] == sparse["recalled_records"] * record_bytes,
        "sparse cache byte counters disagree",
    )
    transfers = {
        "sparse_h2d": _transfer(
            sparse_row,
            records=sparse["recalled_records"],
            record_bytes=record_bytes,
            transport="mapped_host_gather",
            byte_basis="L1 complete-extend scoped cache counters.",
        )
    }
    echo = _echo(
        rows,
        _timeline(timelines, "echo", "extend"),
        ledger["calls"],
        serial_counters=sparse,
        record_bytes=record_bytes,
        peaks=analysis["dense_peaks_tflops"],
    )
    for layer, key, counters_key in (
        (1, "dense_h2d", "selected_layer_phase_cache_counters"),
        (2, "dense_next_h2d", "next_layer_phase_cache_counters"),
    ):
        row = _one(
            rows,
            lambda item, layer=layer: (
                item["method"] == "dense_prefetch"
                and item["phase"] == "extend"
                and item["stage"] == f"dense_history_prefetch_layer_{layer}"
                and item["kind"] == "memcpy"
                and item["io_kind"] == "Host-to-Device"
            ),
            f"complete dense L{layer} H2D DMA",
        )
        counters = dense[counters_key]
        _require(
            counters["host_to_device_bytes"] == 65536 * record_bytes,
            "dense history counters do not cover full H",
        )
        transfers[key] = _transfer(
            row,
            records=65536,
            record_bytes=record_bytes,
            transport="host_to_device_dma",
            byte_basis="Native H2D memcpy bytes equal per-layer counters and H * record_bytes.",
        )
    next_raw = _one(
        dense["next_layer_prefetch_activities"],
        lambda row: row["kind"] == "memcpy",
        "complete next-layer DMA metadata",
    )
    next_activity = transfers["dense_next_h2d"]["activity"]
    _require(
        next_raw["start"] == next_activity["start_ns"]
        and next_raw["end"] == next_activity["end_ns"],
        "next DMA endpoints mismatch",
    )
    sql_extend = _one(dense_sql, lambda row: row["phase"] == "extend", "dense extend SQL audit")
    _require(
        sql_extend["prefetch_full_union_ns"] == transfers["dense_next_h2d"]["duration_ns"]
        and sql_extend["prefetch_h2d_memcpy_bytes"] == transfers["dense_next_h2d"]["bytes"],
        "next DMA does not match independent native SQL audit",
    )
    source_captures = [
        capture
        for capture in analysis["captures"]
        if (capture["mode"], capture["phase"])
        in {
            ("hbm", "prefill_annotated"),
            ("hbm", "extend_annotated"),
            ("serial_sparse", "prefill_annotated"),
            ("serial_sparse", "extend_annotated"),
            ("dense_prefetch", "extend_annotated"),
            ("echo", "extend_annotated"),
        }
    ]
    _require(len(source_captures) == 6, "missing source capture")
    _require(
        all(
            capture["audit"]["kernel_count_and_time_conserved"]
            and capture["graph_attribution"]["every_replay_gpu_node_verified"]
            for capture in source_captures
        ),
        "source attribution audit failed",
    )
    source_files = [report / name for name in (*SOURCE_FILES, "publication_manifest.json")]
    source_files.append(ledger_path)
    source_files.append(Path(__file__).resolve())
    extend_compute = _compute(rows, hbm["extend"], ledger["calls"], **work_arguments)
    dense_pipeline_context = _dense_pipeline_context(
        operator_rows, analysis, ledger["calls"], extend_compute
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "source": {
            "profile_run_id": PROFILE_RUN_ID,
            "profile_source_sha256": summary["source_sha256"],
            "canonical_execution_identity_sha256": acceptance[
                "canonical_execution_identity_sha256"
            ],
            "files": [
                {
                    "path": str(path.relative_to(REPOSITORY))
                    if path.is_relative_to(REPOSITORY)
                    else str(path),
                    "sha256": digest(path),
                }
                for path in source_files
            ],
            "raw_captures": [
                {
                    "phase": row["phase"],
                    "scheme": row["mode"],
                    "sqlite": row["sqlite"],
                    "sqlite_sha256": row["sqlite_sha256"],
                }
                for row in source_captures
            ],
            "source_boundary": (
                "Accepted MFU real checkpoint layers 0-2; take physical L1, final prefill chunk "
                "and cold extend, with adjacent L0 costs for dense prefetch scheduling. These "
                "shape-matched primitives are projected into a motivation "
                "single-layer simulation, not measured C10 GR request latency. Persistent-extend "
                "candidate D2H is deliberately omitted for the simulated discard semantics."
            ),
        },
        "workload": {
            "physical_layer": 1,
            "history_tokens": 65536,
            "candidate_tokens": 128,
            "pool_tokens": config["sparse_pool_tokens"],
            "host_arena_tokens": 65664,
            "prefill_request_id": None,
            "prefill_chunk": 63,
            "prefill_query_tokens": 1024,
            "prefill_prefix_tokens": 64512,
            "extend_request_id": None,
            "record_bytes": record_bytes,
            "hardware": summary["hardware"],
            "config": config,
            "source_model_scope": "Real checkpoint layers 0-2, sequential hidden/residual propagation.",
            "simulation_scope": "Shape-matched single layer; transient candidate discard; no C10 token identity claim.",
        },
        "prefill": {
            "compute": _compute(rows, hbm["prefill"], ledger["calls"], **work_arguments),
            "d2h": _transfer(
                d2h_row,
                records=1024,
                record_bytes=record_bytes,
                transport="device_to_host_memcpy",
                byte_basis="Selected chunk native memcpy bytes.",
            ),
        },
        "extend": {
            "compute": extend_compute,
            **transfers,
            "echo": echo,
            "dense_pipeline_context": dense_pipeline_context,
        },
        "audit": {
            "published_manifest_hashes_match": True,
            "operator_call_ledger_hash_matches_accepted_summary": True,
            "source_output_and_attribution_checks_passed": True,
            "complete_dense_dma_matches_native_sql_audit": True,
            "gather_durations_use_full_raw_intervals": True,
            "candidate_d2h_bytes": 0,
            "source_persistent_candidate_d2h_bytes_per_layer": sparse["device_to_host_bytes"],
            "excluded": [
                "Every CPU API/scope duration and GPU idle gap.",
                "Cache/control kernels, D2D memcpy and memset.",
                "Source persistent candidate D2H: 147456 bytes per layer; simulated candidates are discarded.",
                "Embedding, LM head and other-layer outputs; L0 costs only establish L1 prefetch timing.",
            ],
            "limitations": [
                "One invasive profile sample per selected method/phase; no confidence interval.",
                "Independent compute/IO scheduling is ideal; sparse mapped-host gather uses SM resources.",
                "Final prefill chunk is not a measurement or extrapolation of the whole history.",
                "MFU real-three-layer inputs differ from motivation C10 requests and transient execution.",
                "Fused ECHO is one indivisible measured kernel; its internal compute, IO and bookkeeping timing cannot be separated.",
                "Dense uses adjacent L0 scheduling context and serial H2D transfers; the plotted L1 window starts at L1 projection, with an L2 transfer prefix at its end.",
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-directory", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--operator-calls", type=Path, default=DEFAULT_OPERATOR_CALLS)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        REPOSITORY / "experiments/deepseek_v32_motivation/output/data" / args.run_id / "inputs.json"
    )
    inputs = build_inputs(
        args.report_directory, run_id=args.run_id, operator_calls=args.operator_calls
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(inputs, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
