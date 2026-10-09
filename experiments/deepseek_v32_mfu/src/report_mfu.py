"""Verify and publish four-method prefill/extend MFU and single-layer timelines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from itertools import pairwise
from pathlib import Path

from experiments.deepseek_v32_mfu.src import timeline
from experiments.deepseek_v32_mfu.src.execution_utilization import precision_normalized_utilization
from experiments.deepseek_v32_mfu.src.operator_report import analyze_captures, read_calls
from experiments.deepseek_v32_mfu.src.publish_layers import STAGES
from experiments.deepseek_v32_mfu.src.run_contract import METHODS, benchmark_view, digest


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, sort_keys=True)
                if key == "actual_io" and value is not None
                else value
                for key, value in row.items()
            }
            for row in rows
        )


def _verify_native_file(item):
    if (
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or not Path(item["path"]).is_absolute()
        or type(item.get("bytes")) is not int
        or item["bytes"] <= 0
        or not isinstance(item.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
    ):
        raise ValueError("recorded loaded native artifact lacks a valid path/hash/byte identity")
    path = Path(item["path"])

    def identity(value):
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size != item["bytes"]
            or actual_sha256 != item["sha256"]
            or identity(before) != identity(after)
            or identity(after) != identity(path.stat())
        ):
            raise ValueError(f"recorded loaded native artifact changed: {path}")


def audit_local_native_artifacts(results, *, require_offload=True):
    """Require complete, unchanged mapped-DSO evidence from all three runs."""
    if set(results) != {"check", "bench", "profile"}:
        raise ValueError("local native verification requires check, bench and profile runs")
    recorded = {}
    for run, result in results.items():
        snapshots = []
        for field in ("execution_runtime_artifacts", "flashinfer_runtime_artifacts"):
            runtime = result.get(field)
            entries = runtime.get("local_native_jit") if isinstance(runtime, dict) else None
            if not isinstance(entries, list):
                raise ValueError(  # noqa: TRY004 - invalid or missing serialized evidence
                    f"{run} lacks {field}['local_native_jit']; formal native-bound publication "
                    "requires before/after mapped-library evidence from the original run. "
                    "Older runs cannot acquire this evidence retrospectively."
                )
            counts, paths = Counter(), set()
            for entry in entries:
                if (
                    not isinstance(entry, dict)
                    or set(entry) != {"name", "category", "library"}
                    or not isinstance(entry["name"], str)
                    or re.fullmatch(r"(?:lib)?cxldsagr_[^/]+\.so(?:\.[0-9]+)*", entry["name"])
                    is None
                    or entry["category"]
                    not in ("echo_indexer", "record_transfer", "other_local_native")
                    or not isinstance(entry["library"], dict)
                    or set(entry["library"]) != {"path", "sha256", "bytes"}
                    or not isinstance(entry["library"]["path"], str)
                    or Path(entry["library"]["path"]).name != entry["name"]
                ):
                    raise ValueError(f"{run} has a malformed local native library identity")
                name = entry["name"].removeprefix("lib")
                category = "other_local_native"
                for stem, expected in (
                    ("cxldsagr_echo_indexer", "echo_indexer"),
                    ("cxldsagr_kv_transfer", "record_transfer"),
                ):
                    if re.fullmatch(stem + r"(?:_[^/]+)?\.so(?:\.[0-9]+)*", name):
                        category = expected
                if entry["category"] != category:
                    raise ValueError(f"{run} has an incorrect local native library category")
                path = entry["library"]["path"]
                if path in paths:
                    raise ValueError(f"{run} repeats a local native library path")
                paths.add(path)
                counts[category] += 1
            if require_offload and any(
                counts[category] != 1 for category in ("echo_indexer", "record_transfer")
            ):
                raise ValueError(
                    f"{run} requires exactly one echo_indexer and one record_transfer library "
                    f"in {field}['local_native_jit'] for formal native-bound publication"
                )
            snapshots.append(entries)
        if snapshots[0] != snapshots[1]:
            raise ValueError(f"{run} local native libraries changed between before/after records")
        recorded[run] = snapshots[0]
    if any(entries != recorded["profile"] for entries in recorded.values()):
        raise ValueError("check, bench and profile have different local native libraries")
    native_files = []
    for entry in recorded["profile"]:
        _verify_native_file(entry["library"])
        native_files.append(
            {
                "name": entry["name"],
                "category": entry["category"],
                "kind": "library",
                "source": "local_native_jit",
                **entry["library"],
            }
        )
    return native_files


def _independent_runs(profile_directory, profile):
    from experiments.deepseek_v32_mfu.src.run_contract import validated_receipt

    receipt = validated_receipt(profile)
    directories = {
        "check": Path(receipt["receipt_path"]).parent,
        "bench": Path(profile["benchmark"]["directory"]),
        "profile": profile_directory,
    }
    results = {
        name: json.loads((directory / "result.json").read_text())
        for name, directory in directories.items()
    }
    return receipt, directories, results


def audit_graph_shapes(result, calls, templates):
    """Check every layer's captured shapes and each phase's actual replay counts."""
    from experiments.deepseek_v32_motivation.src.graph_instrumentation import REPLAY_PATTERN

    chunks = {}
    for phase, tokens, size in (
        ("prefill", result["prefix_tokens"], result["chunk_size"]),
        ("extend", result["extend_tokens"], result["extend_chunk_size"] or result["extend_tokens"]),
    ):
        chunks[phase] = [min(size, tokens - start) for start in range(0, tokens, size)]
    shapes = sorted({size for sizes in chunks.values() for size in sizes})
    expected_templates = {
        (part, layer, size)
        for part in ("projection", "finish")
        for layer in range(result["num_layers"])
        for size in shapes
    }
    actual_templates = {
        (item["part"], item["layer"], item["queries"]): item for item in templates["graphs"]
    }
    if (
        len(actual_templates) != len(templates["graphs"])
        or set(actual_templates) != expected_templates
    ):
        raise ValueError("captured graph shapes do not cover each checkpoint layer exactly")
    expected = Counter(
        (method, phase, layer, part, size)
        for method in METHODS
        for phase, sizes in chunks.items()
        for size in sizes
        for layer in range(result["num_layers"])
        for part in ("projection", "finish")
    )
    observed = Counter()
    for call in calls:
        if not call.get("graph_replay"):
            continue
        match = REPLAY_PATTERN.fullmatch(call["stage"])
        if match is None:
            raise ValueError("graph replay has no captured-shape stage")
        part, layer, size = match[1], int(match[2]), int(match[3])
        template = actual_templates.get((part, layer, size))
        if (
            template is None
            or call["layer"] != f"layer_{layer}"
            or call["graph_capture_id"] != template["capture_graph_id"]
            or call["graph_id"] != template["executable_graph_id"]
            or set(call["graph_gpu_node_ids"]) != set(template["gpu_node_ids"])
        ):
            raise ValueError("graph replay differs from its layer/shape capture template")
        observed[call["mode"], call["phase"].removesuffix("_annotated"), layer, part, size] += 1
    if observed != expected:
        raise ValueError("graph replay counts do not match every layer and query chunk")
    return {
        "query_shapes": shapes,
        "capture_template_count": len(actual_templates),
        "every_layer_shape_and_phase_replay_verified": True,
        "phase_query_chunks": chunks,
        "replay_counts": [
            {
                "method": method,
                "phase": phase,
                "layer": layer,
                "part": part,
                "queries": size,
                "count": count,
            }
            for (method, phase, layer, part, size), count in sorted(observed.items())
        ],
    }


def other_gpu_process_observations(rows, selected_uuid):
    """Retain other-device activity without inferring continuous device isolation."""
    observations = []
    for row in rows:
        devices = {}
        for fields in csv.reader(row["gpu_utilization"].splitlines()):
            if not fields:
                continue
            index, uuid, utilization = (value.strip() for value in fields[:3])
            devices[uuid] = (int(index), float(utilization.removesuffix("%").strip()))
        if selected_uuid not in devices:
            raise ValueError("observer lacks selected-device utilization evidence")
        for fields in csv.reader(row["stdout"].splitlines()):
            if not fields:
                continue
            pid, uuid = (value.strip() for value in fields[:2])
            if uuid == selected_uuid:
                continue
            if uuid not in devices:
                raise ValueError("observed process has no matching device utilization sample")
            index, utilization = devices[uuid]
            observations.append(
                {
                    "time_utc": row["time_utc"],
                    "pid": int(pid),
                    "gpu_uuid": uuid,
                    "gpu_index": index,
                    "sampled_gpu_utilization_percent": utilization,
                }
            )
    return observations


def observer_acceptance_text(observation):
    statuses = "; ".join(
        f"{name}: child {item['child_exit_code']}, observer wrapper {item['observer_wrapper_exit_code']}"
        for name, item in observation.items()
    )
    audited = [name for name, item in observation.items() if item["reconciliation"] is not None]
    reconciled = [name for name in audited if observation[name]["observer_wrapper_exit_code"] != 0]
    reconciliation_note = (
        f" Independent raw-evidence audits are retained for {', '.join(audited)}."
        if audited
        else ""
    )
    reconciliation_note += (
        f" Original observer failures and independent reconciliations are retained for {', '.join(reconciled)}."
        if reconciled
        else " No observer reconciliation was required."
    )
    return (
        "[Run acceptance](run_acceptance.json) binds check, benchmark, profile, saved outputs, "
        f"native identities and raw observers. Exit statuses are {statuses}."
        + reconciliation_note
        + " No unresolved selected-GPU foreign process remains. Other-device process observations "
        "and their sampled utilization are retained separately. Discrete observations do not prove "
        "continuous device isolation or exclusive CPU use."
    )


def audit_dense_sql(path, summary, rows):
    """Verify native compute/transport rows, then independently intersect their times."""
    compute = [
        row
        for row in rows
        if row["kind"] == "kernel"
        and row["lane"] == "Compute"
        and row["layer"] == f"layer_{summary['layer']}"
    ]
    intervals = {
        "compute": [(row["raw_start_ns"], row["raw_end_ns"]) for row in compute],
        "prefetch": summary["next_layer_prefetch_intervals_ns"],
    }
    source = {
        "compute": [
            {
                "kind": row["kind"],
                "start": row["raw_start_ns"],
                "end": row["raw_end_ns"],
                "device_id": row["device_id"],
                "stream_id": row["stream"],
                "process": row["process"],
                "correlation": row["correlation"],
            }
            for row in compute
        ],
        "prefetch": summary["next_layer_prefetch_activities"],
    }
    if [(item["start"], item["end"]) for item in source["prefetch"]] != [
        tuple(interval) for interval in intervals["prefetch"]
    ]:
        raise ValueError("prefetch activity records differ from reported full intervals")
    native = []
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        kernel_table = next(
            name
            for name in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")
            if name in tables
        )
        copy_table = next(
            (
                name
                for name in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMCPY2")
                if name in tables
            ),
            None,
        )
        strings = (
            dict(connection.execute('SELECT id, value FROM "StringIds"'))
            if "StringIds" in tables
            else {}
        )
        copy_labels = (
            dict(connection.execute('SELECT id, label FROM "ENUM_CUDA_MEMCPY_OPER"'))
            if "ENUM_CUDA_MEMCPY_OPER" in tables
            else {}
        )
        for group, activities in source.items():
            for activity in activities:
                table = copy_table if activity["kind"] == "memcpy" else kernel_table
                if table is None:
                    raise ValueError("prefetch memcpy has no native CUPTI table")
                matches = connection.execute(
                    f'SELECT rowid AS native_row_id, * FROM "{table}" WHERE start=? AND end=? AND deviceId=? AND streamId=? AND correlationId=?',
                    (
                        activity["start"],
                        activity["end"],
                        activity["device_id"],
                        activity["stream_id"],
                        activity["correlation"],
                    ),
                ).fetchall()
                if len(matches) != 1:
                    raise ValueError("timeline activity does not identify one native CUPTI row")
                row = dict(matches[0])
                if activity["process"] is not None and row.get("globalPid") != activity["process"]:
                    raise ValueError("native activity process differs from timeline")
                launch = None
                if activity["kind"] == "memcpy":
                    label = copy_labels.get(row.get("copyKind"), "")
                    if label.lower() not in {"host-to-device", "unified host-to-device"}:
                        raise ValueError("dense prefetch copy is not native H2D")
                    if row.get("bytes") != activity["bytes"] or not row["bytes"] > 0:
                        raise ValueError("dense prefetch memcpy byte count differs from native row")
                    api = activity["api"]
                    scope = summary["next_layer_prefetch_scope"]
                    if (
                        api is None
                        or api["source"] not in {"RUNTIME", "DRIVER"}
                        or not scope["start"] <= api["start"] < scope["end"]
                        or scope["thread"] != api["thread"]
                    ):
                        raise ValueError("dense H2D launch is not owned by its prefetch scope")
                    api_table = "CUPTI_ACTIVITY_KIND_" + api["source"]
                    api_rows = connection.execute(
                        f'SELECT rowid AS native_row_id, * FROM "{api_table}" WHERE start=? AND end=? AND globalTid=? AND correlationId=?',
                        (api["start"], api["end"], api["thread"], activity["correlation"]),
                    ).fetchall()
                    if len(api_rows) != 1:
                        raise ValueError("dense H2D lacks a unique correlated native launch API")
                    launch = {"table": api_table, **dict(api_rows[0])}
                    raw_name = launch.get("name") or launch.get("nameId")
                    name = raw_name if isinstance(raw_name, str) else strings.get(raw_name, "")
                    if name != api["name"] or not re.match(
                        r"^(?:cudaMemcpyAsync|cuMemcpy(?:HtoD)?Async)(?:_|$)", name
                    ):
                        raise ValueError("dense H2D was not launched by an asynchronous memcpy API")
                    launch["resolved_name"] = name
                native.append(
                    {
                        "group": group,
                        "kind": activity["kind"],
                        "table": table,
                        "launch": launch,
                        **{
                            key: row.get(key)
                            for key in (
                                "native_row_id",
                                "start",
                                "end",
                                "deviceId",
                                "globalPid",
                                "streamId",
                                "correlationId",
                                "graphId",
                                "graphNodeId",
                                "demangledName",
                                "copyKind",
                                "bytes",
                            )
                        },
                    }
                )
    events = defaultdict(lambda: [0, 0])
    for index, group in enumerate(("compute", "prefetch")):
        for start, end in intervals[group]:
            events[start][index] += 1
            events[end][index] -= 1
    active, previous, overlap = [0, 0], None, 0
    for timestamp, change in sorted(events.items()):
        if previous is not None and active[0] > 0 and active[1] > 0:
            overlap += timestamp - previous
        active[0] += change[0]
        active[1] += change[1]
        previous = timestamp
    if active != [0, 0] or abs(overlap / 1e6 - summary["next_layer_prefetch_overlap_ms"]) > 1e-12:
        raise ValueError("independent native overlap arithmetic differs from timeline")
    h2d_bytes = sum(
        row["bytes"] for row in native if row["group"] == "prefetch" and row["kind"] == "memcpy"
    )
    if h2d_bytes != summary["next_layer_prefetch_h2d_memcpy_bytes"]:
        raise ValueError("native H2D bytes differ from timeline")
    return {
        "sqlite": str(path),
        "sqlite_sha256": digest(path),
        "method": "dense_prefetch",
        "phase": summary["phase"],
        "tables": sorted({row["table"] for row in native}),
        "source_query": "Match native activity by start/end/device/stream/correlation; match H2D asynchronous launch by start/end/thread/correlation",
        "native_activity_rows": native,
        "compute_kernel_count": len(compute),
        "prefetch_activity_count": len(intervals["prefetch"]),
        "prefetch_h2d_memcpy_bytes": h2d_bytes,
        "prefetch_streams": sorted(
            {row["streamId"] for row in native if row["group"] == "prefetch"}
        ),
        "prefetch_full_union_ns": timeline._union_ns(intervals["prefetch"]),
        "prefetch_full_span_ns": (
            max(end for _, end in intervals["prefetch"])
            - min(start for start, _ in intervals["prefetch"])
        )
        if intervals["prefetch"]
        else 0,
        "overlap_ns": overlap,
        "verification": "native row timestamps and independent event-sweep intersection agree",
    }


def audit_dense_transport(result, summary, phase_counters):
    declared = result.get("dense_history_transport", {}).get("dense_prefetch")
    if declared is None:
        return None
    if declared != "cuda_memcpy_async_contiguous":
        raise ValueError("unknown dense history transport declaration")
    totals = summary["phase_dense_prefetch_totals"]
    if (
        set(totals) != set(range(result["num_layers"]))
        or len(phase_counters) != result["num_layers"]
    ):
        raise ValueError("dense DMA capture lacks prefetch scopes for every layer")
    for layer, counters in enumerate(phase_counters):
        measured = totals[layer]
        if measured["mapped_gather_count"]:
            raise ValueError("declared dense DMA run contains mapped-host gather transport")
        if measured["h2d_memcpy_bytes"] != counters["host_to_device_bytes"]:
            raise ValueError("dense DMA phase bytes differ from per-layer cache counters")
    return {
        "declared": declared,
        "phase": summary["phase"],
        "per_layer": totals,
        "all_layer_phase_bytes_match_cache_counters": True,
        "mapped_host_gather_count": 0,
    }


def audit_work(result, calls, peaks, *, methods=METHODS):
    """Require full query coverage and equal useful work for every method."""
    expected_stages = set(STAGES) - {"indexer_qk", "indexer_fused", "lm_head"}
    totals, rows, intervals = {}, [], []
    for phase in ("prefill", "extend"):
        start = 0 if phase == "prefill" else result["prefix_tokens"]
        tokens = result["prefix_tokens"] if phase == "prefill" else result["extend_tokens"]
        for method in methods:
            selected = [
                call
                for call in calls
                if (call["mode"], call["phase"]) == (method, phase + "_annotated")
            ]
            matrix = [call for call in selected if call["useful_flops"] is not None]
            for layer in range(3):
                layer_calls = [call for call in matrix if call["layer"] == f"layer_{layer}"]
                stages = {call["stage"] for call in layer_calls}
                indexer = stages & {"indexer_qk", "indexer_fused"}
                if not indexer or stages != expected_stages | indexer:
                    raise ValueError(f"incomplete matrix ledger: {method}/{phase}/layer_{layer}")
                for category in ({"indexer_qk", "indexer_fused"}, {"mla_qk_pv"}):
                    cursor = start
                    chunks = sorted(
                        (call for call in layer_calls if call["stage"] in category),
                        key=lambda call: call["query_start"],
                    )
                    for call in chunks:
                        if call["query_start"] != cursor or call["query_tokens"] < 1:
                            raise ValueError("query ledger has a gap, overlap, or empty interval")
                        cursor += call["query_tokens"]
                    if cursor != start + tokens:
                        raise ValueError("matrix ledger does not cover every query")
                    intervals.append(
                        {
                            "method": method,
                            "phase": phase,
                            "layer": layer,
                            "operator": "indexer" if "indexer_qk" in category else "mla",
                            "queries": tokens,
                            "calls": len(chunks),
                        }
                    )
            shared = [call for call in matrix if call["layer"] == "shared"]
            if len(shared) != 1 or shared[0]["stage"] != "lm_head":
                raise ValueError("expected exactly one shared last-token LM head")
            work = defaultdict(int)
            for call in matrix:
                if call["layer"] not in {"layer_0", "layer_1", "layer_2", "shared"}:
                    raise ValueError("matrix work lies outside the first three layers")
                stage = (
                    "indexer" if call["stage"] in {"indexer_qk", "indexer_fused"} else call["stage"]
                )
                work[call["layer"], stage, call["precision"]] += call["useful_flops"]
            totals[method, phase] = dict(work)
            utilization = precision_normalized_utilization(
                selected,
                result["measurements"][method][phase + "_samples_ms"],
                peaks_tflops=peaks,
                scope=result["scope"],
            )
            rows.append(
                {
                    "method": method,
                    "phase": phase,
                    "wall_median_ms": utilization["wall_median_ms"],
                    "mfu_percent": utilization["utilization_at_median_wall_percent"],
                    "ideal_compute_ms": utilization["ideal_compute_ms"],
                    "details": utilization,
                }
            )
        if any(totals[method, phase] != totals[methods[0], phase] for method in methods[1:]):
            raise ValueError("selected methods have different useful matrix work")
    return rows, intervals


def plot_mfu(rows, operators, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = ("#0072B2", "#8F4C9B", "#4C956C", "#D18A21")
    hatches = ("", "//", "..", "xx")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, method in enumerate(METHODS):
        chosen = [
            next(row for row in rows if row["method"] == method and row["phase"] == phase)
            for phase in ("prefill", "extend")
        ]
        x = np.arange(2) + (index - 1.5) * 0.19
        for axis, key in zip(axes, ("wall_median_ms", "mfu_percent"), strict=True):
            axis.bar(
                x,
                [row[key] for row in chosen],
                0.18,
                label=method,
                color=colors[index],
                hatch=hatches[index],
                edgecolor="#30343B",
                linewidth=0.35,
            )
            axis.set_xticks([0, 1], ["Prefill", "Extend"])
    axes[0].set_ylabel("Synchronized wall time (ms)")
    axes[0].set_yscale("log")
    axes[1].set_ylabel("Final precision-normalized MFU (%)")
    fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    for suffix in ("svg", "png"):
        fig.savefig(output / f"final_mfu.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    stages = sorted({row["stage"] for row in operators if row["kernel_mfu_percent"] is not None})
    for axis, phase in zip(axes, ("prefill", "extend"), strict=True):
        for index, method in enumerate(METHODS):
            lookup = {
                row["stage"]: row
                for row in operators
                if (row["mode"], row["phase"]) == (method, phase + "_annotated")
            }
            values = [lookup.get(stage, {}).get("kernel_mfu_percent") for stage in stages]
            axis.bar(
                np.arange(len(stages)) + (index - 1.5) * 0.19,
                [float("nan") if value is None else value for value in values],
                0.18,
                label=method,
                color=colors[index],
                hatch=hatches[index],
                edgecolor="#30343B",
                linewidth=0.35,
            )
        axis.set_title(phase, loc="left")
        axis.set_xticks(np.arange(len(stages)), stages, rotation=35, ha="right", fontsize=8)
        axis.set_ylabel("Matrix API MFU (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    for suffix in ("svg", "png"):
        fig.savefig(output / f"operator_mfu.{suffix}", dpi=180)
    plt.close(fig)


def publish(directory, output, *, layer=1):
    directory = Path(directory).resolve(strict=True)
    result = json.loads((directory / "result.json").read_text())
    if (
        result.get("schema_version") != 3
        or result.get("mode") != "profile"
        or not result.get("accepted")
    ):
        raise ValueError("four-method report requires a successful schema-3 profile")
    if result["num_layers"] != 3 or result["methods"] != list(METHODS):
        raise ValueError("report requires all four methods and three checkpoint layers")
    _, _, independent_results = _independent_runs(directory, result)
    local_native_files = audit_local_native_artifacts(independent_results)
    if digest(directory / "request.json") != result["request_sha256"]:
        raise ValueError("profile request changed")
    for name, expected in result["source_sha256"].items():
        if digest(directory / "source" / name) != expected:
            raise ValueError(f"profile source snapshot changed: {name}")
    wall = benchmark_view(directory, result)
    calls, metadata = read_calls(directory / "operator_calls.json")
    if metadata["run_id"] != result["run_id"]:
        raise ValueError("operator ledger belongs to another run")
    graph_shapes = (
        audit_graph_shapes(
            result, calls, json.loads((directory / "graph_templates.json").read_text())
        )
        if result["compute_graphs"]
        else None
    )
    setup, paths = [], []
    for index, label in enumerate(result["nsys_capture_order"], 1):
        (setup if label == "graph_setup" else paths).append(directory / f"capture_{index}.sqlite")
    analysis = analyze_captures(paths, calls, metadata=metadata, graph_setup_paths=setup)
    expected = {
        (method, phase + "_annotated") for method in METHODS for phase in ("prefill", "extend")
    }
    if (
        len(analysis["captures"]) != 8
        or {(row["mode"], row["phase"]) for row in analysis["captures"]} != expected
    ):
        raise ValueError("profile lacks one of the eight method/phase captures")
    if analysis["calls_outside_selected_captures"]:
        raise ValueError("operator ledger contains uncaptured calls")
    for capture in analysis["captures"]:
        audit = capture["audit"]
        if not audit["kernel_count_and_time_conserved"] or not audit["metadata_call_counts_match"]:
            raise ValueError("operator attribution failed conservation or call coverage")
        if audit["layer_unscoped_kernel_count"] or any(
            row["unattributed_count"] for row in audit["activity_counts_and_ns"].values()
        ):
            raise ValueError("profile contains unattributed GPU work")
        if (
            result["compute_graphs"]
            and not capture["graph_attribution"]["every_replay_gpu_node_verified"]
        ):
            raise ValueError("graph node ownership was not independently verified")
    final, coverage = audit_work(wall, calls, analysis["dense_peaks_tflops"])
    parents = None
    if setup:
        from experiments.deepseek_v32_motivation.src.graph_attribution import read_lineage

        parents = read_lineage(setup)
    timelines, activity_rows, sql_audits = [], [], []
    for path in paths:
        summary, rows = timeline.extract(
            path, calls, parents, layer=layer, measurements=result["measurements"]
        )
        measurements = result["measurements"][summary["method"]]
        phase_key = (
            "prefix_cache_per_layer" if summary["phase"] == "prefill" else "extend_cache_per_layer"
        )
        summary["cache_counters_scope"] = "complete measured phase, not an individual kernel"
        summary["selected_layer_phase_cache_counters"] = measurements[phase_key][layer]
        summary["next_layer_phase_cache_counters"] = (
            measurements[phase_key][layer + 1] if layer + 1 < 3 else None
        )
        if summary["method"] == "dense_prefetch":
            summary["dense_transport_audit"] = audit_dense_transport(
                result, summary, measurements[phase_key]
            )
            sql_audits.append(audit_dense_sql(path, summary, rows))
        timelines.append(summary)
        activity_rows.extend(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "analysis.json", analysis)
    save_csv(output / "operator_mfu.csv", analysis["operators"])
    save_csv(output / "operator_mfu_by_layer.csv", analysis["operators_by_layer"])
    save_csv(
        output / "final_mfu.csv",
        [{key: value for key, value in row.items() if key != "details"} for row in final],
    )
    save_csv(output / "timeline_activities.csv", activity_rows)
    save_json(output / "timeline_summary.json", timelines)
    save_json(output / "dense_overlap_sql_audit.json", sql_audits)
    plot_mfu(final, analysis["operators"], output)
    for phase in ("prefill", "extend"):
        timeline.draw(timelines, activity_rows, output, phase=phase)
    summary = {
        "profile_run_id": result["run_id"],
        "benchmark": result["benchmark"],
        "profile_result_sha256": digest(directory / "result.json"),
        "validation_receipt": result["validation_receipt"],
        "hardware": result["hardware"],
        "source_sha256": result["source_sha256"],
        "backend_provenance": result["backend_provenance"],
        "current_recorded_local_native_artifacts": local_native_files,
        "configuration": {
            key: wall[key]
            for key in (
                "prefix_tokens",
                "extend_tokens",
                "chunk_size",
                "sparse_pool_tokens",
                "extend_residency",
                "compute_graphs",
                "hbm_cache_budget_bytes",
                "dram_cache_budget_bytes",
                "warmups",
                "repeats",
                "prefill_repeats",
            )
        },
        "dense_history_transport": wall.get("dense_history_transport"),
        "measurements": wall["measurements"],
        "dense_peaks_tflops": analysis["dense_peaks_tflops"],
        "peak_reference": analysis["peak_reference"],
        "final_mfu": final,
        "query_coverage": coverage,
        "graph_shapes": graph_shapes,
        "timelines": timelines,
        "generator_sha256": {
            path.name: digest(path) for path in (Path(__file__), Path(timeline.__file__))
        },
        "input_sha256": {
            path.name: digest(path)
            for path in [
                directory / "operator_calls.json",
                *([directory / "graph_templates.json"] if result["compute_graphs"] else []),
                *paths,
                *setup,
            ]
        },
    }
    repository = Path(__file__).resolve().parents[3]
    summary["analysis_helper_source_sha256"] = {
        str(path.relative_to(repository)): digest(path)
        for path in [
            Path(__file__),
            Path(timeline.__file__),
            *(
                Path(__file__).with_name(name)
                for name in (
                    "operator_report.py",
                    "analyze_nsys.py",
                    "launch_gap.py",
                    "execution_utilization.py",
                    "run_contract.py",
                )
            ),
            *(
                repository / "experiments/deepseek_v32_motivation/src" / name
                for name in ("graph_attribution.py", "graph_instrumentation.py")
            ),
        ]
    }
    summary["analysis_source_boundary"] = (
        "actual post-run analysis helpers; measurement source_sha256 separately preserves the original execution snapshot"
    )
    save_json(output / "summary.json", summary)
    text = [
        "# DeepSeek V3.2: four-method MFU",
        "",
        f"Profile `{result['run_id']}`; independent benchmark `{result['benchmark']['run_id']}`.",
        "",
        "The real checkpoint layers 0–2 propagate hidden/residual sequentially. The execution includes embedding, three dense MLPs, final norm and last-token LM head. Results do not extrapolate to the full 61-layer model or GR serving.",
        "",
        f"Prefill {result['prefix_tokens']:,}; extend {result['extend_tokens']:,}; chunk {result['chunk_size']:,}; pool {result['sparse_pool_tokens']:,}; extend residency `{result['extend_residency']}`; compute graphs `{result['compute_graphs']}`.",
        "",
        "| Method | Phase | Median wall ms | Final MFU % |",
        "|---|---|---:|---:|",
    ]
    text.extend(
        f"| {row['method']} | {row['phase']} | {row['wall_median_ms']:.3f} | {row['mfu_percent']:.2f} |"
        for row in final
    )
    text.extend(
        [
            "",
            "![Final MFU](final_mfu.svg)",
            "",
            "Final MFU is 100 × Σ_precision(useful matrix FLOPs / dense peak) / independent synchronized wall time. It includes nonmatrix computation, communication, host scheduling and gaps in the denominator. FP8/BF16/FP32 nominal H200 dense peaks are 1979/989.5/67 TFLOP/s; this is not Tensor Core activity.",
            "",
            "![Operator MFU](operator_mfu.svg)",
            "",
            "[Operator table](operator_mfu.csv) and [per-layer table](operator_mfu_by_layer.csv) use each matrix API's exclusively attributed GPU kernel-duration sum. Quantization and fused prefetch remain in their API denominator. Nonmatrix rows have no FLOPs or MFU. Graph replay ownership is verified against capture-time API nodes and Nsight clone lineage.",
            "",
            "![Prefill timeline](timeline_prefill.svg)",
            "",
            "![Extend timeline](timeline_extend.svg)",
            "",
            f"The prefill timeline selects the last chunk of layer {layer}; extend selects its complete query batch. Each window runs from the preceding layer's GPU computation completion to the selected layer's completion. IO from another stream remains visible. Raw inclusive CPU scopes remain in the CSV. D2D copies are GPU control; IO contains host-direction copies and mapped-host gather kernels with positive or unresolved transfer counts. CUDA API intervals include synchronization. Complete next-layer prefetch activities are retained even outside the selected window; the dotted line marks the window end. Native transport kind, launch correlation, stream, full activity intervals and H2D bytes are reported explicitly. Same-profile phase counters prove zero-transfer gather calls to be GPU control and zero-transfer fused calls to be Compute. Other fused ECHO work remains Compute + IO without inferring separate transfer intervals. Per-call evidence and unresolved reasons are preserved in the CSV and JSON; phase totals are assigned to positive calls only when the transport identity is unique. Unresolved gather calls make the displayed no-IO ratios descriptive and prevent gate certification. GPU idle and the gap including cache/control are reported separately.",
            "",
            "[Timeline intervals](timeline_activities.csv), [gap/overlap metrics](timeline_summary.json), [native SQL overlap audit](dense_overlap_sql_audit.json) and [source and measurement binding](summary.json) preserve the measured boundaries. A single profile per method/phase has no repeated-profile confidence interval. No NCU hardware utilization was collected by this experiment.",
            "",
        ]
    )
    (output / "results.md").write_text("\n".join(text))
    save_json(
        output / "publication_manifest.json",
        {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()},
    )
    return summary


def attach_run_acceptance(output, profile_directory, observers, reconciliation_source):
    """Bind independent numerical, timing, native identity and observer evidence."""
    from evaluation.validation import identity_digest

    profile_directory, output = Path(profile_directory).resolve(), Path(output).resolve()
    profile = json.loads((profile_directory / "result.json").read_text())
    receipt, directories, results = _independent_runs(profile_directory, profile)
    native_files = audit_local_native_artifacts(results)
    same_fields = (
        "execution_identity",
        "source_sha256",
        "backend_provenance",
        "indexer_build",
        "execution_runtime_artifacts",
        "request_sha256",
        "checkpoint_identity",
    )
    equality = {
        key: len({identity_digest(result[key]) for result in results.values()}) == 1
        for key in same_fields
    }
    if not all(equality.values()) or any(not result["accepted"] for result in results.values()):
        raise ValueError("independent runs differ in execution/source/native identity")
    if results["bench"]["validation_receipt"] != profile["validation_receipt"]:
        raise ValueError("benchmark and profile do not bind the same check")
    tensors = []
    import torch

    for method in METHODS:
        control = torch.load(
            Path(receipt["artifact_paths"][method + "_control.pt"]),
            map_location="cpu",
            weights_only=True,
        )
        actual = torch.load(
            profile_directory / (method + "_profile_output.pt"),
            map_location="cpu",
            weights_only=True,
        )
        for field in ("hidden", "logits"):
            if (
                not torch.equal(actual[field], control[field])
                or not torch.isfinite(actual[field]).all()
            ):
                raise ValueError("saved profile tensor differs from accepted control")
            tensors.append(
                {
                    "method": method,
                    "output": field,
                    "shape": list(actual[field].shape),
                    "bitwise_equal": True,
                }
            )
    for entry in profile["execution_runtime_artifacts"]["native_jit"]:
        for kind in ("library", "build_metadata"):
            item = entry[kind]
            _verify_native_file(item)
            native_files.append({"name": entry["name"], "kind": kind, **item})
    observation = {}
    for name, source in observers.items():
        source = Path(source).resolve()
        run = json.loads((source / "run.json").read_text())
        original = json.loads((source / "monitor_audit.json").read_text())
        rows = [
            json.loads(line) for line in (source / "gpu_monitor.jsonl").read_text().splitlines()
        ]
        if not rows or len(rows) != original["samples"] or run["exit_code"] != 0:
            raise ValueError("observer evidence lacks complete successful child execution")
        if any(row["returncode"] or row["gpu_utilization_returncode"] for row in rows):
            raise ValueError("GPU observer has a query error")
        if original["monitor_errors"]:
            raise ValueError("GPU observer has an unresolved monitor error")
        if digest(source / "driver.py") != run["driver_sha256"]:
            raise ValueError("observer source differs from execution record")
        selected = [item for item in original["foreign_processes"] if item["selected_gpu"]]
        wrapper_exit = run["exit_code"] or int(bool(selected or original["monitor_errors"]))
        reconciled = None
        if selected or (source / "reconciliation.json").is_file():
            if reconciliation_source is None:
                raise ValueError("selected-GPU observation requires a reconciliation source")
            reconciled = json.loads((source / "reconciliation.json").read_text())
            if (
                not reconciled["accepted_after_reconciliation"]
                or reconciled["unresolved_selected_gpu_foreign"]
            ):
                raise ValueError("selected GPU has an unresolved foreign process")
            if (
                reconciled["original_wrapper_exit_code"] != wrapper_exit
                or reconciled["original_audit"] != original
            ):
                raise ValueError("reconciliation does not preserve original observer outcome")
            for filename, expected in reconciled["evidence_sha256"].items():
                if digest(source / filename) != expected:
                    raise ValueError("reconciliation raw evidence changed")
            if digest(reconciliation_source) != reconciled["source_sha256"]:
                raise ValueError("reconciliation source changed")
        other_devices = other_gpu_process_observations(
            rows, results[name]["hardware"]["gpu"]["uuid"]
        )
        timestamps = [datetime.fromisoformat(row["time_utc"]).timestamp() for row in rows]
        started, completed = (
            datetime.fromisoformat(run[key]).timestamp() for key in ("started_utc", "completed_utc")
        )
        if (
            any(b < a for a, b in pairwise(timestamps))
            or not started <= timestamps[0] <= timestamps[-1] <= completed
        ):
            raise ValueError("observer timestamps do not lie within the recorded run")
        retained = (
            directories[name] / "observer"
            if name == "check"
            else directories[name].parent.parent / "log" / results[name]["run_id"] / "observer"
        )
        retained.mkdir(parents=True, exist_ok=True)
        paths = [path for path in source.iterdir() if path.is_file()]
        evidence = {}
        for path in paths:
            target = retained / path.name
            if target.exists() and digest(target) != digest(path):
                raise ValueError("retained observer evidence would overwrite a different artifact")
            if not target.exists():
                shutil.copyfile(path, target)
            evidence[path.name] = {
                "path": str(target),
                "sha256": digest(target),
                "bytes": target.stat().st_size,
            }
        if reconciled is not None:
            target = retained / "reconciliation_source.py"
            if not target.exists():
                shutil.copyfile(reconciliation_source, target)
            evidence[target.name] = {
                "path": str(target),
                "sha256": digest(target),
                "bytes": target.stat().st_size,
            }
        observation[name] = {
            "run_id": results[name]["run_id"],
            "command": run["command"],
            "started_utc": run["started_utc"],
            "completed_utc": run["completed_utc"],
            "child_exit_code": run["exit_code"],
            "observer_wrapper_exit_code": wrapper_exit,
            "samples": len(rows),
            "first_sample_utc": rows[0]["time_utc"],
            "last_sample_utc": rows[-1]["time_utc"],
            "maximum_sample_gap_seconds": max((b - a for a, b in pairwise(timestamps)), default=0),
            "raw_selected_gpu_foreign_observations": selected,
            "unresolved_selected_gpu_foreign": [],
            "other_gpu_process_observations": other_devices,
            "reconciliation": reconciled,
            "retained_evidence": evidence,
            "boundary": "Discrete samples; no claim of continuous device isolation or exclusive CPU use. Raw observer failure is preserved separately from successful measurement child.",
        }
    acceptance = {
        "schema": "deepseek-v32-mfu-run-acceptance-v1",
        "execution_identity_fields_equal": equality,
        "canonical_execution_identity_sha256": identity_digest(profile["execution_identity"]),
        "validation_receipt": profile["validation_receipt"],
        "run_results": {
            name: {
                "run_id": result["run_id"],
                "result_path": str(directories[name] / "result.json"),
                "result_sha256": digest(directories[name] / "result.json"),
                "correctness_comparisons": len(result["correctness"]),
            }
            for name, result in results.items()
        },
        "saved_profile_tensor_checks": tensors,
        "current_recorded_native_artifacts": native_files,
        "observations": observation,
        "native_boundary": "Exact equality of recorded source/build flags, loaded backend and runtime identities. Check, bench and profile each require the same ECHO, record-transfer and other local native libraries before and after execution. Current file hashes and byte counts verify those recorded mapped libraries and FlashInfer native libraries/build metadata. Missing original mapped-library records cannot be supplied retrospectively. Graph node attribution separately proves measured GPU ownership.",
    }
    save_json(output / "run_acceptance.json", acceptance)
    with (output / "results.md").open("a") as stream:
        stream.write("\n" + observer_acceptance_text(observation) + "\n")
    save_json(
        output / "publication_manifest.json",
        {
            path.name: digest(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "publication_manifest.json"
        },
    )
    return acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=(1, 2), default=1)
    parser.add_argument("--check-observer", type=Path)
    parser.add_argument("--benchmark-observer", type=Path)
    parser.add_argument("--profile-observer", type=Path)
    parser.add_argument("--reconciliation-source", type=Path)
    args = parser.parse_args()
    observer_args = (
        args.check_observer,
        args.benchmark_observer,
        args.profile_observer,
    )
    if any(observer_args) and not all(observer_args):
        parser.error("provide all three observers together")
    if args.reconciliation_source is not None and not all(observer_args):
        parser.error("reconciliation source requires all three observers")
    summary = publish(args.profile_run, args.output_dir, layer=args.layer)
    if all(observer_args):
        attach_run_acceptance(
            args.output_dir,
            args.profile_run,
            {
                "check": args.check_observer,
                "bench": args.benchmark_observer,
                "profile": args.profile_observer,
            },
            args.reconciliation_source,
        )
    print(json.dumps({"profile_run_id": summary["profile_run_id"], "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
