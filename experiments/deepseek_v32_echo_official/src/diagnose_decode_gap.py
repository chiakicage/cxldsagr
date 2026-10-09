"""Audit the published Q1 timeline gap using CPU-only NSYS interval analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

from experiments.deepseek_v32_echo_official.src import compare_timeline as local_timeline

EXPERIMENT = Path(__file__).resolve().parents[1]
MFU = EXPERIMENT.parent / "deepseek_v32_mfu"
PROCESS_MASK = 0xFFFFFFFFFF000000


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind(path, digest, sources):
    path = Path(path).resolve(strict=True)
    actual = sha256(path)
    require(actual == digest, f"Source digest differs: {path}")
    sources[str(path)] = actual
    return path


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def duration(intervals):
    return sum(end - start for start, end in union(intervals))


def category(row):
    return "GPU control" if row["lane"] == "GPU control" else row["category"]


def kernel_family(row):
    name = row["name"].lower()
    if "mqa_logits" in name:
        return "MQA metadata" if "metadata" in name else "MQA core (fused IO retained)"
    if row["category"] == "Attention":
        return "Main attention"
    if row["category"] == "Projection / RoPE":
        return "Projection / norm / RoPE"
    if row["category"] == "Indexer / top-k" and "gemm_1d2d" in name:
        return "Indexer projection GEMM (official category)"
    if row["category"] == "Indexer / top-k" and (
        "topk" in name or row.get("stage") in {"topk", "exact_topk"}
    ):
        return "Indexer top-k"
    return category(row)


def summarize(rows, start, end, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    result = {}
    for label, members in groups.items():
        intervals = [(max(start, row["start_ns"]), min(end, row["end_ns"])) for row in members]
        union_ns = duration(intervals)
        summed_ns = sum(right - left for left, right in intervals)
        result[label] = {
            "activity_count": len(members),
            "union_ns": union_ns,
            "union_ms": union_ns / 1e6,
            "sum_ns": summed_ns,
            "sum_ms": summed_ns / 1e6,
        }
    return result


def partition(rows, start, end):
    events = defaultdict(list)
    for index, row in enumerate(rows):
        events[max(start, row["start_ns"])].append((index, category(row)))
        events[min(end, row["end_ns"])].append((index, None))
    active, totals, cursor = {}, Counter(), start
    for point, changes in sorted(events.items()):
        labels = tuple(sorted(set(active.values()))) or ("GPU idle",)
        totals[labels] += point - cursor
        for index, label in changes:
            if label is None:
                active.pop(index)
            else:
                active[index] = label
        cursor = point
    require(sum(totals.values()) == end - start, "Partition does not conserve elapsed time")
    return [
        {"active_categories": list(labels), "duration_ns": ns, "duration_ms": ns / 1e6}
        for labels, ns in sorted(totals.items(), key=lambda item: item[1], reverse=True)
        if ns
    ]


def load_panels(comparison, official, local, sources):
    provenance = read(comparison / "provenance.json")
    sources[str((comparison / "provenance.json").resolve())] = sha256(
        comparison / "provenance.json"
    )
    snapshots = provenance["source_snapshot_sha256"]
    for name, digest in snapshots.items():
        bind(comparison / "source_snapshot" / name, digest, sources)
    for name, digest in provenance["inputs_and_sources_sha256"].items():
        # Renderer source is qualified by its saved snapshot, not a later working tree.
        if Path(name).name in snapshots and snapshots[Path(name).name] == digest:
            continue
        bind(name, digest, sources)
    panel_path = bind(
        comparison / "panels.json", provenance["artifacts_sha256"]["panels.json"], sources
    )
    panels = read(panel_path)
    require(len(panels) == 6, "Expected the published six-panel comparison")

    publication = read(official / "publication.json")
    for name in ("report.json", "windows.json", "activities.json"):
        bind(official / name, publication["files"][name], sources)
    official_report = read(official / "report.json")
    official_rows = read(official / "activities.json")
    require(official_report["numerical_acceptance"] is False, "Unexpected numerical claim")
    for panel, window in zip(panels[:2], official_report["windows"], strict=True):
        require(panel["window"] == window, "Official window changed")
        require(panel["rows"] == official_rows[panel["method"]], "Official activities changed")

    publication = read(local / "publication_manifest.json")
    if publication.get("schema") == "isolated-method-artifacts-v1":
        original_panels, paths = local_timeline.mfu_panels(local, None, "extend")
        for path in paths:
            sources[str(path.resolve())] = sha256(path)
        for panel, original in zip(panels[2:], original_panels, strict=True):
            require(
                all(panel.get(key) == value for key, value in original.items() if key != "source"),
                "Local isolated panel differs from the published method panel",
            )
        return (
            panels,
            official_report,
            {
                "format": "isolated-method-artifacts-v1",
                "input_hashes": read(local / "input_hashes.json"),
            },
        )

    manifest = publication["files_sha256"]
    receipt_path = bind(
        local / "h65536_a1/compact_receipt.json",
        manifest["h65536_a1/compact_receipt.json"],
        sources,
    )
    receipt = read(receipt_path)
    local_provenance = read(local / "provenance.json")
    raw_receipts = [
        Path(name)
        for name, digest in local_provenance["inputs_sha256"].items()
        if Path(name).name == "compact_receipt.json" and digest == sha256(receipt_path)
    ]
    require(len(raw_receipts) == 1, "Ambiguous local timeline source")
    local_rows_path = bind(
        raw_receipts[0].parent / "window_rows.json",
        receipt["artifacts_sha256"]["window_rows.json"],
        sources,
    )
    local_panels = read(local_rows_path)["extend"]
    require(receipt["shape"]["extend_tokens"] == 1, "Local trace is not A1")
    require(receipt["shape"]["prefix_tokens"] == 65536, "Local history is not H64K")
    for panel, original in zip(panels[2:], local_panels, strict=True):
        require(panel["method"] == original["method"], "Local method differs")
        require(panel["window"] == original["window"], "Local window differs")
        rows = [row for row in original["rows"] if row["kind"] != "api"]
        require(len(rows) == len(panel["rows"]), "Local activity count differs")
        for selected, source in zip(panel["rows"], rows, strict=True):
            require(
                all(
                    selected[key] == value
                    for key, value in source.items()
                    if key not in {"start_ns", "end_ns", "lane"}
                ),
                "Local activity identity differs",
            )
            require(
                selected["start_ns"] == source["raw_start_ns"]
                and selected["end_ns"] == source["raw_end_ns"],
                "Local raw timestamps differ",
            )
    return panels, official_report, {"format": "mixed-method-compact-receipt", "receipt": receipt}


def local_profile_input(panel, context, sources):
    """Bind one method's result and capture without synthesizing a shared profile."""
    method = panel["method"]
    isolated = context["format"] == "isolated-method-artifacts-v1"
    if isolated:
        provenance = panel["provenance"]
        hashes = context["input_hashes"]
        result_path = Path(provenance["profile_directory"]) / "result.json"
        require(
            hashes[str(result_path)] == provenance["result_sha256"],
            "Local isolated result is not bound by the report",
        )
    else:
        hashes = context["receipt"]["inputs_and_sources_sha256"]
        result_paths = [Path(name) for name in hashes if Path(name).name == "result.json"]
        require(len(result_paths) == 1, "Expected one mixed-method local profile result")
        result_path = result_paths[0]
    bind(result_path, hashes[str(result_path)], sources)
    result = read(result_path)
    label = f"{method}/extend_annotated"
    order = result["nsys_capture_order"]
    require(order.count(label) == 1, "Expected one capture for the selected method")
    path = result_path.parent / f"capture_{order.index(label) + 1}.sqlite"
    if isolated:
        require(
            result.get("schema_version") == 4
            and result.get("selected_method") == method
            and result.get("methods") == [method]
            and result.get("method_isolation") == "fresh-process-one-method-v1"
            and result["run_id"] == panel["profile_run_id"] == provenance["profile_run_id"],
            "Local isolated result belongs to another method or process",
        )
        require(
            result["benchmark"] == provenance["benchmark"]
            and result["validation_receipt"] == provenance["validation_receipt"],
            "Local isolated benchmark or validation binding differs",
        )
        require(
            str(path) == provenance["sqlite"] and hashes[str(path)] == provenance["sqlite_sha256"],
            "Local isolated capture does not match its own capture order",
        )
    bind(path, hashes[str(path)], sources)
    return {
        "profile_run_id": result["run_id"],
        "result_path": str(result_path),
        "result_sha256": hashes[str(result_path)],
        "sqlite": str(path),
        "sqlite_sha256": hashes[str(path)],
        "measurement": {
            key: value
            for key, value in result["measurement_identity"].items()
            if key != "source_sha256"
        },
        "target_pid": result["process_provenance"]["pid"] if isolated else None,
        **(
            {
                "benchmark": result["benchmark"],
                "validation_receipt": result["validation_receipt"],
            }
            if isolated
            else {}
        ),
    }


def raw_capture(panel, path, *, expected_pid=None):
    rows = panel["rows"]
    correlations = {row["correlation"] for row in rows}
    processes = {row["process"] for row in rows}
    require(len(correlations) == len(processes) == 1, "Multiple replay identities")
    correlation, process = next(iter(correlations)), next(iter(processes))
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        strings = dict(db.execute("SELECT id,value FROM StringIds"))
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master")}
        if expected_pid is not None:
            require("PROCESSES" in tables, "Missing raw process identity table")
            pids = {
                row[0]
                for row in db.execute("SELECT pid FROM PROCESSES WHERE globalPid=?", (process,))
            }
            require(pids == {expected_pid}, "Raw capture process differs from the method result")
        gpu = []
        for kind, table in (
            ("kernel", "CUPTI_ACTIVITY_KIND_KERNEL"),
            ("memcpy", "CUPTI_ACTIVITY_KIND_MEMCPY"),
            ("memset", "CUPTI_ACTIVITY_KIND_MEMSET"),
        ):
            if table not in tables:
                continue
            query = f"SELECT * FROM {table} WHERE correlationId=? AND globalPid=?"
            for record in db.execute(query, (correlation, process)):
                row = {**dict(record), "kind": kind}
                if kind == "kernel":
                    row["name"] = strings[row["demangledName"]]
                gpu.append(row)
        by_node = {(row["kind"], row["graphNodeId"]): row for row in gpu}
        require(len(by_node) == len(gpu), "Repeated graph activity identity")
        for row in rows:
            raw = by_node[(row["kind"], row["graph_node_id"])]
            require(
                raw["start"] == row["start_ns"] and raw["end"] == row["end_ns"],
                "Published timestamp differs from raw CUPTI",
            )
            if row["kind"] == "kernel":
                require(raw["name"] == row["name"], "Published kernel name differs")
        apis = [
            dict(row)
            for row in db.execute(
                "SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=?", (correlation,)
            )
        ]
        require(len(apis) == 1, "Replay does not have a unique runtime API")
        api = apis[0]
        require("GraphLaunch" in strings[api["nameId"]], "Selected API is not GraphLaunch")
        require(api["globalTid"] & PROCESS_MASK == process, "Launch process differs")
        scopes = []
        for row in db.execute("SELECT * FROM NVTX_EVENTS"):
            label = row["text"] or strings.get(row["textId"], "")
            if (
                row["end"] is not None
                and row["globalTid"] == api["globalTid"]
                and row["start"] <= api["start"] <= api["end"] <= row["end"]
            ):
                scopes.append({"label": label, "start_ns": row["start"], "end_ns": row["end"]})
        warnings = []
        if "DIAGNOSTIC_EVENT" in tables:
            warnings = [
                row[0]
                for row in db.execute(
                    "SELECT text FROM DIAGNOSTIC_EVENT WHERE globalPid=? AND severity>=2",
                    (process,),
                )
            ]
    start, end = panel["window"]["start_ns"], panel["window"]["end_ns"]
    require(
        len([row for row in gpu if row["start"] < end and row["end"] > start]) == len(rows),
        "A replay activity is missing from the selected window",
    )
    graph = {
        "sqlite": str(path),
        "process": process,
        **({"os_pid": expected_pid} if expected_pid is not None else {}),
        "correlation": correlation,
        "runtime_api": strings[api["nameId"]],
        "api_start_ns": api["start"],
        "api_end_ns": api["end"],
        "api_duration_ms": (api["end"] - api["start"]) / 1e6,
        "api_overlap_with_window_ms": max(0, min(end, api["end"]) - max(start, api["start"])) / 1e6,
        "full_replay_gpu_nodes": len(gpu),
        "full_replay_kind_counts": dict(Counter(row["kind"] for row in gpu)),
        "full_replay_gpu_span_ms": (
            max(row["end"] for row in gpu) - min(row["start"] for row in gpu)
        )
        / 1e6,
        "enclosing_cpu_nvtx": scopes,
        "collector_warnings_for_selected_process": warnings,
        "every_published_activity_matches_raw_cupti": True,
    }
    return graph, by_node


def panel_statistics(panel):
    rows, window = panel["rows"], panel["window"]
    start, end = window["start_ns"], window["end_ns"]
    categories = summarize(rows, start, end, category)
    busy_ns = duration((max(start, row["start_ns"]), min(end, row["end_ns"])) for row in rows)
    idle_ns = end - start - busy_ns
    require(idle_ns == round(window["gpu_idle_ms"] * 1e6), "Idle changed")
    overlap_ns = sum(row["union_ns"] for row in categories.values()) - busy_ns
    combined = summarize(
        rows,
        start,
        end,
        lambda row: (
            "Projection + indexer"
            if category(row) in {"Projection / RoPE", "Indexer / top-k"}
            else category(row)
        ),
    )
    return {
        "source": panel["source"],
        "method": panel["method"],
        "condition": panel["condition"],
        "start_ns": start,
        "end_ns": end,
        "window_ns": end - start,
        "window_ms": (end - start) / 1e6,
        "busy_ns": busy_ns,
        "busy_ms": busy_ns / 1e6,
        "idle_ns": idle_ns,
        "idle_ms": idle_ns / 1e6,
        "cross_category_overlap_ns": overlap_ns,
        "cross_category_overlap_ms": overlap_ns / 1e6,
        "activity_count": len(rows),
        "stream_counts": dict(Counter(row["stream"] for row in rows)),
        "categories": categories,
        "projection_indexer_combined": combined["Projection + indexer"],
        "exclusive_time_partition": partition(rows, start, end),
        "kernel_families": summarize(rows, start, end, kernel_family),
        "kernel_names": summarize(rows, start, end, lambda row: row["name"]),
    }


def differences(official, local):
    labels = official["categories"].keys() | local["categories"].keys()
    rows = []
    for label in labels:
        before = official["categories"].get(label, {}).get("union_ns", 0)
        after = local["categories"].get(label, {}).get("union_ns", 0)
        rows.append({"component": label, "official_ns": before, "local_ns": after})
    rows += [
        {
            "component": "GPU idle",
            "official_ns": official["idle_ns"],
            "local_ns": local["idle_ns"],
        },
        {
            "component": "Subtract cross-category overlap",
            "official_ns": -official["cross_category_overlap_ns"],
            "local_ns": -local["cross_category_overlap_ns"],
        },
    ]
    for row in rows:
        row.update(
            official_ms=row["official_ns"] / 1e6,
            local_ms=row["local_ns"] / 1e6,
            delta_ns=row["local_ns"] - row["official_ns"],
            delta_ms=(row["local_ns"] - row["official_ns"]) / 1e6,
        )
    expected = local["window_ns"] - official["window_ns"]
    require(sum(row["delta_ns"] for row in rows) == expected, "Gap decomposition lost time")
    return {
        "method": local["method"],
        "local_minus_official_ms": expected / 1e6,
        "components": sorted(rows, key=lambda row: row["delta_ns"], reverse=True),
        "interpretation": "An exact accounting identity for the captured timelines, not isolated replacement savings.",
    }


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=EXPERIMENT / "output/data/engine_mfu_decode_q1_20261008_01",
    )
    parser.add_argument("--official-report", type=Path, default=EXPERIMENT / "report/sglang_decode")
    parser.add_argument("--local-report", type=Path, default=MFU / "report/h64k_a1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(output.is_relative_to(EXPERIMENT / "output"), "Output must stay in this experiment")
    require(not output.exists(), "Use a new analysis run ID")
    sources = {}
    panels, official_report, local_context = load_panels(
        args.comparison, args.official_report, args.local_report, sources
    )
    selected = [panels[0], panels[2], panels[1], panels[3]]
    results, details, graphs, local_profiles = {}, [], {}, {}
    for panel in selected:
        official = panel["source"] == "Official SGLang"
        method = panel["method"]
        key = f"{'official' if official else 'local'}_{method}"
        expected_pid = None
        if official:
            record = official_report["sources"][f"{method}_profile"]
            run_path = EXPERIMENT / "output/data" / record["run_id"] / "run.json"
            bind(run_path, record["run_sha256"], sources)
            run = read(run_path)
            path = Path(run["workers"][0]["nsys"]["path"]).with_suffix(".sqlite")
            bind(path, official_report["sources"][method]["sqlite_sha256"], sources)
        else:
            profile = local_profile_input(panel, local_context, sources)
            local_profiles[method] = profile
            path, expected_pid = Path(profile["sqlite"]), profile["target_pid"]
        graph, raw_rows = raw_capture(panel, path, expected_pid=expected_pid)
        graphs[key] = graph
        results[key] = panel_statistics(panel)
        for row in panel["rows"]:
            raw = raw_rows[(row["kind"], row["graph_node_id"])]
            details.append(
                {
                    "panel": key,
                    "family": kernel_family(row),
                    "category": category(row),
                    "original_category": row["category"],
                    "lane": row["lane"],
                    "kind": row["kind"],
                    "layer": row["layer"],
                    "stage": row["stage"],
                    "source_stage": row.get("source_stage"),
                    "name": row["name"],
                    "start_ns": row["start_ns"],
                    "end_ns": row["end_ns"],
                    "clipped_duration_ms": (
                        min(panel["window"]["end_ns"], row["end_ns"])
                        - max(panel["window"]["start_ns"], row["start_ns"])
                    )
                    / 1e6,
                    "stream": row["stream"],
                    "graph_node_id": row["graph_node_id"],
                    "potential_io": row["potential_io"],
                    "bytes": row.get("bytes"),
                    "actual_io": json.dumps(row.get("actual_io"), sort_keys=True),
                    **{
                        name: raw.get(name)
                        for name in (
                            "gridX",
                            "gridY",
                            "gridZ",
                            "blockX",
                            "blockY",
                            "blockZ",
                            "registersPerThread",
                            "staticSharedMemory",
                            "dynamicSharedMemory",
                        )
                    },
                }
            )
    deltas = {
        method: differences(results[f"official_{method}"], results[f"local_{method}"])
        for method in ("hbm", "echo")
    }
    report = {
        "schema": "official-local-q1-gap-audit-v1",
        "run_id": output.name,
        "numerical_acceptance_between_implementations": False,
        "panels": results,
        "local_minus_official": deltas,
        "raw_graphs": graphs,
        "local_report_format": local_context["format"],
        "local_profiles": local_profiles,
        "local_profile_measurements": {
            method: profile["measurement"] for method, profile in local_profiles.items()
        },
        "accounting": [
            "All intervals use integer nanoseconds and are clipped to the published L0-L2 window.",
            "window = sum(category unions) - cross-category overlap + GPU idle.",
            "The exclusive time partition is disjoint and conserves the complete window.",
            "Kernel sums include concurrent activity and must not be added as elapsed latency.",
            "The official indexer category includes indexer projection GEMMs; local classification puts these in Projection / RoPE. Combined Projection + indexer is also reported.",
            "Fused MQA intervals retain their complete duration, including any internal IO; no compute/IO split is inferred.",
            "Official HBM has a 320 ns shared final-RMSNorm overlap at the window end, classified as GPU control by the original graph_body owner. It does not change window, busy or idle.",
        ],
        "unisolated_factors": [
            "Different query tokens (official sampled 57841; local supplied 111090), backend kernels, framework versions, physical GPUs and capacities.",
            "Local offload history KV is cold in DRAM; official preserves natural residency and its synchronized snapshot is only after the request.",
            "Both profiles use node-level CUDA Graph tracing. Each selected step has one GraphLaunch; API duration is recorded separately and is not a GPU lane.",
            "Local capture includes one explicitly excluded in-capture warmup; official has two earlier normal warmup requests. Their collection boundaries differ.",
            "There is no paired profiler-on/off decode-only timing for both implementations. The relative profiling inflation is not isolated by these traces.",
            "A grid size proves launched CTA count, not effective work distribution, occupancy or clock behavior. No NCU execution is performed here.",
            "Collector warnings are retained; this audit verifies the selected activities against raw CUPTI, not every process or setup/warmup event.",
            "Category and kernel differences identify measured work, not the counterfactual wall-time benefit of swapping a kernel.",
        ],
    }
    if local_context["format"] == "mixed-method-compact-receipt":
        report["local_profile_measurement"] = local_profiles["hbm"]["measurement"]
    source_paths = [
        Path(module.__file__).resolve()
        for module in (
            local_timeline,
            local_timeline.engine,
            local_timeline.compact,
            local_timeline.timeline,
        )
    ] + [Path(__file__).resolve()]
    for path in source_paths:
        sources[str(path)] = sha256(path)
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(
        output / "gap_decomposition.csv",
        [
            {"method": method, **row}
            for method, delta in deltas.items()
            for row in delta["components"]
        ],
    )
    write_csv(output / "kernel_details.csv", details)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in source_paths:
        shutil.copyfile(path, snapshot / path.name)
    provenance = {
        "schema": "cpu-gap-analysis-provenance-v1",
        "argv": sys.argv,
        "sources_sha256": sources,
        "artifacts_sha256": {
            path.name: sha256(path) for path in output.iterdir() if path.is_file()
        },
        "source_snapshot_sha256": {
            path.name: sha256(snapshot / path.name) for path in source_paths
        },
        "execution": "CPU-only SQLite/JSON analysis; no model imports, GPU work or source measurements changed.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        json.dumps({method: delta["local_minus_official_ms"] for method, delta in deltas.items()})
    )


if __name__ == "__main__":
    main()
