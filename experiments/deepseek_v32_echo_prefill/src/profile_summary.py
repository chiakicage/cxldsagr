"""Curate existing Nsight analyses without diagnosing or reinterpreting metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
DATA = EXPERIMENT / "output" / "data"
OUTPUTS = ("profile_summary.json", "nsys_stages.csv", "ncu_metrics.csv")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def digest(path, cache):
    path = Path(path).resolve()
    if path not in cache:
        with path.open("rb") as handle:
            cache[path] = hashlib.file_digest(handle, "sha256").hexdigest()
    return cache[path]


def run_directory(run_id):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id), "Invalid run ID")
    path = DATA / run_id
    require(path.is_dir(), f"Missing run directory: {path}")
    return path


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def compact_metric(metric):
    return {
        key: metric.get(key)
        for key in ("unit", "value", "value_status", "instance_count", "errors")
    }


def pm_bins(metric):
    """Keep original sample order; bin boundaries never represent elapsed time."""
    values = metric["values"]
    require(len(values) == metric["instance_count"], "PM instance count mismatch")
    bins = []
    for index in range(32):
        start, end = len(values) * index // 32, len(values) * (index + 1) // 32
        finite = [value for value in values[start:end] if numeric(value)]
        bins.append(
            {
                "sample_start": start,
                "sample_end_exclusive": end,
                "sample_count": end - start,
                "finite_numeric_count": len(finite),
                "mean": statistics.fmean(finite) if finite else None,
            }
        )
    return {
        "unit": metric["unit"],
        "instance_count": len(values),
        "statistics": metric["statistics"],
        "errors": metric.get("errors", []),
        "correlation_count_matches_values": metric.get("correlation_count_matches_values"),
        "correlation_metadata": metric.get("correlation_metadata"),
        "bins": bins,
    }


def source_stalls(action):
    selected = {}
    for name, metric in action["pc_sampling"]["metrics"].items():
        if "warps_issue_stalled" not in name:
            continue
        lines = sorted(
            metric["source_lines"],
            key=lambda row: (
                -row["sample_sum"] if numeric(row["sample_sum"]) else math.inf,
                row["file_name"],
                row["line"],
            ),
        )
        selected[name] = {
            "unit": metric["unit"],
            "instance_count": metric["instance_count"],
            "statistics": metric["statistics"],
            "unmapped_statistics": metric["unmapped_statistics"],
            "errors": metric.get("errors", []),
            "correlation_count_matches_values": metric.get("correlation_count_matches_values"),
            "mapped_source_line_count": len(lines),
            "top_source_lines": lines[:5],
        }
    return selected


def ncu_action(action, range_index):
    names = set(action["key_metrics"])
    for name in action["metrics"]:
        if (
            name.startswith(("launch__", "gpu__dram_throughput."))
            or name == "derived__local_spilling_requests"
            or name.startswith(("dram__", "l1tex__", "lts__"))
            and ("throughput" in name or "hit_rate" in name)
            or "long_scoreboard" in name
            and not name.startswith(("pmsampling:", "smsp__pcsamp_"))
        ):
            names.add(name)
    metrics = {name: compact_metric(action["metrics"][name]) for name in sorted(names)}
    return {
        "range_index": range_index,
        "action_index": action["action_index"],
        "name": action["name"],
        "names": action["names"],
        "metrics": metrics,
        "absent_key_metric_candidates": action["absent_key_metric_candidates"],
        "source_pc_stalls": source_stalls(action),
        "pm_sampling": {name: pm_bins(metric) for name, metric in action["pm_sampling"].items()},
        "extraction_errors": action["errors"],
    }


def collect_ncu(directory, full_run_id, hashes):
    analysis_path = directory / "ncu_analysis.json"
    analysis = read_json(analysis_path)
    captures = {}
    for label in ("full", "source"):
        meta_path = directory / f"{label}.json"
        meta = read_json(meta_path)
        require(
            meta["accepted"] is True
            and meta["run_id"] == directory.name
            and meta["source_run_id"] == full_run_id
            and meta["capture_label"] == label,
            f"NCU metadata does not identify the accepted capture: {meta_path}",
        )
        require(
            digest(meta["input"], hashes) == meta["input_sha256"],
            f"Captured activation SHA256 mismatch: {meta_path}",
        )
        reports = [
            report
            for report in analysis["reports"]
            if Path(report["path"]).name == f"{label}_{meta['kernel']}.ncu-rep"
        ]
        require(len(reports) == 1, f"Expected one native {label} report in {analysis_path}")
        report = reports[0]
        require(
            digest(report["path"], hashes) == report["sha256"],
            f"Native NCU report SHA256 mismatch: {report['path']}",
        )
        actions = [
            ncu_action(action, region["range_index"])
            for region in report["ranges"]
            for action in region["actions"]
        ]
        require(
            len(actions)
            == report["action_count"]
            == meta["selected_launches_inside_profiler_api_range"],
            f"NCU launch count mismatch: {report['path']}",
        )
        require(
            all(re.search(meta["ncu_kernel_regex"], action["name"]) for action in actions),
            f"NCU kernel name differs from capture metadata: {report['path']}",
        )
        captures[label] = {
            "metadata": meta,
            "metadata_sha256": digest(meta_path, hashes),
            "native_report": report["path"],
            "native_report_sha256": report["sha256"],
            "validation": {
                "metadata_accepted": True,
                "input_sha256_matches": True,
                "native_report_sha256_matches": True,
                "launch_count_matches": True,
                "kernel_name_matches": True,
            },
            "actions": actions,
        }
    for field in (
        "input_sha256",
        "kernel",
        "layer",
        "query_start",
        "tensor_metadata",
        "measurement_boundary",
        "source_sha256",
    ):
        require(
            captures["full"]["metadata"][field] == captures["source"]["metadata"][field],
            f"Full/source capture metadata differ: {directory.name}/{field}",
        )
    return {
        "run_id": directory.name,
        "analysis_path": str(analysis_path),
        "analysis_sha256": digest(analysis_path, hashes),
        "analyzer_sha256": analysis["analyzer_sha256"],
        "ncu_report_module": analysis["ncu_report_module"],
        "extraction_notes": analysis["notes"],
        "full_source_metadata_match": True,
        "captures": captures,
    }


def collect_nsys(directory, mode, result_sha, request_sha, hashes):
    path = directory / f"{mode}_nsys.json"
    analysis = read_json(path)
    counters = analysis["kv_transfers"]
    require(
        analysis["mode"] == mode
        and counters["source_run_id"] == directory.name
        and counters["source_accepted"] is True
        and counters["source_sha256"] == result_sha,
        f"Nsight analysis does not reference the accepted full run: {path}",
    )
    selected = {
        key: analysis[key]
        for key in (
            "analyzer_sha256",
            "scope_prefix",
            "nvtx_scope_count",
            "cpu_nvtx_envelope",
            "capture_gpu_envelope",
            "stages",
            "kernel_categories",
            "api_summary",
            "kv_transfers",
            "notes",
        )
    }
    selected.update(
        {
            "analysis_path": str(path),
            "analysis_sha256": digest(path, hashes),
            "sqlite_path": analysis["sqlite"],
            "sqlite_sha256": digest(analysis["sqlite"], hashes),
            "request_sha256": request_sha,
            "devices": [
                {key: value for key, value in row.items() if key != "largest_gaps"}
                for row in analysis["devices"]
            ],
            "attribution": {
                key: value
                for key, value in analysis["attribution"].items()
                if key != "unattributed_activities"
            },
            "global_wait_and_synchronization_apis": [
                row
                for row in analysis["api_summary"]["calls_by_name"]
                if "Synchronize" in row["name"] or "Wait" in row["name"]
            ],
        }
    )
    return selected


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate(full_run_id, ncu_run_ids, *, publish=False):
    directory = run_directory(full_run_id)
    require(len(set(ncu_run_ids)) == len(ncu_run_ids), "Duplicate NCU run IDs")
    hashes = {}
    full = read_json(directory / "summary.json")
    result_sha = digest(directory / "result.json", hashes)
    require(
        full["accepted"] is True
        and full["run_id"] == full_run_id
        and full["provenance"]["result_sha256"] == result_sha,
        "Full-run summary does not identify the accepted result",
    )
    summary = {
        "schema_version": 1,
        "full_run_id": full_run_id,
        "result_sha256": result_sha,
        "request_sha256": full["provenance"]["request_sha256"],
        "curator_sha256": digest(__file__, hashes),
        "notes": [
            "Existing Nsight analysis data are selected without performance diagnosis.",
            "Nsight capture envelopes, actual GPU kernel duration sums, CPU API unions and uninstrumented model latency are different quantities; none are added together.",
            "Wait/synchronization API rows are global capture totals. Stage api_ms includes non-wait APIs and is not CPU waiting time.",
            "Mapped-host KV bytes come from accepted result cache counters, not CUDA memcpy bytes.",
            "NCU full/source replay boundaries and verification metadata are retained separately; they do not reconstruct the original model cache state.",
            "All NCU metrics retain their exact names and units. Null or unavailable values are not replaced by zero.",
            "PC source lines are ranked by reported sample_sum for each stall metric; at most five lines are retained, with PCs and completeness metadata.",
            "Each PM metric is split into 32 contiguous sample-index ranges using floor(n*i/32). Means include finite numeric values only. These bins are not time intervals and are not interpreted.",
        ],
        "nsys": {
            mode: collect_nsys(
                directory, mode, result_sha, full["provenance"]["request_sha256"], hashes
            )
            for mode in ("resident", "offload")
        },
        "ncu": [collect_ncu(run_directory(run_id), full_run_id, hashes) for run_id in ncu_run_ids],
    }
    nsys_rows = [
        {"full_run_id": full_run_id, "mode": mode, **row}
        for mode, analysis in summary["nsys"].items()
        for row in analysis["stages"]
    ]
    ncu_rows = [
        {
            "full_run_id": full_run_id,
            "ncu_run_id": run["run_id"],
            "capture": label,
            "range_index": action["range_index"],
            "action_index": action["action_index"],
            "kernel": capture["metadata"]["kernel"],
            "metric": name,
            **{key: value for key, value in metric.items() if key != "errors"},
            "extraction_error_count": len(metric["errors"] or []),
        }
        for run in summary["ncu"]
        for label, capture in run["captures"].items()
        for action in capture["actions"]
        for name, metric in action["metrics"].items()
    ]
    serialized = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (directory / OUTPUTS[0]).write_text(serialized)
    write_csv(directory / OUTPUTS[1], nsys_rows)
    write_csv(directory / OUTPUTS[2], ncu_rows)
    if publish:
        report = EXPERIMENT / "report"
        report.mkdir(exist_ok=True)
        for name in OUTPUTS:
            shutil.copyfile(directory / name, report / name)
    return {
        "full_run_id": full_run_id,
        "ncu_runs": len(summary["ncu"]),
        "nsys_stage_rows": len(nsys_rows),
        "ncu_metric_rows": len(ncu_rows),
        "published": publish,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-id", required=True)
    parser.add_argument("--ncu-run-id", action="append", required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    try:
        status = generate(args.full_run_id, args.ncu_run_id, publish=args.publish)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Profile summary rejected: {exc}\n")
    print(json.dumps(status))


if __name__ == "__main__":
    main()
