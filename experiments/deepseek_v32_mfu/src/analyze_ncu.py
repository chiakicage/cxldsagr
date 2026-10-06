"""Extract Nsight Compute observations without launching CUDA or the profiler.

The native ``ncu_report`` import is deferred until report loading. Report/range/
action boundaries and metric names are retained; no kernel name is assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

_EXPERIMENT = Path(__file__).resolve().parents[1]
_KEY_METRICS = (
    "gpu__time_duration.sum",
    "launch__block_size",
    "launch__grid_size",
    "launch__waves_per_multiprocessor",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__occupancy_limit_blocks",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_read.sum.per_second",
    "dram__bytes_write.sum",
    "dram__bytes_write.sum.per_second",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
)
_NOTES = [
    "All aggregates are values reported by NCU, not recomputed from metric instances.",
    (
        "Missing values are null. Nonfinite values use a {nonfinite: NaN/Infinity/-Infinity} "
        "object so strict JSON preserves their distinction from missing observations. "
        "Getter failures include their method and instance index in errors."
    ),
    (
        "PM values and correlation IDs retain their original instance order, including zeros, "
        "missing entries, and duplicate IDs. IDs are not assumed to be timestamps or converted "
        "to time units; their API metadata is recorded separately. JSON integers retain uint64 "
        "precision; consumers must not parse them through binary64."
    ),
    (
        "Statistics describe finite numeric values only; standard deviation is population "
        "standard deviation. Missing, nonfinite, nonnumeric and zero counts are separate. "
        "No time weighting or interpolation is applied."
    ),
    (
        "PC metric correlation_ids join to pc_sampling.source_by_pc using decimal integer "
        "keys. Unknown source locations stay explicit. Source-line sums are emitted only for "
        "raw sample-count metrics, separately for each metric; derived ratios are not added. "
        "Embedded source text is read from the report, never from the current source checkout."
    ),
    "These are extracted observations, not performance diagnoses or standalone timing results.",
]


def _call(obj, method, errors, *args):
    try:
        return getattr(obj, method)(*args)
    except Exception as exc:  # noqa: BLE001 - Preserve independent native getter failures as data.
        errors.append(
            {"method": method, "args": list(args), "error": f"{type(exc).__name__}: {exc}"}
        )
        return None


def _json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": "NaN" if math.isnan(value) else str(value).replace("inf", "Infinity")}
    return value


def _value(metric, errors, index=None):
    args = () if index is None else (index,)
    before = len(errors)
    # NCU 2026.1.1's native has_value argument is ValueKind, despite the
    # Python docstring calling it an instance index. Passing indices silently
    # discards almost every PC/PM sample. value(index) dispatches to the typed
    # as_double/as_uint64/as_string instance getter and preserves real zeros.
    if index is None and _call(metric, "has_value", errors) is False:
        return None, "missing"
    value = _call(metric, "value", errors, *args)
    if value is None:
        return None, "error" if len(errors) > before else "missing"
    value = _json_value(value)
    return value, "nonfinite" if isinstance(value, dict) else "present"


def _enum(obj, prefix, suffixes, code):
    label = next(
        (
            name
            for name in suffixes
            if code is not None and getattr(obj, prefix + name, None) == code
        ),
        None,
    )
    return {"code": code, "name": label}


def _metric_metadata(metric, errors):
    kind = _call(metric, "kind", errors)
    rollup = _call(metric, "rollup_operation", errors)
    return {
        "unit": _call(metric, "unit", errors),
        "description": _call(metric, "description", errors),
        "kind": _enum(
            metric,
            "ValueKind_",
            ("UNKNOWN", "ANY", "STRING", "FLOAT", "DOUBLE", "UINT32", "UINT64"),
            kind,
        ),
        "rollup": _enum(metric, "RollupOperation_", ("NONE", "AVG", "MAX", "MIN", "SUM"), rollup),
        "instance_count": _call(metric, "num_instances", errors),
    }


def _aggregate(metric):
    errors = []
    result = _metric_metadata(metric, errors)
    result["value"], result["value_status"] = _value(metric, errors)
    result["errors"] = errors
    return result


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _statistics(values):
    finite = [value for value in values if _is_number(value) and math.isfinite(value)]
    return {
        "count": len(values),
        "finite_numeric_count": len(finite),
        "missing_count": sum(value is None for value in values),
        "nonfinite_count": sum(
            isinstance(value, dict) and "nonfinite" in value for value in values
        ),
        "nonnumeric_count": sum(isinstance(value, (str, bool)) for value in values),
        "zero_count": sum(value == 0 for value in finite),
        "minimum": min(finite) if finite else None,
        "maximum": max(finite) if finite else None,
        "mean": _json_value(statistics.fmean(finite)) if finite else None,
        "population_stddev": _json_value(statistics.pstdev(finite)) if finite else None,
    }


def _instances(metric, count, errors):
    values, statuses = [], []
    for index in range(count or 0):
        value, status = _value(metric, errors, index)
        values.append(value)
        statuses.append(status)
    return values, statuses


def _series(metric, aggregate):
    errors = []
    values, statuses = _instances(metric, aggregate["instance_count"], errors)
    has_ids = _call(metric, "has_correlation_ids", errors)
    ids, id_statuses, id_metadata = None, None, None
    if has_ids:
        correlation = _call(metric, "correlation_ids", errors)
        if correlation is not None:
            id_errors = []
            id_metadata = _metric_metadata(correlation, id_errors)
            id_metadata["name"] = _call(correlation, "name", id_errors)
            ids, id_statuses = _instances(correlation, id_metadata["instance_count"], id_errors)
            id_metadata["errors"] = id_errors
    return {
        "unit": aggregate["unit"],
        "instance_count": aggregate["instance_count"],
        "values": values,
        "value_status": statuses,
        "has_correlation_ids": has_ids,
        "correlation_ids": ids,
        "correlation_id_status": id_statuses,
        "correlation_metadata": id_metadata,
        "correlation_count_matches_values": len(ids) == len(values) if ids is not None else None,
        "statistics": _statistics(values),
        "errors": errors,
    }


def _source(action, pc, embedded_sources):
    errors = []
    info = _call(action, "source_info", errors, pc)
    file_name = _call(info, "file_name", errors) if info is not None else None
    line = _call(info, "line", errors) if info is not None else None
    known = bool(file_name and isinstance(line, int) and line > 0)
    text = None
    if known:
        lines = embedded_sources.get(file_name, "").splitlines()
        if line <= len(lines):
            text = lines[line - 1]
    return {
        "pc": pc,
        "pc_hex": hex(pc),
        "file_name": file_name,
        "line": line,
        "source_known": known,
        "source_text": text,
        "sass": _call(action, "sass_by_pc", errors, pc) or None,
        "errors": errors,
    }


def _line_summaries(name, series, sources):
    # Base pcsamp counters are additive samples. Derived metrics (.pct, .ratio,
    # per-cycle averages, etc.) retain per-line statistics without an invented sum.
    additive = "." not in name and (
        name == "smsp__pcsamp_sample_count" or name.startswith("smsp__pcsamp_warps_issue_stalled_")
    )
    groups = defaultdict(lambda: {"values": [], "pcs": set()})
    unmapped = []
    ids = series["correlation_ids"] or []
    for index, value in enumerate(series["values"]):
        pc = ids[index] if index < len(ids) else None
        source = sources.get(str(pc))
        if source is None or not source["source_known"]:
            unmapped.append(value)
            continue
        group = groups[(source["file_name"], source["line"])]
        group["values"].append(value)
        group["pcs"].add(pc)
    rows = []
    for (file_name, line), group in sorted(groups.items()):
        stats = _statistics(group["values"])
        numbers = [value for value in group["values"] if _is_number(value)]
        rows.append(
            {
                "file_name": file_name,
                "line": line,
                "pcs": sorted(group["pcs"]),
                "statistics": stats,
                "sample_sum": _json_value(sum(numbers)) if additive and numbers else None,
                "sample_sum_complete": stats["finite_numeric_count"] == stats["count"]
                if additive
                else None,
            }
        )
    return rows, _statistics(unmapped)


def analyze_action(action, action_index=0):
    """Extract one API action; accepts CPU fakes for schema/edge-case tests."""
    errors = []
    names = {"function": _call(action, "name", errors)}
    for base in ("DEMANGLED", "MANGLED"):
        enum_value = getattr(action, "NameBase_" + base, None)
        names[base.lower()] = (
            _call(action, "name", errors, enum_value) if enum_value is not None else None
        )
    workload = _call(action, "workload_type", errors)
    metric_names = _call(action, "metric_names", errors)
    if metric_names is None:
        raise ValueError(f"Cannot enumerate metrics for action {action_index}: {errors}")
    metrics, pc_metrics, pm_metrics = {}, {}, {}
    missing = []
    for name in metric_names:
        getter_errors = []
        metric = _call(action, "metric_by_name", getter_errors, name)
        if metric is None:
            metrics[name] = {"value": None, "value_status": "unavailable", "errors": getter_errors}
            missing.append(name)
            continue
        aggregate = _aggregate(metric)
        metrics[name] = aggregate
        if name.startswith("pmsampling:") or ".TriageCompute." in name:
            pm_metrics[name] = _series(metric, aggregate)
        elif name.startswith("smsp__pcsamp_"):
            pc_metrics[name] = _series(metric, aggregate)

    # The installed SWIG API returns map_string_string despite documenting dict.
    embedded = dict(_call(action, "source_files", errors) or {}) if pc_metrics else {}
    pcs = {
        pc
        for series in pc_metrics.values()
        for pc in series["correlation_ids"] or []
        if isinstance(pc, int) and not isinstance(pc, bool) and pc >= 0
    }
    sources = {str(pc): _source(action, pc, embedded) for pc in sorted(pcs)}
    for name, series in pc_metrics.items():
        series["source_lines"], series["unmapped_statistics"] = _line_summaries(
            name, series, sources
        )
    key_names = {
        name
        for name in metrics
        if name in _KEY_METRICS
        or name.startswith(("launch__", "smsp__average_warps_issue_stalled_"))
        or "__pipe_tensor_cycles_active" in name
    }
    return {
        "action_index": action_index,
        "name": names["demangled"] or names["function"],
        "names": names,
        "workload_type": _enum(
            action, "WorkloadType_", ("KERNEL", "RANGE", "CMDLIST", "GRAPH"), workload
        ),
        "metric_count": len(metrics),
        "metrics": metrics,
        "unavailable_metric_names": missing,
        "key_metrics": {name: metrics[name] for name in sorted(key_names)},
        "absent_key_metric_candidates": [name for name in _KEY_METRICS if name not in metrics],
        "pc_sampling": {
            "metrics": pc_metrics,
            "source_by_pc": sources,
            "embedded_source_files": [
                {"file_name": name, "content_available": bool(content)}
                for name, content in sorted(embedded.items())
            ],
        },
        "pm_sampling": pm_metrics,
        "errors": errors,
    }


def analyze_report(report, report_path):
    """Keep the loaded report alive while visiting every range and action."""
    ranges = []
    for range_index in range(report.num_ranges()):
        report_range = report.range_by_idx(range_index)
        if report_range is None:
            raise ValueError(f"Missing range {range_index} in {report_path}")
        actions = []
        for action_index in range(report_range.num_actions()):
            action = report_range.action_by_idx(action_index)
            if action is None:
                raise ValueError(f"Missing action {range_index}/{action_index} in {report_path}")
            actions.append(analyze_action(action, action_index))
        ranges.append(
            {"range_index": range_index, "action_count": len(actions), "actions": actions}
        )
    return {
        "path": str(Path(report_path).resolve()),
        "range_count": len(ranges),
        "action_count": sum(row["action_count"] for row in ranges),
        "ranges": ranges,
    }


def _load_api(python_path=None):
    if python_path is not None:
        candidate = Path(python_path).resolve()
        if not (candidate / "ncu_report.py").is_file():
            raise ValueError(f"ncu_report.py not found in {candidate}")
        sys.path.insert(0, str(candidate))
        return importlib.import_module("ncu_report")
    try:
        return importlib.import_module("ncu_report")
    except ModuleNotFoundError as exc:
        if exc.name != "ncu_report":
            raise
    candidates = []
    executable = shutil.which("ncu")
    if executable:
        candidates.append(Path(executable).resolve().parent / "extras/python")
    candidates.extend(
        sorted(Path("/opt/nvidia/nsight-compute").glob("*/extras/python"), reverse=True)
    )
    for candidate in candidates:
        if (candidate / "ncu_report.py").is_file():
            sys.path.insert(0, str(candidate))
            return importlib.import_module("ncu_report")
    raise RuntimeError(
        "Cannot find ncu_report; pass --ncu-python-path <Nsight Compute>/extras/python"
    )


def analyze_paths(report_paths, *, python_path=None):
    api = _load_api(python_path)
    reports = []
    for path in report_paths:
        path = Path(path).resolve(strict=True)
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        report = api.load_report(str(path))
        if report is None:
            raise ValueError(f"NCU could not load report {path}")
        extracted = analyze_report(report, path)
        extracted["sha256"] = digest
        reports.append(extracted)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "ncu_report_module": getattr(api, "__file__", None),
        "notes": _NOTES,
        "reports": reports,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        action="append",
        required=True,
        help="Repeat for multiple .ncu-rep files.",
    )
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--run-id", help="Write output/data/<run_id>/ncu_analysis.json in this experiment."
    )
    destination.add_argument(
        "--output",
        type=Path,
        help="New JSON path, normally output/data/<run_id>/ncu_analysis.json.",
    )
    parser.add_argument(
        "--ncu-python-path", type=Path, help="Directory containing the installed ncu_report.py."
    )
    args = parser.parse_args(argv)
    if args.run_id is not None and (
        not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in (".", "..")
    ):
        parser.error("--run-id must be a single directory name")
    output = args.output or _EXPERIMENT / "output/data" / args.run_id / "ncu_analysis.json"
    output = output.resolve()
    if output in [path.resolve() for path in args.report]:
        parser.error("--output must not replace an input report")
    if output.suffix != ".json":
        parser.error("--output must have a .json suffix")
    if output.exists():
        parser.error(f"Output already exists: {output}; choose a new output path")
    for path in args.report:
        if not path.is_file():
            parser.error(f"Report does not exist: {path}")
    result = analyze_paths(args.report, python_path=args.ncu_python_path)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as destination_file:
        destination_file.write(serialized)
    print(
        json.dumps(
            {
                "output": str(output),
                "reports": len(result["reports"]),
                "actions": sum(row["action_count"] for row in result["reports"]),
            }
        )
    )


if __name__ == "__main__":
    main()
