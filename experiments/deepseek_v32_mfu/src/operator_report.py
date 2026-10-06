"""Account for every kernel in annotated ECHO captures and report useful matrix MFU.

The call ledger supplies semantic work counts; Nsight supplies actual GPU times.
No FLOPs are guessed from a kernel name or inferred for missing ledger entries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from experiments.deepseek_v32_mfu.src.analyze_nsys import (
    _api_summary,
    _assign_scopes,
    _attribute,
    _device_summary,
    _read_capture,
    _union_ns,
    kernel_category,
)

_SCOPE = re.compile(
    r"^echo/(?P<mode>resident|offload|hbm|echo|serial_sparse|dense_prefetch)/"
    r"(?P<phase>prefill_annotated|extend_annotated)/"
    r"(?P<layer>layer_\d+|shared)(?:/(?P<stage>[^/]+))?(?:/call_\d+)?$"
)
_KEY = ("mode", "phase", "layer", "stage")
_DEFAULT_PEAKS = {"FP8": 1979.0, "BF16": 989.5, "FP32": 67.0}
_NOTES = [
    (
        "MFU uses useful matrix FLOPs only, divided by dense precision-specific peak and actual "
        "GPU kernel-duration sum. Masked/padded/recomputed arithmetic, quantization, scalar math, "
        "selection and transfers do not add useful matrix FLOPs; their time stays in their scope."
    ),
    (
        "Unknown or nonmatrix work has null FLOPs/MFU, never an invented zero-MFU result. "
        "Mixed precision or partly unknown groups have no single aggregate MFU."
    ),
    (
        "GPU activities are attributed through runtime/driver launch correlation to the innermost "
        "matching CPU NVTX scope. GPU timestamps need not fit in the CPU NVTX interval."
    ),
    (
        "Scope host time is the inclusive per-thread union of CPU NVTX ranges. API time is the "
        "per-thread union of non-event CUDA runtime/driver calls attributed exclusively to the scope. "
        "Neither is GPU execution time. Parent and child inclusive host times are not additive."
    ),
    (
        "Kernel sum counts each activity exactly once. It can exceed elapsed wall time when kernels "
        "overlap. Per-device busy/gap unions include kernels, copies and memsets, even if unattributed."
    ),
    (
        "Cross-layer rows pool all matching calls and durations; they are duration-weighted work/time "
        "ratios, not arithmetic means of per-layer MFU values. They are not end-to-end model MFU."
    ),
    (
        "H200 SXM reference peaks: NVIDIA lists BF16 1979 and FP8 3958 TFLOPS with sparsity; "
        "dense peaks are half, 989.5 and 1979. FP32 is 67 TFLOPS without sparsity. Explicit overrides "
        "must match the recorded device and actual arithmetic. Device identity is a separate record."
    ),
]


def _key(row):
    return tuple(row.get(field) or "layer_unscoped" for field in _KEY)


def _sum_ns(rows):
    return sum(row["end"] - row["start"] for row in rows)


def _thread_union_ns(rows):
    by_thread = defaultdict(list)
    for row in rows:
        by_thread[row["thread"]].append((row["start"], row["end"]))
    return sum(_union_ns(intervals) for intervals in by_thread.values())


def read_calls(path):
    """Validate the semantic ledger without loading model or CUDA dependencies."""
    document = json.loads(Path(path).read_text())
    if not isinstance(document, (list, dict)):
        raise TypeError("operator call document must be a list or an object")
    calls = document if isinstance(document, list) else document.get("calls")
    if not isinstance(calls, list):
        raise TypeError("operator calls must be a list or an object with a calls list")
    validated = []
    for index, original in enumerate(calls):
        if not isinstance(original, dict):
            raise TypeError(f"call {index} must be an object")
        row = dict(original)
        if any(not isinstance(row.get(field), str) or not row[field] for field in _KEY):
            raise ValueError(f"call {index} lacks a nonempty mode/phase/layer/stage")
        if not _SCOPE.fullmatch("echo/" + "/".join(row[field] for field in _KEY)):
            raise ValueError(f"call {index} does not identify an annotated ECHO stage")
        flops = row.get("useful_flops")
        if flops is not None and (
            isinstance(flops, bool)
            or not isinstance(flops, (float, int))
            or not math.isfinite(flops)
            or flops < 0
        ):
            raise ValueError(f"call {index} useful_flops must be finite, nonnegative or null")
        precision = row.get("precision")
        if precision is not None:
            if not isinstance(precision, str) or precision.upper() not in _DEFAULT_PEAKS:
                raise ValueError(f"call {index} precision must be FP8, BF16, FP32 or null")
            row["precision"] = precision.upper()
        if flops is not None and flops > 0 and precision is None:
            raise ValueError(f"call {index} positive useful_flops requires a precision")
        validated.append(row)
    metadata = (
        {} if isinstance(document, list) else {k: v for k, v in document.items() if k != "calls"}
    )
    return validated, metadata


def _mfu(calls, kernel_ns, peaks):
    if not calls:
        return None, None, None, None, "missing_call_metadata"
    if any(row.get("useful_flops") is None for row in calls):
        reasons = sorted({row.get("flop_reason") or "unknown_or_nonmatrix_work" for row in calls})
        return None, None, None, None, "; ".join(reasons)
    flops = sum(row["useful_flops"] for row in calls)
    precisions = {row.get("precision") for row in calls}
    if len(precisions) != 1 or None in precisions:
        return flops, None, None, None, "mixed_or_unspecified_precision"
    precision = precisions.pop()
    peak = peaks[precision]
    if not kernel_ns or not flops:
        return flops, precision, peak, None, "no_kernel_time" if not kernel_ns else "nonmatrix_work"
    return flops, precision, peak, flops / (kernel_ns * peak * 1000) * 100, None


def _operator_row(key, scopes, apis, activities, calls, peaks):
    kernels = [row for row in activities if row["kind"] == "kernel"]
    kernel_ns = _sum_ns(kernels)
    non_event = [row for row in apis if not row["bookkeeping"]]
    flops, precision, peak, mfu, reason = _mfu(calls, kernel_ns, peaks)
    row = dict(zip(_KEY, key))
    row.update(
        scope_count=len(scopes),
        metadata_call_count=len(calls),
        metadata_call_count_matches=len(scopes) == len(calls),
        useful_flops=flops,
        precision=precision,
        dense_peak_tflops=peak,
        kernel_mfu_percent=mfu,
        mfu_unavailable_reason=reason,
        scope_host_union_ms=_thread_union_ns(scopes) / 1e6,
        scope_host_duration_sum_ms=_sum_ns(scopes) / 1e6,
        api_thread_union_ms=_thread_union_ns(non_event) / 1e6,
        api_count=len(non_event),
        runtime_api_count=sum(api["source"] == "RUNTIME" for api in non_event),
        driver_api_count=sum(api["source"] == "DRIVER" for api in non_event),
        kernel_launch_count=len({(r["process"], r["correlation"]) for r in kernels}),
        useful_tflops=flops / (kernel_ns * 1000) if flops is not None and kernel_ns else None,
    )
    for kind in ("kernel", "memcpy", "memset"):
        selected = [activity for activity in activities if activity["kind"] == kind]
        row[kind + "_count"] = len(selected)
        row[kind + "_ns"] = _sum_ns(selected)
        row[kind + "_ms"] = _sum_ns(selected) / 1e6
    if calls and len(scopes) != len(calls):
        row["kernel_mfu_percent"] = None
        row["mfu_unavailable_reason"] = "scope_metadata_call_count_mismatch"
    return row


def _kernel_inventory(activities, mode, phase, source):
    buckets = defaultdict(list)
    for activity in activities:
        if activity["kind"] != "kernel":
            continue
        scope = activity["scope"]
        key = (
            scope["layer"] if scope else None,
            (scope["stage"] or "layer_unscoped") if scope else None,
            activity["device_id"],
            activity["name"],
            activity["unattributed_reason"],
        )
        buckets[key].append(activity)
    output = []
    for (layer, stage, device, name, reason), group in sorted(
        buckets.items(), key=lambda item: -_sum_ns(item[1])
    ):
        times = [row["end"] - row["start"] for row in group]
        output.append(
            {
                "mode": mode,
                "phase": phase,
                "layer": layer,
                "stage": stage,
                "device_id": device,
                "kernel_name": name,
                "kernel_category": kernel_category(name),
                "count": len(group),
                "total_ns": sum(times),
                "total_ms": sum(times) / 1e6,
                "min_us": min(times) / 1e3,
                "max_us": max(times) / 1e3,
                "mean_us": sum(times) / len(times) / 1e3,
                "unattributed_reason": reason,
                "sqlite": source,
            }
        )
    return output


def analyze_captures(sqlite_paths, calls, *, metadata=None, peaks=None, graph_setup_paths=()):
    """Read one capture per mode/phase; emit exact coverage accounting and work ratios."""
    peaks = dict(_DEFAULT_PEAKS if peaks is None else peaks)
    if set(peaks) != set(_DEFAULT_PEAKS) or any(
        not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
        for value in peaks.values()
    ):
        raise ValueError("dense peaks must provide positive finite FP8, BF16 and FP32 TFLOPS")
    calls_by_key = defaultdict(list)
    for call in calls:
        calls_by_key[_key(call)].append(call)
    captures, inventory, rows = [], [], []
    aggregate = defaultdict(lambda: {"scopes": [], "apis": [], "activities": [], "calls": []})
    seen = set()
    parents = None
    if graph_setup_paths:
        from experiments.deepseek_v32_motivation.src.graph_attribution import read_lineage

        parents = read_lineage(graph_setup_paths)
    for source in sqlite_paths:
        path = Path(source).resolve(strict=True)
        scopes, apis, activities, tables = _read_capture(path, scope_pattern=_SCOPE)
        identities = {(scope["mode"], scope["phase"]) for scope in scopes}
        if len(identities) != 1:
            raise ValueError(f"{path}: expected exactly one annotated ECHO mode/phase")
        mode, phase = identities.pop()
        if (mode, phase) in seen:
            raise ValueError(f"duplicate mode/phase capture: {mode}/{phase}")
        seen.add((mode, phase))
        if not activities:
            raise ValueError(f"{path}: no GPU activities")
        _assign_scopes(apis, scopes)
        _attribute(apis, activities)
        graph_audit = None
        capture_calls = [call for call in calls if (call["mode"], call["phase"]) == (mode, phase)]
        if any(call.get("graph_replay") for call in capture_calls):
            if parents is None:
                raise ValueError("graph replay timing requires its setup capture lineage")
            from experiments.deepseek_v32_motivation.src.graph_attribution import (
                attribute_graph_replays,
            )

            graph_audit = attribute_graph_replays(
                activities, capture_calls, parents, scopes=scopes, require_replays=True
            )
            virtual_scopes = {}
            for call in capture_calls:
                if call.get("graph_api"):
                    virtual_scopes[call["nvtx"]] = {
                        **call,
                        "label": call["nvtx"],
                        "id": len(scopes) + len(virtual_scopes),
                        "start": 0,
                        "end": 0,
                        "thread": 0,
                    }
            for activity in activities:
                if activity.get("graph_call") is not None:
                    activity["scope"] = virtual_scopes[activity["graph_call"]["nvtx"]]
            scopes.extend(virtual_scopes.values())
        buckets = defaultdict(lambda: {"scopes": [], "apis": [], "activities": []})
        for scope in scopes:
            if scope["stage"] is not None:
                buckets[_key(scope)]["scopes"].append(scope)
        for api in apis:
            if api["scope"]:
                buckets[_key(api["scope"])]["apis"].append(api)
        for activity in activities:
            if activity["scope"]:
                buckets[_key(activity["scope"])]["activities"].append(activity)
        for key in calls_by_key:
            if key[:2] == (mode, phase):
                buckets[key]  # Preserve metadata calls that have no matching NVTX scope.
        local_rows = []
        for key, data in sorted(buckets.items()):
            data_calls = calls_by_key[key]
            row = _operator_row(key, **data, calls=data_calls, peaks=peaks)
            local_rows.append(row)
            aggregate_key = key[:2] + ("all_layers", key[3])
            for field in ("scopes", "apis", "activities"):
                aggregate[aggregate_key][field].extend(data[field])
            aggregate[aggregate_key]["calls"].extend(data_calls)
        rows.extend(local_rows)
        current_inventory = _kernel_inventory(activities, mode, phase, str(path))
        inventory.extend(current_inventory)
        kernels = [row for row in activities if row["kind"] == "kernel"]
        unassigned = [row for row in activities if row["scope"] is None]
        unassigned_kernels = [row for row in unassigned if row["kind"] == "kernel"]
        devices, envelope = _device_summary(activities)
        by_kind = {}
        for kind in ("kernel", "memcpy", "memset"):
            selected = [row for row in activities if row["kind"] == kind]
            assigned = [row for row in selected if row["scope"] is not None]
            by_kind[kind] = {
                "count": len(selected),
                "attributed_count": len(assigned),
                "unattributed_count": len(selected) - len(assigned),
                "total_ns": _sum_ns(selected),
                "attributed_ns": _sum_ns(assigned),
                "unattributed_ns": _sum_ns(selected) - _sum_ns(assigned),
            }
        mismatches = [
            {field: row[field] for field in (*_KEY, "scope_count", "metadata_call_count")}
            for row in local_rows
            if row["stage"] != "layer_unscoped" and not row["metadata_call_count_matches"]
        ]
        kernel_ns = _sum_ns(kernels)
        operator_ns = sum(row["kernel_ns"] for row in local_rows)
        inventory_ns = sum(row["total_ns"] for row in current_inventory)
        conserved = operator_ns + _sum_ns(unassigned_kernels) == kernel_ns == inventory_ns and sum(
            row["kernel_count"] for row in local_rows
        ) + len(unassigned_kernels) == len(kernels) == sum(
            row["count"] for row in current_inventory
        )
        if not conserved:
            raise RuntimeError(
                "kernel inventory or operator attribution failed time/count conservation"
            )
        captures.append(
            {
                "mode": mode,
                "phase": phase,
                "sqlite": str(path),
                "sqlite_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "tables_read": tables,
                "devices": devices,
                "capture_gpu_envelope": envelope,
                "api_summary": _api_summary(apis),
                "graph_attribution": graph_audit,
                "audit": {
                    "kernel_count_and_time_conserved": conserved,
                    "activity_counts_and_ns": by_kind,
                    "operator_kernel_ns": operator_ns,
                    "inventory_kernel_ns": inventory_ns,
                    "metadata_call_count_mismatches": mismatches,
                    "metadata_call_counts_match": not mismatches,
                    "unattributed_reasons": dict(
                        Counter(row["unattributed_reason"] for row in unassigned)
                    ),
                    "layer_unscoped_kernel_count": sum(
                        row["kernel_count"]
                        for row in local_rows
                        if row["stage"] == "layer_unscoped"
                    ),
                    "excluded_event_api_count": sum(row["bookkeeping"] for row in apis),
                },
            }
        )
    unused_calls = [call for call in calls if (call["mode"], call["phase"]) not in seen]
    pooled = [_operator_row(key, **data, peaks=peaks) for key, data in sorted(aggregate.items())]
    invalid_keys = {
        (row["mode"], row["phase"], row["stage"])
        for row in rows
        if row["stage"] != "layer_unscoped" and not row["metadata_call_count_matches"]
    }
    for row in pooled:
        if (row["mode"], row["phase"], row["stage"]) in invalid_keys:
            # Missing calls in one layer cannot cancel extra calls in another.
            row["metadata_call_count_matches"] = False
            row["kernel_mfu_percent"] = None
            row["mfu_unavailable_reason"] = "scope_metadata_call_count_mismatch"
    return {
        "schema_version": 1,
        "ledger_metadata": metadata or {},
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "attribution_source_sha256": hashlib.sha256(
            Path(__file__).with_name("analyze_nsys.py").read_bytes()
        ).hexdigest(),
        "dense_peaks_tflops": peaks,
        "peak_reference": "https://www.nvidia.com/en-us/data-center/h200/",
        "notes": _NOTES,
        "captures": captures,
        "operators_by_layer": rows,
        "operators": pooled,
        "kernel_inventory": inventory,
        "calls_outside_selected_captures": len(unused_calls),
    }


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, action="append", required=True)
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--graph-setup", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--peak-fp8", type=float, default=_DEFAULT_PEAKS["FP8"])
    parser.add_argument("--peak-bf16", type=float, default=_DEFAULT_PEAKS["BF16"])
    parser.add_argument("--peak-fp32", type=float, default=_DEFAULT_PEAKS["FP32"])
    args = parser.parse_args()
    calls, metadata = read_calls(args.calls)
    report = analyze_captures(
        args.sqlite,
        calls,
        metadata=metadata,
        graph_setup_paths=args.graph_setup,
        peaks={"FP8": args.peak_fp8, "BF16": args.peak_bf16, "FP32": args.peak_fp32},
    )
    outputs = [
        args.output_dir / f"{name}.csv"
        for name in ("operators_by_layer", "operators", "kernel_inventory")
    ]
    outputs.append(args.output_dir / "analysis.json")
    inputs = [path.resolve() for path in (*args.sqlite, args.calls)]
    if any(path.exists() for path in outputs):
        parser.error("output files already exist; choose a fresh output directory")
    if any(path.resolve() in inputs for path in outputs):
        parser.error("analysis output must not overwrite an input")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report["calls_sha256"] = hashlib.sha256(args.calls.read_bytes()).hexdigest()
    (args.output_dir / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in ("operators_by_layer", "operators", "kernel_inventory"):
        _write_csv(args.output_dir / f"{name}.csv", report[name])
    print(json.dumps({"output_dir": str(args.output_dir), "captures": len(report["captures"])}))


if __name__ == "__main__":
    main()
