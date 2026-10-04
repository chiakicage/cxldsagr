"""Calculate operator MFU from captured calls and correlated GPU activities."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from experiments.deepseek_v32_echo_prefill.src.analyze_nsys import (
    _assign_scopes,
    _attribute,
    _read_capture,
    _union_ns,
)
from experiments.deepseek_v32_motivation.src.analyze_pipeline import (
    PATTERN,
    is_matrix_kernel,
    sha,
    write_csv,
    write_json,
)
from experiments.deepseek_v32_motivation.src.graph_attribution import (
    attribute_graph_replays,
    profile_graph_inputs,
)

IDENTITY = ("capture_index", "scheme", "phase", "segment", "stage", "precision")


def primary_kernel(name):
    """Identify matrix kernels; keep split-K output reductions in the API only."""
    return is_matrix_kernel(name)


def bind_scopes(scopes, calls):
    """Bind unique matrix labels and inherit their ownership into helper ranges."""
    labels = {call["nvtx"]: call for call in calls}
    if len(labels) != len(calls):
        raise ValueError("matrix call labels must be unique within a capture")
    by_thread = defaultdict(list)
    for scope in scopes:
        by_thread[scope["thread"]].append(scope)
    owners, matched = {}, set()
    for ranges in by_thread.values():
        stack = []
        for scope in sorted(ranges, key=lambda s: (s["start"], -s["end"], s["id"])):
            while stack and stack[-1]["end"] <= scope["start"]:
                stack.pop()
            inherited = owners[stack[-1]["id"]] if stack else None
            if stack and scope["end"] > stack[-1]["end"]:
                raise ValueError("crossing CPU scopes")
            if scope["label"] in labels:
                if inherited is not None or scope["label"] in matched:
                    raise ValueError("nested or repeated matrix scope would double count work")
                matched.add(scope["label"])
                owners[scope["id"]] = labels[scope["label"]]
            else:
                owners[scope["id"]] = inherited
            stack.append(scope)
    if matched != set(labels):
        raise ValueError("matrix ledger and NVTX scopes differ")
    return owners


def invocation_row(call, activities, peaks):
    if not activities:
        raise ValueError(f"matrix invocation has no GPU work: {call['nvtx']}")
    if len({a["device_id"] for a in activities}) != 1:
        raise ValueError("operator MFU requires one GPU")
    kernels = [a for a in activities if a["kind"] == "kernel"]
    core = [a for a in kernels if primary_kernel(a["name"])]
    if len(core) != 1:
        raise ValueError(f"expected one matrix entry point in this workload: {call['nvtx']}")
    precision = call["precision"].lower()
    ideal_ns = call["useful_flops"] / (peaks[precision] * 1e3)
    return {
        **{key: call[key] for key in IDENTITY},
        "chunk": call["chunk"],
        "layer": call["layer"],
        "call_id": call["call_id"],
        "useful_flops": call["useful_flops"],
        "executed_matmul_flops": call["executed_matmul_flops"],
        "ideal_ns": ideal_ns,
        "primary_kernel_count": len(core),
        "gpu_kernel_count": len(kernels),
        "gpu_activity_count": len(activities),
        "primary_kernel_ns": _union_ns((a["start"], a["end"]) for a in core),
        "operator_gpu_active_ns": _union_ns((a["start"], a["end"]) for a in activities),
        "operator_gpu_span_ns": max(a["end"] for a in activities)
        - min(a["start"] for a in activities),
        "operator_gpu_duration_sum_ns": sum(a["end"] - a["start"] for a in activities),
        "dimensions": call["dimensions"],
        "nvtx": call["nvtx"],
    }


def summarize(rows, *, by_layer=False):
    groups = defaultdict(list)
    keys = IDENTITY + (("layer",) if by_layer else ())
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity, group in sorted(groups.items()):
        total = {
            key: sum(row[key] for row in group)
            for key in (
                "useful_flops",
                "ideal_ns",
                "primary_kernel_count",
                "gpu_kernel_count",
                "gpu_activity_count",
                "primary_kernel_ns",
                "operator_gpu_active_ns",
                "operator_gpu_span_ns",
                "operator_gpu_duration_sum_ns",
            )
        }
        result = dict(zip(keys, identity, strict=True))
        result.update(
            calls=len(group),
            useful_flops=total["useful_flops"],
            executed_matmul_flops=(
                sum(row["executed_matmul_flops"] for row in group)
                if all(row["executed_matmul_flops"] is not None for row in group)
                else None
            ),
            ideal_ms=total["ideal_ns"] / 1e6,
            primary_kernel_count=total["primary_kernel_count"],
            gpu_kernel_count=total["gpu_kernel_count"],
            gpu_activity_count=total["gpu_activity_count"],
        )
        for name in ("primary_kernel", "operator_gpu_active", "operator_gpu_span"):
            duration = total[f"{name}_ns"]
            result[f"{name}_ms"] = duration / 1e6
            result[f"{name}_mfu_pct"] = 100 * total["ideal_ns"] / duration
        result["operator_gpu_duration_sum_ms"] = total["operator_gpu_duration_sum_ns"] / 1e6
        output.append(result)
    return output


def analyze(profile, flops, output):
    metadata = json.loads((profile / "metadata.json").read_text())
    result = json.loads((profile / "result.json").read_text())
    verified = json.loads((profile / "analysis/flops_verification.json").read_text())
    work = json.loads(flops.read_text())
    if not result["accepted"] or not verified["passed"] or verified["mismatches"]:
        raise ValueError("accepted profile and verified matrix work are required")
    if work["source_run_id"] != metadata["reference_run_id"]:
        raise ValueError("profile and FLOPs refer to different formal runs")
    verified_inputs = {Path(name).name: value for name, value in verified["input_sha256"].items()}
    for path in (profile / "operator_calls.json", profile / "metadata.json", flops):
        if sha(path) != verified_inputs[path.name]:
            raise ValueError(f"verified input changed: {path}")
    calls = json.loads((profile / "operator_calls.json").read_text())
    matrix = [call for call in calls if call["useful_flops"] is not None]
    peaks = work["dense_peaks_tflops"]
    rows, audits, inventory = [], [], []
    graph_parents, graph_paths = {}, []
    if metadata.get("graph_setup_captures"):
        _, graph_parents, graph_paths = profile_graph_inputs(profile)
    for capture in metadata["captures"]:
        path = profile / capture["sqlite"]
        scopes, apis, activities, _ = _read_capture(path.resolve(), scope_pattern=PATTERN)
        _assign_scopes(apis, scopes)
        _attribute(apis, activities)
        if any(a["scope"] is None for a in activities):
            raise ValueError("unattributed GPU activity in capture")
        selected = [call for call in matrix if call["capture_index"] == capture["capture_index"]]
        capture_calls = [
            call for call in calls if call["capture_index"] == capture["capture_index"]
        ]
        graph_audit = attribute_graph_replays(
            activities,
            capture_calls,
            graph_parents,
            scopes=scopes,
            require_replays=metadata["config"].get("enable_compute_graphs", False),
        )
        owners = bind_scopes(scopes, [call for call in selected if not call.get("graph_api")])
        grouped, kernels = defaultdict(list), defaultdict(list)
        for activity in activities:
            call = activity.get("graph_call") or owners[activity["scope"]["id"]]
            if call is not None:
                grouped[call["call_id"]].append(activity)
                if activity["kind"] == "kernel":
                    kernels[(call["stage"], activity["name"])].append(activity)
        capture_rows = [invocation_row(call, grouped[call["call_id"]], peaks) for call in selected]
        rows.extend(capture_rows)
        for (stage, name), group in sorted(kernels.items()):
            inventory.append(
                {
                    "capture_index": capture["capture_index"],
                    "scheme": capture["scheme"],
                    "phase": capture["phase"],
                    "stage": stage,
                    "kernel": name,
                    "primary_matrix_kernel": primary_kernel(name),
                    "count": len(group),
                    "duration_sum_ms": sum(a["end"] - a["start"] for a in group) / 1e6,
                }
            )
        audits.append(
            {
                "capture_index": capture["capture_index"],
                "scheme": capture["scheme"],
                "phase": capture["phase"],
                "sqlite": str(path),
                "sqlite_sha256": sha(path),
                "matrix_calls": len(selected),
                "bound_matrix_calls": len(capture_rows),
                "matrix_scope_gpu_activity_count": sum(len(group) for group in grouped.values()),
                "call_activity_count_conserved": sum(
                    row["gpu_activity_count"] for row in capture_rows
                )
                == sum(len(group) for group in grouped.values()),
                "graph_attribution": graph_audit,
            }
        )
        print(
            f"operator MFU: {capture['scheme']} {capture['phase']}, {len(selected)} calls",
            flush=True,
        )
    if len(rows) != verified["matrix_calls"]:
        raise ValueError("not all verified matrix invocations were processed")
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    write_csv(output / "operator_mfu.csv", summary)
    write_csv(output / "operator_mfu_by_layer.csv", summarize(rows, by_layer=True))
    write_csv(output / "operator_kernel_inventory.csv", inventory)
    with (output / "operator_mfu_calls.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    write_json(
        output / "operator_mfu_summary.json",
        {
            "analysis_run_id": output.name,
            "profile_run_id": profile.name,
            "formal_run_id": metadata["reference_run_id"],
            "matrix_calls": len(rows),
            "operator_groups": len(summary),
            "captures": audits,
            "dense_peaks_tflops": peaks,
            "precision_settings": metadata["precision_settings"],
            "peak_reference": work["peak_reference"],
            "input_sha256": {
                str(p): sha(p)
                for p in (
                    profile / "operator_calls.json",
                    profile / "metadata.json",
                    profile / "result.json",
                    profile / "analysis/flops_verification.json",
                    flops,
                    *graph_paths,
                )
            },
            "analysis_source_sha256": {
                str(p.resolve()): sha(p)
                for p in (
                    Path(__file__),
                    Path(__file__).with_name("analyze_pipeline.py"),
                    Path(__file__).with_name("graph_attribution.py"),
                    Path("experiments/deepseek_v32_echo_prefill/src/analyze_nsys.py"),
                )
            },
            "definitions": {
                "mfu": "100 * sum(useful FLOPs / precision dense peak) / sum(per-call duration); never mean individual percentages",
                "primary_kernel": "Actual matrix kernels only; split-K reductions are helpers. Fused ECHO includes inseparable fetch/reduction work.",
                "operator_gpu_active": "Per-call union of all attributed GPU kernel/memcpy/memset activities, including helpers; summed across calls.",
                "operator_gpu_span": "Per-call first GPU activity start to last GPU activity end, including internal gaps; summed across calls.",
                "attribution": "CPU launch correlation to innermost NVTX, inheriting eager matrix ownership into descendants. Graph APIs use capture-time node sets and recorded executable clone lineage; no CPU/GPU timestamp clipping.",
            },
            "boundaries": [
                "Existing instrumented Nsight captures, one cold and one revisit per scheme. No new GPU measurements.",
                "Operator GPU-active MFU excludes host overhead and gaps. It is not formal request MFU or a Tensor Core hardware counter.",
                "ECHO fused duration includes internal host loads; no pure-QK throughput or internal overlap is inferred.",
                "All ten independent blocks count; useful FLOPs exclude known padding, scalar math and copies. Unknown executed GEMM FLOPs remain null.",
                "Top-k, norms, cache metadata and standalone fetch have no matrix MFU; their times remain in the prior stage report.",
            ],
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-run", type=Path, required=True)
    parser.add_argument("--flops", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    analyze(args.profile_run, args.flops, args.output_dir)


if __name__ == "__main__":
    main()
