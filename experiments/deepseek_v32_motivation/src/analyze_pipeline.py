"""Attribute diagnostic Nsight captures to actual launches and draw stream timelines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from experiments.deepseek_v32_echo_prefill.src.analyze_nsys import (
    _assign_scopes,
    _attribute,
    _read_capture,
    _union,
    _union_ns,
)
from experiments.deepseek_v32_motivation.src.graph_attribution import (
    attribute_graph_replays,
    profile_graph_inputs,
)

PATTERN = re.compile(
    r"^motivation/(?P<scheme>[^/]+)/(?P<phase>cold|revisit)/(?P<segment>[^/]+)/"
    r"chunk_(?P<chunk>[^/]+)/layer_(?P<layer>[^/]+)/(?P<stage>[^/]+)$"
)
COLORS = {
    "matrix_compute": "#0072B2",
    "indexer_prefetch_fused": "#CC79A7",
    "host_gather": "#D55E00",
    "D2H": "#E69F00",
    "H2D": "#009E73",
    "cache_metadata": "#999999",
    "model_auxiliary": "#56B4E9",
    "other_copy": "#F0E442",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def is_matrix_kernel(name):
    """Recognize measured matrix entry points, not the deep_gemm namespace."""
    name = name.lower()
    if "execute_split_k_kernel" in name or "splitkreduce_kernel" in name:
        return False
    return any(
        token in name
        for token in (
            "sm90_fp8_gemm_1d2d_impl",
            "sm90_fp8_mqa_logits",
            "sparse_attn_fwd_kernel",
            "nvjet_",
            "_xmma_gemm_",
            "_simt_sgemm_",
            "cutlass_80_wmma_tensorop_",
            "gemmsn_tn_kernel<",
        )
    )


def category(activity):
    name = activity["name"].lower()
    scope = activity.get("scope") or {}
    stage = scope.get("stage", "").lower()
    if activity["kind"] == "memcpy":
        if "dtoh" in name or "device-to-host" in name or "device to host" in name:
            return "D2H"
        if "htod" in name or "host-to-device" in name or "host to device" in name:
            return "H2D"
        return "other_copy"
    if "mqa_logits_fuse_prefetch" in name:
        return "indexer_prefetch_fused"
    if "gather_records" in name:
        return "host_gather"
    if is_matrix_kernel(name):
        return "matrix_compute"
    if any(
        x in stage
        for x in ("cache", "pool", "slots", "host", "recall", "prefetch", "logical_to_global")
    ):
        return "cache_metadata"
    return "model_auxiliary"


def intersection_ns(left, right):
    left, right = _union(left), _union(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        total += max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] < b[1]:
            i += 1
        else:
            j += 1
    return total


def exclusive_scope_ns(scopes):
    """Subtract nested ranges on each CPU thread; retain innermost attribution."""
    by_thread = defaultdict(list)
    children = defaultdict(list)
    for scope in scopes:
        by_thread[scope["thread"]].append(scope)
    for values in by_thread.values():
        active = []
        for scope in sorted(values, key=lambda s: (s["start"], -s["end"], s["id"])):
            while active and active[-1]["end"] <= scope["start"]:
                active.pop()
            if active:
                parent = active[-1]
                if scope["end"] > parent["end"]:
                    raise ValueError("crossing NVTX ranges cannot be treated as nested scopes")
                children[parent["id"]].append((scope["start"], scope["end"]))
            active.append(scope)
    return {
        scope["id"]: scope["end"] - scope["start"] - _union_ns(children[scope["id"]])
        for scope in scopes
    }


def grouped_activities(activities, field):
    groups = defaultdict(list)
    for activity in activities:
        groups[field(activity)].append(activity)
    return [
        {
            "group": label,
            "count": len(group),
            "duration_sum_ms": sum(a["end"] - a["start"] for a in group) / 1e6,
            "duration_union_ms": _union_ns((a["start"], a["end"]) for a in group) / 1e6,
            "memcpy_bytes": sum(a.get("bytes") or 0 for a in group if a["kind"] == "memcpy"),
        }
        for label, group in sorted(groups.items())
    ]


def activity_summary(activities):
    busy = _union_ns((a["start"], a["end"]) for a in activities)
    span = max(a["end"] for a in activities) - min(a["start"] for a in activities)
    return {
        "gpu_activity_count": len(activities),
        "gpu_kernel_count": sum(a["kind"] == "kernel" for a in activities),
        "gpu_busy_union_ms": busy / 1e6,
        "gpu_span_ms": span / 1e6,
        "gpu_gap_ms": (span - busy) / 1e6,
        "gpu_categories": grouped_activities(activities, lambda a: a["category"]),
    }


def analyze_capture(path):
    path = Path(path).resolve(strict=True)
    scopes, apis, activities, tables = _read_capture(path, scope_pattern=PATTERN)
    if not scopes or not activities:
        raise ValueError(f"missing motivation scopes or GPU activities: {path}")
    if len({a["device_id"] for a in activities}) != 1:
        raise ValueError("pipeline analysis requires a single GPU capture")
    if any(a["kind"] == "memcpy" and a["name"].startswith("copy_kind_") for a in activities):
        raise ValueError("memcpy kind labels must be resolved before transfer analysis")
    identities = {(s["scheme"], s["phase"]) for s in scopes}
    if len(identities) != 1:
        raise ValueError("each capture must contain one scheme/request phase")
    scheme, phase = identities.pop()
    _assign_scopes(apis, scopes)
    _attribute(apis, activities)
    graph_audit = None
    graph_calls = []
    graph_paths = []
    metadata_path = path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if metadata.get("config", {}).get("enable_compute_graphs") or any(
        activity.get("graph_id") or activity.get("graph_node_id") for activity in activities
    ):
        graph_calls, parents, graph_paths = profile_graph_inputs(path.parent)
        captures = [row for row in metadata["captures"] if row["sqlite"] == path.name]
        if len(captures) != 1 or (captures[0]["scheme"], captures[0]["phase"]) != (scheme, phase):
            raise ValueError("graph capture identity differs from profile metadata")
        graph_calls = [
            row for row in graph_calls if row["capture_index"] == captures[0]["capture_index"]
        ]
        graph_audit = attribute_graph_replays(
            activities, graph_calls, parents, scopes=scopes, require_replays=True
        )
        graph_paths = [metadata_path, path.parent / "operator_calls.json", *graph_paths]
    for activity in activities:
        activity["category"] = category(activity)
    exclusive = exclusive_scope_ns(scopes)
    cpu_ranges = defaultdict(list)
    for scope in scopes:
        cpu_ranges[scope["thread"]].append((scope["start"], scope["end"]))
    cpu_union_ns = sum(_union_ns(ranges) for ranges in cpu_ranges.values())
    if sum(exclusive.values()) != cpu_union_ns:
        raise ValueError("exclusive CPU ranges do not conserve per-thread NVTX unions")
    stages = defaultdict(
        lambda: {
            "calls": 0,
            "cpu_inclusive_ns": 0,
            "cpu_exclusive_ns": 0,
            "gpu_count": 0,
            "gpu_ns": 0,
            "api_intervals": [],
        }
    )
    for scope in scopes:
        row = stages[(scope["segment"], scope["stage"])]
        row["calls"] += 1
        row["cpu_inclusive_ns"] += scope["end"] - scope["start"]
        row["cpu_exclusive_ns"] += exclusive[scope["id"]]
    for activity in activities:
        scope = activity["scope"]
        key = (
            (scope["segment"], activity.get("graph_stage", scope["stage"]))
            if scope
            else ("unattributed", "unattributed")
        )
        stages[key]["gpu_count"] += 1
        stages[key]["gpu_ns"] += activity["end"] - activity["start"]
    for call in graph_calls:
        if call.get("graph_api"):
            stages[call["segment"], call["stage"]]["calls"] += 1
    for api in apis:
        scope = api["scope"]
        if scope:
            stages[(scope["segment"], scope["stage"])]["api_intervals"].append(
                (api["start"], api["end"])
            )
    stage_rows = [
        {
            "scheme": scheme,
            "phase": phase,
            "segment": key[0],
            "stage": key[1],
            "calls": row["calls"],
            "cpu_inclusive_ms": row["cpu_inclusive_ns"] / 1e6,
            "cpu_exclusive_ms": row["cpu_exclusive_ns"] / 1e6,
            "gpu_activity_count": row["gpu_count"],
            "gpu_duration_sum_ms": row["gpu_ns"] / 1e6,
            "cuda_api_union_ms": _union_ns(row["api_intervals"]) / 1e6,
        }
        for key, row in sorted(stages.items())
    ]
    api_groups = defaultdict(list)
    for api in apis:
        api_groups[(api["source"], api["name"])].append(api)
    api_rows = [
        {
            "scheme": scheme,
            "phase": phase,
            "source": source,
            "name": name,
            "count": len(values),
            "duration_sum_ms": sum(a["end"] - a["start"] for a in values) / 1e6,
        }
        for (source, name), values in sorted(api_groups.items())
    ]
    busy = _union_ns((a["start"], a["end"]) for a in activities)
    gpu_start = min(a["start"] for a in activities)
    gpu_end = max(a["end"] for a in activities)
    compute = [(a["start"], a["end"]) for a in activities if a["category"] == "matrix_compute"]
    transfers = [
        (a["start"], a["end"]) for a in activities if a["category"] in ("D2H", "H2D", "host_gather")
    ]
    summary = {
        "scheme": scheme,
        "phase": phase,
        "sqlite": str(path),
        "sqlite_sha256": sha(path),
        "tables": tables,
        "gpu_activity_count": len(activities),
        "gpu_kernel_count": sum(a["kind"] == "kernel" for a in activities),
        "gpu_busy_union_ms": busy / 1e6,
        "gpu_span_ms": (gpu_end - gpu_start) / 1e6,
        "gpu_gap_ms": (gpu_end - gpu_start - busy) / 1e6,
        "cpu_nvtx_span_ms": (max(s["end"] for s in scopes) - min(s["start"] for s in scopes)) / 1e6,
        "cpu_exclusive_sum_ms": sum(exclusive.values()) / 1e6,
        "cpu_exclusive_conserves_thread_unions": True,
        "unattributed_gpu_count": sum(a["scope"] is None for a in activities),
        "graph_attribution": graph_audit,
        "graph_input_sha256": {str(p): sha(p) for p in graph_paths},
        "nonfused_matrix_and_separate_transfer_overlap_ms": intersection_ns(compute, transfers)
        / 1e6,
        "gpu_categories": grouped_activities(activities, lambda a: a["category"]),
        "gpu_kernels": grouped_activities(
            [a for a in activities if a["kind"] == "kernel"], lambda a: a["name"]
        ),
        "gpu_streams": grouped_activities(activities, lambda a: str(a["stream_id"])),
        "segments": {
            segment: activity_summary(
                [a for a in activities if a["scope"] and a["scope"]["segment"] == segment]
            )
            for segment in sorted({a["scope"]["segment"] for a in activities if a["scope"]})
        },
        "stage_totals_conserve_gpu_count": sum(r["gpu_activity_count"] for r in stage_rows)
        == len(activities),
        "stage_totals_conserve_gpu_ns": sum(r["gpu_ns"] for r in stages.values())
        == sum(a["end"] - a["start"] for a in activities),
    }
    return summary, stage_rows, api_rows, activities


def draw_timeline(summary, activities, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    history = [
        a
        for a in activities
        if a["scope"] and a["scope"]["segment"] == "history" and a["scope"]["chunk"].isdigit()
    ]
    if summary["phase"] == "cold" and history:
        chunk = max(int(a["scope"]["chunk"]) for a in history)
        focus = [a for a in history if int(a["scope"]["chunk"]) == chunk]
        title = f"Last history chunk ({chunk}); all 10 layers"
    else:
        focus = [a for a in activities if a["scope"] and a["scope"]["segment"] == "candidate"]
        title = "Candidate; all 10 layers"
    if not focus:
        raise ValueError("timeline has no requested chunk/segment GPU work")
    lo, hi = min(a["start"] for a in focus), max(a["end"] for a in focus)
    streams = sorted({a["stream_id"] for a in activities}, key=str)
    first_layers = [a for a in focus if a["scope"]["layer"] in ("0", "1")]
    detail_lo = min(a["start"] for a in first_layers)
    detail_hi = max(a["end"] for a in first_layers)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), constrained_layout=True)
    origin = min(a["start"] for a in activities)
    for axis, chosen, start, heading in (
        (axes[0], activities, origin, "Whole captured request"),
        (axes[1], [a for a in activities if a["end"] >= lo and a["start"] <= hi], lo, title),
        (
            axes[2],
            [a for a in activities if a["end"] >= detail_lo and a["start"] <= detail_hi],
            detail_lo,
            "First two layers of the window above (actual GPU execution)",
        ),
    ):
        for y, stream in enumerate(streams):
            for label, color in COLORS.items():
                spans = [
                    (
                        max(a["start"], start) / 1e6 - start / 1e6,
                        (a["end"] - max(a["start"], start)) / 1e6,
                    )
                    for a in chosen
                    if a["stream_id"] == stream and a["category"] == label
                ]
                axis.broken_barh(spans, (y - 0.35, 0.7), facecolors=color, rasterized=True)
        axis.set_yticks(range(len(streams)), [f"CUDA stream {s}" for s in streams])
        axis.set_xlabel("GPU activity time (ms)")
        axis.set_title(heading)
        axis.grid(axis="x", alpha=0.2)
        axis.set_xlim(0, max(a["end"] for a in chosen) / 1e6 - start / 1e6)
        if axis is axes[2]:
            for a in chosen:
                scope = a["scope"] or {}
                if scope.get("layer") not in ("0", "1", "2"):
                    continue
                label = None
                if a["category"] == "host_gather":
                    label = "fetch"
                elif a["category"] == "indexer_prefetch_fused":
                    label = "index+fetch"
                elif "mqa_logits" in a["name"]:
                    label = "index"
                elif "sparse_attn_fwd" in a["name"]:
                    label = "MLA"
                if label:
                    x = (max(a["start"], start) - start) / 1e6
                    y = streams.index(a["stream_id"])
                    axis.annotate(
                        f"L{scope['layer']} {label}",
                        (x, y + 0.36),
                        xytext=(2, 3),
                        textcoords="offset points",
                        fontsize=8,
                        rotation=25,
                        va="bottom",
                    )
            axis.set_ylim(-0.5, len(streams) - 0.15)
    fig.suptitle(f"{summary['scheme']} / {summary['phase']} — diagnostic Nsight capture")
    fig.legend(
        handles=[Patch(facecolor=c, label=k) for k, c in COLORS.items()],
        loc="outside lower center",
        ncol=4,
        fontsize=8,
    )
    for extension in ("png", "svg"):
        fig.savefig(
            output / f"pipeline_{summary['scheme']}_{summary['phase']}.{extension}", dpi=170
        )
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    captures, stages, apis = [], [], []
    for path in args.sqlite:
        summary, stage_rows, api_rows, activities = analyze_capture(path)
        captures.append(summary)
        stages.extend(stage_rows)
        apis.extend(api_rows)
        draw_timeline(summary, activities, args.output_dir)
    write_csv(args.output_dir / "stage_costs.csv", stages)
    write_csv(args.output_dir / "cuda_apis.csv", apis)
    write_json(
        args.output_dir / "pipeline.json",
        {
            "captures": captures,
            "analysis_sources": {
                str(p): sha(p)
                for p in (
                    Path(__file__),
                    Path(__file__).with_name("graph_attribution.py"),
                    Path("experiments/deepseek_v32_echo_prefill/src/analyze_nsys.py"),
                )
            },
            "boundaries": [
                "Diagnostic instrumented captures; use the accepted uninstrumented run for latency and MFU.",
                "GPU attribution follows runtime/driver launch correlation to the innermost CPU NVTX scope.",
                "Graph matrix attribution follows captured API node membership and recorded executable clone lineage; only actual GPU intervals contribute duration.",
                "CPU exclusive scopes subtract children; CPU inclusive times must not be summed.",
                "CPU exclusive wall time includes blocking synchronization APIs; it is not pure CPU compute time.",
                "GPU busy/gap are activity unions, not SM utilization; sums may include concurrent streams.",
                "Fused ECHO indexer/prefetch is one indivisible measured kernel; no internal fetch overlap is inferred.",
                "The separate-transfer overlap number excludes the entire fused ECHO kernel from matrix intervals.",
                "D2H/H2D memcpy bytes exclude mapped-host loads inside gather and fused kernels.",
                "Matrix kernels are classified by concrete GEMM/GEMV/MLA/indexer names, including cuBLAS nvjet.",
                "Segment GPU spans follow the attributed launches, not CPU scope clipping.",
                "API sums may overlap runtime/driver wrappers and are not an additive wall-time decomposition.",
            ],
        },
    )
    print(f"analyzed {len(captures)} captures: {args.output_dir}")


if __name__ == "__main__":
    main()
