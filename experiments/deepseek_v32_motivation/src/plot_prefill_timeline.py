"""Render one measured history-prefill layer from the published Nsight captures.

The window starts at the preceding block's final GPU activity in its finish
graph and ends at the selected block's final finish activity, within one token
chunk. Every intersecting GPU activity is retained, including asynchronous KV
writeback from another layer or chunk. Runtime correlation and CUDA-graph node
attribution are reused from the published pipeline analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from experiments.deepseek_v32_mfu.src.analyze_nsys import _union_ns
from experiments.deepseek_v32_motivation.src.analyze_pipeline import (
    analyze_capture,
    intersection_ns,
    sha,
)
from experiments.deepseek_v32_motivation.src.plot_single_layer import (
    COLORS,
    NAMES,
    SCHEMES,
    STAGES,
    display_group,
)

GROUP_COLORS = {**COLORS, "writeback": "#D55E00"}
H2D_CONTROL_STAGES = {"pool_stamp", "pool_operation", "pool_protect", "cache_operation"}


def verify_hashes(expected, description):
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError(f"{description} changed: {path}")


def classify(activity):
    """Separate persistent KV writeback from small control copies."""
    scope_stage = activity["scope"]["stage"]
    category = activity["category"]
    if category == "D2H":
        if scope_stage == "pool_write_host":
            return "writeback", "kv_d2h"
        return "aux", "control_d2h"
    if category == "H2D":
        if scope_stage == "host_dma":
            return "transfer", "kv_h2d_dma"
        role = "control_h2d" if scope_stage in H2D_CONTROL_STAGES else "other_h2d"
        return "aux", role
    if category == "host_gather":
        return "transfer", "kv_h2d_gather"
    if category == "indexer_prefetch_fused":
        return "fused", "fused_compute_and_kv_h2d"
    return display_group(activity), "device_only"


def group_metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    return {
        group: {
            "activity_count": len(values),
            "kernel_count": sum(row["kind"] == "kernel" for row in values),
            "streams": sorted({row["stream_id"] for row in values}),
            "layers": sorted({row["layer"] for row in values}),
            "start_ms": min(row["start_ms"] for row in values),
            "end_ms": max(row["end_ms"] for row in values),
            "duration_sum_ms": sum(row["visible_duration_ns"] for row in values) / 1e6,
            "duration_union_ms": _union_ns(
                (row["visible_start_ns"], row["visible_end_ns"]) for row in values
            )
            / 1e6,
            "whole_activity_memcpy_bytes": sum(
                row["bytes"] or 0 for row in values if row["kind"] == "memcpy"
            ),
        }
        for group, values in grouped.items()
    }


def extract(capture, chunk, layer):
    path = Path(capture["sqlite"])
    verify_hashes({path: capture["sqlite_sha256"]}, "raw capture")
    verify_hashes(capture["graph_input_sha256"], "graph attribution input")
    summary, _, _, activities = analyze_capture(path)
    if (summary["scheme"], summary["phase"]) != (capture["scheme"], "cold"):
        raise ValueError("prefill requires the matching cold-request capture")
    if summary["graph_input_sha256"] != capture["graph_input_sha256"]:
        raise ValueError("graph attribution inputs differ from the published analysis")
    if summary["graph_attribution"] != capture["graph_attribution"]:
        raise ValueError("graph attribution audit differs from the published analysis")
    metadata = json.loads((path.parent / "metadata.json").read_text())
    config = metadata["config"]
    history, chunk_size = config["history_tokens"], config["chunk_size"]
    chunks = math.ceil(history / chunk_size)
    selected_chunk = chunks - 1 if chunk == "last" else int(chunk)
    if not 0 <= selected_chunk < chunks:
        raise ValueError(f"chunk must be in [0, {chunks - 1}]")
    if not 1 <= layer < config["layers"]:
        raise ValueError("layer must have a preceding physical block")
    chunk_rows = [
        activity
        for activity in activities
        if activity["scope"]
        and activity["scope"]["segment"] == "history"
        and activity["scope"]["chunk"] == str(selected_chunk)
    ]

    def finish_end(which):
        prefix = f"compute_graph_finish_layer_{which}_"
        selected = [a for a in chunk_rows if a["scope"]["stage"].startswith(prefix)]
        if not selected:
            raise ValueError(f"no finish graph for chunk {selected_chunk}, layer {which}")
        return max(activity["end"] for activity in selected)

    origin, end = finish_end(layer - 1), finish_end(layer)
    if end <= origin:
        raise ValueError("prefill GPU window is empty or reversed")
    visible = [a for a in activities if a["end"] > origin and a["start"] < end]
    rows = []
    for activity in sorted(visible, key=lambda a: (a["start"], a["stream_id"], a["id"])):
        scope = activity["scope"]
        if not scope or scope["segment"] != "history":
            raise ValueError("prefill GPU window contains unattributed or non-history work")
        group, role = classify(activity)
        if group in (*STAGES, "fused") and (
            scope["layer"] != str(layer) or scope["chunk"] != str(selected_chunk)
        ):
            raise ValueError("window includes another layer/chunk's computation")
        start_ns, end_ns = max(origin, activity["start"]), min(end, activity["end"])
        rows.append(
            {
                "scheme": summary["scheme"],
                "phase": "cold",
                "segment": scope["segment"],
                "chunk": scope["chunk"],
                "layer": scope["layer"],
                "activity_id": activity["id"],
                "stream_id": activity["stream_id"],
                "correlation_id": activity["correlation"],
                "group": group,
                "transfer_role": role,
                "kind": activity["kind"],
                "category": activity["category"],
                "stage": activity.get("graph_stage", scope["stage"]),
                "scope_stage": scope["stage"],
                "start_ns": activity["start"],
                "end_ns": activity["end"],
                "visible_start_ns": start_ns,
                "visible_end_ns": end_ns,
                "visible_duration_ns": end_ns - start_ns,
                "start_ms": (start_ns - origin) / 1e6,
                "end_ms": (end_ns - origin) / 1e6,
                "clipped": start_ns != activity["start"] or end_ns != activity["end"],
                "bytes": activity.get("bytes"),
                "name": activity["name"],
            }
        )
    if len(rows) != len(visible) or len({row["activity_id"] for row in rows}) != len(rows):
        raise ValueError("visible activity conservation failed")
    groups = group_metrics(rows)
    if any(stage not in groups for stage in STAGES if not (stage == "index" and "fused" in groups)):
        raise ValueError("the layer window is missing a required compute stage")
    compute = [
        (row["visible_start_ns"], row["visible_end_ns"])
        for row in rows
        if row["group"] in STAGES and row["kind"] == "kernel"
    ]
    transfers = defaultdict(list)
    for row in rows:
        if row["transfer_role"] != "device_only":
            transfers[row["transfer_role"]].append(row)
    transfer_metrics = {}
    for role in (
        "kv_d2h",
        "kv_h2d_gather",
        "kv_h2d_dma",
        "fused_compute_and_kv_h2d",
        "control_d2h",
        "control_h2d",
        "other_h2d",
    ):
        values = transfers[role]
        intervals = [(row["visible_start_ns"], row["visible_end_ns"]) for row in values]
        transfer_metrics[role] = {
            "activity_count": len(values),
            "whole_activity_memcpy_bytes": sum(row["bytes"] or 0 for row in values),
            "kernel_transfer_bytes": None
            if any(row["kind"] == "kernel" for row in values)
            and "h2d" in role
            and "control" not in role
            else 0,
            "duration_union_ms": _union_ns(intervals) / 1e6,
            "compute_kernel_overlap_ms": (
                None
                if role == "fused_compute_and_kv_h2d" and values
                else intersection_ns(compute, intervals) / 1e6
            ),
            "streams": sorted({row["stream_id"] for row in values}),
            "layers": sorted({row["layer"] for row in values}),
        }
    case = next(c for c in metadata["captures"] if c["sqlite"] == path.name)
    token_start, token_end = (
        selected_chunk * chunk_size,
        min((selected_chunk + 1) * chunk_size, history),
    )
    result = {
        "scheme": summary["scheme"],
        "phase": "cold",
        "segment": "history",
        "request_id": case["request_id"],
        "chunk": selected_chunk,
        "layer": layer,
        "query_tokens": token_end - token_start,
        "token_start_inclusive": token_start,
        "token_end_inclusive": token_end - 1,
        "prefix_tokens_before_chunk": token_start,
        "history_tokens_after_prefill": history,
        "origin_ns": origin,
        "end_ns": end,
        "window_ms": (end - origin) / 1e6,
        "config": config,
        "hardware": metadata["hardware"],
        "profile_run_id": metadata["run_id"],
        "profile_source_sha256": metadata["source_sha256"],
        "sqlite": str(path),
        "sqlite_sha256": summary["sqlite_sha256"],
        "graph_input_sha256": summary["graph_input_sha256"],
        "graph_attribution": summary["graph_attribution"],
        "visible_activity_count": len(rows),
        "clipped_activity_count": sum(row["clipped"] for row in rows),
        "visible_activities_conserved": True,
        "visible_activity_layers": sorted({row["layer"] for row in rows}),
        "adjacent_layer_activity_count": sum(row["layer"] != str(layer) for row in rows),
        "groups": groups,
        "transfers": transfer_metrics,
        "whole_history_segment_layer_counters": case["segment_counters"]["history"]["layers"][
            layer
        ],
        "counter_scope": "History counters cover the entire prefill, not just this chunk/window.",
        "selected_layer_chunk_kv_writebacks": [
            {
                "start_ns": a["start"],
                "end_ns": a["end"],
                "bytes": a.get("bytes"),
                "stream_id": a["stream_id"],
                "scope_stage": a["scope"]["stage"],
            }
            for a in chunk_rows
            if a["scope"]["layer"] == str(layer) and classify(a)[0] == "writeback"
        ],
    }
    return result, rows


def draw(captures, rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "deepseek-history-prefill-layer",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
        }
    )
    first = captures[0]
    layer, chunk = first["layer"], first["chunk"]
    has_h2d = any(row["group"] in ("transfer", "fused") for row in rows)
    lane_y = {"compute": 3 if has_h2d else 2, "h2d": 2, "writeback": 1, "aux": 0}
    compute_y = lane_y["compute"]
    figure, axes = plt.subplots(4, 1, figsize=(13.6, 10.4), sharex=True)
    figure.subplots_adjust(left=0.145, right=0.975, top=0.835, bottom=0.205, hspace=0.55)
    figure.suptitle(
        "DeepSeek V3.2 | History prefill: one-layer timeline",
        x=0.035,
        ha="left",
        y=0.976,
        fontsize=19,
        fontweight="bold",
    )
    figure.text(
        0.035,
        0.937,
        f"Measured Nsight schedule · H200 / SM90 · Q={first['query_tokens']:,} · "
        f"chunk {chunk} · tokens {first['token_start_inclusive']:,}–{first['token_end_inclusive']:,}",
        fontsize=11,
    )
    figure.text(
        0.035,
        0.906,
        f"Prefix before chunk: {first['prefix_tokens_before_chunk']:,} tokens · "
        f"final history H={first['history_tokens_after_prefill']:,} · "
        f"pool P={first['config']['sparse_pool_tokens']:,}",
        fontsize=10.5,
        color="#4B5563",
    )
    figure.text(
        0.035,
        0.876,
        f"Physical block L{layer} (zero-based) · cold request {first['request_id']} · "
        f"each row starts at this chunk's L{layer - 1} GPU finish · same time scale",
        fontsize=10.5,
        color="#4B5563",
    )
    xmax = math.ceil(max(capture["window_ms"] for capture in captures) * 2) / 2 + 0.12
    for axis, capture, name in zip(axes, captures, NAMES, strict=True):
        chosen = [row for row in rows if row["scheme"] == capture["scheme"]]
        groups = defaultdict(list)
        for row in chosen:
            groups[row["group"]].append(row)
        compute_streams = sorted({row["stream_id"] for row in chosen if row["group"] in STAGES})
        axis.set_title(name, loc="left", fontsize=12, fontweight="bold", pad=20)
        axis.set_title(
            f"GPU window {capture['window_ms']:.3f} ms", loc="right", fontsize=10, pad=20
        )
        ticks = [compute_y, 1, 0]
        labels = [
            f"L{layer} compute\nstream {','.join(map(str, compute_streams))}",
            "GPU → host\nnew KV writeback",
            "Aux / cache\nD2D / control",
        ]
        if has_h2d:
            ticks.insert(1, 2)
            labels.insert(1, "Host → HBM\nhistory KV")
        axis.set_yticks(ticks, labels)
        axis.tick_params(axis="y", length=0, labelsize=9)
        axis.set_ylim(-0.43, compute_y + 0.72)
        axis.set_xlim(0, xmax)
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines["bottom"].set_color("#BBC1C9")
        for y in ticks:
            axis.axhline(y, color="#E5E7EB", linewidth=0.65, zorder=0)
        for group, values in groups.items():
            if group == "fused":
                for row in values:
                    axis.add_patch(
                        Rectangle(
                            (row["start_ms"], lane_y["h2d"] - 0.27),
                            row["end_ms"] - row["start_ms"],
                            1.54,
                            facecolor=GROUP_COLORS[group],
                            edgecolor="#603359",
                            lw=0.7,
                        )
                    )
                continue
            y = (
                lane_y["h2d"]
                if group == "transfer"
                else lane_y["writeback"]
                if group == "writeback"
                else lane_y["aux"]
                if group == "aux"
                else compute_y
            )
            axis.broken_barh(
                [(row["start_ms"], row["end_ms"] - row["start_ms"]) for row in values],
                (y - 0.26, 0.52),
                facecolors=GROUP_COLORS[group],
                linewidth=0,
            )
            if group in STAGES:
                lo, hi = (
                    min(row["start_ms"] for row in values),
                    max(row["end_ms"] for row in values),
                )
                axis.plot(
                    [lo, lo, hi, hi],
                    [compute_y + 0.36, compute_y + 0.43, compute_y + 0.43, compute_y + 0.36],
                    color="#374151",
                    lw=0.7,
                )
                axis.text(
                    (lo + hi) / 2,
                    compute_y + 0.47,
                    str(STAGES.index(group) + 1),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
        if not groups["writeback"]:
            axis.text(
                0.1,
                1,
                "KV remains in HBM; no KV writeback",
                va="center",
                fontsize=9,
                color="#6B7280",
            )
        for index, row in enumerate(groups["writeback"]):
            center = (row["start_ms"] + row["end_ms"]) / 2
            label = f"L{row['layer']} · {row['bytes'] / 2**20:.3f} MiB · stream {row['stream_id']}"
            text_x = min(center + 0.35, xmax - 1.1)
            axis.annotate(
                label,
                xy=(center, 1.02),
                xytext=(text_x, 1.18 - index * 0.29),
                ha="left",
                va="center",
                fontsize=9,
                color="#9D4100",
                arrowprops={"arrowstyle": "-", "lw": 0.7, "color": "#9D4100"},
            )
        for row in groups["transfer"]:
            axis.text(
                (row["start_ms"] + row["end_ms"]) / 2,
                lane_y["h2d"] - 0.34,
                f"L{row['layer']} gather · stream {row['stream_id']}",
                ha="center",
                va="top",
                fontsize=8.5,
                color="#8A5100",
            )
        axis.axvline(capture["window_ms"], color="#526071", linewidth=0.8, linestyle="--")
    axes[-1].set_xlabel(f"Time since this chunk's block L{layer - 1} GPU finish (ms)", labelpad=10)
    figure.text(
        0.145,
        0.139,
        "1  Norm + projections     2  Indexer logits     3  Exact top-k     "
        "4  Sparse MLA     5  V expansion + output + norm + MLP",
        fontsize=9.5,
    )
    figure.legend(
        handles=[
            Patch(color=GROUP_COLORS[key], label=label)
            for key, label in (
                ("projection", "Compute path"),
                ("writeback", "New KV: GPU → host"),
                ("aux", "Cache / D2D / control"),
            )
        ],
        loc="lower center",
        bbox_to_anchor=(0.55, 0.091),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    no_fetch = not any(row["group"] in ("transfer", "fused") for row in rows)
    if no_fetch:
        note = "No history KV H2D in these windows; ECHO uses its resident indexer. "
        note += "Small control H2D copies remain in Aux / cache."
    else:
        note = "Main-KV H2D DMA uses the copy engine; gathers and fused host reads use SMs. "
        note += "Control copies remain in Aux / cache."
    figure.text(0.035, 0.075, note, fontsize=9.5, color="#4B5563")
    figure.text(
        0.035,
        0.050,
        "All intersecting GPU activities and gaps are retained, including adjacent-layer "
        "asynchronous work. Writeback labels identify its owning layer.",
        fontsize=9.1,
        color="#4B5563",
    )
    figure.text(
        0.035,
        0.026,
        "One instrumented cold-request sample; these windows are not formal layer latencies. "
        + first["profile_run_id"],
        fontsize=9.1,
        color="#4B5563",
    )
    for suffix in ("svg", "pdf", "png"):
        metadata = {"Creator": "plot_prefill_timeline.py"}
        if suffix == "svg":
            metadata["Date"] = None
        elif suffix == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        figure.savefig(output / f"timeline.{suffix}", dpi=200, facecolor="white", metadata=metadata)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-json", type=Path, required=True)
    parser.add_argument(
        "--chunk", default="last", help="Zero-based history chunk, or last (default)."
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.chunk != "last" and (not args.chunk.isdigit() or int(args.chunk) < 0):
        parser.error("--chunk must be a nonnegative integer or last")
    published = json.loads(args.pipeline_json.read_text())
    verify_hashes(published["analysis_sources"], "published analysis source")
    inputs = {
        str(Path(__file__).resolve()): sha(__file__),
        str(Path(__file__).with_name("plot_single_layer.py").resolve()): sha(
            Path(__file__).with_name("plot_single_layer.py")
        ),
        **published["analysis_sources"],
    }
    captures, rows = [], []
    for scheme in SCHEMES:
        selected = [
            capture
            for capture in published["captures"]
            if (capture["scheme"], capture["phase"]) == (scheme, "cold")
        ]
        if len(selected) != 1:
            raise ValueError(f"expected exactly one published {scheme}/cold capture")
        capture, events = extract(selected[0], args.chunk, args.layer)
        captures.append(capture)
        rows.extend(events)
        print(
            f"{scheme}: {capture['window_ms']:.6f} ms, {len(events)} activities, "
            f"KV D2H={capture['transfers']['kv_d2h']['whole_activity_memcpy_bytes']:,} B",
            flush=True,
        )
    identities = {
        (
            c["chunk"],
            c["query_tokens"],
            c["prefix_tokens_before_chunk"],
            c["history_tokens_after_prefill"],
            c["config"]["sparse_pool_tokens"],
        )
        for c in captures
    }
    if len(identities) != 1:
        raise ValueError("the four captures use different history/chunk/pool configurations")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "intervals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    draw(captures, rows, args.output_dir)
    verify_hashes(inputs, "analysis source during rendering")
    payload = {
        "schema": "deepseek_prefill_layer_timeline_v1",
        "analysis_id": args.output_dir.name,
        "pipeline_json": str(args.pipeline_json.resolve()),
        "pipeline_sha256": sha(args.pipeline_json),
        "analysis_sources": inputs,
        "boundary": "Same history chunk: previous physical block finish graph last GPU end to selected block finish graph last GPU end.",
        "activity_boundary": "Every intersecting KERNEL, MEMCPY, and MEMSET is preserved; timestamps are clipped only for display and duration within the window.",
        "byte_boundary": "Memcpy byte totals count entire activities intersecting the window, not proportional bytes for clipped intervals. Kernel host-transfer bytes are unknown when no scoped byte counter is available.",
        "transfer_classification": "D2H under pool_write_host is persistent KV writeback. H2D under host_dma is contiguous main-KV DMA. Other D2H and recognized small cache H2D are control; other H2D is explicit rather than silently counted as KV.",
        "counter_boundary": "History segment counters are retained as context only and are not per-chunk measurements.",
        "captures": captures,
        "artifacts_sha256": {
            name: sha(args.output_dir / name)
            for name in ("intervals.csv", "timeline.svg", "timeline.pdf", "timeline.png")
        },
    }
    (args.output_dir / "timeline.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
