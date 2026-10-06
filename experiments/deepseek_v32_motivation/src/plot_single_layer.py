"""Draw one block's actual GPU schedule from the accepted Nsight captures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from experiments.deepseek_v32_motivation.src.analyze_pipeline import (
    analyze_capture,
    intersection_ns,
    is_host_kv_transfer,
    sha,
)

SCHEMES = ("hbm", "echo", "serial_sparse", "dense_prefetch")
NAMES = ("HBM-only", "ECHO", "Sparse fetch (serial_sparse)", "Dense prefetch")
COLORS = {
    "projection": "#0072B2",
    "index": "#0072B2",
    "topk": "#0072B2",
    "attention": "#0072B2",
    "finish": "#0072B2",
    "fused": "#995C91",
    "transfer": "#C77700",
    "aux": "#A6ACB4",
}
STAGES = ("projection", "index", "topk", "attention", "finish")


def dense_pair(capture, rows):
    layer = capture["layer"]
    selected = [r for r in rows if r["scheme"] == "dense_prefetch"]
    compute = [r for r in selected if r["group"] in STAGES and r["kind"] == "kernel"]
    transfer = [r for r in selected if r["group"] == "transfer"]
    if not compute:
        raise ValueError("dense pair requires measured computation")
    if {r["layer"] for r in compute} != {str(layer)}:
        raise ValueError("dense compute lane must contain exactly the selected layer")
    if not {r["layer"] for r in transfer} <= {str(layer), str(layer + 1)}:
        raise ValueError("dense window contains an unexpected layer's transfer")
    target = capture["next_layer_host_transfers"]
    tail = [r for r in transfer if r["layer"] == str(layer)]
    compute_streams = sorted({r["stream_id"] for r in compute})
    transfer_streams = sorted({r["stream_id"] for r in target})
    if len(compute_streams) != 1 or len(transfer_streams) > 1:
        raise ValueError("expected one compute stream and at most one prefetch stream")
    target_intervals = [(r["start_ns"], r["end_ns"]) for r in target]
    target_ns = intersection_ns(target_intervals, target_intervals)
    overlap_ns = intersection_ns([(r["start_ns"], r["end_ns"]) for r in compute], target_intervals)
    return {
        "compute_layer": layer,
        "prefetch_layer": layer + 1,
        "compute_stream": compute_streams[0],
        "prefetch_stream": transfer_streams[0] if target else None,
        "prefetch_transport": (
            "cuda_memcpy_async"
            if target and all(r["kind"] == "memcpy" for r in target)
            else "mapped_host_gather"
            if target
            else "none"
        ),
        "prefetch_memcpy_bytes": sum(r["bytes"] for r in target if r["kind"] == "memcpy"),
        "compute_start_ms": min(r["start_ms"] for r in compute),
        "compute_end_ms": max(r["end_ms"] for r in compute),
        "prefetch_start_ms": min((r["start_ms"] for r in target), default=None),
        "prefetch_end_ms": max((r["end_ms"] for r in target), default=None),
        "prefetch_full_union_ms": target_ns / 1e6,
        "overlap_ms": overlap_ns / 1e6,
        "overlap_fraction_of_full_prefetch": overlap_ns / target_ns if target_ns else None,
        "current_layer_transfer_tail_ms": intersection_ns(
            [(r["start_ns"], r["end_ns"]) for r in tail],
            [(capture["origin_ns"], capture["end_ns"])],
        )
        / 1e6,
        "boundary": f"L{layer} compute kernels intersect complete L{layer + 1} transfer intervals; denominator is the complete target transfer union. Current-layer transfer tails and D2D/control activities are excluded from this overlap.",
    }


def display_group(activity):
    scope_stage = activity["scope"]["stage"]
    stage = activity.get("graph_stage", scope_stage)
    if activity["category"] == "indexer_prefetch_fused":
        return "fused"
    if is_host_kv_transfer(activity):
        return "transfer"
    if scope_stage.startswith("compute_graph_projection_layer_"):
        return "projection"
    if scope_stage.startswith("compute_graph_finish_layer_") or stage == "v_expand":
        return "finish"
    if stage == "indexer_qk":
        return "index"
    if stage == "exact_topk":
        return "topk"
    if stage in ("mla_qk_pv", "sparse_mla"):
        return "attention"
    return "aux"


def extract(capture, layer):
    path = Path(capture["sqlite"])
    if sha(path) != capture["sqlite_sha256"]:
        raise ValueError(f"capture no longer matches the published analysis: {path}")
    summary, _, _, activities = analyze_capture(path)
    if (summary["scheme"], summary["phase"]) != (capture["scheme"], capture["phase"]):
        raise ValueError("capture identity mismatch")
    for key, value in capture["graph_input_sha256"].items():
        if sha(key) != value:
            raise ValueError(f"graph attribution input changed: {key}")
    candidate = [a for a in activities if a["scope"] and a["scope"]["segment"] == "candidate"]

    def finish_end(which):
        prefix = f"compute_graph_finish_layer_{which}_"
        return max(a["end"] for a in candidate if a["scope"]["stage"].startswith(prefix))

    origin, end = finish_end(layer - 1), finish_end(layer)
    visible = [a for a in activities if a["end"] > origin and a["start"] < end]
    if any(not a["scope"] or a["scope"]["segment"] != "candidate" for a in visible):
        raise ValueError("the requested GPU window contains unattributed or non-candidate work")
    rows = []
    for activity in sorted(visible, key=lambda a: (a["start"], a["stream_id"])):
        scope = activity["scope"]
        group = display_group(activity)
        if group in STAGES and scope["layer"] != str(layer):
            raise ValueError("the requested window includes another block's computation")
        rows.append(
            {
                "scheme": summary["scheme"],
                "phase": summary["phase"],
                "layer": scope["layer"],
                "stream_id": activity["stream_id"],
                "group": group,
                "kind": activity["kind"],
                "category": activity["category"],
                "stage": activity.get("graph_stage", scope["stage"]),
                "scope_stage": scope["stage"],
                "start_ns": activity["start"],
                "end_ns": activity["end"],
                "start_ms": (max(origin, activity["start"]) - origin) / 1e6,
                "end_ms": (min(end, activity["end"]) - origin) / 1e6,
                "bytes": activity.get("bytes") or 0,
                "name": activity["name"],
            }
        )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    transfers = [a for a in candidate if is_host_kv_transfer(a)]
    own_transfers = [a for a in transfers if a["scope"]["layer"] == str(layer)]
    compute = [
        (r["start_ns"], r["end_ns"]) for r in rows if r["group"] in STAGES and r["kind"] == "kernel"
    ]
    separate_transfer = [(r["start_ns"], r["end_ns"]) for r in rows if r["group"] == "transfer"]
    metadata = json.loads((path.parent / "metadata.json").read_text())
    case = next(c for c in metadata["captures"] if c["sqlite"] == path.name)
    result = {
        "scheme": summary["scheme"],
        "phase": summary["phase"],
        "request_id": case["request_id"],
        "layer": layer,
        "origin_ns": origin,
        "end_ns": end,
        "window_ms": (end - origin) / 1e6,
        "config": metadata["config"],
        "profile_run_id": metadata["run_id"],
        "profile_source_sha256": metadata["source_sha256"],
        "sqlite": str(path),
        "sqlite_sha256": summary["sqlite_sha256"],
        "graph_input_sha256": summary["graph_input_sha256"],
        "graph_attribution": summary["graph_attribution"],
        "visible_activity_count": len(rows),
        "separate_transfer_compute_overlap_ms": intersection_ns(compute, separate_transfer) / 1e6,
        "layer_kv_counters": case["segment_counters"]["candidate"]["layers"][layer],
        "layer_host_transfers_relative_to_origin_ms": [
            [(a["start"] - origin) / 1e6, (a["end"] - origin) / 1e6] for a in own_transfers
        ],
        "next_layer_host_transfers": [
            {
                "start_ns": a["start"],
                "end_ns": a["end"],
                "start_ms": (a["start"] - origin) / 1e6,
                "end_ms": (a["end"] - origin) / 1e6,
                "stream_id": a["stream_id"],
                "correlation_id": a["correlation"],
                "name": a["name"],
                "kind": a["kind"],
                "category": a["category"],
                "bytes": a.get("bytes") or 0,
            }
            for a in transfers
            if a["scope"]["layer"] == str(layer + 1)
        ],
        "stage_envelopes_ms": {
            group: [min(r["start_ms"] for r in rr), max(r["end_ms"] for r in rr)]
            for group, rr in grouped.items()
        },
    }
    return result, rows


def draw(captures, rows, output, layer):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
        }
    )
    fig, axes = plt.subplots(4, 1, figsize=(13, 9.6), sharex=True)
    fig.subplots_adjust(left=0.115, right=0.975, top=0.855, bottom=0.18, hspace=0.53)
    fig.suptitle(
        "DeepSeek V3.2 | One-layer computation and communication",
        x=0.04,
        ha="left",
        y=0.977,
        fontsize=18,
        fontweight="bold",
    )
    config = captures[0]["config"]
    fig.text(
        0.04,
        0.941,
        f"Measured Nsight schedule · H200 / SM90 · candidate Q={config['candidate_tokens']:,} · "
        f"history H={config['history_tokens']:,} · pool P={config['sparse_pool_tokens']:,}",
        fontsize=10.5,
    )
    fig.text(
        0.04,
        0.914,
        f"Block L{layer} (zero-based) · {captures[0]['phase']} request "
        f"{captures[0]['request_id']} · each row starts at L{layer - 1} GPU finish · same time scale",
        fontsize=10.5,
        color="#4B5563",
    )
    xmax = math.ceil(max(c["window_ms"] for c in captures) * 2) / 2 + 0.1
    for ax, capture, title in zip(axes, captures, NAMES, strict=True):
        chosen = [r for r in rows if r["scheme"] == capture["scheme"]]
        groups = defaultdict(list)
        for row in chosen:
            groups[row["group"]].append(row)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=21)
        timing_label = f"GPU window {capture['window_ms']:.3f} ms"
        pair = dense_pair(capture, rows) if capture["scheme"] == "dense_prefetch" else None
        if pair is not None:
            timing_label += f" · overlap {pair['overlap_ms']:.3f} ms"
        ax.set_title(timing_label, loc="right", fontsize=10, pad=21)
        for y in (0, 1, 2):
            ax.axhline(y, color="#E5E7EB", linewidth=0.65, zorder=0)
        labels = ["Compute", "Host → HBM", "Aux / cache"]
        if pair is not None:
            transfer_name = "DMA" if pair["prefetch_transport"] == "cuda_memcpy_async" else "gather"
            labels = [
                f"L{layer} compute\nstream {pair['compute_stream']}",
                f"KV {transfer_name}\n"
                + (
                    f"stream {pair['prefetch_stream']}"
                    if pair["prefetch_stream"]
                    else "no transfer"
                ),
                "Aux / cache",
            ]
        ax.set_yticks([2, 1, 0], labels)
        ax.tick_params(axis="y", length=0, labelsize=9)
        ax.set_ylim(-0.42, 2.72)
        ax.set_xlim(0, xmax)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
        ax.spines["bottom"].set_color("#BBC1C9")
        for group, rr in groups.items():
            if group == "fused":
                for row in rr:
                    ax.add_patch(
                        Rectangle(
                            (row["start_ms"], 0.73),
                            row["end_ms"] - row["start_ms"],
                            1.54,
                            facecolor=COLORS[group],
                            edgecolor="#603359",
                            linewidth=0.7,
                        )
                    )
                    ax.text(
                        (row["start_ms"] + row["end_ms"]) / 2,
                        1.5,
                        "2  Index + prefetch\n(one fused kernel)",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=10,
                    )
                continue
            y = 1 if group == "transfer" else 0 if group == "aux" else 2
            if group == "transfer" and pair is not None:
                for row in rr:
                    tail = row["layer"] == str(layer)
                    ax.broken_barh(
                        [(row["start_ms"], row["end_ms"] - row["start_ms"])],
                        (y - 0.27, 0.54),
                        facecolors="#E3BD80" if tail else COLORS[group],
                        edgecolors="#8A5B14" if tail else "none",
                        hatch="///" if tail else None,
                        linewidth=0.5 if tail else 0,
                    )
            else:
                ax.broken_barh(
                    [(r["start_ms"], r["end_ms"] - r["start_ms"]) for r in rr],
                    (y - 0.27, 0.54),
                    facecolors=COLORS[group],
                    linewidth=0,
                )
            if group in STAGES:
                lo, hi = min(r["start_ms"] for r in rr), max(r["end_ms"] for r in rr)
                ax.plot([lo, lo, hi, hi], [2.37, 2.44, 2.44, 2.37], color="#374151", lw=0.7)
                ax.text(
                    (lo + hi) / 2,
                    2.48,
                    str(STAGES.index(group) + 1),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
        for row in groups.get("transfer", []):
            label = (
                f"L{row['layer']} "
                + ("transfer tail" if row["layer"] == str(layer) else "prefetch")
                + f" · stream {row['stream_id']}"
                if capture["scheme"] == "dense_prefetch"
                else f"L{row['layer']} "
                + ("residual recall" if capture["scheme"] == "echo" else "miss recall")
            )
            ax.text(
                (row["start_ms"] + row["end_ms"]) / 2,
                0.63,
                label,
                ha="center",
                va="top",
                fontsize=8.5,
                color="#8A5100",
            )
        if capture["scheme"] == "hbm":
            ax.text(
                0.14,
                1,
                "History already in HBM; no KV transfer",
                va="center",
                fontsize=9,
                color="#6B7280",
            )
        if capture["scheme"] == "dense_prefetch":
            readiness = (
                f"L{layer} waits for its transfer tail"
                if pair["current_layer_transfer_tail_ms"] > 0
                else f"L{layer} history ready before t = 0"
            )
            ax.text(0.03, 2.05, readiness, fontsize=9, color="#4B5563")
        ax.axvline(capture["window_ms"], color="#526071", linewidth=0.8, linestyle="--")
    axes[-1].set_xlabel(f"Time since previous block L{layer - 1} GPU finish (ms)", labelpad=9)
    fig.text(
        0.115,
        0.114,
        "1  Norm + projections     2  Indexer logits     3  Exact top-k     "
        "4  Sparse MLA     5  V expansion + output + norm + MLP",
        fontsize=9.5,
    )
    fig.legend(
        handles=[
            Patch(color=COLORS[k], label=v)
            for k, v in (
                ("projection", "Compute path"),
                ("fused", "Compute + host reads"),
                ("transfer", "KV gather / DMA"),
                ("aux", "Cache / D2D / control"),
            )
        ],
        loc="lower center",
        bbox_to_anchor=(0.55, 0.068),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.04,
        0.048,
        "Logical lanes preserve GPU timestamps. H2D DMA uses copy engines; "
        "gather kernels and fused host reads use SMs.",
        fontsize=9,
        color="#4B5563",
    )
    fig.text(
        0.04,
        0.027,
        "White gaps are retained. Fused-kernel internals are not resolved. "
        "One instrumented sample; these windows are not benchmark layer latencies.",
        fontsize=9,
        color="#4B5563",
    )
    for extension in ("svg", "pdf", "png"):
        fig.savefig(output / f"timeline.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def draw_dense_actual(capture, rows, output):
    import matplotlib.pyplot as plt

    pair = dense_pair(capture, rows)
    if pair["prefetch_stream"] is None:
        return pair
    layer = capture["layer"]
    transfer_name = "DMA" if pair["prefetch_transport"] == "cuda_memcpy_async" else "gather"
    selected = [r for r in rows if r["scheme"] == "dense_prefetch"]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.subplots_adjust(left=0.16, right=0.975, top=0.75, bottom=0.24)
    fig.text(
        0.035,
        0.925,
        f"Dense prefetch | Measured L{layer} compute and L{layer + 1} prefetch",
        fontsize=18,
        fontweight="bold",
    )
    fraction = pair["overlap_fraction_of_full_prefetch"]
    fig.text(
        0.035,
        0.875,
        f"H200 · H={capture['config']['history_tokens']:,} · "
        f"Q={capture['config']['candidate_tokens']} · "
        f"{capture['phase']} request {capture['request_id']} · "
        f"overlap {pair['overlap_ms']:.6f} / {pair['prefetch_full_union_ms']:.6f} ms "
        f"({fraction:.1%} of complete L{layer + 1} {transfer_name})",
        fontsize=10.5,
    )
    end = max(capture["window_ms"], pair["prefetch_end_ms"])
    ax.set_xlim(0, math.ceil(end * 2) / 2 + 0.08)
    ax.set_ylim(-0.4, 3.65)
    ax.set_yticks(
        [3, 2, 1, 0],
        [
            f"L{layer} compute\nstream {pair['compute_stream']}",
            f"Complete L{layer + 1} {transfer_name}\nstream {pair['prefetch_stream']}",
            f"L{layer} transfer tail\nwithin layer window",
            "GPU control / cache",
        ],
    )
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_color("#BBC1C9")
    for y in (0, 1, 2, 3):
        ax.axhline(y, color="#E5E7EB", linewidth=0.7, zorder=0)
    for group in (*STAGES, "aux"):
        rr = [r for r in selected if r["group"] == group]
        if not rr:
            continue
        y = 0 if group == "aux" else 3
        ax.broken_barh(
            [(r["start_ms"], r["end_ms"] - r["start_ms"]) for r in rr],
            (y - 0.27, 0.54),
            facecolors=COLORS[group],
            linewidth=0,
        )
        if group in STAGES:
            lo, hi = min(r["start_ms"] for r in rr), max(r["end_ms"] for r in rr)
            ax.text(
                (lo + hi) / 2,
                3.37,
                str(STAGES.index(group) + 1),
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
    target = capture["next_layer_host_transfers"]
    ax.broken_barh(
        [(r["start_ms"], r["end_ms"] - r["start_ms"]) for r in target],
        (2 - 0.27, 0.54),
        facecolors=COLORS["transfer"],
        linewidth=0,
    )
    tail = [r for r in selected if r["group"] == "transfer" and r["layer"] == str(layer)]
    if tail:
        ax.broken_barh(
            [(r["start_ms"], r["end_ms"] - r["start_ms"]) for r in tail],
            (1 - 0.27, 0.54),
            facecolors="#E3BD80",
            edgecolors="#8A5B14",
            hatch="///",
            linewidth=0.5,
        )
    else:
        ax.text(
            0.02,
            1,
            "No current-layer transfer inside this window",
            va="center",
            color="#6B7280",
            fontsize=9,
        )
    prefetch_start, prefetch_end = pair["prefetch_start_ms"], pair["prefetch_end_ms"]
    ax.text(
        (prefetch_start + prefetch_end) / 2,
        2,
        f"L{layer + 1}: {prefetch_start:.6f}–{prefetch_end:.6f} ms",
        ha="center",
        va="center",
        fontsize=10,
        color="white",
    )
    ax.axvline(capture["window_ms"], color="#526071", linestyle="--", linewidth=0.9)
    if prefetch_end > capture["window_ms"]:
        ax.axvspan(capture["window_ms"], prefetch_end, facecolor="#CBD5E1", alpha=0.18, zorder=0)
    ax.text(
        capture["window_ms"], -0.28, f" L{layer} finish", fontsize=9, color="#526071", va="center"
    )
    ax.set_xlabel(
        f"Time since L{layer - 1} GPU finish (ms); all lanes share the capture clock", labelpad=10
    )
    fig.text(
        0.16,
        0.13,
        "1 Norm + projections    2 Indexer logits    3 Exact top-k    4 Sparse MLA    "
        "5 V expansion + output + norm + MLP",
        fontsize=9,
    )
    fig.text(
        0.035,
        0.075,
        "Overlap includes compute kernels only. D2D copies stay in GPU control; "
        "the target transfer retains its complete duration beyond the layer boundary.",
        fontsize=9,
        color="#4B5563",
    )
    fig.text(0.035, 0.033, capture["profile_run_id"], fontsize=9, color="#4B5563")
    for extension in ("svg", "pdf", "png"):
        fig.savefig(output / f"dense_prefetch_actual.{extension}", dpi=180, facecolor="white")
    plt.close(fig)
    return pair


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-json", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--phase", choices=("cold", "revisit"), default="revisit")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.layer <= 8:
        parser.error("choose a non-boundary C10 layer in [1, 8]")
    published = json.loads(args.pipeline_json.read_text())
    captures, rows = [], []
    for scheme in SCHEMES:
        selected = [
            c for c in published["captures"] if (c["scheme"], c["phase"]) == (scheme, args.phase)
        ]
        if len(selected) != 1:
            raise ValueError(f"expected exactly one {scheme}/{args.phase} capture")
        capture, events = extract(selected[0], args.layer)
        captures.append(capture)
        rows.extend(events)
        print(f"{scheme}: {capture['window_ms']:.6f} ms, {len(events)} GPU activities", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "intervals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    draw(captures, rows, args.output_dir, args.layer)
    dense = next(capture for capture in captures if capture["scheme"] == "dense_prefetch")
    pair = draw_dense_actual(dense, rows, args.output_dir)
    payload = {
        "schema": "deepseek_single_layer_timeline_v2",
        "pipeline_json": str(args.pipeline_json),
        "pipeline_sha256": sha(args.pipeline_json),
        "generator_sha256": sha(__file__),
        "analysis_sources": {path: sha(path) for path in published["analysis_sources"]},
        "boundary": "Previous block finish graph last GPU end to selected block finish graph last GPU end.",
        "fused_boundary": "One indivisible measured kernel; internal compute and transfer intervals are unknown.",
        "dense_boundary": "The main timeline preserves every activity intersecting the layer window, including a current-layer transfer tail and next-layer prefetch. The dense detail shows complete next-layer transfer duration, even beyond the selected layer window; its overlap includes only selected-layer compute kernels. Scoped main-KV H2D memcpy is separate from control H2D copies.",
        "captures": captures,
        "dense_compute_prefetch_pair": pair,
        "artifacts_sha256": {
            name: sha(args.output_dir / name)
            for name in (
                "intervals.csv",
                "timeline.svg",
                "timeline.pdf",
                "timeline.png",
                "dense_prefetch_actual.svg",
                "dense_prefetch_actual.pdf",
                "dense_prefetch_actual.png",
            )
            if name.startswith("dense_prefetch_actual") is False
            or pair["prefetch_stream"] is not None
        },
    }
    (args.output_dir / "timeline.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
