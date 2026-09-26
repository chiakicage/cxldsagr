"""Estimate fetch hidden by attention and preceding-layer prefetch windows.

Heads below the coverage threshold fetch their selected KV union during their
layer's attention. Other heads fetch full KV using the previous layer's
non-attention work. Attention FLOPs are unchanged for both classes; an optional
MFU multiplier scales attention time inversely, leaving other compute unchanged.
This is an ideal window-budget calculation, not an executed transfer schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from experiments.nosa_indexer_pattern_65536_1024.src.analyze import _write_csv

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments/nosa_indexer_pattern_65536_1024/output/data"
DEFAULT_ESTIMATES = (
    DATA / "estimate32_mfu_pcie50_20260926_01",
    DATA / "estimate_mfu_pcie50_20260925_01",
)


def _ratio(hidden, fetch):
    return hidden / fetch if fetch else None


def estimate_overlap(
    source_report, *, threshold_pct=30.0, bandwidth_gbps=50.0, attention_mfu_scale=1.0
):
    if not math.isfinite(threshold_pct) or not 0 <= threshold_pct <= 100:
        raise ValueError("threshold_pct must be finite and between 0 and 100")
    if not math.isfinite(bandwidth_gbps) or bandwidth_gbps <= 0:
        raise ValueError("bandwidth_gbps must be positive and finite")
    if not math.isfinite(attention_mfu_scale) or attention_mfu_scale <= 0:
        raise ValueError("attention_mfu_scale must be positive and finite")
    params = source_report["parameters"]
    layers, num_heads = params["num_layers"], params["num_kv_heads"]
    timings = source_report["layers"]
    if [row["layer"] for row in timings] != list(range(layers)):
        raise ValueError("Timing must contain every layer exactly once in order")
    source_heads = source_report["heads"]
    expected_heads = {(layer, head) for layer in range(layers) for head in range(num_heads)}
    if (
        len(source_heads) != len(expected_heads)
        or {(head["layer"], head["kv_head"]) for head in source_heads} != expected_heads
    ):
        raise ValueError("Each layer/KV head must occur exactly once")
    heads = []
    for head in source_heads:
        union, full = head["sparse_union_bytes"], head["full_kv_bytes"]
        if type(union) is not int or type(full) is not int or not 0 <= union <= full or full <= 0:
            raise ValueError("KV byte counts must be integers with 0 <= union <= full and full > 0")
        coverage_pct = 100 * union / full
        sparse = coverage_pct < threshold_pct
        heads.append(
            {
                "layer": head["layer"],
                "kv_head": head["kv_head"],
                "coverage_pct": coverage_pct,
                "head_class": "sparse" if sparse else "dense",
                "selected_union_bytes": union,
                "full_kv_bytes": full,
                "fetch_bytes": union if sparse else full,
            }
        )
    for row in timings:
        for key in ("sparse_attention_gpu_ms", "other_gpu_ms"):
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError("Compute windows must be finite and nonnegative")
    rows = []
    for layer, timing in enumerate(timings):
        layer_heads = [head for head in heads if head["layer"] == layer]
        if (
            sum(head["selected_union_bytes"] for head in layer_heads)
            != timing["sparse_union_bytes"]
            or sum(head["full_kv_bytes"] for head in layer_heads) != timing["full_kv_bytes"]
        ):
            raise ValueError("Head payloads disagree with the source layer totals")
        sparse = [head for head in layer_heads if head["head_class"] == "sparse"]
        dense = [head for head in layer_heads if head["head_class"] == "dense"]
        sparse_bytes = sum(head["fetch_bytes"] for head in sparse)
        dense_bytes = sum(head["fetch_bytes"] for head in dense)
        sparse_fetch = sparse_bytes / (bandwidth_gbps * 1e6)
        dense_fetch = dense_bytes / (bandwidth_gbps * 1e6)
        base_attention = timing["sparse_attention_gpu_ms"]
        attention_window = base_attention / attention_mfu_scale
        # Layer l-1 can prefetch layer l. Do not use post-attention work
        # retroactively to hide that same layer's already-consumed KV.
        prefetch_window = timings[layer - 1]["other_gpu_ms"] if layer else 0.0
        sparse_hidden = min(sparse_fetch, attention_window)
        dense_hidden = min(dense_fetch, prefetch_window)
        hidden = sparse_hidden + dense_hidden
        fetch = sparse_fetch + dense_fetch
        rows.append(
            {
                "layer": layer,
                "sparse_head_count": len(sparse),
                "dense_head_count": len(dense),
                "sparse_fetch_bytes": sparse_bytes,
                "dense_fetch_bytes": dense_bytes,
                "sparse_fetch_ms": sparse_fetch,
                "dense_fetch_ms": dense_fetch,
                "base_attention_ms": base_attention,
                "attention_window_ms": attention_window,
                "other_gpu_ms": timing["other_gpu_ms"],
                "compute_ms": attention_window + timing["other_gpu_ms"],
                "prefetch_source_layer": layer - 1 if layer else None,
                "prefetch_window_ms": prefetch_window,
                "sparse_hidden_ms": sparse_hidden,
                "dense_hidden_ms": dense_hidden,
                "sparse_unhidden_ms": sparse_fetch - sparse_hidden,
                "dense_unhidden_ms": dense_fetch - dense_hidden,
                "hidden_ms": hidden,
                "unhidden_ms": fetch - hidden,
                "fetch_ms": fetch,
                "compute_plus_unhidden_ms": attention_window
                + timing["other_gpu_ms"]
                + fetch
                - hidden,
                "overlap_efficiency": _ratio(hidden, fetch),
            }
        )
    additive = (
        "sparse_head_count",
        "dense_head_count",
        "sparse_fetch_bytes",
        "dense_fetch_bytes",
        "sparse_fetch_ms",
        "dense_fetch_ms",
        "base_attention_ms",
        "attention_window_ms",
        "other_gpu_ms",
        "compute_ms",
        "sparse_hidden_ms",
        "dense_hidden_ms",
        "sparse_unhidden_ms",
        "dense_unhidden_ms",
        "hidden_ms",
        "unhidden_ms",
        "fetch_ms",
        "compute_plus_unhidden_ms",
    )
    totals = {key: sum(row[key] for row in rows) for key in additive}
    totals.update(
        overlap_efficiency=_ratio(totals["hidden_ms"], totals["fetch_ms"]),
        sparse_overlap_efficiency=_ratio(totals["sparse_hidden_ms"], totals["sparse_fetch_ms"]),
        dense_overlap_efficiency=_ratio(totals["dense_hidden_ms"], totals["dense_fetch_ms"]),
        all_sparse_layers=sum(row["sparse_head_count"] == num_heads for row in rows),
        all_dense_layers=sum(row["dense_head_count"] == num_heads for row in rows),
        mixed_layers=sum(0 < row["sparse_head_count"] < num_heads for row in rows),
    )
    return {
        "parameters": {
            **params,
            "threshold_pct": threshold_pct,
            "bandwidth_gbps": bandwidth_gbps,
            "bytes_per_second": bandwidth_gbps * 1e9,
            "attention_mfu_scale": attention_mfu_scale,
        },
        "heads": heads,
        "layers": rows,
        "totals": totals,
    }


def report_markdown(reports):
    lines = [
        "# Fetch overlap window estimate",
        "",
        "Sparse: selected KV union; dense: full KV. Attention FLOPs remain unchanged.",
        "Attention time = source attention time / MFU scale; non-attention work is unchanged.",
        "Hidden = min(sparse fetch, current attention) + min(dense fetch, previous-layer non-attention).",
        "Layer 0 has no predecessor prefetch window. All heads share one link; budgets are used once.",
        "This ideal capacity estimate excludes startup, first-tile wait and indexer overhead.",
        "",
        "| Budget | MFU scale | Sparse / dense heads | Fetch ms | Hidden ms | Unhidden ms | Efficiency |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        t = report["totals"]
        efficiency = (
            f"{100 * t['overlap_efficiency']:.4f}%"
            if t["overlap_efficiency"] is not None
            else "N/A"
        )
        lines.append(
            f"| {report['parameters']['selection_slots']} | {report['parameters']['attention_mfu_scale']:g} | "
            f"{t['sparse_head_count']} / {t['dense_head_count']} | "
            f"{t['fetch_ms']:.8f} | {t['hidden_ms']:.8f} | {t['unhidden_ms']:.8f} | {efficiency} |"
        )
    lines += [
        "",
        "Compute + unhidden fetch is the same ideal algebraic budget, not measured end-to-end latency.",
        "Decoder layers only; embedding/final norm, indexer and launch/startup costs are excluded.",
        "",
        "| Budget | Attention ms | Other compute ms | Decoder compute ms | Compute + unhidden fetch ms |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        t = report["totals"]
        lines.append(
            f"| {report['parameters']['selection_slots']} | {t['attention_window_ms']:.8f} | "
            f"{t['other_gpu_ms']:.8f} | {t['compute_ms']:.8f} | {t['compute_plus_unhidden_ms']:.8f} |"
        )
    return "\n".join(lines) + "\n"


def _hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--estimate-data-dir",
        type=Path,
        action="append",
        help="Repeat to compare; defaults to both 32/64-block estimates",
    )
    parser.add_argument("--threshold-pct", type=float, default=30.0)
    parser.add_argument("--bandwidth-gbps", type=float, default=50.0)
    parser.add_argument(
        "--attention-mfu-scale",
        type=float,
        default=1.0,
        help="Attention MFU relative to the source; 0.5 doubles attention time",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    inputs = [path / "estimates.json" for path in (args.estimate_data_dir or DEFAULT_ESTIMATES)]
    reports = []
    for path in inputs:
        source = json.loads(path.read_text())
        report = estimate_overlap(
            source,
            threshold_pct=args.threshold_pct,
            bandwidth_gbps=args.bandwidth_gbps,
            attention_mfu_scale=args.attention_mfu_scale,
        )
        report["source_estimate_run_id"] = path.parent.name
        report["source_provenance"] = source["provenance"]
        reports.append(report)
    sources = [
        Path(__file__),
        Path(__file__).with_name("analyze.py"),
        Path(__file__).parent.parent / "scripts/overlap.sh",
    ]
    output = {
        "kind": "ideal_fetch_overlap_window_budget",
        "assumptions": {
            "classification": "union/full < threshold is sparse; equality is dense; labels assumed known before scheduling",
            "fetch": "sparse heads fetch their combined union, dense heads fetch full prefix+candidate KV",
            "attention": "original sparse attention FLOPs; each layer's source MFU multiplied by attention_mfu_scale for both head classes",
            "non_attention": "original non-attention times, unaffected by attention MFU scale",
            "compute_plus_unhidden": "decoder compute plus residual transfer is an ideal algebraic budget, not end-to-end latency",
            "sparse_window": "whole current-layer attention, including work of prefetched dense-class heads",
            "dense_window": "previous layer's non-attention work, with no predecessor for layer 0",
            "bandwidth": "one shared link; sum each class's head payloads before applying the window cap",
            "efficiency": "sum hidden transfer time / sum transfer time, not mean of layer/head efficiencies",
            "excluded": [
                "indexer",
                "first-tile wait",
                "transfer startup",
                "kernel resource contention",
                "host gaps",
            ],
            "schedule": "ideal windows, head labels and transferable payload assumed ready; no executed DMA schedule",
        },
        "configurations": reports,
        "provenance": {
            "run_id": args.run_id,
            "computed_at_utc": datetime.now(UTC).isoformat(),
            "input_sha256": {str(path.resolve()): _hash(path) for path in inputs},
            "source_sha256": {str(path.relative_to(ROOT)): _hash(path) for path in sources},
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "overlap.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "overlap.md").write_text(report_markdown(reports))
    for key, filename in (("layers", "layer_overlap.csv"), ("heads", "head_classes.csv")):
        _write_csv(
            args.output_dir / filename,
            [
                {
                    "block_budget": r["parameters"]["selection_slots"],
                    "estimate_run_id": r["source_estimate_run_id"],
                    **row,
                }
                for r in reports
                for row in r[key]
            ],
        )
    for source in sources:
        target = args.output_dir / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    print(report_markdown(reports))


if __name__ == "__main__":
    main()
