"""Give authenticated Q1 activities the same functional boundary on both sides."""

from __future__ import annotations

from copy import deepcopy

from experiments.deepseek_v32_echo_official.src import engine_timeline as engine

POLICY = {
    "schema": "q1-score-topk-functional-boundary-v1",
    "indexer": "Paged score kernels, causal score cleaning, and the complete top-k output path, including initialization, masks, transforms and sorting.",
    "projection": "Main and indexer Q/K projections, norm, RoPE, Hadamard, quantization and head weights.",
    "control": "Scheduler metadata, index cache writes, page packing/tables, staging preparation/promotion, pool management and hint updates.",
    "timing": "Clipped GPU interval unions across all streams; activity and kernel sums are reported separately. No activity, gap or timestamp is removed.",
    "source": "Original authenticated panels remain unchanged; aligned panels retain source_lane/source_category and a per-row boundary reason.",
}


def _layer(value):
    text = str(value).removeprefix("layer_")
    return int(text) if text in {"0", "1", "2"} else None


def align_panel(panel):
    """Reclassify only after source ownership and publication checks have passed."""
    aligned = deepcopy(panel)
    official = panel["source"] == "Official SGLang"
    topk_ends = {}
    for row in panel["rows"]:
        if row.get("stage") == "topk":
            layer = _layer(row["layer"])
            topk_ends[layer] = max(topk_ends.get(layer, 0), row["end_ns"])
    for row in aligned["rows"]:
        stage, name = row.get("stage", ""), row.get("name", "")
        row["source_lane"], row["source_category"] = row["lane"], row["category"]
        reason = "unchanged functional category"
        if stage in {"topk", "exact_topk"}:
            row.update(lane="Compute", category="Indexer / top-k")
            reason = "complete top-k output path, including initialization and masking"
        elif "paged_mqa_logits_metadata<" in name:
            row.update(lane="GPU control", category="GPU control")
            reason = "score scheduling metadata"
        elif "sm90_fp8_paged_mqa_logits" in name or "smxx_clean_logits<" in name:
            row.update(lane="Compute", category="Indexer / top-k")
            reason = "paged scoring or causal score cleaning"
        elif official and stage == "indexer":
            layer = _layer(row["layer"])
            if layer not in topk_ends:
                raise ValueError("Official indexer has no authenticated top-k boundary")
            if row["start_ns"] >= topk_ends[layer]:
                row.update(lane="GPU control", category="ECHO hint")
                reason = "post-top-k decode threshold update"
            elif row["lane"] == "Compute":
                row["category"] = "Projection / RoPE"
                reason = "indexer projection, normalization, rotary, quantization or head weights"
            else:
                reason = "prediction counter preparation"
        elif stage in {"indexer_prefetch", "indexer_fused", "indexer_qk", "indexer"}:
            if row["lane"] == "Compute":
                row["category"] = "Indexer / top-k"
                reason = "paged scoring or causal score cleaning"
            else:
                reason = "index input/page/staging preparation"
        row["boundary_reason"] = reason
    if (
        aligned["window"] != panel["window"]
        or aligned["layers"] != panel["layers"]
        or aligned["idle_intervals_ns"] != panel["idle_intervals_ns"]
        or len(aligned["rows"]) != len(panel["rows"])
        or any(
            {
                key: value
                for key, value in result.items()
                if key
                not in {"lane", "category", "source_lane", "source_category", "boundary_reason"}
            }
            != {key: value for key, value in source.items() if key not in {"lane", "category"}}
            for source, result in zip(panel["rows"], aligned["rows"], strict=True)
        )
    ):
        raise ValueError("Functional alignment changed source activities or their boundaries")
    aligned["category_policy"] = POLICY["schema"]
    return aligned


def category_metrics(panel):
    """Report functional interval unions, separately for each layer and all layers."""
    start, end = panel["window"]["start_ns"], panel["window"]["end_ns"]
    metrics = []
    for layer in (0, 1, 2, "all"):
        rows = [row for row in panel["rows"] if layer == "all" or _layer(row.get("layer")) == layer]
        for category in ("Projection / RoPE", "Indexer / top-k", "GPU control", "ECHO hint"):
            selected = [row for row in rows if row["category"] == category]
            intervals = [
                (max(start, row["start_ns"]), min(end, row["end_ns"]))
                for row in selected
                if row["end_ns"] > start and row["start_ns"] < end
            ]
            metrics.append(
                {
                    "source": panel["source"],
                    "method": panel["method"],
                    "layer": layer,
                    "category": category,
                    "activity_count": len(selected),
                    "union_ns": engine.duration(intervals),
                    "activity_sum_ns": sum(right - left for left, right in intervals),
                    "kernel_sum_ns": sum(
                        min(end, row["end_ns"]) - max(start, row["start_ns"])
                        for row in selected
                        if row["kind"] == "kernel"
                        and row["end_ns"] > start
                        and row["start_ns"] < end
                    ),
                    "union_us": engine.duration(intervals) / 1000,
                }
            )
    return metrics
