"""Estimate sparse GPU work at unchanged MFU and KV fetch at a fixed bandwidth.

This is a calculation from existing dense timings and block selections, not a
new sparse kernel or PCIe measurement. GPU module activity durations follow the
original MFU denominator; host gaps and indexer overhead are not estimated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.nosa_gr_65536_1024.src.mfu import matrix_flops
from experiments.nosa_indexer_pattern_65536_1024.src.analyze import summarize

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DENSE = (
    ROOT
    / "experiments/nosa_gr_65536_1024/output/data/flashinfer_merged_gemm_65536_1024_20260925_02"
)
DEFAULT_PATTERN = (
    ROOT
    / "experiments/nosa_indexer_pattern_65536_1024/output/data/query_aware_fp32_65536_1024_20260925_01"
)


def estimate(
    block_ids,
    valid_mask,
    *,
    config,
    execution,
    dense_timing,
    peak_tflops=989.0,
    bandwidth_gbps=50.0,
    element_size=2,
    block_size=64,
):
    """Return estimates with separate query-pair FLOPs and union-byte accounting."""
    if not math.isfinite(peak_tflops) or peak_tflops <= 0:
        raise ValueError("peak_tflops must be positive and finite")
    if not math.isfinite(bandwidth_gbps) or bandwidth_gbps <= 0:
        raise ValueError("bandwidth_gbps must be positive and finite")
    prefix, total = execution["prefix_tokens"], execution["total_tokens"]
    layers, query_heads, kv_heads = (
        config["num_hidden_layers"],
        config["num_attention_heads"],
        config["num_key_value_heads"],
    )
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise ValueError("Query heads must divide into KV head groups")
    _, payload = summarize(
        block_ids,
        valid_mask,
        prefix_tokens=prefix,
        total_tokens=total,
        head_dim=config["head_dim"],
        element_size=element_size,
        block_size=block_size,
    )
    if block_ids.shape[0] != layers or block_ids.shape[2] != kv_heads:
        raise ValueError("Selections do not match the model's layers and KV heads")
    timings = dense_timing["layers"]
    if len(timings) != layers or [row["layer"] for row in timings] != list(range(layers)):
        raise ValueError("Dense timing must contain each model layer exactly once in order")

    dense_flops = matrix_flops(config, prefix, total - prefix)
    dense_attention_flops = dense_flops["attention_core"] // layers
    projection_flops = (
        sum(value for key, value in dense_flops.items() if key != "attention_core") // layers
    )
    # A block is fetched in full, but only its causal token pairs contribute to
    # effective attention FLOPs. QK and PV cost 4*head_dim per Q-head/key pair.
    positions = np.arange(prefix, total, dtype=np.int64)
    visible_tokens = np.clip(
        positions[None, :, None, None] - block_ids.astype(np.int64) * block_size + 1,
        0,
        block_size,
    )
    pairs_per_kv_head = np.where(valid_mask, visible_tokens, 0).sum(axis=(1, 3))
    sparse_flops = pairs_per_kv_head.sum(axis=-1) * (
        4 * config["head_dim"] * (query_heads // kv_heads)
    )
    bytes_per_ms = bandwidth_gbps * 1e6  # decimal GB/s: 50e9 B/s = 50e6 B/ms
    rows = []
    for layer, (timing, capacity) in enumerate(zip(timings, payload["layer_stats"], strict=True)):
        modules = timing["modules"]
        if any(not math.isfinite(value) or value < 0 for value in modules.values()):
            raise ValueError("GPU module times must be finite and nonnegative")
        dense_attention_ms = modules["attention_core"]
        if dense_attention_ms <= 0:
            raise ValueError("Each layer requires a positive dense attention duration")
        dense_layer_ms = sum(modules.values())
        if not math.isclose(dense_layer_ms, timing["gpu_ms"], abs_tol=1e-9):
            raise ValueError("Layer duration must equal its module activity duration sum")
        ratio = int(sparse_flops[layer]) / dense_attention_flops
        other_ms = dense_layer_ms - dense_attention_ms
        sparse_attention_ms = dense_attention_ms * ratio
        rows.append(
            {
                "layer": layer,
                "dense_attention_flops": dense_attention_flops,
                "sparse_attention_flops": int(sparse_flops[layer]),
                "attention_flops_ratio": ratio,
                "projection_flops": projection_flops,
                "dense_attention_mfu_pct": dense_attention_flops
                / (dense_attention_ms * 1e9 * peak_tflops)
                * 100,
                "dense_attention_gpu_ms": dense_attention_ms,
                "dense_layer_gpu_ms": dense_layer_ms,
                "other_gpu_ms": other_ms,
                "sparse_attention_gpu_ms": sparse_attention_ms,
                "sparse_layer_gpu_ms": other_ms + sparse_attention_ms,
                "sparse_union_bytes": capacity["union_bytes"],
                "full_kv_bytes": capacity["kv_cache_bytes"],
                "sparse_fetch_ms": capacity["union_bytes"] / bytes_per_ms,
                "full_fetch_ms": capacity["kv_cache_bytes"] / bytes_per_ms,
                "prefix_sparse_fetch_ms": capacity["prefix_union_bytes"] / bytes_per_ms,
                "prefix_full_fetch_ms": capacity["prefix_kv_cache_bytes"] / bytes_per_ms,
            }
        )
    heads = [
        {
            "layer": row["layer"],
            "kv_head": row["kv_head"],
            "sparse_union_bytes": row["union_bytes"],
            "full_kv_bytes": row["kv_cache_bytes"],
            "sparse_fetch_ms": row["union_bytes"] / bytes_per_ms,
            "full_fetch_ms": row["kv_cache_bytes"] / bytes_per_ms,
        }
        for row in payload["head_stats"]
    ]
    additive = (
        "dense_attention_flops",
        "sparse_attention_flops",
        "projection_flops",
        "dense_attention_gpu_ms",
        "dense_layer_gpu_ms",
        "other_gpu_ms",
        "sparse_attention_gpu_ms",
        "sparse_layer_gpu_ms",
        "sparse_union_bytes",
        "full_kv_bytes",
        "sparse_fetch_ms",
        "full_fetch_ms",
        "prefix_sparse_fetch_ms",
        "prefix_full_fetch_ms",
    )
    totals = {key: sum(row[key] for row in rows) for key in additive}
    shared_times = dense_timing["shared_modules"].values()
    if any(not math.isfinite(value) or value < 0 for value in shared_times):
        raise ValueError("Shared GPU module times must be finite and nonnegative")
    shared_ms = sum(shared_times)
    totals.update(
        {
            "shared_gpu_ms": shared_ms,
            "sparse_model_gpu_ms_including_shared": totals["sparse_layer_gpu_ms"] + shared_ms,
            "mean_sparse_attention_gpu_ms": totals["sparse_attention_gpu_ms"] / layers,
            "mean_sparse_layer_gpu_ms": totals["sparse_layer_gpu_ms"] / layers,
            "attention_flops_ratio": totals["sparse_attention_flops"]
            / totals["dense_attention_flops"],
            "dense_attention_mfu_pct": totals["dense_attention_flops"]
            / (totals["dense_attention_gpu_ms"] * 1e9 * peak_tflops)
            * 100,
            "sparse_effective_matrix_flops": totals["projection_flops"]
            + totals["sparse_attention_flops"],
        }
    )
    return {
        "kind": "estimate_from_existing_measurements",
        "parameters": {
            "peak_tflops": peak_tflops,
            "bandwidth_gbps": bandwidth_gbps,
            "bytes_per_second": bandwidth_gbps * 1e9,
            **payload["parameters"],
        },
        "assumptions": {
            "mfu": "preserve each layer's measured dense attention MFU",
            "other_modules": "preserve each layer's measured non-attention GPU activity durations",
            "flops": "causal effective QK + PV; each KV head serves its GQA query heads",
            "layer_time": "sum of GPU module activity durations, as in the source MFU denominator",
            "fetch": "full selected K+V blocks including candidate KV, both heads share one bandwidth",
            "excluded": [
                "indexer",
                "host gaps and launch overhead",
                "fetch startup overhead",
                "transfer/compute overlap",
            ],
        },
        "layers": rows,
        "heads": heads,
        "totals": totals,
        "dense_shared_modules": dense_timing["shared_modules"],
        "sparse_query_key_pairs_per_kv_head": pairs_per_kv_head.tolist(),
    }


def _csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figure, axis = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    layers = report["layers"]
    for key, label in (
        ("sparse_attention_gpu_ms", "Sparse attention, unchanged MFU"),
        ("sparse_layer_gpu_ms", "Sparse layer GPU work"),
        ("sparse_fetch_ms", "Sparse KV union fetch"),
        ("full_fetch_ms", "Full KV fetch"),
    ):
        axis.plot(
            [row["layer"] for row in layers], [row[key] for row in layers], marker=".", label=label
        )
    axis.set(
        xlabel="Layer (zero based)",
        ylabel="Estimated time (ms)",
        ylim=(0, None),
        title=(
            f"NOSA 64K + 1K, {report['parameters']['selection_slots']} blocks/query/head: "
            f"unchanged MFU, {report['parameters']['bandwidth_gbps']:g} GB/s fetch"
        ),
    )
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=0.25)
    axis.legend()
    for extension in ("png", "svg"):
        figure.savefig(path / f"time_estimates.{extension}", dpi=160)
    plt.close(figure)


def report_markdown(report):
    totals = report["totals"]
    lines = [
        f"# 固定 MFU 与 {report['parameters']['bandwidth_gbps']:g} GB/s KV fetch 估算",
        "",
        "由已有 dense GPU 测量和 sparse pattern 计算；非 sparse kernel / PCIe 实测。",
        f"每 query/head 选择槽位：{report['parameters']['selection_slots']} blocks。",
        "计算时间为模块 GPU 活动时长之和，不含 indexer、host gap、fetch 或 overlap。",
        "",
        (
            f"{len(report['layers'])} 层 sparse attention 合计 {totals['sparse_attention_gpu_ms']:.6f} ms；"
            f"decoder 层合计 {totals['sparse_layer_gpu_ms']:.6f} ms，"
            f"embedding/final norm 另 {totals['shared_gpu_ms']:.6f} ms。"
        ),
        "",
        (
            f"Sparse union fetch 合计 {totals['sparse_fetch_ms']:.6f} ms；"
            f"full KV fetch 合计 {totals['full_fetch_ms']:.6f} ms。"
        ),
        "",
        "| 层 | Dense attention MFU | Sparse attention ms | 其他模块 ms | Sparse 层 GPU ms | Sparse fetch ms | Full fetch ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["layers"]:
        lines.append(
            f"| {row['layer']} | {row['dense_attention_mfu_pct']:.2f}% | {row['sparse_attention_gpu_ms']:.6f} | {row['other_gpu_ms']:.6f} | {row['sparse_layer_gpu_ms']:.6f} | {row['sparse_fetch_ms']:.6f} | {row['full_fetch_ms']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def _hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    from experiments.nosa_indexer_pattern_65536_1024.src.dense_timing import load_dense_layer_timing

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-data-dir", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--pattern-data-dir", type=Path, default=DEFAULT_PATTERN)
    parser.add_argument("--bandwidth-gbps", type=float, default=50.0, help="Decimal GB/s")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    dense, pattern = args.dense_data_dir, args.pattern_data_dir
    mfu = json.loads((dense / "mfu.json").read_text())
    metadata = json.loads((pattern / "metadata.json").read_text())
    if mfu["model_config"] != metadata["model_config"]:
        raise ValueError("Dense and pattern model configurations differ")
    for key in ("prefix_tokens", "new_tokens", "total_tokens"):
        if mfu["execution"][key] != metadata["execution"][key]:
            raise ValueError(f"Dense and pattern execution lengths differ: {key}")
    if (dense / "request.json").read_bytes() != (pattern / "request.json").read_bytes():
        raise ValueError("Dense and pattern captures must use the exact same request")
    timing = load_dense_layer_timing(
        dense / "nsys.sqlite", mfu["model_config"]["num_hidden_layers"]
    )
    original = next(
        row
        for row in json.loads((dense / "analysis.json").read_text())
        if row["range"] == "GR/detailed/extend/0"
    )
    if timing["modules"].keys() != original["modules"].keys():
        raise ValueError("Reconstructed module names disagree with dense analysis")
    for module, row in original["modules"].items():
        if not math.isclose(
            timing["modules"].get(module, -1), row["gpu_ms"], rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(f"Reconstructed module timing disagrees with dense analysis: {module}")
    expected_flops = matrix_flops(
        mfu["model_config"], mfu["execution"]["prefix_tokens"], mfu["execution"]["new_tokens"]
    )
    for module, flops in expected_flops.items():
        if flops != mfu["extend"][module]["flops"] or not math.isclose(
            timing["modules"][module], mfu["extend"][module]["gpu_ms"], abs_tol=1e-9
        ):
            raise ValueError(f"Dense timing/FLOPs do not match the MFU baseline: {module}")
    report = estimate(
        np.load(pattern / "block_ids.npy", allow_pickle=False),
        np.load(pattern / "valid_mask.npy", allow_pickle=False),
        config=metadata["model_config"],
        execution=metadata["execution"],
        dense_timing=timing,
        peak_tflops=mfu["peak_tflops"],
        bandwidth_gbps=args.bandwidth_gbps,
        element_size=metadata["element_size"],
        block_size=metadata["block_size"],
    )
    sources = [
        Path(__file__),
        Path(__file__).with_name("dense_timing.py"),
        Path(__file__).with_name("analyze.py"),
        Path(__file__).parent.parent / "scripts/estimate.sh",
        ROOT / "experiments/nosa_gr_65536_1024/src/mfu.py",
    ]
    inputs = [
        dense / name
        for name in ("request.json", "metadata.json", "analysis.json", "mfu.json", "nsys.sqlite")
    ]
    inputs += [
        pattern / name
        for name in ("request.json", "metadata.json", "block_ids.npy", "valid_mask.npy")
    ]
    report["provenance"] = {
        "run_id": args.run_id,
        "computed_at_utc": datetime.now(UTC).isoformat(),
        "dense_run_id": dense.name,
        "pattern_run_id": pattern.name,
        "input_sha256": {str(path.resolve()): _hash(path) for path in inputs},
        "source_sha256": {str(path.relative_to(ROOT)): _hash(path) for path in sources},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _plot(args.output_dir, report)
    (args.output_dir / "estimates.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "dense_layer_timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    (args.output_dir / "estimate.md").write_text(report_markdown(report))
    _csv(args.output_dir / "layer_estimates.csv", report["layers"])
    _csv(args.output_dir / "head_fetch_estimates.csv", report["heads"])
    for source in sources:
        target = args.output_dir / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    print(json.dumps(report["totals"], indent=2))


if __name__ == "__main__":
    main()
