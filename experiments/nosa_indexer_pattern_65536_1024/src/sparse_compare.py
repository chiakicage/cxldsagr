"""Compare logical NOSA block choices on dense and sparse activation trajectories."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.nosa_indexer_pattern_65536_1024.src.analyze import _write_csv, summarize

ROOT = Path(__file__).resolve().parents[3]
ARMS = ("dense_qa64", "dense_nosa64", "sparse_nosa64")
LABELS = {
    "dense_qa64": "dense_qa64: dense / QA-only, 16 local, FP32",
    "dense_nosa64": "dense_nosa64: dense / NOSA+CIS, 17 local, SM90 dispatcher",
    "sparse_nosa64": "sparse_nosa64: sparse / NOSA+CIS, 17 local, SM90 dispatcher",
}
PAIRS = {
    "policy_and_arithmetic": (
        "dense_qa64",
        "dense_nosa64",
        "Shared dense activations; policy, CIS, local boundary and arithmetic change together",
    ),
    "sparse_propagation": (
        "dense_nosa64",
        "sparse_nosa64",
        "Same complete NOSA policy/backend; differences include sparse activation propagation",
    ),
}
EXECUTION = {
    "prefix_tokens": 65536,
    "new_tokens": 1024,
    "total_tokens": 66560,
    "prefix_chunk_size": 1024,
}


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _choice_arrays(ids, valid, block_count):
    ids, valid = np.asarray(ids), np.asarray(valid)
    if ids.ndim != 4 or any(size == 0 for size in ids.shape):
        raise ValueError("Selections need nonempty [layer, query, KV head, budget] dimensions")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Block IDs must have an integer dtype")
    if valid.shape != ids.shape or valid.dtype != np.bool_:
        raise ValueError("Validity must be boolean with the same shape as block IDs")
    if np.any(valid & ((ids < 0) | (ids >= block_count))):
        raise ValueError("Valid block IDs must be within the cache block range")
    ordered = np.sort(np.where(valid, ids, block_count), axis=-1)
    if np.any((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] < block_count)):
        raise ValueError("Duplicate valid block IDs within a selection")
    return ids, valid


def validate_choices(ids, valid, *, prefix_tokens, total_tokens, local_blocks, block_size=64):
    """Check full budgets, causality, sink and each policy's inclusive local set."""
    if block_size <= 0 or total_tokens % block_size or prefix_tokens % block_size:
        raise ValueError("Prefix and total token lengths must be block aligned")
    if not 0 <= prefix_tokens < total_tokens or local_blocks < 1:
        raise ValueError("Invalid query range or local block count")
    ids, valid = _choice_arrays(ids, valid, total_tokens // block_size)
    if ids.shape[1] != total_tokens - prefix_tokens:
        raise ValueError("Query dimension must match the captured suffix")
    if not valid.all():
        raise ValueError("Every candidate query/head must fill its complete block budget")
    query_blocks = (prefix_tokens + np.arange(ids.shape[1])) // block_size
    if np.any(ids > query_blocks[None, :, None, None]):
        raise ValueError("Selections contain future noncausal blocks")
    if not np.all(np.any(ids == 0, axis=-1)):
        raise ValueError("Every selection must contain the sink block 0")
    for offset in range(local_blocks):
        required = query_blocks - offset
        present = np.any(ids == required[None, :, None, None], axis=-1)
        if not np.all(present | (required[None, :, None] < 0)):
            raise ValueError(f"Selection is missing mandatory local block offset {offset}")


def _comparison_row(left_union, right_union, block_bytes):
    left_count, right_count = int(left_union.sum()), int(right_union.sum())
    capacity = left_union.size
    return {
        "left_union_blocks": left_count,
        "right_union_blocks": right_count,
        "left_coverage_fraction": left_count / capacity,
        "right_coverage_fraction": right_count / capacity,
        "coverage_delta_right_minus_left": (right_count - left_count) / capacity,
        "left_union_mib": left_count * block_bytes / 2**20,
        "right_union_mib": right_count * block_bytes / 2**20,
        "union_delta_mib_right_minus_left": (right_count - left_count) * block_bytes / 2**20,
    }


def compare_pair(left_union, right_union, *, block_bytes):
    """Compare physical K+V capacity, counting each layer/head separately."""
    if left_union.shape != right_union.shape or left_union.ndim != 3:
        raise ValueError("Union masks must have matching [layer, KV head, block] shapes")
    layers, heads = left_union.shape[:2]
    layer_stats, head_stats = [], []
    for layer in range(layers):
        layer_stats.append(
            {
                "layer": layer,
                **_comparison_row(left_union[layer], right_union[layer], block_bytes),
            }
        )
        for head in range(heads):
            head_stats.append(
                {
                    "layer": layer,
                    "kv_head": head,
                    **_comparison_row(
                        left_union[layer, head], right_union[layer, head], block_bytes
                    ),
                }
            )
    return {
        "summary": _comparison_row(left_union, right_union, block_bytes),
        "layer_stats": layer_stats,
        "head_stats": head_stats,
    }


def _validate_metadata(metadata):
    if metadata.get("schema_version") != 1 or not metadata.get("run_id"):
        raise ValueError("Expected a version-1 three-arm capture with a run_id")
    for name, expected in EXECUTION.items():
        if metadata["execution"].get(name) != expected:
            raise ValueError(f"This comparison requires execution.{name}={expected}")
    config = metadata["model_config"]
    if tuple(
        config.get(name)
        for name in ("num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim")
    ) != (32, 32, 2, 128):
        raise ValueError(
            "This comparison requires NOSA-8B with 32 layers, 32 Q heads, 2 KV heads and D=128"
        )
    if metadata.get("dtype") != "bfloat16":
        raise ValueError("This comparison requires bfloat16 model activations and K/V")
    if metadata.get("element_size") != 2 or metadata.get("block_size") != 64:
        raise ValueError("This comparison requires 64-token blocks and 2-byte K/V elements")
    batching = metadata.get("indexer_query_chunk_sizes")
    if batching is None:
        if metadata.get("indexer_query_chunk_size") != 64:
            raise ValueError("Legacy captures require indexer_query_chunk_size=64")
    elif batching != {"dense_qa64": 64, "dense_nosa64": None, "sparse_nosa64": None}:
        raise ValueError("Expected reference chunks of 64 and full Triton query batches")
    if set(metadata.get("arms", {})) != set(ARMS):
        raise ValueError("Capture must contain exactly the three named comparison arms")
    for arm in ARMS:
        info = metadata["arms"][arm]
        qa_only = arm == "dense_qa64"
        propagation = "sparse" if arm == "sparse_nosa64" else "dense"
        expected_sources = {
            "prefix_propagation": propagation,
            "candidate_propagation": propagation,
            "activation_source": "dense_full_attention_post_rope"
            if propagation == "dense"
            else "resident_block_sparse_attention_with_cis_post_rope",
            "selection_source": "sidecar_same_dense_activations"
            if propagation == "dense"
            else "actual_attention_input_without_reselection",
        }
        for name, expected in expected_sources.items():
            if info.get(name) != expected:
                raise ValueError(f"Unexpected {name} for {arm}: expected {expected}")
        if info.get("indexer_mode") != ("query_aware" if qa_only else "nosa") or info.get(
            "indexer_backend"
        ) != ("reference" if qa_only else "triton"):
            raise ValueError(f"Unexpected indexer mode/backend for {arm}")
        policy = {
            "block_size": 64,
            "block_budget": 64,
            "sink_blocks": 1,
            "local_blocks": 16 if qa_only else 17,
            "topk_blocks": 47 if qa_only else 15,
            "compression_kernel_size": 32,
            "compression_stride": 16,
        }
        if any(info["indexer_policy"].get(name) != value for name, value in policy.items()):
            raise ValueError(f"Unexpected NOSA selection policy for {arm}")
        if not qa_only and info.get("query_stage_blocks") != 33:
            raise ValueError("Full NOSA must protect 33 blocks in its query-aware stage")


def _verify_request(data_dir, metadata):
    path = data_dir / "request.json"
    if _sha256(path) != metadata.get("request_sha256"):
        raise ValueError("request.json SHA256 disagrees with capture metadata")
    request = json.loads(path.read_text())
    tokens = np.asarray(request["input_ids"])
    if (
        tokens.ndim != 1
        or len(tokens) != metadata["execution"]["total_tokens"]
        or not np.issubdtype(tokens.dtype, np.integer)
        or np.any(tokens < 0)
        or np.any(tokens >= metadata["model_config"]["vocab_size"])
    ):
        raise ValueError("Request token IDs do not match the captured sequence/model")
    digest = hashlib.sha256(tokens.astype("<i8").tobytes()).hexdigest()
    if digest != metadata.get("input_ids_sha256"):
        raise ValueError("Request input_ids SHA256 disagrees with capture metadata")


def baseline_compatibility(metadata, baseline):
    """Require the same exact request, model, checkpoint and execution boundary."""
    for name in ("request_sha256", "input_ids_sha256", "model_config", "checkpoint_sha256"):
        if not metadata.get(name) or metadata[name] != baseline.get(name):
            raise ValueError(f"Historical baseline differs in {name}")
    for name in EXECUTION:
        if metadata["execution"].get(name) != baseline["execution"].get(name):
            raise ValueError(f"Historical baseline differs in execution.{name}")
    if baseline.get("activation_source") != "dense_full_attention_post_rope":
        raise ValueError("Historical baseline must use dense post-RoPE activations")
    if baseline.get("indexer_compute_dtype") != "float32":
        raise ValueError("Historical baseline must use the FP32 QA-only indexer")
    if baseline.get("indexer_policy") != metadata["arms"]["dense_qa64"]["indexer_policy"]:
        raise ValueError("Historical baseline must use the same 64-block QA-only policy")
    if baseline.get("element_size") != 2 or baseline.get("block_size") != 64:
        raise ValueError("Historical baseline KV/block layout differs")


def _load_choices(path, metadata, arm):
    ids = np.load(path / "block_ids.npy", allow_pickle=False, mmap_mode="r")
    valid = np.load(path / "valid_mask.npy", allow_pickle=False, mmap_mode="r")
    if ids.shape != (32, 1024, 2, 64):
        raise ValueError(f"{arm} must have [32,1024,2,64] selections")
    validate_choices(
        ids,
        valid,
        prefix_tokens=metadata["execution"]["prefix_tokens"],
        total_tokens=metadata["execution"]["total_tokens"],
        local_blocks=16 if arm == "dense_qa64" else 17,
    )
    return ids, valid


def _validate_first_layer_choices(dense_ids, sparse_ids):
    if not np.array_equal(np.sort(dense_ids[0], axis=-1), np.sort(sparse_ids[0], axis=-1)):
        raise ValueError("Dense and sparse full NOSA must select identical first-layer block sets")


def _plot(data_dir, unions, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import MaxNLocator

    colors = ("#2471a3", "#d68910", "#239b56")
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for head in range(2):
        line, ecdf = axes[:, head]
        for arm, color in zip(ARMS, colors, strict=True):
            coverage = unions[arm][:, head].mean(-1) * 100
            line.plot(np.arange(len(coverage)), coverage, color=color, label=arm)
            ordered = np.sort(coverage)
            ecdf.step(
                np.r_[0, ordered, 100],
                np.r_[0, np.arange(1, len(ordered) + 1) / len(ordered), 1],
                where="post",
                color=color,
                label=arm,
            )
        line.set(
            title=f"KV head {head}: union K+V coverage by layer",
            xlabel="Layer",
            ylabel="Full per-head KV coverage (%)",
            ylim=(0, 105),
        )
        line.xaxis.set_major_locator(MaxNLocator(integer=True))
        ecdf.set(
            title=f"KV head {head}: coverage distribution across 32 layers",
            xlabel="Full per-head KV coverage (%)",
            ylabel="Empirical cumulative probability",
            xlim=(0, 100),
            ylim=(0, 1.02),
        )
        for axis in (line, ecdf):
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
    figure.suptitle("NOSA selections: dense QA-only, dense NOSA+CIS, sparse NOSA+CIS")
    for extension in ("png", "svg"):
        figure.savefig(data_dir / f"comparison.{extension}", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True, constrained_layout=True)
    for row, arm in enumerate(ARMS):
        for head in range(2):
            axis = axes[row, head]
            axis.imshow(
                unions[arm][:, head],
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap=ListedColormap(["#f4f4f4", colors[row]]),
                vmin=0,
                vmax=1,
            )
            axis.axvline(65536 / 64 - 0.5, color="#222222", linestyle="--", linewidth=0.8)
            axis.set(title=f"{LABELS[arm]}\nKV head {head}", ylabel="Layer")
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            if row == 2:
                axis.set_xlabel("Logical 64-token block ID")
    figure.suptitle("Union across 1024 candidate queries; dashed line starts candidate blocks")
    for extension in ("png", "svg"):
        figure.savefig(data_dir / f"union_heatmap.{extension}", dpi=170)
    plt.close(figure)


def _markdown(report):
    lines = [
        f"# NOSA selection comparison: {report['run_id']}",
        "",
        "P=65536, Q=1024; 32 layers, 2 KV heads, 64-token blocks, 64 selection slots.",
        "All union payloads count K and V separately for each layer/KV head; only queries are deduplicated.",
        "These are selection/coverage measurements, with no timing, transfer-performance or model-quality metric.",
        "",
        "| Arm | Activation trajectory and policy | Union K+V MiB | Full KV coverage % |",
        "| --- | --- | ---: | ---: |",
    ]
    for arm in ARMS:
        summary = report["arms"][arm]["summary"]
        lines.append(
            f"| {arm} | {LABELS[arm]} | {summary['union_mib']:.3f} | {summary['fraction'] * 100:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Pair | K+V capacity change, MiB | Coverage change, pp |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, pair in report["pairs"].items():
        row = pair["summary"]
        lines.append(
            f"| {name} | {row['union_delta_mib_right_minus_left']:+.3f} | "
            f"{row['coverage_delta_right_minus_left'] * 100:+.3f} |"
        )
    lines.append("")
    for name, pair in report["pairs"].items():
        lines.extend([f"{name}: {pair['interpretation']}.", ""])
    baseline = report["historical_baseline"]
    if baseline["status"] == "compared":
        row = baseline["summary"]
        lines.extend(
            [
                (
                    f"Historical dense QA baseline `{baseline['run_id']}`: "
                    f"K+V capacity change {row['union_delta_mib_right_minus_left']:+.3f} MiB."
                ),
                "",
            ]
        )
    else:
        lines.extend([f"Historical baseline: {baseline['status']}.", ""])
    if report["plots_generated"]:
        lines.extend(
            [
                "![Three-arm comparison](comparison.png)",
                "",
                "![Union heatmaps](union_heatmap.png)",
                "",
            ]
        )
    return "\n".join(lines)


def analyze(data_dir, *, baseline_data_dir=None, plots=True):
    """Analyze one complete three-arm capture; optional history is a separate check."""
    data_dir = Path(data_dir).resolve()
    metadata = json.loads((data_dir / "metadata.json").read_text())
    _validate_metadata(metadata)
    _verify_request(data_dir, metadata)
    choices, unions, arms = {}, {}, {}
    input_paths = [data_dir / name for name in ("metadata.json", "request.json")]
    for arm in ARMS:
        arm_dir = data_dir / "arms" / arm
        choices[arm] = _load_choices(arm_dir, metadata, arm)
        unions[arm], arms[arm] = summarize(
            *choices[arm],
            prefix_tokens=EXECUTION["prefix_tokens"],
            total_tokens=EXECUTION["total_tokens"],
            head_dim=metadata["model_config"]["head_dim"],
            element_size=metadata["element_size"],
            block_size=metadata["block_size"],
        )
        arms[arm].update(run_id=metadata["run_id"], arm=arm, capture=metadata["arms"][arm])
        input_paths.extend(arm_dir / name for name in ("block_ids.npy", "valid_mask.npy"))
    _validate_first_layer_choices(choices["dense_nosa64"][0], choices["sparse_nosa64"][0])
    block_bytes = arms[ARMS[0]]["parameters"]["block_bytes"]
    pairs = {}
    for name, (left, right, interpretation) in PAIRS.items():
        pair = compare_pair(unions[left], unions[right], block_bytes=block_bytes)
        pairs[name] = {
            "left_arm": left,
            "right_arm": right,
            "interpretation": interpretation,
            **pair,
        }
    baseline_report = {"status": "disabled"}
    if baseline_data_dir is not None:
        baseline_dir = Path(baseline_data_dir).resolve()
        baseline = json.loads((baseline_dir / "metadata.json").read_text())
        baseline_compatibility(metadata, baseline)
        _verify_request(baseline_dir, baseline)
        old = _load_choices(baseline_dir, baseline, "dense_qa64")
        old_union, _ = summarize(
            *old,
            prefix_tokens=65536,
            total_tokens=66560,
            head_dim=128,
            element_size=2,
            block_size=64,
        )
        comparison = compare_pair(
            old_union,
            unions["dense_qa64"],
            block_bytes=block_bytes,
        )
        baseline_report = {
            "status": "compared",
            "run_id": baseline["run_id"],
            "path": str(baseline_dir),
            "left_arm": "historical_dense_qa64",
            "right_arm": "new_dense_qa64",
            "input_sha256": {
                name: _sha256(baseline_dir / name)
                for name in ("metadata.json", "request.json", "block_ids.npy", "valid_mask.npy")
            },
            **comparison,
        }
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        versions["matplotlib"] = importlib.metadata.version("matplotlib")
    except importlib.metadata.PackageNotFoundError:
        versions["matplotlib"] = None
    report = {
        "schema_version": 1,
        "run_id": metadata["run_id"],
        "execution": metadata["execution"],
        "arms": arms,
        "pairs": pairs,
        "historical_baseline": baseline_report,
        "plots_generated": plots,
        "definitions": {
            "union_metrics": "Physical (layer, KV head, logical block) sets, deduplicated across query only",
            "coverage_delta": "Right arm minus left arm, divided by full K+V capacity; identical denominator",
            "payload": "Selected full blocks * 64 tokens * head_dim * 2 (K,V) * element_size",
            "limits": "No timing, bandwidth, physical traffic or model-quality metric is measured",
        },
        "provenance": {
            "derived_at_utc": datetime.now(UTC).isoformat(),
            "input_sha256": {
                str(path.relative_to(data_dir)): _sha256(path) for path in input_paths
            },
            "analysis_source_sha256": {
                name: _sha256(ROOT / name)
                for name in (
                    "experiments/nosa_indexer_pattern_65536_1024/src/sparse_compare.py",
                    "experiments/nosa_indexer_pattern_65536_1024/src/analyze.py",
                )
            },
            "analysis_source_snapshot_directory": "analysis_sources",
            "capture_source_sha256": metadata.get("source_sha256"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "analysis_versions": versions,
        },
    }
    if plots:
        _plot(data_dir, unions, report)
    for name in report["provenance"]["analysis_source_sha256"]:
        snapshot = data_dir / "analysis_sources" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / name).read_bytes())
    for arm in ARMS:
        arm_dir = data_dir / "arms" / arm
        np.save(arm_dir / "union_mask.npy", unions[arm], allow_pickle=False)
        _write_csv(arm_dir / "head_stats.csv", arms[arm]["head_stats"])
        _write_csv(arm_dir / "layer_stats.csv", arms[arm]["layer_stats"])
        _write_json(arm_dir / "summary.json", arms[arm])
    for level in ("head", "layer"):
        rows = [
            {"pair": name, "left_arm": pair["left_arm"], "right_arm": pair["right_arm"], **row}
            for name, pair in pairs.items()
            for row in pair[f"{level}_stats"]
        ]
        _write_csv(data_dir / f"{level}_comparison.csv", rows)
        if baseline_report["status"] == "compared":
            _write_csv(
                data_dir / f"baseline_{level}_comparison.csv", baseline_report[f"{level}_stats"]
            )
    _write_json(data_dir / "comparison.json", report)
    (data_dir / "comparison.md").write_text(_markdown(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dir", type=Path, help="Run containing metadata and arms/<name> selections"
    )
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument(
        "--baseline-data-dir", type=Path, help="Optional independent QA-only capture"
    )
    baseline.add_argument(
        "--no-baseline", action="store_true", help="Do not compare the historical run"
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Write numerical tables without figures"
    )
    args = parser.parse_args(argv)
    report = analyze(
        args.data_dir,
        baseline_data_dir=None if args.no_baseline else args.baseline_data_dir,
        plots=not args.no_plots,
    )
    print(json.dumps({name: pair["summary"] for name, pair in report["pairs"].items()}, indent=2))


if __name__ == "__main__":
    main()
