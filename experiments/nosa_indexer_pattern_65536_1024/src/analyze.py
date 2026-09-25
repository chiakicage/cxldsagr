"""Count unique NOSA KV block payloads across the candidate query tokens.

Each (layer, KV head) has a separate union. The result describes the bytes
needed if every selected block is fetched once, not measured device traffic.
Only block-aligned prefix and total lengths are supported by this experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
from numbers import Integral
from pathlib import Path

import numpy as np


def _integer(name, value, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _stats(prefix_blocks, candidate_blocks, *, block_bytes, prefix_capacity, total_capacity):
    prefix_bytes = int(prefix_blocks) * block_bytes
    candidate_bytes = int(candidate_blocks) * block_bytes
    union_bytes = prefix_bytes + candidate_bytes
    candidate_capacity = total_capacity - prefix_capacity
    return {
        "union_blocks": int(prefix_blocks + candidate_blocks),
        "union_bytes": union_bytes,
        "union_mib": union_bytes / 2**20,
        "kv_cache_bytes": total_capacity,
        "fraction": union_bytes / total_capacity,
        "prefix_union_blocks": int(prefix_blocks),
        "prefix_union_bytes": prefix_bytes,
        "prefix_kv_cache_bytes": prefix_capacity,
        "prefix_fraction": prefix_bytes / prefix_capacity if prefix_capacity else None,
        "candidate_union_blocks": int(candidate_blocks),
        "candidate_union_bytes": candidate_bytes,
        "candidate_kv_cache_bytes": candidate_capacity,
        "candidate_fraction": candidate_bytes / candidate_capacity,
    }


def summarize(
    block_ids,
    valid_mask,
    *,
    prefix_tokens,
    total_tokens,
    head_dim,
    element_size,
    block_size=64,
):
    """Return ``(union_mask[layer, kv_head, block], report)`` using NumPy only.

    Inputs have shape [layer, query, kv_head, selected_block]. Query positions
    are the contiguous suffix [prefix_tokens, total_tokens). Invalid entries
    are padding and ignored. Valid block IDs must be causal and distinct within
    each query/head selection; repeated IDs across different queries are normal.

    A block costs ``block_size * head_dim * 2 * element_size`` bytes, counting
    both K and V. Every head/layer is counted independently, including when its
    block IDs match another head/layer. No tensors or cache contents are needed.
    """
    prefix_tokens = _integer("prefix_tokens", prefix_tokens, minimum=0)
    total_tokens = _integer("total_tokens", total_tokens)
    head_dim = _integer("head_dim", head_dim)
    element_size = _integer("element_size", element_size)
    block_size = _integer("block_size", block_size)
    if total_tokens <= prefix_tokens:
        raise ValueError("total_tokens must exceed prefix_tokens")
    if prefix_tokens % block_size or total_tokens % block_size:
        raise ValueError("prefix_tokens and total_tokens must be block aligned")
    block_ids = np.asarray(block_ids)
    valid_mask = np.asarray(valid_mask)
    if block_ids.ndim != 4 or any(size == 0 for size in block_ids.shape):
        raise ValueError("block_ids must have nonempty shape [layer, query, kv_head, selection]")
    if not np.issubdtype(block_ids.dtype, np.integer):
        raise ValueError("block_ids must have an integer dtype")
    if valid_mask.shape != block_ids.shape or valid_mask.dtype != np.bool_:
        raise ValueError("valid_mask must be boolean with the same shape as block_ids")
    layers, queries, heads, selections = block_ids.shape
    if queries != total_tokens - prefix_tokens:
        raise ValueError("query dimension must equal total_tokens - prefix_tokens")
    num_blocks = total_tokens // block_size
    if np.any(valid_mask & ((block_ids < 0) | (block_ids >= num_blocks))):
        raise ValueError("valid block IDs must be within the full KV cache range")
    query_blocks = (prefix_tokens + np.arange(queries)) // block_size
    if np.any(valid_mask & (block_ids > query_blocks[None, :, None, None])):
        raise ValueError("valid block IDs must be causal for each query position")
    sorted_ids = np.sort(
        np.where(valid_mask, block_ids.astype(np.int64, copy=False), num_blocks), axis=-1
    )
    if np.any((sorted_ids[..., 1:] == sorted_ids[..., :-1]) & (sorted_ids[..., 1:] < num_blocks)):
        raise ValueError("duplicate valid block IDs within a query/head selection")

    union_mask = np.zeros((layers, heads, num_blocks), dtype=np.bool_)
    for layer in range(layers):
        for head in range(heads):
            selected = block_ids[layer, :, head, :][valid_mask[layer, :, head, :]]
            union_mask[layer, head, np.unique(selected)] = True
    prefix_block_count = prefix_tokens // block_size
    prefix_counts = union_mask[..., :prefix_block_count].sum(axis=-1)
    candidate_counts = union_mask[..., prefix_block_count:].sum(axis=-1)
    block_bytes = block_size * head_dim * 2 * element_size
    head_capacity = num_blocks * block_bytes
    head_prefix_capacity = prefix_block_count * block_bytes

    def stats(prefix_count, candidate_count, instances):
        return _stats(
            prefix_count,
            candidate_count,
            block_bytes=block_bytes,
            prefix_capacity=head_prefix_capacity * instances,
            total_capacity=head_capacity * instances,
        )

    head_stats = [
        {
            "layer": layer,
            "kv_head": head,
            **stats(prefix_counts[layer, head], candidate_counts[layer, head], 1),
        }
        for layer in range(layers)
        for head in range(heads)
    ]
    layer_stats = [
        {
            "layer": layer,
            **stats(prefix_counts[layer].sum(), candidate_counts[layer].sum(), heads),
        }
        for layer in range(layers)
    ]
    report = {
        "schema_version": 1,
        "metric": "unique full-block K+V payload across candidate queries",
        "deduplication_axes": "query only; KV heads and layers remain independent",
        "byte_accounting": "block_size * head_dim * 2 (K and V) * element_size",
        "parameters": {
            "num_layers": layers,
            "num_queries": queries,
            "num_kv_heads": heads,
            "selection_slots": selections,
            "prefix_tokens": prefix_tokens,
            "candidate_tokens": total_tokens - prefix_tokens,
            "total_tokens": total_tokens,
            "head_dim": head_dim,
            "element_size": element_size,
            "block_size": block_size,
            "blocks_per_head": num_blocks,
            "block_bytes": block_bytes,
        },
        "head_stats": head_stats,
        "layer_stats": layer_stats,
        "summary": stats(prefix_counts.sum(), candidate_counts.sum(), layers * heads),
    }
    return union_mask, report


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path, report):
    params = report["parameters"]
    total = report["summary"]
    lines = [
        f"# NOSA query-aware block union ({params['selection_slots']} slots per query/head)",
        "",
        (
            f"Candidate queries: {params['num_queries']}; prefix: {params['prefix_tokens']} tokens; "
            f"full KV cache: {params['total_tokens']} tokens."
        ),
        "",
        (
            f"Each {params['block_size']}-token block occupies {params['block_bytes']:,} bytes "
            "per KV head, including K and V. Unions span candidate query positions only; "
            "different heads and layers retain separate physical KV records."
        ),
        "",
        (
            f"All layers: **{total['union_mib']:.3f} MiB / "
            f"{total['kv_cache_bytes'] / 2**20:.3f} MiB ({100 * total['fraction']:.2f}%)**."
        ),
        "",
        (
            "This is unique block payload assuming each selected block is fetched once. "
            "It excludes indexer reads, cache writes, repeated physical transfers, and latency. "
            "The underlying activations come from the unchanged dense forward."
        ),
        "",
        "## Per layer",
        "",
        "| Layer | Union blocks across heads | Union MiB | Full KV % | Prefix MiB | Candidate MiB |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["layer_stats"]:
        lines.append(
            f"| {row['layer']} | {row['union_blocks']} | {row['union_mib']:.3f} | "
            f"{100 * row['fraction']:.2f} | {row['prefix_union_bytes'] / 2**20:.3f} | "
            f"{row['candidate_union_bytes'] / 2**20:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Per KV head",
            "",
            "| Layer | KV head | Union blocks | Union MiB | Full KV % | Prefix blocks | Candidate blocks |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["head_stats"]:
        lines.append(
            f"| {row['layer']} | {row['kv_head']} | {row['union_blocks']} | "
            f"{row['union_mib']:.3f} | {100 * row['fraction']:.2f} | "
            f"{row['prefix_union_blocks']} | {row['candidate_union_blocks']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(data_dir, union_mask, report):
    # Plotting is an optional analysis dependency; importing summarize needs NumPy only.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import MaxNLocator

    params = report["parameters"]
    layers = np.arange(params["num_layers"])
    heads = params["num_kv_heads"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for head in range(heads):
        rows = [row for row in report["head_stats"] if row["kv_head"] == head]
        axes[0].plot(
            layers, [row["union_mib"] for row in rows], label=f"KV head {head}", marker="."
        )
        axes[1].plot(
            layers, [100 * row["fraction"] for row in rows], label=f"KV head {head}", marker="."
        )
    total_rows = report["layer_stats"]
    axes[0].plot(
        layers,
        [row["union_mib"] for row in total_rows],
        label="Layer total",
        color="black",
        marker=".",
    )
    axes[0].axhline(
        total_rows[0]["kv_cache_bytes"] / 2**20,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Full layer KV",
    )
    axes[1].plot(
        layers,
        [100 * row["fraction"] for row in total_rows],
        label="Layer total",
        color="black",
        linestyle="--",
        marker=".",
    )
    for axis in axes:
        axis.set_xlabel("Layer (zero based)")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlim(-0.5, params["num_layers"] - 0.5)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")
    axes[0].set_ylabel("Unique K+V payload (MiB)")
    axes[1].set_ylabel("Fraction of full KV cache (%)")
    axes[1].set_ylim(0, 105)
    figure.suptitle(
        f"NOSA candidate-query block unions ({params['selection_slots']} blocks/query/head)"
    )
    for extension in ("png", "svg"):
        figure.savefig(data_dir / f"capacity_by_layer.{extension}", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(
        heads,
        1,
        figsize=(15, max(3, 2.8 * heads)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for head, axis in enumerate(axes[:, 0]):
        axis.imshow(
            union_mask[:, head, :],
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap=ListedColormap(["#f4f4f4", "#195a91"]),
            vmin=0,
            vmax=1,
        )
        axis.axvline(
            params["prefix_tokens"] / params["block_size"] - 0.5,
            color="#d15b26",
            linestyle="--",
            linewidth=1,
        )
        axis.set_ylabel("Layer (zero based)")
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_title(f"KV head {head}: blue = selected by at least one candidate query")
    axes[-1, 0].set_xlabel(f"Logical block ID ({params['block_size']} tokens per block)")
    figure.suptitle(
        f"{params['selection_slots']} blocks/query/head; orange line starts candidate blocks"
    )
    for extension in ("png", "svg"):
        figure.savefig(data_dir / f"union_heatmap.{extension}", dpi=160)
    plt.close(figure)


def analyze(data_dir):
    """Read capture artifacts and write tables, the union mask, and static plots."""
    data_dir = Path(data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text())
    execution = metadata["execution"]
    union_mask, report = summarize(
        np.load(data_dir / "block_ids.npy", allow_pickle=False),
        np.load(data_dir / "valid_mask.npy", allow_pickle=False),
        prefix_tokens=execution["prefix_tokens"],
        total_tokens=execution["total_tokens"],
        head_dim=metadata["model_config"]["head_dim"],
        element_size=metadata["element_size"],
        block_size=metadata.get("block_size", 64),
    )
    # Fail on a missing plotting dependency before publishing partial analysis files.
    _plot(data_dir, union_mask, report)
    np.save(data_dir / "union_mask.npy", union_mask, allow_pickle=False)
    _write_csv(data_dir / "head_stats.csv", report["head_stats"])
    _write_csv(data_dir / "layer_stats.csv", report["layer_stats"])
    (data_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_report(data_dir / "report.md", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Run data directory containing capture NPYs")
    args = parser.parse_args()
    report = analyze(args.data_dir)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
