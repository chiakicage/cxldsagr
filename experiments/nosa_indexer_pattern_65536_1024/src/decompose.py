"""Separate sink/local and query-aware KV unions using existing pattern arrays.

No model execution or performance measurement is performed. Fetch estimates use
one shared link across KV heads and count each selected full K+V block once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.nosa_indexer_pattern_65536_1024.src.analyze import _write_csv
from experiments.nosa_indexer_pattern_65536_1024.src.selection_parts import decompose_selections

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATTERN = (
    ROOT
    / "experiments/nosa_indexer_pattern_65536_1024/output/data/query_aware_fp32_65536_1024_20260925_01"
)
LABELS = {
    "sink": "Sink",
    "local": "Local",
    "fixed": "Fixed: sink + local (D)",
    "query_aware": "Query-aware (A)",
    "overlap": "Overlap (A intersect D)",
    "query_aware_additional": "Additional query-aware (A minus D)",
    "combined": "Combined (D union A)",
}


def _flat_rows(rows):
    return [
        {
            **{key: row[key] for key in ("layer", "kv_head") if key in row},
            "component": name,
            **row[name],
        }
        for row in rows
        for name in LABELS
    ]


def report_markdown(report):
    params = report["parameters"]
    lines = [
        f"# NOSA {params['selection_slots']}-block KV union decomposition",
        "",
        "Classify each query first, then union across queries separately per layer/KV head.",
        "D = sink/local union; A = query-aware union. D and A may overlap across queries.",
        "Fetch D once, then A minus D: combined = D union A. Independent D + A double-counts overlap.",
        (
            f"Fetch assumes {params['bandwidth_gbps']:g} decimal GB/s shared across heads; "
            "full-block K+V payload only. This is an offline estimate, not a PCIe measurement."
        ),
        "",
        "## Model totals",
        "",
        "| Component | MiB | Full KV % | Fetch ms | Prefix MiB | Candidate MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in LABELS.items():
        row = report["summary"][name]
        lines.append(
            f"| {label} | {row['union_mib']:.5f} | {100 * row['fraction']:.2f} | "
            f"{row['fetch_ms']:.8f} | {row['prefix_union_bytes'] / 2**20:.5f} | "
            f"{row['candidate_union_bytes'] / 2**20:.5f} |"
        )
    lines += [
        "",
        "## Per layer (both KV heads)",
        "",
        "| Layer | Fixed MiB | QA MiB | Overlap MiB | Additional QA MiB | Fixed fetch ms | Additional QA fetch ms | Combined fetch ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["layer_stats"]:
        fixed, qa, overlap, extra, combined = (
            row[name]
            for name in ("fixed", "query_aware", "overlap", "query_aware_additional", "combined")
        )
        lines.append(
            f"| {row['layer']} | {fixed['union_mib']:.5f} | {qa['union_mib']:.5f} | "
            f"{overlap['union_mib']:.5f} | {extra['union_mib']:.5f} | "
            f"{fixed['fetch_ms']:.8f} | {extra['fetch_ms']:.8f} | {combined['fetch_ms']:.8f} |"
        )
    return "\n".join(lines) + "\n"


def _plot(path, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    rows = report["layer_stats"]
    layers = [row["layer"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for axis, key, ylabel in zip(
        axes,
        ("union_mib", "fetch_ms"),
        ("Unique K+V payload (MiB)", "Estimated fetch time (ms)"),
        strict=True,
    ):
        fixed = [row["fixed"][key] for row in rows]
        extra = [row["query_aware_additional"][key] for row in rows]
        axis.bar(layers, fixed, label="Fixed sink/local union", color="#195a91")
        axis.bar(layers, extra, bottom=fixed, label="Additional query-aware union", color="#e49436")
        axis.set(xlabel="Layer (zero based)", ylabel=ylabel, ylim=(0, None))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend(fontsize="small")
    params = report["parameters"]
    figure.suptitle(
        f"NOSA {params['selection_slots']} blocks/query/head: "
        f"fixed + additional QA, {params['bandwidth_gbps']:g} GB/s; both KV heads"
    )
    for extension in ("png", "svg"):
        figure.savefig(path / f"component_capacity_fetch.{extension}", dpi=160)
    plt.close(figure)


def _hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern-data-dir",
        type=Path,
        default=DEFAULT_PATTERN,
        help="Pattern capture directory; defaults to the 64-block baseline",
    )
    parser.add_argument("--bandwidth-gbps", type=float, default=50.0, help="Decimal GB/s")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    pattern = args.pattern_data_dir
    metadata = json.loads((pattern / "metadata.json").read_text())
    policy = metadata["indexer_policy"]
    execution = metadata["execution"]
    block_ids = np.load(pattern / "block_ids.npy", allow_pickle=False)
    valid = np.load(pattern / "valid_mask.npy", allow_pickle=False)
    config = metadata["model_config"]
    if (
        block_ids.shape
        != (
            config["num_hidden_layers"],
            execution["new_tokens"],
            config["num_key_value_heads"],
            policy["block_budget"],
        )
        or policy["block_size"] != metadata["block_size"]
    ):
        raise ValueError("Selection arrays disagree with the captured model/policy")
    unions, report = decompose_selections(
        block_ids,
        valid,
        prefix_tokens=execution["prefix_tokens"],
        total_tokens=execution["total_tokens"],
        head_dim=config["head_dim"],
        element_size=metadata["element_size"],
        block_size=policy["block_size"],
        sink_blocks=policy["sink_blocks"],
        local_blocks=policy["local_blocks"],
        bandwidth_gbps=args.bandwidth_gbps,
    )
    if not np.array_equal(
        unions["combined"], np.load(pattern / "union_mask.npy", allow_pickle=False)
    ):
        raise ValueError("Decomposition does not reproduce the original combined union")
    sources = [
        Path(__file__),
        Path(__file__).with_name("selection_parts.py"),
        Path(__file__).with_name("analyze.py"),
        Path(__file__).parent.parent / "scripts/decompose.sh",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    inputs = [
        pattern / name
        for name in (
            "metadata.json",
            "request.json",
            "block_ids.npy",
            "valid_mask.npy",
            "union_mask.npy",
        )
    ]
    report["provenance"] = {
        "run_id": args.run_id,
        "computed_at_utc": datetime.now(UTC).isoformat(),
        "pattern_run_id": pattern.name,
        "pattern_data_dir": str(pattern.resolve()),
        "input_sha256": {str(path.resolve()): _hash(path) for path in inputs},
        "source_sha256": {str(path.relative_to(ROOT)): _hash(path) for path in sources},
        "numpy_version": np.__version__,
        "performance_measured": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _plot(args.output_dir, report)
    np.savez_compressed(args.output_dir / "component_union_masks.npz", **unions)
    (args.output_dir / "decomposition.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "decomposition.md").write_text(report_markdown(report))
    _write_csv(args.output_dir / "head_components.csv", _flat_rows(report["head_stats"]))
    _write_csv(args.output_dir / "layer_components.csv", _flat_rows(report["layer_stats"]))
    _write_csv(args.output_dir / "model_components.csv", _flat_rows([report["summary"]]))
    for source in sources:
        target = args.output_dir / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
