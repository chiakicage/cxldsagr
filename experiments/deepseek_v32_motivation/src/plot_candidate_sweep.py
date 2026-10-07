"""Plot H64K offload revisit latency versus candidate length from the accepted matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

from experiments.deepseek_v32_motivation.src import plot, plot_offload_matrix

ROOT = Path(__file__).resolve().parents[3]
NAME = "revisit_offload_h64k_vs_a"
HISTORY = 65536
CANDIDATES = (128, 256, 512, 1024)
SCHEMES = ("echo", "serial_sparse", "dense_prefetch")
LABELS = {"echo": "ECHO", "serial_sparse": "serial_sparse", "dense_prefetch": "dense_prefetch"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(summary_path, output):
    summary_path = Path(summary_path).resolve(strict=True)
    output = Path(output).resolve()
    data = json.loads(summary_path.read_text())
    if data["schema"] != "deepseek-c10-fixed-capacity-matrix-v1" or data["status"] != "accepted":
        raise ValueError("an accepted fixed-capacity matrix summary is required")
    if data["candidates"] != list(CANDIDATES) or HISTORY not in data["histories"]:
        raise ValueError("the accepted matrix must cover H64K and all four candidate lengths")
    rows = {}
    for row in data["summary"]:
        if (
            row["history_tokens"] != HISTORY
            or row["scheme"] not in SCHEMES
            or row["visit_kind"] != "revisit"
        ):
            continue
        key = row["candidate_tokens"], row["scheme"]
        if key in rows or row["requests"] != 16:
            raise ValueError(f"duplicate group or unexpected sample count: {key}")
        value = row["latency_mean_ms"]
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid mean latency: {key}")
        rows[key] = row
    expected = {(candidate, scheme) for candidate in CANDIDATES for scheme in SCHEMES}
    if rows.keys() != expected:
        raise ValueError("expected all twelve H64K offload revisit means")

    output.mkdir(parents=True, exist_ok=False)
    with plot.plt.rc_context({"font.size": 13, "svg.fonttype": "none", "svg.hashsalt": NAME}):
        figure, axis = plot.plt.subplots(figsize=(5.8, 4.5))
        for scheme in SCHEMES:
            marker, linestyle = plot_offload_matrix.STYLES[scheme]
            axis.plot(
                CANDIDATES,
                [rows[candidate, scheme]["latency_mean_ms"] for candidate in CANDIDATES],
                color=plot.COLORS[scheme],
                marker=marker,
                linestyle=linestyle,
                linewidth=2.1,
                markersize=7,
                markerfacecolor="white",
                markeredgewidth=1.6,
                label=LABELS[scheme],
            )
        axis.set_xticks(CANDIDATES, [str(candidate) for candidate in CANDIDATES])
        axis.set_xlim(80, 1072)
        axis.set_ylim(0, math.ceil(max(row["latency_mean_ms"] for row in rows.values()) / 10) * 10)
        axis.set_xlabel("A", fontsize=15)
        axis.set_ylabel("Mean revisit latency (ms)", fontsize=13)
        axis.set_title("History = 64K", fontsize=15)
        axis.grid(axis="y", color="#DDE1E5", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="upper left", fontsize=12)
        figure.tight_layout()
        if len(figure.axes) != 1:
            raise ValueError("expected one coordinate plot")
        figure.savefig(
            output / f"{NAME}.pdf",
            bbox_inches="tight",
            metadata={"CreationDate": None, "ModDate": None},
        )
        plot.save_figure(figure, output, NAME)

    plotted = [
        {
            "history_tokens": HISTORY,
            "candidate_tokens": candidate,
            "scheme": scheme,
            "visit_kind": "revisit",
            "requests": rows[candidate, scheme]["requests"],
            "source_field": "latency_mean_ms",
            "source_value": rows[candidate, scheme]["latency_mean_ms"],
            "plotted_value": rows[candidate, scheme]["latency_mean_ms"],
            "run_id": rows[candidate, scheme]["run_id"],
        }
        for candidate in CANDIDATES
        for scheme in SCHEMES
    ]
    with (output / "candidate_sweep_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plotted[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(plotted)
    source_hashes = {}
    for source in (Path(__file__), Path(plot.__file__), Path(plot_offload_matrix.__file__)):
        relative = source.relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hashes[str(relative)] = digest(source)
    record = {
        "schema": "deepseek-c10-h64k-candidate-sweep-v1",
        "run_id": output.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "input_summary": {"path": str(summary_path), "sha256": digest(summary_path)},
        "source_sha256": source_hashes,
        "source_base": str(output / "source"),
        "artifacts_base": str(output),
        "history_tokens": HISTORY,
        "candidate_tokens": list(CANDIDATES),
        "schemes": list(SCHEMES),
        "benchmark_run_ids": sorted({row["run_id"] for row in plotted}),
        "plotted_values": len(plotted),
        "axes_count": 1,
        "axes": {
            "x": "candidate tokens A, linear",
            "y": "mean revisit latency in ms, linear from zero",
        },
        "boundary": (
            "Twelve values copied from the accepted matrix; no new GPU measurements. "
            "Each mean covers sixteen revisits, with one observation per request. "
            "Revisit is defined by user visit count; latency is synchronized request wall time. "
            "Different candidate lengths use the same generation rule/seed, not necessarily nested tokens."
        ),
        "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
        "artifacts_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    (output / "candidate_sweep_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = render(args.summary, args.output_dir)
    print(f"Rendered {result['plotted_values']} H64K offload values: {args.output_dir}")


if __name__ == "__main__":
    main()
