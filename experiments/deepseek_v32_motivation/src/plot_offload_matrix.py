"""Plot single-panel revisit comparisons versus history for each candidate length."""

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

from experiments.deepseek_v32_motivation.src import plot

SCHEMES = ("hbm", "echo", "serial_sparse", "dense_prefetch")
STYLES = {
    "hbm": ("D", "-."),
    "echo": ("o", "-"),
    "serial_sparse": ("s", "--"),
    "dense_prefetch": ("^", ":"),
}
VIEWS = (
    ("hbm_offload", SCHEMES, "HBM-only and offload", "log"),
    ("offload", SCHEMES[1:], "Offload methods", "linear"),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(summary_path, output):
    summary_path = Path(summary_path).resolve()
    output = Path(output).resolve()
    data = json.loads(summary_path.read_text())
    if data["schema"] != "deepseek-c10-fixed-capacity-matrix-v1" or data["status"] != "accepted":
        raise ValueError("an accepted fixed-capacity matrix summary is required")
    histories, candidates = data["histories"], data["candidates"]
    if histories != [4096, 16384, 65536] or candidates != [128, 256, 512, 1024]:
        raise ValueError("expected the complete 3-by-4 H/A matrix")
    rows = {}
    for row in data["summary"]:
        if row["scheme"] not in SCHEMES or row["visit_kind"] != "revisit":
            continue
        key = (row["history_tokens"], row["candidate_tokens"], row["scheme"])
        if key in rows or row["requests"] != 16:
            raise ValueError(f"duplicate series or unexpected sample count: {key}")
        rows[key] = row
    if len(rows) != 48:
        raise ValueError("expected 48 revisit summary rows")

    output.mkdir(parents=True, exist_ok=False)
    plotted = []
    figures = []
    with plot.plt.rc_context({"font.size": 12, "svg.fonttype": "none", "svg.hashsalt": "offload"}):
        for candidate in candidates:
            for view, schemes, title, yscale in VIEWS:
                name = f"revisit_{view}_a{candidate}"
                figure, axis = plot.plt.subplots(figsize=(5.8, 4.6))
                figure_values = []
                for scheme in schemes:
                    marker, linestyle = STYLES[scheme]
                    values = []
                    for history in histories:
                        row = rows[history, candidate, scheme]
                        value = row["latency_mean_ms"]
                        if not math.isfinite(value) or value <= 0:
                            raise ValueError(f"invalid latency for {history}/{candidate}/{scheme}")
                        values.append(value)
                        figure_values.append(value)
                        plotted.append(
                            {
                                "figure": name,
                                "view": view,
                                "history_tokens": history,
                                "candidate_tokens": candidate,
                                "scheme": scheme,
                                "visit_kind": "revisit",
                                "source_field": "latency_mean_ms",
                                "source_value": value,
                                "plotted_value": value,
                            }
                        )
                    axis.plot(
                        histories,
                        values,
                        marker=marker,
                        linestyle=linestyle,
                        color=plot.COLORS[scheme],
                        label="Sparse fetch" if scheme == "serial_sparse" else plot.LABELS[scheme],
                        linewidth=1.8,
                        markersize=6,
                        markerfacecolor="white",
                        markeredgewidth=1.5,
                    )
                axis.set_xscale("log", base=2)
                axis.set_xticks(histories, [f"{value // 1024}K" for value in histories])
                axis.set_yscale(yscale)
                if yscale == "log":
                    axis.set_ylim(min(figure_values) / 1.3, max(figure_values) * 1.3)
                else:
                    axis.set_ylim(0, max(figure_values) * 1.12)
                axis.set_title(f"{title} | A = {candidate}", fontsize=12)
                axis.set_xlabel("History tokens H (log2)")
                axis.set_ylabel(
                    "Mean revisit latency (ms)" + ("\nlog scale" if yscale == "log" else "")
                )
                axis.grid(axis="y", alpha=0.25)
                axis.spines[["top", "right"]].set_visible(False)
                handles, labels = axis.get_legend_handles_labels()
                figure.legend(
                    handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10.5
                )
                figure.text(
                    0.5,
                    0.145,
                    "DeepSeek C10: P=65,536; NH=16,777,216\n"
                    "Mean of 16 revisits; one observation per request.",
                    ha="center",
                    fontsize=10,
                )
                figure.tight_layout(rect=(0, 0.23, 1, 1))
                figures.append(
                    {
                        "name": name,
                        "candidate_tokens": candidate,
                        "schemes": list(schemes),
                        "axes_count": len(figure.axes),
                        "yscale": yscale,
                    }
                )
                plot.save_figure(figure, output, name)

    with (output / "plot_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(plotted[0]))
        writer.writeheader()
        writer.writerows(plotted)
    source_dir = output / "source"
    source_dir.mkdir()
    for path in (Path(__file__), Path(plot.__file__)):
        shutil.copy2(path, source_dir / path.name)
    provenance = {
        "schema": "deepseek-c10-revisit-figures-v3",
        "run_id": output.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "input_summary": {"path": str(summary_path), "sha256": digest(summary_path)},
        "benchmark_run_ids": sorted({row["run_id"] for row in rows.values()}),
        "schemes": list(SCHEMES),
        "plotted_values": len(plotted),
        "figures": figures,
        "layout": "one axes per file; one HBM/offload comparison and one offload-only view per candidate",
        "axes": {
            "x": "log2 history tokens",
            "y": "log for HBM/offload, zero-based linear for offload-only",
        },
        "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
        "artifacts_sha256": {
            str(path.relative_to(output)): digest(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
        "boundary": (
            "Presentation-only filtering of the accepted matrix; no GPU execution or new "
            "measurements. Original four-method statistics and validation are unchanged. "
            "Only revisit mean latency is plotted, directly in milliseconds."
        ),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = render(args.summary, args.output_dir)
    print(
        f"Rendered {len(result['figures'])} candidate figures, "
        f"{result['plotted_values']} plotted values: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
