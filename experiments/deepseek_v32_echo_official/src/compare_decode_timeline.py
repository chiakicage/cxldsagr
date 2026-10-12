"""Combine the published official decode and local H64K/A1 GPU timelines."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiments.deepseek_v32_echo_official.src import compare_timeline as comparison
from experiments.deepseek_v32_echo_official.src import decode_display
from experiments.deepseek_v32_echo_official.src import q1_timeline_categories as categories

engine = comparison.engine
EXPERIMENT = comparison.EXPERIMENT


def official_panels(directory):
    publication_path = directory / "publication.json"
    publication = comparison.read(publication_path)
    paths = [directory / name for name in ("report.json", "windows.json", "activities.json")]
    for path in paths:
        comparison.require_hash(path, publication["files"][path.name])
    report, windows, activities = (comparison.read(path) for path in paths)
    if report["numerical_acceptance"] is not False or windows != report["windows"]:
        raise ValueError("Official published windows or numerical status differ")
    panels = []
    for method, window in zip(("hbm", "echo"), windows, strict=True):
        if (
            window["case"] != method
            or window["phase"] != "decode"
            or window["history_tokens"] != 65536
            or window["query_tokens"] != 1
            or window["graph_launches"] != 1
            or not window["every_graph_node_verified"]
            or len(activities[method]) != window["activity_count"]
        ):
            raise ValueError("Expected the verified official H64K Q1 decode")
        panels.append(
            comparison.verify_panel(
                {
                    "source": "Official SGLang",
                    "condition": "normal decode; natural residency",
                    "method": method,
                    "window": window,
                    "layers": window["layer_boundaries"],
                    "rows": activities[method],
                    "profile_run_id": report["sources"][f"{method}_profile"]["run_id"],
                }
            )
        )
    return panels, [publication_path, *paths]


def draw(panels, output, *, xlim_ms=(0, 5), figure_title=None, compact_style=False):
    if compact_style:
        return draw_compact_comparison(panels, output, xlim_ms, figure_title)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "h64k-q1-comparison",
        }
    )
    figure, axes = plt.subplots(len(panels), 1, figsize=(12, 14), sharex=True, squeeze=False)
    axes = axes[:, 0]
    lanes = ["Compute", "H2D", "D2H", "GPU control", "GPU idle"]
    colors = {**engine.COLORS, "GPU control": "#777777"}
    for axis, panel in zip(axes, panels, strict=True):
        window = panel["window"]
        start, end = window["start_ns"], window["end_ns"]
        if xlim_ms[0] != 0 or window["window_ms"] > xlim_ms[1]:
            raise ValueError("The shared axis must contain the entire window")
        for row in panel["rows"]:
            lane, category, hatched = decode_display.activity_style(row)
            if lane not in lanes[:-1]:
                raise ValueError(f"Unrecognized GPU lane: {lane}")
            category = "GPU control" if lane == "GPU control" else category
            left, right = max(start, row["start_ns"]), min(end, row["end_ns"])
            axis.broken_barh(
                [((left - start) / 1e6, (right - left) / 1e6)],
                (lanes.index(lane) - 0.27, 0.54),
                facecolors=colors[category],
                edgecolors="#333333" if hatched else "none",
                hatch="////" if hatched else None,
                linewidth=0.25,
            )
        axis.broken_barh(
            [
                ((left - start) / 1e6, (right - left) / 1e6)
                for left, right in panel["idle_intervals_ns"]
            ],
            (3.73, 0.54),
            facecolors=colors["GPU idle"],
        )
        for layer in panel["layers"]:
            left = (layer.get("compute_start_ns", layer["start_ns"]) - start) / 1e6
            axis.axvline(left, color="#777777", linewidth=0.5, linestyle=":")
            axis.text(left, -0.50, f"L{layer['layer']}", va="bottom", fontsize=9)
        axis.axvline(window["window_ms"], color="#777777", linewidth=0.6, linestyle="--")
        axis.set_yticks(range(len(lanes)), lanes)
        axis.set_ylim(4.6, -0.95)
        axis.set_xlim(*xlim_ms)
        if xlim_ms == (0, 5):
            axis.set_xticks(range(6))
        axis.tick_params(axis="x", labelbottom=True)
        axis.set_axisbelow(True)
        axis.grid(axis="x", alpha=0.18)
        axis.set_title(
            f"{panel['source']} · {comparison.METHODS[panel['method']]}"
            f" | {window['window_ms']:.3f} ms | idle {window['gpu_idle_ms']:.3f} ms"
            f" | {panel['condition']}",
            loc="left",
            fontsize=10,
            pad=8,
        )
    axes[-1].set_xlabel("Elapsed GPU time from window start (ms)")
    categories = [
        "Projection / RoPE",
        "Indexer / top-k",
        "Attention",
        "Output / MLP",
        "H2D",
        "D2H",
        "GPU control",
        "GPU idle",
    ]
    handles = [Patch(facecolor=colors[name], label=name) for name in categories]
    handles.append(
        Patch(
            facecolor="white",
            edgecolor="#333333",
            hatch="////",
            label="Fused / potential mapped-host IO",
        )
    )
    figure.suptitle(
        figure_title or "H64K · Q1 · L0–L2 | Official decode and our single-token step",
        y=0.992,
        fontsize=13,
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.970),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.918), h_pad=1.2)
    for suffix in ("svg", "png"):
        figure.savefig(
            output / f"timeline_decode.{suffix}",
            dpi=180,
            metadata={"Date": None} if suffix == "svg" else None,
        )
    plt.close(figure)


def draw_compact_comparison(panels, output, xlim_ms, figure_title):
    """Use the local phase palette and typography with all GPU control visible."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if (
        not panels
        or xlim_ms[0] != 0
        or any(panel["window"]["window_ms"] > xlim_ms[1] for panel in panels)
    ):
        raise ValueError("The shared axis must contain every complete window")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "q1-aligned-decode",
        }
    )
    panel_height, panel_gap, top_margin, bottom_margin = 1.25, 0.60, 0.90, 2.10
    figure_height = (
        top_margin + bottom_margin + len(panels) * panel_height + (len(panels) - 1) * panel_gap
    )
    figure, axes = plt.subplots(len(panels), 1, figsize=(13, figure_height), squeeze=False)
    lanes = ["Compute", "H2D", "D2H", "GPU control", "GPU idle"]
    colors = {**engine.COLORS, "GPU control": "#777777"}
    for axis, panel in zip(axes[:, 0], panels, strict=True):
        window = panel["window"]
        start, end = window["start_ns"], window["end_ns"]
        for row in panel["rows"]:
            lane, category, hatched = decode_display.activity_style(row)
            if lane not in lanes[:-1]:
                raise ValueError(f"Unrecognized GPU lane: {lane}")
            if lane == "GPU control" and category not in comparison.compact.ANNOTATION_COLORS:
                category = "GPU control"
            left, right = max(start, row["start_ns"]), min(end, row["end_ns"])
            axis.broken_barh(
                [((left - start) / 1e6, (right - left) / 1e6)],
                (lanes.index(lane) - 0.25, 0.50),
                facecolors=colors[category],
                edgecolors="white" if hatched else "none",
                hatch="////" if hatched else None,
                linewidth=0,
            )
        axis.broken_barh(
            [
                ((left - start) / 1e6, (right - left) / 1e6)
                for left, right in panel["idle_intervals_ns"]
            ],
            (3.75, 0.50),
            facecolors=colors["GPU idle"],
        )
        for layer in panel["layers"]:
            left, right = (layer["start_ns"] - start) / 1e6, (layer["end_ns"] - start) / 1e6
            axis.text((left + right) / 2, -0.52, f"L{layer['layer']}", ha="center", fontsize=9)
            axis.axvline(right, color="#66717D", linestyle=":", alpha=0.5, linewidth=0.6)
        axis.axvline(window["window_ms"], color="#66717D", linestyle="--", linewidth=0.8)
        axis.set_yticks(range(len(lanes)), lanes)
        axis.set_ylim(4.58, -0.98)
        axis.set_xlim(*xlim_ms)
        axis.tick_params(axis="both", labelsize=9)
        axis.set_axisbelow(True)
        axis.grid(axis="x", alpha=0.12)
        for edge in ("left", "top", "right"):
            axis.spines[edge].set_visible(False)
        indexer_us = next(
            metric["union_us"]
            for metric in categories.category_metrics(panel)
            if metric["layer"] == "all" and metric["category"] == "Indexer / top-k"
        )
        axis.set_title(
            f"{panel['source']} {comparison.METHODS[panel['method']]}  |  {window['window_ms']:.3f} ms"
            f"  |  GPU idle {window['gpu_idle_ms']:.3f} ms"
            f"  |  Indexer {indexer_us:.3f} µs",
            loc="left",
            fontsize=10.5,
            pad=9,
        )
    axes[-1, 0].set_xlabel("Time from L0 first compute kernel start (ms)", fontsize=10)
    figure.text(
        0.09,
        1 - 0.32 / figure_height,
        figure_title or "ECHO | H=65,536, Q=1, L0-L2 | Official and local warm",
        fontsize=15,
    )
    names = [
        "Projection / RoPE",
        "Indexer / top-k",
        "Attention",
        "Output / MLP",
        "H2D",
        "D2H",
        "GPU control",
        "GPU idle",
        "ECHO prepare",
        "ECHO finalize",
        "ECHO hint",
    ]
    handles = [
        Patch(
            facecolor=colors[name],
            label={
                "H2D": "H2D / official recall kernel",
                "D2H": "D2H: GPU to DRAM",
                "Indexer / top-k": "Indexer: scores + clean + full top-k",
            }.get(name, name),
        )
        for name in names
    ]
    handles.append(
        Patch(
            facecolor=colors["Indexer / top-k"],
            edgecolor="white",
            hatch="////",
            label="Fused / potential mapped-host IO",
        )
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.20 / figure_height),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        top=1 - top_margin / figure_height,
        bottom=bottom_margin / figure_height,
        hspace=panel_gap / panel_height,
    )
    for suffix in ("svg", "png"):
        figure.savefig(
            output / f"timeline_decode.{suffix}",
            dpi=180,
            metadata={"Date": None} if suffix == "svg" else None,
        )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-report", type=Path, default=EXPERIMENT / "report/sglang_decode")
    parser.add_argument("--mfu-report", type=Path, default=comparison.MFU / "report/h64k_a1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(EXPERIMENT / "output"):
        parser.error("--output-dir must be inside this experiment's output directory")
    if args.publish_dir and not args.publish_dir.resolve().is_relative_to(EXPERIMENT / "report"):
        parser.error("--publish-dir must be inside this experiment's report directory")
    if output.exists() or (args.publish_dir and args.publish_dir.exists()):
        raise FileExistsError("Use new output and publication directories")
    official, official_paths = official_panels(args.official_report)
    local, local_paths = comparison.mfu_panels(
        args.mfu_report, None, "extend", receipt_subdir="h65536_a1"
    )
    shape = comparison.mfu_report_shape(args.mfu_report, receipt_subdir="h65536_a1")
    if shape["prefix_tokens"] != 65536 or shape["extend_tokens"] != 1:
        raise ValueError("Expected the published H64K/A1 local shape")
    for panel in local:
        panel.update(
            source="Ours (local MFU)",
            condition="fixed token; HBM resident"
            if panel["method"] == "hbm"
            else "fixed token; cold history KV",
        )
    panels = official + local
    aligned = [categories.align_panel(panel) for panel in panels]
    sources = [
        Path(__file__),
        Path(comparison.__file__),
        Path(engine.__file__),
        Path(comparison.compact.__file__),
        Path(comparison.timeline.__file__),
        Path(decode_display.__file__),
        Path(categories.__file__),
    ]
    paths = [*official_paths, *local_paths, *sources]
    before = {str(path.resolve()): engine.sha256(path) for path in paths}
    output.mkdir(parents=True)
    draw(
        aligned,
        output,
        compact_style=True,
        figure_title="H64K · Q1 · L0–L2 | Official decode and local cold steps",
    )
    engine.write_csv(
        output / "windows.csv",
        [
            {
                "source": panel["source"],
                "method": panel["method"],
                "condition": panel["condition"],
                "profile_run_id": panel["profile_run_id"],
                **{
                    key: panel["window"][key]
                    for key in ("start_ns", "end_ns", "window_ms", "gpu_idle_ms")
                },
            }
            for panel in panels
        ],
    )
    (output / "panels.json").write_text(json.dumps(panels, indent=2) + "\n")
    (output / "aligned_panels.json").write_text(json.dumps(aligned, indent=2) + "\n")
    engine.write_csv(
        output / "category_metrics.csv",
        [row for panel in aligned for row in categories.category_metrics(panel)],
    )
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshot / path.name)
    if before != {str(path.resolve()): engine.sha256(path) for path in paths}:
        raise ValueError("Input or renderer changed during rendering")
    provenance = {
        "schema": "official-local-q1-timeline-comparison-v1",
        "run_id": output.name,
        "measurement": "Presentation of existing verified NSYS windows; no new GPU execution or clean timing.",
        "inputs_and_sources_sha256": before,
        "shared_xlim_ms": [0, 5],
        "category_policy": categories.POLICY,
        "window": "Published L0 first compute to L2 last compute windows; local dense also retains earlier L0 history H2D when present.",
        "io_display": decode_display.RECALL_DISPLAY["boundary"],
        "display_policy": decode_display.RECALL_DISPLAY,
        "fused_display_policy": decode_display.FUSED_DISPLAY,
        "numerical_acceptance_between_implementations": False,
        "local_profile_run_ids": {panel["method"]: panel["profile_run_id"] for panel in local},
        "local_panel_provenance": {
            panel["method"]: panel["provenance"] for panel in local if "provenance" in panel
        },
        "conditions": {
            "official": "Normal autoregressive decode, input token 57841, natural radix/cache residency, GPU7; offload has 65664 device and 16777216 host token slots.",
            "local": "One supplied token 111090, persistent append; offload history main KV starts in DRAM with its HBM residency cleared, indexer remains in HBM; GPU0, P=NH=65600.",
        },
        "limitations": [
            "Different input tokens, cache state, capacities, framework environments and physical GPUs; this figure does not establish equivalent-workload speedups.",
            "All durations include profiling effects and exclude the full Engine request boundary.",
            "Official numerical equivalence remains unaccepted; selected decode scope completeness does not imply whole-process trace completeness.",
        ],
        "audit": {
            "published_hashes": True,
            "exact_window_and_idle_durations": True,
            "local_compute_categories_and_idle_intervals": True,
            "source_to_aligned_activity_identity": True,
        },
        "artifacts_sha256": {
            path.name: engine.sha256(path) for path in output.iterdir() if path.is_file()
        },
        "source_snapshot_sha256": {path.name: engine.sha256(path) for path in snapshot.iterdir()},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if args.publish_dir:
        # Renderer snapshots remain in the run directory; publish report artifacts only.
        args.publish_dir.mkdir(parents=True)
        for path in output.iterdir():
            if path.is_file():
                shutil.copyfile(path, args.publish_dir / path.name)
    print(output)


if __name__ == "__main__":
    main()
