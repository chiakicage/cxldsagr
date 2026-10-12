"""Expand authenticated L0 graph activities around Indexer; no GPU execution."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiments.deepseek_v32_echo_official.src import compare_warm_decode_timeline as warm
from experiments.deepseek_v32_echo_official.src import decode_display
from experiments.deepseek_v32_echo_official.src import engine_timeline as engine

EXPERIMENT = Path(__file__).resolve().parents[1]
ALIGNED_PANELS_SHA256 = "9179a5a9e3e77b3de64938673a6aa695723f7782bdb9374763150b7a953ef99f"
PUBLISHED_FILES = (
    "timeline_indexer_detail.svg",
    "timeline_indexer_detail.png",
    "segments.csv",
    "activities.json",
    "summary.json",
    "provenance.json",
    "publication_manifest.json",
)
# These IDs identify nodes in the two fixed, capture-authenticated profiles.
# Names/repetition are never used to infer layer ownership.
SPECS = (
    {
        "source": "Official SGLang",
        "profile_run_id": "h64k_decode_echo_profile_20261008_02",
        "layer": 0,
        "prefix": "858993",
        "groups": (
            ("O1", 4615, 4619, "Write index K/scales; compute head weights"),
            ("O2", 4620, 4621, "Build scheduler metadata; reset prefetch count"),
            ("O3", 4622, 4622, "Fused scoring / predictive staging prefetch"),
            ("O4", 4623, 4628, "Top-k: init, select + host IDs, padded-row masks"),
            ("O5", 4629, 4632, "Update decode EMA; D2D writeback"),
            ("O6", 4633, 4637, "Append capacity; clone / mask slot priorities"),
            ("O7", 4638, 4638, "Append argmin: scan 65,665 priority entries"),
            ("O8", 4639, 4648, "Free / allocate; maps; append KV + host write"),
            ("O9", 4649, 4656, "Map selected host IDs; count slot demand"),
            ("O10", 4657, 4674, "Recall capacity; protect hits; free / allocate"),
            ("O11", 4675, 4678, "Recall + maps; clear staging; bounds check"),
            ("O12", 4679, 4681, "FA3 setup, attention and combine"),
        ),
    },
    {
        "source": "Local warm",
        "profile_run_id": "deepseek_h64k_a1_official_input_warm_20261011_01_echo_profile",
        "layer": "layer_0",
        "prefix": "176093659",
        "groups": (
            ("L1", 175, 176, "Write index K/scales: two D2D copies"),
            ("L2", 177, 179, "Verify free slots; select 64; reset scratch"),
            ("L3", 180, 183, "Tables; pack full visible index; scheduler"),
            ("L4", 184, 184, "Fused scoring / predictive staging prefetch"),
            ("L5", 185, 185, "Causal score clean"),
            ("L6", 186, 186, "FlashInfer SMALL top-k selection"),
            ("L7", 187, 187, "CUB sort + invalid-ID mask"),
            ("L8", 188, 188, "Validate promotion; protect exact hits"),
            ("L9", 189, 194, "Publish staging; clear tags; finalize FIFO"),
            ("L10", 195, 196, "Score mean hint + decode EMA"),
            ("L11", 197, 200, "Allocate append slot; KV / maps; async D2H"),
            ("L12", 201, 203, "Exact selection union; classify hits / misses"),
            ("L13", 204, 209, "Prepare recall; gather (0 records); publish / map"),
            ("L14", 210, 210, "Convert attention inputs"),
            ("L15", 211, 212, "Split FlashMLA attention + LSE combine"),
        ),
    },
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def select_detail(panel, spec):
    """Require exact coverage of the chosen window, including concurrent copies."""
    warm.require(panel["profile_run_id"] == spec["profile_run_id"], "Unexpected profile ID")
    warm.require(panel["source"] == spec["source"], "Unexpected source")
    groups, selected = [], []
    all_ids = [row["graph_node_id"] for row in panel["rows"]]
    warm.require(len(all_ids) == len(set(all_ids)), "Duplicate source graph nodes")
    by_id = {row["graph_node_id"]: row for row in panel["rows"]}
    assigned = set()
    for job_id, first, last, action in spec["groups"]:
        ids = [int(spec["prefix"] + str(suffix)) for suffix in range(first, last + 1)]
        warm.require(not assigned.intersection(ids), "Graph node assigned to multiple jobs")
        warm.require(all(node in by_id for node in ids), f"Missing graph node for {job_id}")
        rows = sorted((by_id[node] for node in ids), key=lambda row: row["start_ns"])
        warm.require(all(row["layer"] == spec["layer"] for row in rows), "Layer differs")
        warm.require(all(row["end_ns"] > row["start_ns"] for row in rows), "Empty activity")
        intervals = [(row["start_ns"], row["end_ns"]) for row in rows]
        groups.append(
            {
                "job_id": job_id,
                "action": action,
                "graph_node_ids": ids,
                "start_ns": min(left for left, _ in intervals),
                "end_ns": max(right for _, right in intervals),
                "gpu_active_ns": engine.duration(intervals),
                "activity_sum_ns": sum(right - left for left, right in intervals),
                "rows": rows,
            }
        )
        selected.extend(rows)
        assigned.update(ids)
    start, end = groups[0]["start_ns"], groups[-1]["end_ns"]
    warm.require(
        {
            row["graph_node_id"]
            for row in panel["rows"]
            if row["start_ns"] < end and row["end_ns"] > start
        }
        == assigned,
        "Zoom has unassigned or outside graph activities",
    )
    warm.require(
        all(start <= row["start_ns"] < row["end_ns"] <= end for row in selected), "Clipped node"
    )
    # NSYS omits the executable graph ID on official memcpy records and writes 0
    # on local memcpy records. Their authenticated native node IDs carry ownership.
    warm.require(
        len({row["graph_id"] for row in selected if row["kind"] == "kernel"}) == 1,
        "Multiple executable kernel graphs",
    )
    idle = engine.union(
        (max(start, left), min(end, right)) for left, right in panel["idle_intervals_ns"]
    )
    busy = [(row["start_ns"], row["end_ns"]) for row in selected]
    warm.require(idle == engine.complement(busy, start, end), "Saved idle intervals differ")
    warm.require(
        engine.duration(busy) + engine.duration(idle) == end - start, "Window conservation failed"
    )
    return {
        "source": panel["source"],
        "profile_run_id": panel["profile_run_id"],
        "layer": 0,
        "anchor": "First current-token index-cache write",
        "start_ns": start,
        "end_ns": end,
        "anchor_offset_from_l0_compute_ns": start - panel["layers"][0]["start_ns"],
        "gpu_busy_ns": engine.duration(busy),
        "gpu_idle_ns": engine.duration(idle),
        "idle_intervals_ns": idle,
        "groups": groups,
    }


def draw(details, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "q1-indexer-detail",
        }
    )
    colors = {**engine.COLORS, "GPU control": "#777777"}
    pitch, top, gap, bottom = 0.25, 1.10, 0.75, 2.0
    heights = [(len(detail["groups"]) + 1) * pitch for detail in details]
    height = top + sum(heights) + gap + bottom
    figure = plt.figure(figsize=(13, height))
    cursor = height - top
    for detail, panel_height in zip(details, heights, strict=True):
        cursor -= panel_height
        axis = figure.add_axes([0.345, cursor / height, 0.58, panel_height / height])
        start, end = detail["start_ns"], detail["end_ns"]
        for y, group in enumerate(detail["groups"]):
            for row in group["rows"]:
                lane, category, hatched = decode_display.activity_style(row)
                if lane == "GPU control" and category not in colors:
                    category = "GPU control"
                axis.broken_barh(
                    [((row["start_ns"] - start) / 1000, (row["end_ns"] - row["start_ns"]) / 1000)],
                    (y - 0.25, 0.50),
                    facecolors=colors[category],
                    edgecolors="white" if hatched else "none",
                    hatch=decode_display.FUSED_DISPLAY["hatch"] if hatched else None,
                    linewidth=0,
                )
            axis.text(
                1.025,
                y,
                f"{group['gpu_active_ns'] / 1000:.3f}",
                transform=axis.get_yaxis_transform(),
                va="center",
                fontsize=9,
                weight="bold" if group["job_id"] in ("O7", "L8") else "normal",
            )
        idle_y = len(detail["groups"])
        axis.broken_barh(
            [
                ((left - start) / 1000, (right - left) / 1000)
                for left, right in detail["idle_intervals_ns"]
            ],
            (idle_y - 0.25, 0.50),
            facecolors=colors["GPU idle"],
        )
        axis.text(
            1.025,
            idle_y,
            f"{detail['gpu_idle_ns'] / 1000:.3f}",
            transform=axis.get_yaxis_transform(),
            va="center",
            fontsize=9,
        )
        axis.axvline((end - start) / 1000, color="#66717D", linestyle="--", linewidth=0.8)
        axis.set_yticks(
            range(idle_y + 1),
            [f"{group['job_id']}  {group['action']}" for group in detail["groups"]]
            + ["GPU idle  (all streams)"],
        )
        axis.set_ylim(idle_y + 0.65, -0.65)
        axis.set_xlim(0, 260)
        axis.set_xticks(range(0, 261, 20))
        axis.tick_params(axis="both", labelsize=9, length=0)
        axis.tick_params(axis="y", pad=8)
        axis.set_axisbelow(True)
        axis.grid(axis="x", alpha=0.12)
        for edge in ("left", "top", "right"):
            axis.spines[edge].set_visible(False)
        title_y = (cursor + panel_height + 0.20) / height
        figure.text(
            0.035,
            title_y,
            f"{detail['source']} ECHO  |  L0 zoom {(end - start) / 1000:.3f} µs",
            fontsize=10.5,
        )
        figure.text(0.94, title_y, "GPU active\n(µs)", fontsize=9, va="center")
        axis.set_xlabel("Time from first L0 current-index-cache write (µs)", fontsize=10)
        cursor -= gap
    figure.text(
        0.035, 1 - 0.33 / height, "H64K · Q1 · L0 | GPU work before and after Indexer", fontsize=15
    )
    figure.text(
        0.035,
        1 - 0.65 / height,
        "Same scale and anchor on both sides; each bar is an original GPU activity interval.",
        fontsize=10,
    )
    names = [
        "Projection / RoPE",
        "Indexer / top-k",
        "Attention",
        "GPU control",
        "ECHO prepare",
        "ECHO finalize",
        "ECHO hint",
        "H2D",
        "D2H",
        "GPU idle",
    ]
    handles = [
        Patch(
            facecolor=colors[name],
            label={"H2D": "H2D / official recall kernel", "D2H": "D2H: GPU to DRAM"}.get(
                name, name
            ),
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
        bbox_to_anchor=(0.5, 0.68 / height),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.text(
        0.035,
        0.45 / height,
        "Active times are interval unions per row; overlapping rows cannot be added as elapsed time.",
        fontsize=9,
    )
    figure.text(
        0.035,
        0.25 / height,
        "Hatching marks IO capability. Local warm H2D = 0 B; official profile bytes unknown. White after the dashed end is outside the zoom.",
        fontsize=9,
    )
    for suffix in ("svg", "png"):
        figure.savefig(
            output / f"timeline_indexer_detail.{suffix}",
            dpi=180,
            metadata={"Date": None} if suffix == "svg" else None,
        )
    plt.close(figure)


def build_report(comparison_report, output, publish_dir=None):
    comparison_report, output = comparison_report.resolve(), output.resolve()
    warm.require(
        output.parent == (EXPERIMENT / "output/data").resolve(),
        "Output must use a new run ID under output/data",
    )
    if publish_dir is not None:
        publish_dir = publish_dir.resolve()
        warm.require(
            publish_dir.is_relative_to((EXPERIMENT / "report").resolve()),
            "Publication must be under report",
        )
    if output.exists() or (publish_dir is not None and publish_dir.exists()):
        raise FileExistsError("Use new output and publication directories")
    publication = warm.validate_manifest(comparison_report)
    warm.require(
        publication["schema"] == "official-local-warm-echo-display-artifacts-v1",
        "Expected authenticated warm comparison",
    )
    warm.require("aligned_panels.json" in publication["files_sha256"], "Unbound aligned panels")
    warm.require(
        publication["files_sha256"]["aligned_panels.json"] == ALIGNED_PANELS_SHA256,
        "Node grouping and labels are bound to a different saved profile payload",
    )
    panels = json.loads((comparison_report / "aligned_panels.json").read_text())
    warm.require(len(panels) == 2, "Expected exactly two comparison panels")
    sources = [
        Path(__file__),
        Path(warm.__file__),
        Path(warm.comparison.__file__),
        Path(engine.__file__),
        Path(decode_display.__file__),
    ]
    inputs = [
        comparison_report / name
        for name in (*publication["files_sha256"], "publication_manifest.json")
    ]
    before = {str(path): engine.sha256(path) for path in [*sources, *inputs]}
    details = [select_detail(panel, spec) for panel, spec in zip(panels, SPECS, strict=True)]
    output.mkdir(parents=True)
    draw(details, output)
    write_json(output / "activities.json", details)
    segments = []
    for detail in details:
        for group in detail["groups"]:
            segments.append(
                {
                    "source": detail["source"],
                    "profile_run_id": detail["profile_run_id"],
                    "layer": 0,
                    **{key: value for key, value in group.items() if key != "rows"},
                    "zoom_start_us": (group["start_ns"] - detail["start_ns"]) / 1000,
                    "zoom_end_us": (group["end_ns"] - detail["start_ns"]) / 1000,
                    "gpu_active_us": group["gpu_active_ns"] / 1000,
                }
            )
    engine.write_csv(output / "segments.csv", segments)
    summary = {
        "schema": "q1-indexer-detail-v1",
        "run_id": output.name,
        "new_gpu_measurement": False,
        "numerical_acceptance_between_implementations": False,
        "shared_xlim_us": [0, 260],
        "anchor": "First current-token L0 index-cache write to attention end",
        "panels": [
            {
                **{
                    key: value
                    for key, value in detail.items()
                    if key not in ("groups", "idle_intervals_ns")
                },
                "activity_count": sum(len(group["rows"]) for group in detail["groups"]),
            }
            for detail in details
        ],
        "io_boundary": "Local warm fused prefetch and exact gather have verified 0 records / 0 B; official mapped-host bytes unknown. Observer counters are not assigned to this profile.",
    }
    write_json(output / "summary.json", summary)
    shutil.copytree(comparison_report, output / "input_report")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshot / path.name)
    warm.require(
        before == {str(path): engine.sha256(path) for path in [*sources, *inputs]},
        "Input or renderer changed during rendering",
    )
    provenance = {
        "schema": "q1-indexer-detail-provenance-v1",
        "run_id": output.name,
        "inputs_and_sources_sha256": before,
        "source_snapshot_sha256": {path.name: engine.sha256(path) for path in snapshot.iterdir()},
        "source_publication_run_id": publication["run_id"],
        "grouping_policy": "Explicit native graph node IDs bound to the exact saved profile IDs; authenticated layer ownership retained. No name or repetition-based layer inference.",
        "io_display_policy": decode_display.RECALL_DISPLAY,
        "fused_display_policy": decode_display.FUSED_DISPLAY,
        "timing_boundary": "Original integer GPU activity intervals; row duration is interval union. Idle is the original all-stream complement clipped to each zoom. Group overlap is retained.",
        "audit": {
            "exact_activity_coverage": True,
            "saved_idle_conservation": True,
            "unique_graph_node_and_layer_ownership": True,
            "input_and_renderer_stable": True,
            "new_gpu_execution": False,
        },
    }
    write_json(output / "provenance.json", provenance)
    write_json(
        output / "publication_manifest.json",
        {
            "schema": "q1-indexer-detail-publication-v1",
            "run_id": output.name,
            "files_sha256": {
                name: engine.sha256(output / name)
                for name in PUBLISHED_FILES
                if name != "publication_manifest.json"
            },
        },
    )
    warm.validate_manifest(output)
    if publish_dir is not None:
        publish_dir.mkdir(parents=True)
        for name in PUBLISHED_FILES:
            shutil.copyfile(output / name, publish_dir / name)
        warm.validate_manifest(publish_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-report", type=Path, default=EXPERIMENT / "report/warm_decode_aligned"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish-dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_report(args.comparison_report, args.output_dir, args.publish_dir), indent=2
        )
    )


if __name__ == "__main__":
    main()
