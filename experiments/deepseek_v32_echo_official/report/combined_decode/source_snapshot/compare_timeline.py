"""Combine published Engine and local MFU timelines without GPU execution."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from experiments.deepseek_v32_echo_official.src import engine_timeline as engine
from experiments.deepseek_v32_mfu.src import compact_timeline as compact
from experiments.deepseek_v32_mfu.src import timeline

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/deepseek_v32_echo_official"
MFU = ROOT / "experiments/deepseek_v32_mfu"
METHODS = {
    "hbm": "HBM-only",
    "echo": "ECHO",
    "serial_sparse": "Serial sparse",
    "dense_prefetch": "Dense prefetch",
}


def read(path):
    return json.loads(path.read_text())


def require_hash(path, expected):
    if engine.sha256(path) != expected:
        raise ValueError(f"Published input changed: {path}")


def verify_panel(panel):
    window = panel["window"]
    start, end = window["start_ns"], window["end_ns"]
    if end - start != round(window["window_ms"] * 1e6):
        raise ValueError("Window duration differs from the published duration")
    idle = engine.complement(
        [(row["start_ns"], row["end_ns"]) for row in panel["rows"]], start, end
    )
    if engine.duration(idle) != round(window["gpu_idle_ms"] * 1e6):
        raise ValueError("GPU idle differs from the published duration")
    panel["idle_intervals_ns"] = idle
    if [layer["layer"] for layer in panel["layers"]] != [0, 1, 2]:
        raise ValueError("Expected exactly L0-L2")
    if start != panel["layers"][0]["start_ns"] or end != panel["layers"][-1]["end_ns"]:
        raise ValueError("Layer endpoints differ from the published window")
    for row in panel["rows"]:
        if row["start_ns"] >= end or row["end_ns"] <= start:
            raise ValueError("Activity does not overlap the selected window")
    return panel


def engine_panels(report_dir, phase):
    publication = read(report_dir / "publication.json")
    paths = [report_dir / name for name in ("report.json", "selected_activities.csv")]
    for path in paths:
        require_hash(path, publication["files"][path.name]["sha256"])
    report = read(paths[0])
    with paths[1].open(newline="") as stream:
        activities = list(csv.DictReader(stream))
    panels = []
    for method in ("hbm", "echo"):
        panel = report["timeline"]["panels"][method][phase]
        query = (
            {"start": 64512, "end": 65536, "q": 1024}
            if phase == "prefill"
            else {"start": 65536, "end": 65664, "q": 128}
        )
        if {key: panel["query"][key] for key in query} != query:
            raise ValueError("Engine query differs from the selected MFU phase")
        if phase == "prefill" and panel["prefill_chunks"] != 64:
            raise ValueError("Expected all 64 Engine history chunks")
        rows = [
            {
                **row,
                "start_ns": int(row["start_ns"]),
                "end_ns": int(row["end_ns"]),
                "potential_io": row["potential_io"] == "True",
            }
            for row in activities
            if row["case"] == method and row["selected_phase"] == phase
        ]
        if len(rows) != panel["window"]["activity_count"]:
            raise ValueError("Engine activity count differs from the published window")
        panels.append(
            verify_panel({**panel, "source": "SGLang Engine", "method": method, "rows": rows})
        )
    return panels, [*paths, report_dir / "publication.json"]


def mfu_panels(report_dir, rows_path, phase, *, receipt_subdir=""):
    manifest_path = report_dir / "publication_manifest.json"
    publication = read(manifest_path)
    if publication.get("schema") == "isolated-method-artifacts-v1":
        if receipt_subdir not in ("", "h65536_a1"):
            raise ValueError("Isolated comparison uses the report's three-layer root panels")
        return isolated_mfu_panels(report_dir, rows_path, phase, publication)
    receipt_path = report_dir / receipt_subdir / "compact_receipt.json"
    provenance_path = report_dir / "provenance.json"
    receipt = read(receipt_path)
    manifest = publication["files_sha256"]
    receipt_digest = manifest[receipt_path.relative_to(report_dir).as_posix()]
    require_hash(receipt_path, receipt_digest)
    require_hash(provenance_path, manifest[provenance_path.name])
    if rows_path is None:
        inputs = read(provenance_path)["inputs_sha256"]
        if receipt_subdir:
            candidates = [
                Path(path).parent / "window_rows.json"
                for path, digest in inputs.items()
                if Path(path).name == "compact_receipt.json" and digest == receipt_digest
            ]
        else:
            candidates = [
                Path(path)
                for path, digest in inputs.items()
                if Path(path).name == "window_rows.json"
                and digest == receipt["artifacts_sha256"]["window_rows.json"]
            ]
        if len(candidates) != 1:
            raise ValueError("Expected one published MFU window_rows.json source")
        rows_path = candidates[0]
    require_hash(rows_path, receipt["artifacts_sha256"]["window_rows.json"])
    saved = read(rows_path)[phase]
    if [panel["method"] for panel in saved] != list(METHODS):
        raise ValueError("Expected the four published MFU methods")
    panels = []
    for panel in saved:
        method = panel["method"]
        expected = receipt["metrics"][f"{phase}/{method}"]
        if panel["phase"] != phase or (phase == "prefill" and panel["chunk"] != 63):
            raise ValueError("Unexpected MFU phase or prefill chunk")
        if panel["window"] != expected["window"]:
            raise ValueError("MFU window differs from the published figure")
        normalized, counts = normalize_mfu_panel(panel, receipt["profile_run_id"])
        if counts != expected["compute_categories"]:
            raise ValueError("MFU compute categories differ from the published figure")
        if normalized["idle_intervals_ns"] != expected["annotation_intervals_ns"]["GPU idle"]:
            raise ValueError("MFU idle intervals differ from the published figure")
        panels.append(normalized)
    return panels, [receipt_path, manifest_path, provenance_path, rows_path]


def normalize_mfu_panel(panel, profile_run_id):
    """Reuse the published lane/purpose classification and verify GPU interval arithmetic."""
    method = panel["method"]
    rows, counts = [], Counter()
    for row in panel["rows"]:
        if row["kind"] == "api":
            continue
        lane, category = row["lane"], "GPU control"
        if lane not in {"Compute", "Compute + IO", "IO", "GPU control"}:
            raise ValueError(f"Unrecognized local GPU lane: {lane}")
        if lane in {"Compute", "Compute + IO"}:
            purpose = row.get("purpose") or timeline.computation_segments([row])[0]["label"]
            category = compact.GROUPS[purpose]
            counts[category] += 1
            category = category.replace("Output / MLP / head", "Output / MLP")
            lane = "Compute"
        elif lane == "IO":
            category = compact.io_direction(row).split(":")[0]
            lane = category
        elif method == "echo" and row["stage"] in compact.ECHO_STAGES:
            category = compact.ECHO_STAGES[row["stage"]]
        rows.append(
            {
                **row,
                "start_ns": row["raw_start_ns"],
                "end_ns": row["raw_end_ns"],
                "lane": lane,
                "category": category,
                "potential_io": row["lane"] == "Compute + IO",
            }
        )
    normalized = verify_panel(
        {**panel, "source": "Local MFU", "rows": rows, "profile_run_id": profile_run_id}
    )
    return normalized, counts


def isolated_summary(report_dir, publication):
    """Verify saved report acceptance and shape without reopening GPU measurements."""
    path = report_dir / "summary.json"
    require_hash(path, publication["files_sha256"][path.name])
    summary = read(path)
    shape = summary.get("configuration", {})
    expected = {
        "prefix_tokens": 65536,
        "extend_tokens": 1,
        "chunk_size": 1024,
        "extend_chunk_size": 1,
        "num_layers": 3,
        "extend_residency": "cold",
        "method_isolation": "fresh-process-one-method-v1",
    }
    if (
        summary.get("schema") != "deepseek-v32-isolated-method-report-v1"
        or summary.get("passed") is not True
        or any(shape.get(key) != value for key, value in expected.items())
        or set(summary.get("methods", {})) != set(METHODS)
        or set(summary.get("children", {})) != set(METHODS)
    ):
        raise ValueError("Expected an accepted isolated H64K/A1 report")
    return summary


def mfu_report_shape(report_dir, *, receipt_subdir=""):
    """Read the publication-bound shape from the format's own metadata."""
    publication = read(report_dir / "publication_manifest.json")
    if publication.get("schema") == "isolated-method-artifacts-v1":
        return isolated_summary(report_dir, publication)["configuration"]
    path = report_dir / receipt_subdir / "compact_receipt.json"
    require_hash(path, publication["files_sha256"][path.relative_to(report_dir).as_posix()])
    return read(path)["shape"]


def isolated_mfu_panels(report_dir, rows_path, phase, publication):
    """Read the four audited child panels while retaining every original binding."""
    if phase not in {"prefill", "extend"}:
        raise ValueError("Unsupported isolated timeline phase")
    summary = isolated_summary(report_dir, publication)
    rows_path = report_dir / "window_rows.json" if rows_path is None else rows_path
    hashes_path = report_dir / "input_hashes.json"
    require_hash(rows_path, publication["files_sha256"]["window_rows.json"])
    require_hash(hashes_path, publication["files_sha256"][hashes_path.name])
    hashes = read(hashes_path)
    saved = read(rows_path)
    if set(saved) != {"prefill", "extend"}:
        raise ValueError("Isolated report must retain both measured phase panels")
    panels, profile_ids, directories = [], set(), set()
    expected_shape = compact.profile_shape(summary["configuration"])
    for requested_phase in ("prefill", "extend"):
        if [panel["method"] for panel in saved[requested_phase]] != list(METHODS):
            raise ValueError("Expected the four published isolated MFU methods")
    for panel in saved[phase]:
        method = panel["method"]
        if (
            panel.get("shape") != expected_shape
            or panel.get("phase") != phase
            or panel.get("chunk") != (63 if phase == "prefill" else None)
        ):
            raise ValueError("Isolated panel differs from its published shape or phase")
        provenance = panel["provenance"]
        child = summary["children"][method]
        profile, benchmark, check = (child[name] for name in ("profile", "bench", "check"))
        profile_id = provenance["profile_run_id"]
        directory = Path(provenance["profile_directory"])
        sqlite = Path(provenance["sqlite"])
        if (
            profile_id != profile["run_id"]
            or str(directory) != profile["directory"]
            or directory.name != profile_id
            or sqlite.parent != directory
            or not directory.is_absolute()
            or profile_id in profile_ids
            or directory in directories
        ):
            raise ValueError("Isolated panel lost its method's original profile process binding")
        profile_ids.add(profile_id)
        directories.add(directory)
        binding = provenance["benchmark"]
        receipt = provenance["validation_receipt"]
        if (
            provenance["result_sha256"] != profile["result_sha256"]
            or binding["run_id"] != benchmark["run_id"]
            or binding["directory"] != benchmark["directory"]
            or binding["result_sha256"] != benchmark["result_sha256"]
            or receipt["receipt_path"] != check["receipt_path"]
            or hashes.get(check["receipt_path"]) != check["receipt_sha256"]
        ):
            raise ValueError("Isolated panel differs from its checked profile/benchmark binding")
        for path, expected in (
            (sqlite, provenance["sqlite_sha256"]),
            (directory / "result.json", provenance["result_sha256"]),
            (Path(binding["directory"]) / "result.json", binding["result_sha256"]),
        ):
            if hashes.get(str(path)) != expected:
                raise ValueError("Isolated panel artifact is absent from the audited input hashes")
        audited = summary["raw_windows"][method][phase]
        if any(
            panel["window"].get(key) != value
            for key, value in audited.items()
            if key in panel["window"]
        ) or audited["native_gpu_intervals"] != sum(row["kind"] != "api" for row in panel["rows"]):
            raise ValueError("Isolated window differs from the report's raw interval audit")
        expected_idle = [
            list(interval) for interval in compact.idle_echo_annotations(panel)["GPU idle"]
        ]
        normalized, _ = normalize_mfu_panel(panel, profile_id)
        if normalized["idle_intervals_ns"] != expected_idle:
            raise ValueError("Isolated MFU idle intervals differ from the saved activities")
        panels.append(normalized)
    return panels, [
        report_dir / "summary.json",
        report_dir / "publication_manifest.json",
        rows_path,
        hashes_path,
    ]


def draw(panels, output, phase):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": f"{phase}-comparison",
        }
    )
    fig, axes = plt.subplots(6, 1, figsize=(12, 13), sharex=True)
    limit = 30.0 if phase == "prefill" else 15.0
    if max(panel["window"]["window_ms"] for panel in panels) >= limit:
        raise ValueError("The shared time axis must contain every complete window")
    shown = set()
    for ax, panel in zip(axes, panels, strict=True):
        window = panel["window"]
        start, end = window["start_ns"], window["end_ns"]
        labels = [
            "GPU idle",
            *(["ECHO control"] if panel["method"] == "echo" else []),
            "D2H",
            "H2D",
            "Compute",
        ]
        lane_y = {label: index for index, label in enumerate(labels)}
        for row in panel["rows"]:
            recall = row["category"] == "ECHO recall (IO unknown)"
            category = "H2D" if recall else row["category"]
            hatched = row["potential_io"] and not recall
            lane = (
                "H2D"
                if recall
                else "ECHO control"
                if category in engine.CONTROL_CATEGORIES
                else row["lane"]
            )
            if lane not in lane_y:
                continue
            shown.add(category)
            left, right = max(start, row["start_ns"]), min(end, row["end_ns"])
            ax.broken_barh(
                [((left - start) / 1e6, (right - left) / 1e6)],
                (lane_y[lane] - 0.27, 0.54),
                facecolors=engine.COLORS[category],
                edgecolors="white" if hatched else "none",
                hatch="////" if hatched else None,
                linewidth=0.35,
            )
        ax.broken_barh(
            [
                ((left - start) / 1e6, (right - left) / 1e6)
                for left, right in panel["idle_intervals_ns"]
            ],
            (-0.27, 0.54),
            facecolors=engine.COLORS["GPU idle"],
        )
        for layer in panel["layers"]:
            left = (layer.get("compute_start_ns", layer["start_ns"]) - start) / 1e6
            right = (layer["end_ns"] - start) / 1e6
            ax.text(
                (left + right) / 2,
                lane_y["Compute"] + 0.44,
                f"L{layer['layer']}",
                ha="center",
                fontsize=10,
            )
            if layer["layer"]:
                ax.axvline(left, color="#777777", linewidth=0.6, linestyle=":")
        ax.axvline(window["window_ms"], color="#777777", linewidth=0.7, linestyle="--")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_ylim(-0.5, lane_y["Compute"] + 1.0)
        ax.set_xlim(0, limit)
        ax.set_xticks([index * limit / 6 for index in range(7)])
        ax.set_title(
            f"{panel['source']} · {METHODS[panel['method']]}"
            f"    |    {window['window_ms']:.3f} ms    |    GPU idle {window['gpu_idle_ms']:.3f} ms",
            loc="left",
            fontsize=11,
            pad=8,
        )
        ax.grid(axis="x", color="#dddddd", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelbottom=True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel(
        "Elapsed time from first L0 compute kernel (ms)"
        if phase == "prefill"
        else "Elapsed time from window start (ms)"
    )
    handles = [
        Patch(facecolor=color, label=key)
        for key, color in engine.COLORS.items()
        if key in shown or key == "GPU idle"
    ]
    handles.append(
        Patch(
            facecolor="white", edgecolor="#555555", hatch="////", label="Dynamic mapped-host path"
        )
    )
    title = (
        "Prefill | chunk 64/64 · 1,024 tokens · L0–L2"
        if phase == "prefill"
        else "Extend | 128 candidate tokens · L0–L2"
    )
    fig.suptitle(title, y=0.985, fontsize=13)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.957),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=1.35)
    for suffix in ("svg", "png"):
        fig.savefig(
            output / f"timeline_{phase}.{suffix}",
            dpi=180,
            facecolor="white",
            metadata={"Date": None} if suffix == "svg" else None,
        )
    plt.close(fig)
    return [0.0, limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prefill", "extend"), default="prefill")
    parser.add_argument("--engine-report", type=Path, default=EXPERIMENT / "report/sglang_engine")
    parser.add_argument("--mfu-report", type=Path, default=MFU / "report/full_extend_graph")
    parser.add_argument(
        "--mfu-rows",
        type=Path,
        help="Default: the window_rows.json bound by the current MFU publication",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(EXPERIMENT / "output"):
        parser.error("--output-dir must be inside this experiment's output directory")
    official, official_paths = engine_panels(args.engine_report, args.phase)
    local, local_paths = mfu_panels(args.mfu_report, args.mfu_rows, args.phase)
    if (
        read(args.mfu_report / "publication_manifest.json").get("schema")
        == "isolated-method-artifacts-v1"
        and args.phase == "extend"
    ):
        raise ValueError("Engine extend compares A128; use compare_decode_timeline for isolated A1")
    panels = official + local
    paths = [
        *official_paths,
        *local_paths,
        Path(__file__),
        Path(engine.__file__),
        Path(compact.__file__),
        Path(timeline.__file__),
    ]
    before = {str(path.resolve()): engine.sha256(path) for path in paths}
    output.mkdir(parents=True, exist_ok=False)
    axis = draw(panels, output, args.phase)
    engine.write_csv(
        output / "windows.csv",
        [
            {
                "source": panel["source"],
                "method": panel["method"],
                "profile_run_id": panel.get("profile_run_id"),
                **{
                    key: panel["window"][key]
                    for key in ("start_ns", "end_ns", "window_ms", "gpu_idle_ms")
                },
            }
            for panel in panels
        ],
    )
    (output / "panels.json").write_text(json.dumps(panels, indent=2) + "\n")
    for path in paths[-4:]:
        shutil.copyfile(path, output / path.name)
    if before != {str(path.resolve()): engine.sha256(path) for path in paths}:
        raise ValueError("An input or renderer changed during rendering")
    provenance = {
        "schema": "engine-local-mfu-timeline-comparison-v2",
        "phase": args.phase,
        "run_id": output.name,
        "measurement": "Presentation of existing NSYS profiles; no new GPU execution or benchmark.",
        "inputs_and_sources_sha256": before,
        "mfu_profile_run_ids": {panel["method"]: panel["profile_run_id"] for panel in local},
        **(
            {"mfu_profile_run_id": local[0]["profile_run_id"]}
            if len({panel["profile_run_id"] for panel in local}) == 1
            else {}
        ),
        "mfu_panel_provenance": {
            panel["method"]: panel["provenance"] for panel in local if "provenance" in panel
        },
        "engine_profile_run_ids": [
            "first3_engine_warm_hbm_profile_20261006_02",
            "first3_engine_warm_echo_profile_20261006_01",
        ],
        "boundary": (
            "Last prefill chunk (64/64), positions 64512-65535; first L0 compute start to last L2 compute end."
            if args.phase == "prefill"
            else "Complete 128-token extend, positions 65536-65663; first L0 compute start to last L2 compute end, including earlier L0 history H2D for local dense when present."
        ),
        "panel_boundaries": [
            {
                "source": panel["source"],
                "method": panel["method"],
                "start_basis": panel["window"].get("start_basis", "L0 first compute"),
                "l0_compute_start_ns": panel["layers"][0].get(
                    "compute_start_ns", panel["layers"][0]["start_ns"]
                ),
                "l0_history_h2d_start_ns": panel["window"].get("l0_history_h2d_start_ns"),
            }
            for panel in panels
        ],
        "shared_xlim_ms": axis,
        "footer": None,
        "display_policy": {
            "ECHO recall (IO unknown)": {
                "lane": "H2D",
                "label": "H2D",
                "color": engine.COLORS["H2D"],
                "hatch": None,
                "basis": "Kernel function; original category, IO evidence and timestamps are unchanged.",
            }
        },
        "audit": {
            "published_input_hashes": True,
            "published_window_and_idle_durations": True,
            "mfu_compute_category_counts": True,
            "mfu_exact_idle_intervals": True,
        },
        "limitations": [
            "Profile windows include profiler overhead and are not standalone benchmark latencies.",
            "Official cross-implementation numerical acceptance remains false.",
            "Official and local execution, precision and cache semantics retain their published differences.",
            "Only captured GPU idle is red; omitted control activities still count as busy.",
            "Official recall is displayed in H2D by kernel function. Fused/recall paths may transfer zero records; mapped-host bytes remain unknown.",
        ],
        "artifacts_sha256": {
            path.name: engine.sha256(path) for path in sorted(output.iterdir()) if path.is_file()
        },
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
