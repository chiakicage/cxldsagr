"""Redraw verified official decode activities without executing a model or changing data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiments.deepseek_v32_echo_official.src import compare_decode_timeline as comparison
from experiments.deepseek_v32_echo_official.src import decode_display, decode_timeline

EXPERIMENT = Path(__file__).resolve().parents[1]
DATA_FILES = ("activities.json", "report.json", "timing_samples.csv", "windows.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, default=EXPERIMENT / "report/sglang_decode")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source_report.resolve(), args.output_dir.resolve()
    if not output.is_relative_to(EXPERIMENT / "output/data"):
        parser.error("--output-dir must be inside this experiment's output/data")
    if output.exists():
        raise FileExistsError("Use a new display run ID")
    publication = comparison.comparison.read(source / "publication.json")
    paths = [source / "publication.json"]
    for name, expected in publication["files"].items():
        if Path(name).name != name:
            raise ValueError("Expected publication files directly below the report directory")
        comparison.comparison.require_hash(source / name, expected)
        paths.append(source / name)
    panels, _ = comparison.official_panels(source)
    sources = [
        Path(__file__),
        Path(decode_display.__file__),
        Path(decode_timeline.__file__),
        Path(comparison.__file__),
        Path(comparison.comparison.__file__),
        Path(comparison.engine.__file__),
    ]
    before = {str(path): comparison.engine.sha256(path) for path in (*paths, *sources)}
    output.mkdir(parents=True)
    archived = output / "input_report"
    archived.mkdir()
    for path in paths:
        shutil.copyfile(path, archived / path.name)
        comparison.comparison.require_hash(archived / path.name, before[str(path)])
    for name in DATA_FILES:
        shutil.copyfile(source / name, output / name)
    decode_timeline.plot(
        {panel["method"]: (panel["window"], panel["rows"]) for panel in panels},
        output / "timeline_decode.svg",
    )
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshot / path.name)
    if before != {str(path): comparison.engine.sha256(path) for path in (*paths, *sources)}:
        raise ValueError("Published input or renderer changed during redraw")
    for name in DATA_FILES:
        comparison.comparison.require_hash(output / name, publication["files"][name])
    recall = [
        {
            "case": panel["method"],
            **{
                key: row[key]
                for key in (
                    "name",
                    "layer",
                    "stage",
                    "kind",
                    "lane",
                    "category",
                    "potential_io",
                    "start_ns",
                    "end_ns",
                    "graph_node_id",
                    "io_evidence",
                )
            },
            "display_lane": "H2D",
            "measured_io_bytes": row.get("measured_io_bytes"),
        }
        for panel in panels
        for row in panel["rows"]
        if row["kind"] == "kernel" and row["category"] == decode_display.RECALL_CATEGORY
    ]
    updated = {
        **publication,
        "files": {name: comparison.engine.sha256(output / name) for name in publication["files"]},
        "display_update": {
            "schema": "official-decode-recall-display-v1",
            "run_id": output.name,
            "measurement": "Existing verified official NSYS activities; no new GPU execution or timing.",
            "original_publication": {
                "path": str(archived / "publication.json"),
                "sha256": comparison.engine.sha256(archived / "publication.json"),
            },
            "display_policy": decode_display.RECALL_DISPLAY,
            "recall_activities": recall,
            "source_snapshot_sha256": {
                path.name: comparison.engine.sha256(path) for path in snapshot.iterdir()
            },
            "unchanged_data_sha256": {name: publication["files"][name] for name in DATA_FILES},
            "audit": {
                "published_hashes": True,
                "window_and_idle_unions": True,
                "original_activity_rows": True,
                "numerical_acceptance": False,
            },
        },
    }
    (output / "publication.json").write_text(json.dumps(updated, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(output), "recall_kernels": len(recall), "data_unchanged": True}, indent=2
        )
    )


if __name__ == "__main__":
    main()
