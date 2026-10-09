"""Confirm the accepted A1/B1/B2/A2 process sequence from raw CPU evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sqlite3
import statistics
from itertools import pairwise
from pathlib import Path

from evaluation.validation import require_receipt
from experiments.deepseek_v32_mfu.src.analyze_event_boundary import signature
from experiments.deepseek_v32_motivation.src.graph_attribution import original_node, read_lineage

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments/deepseek_v32_mfu/output/data"
ORDER = (
    ("A1", "matched", "01"),
    ("B1", "prelude", "01"),
    ("B2", "prelude", "02"),
    ("A2", "matched", "02"),
)
LAYERS = ("layer_0", "layer_1", "layer_2")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def intervals(rows):
    merged = []
    for left, right in sorted((row["start"], row["end"]) for row in rows):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    busy = sum(right - left for left, right in merged)
    gaps = [second[0] - first[1] for first, second in pairwise(merged)]
    return busy, gaps


def clean_summary(rows):
    require(len(rows) == 100, "Expected 50 complete pairs per clean process")
    require(
        [(row["pair"], row["arm"], row["order"]) for row in rows]
        == [
            (pair, arm, "AB" if pair % 2 == 0 else "BA")
            for pair in range(50)
            for arm in (("plain", "timed") if pair % 2 == 0 else ("timed", "plain"))
        ],
        "Clean sample sequence differs",
    )
    require(
        all(math.isfinite(row["wall_ms"]) and row["wall_ms"] > 0 for row in rows),
        "Invalid clean sample",
    )
    values = {(row["pair"], row["arm"]): row["wall_ms"] for row in rows}
    deltas = [values[pair, "timed"] - values[pair, "plain"] for pair in range(50)]
    return {
        "wall_ms": {
            "median": {
                arm: statistics.median(values[pair, arm] for pair in range(50))
                for arm in ("plain", "timed")
            },
            "paired_delta_timed_minus_plain": deltas,
            "paired_median_delta": statistics.median(deltas),
            "timed_faster_pairs": sum(value < 0 for value in deltas),
            "order_median_delta": {
                "AB": statistics.median(deltas[::2]),
                "BA": statistics.median(deltas[1::2]),
            },
        }
    }


def audit_rows(audit, profile, directory):
    parents = read_lineage([directory / "capture.1.sqlite"])
    template = profile["template"]
    connection = sqlite3.connect((directory / "capture.2.sqlite").as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    windows = []
    try:
        strings = dict(connection.execute("SELECT id,value FROM StringIds"))
        for scope in audit["scopes"]:
            require(scope["process"] == parents.process, "Profile process differs from lineage")
            nodes = []
            for kind in ("kernel", "memcpy", "memset"):
                for value in connection.execute(
                    f"SELECT * FROM CUPTI_ACTIVITY_KIND_{kind.upper()} WHERE globalPid=? AND correlationId=?",
                    (parents.process, scope["correlation"]),
                ):
                    row = dict(value)
                    row["kind"] = kind
                    row["name"] = (
                        strings[row["demangledName"]]
                        if kind == "kernel"
                        else "Device-to-Device"
                        if kind == "memcpy" and row["copyKind"] == 8
                        else kind
                    )
                    row["capture_node"] = original_node(row["graphNodeId"], parents)
                    row["owner"] = template["node_owners"][str(row["capture_node"])]
                    if kind == "kernel" or "graphId" in row:
                        require(row["graphId"] == template["executable_graph_id"], "Wrong graph")
                    nodes.append(row)
            nodes.sort(key=lambda row: row["capture_node"])
            require(nodes == scope["nodes"] and len(nodes) == 197, "Raw full-graph rows differ")
            require(
                {row["capture_node"] for row in nodes} == set(template["gpu_node_ids"]),
                "Native node membership differs",
            )
            selected = [row for row in nodes if row["owner"]["layer"] in LAYERS]
            require(len(selected) == 192, "Wrong layer ownership")
            start, end = min(row["start"] for row in selected), max(row["end"] for row in selected)
            busy, gaps = intervals(selected)
            stats = scope["statistics"]
            require(
                (start, end, busy, end - start - busy, statistics.median(gaps), max(gaps))
                == (
                    stats["start_ns"],
                    stats["end_ns"],
                    round(stats["busy_us"] * 1000),
                    round(stats["idle_us"] * 1000),
                    stats["gap_median_ns"],
                    stats["gap_max_ns"],
                ),
                "Raw integer layer statistics differ",
            )
            raw = []
            for kind in ("KERNEL", "MEMCPY", "MEMSET"):
                raw.extend(
                    dict(row)
                    for row in connection.execute(
                        f"SELECT * FROM CUPTI_ACTIVITY_KIND_{kind} WHERE end>? AND start<?",
                        (start, end),
                    )
                )
            require(
                {row["globalPid"] for row in raw} == {parents.process}, "Foreign process activity"
            )
            require({row["deviceId"] for row in raw} == {0}, "Foreign device activity")
            require(
                {row["graphNodeId"] for row in raw} <= {row["graphNodeId"] for row in nodes},
                "Foreign graph activity",
            )
            clipped = [
                {**row, "start": max(start, row["start"]), "end": min(end, row["end"])}
                for row in raw
            ]
            require(intervals(clipped)[0] == busy, "All-process clipped union differs")
            windows.append(
                {
                    "label": scope["label"],
                    "arm": scope["arm"],
                    "phase": scope["phase"],
                    "start_ns": start,
                    "end_ns": end,
                    "span_us": (end - start) / 1000,
                    "busy_us": busy / 1000,
                    "idle_us": (end - start - busy) / 1000,
                    "gap_median_ns": statistics.median(gaps),
                    "gap_max_ns": max(gaps),
                    "intersecting_rows": len(raw),
                    "foreign_process_device_graph_rows": 0,
                    "full_graph_nodes": len(nodes),
                    "layer_nodes": len(selected),
                    "signature_and_owner_sha256": canonical(
                        [{"signature": signature(row), "owner": row["owner"]} for row in nodes]
                    ),
                }
            )
    finally:
        connection.close()
    require(len(windows) == 6, "Expected six profile scopes")
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "Output already exists")
    inputs, sources, checks, cases, samples = {}, {}, {}, [], []
    arm_identities = {}
    for label, treatment, suffix in ORDER:
        paths = {
            kind: DATA / f"q1_replay_{treatment}_{kind}_20261008_{suffix}"
            for kind in ("audit", "bench", "profile")
        }
        audit, bench, profile = (read(paths[kind] / "result.json") for kind in paths)
        require(
            all(item["accepted"] is True for item in (audit, bench, profile)), "Unaccepted input"
        )
        require(bench["mode"] == "bench" and profile["mode"] == "profile", "Wrong run modes")
        require(
            bench["identity"] == profile["identity"] and bench["receipt"] == profile["receipt"],
            "Bench/profile identity or check differs",
        )
        identity = profile["identity"]
        if treatment in arm_identities:
            require(identity == arm_identities[treatment], "Within-treatment identity changed")
        arm_identities[treatment] = identity
        for name, expected in audit["inputs_sha256"].items():
            require(digest(name) == expected, "Accepted audit input changed")
            inputs[name] = expected
        for name, expected in audit["analysis_sources_sha256"].items():
            require(digest(name) == expected, "Accepted analyzer source changed")
            sources[name] = expected
        kind = (
            "deepseek-q1-replay-matched-v1"
            if treatment == "matched"
            else "deepseek-q1-replay-prelude-v1"
        )
        receipt = profile["receipt"]
        require(digest(receipt["path"]) == receipt["sha256"], "Check receipt changed")
        require_receipt(receipt["path"], kind=kind, identity=identity)
        checks[receipt["path"]] = {"sha256": receipt["sha256"], "kind": kind}
        for category in ("bench", "profile"):
            directory = paths[category]
            for relative, expected in identity["source"]["sources"].items():
                require(
                    digest(directory / "source" / relative) == expected, "Source archive changed"
                )
            for relative, expected in identity["runtime"]["formal_match"][
                "archived_artifacts"
            ].items():
                require(digest(directory / relative) == expected, "Runtime archive changed")
        summary = clean_summary(bench["result"]["samples"])
        require(
            summary == bench["result"]["summary"] == audit["clean_benchmark"]["summary"],
            "Clean sample summary differs",
        )
        samples.extend(
            {"process_label": label, "run_id": bench["run_id"], **row}
            for row in bench["result"]["samples"]
        )
        windows = audit_rows(audit, profile, paths["profile"])
        files = [paths[kind] / "result.json" for kind in paths]
        files.extend(paths["profile"] / name for name in ("capture.1.sqlite", "capture.2.sqlite"))
        inputs.update({str(path): digest(path) for path in files})
        cases.append(
            {
                "process_label": label,
                "treatment": treatment,
                "audit_run_id": paths["audit"].name,
                "check": receipt,
                "execution_identity_sha256": canonical(identity),
                "bench_result_mtime_ns": (paths["bench"] / "result.json").stat().st_mtime_ns,
                "profile_result_mtime_ns": (paths["profile"] / "result.json").stat().st_mtime_ns,
                "clean_summary": summary,
                "profile_windows": windows,
            }
        )
    for field in ("bench_result_mtime_ns", "profile_result_mtime_ns"):
        stamps = [row[field] for row in cases]
        require(
            all(left < right for left, right in pairwise(stamps)),
            "Result completion timestamps do not corroborate ABBA order",
        )
    require(
        len(
            {
                window["signature_and_owner_sha256"]
                for case in cases
                for window in case["profile_windows"]
            }
        )
        == 1,
        "Cross-process GPU signatures differ",
    )
    sources[str(Path(__file__).resolve())] = digest(__file__)
    output = {
        "accepted": True,
        "run_id": args.output_dir.name,
        "inputs_sha256": inputs,
        "analysis_sources_sha256": sources,
        "checks": checks,
        "process_order_per_metric": [row[0] for row in ORDER],
        "cases": cases,
        "clean_sample_count": len(samples),
        "boundary": "Root executed fresh processes in A1/B1/B2/A2 order separately for clean "
        "benchmarks and profiles. Saved result mtimes corroborate completion order, "
        "not exact process start times. Two independent processes per treatment per "
        "metric; within-process event pairs are not additional process replicates. "
        "All samples retained. This confirms the preparation-package trigger under "
        "recorded source/runtime conditions, not an allocator/stream/hardware mechanism. "
        "DeepGEMM per-launch binaries and retained CuTe MLIR remain outside the formal ledger.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output_dir / Path(__file__).name)
    with (args.output_dir / "clean_samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    (args.output_dir / "result.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print(args.output_dir / "result.json")


if __name__ == "__main__":
    main()
