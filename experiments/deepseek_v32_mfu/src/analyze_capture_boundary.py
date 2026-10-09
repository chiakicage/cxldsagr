"""Audit profiler-boundary A/B by unique matching to a saved 198-node template."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import sqlite3
import statistics
from pathlib import Path

PROCESS_MASK = 0xFFFFFFFFFF000000


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same(actual, expected):
    if actual["kind"] != expected["kind"] or actual["name"] != expected["name"]:
        return False
    if expected["kind"] == "kernel":
        return all(actual[key] == value for key, value in expected["launch_metadata"].items())
    return actual["bytes"] == expected["bytes"] and actual["copyKind"] == 8


def summary(nodes):
    start, end = min(r["start"] for r in nodes), max(r["end"] for r in nodes)
    cursor, gaps = start, []
    for row in sorted(nodes, key=lambda r: r["start"]):
        if row["start"] > cursor:
            gaps.append({"start_ns": cursor, "end_ns": row["start"], "ns": row["start"] - cursor})
        cursor = max(cursor, row["end"])
    idle = sum(row["ns"] for row in gaps)
    return {
        "start_ns": start,
        "end_ns": end,
        "span_us": (end - start) / 1000,
        "busy_us": (end - start - idle) / 1000,
        "idle_us": idle / 1000,
        "activities": len(nodes),
        "kinds": dict(collections.Counter(row["kind"] for row in nodes)),
        "summed_kernel_us": sum(r["end"] - r["start"] for r in nodes if r["kind"] == "kernel")
        / 1000,
        "gaps": gaps,
        "gap_count": len(gaps),
        "gap_median_ns": statistics.median(row["ns"] for row in gaps) if gaps else 0,
        "gap_max_ns": max((row["ns"] for row in gaps), default=0),
        "gap_histogram_ns": dict(collections.Counter(row["ns"] for row in gaps)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    required = reference["nodes"]
    assert len(required) == 198
    assert digest(reference["source_csv"]) == reference["source_csv_sha256"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.reference, args.output_dir / "reference.json")
    shutil.copy2(__file__, args.output_dir / "analyze_capture_boundary.py")
    c = sqlite3.connect(args.sqlite.as_uri() + "?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    strings = dict(c.execute("SELECT id,value FROM StringIds"))
    scopes = c.execute(
        "SELECT * FROM NVTX_EVENTS WHERE text LIKE 'q1_capture_boundary|pair=%' ORDER BY start"
    ).fetchall()
    assert len(scopes) == 4, len(scopes)
    tables = {row[0] for row in c.execute("SELECT name FROM sqlite_master")}
    results = []
    for scope in scopes:
        label = scope["text"]
        fields = dict(item.split("=", 1) for item in label.split("|")[1:])
        runtime = c.execute(
            "SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND end<=? AND globalTid=?",
            (scope["start"], scope["end"], scope["globalTid"]),
        ).fetchall()
        launches = [row for row in runtime if "cudaGraphLaunch" in strings[row["nameId"]]]
        assert len(launches) == 1, label
        launch = launches[0]
        pid = scope["globalTid"] & PROCESS_MASK
        nodes = []
        for kind in ("kernel", "memcpy", "memset"):
            table = "CUPTI_ACTIVITY_KIND_" + kind.upper()
            if table not in tables:
                continue
            for row in c.execute(
                f"SELECT * FROM {table} WHERE correlationId=? AND globalPid=?",
                (launch["correlationId"], pid),
            ):
                row = dict(row)
                assert row["graphNodeId"]
                row["kind"] = kind
                row["name"] = (
                    strings[row["demangledName"]]
                    if kind == "kernel"
                    else "Device-to-Device"
                    if kind == "memcpy" and row["copyKind"] == 8
                    else kind
                )
                nodes.append(row)
        nodes.sort(key=lambda r: r["graphNodeId"])
        assert len(nodes) == 203, (label, len(nodes))
        assert len({r["graphNodeId"] for r in nodes}) == 203
        candidates = [
            i
            for i in range(len(nodes) - len(required) + 1)
            if all(
                same(actual, expected)
                for actual, expected in zip(nodes[i : i + 198], required, strict=True)
            )
        ]
        assert len(candidates) == 1, (label, candidates)
        offset = candidates[0]
        selected = nodes[offset : offset + 198]
        statistics_ = summary(selected)
        assert all(scope["start"] <= row["start"] < row["end"] <= scope["end"] for row in nodes)
        intersecting = [
            row
            for row in nodes
            if row["end"] > statistics_["start_ns"] and row["start"] < statistics_["end_ns"]
        ]
        clipped = [
            {
                **row,
                "start": max(row["start"], statistics_["start_ns"]),
                "end": min(row["end"], statistics_["end_ns"]),
            }
            for row in intersecting
        ]
        # A shared final-norm kernel may overlap the final MLP kernel's tail.
        # Preserve that boundary crossing and prove it fills no selected gap.
        clipped_statistics = summary(clipped)
        assert clipped_statistics["busy_us"] == statistics_["busy_us"]
        assert clipped_statistics["idle_us"] == statistics_["idle_us"]
        for row, expected in zip(selected, required, strict=True):
            row["layer"] = expected["layer"]
            row["stage"] = expected["stage"]
        results.append(
            {
                "label": label,
                "pair": int(fields["pair"]),
                "arm": fields["arm"],
                "correlation": launch["correlationId"],
                "process": pid,
                "full_replay_activity_count": len(nodes),
                "matched_ordinal_start": offset,
                "same_198_node_signature": True,
                "all_intersecting_gpu_activities": len(intersecting),
                "boundary_crossing_activities": [
                    row for row in intersecting if row not in selected
                ],
                "all_activity_clipped_union_matches_selected_union": True,
                "statistics": statistics_,
                "nodes": selected,
            }
        )
    output = {
        "sqlite": str(args.sqlite),
        "sqlite_sha256": digest(args.sqlite),
        "reference_sha256": digest(args.reference),
        "analysis_sha256": digest(__file__),
        "boundary": "Unique match of the entire saved 198-node L0-L2 signature, including kind/name and all 17 launch fields or D2D bytes, after excluding non-GPU event nodes. Complete replay has 203 GPU activities. A shared final-norm activity crosses the window end and is separately retained; the clipped union of all intersecting activities must equal the selected-node union. Scope identity does not establish equal graph edges; no graph inspector was installed. Busy is interval union and idle its complement.",
        "scopes": results,
    }
    (args.output_dir / "result.json").write_text(json.dumps(output, indent=2) + "\n")
    for result in results:
        print(
            result["label"],
            {
                key: result["statistics"][key]
                for key in (
                    "span_us",
                    "busy_us",
                    "idle_us",
                    "gap_count",
                    "gap_median_ns",
                    "gap_max_ns",
                )
            },
        )


if __name__ == "__main__":
    main()
