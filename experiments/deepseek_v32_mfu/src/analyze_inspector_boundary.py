"""Bind inspector-control replays to the native ledger and compare GPU idle."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

from experiments.deepseek_v32_mfu.src.analyze_capture_boundary import (
    PROCESS_MASK,
    digest,
    summary,
)
from experiments.deepseek_v32_motivation.src.graph_attribution import (
    original_node,
    read_lineage,
)

FIELDS = (
    "launchType",
    "cacheConfig",
    "registersPerThread",
    "gridX",
    "gridY",
    "gridZ",
    "blockX",
    "blockY",
    "blockZ",
    "staticSharedMemory",
    "dynamicSharedMemory",
    "sharedMemoryExecuted",
    "sharedMemoryLimitConfig",
    "clusterX",
    "clusterY",
    "clusterZ",
    "clusterSchedulingPolicy",
)


def signature(row):
    result = {"kind": row["kind"], "name": row["name"]}
    if row["kind"] == "kernel":
        result.update({key: row[key] for key in FIELDS})
    else:
        result["bytes"] = row["bytes"]
        if row["kind"] == "memcpy":
            result["copyKind"] = row["copyKind"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    profile = args.profile_dir.resolve()
    result_path = profile / "result.json"
    result = json.loads(result_path.read_text())
    template = result["templates"]["inspect"]
    expected_ids = set(template["gpu_node_ids"])
    assert len(expected_ids) == 197
    paths = [profile / f"capture.{i}.sqlite" for i in (1, 2, 3)]
    parents = read_lineage(paths[:2])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for source in (Path(__file__), Path(__file__).with_name("analyze_capture_boundary.py")):
        shutil.copy2(source, args.output_dir / source.name)
    coverage = {}
    for index, path in enumerate(paths, 1):
        c = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        strings = dict(c.execute("SELECT id,value FROM StringIds"))
        counts = {}
        for name, count in c.execute(
            "SELECT nameId,count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME GROUP BY nameId"
        ):
            name = strings[name]
            if any(
                token in name
                for token in ("Capture", "GraphGet", "GraphNode", "GraphLaunch", "GraphInstantiate")
            ):
                counts[name] = count
        labels = c.execute(
            "SELECT text,count(*) FROM NVTX_EVENTS WHERE text LIKE 'q1_inspector_boundary|%' GROUP BY text"
        ).fetchall()
        coverage[str(index)] = {"api_counts": counts, "nvtx": labels}
    c = sqlite3.connect(paths[2].as_uri() + "?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    strings = dict(c.execute("SELECT id,value FROM StringIds"))
    scopes = c.execute(
        "SELECT * FROM NVTX_EVENTS WHERE text LIKE 'q1_inspector_boundary|pair=%' ORDER BY start"
    ).fetchall()
    assert len(scopes) == 4
    tables = {row[0] for row in c.execute("SELECT name FROM sqlite_master")}
    runs = []
    for scope in scopes:
        fields = dict(item.split("=", 1) for item in scope["text"].split("|")[1:])
        pid = scope["globalTid"] & PROCESS_MASK
        assert pid == parents.process
        apis = c.execute(
            "SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND end<=? AND globalTid=?",
            (scope["start"], scope["end"], scope["globalTid"]),
        ).fetchall()
        launches = [row for row in apis if "cudaGraphLaunch" in strings[row["nameId"]]]
        assert len(launches) == 1
        launch = launches[0]
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
                row["capture_node"] = original_node(row["graphNodeId"], parents)
                row["kind"] = kind
                row["name"] = (
                    strings[row["demangledName"]]
                    if kind == "kernel"
                    else "Device-to-Device"
                    if kind == "memcpy" and row["copyKind"] == 8
                    else kind
                )
                assert scope["start"] <= row["start"] < row["end"] <= scope["end"]
                nodes.append(row)
        nodes.sort(key=lambda row: row["capture_node"])
        assert len(nodes) == len({row["capture_node"] for row in nodes}) == 197
        if fields["arm"] == "inspect":
            assert {row["capture_node"] for row in nodes} == expected_ids
            assert all(
                row.get("graphId", template["executable_graph_id"])
                == template["executable_graph_id"]
                for row in nodes
            )
        runs.append(
            {
                "label": scope["text"],
                "arm": fields["arm"],
                "pair": int(fields["pair"]),
                "correlation": launch["correlationId"],
                "process": pid,
                "nodes": nodes,
            }
        )
    inspected = next(run for run in runs if run["arm"] == "inspect")["nodes"]
    reference = [signature(row) for row in inspected]
    owners = [template["node_owners"][str(row["capture_node"])] for row in inspected]
    for run in runs:
        assert [signature(row) for row in run["nodes"]] == reference, run["label"]
        for row, owner in zip(run["nodes"], owners, strict=True):
            row["owner"] = owner
        selected = [
            row
            for row in run["nodes"]
            if row["owner"]["layer"] in ("layer_0", "layer_1", "layer_2")
        ]
        assert len(selected) == 192
        stats = summary(selected)
        intersecting = [
            row
            for row in run["nodes"]
            if row["end"] > stats["start_ns"] and row["start"] < stats["end_ns"]
        ]
        clipped = [
            {
                **row,
                "start": max(row["start"], stats["start_ns"]),
                "end": min(row["end"], stats["end_ns"]),
            }
            for row in intersecting
        ]
        all_stats = summary(clipped)
        assert stats["busy_us"] == all_stats["busy_us"] and stats["idle_us"] == all_stats["idle_us"]
        run.update(
            {
                "statistics": stats,
                "boundary_crossing_activities": [
                    row for row in intersecting if row not in selected
                ],
                "all_activity_clipped_union_matches_selected_union": True,
                "same_complete_197_node_signature": True,
            }
        )
    output = {
        "inputs_sha256": {str(path): digest(path) for path in (*paths, result_path)},
        "analysis_sha256": digest(__file__),
        "coverage": coverage,
        "clone_edges": len(parents),
        "scopes": runs,
        "boundary": "Inspect-arm capture IDs/owners are verified through same-process native lineage and exact executable graph ID. Both arms match the complete ordered 197-node GPU signature, including 17 kernel launch fields and D2D bytes. The native ledger's layer owners identify 192 L0-L2 nodes; the clipped union of every intersecting activity is verified separately. This establishes node/launch and scope identity, not equal graph edges. The treatment is the complete FullExtendGraphCapture scope, not one individual API.",
    }
    (args.output_dir / "result.json").write_text(json.dumps(output, indent=2) + "\n")
    for run in runs:
        print(
            run["label"],
            {
                key: run["statistics"][key]
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
