"""Audit zero/two graph-event controls against both native ledgers and formal production."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

from evaluation.validation import require_receipt
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
    parser.add_argument("--formal-reference", type=Path, required=True)
    args = parser.parse_args()
    profile = args.profile_dir.resolve()
    result_path = profile / "result.json"
    result = json.loads(result_path.read_text())
    assert result["accepted"] is True and result["mode"] == "profile"
    assert result["run_id"] == profile.name
    receipt_path = Path(result["receipt"]["path"])
    assert digest(receipt_path) == result["receipt"]["sha256"]
    require_receipt(receipt_path, kind="deepseek-q1-event-boundary-v1", identity=result["identity"])
    for relative, expected_hash in result["identity"]["source"]["sources"].items():
        assert digest(profile / "source" / relative) == expected_hash
    assert digest(profile / "request.json") == result["identity"]["source"]["request"]["sha256"]
    templates = result["templates"]
    assert set(templates) == {"no_events", "events"}
    formal = json.loads(args.formal_reference.read_text())
    for path, expected_hash in formal["input_sha256"].items():
        assert digest(path) == expected_hash
    assert len(formal["nodes"]) == len(formal["signatures"]) == 197
    assert [signature(row) for row in formal["nodes"]] == formal["signatures"]
    assert formal["gpu_node_count"] == 197 and formal["non_gpu_node_count"] == 0
    for arm, template in templates.items():
        assert len(template["gpu_node_ids"]) == 197
        assert list(template["node_types"].values()).count(7) == (2 if arm == "events" else 0)
    paths = [profile / f"capture.{i}.sqlite" for i in (1, 2, 3)]
    parents = read_lineage(paths[:2])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for source in (Path(__file__), Path(__file__).with_name("analyze_capture_boundary.py")):
        shutil.copy2(source, args.output_dir / source.name)
    shutil.copy2(args.formal_reference, args.output_dir / "formal_reference.json")
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
            "SELECT text,count(*) FROM NVTX_EVENTS WHERE text LIKE 'q1_event_boundary|%' GROUP BY text"
        ).fetchall()
        coverage[str(index)] = {"api_counts": counts, "nvtx": labels}
    c = sqlite3.connect(paths[2].as_uri() + "?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    strings = dict(c.execute("SELECT id,value FROM StringIds"))
    scopes = c.execute(
        "SELECT * FROM NVTX_EVENTS WHERE text LIKE 'q1_event_boundary|pair=%' ORDER BY start"
    ).fetchall()
    expected_labels = [
        f"q1_event_boundary|pair={pair}|arm={arm}"
        for pair in range(2)
        for arm in (("no_events", "events") if pair == 0 else ("events", "no_events"))
    ]
    assert [scope["text"] for scope in scopes] == expected_labels
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
        template = templates[fields["arm"]]
        assert {row["capture_node"] for row in nodes} == set(template["gpu_node_ids"])
        assert all(
            row.get("graphId", template["executable_graph_id"]) == template["executable_graph_id"]
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
    no_events = next(run for run in runs if run["arm"] == "no_events")["nodes"]
    reference = [signature(row) for row in no_events]
    assert reference == formal["signatures"], (
        "Zero-event graph differs from current formal GPU signature"
    )
    for run in runs:
        assert [signature(row) for row in run["nodes"]] == reference, run["label"]
        template = templates[run["arm"]]
        for row in run["nodes"]:
            row["owner"] = template["node_owners"][str(row["capture_node"])]
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
        # Also test every same-process GPU row, including other launches.
        all_process_rows = []
        for kind in ("kernel", "memcpy", "memset"):
            table = "CUPTI_ACTIVITY_KIND_" + kind.upper()
            if table not in tables:
                continue
            for row in c.execute(
                f"SELECT * FROM {table} WHERE globalPid=? AND end>? AND start<?",
                (run["process"], stats["start_ns"], stats["end_ns"]),
            ):
                row = dict(row)
                row["kind"] = kind
                row["start"] = max(row["start"], stats["start_ns"])
                row["end"] = min(row["end"], stats["end_ns"])
                all_process_rows.append(row)
        all_process_stats = summary(all_process_rows)
        assert (stats["busy_us"], stats["idle_us"]) == (
            all_process_stats["busy_us"],
            all_process_stats["idle_us"],
        )
        run.update(
            {
                "statistics": stats,
                "same_process_intersecting_gpu_activity_count": len(all_process_rows),
                "other_launch_intersecting_activity_count": sum(
                    row["correlationId"] != run["correlation"] for row in all_process_rows
                ),
                "all_same_process_activity_clipped_union_matches_selected_union": True,
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
        "analysis_dependencies_sha256": {
            str(Path(__file__).with_name("analyze_capture_boundary.py")): digest(
                Path(__file__).with_name("analyze_capture_boundary.py")
            ),
            "experiments/deepseek_v32_motivation/src/graph_attribution.py": digest(
                Path(__file__).resolve().parents[3]
                / "experiments/deepseek_v32_motivation/src/graph_attribution.py"
            ),
        },
        "receipt_sha256": digest(receipt_path),
        "exact_ABBA_scope_order": expected_labels,
        "formal_reference_sha256": digest(args.formal_reference),
        "zero_event_graph_matches_current_formal_signature": True,
        "node_type_counts": result["identity"]["node_type_counts"],
        "coverage": coverage,
        "clone_edges": len(parents),
        "scopes": runs,
        "boundary": "Both arms are bound to their own native capture ID/ownership ledger through same-process clone lineage and exact executable graph ID. Node types differ by exactly two graph-internal type7 event records; GPU membership is197 in both. The zero-event arm exactly matches the current formal197-node GPU signature, including all17 kernel launch fields and D2D bytes, and the event arm matches it too. Native layer ownership identifies192 L0-L2 nodes. The clipped union of every intersecting GPU activity is verified separately. Outside ordinary CUDA events time both replays; they are not graph nodes. Node/launch equivalence is not a complete edge-equivalence proof. Separate actual no-NSYS replay measurements from NSYS recorded-gap effects.",
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
