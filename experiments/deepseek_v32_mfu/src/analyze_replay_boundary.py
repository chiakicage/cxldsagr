"""Audit first-replay collection controls, receipts, native ownership and GPU intervals."""

from __future__ import annotations

import argparse
import collections
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
from experiments.deepseek_v32_mfu.src.analyze_event_boundary import signature
from experiments.deepseek_v32_motivation.src.graph_attribution import (
    original_node,
    read_lineage,
)

ARMS = ("first5_outside", "first5_inside")
LABEL = "q1_replay_boundary|"
LAYERS = ("layer_0", "layer_1", "layer_2")
KIND = "deepseek-q1-replay-boundary-v1"


def read(path):
    return json.loads(path.read_text())


def validate_execution(profile):
    result = read(profile / "result.json")
    assert result["accepted"] is True and result["mode"] == "profile"
    assert result["run_id"] == profile.name
    receipt = Path(result["receipt"]["path"])
    assert digest(receipt) == result["receipt"]["sha256"]
    require_receipt(receipt, kind=KIND, identity=result["identity"])
    source = result["identity"]["source"]
    for relative, expected in source["sources"].items():
        assert digest(profile / "source" / relative) == expected, relative
    assert digest(profile / "request.json") == source["request"]["sha256"]
    assert set(result["templates"]) == set(ARMS)
    for arm, template in result["templates"].items():
        counts = dict(collections.Counter(template["node_types"].values()))
        assert counts == {0: 181, 1: 15, 2: 1}
        assert result["identity"]["node_type_counts"][arm] == {
            str(key): value for key, value in counts.items()
        }
        assert len(template["gpu_node_ids"]) == 197
        assert len(set(template["gpu_node_ids"])) == 197
        assert template["runtime"]["replays"] == 0
        assert result["memory"][arm]["graph"]["replays"] == 5
        assert result["lifecycle"][arm] == {
            "after_construction": 0,
            "after_first_warmups": 5,
        }
        assert [row["index"] for row in result["first_warmups"][arm]] == list(range(5))
        assert all(
            row["collection"] == (arm == "first5_inside") for row in result["first_warmups"][arm]
        )
    assert [(row["pair"], row["arm"]) for row in result["result"]["samples"]] == [
        (0, ARMS[0]),
        (0, ARMS[1]),
        (1, ARMS[1]),
        (1, ARMS[0]),
    ]
    return result


def validate_formal(path):
    formal = read(path)
    assert formal["gpu_node_count"] == 197 and formal["non_gpu_node_count"] == 0
    assert formal["layer_node_count"] == 192
    paths = {Path(name).name: Path(name) for name in formal["input_sha256"]}
    for name, expected in formal["input_sha256"].items():
        assert digest(name) == expected, name
    template = next(
        item for item in read(paths["full_graph_templates.json"]) if item["method"] == "hbm"
    )
    parents = read_lineage([paths["capture_3.sqlite"]])
    assert formal["process"] == parents.process
    assert {row["capture_node"] for row in formal["nodes"]} == set(template["gpu_node_ids"])
    assert set(template["node_types"].values()) == {0, 1, 2}
    assert len(formal["nodes"]) == 197
    connection = database(paths["capture_4.sqlite"].resolve())
    strings = dict(connection.execute("SELECT id,value FROM StringIds"))
    scopes = connection.execute(
        "SELECT * FROM NVTX_EVENTS WHERE text=?", (formal["scope"],)
    ).fetchall()
    assert len(scopes) == 1
    scope = scopes[0]
    assert scope["globalTid"] & PROCESS_MASK == parents.process
    apis = connection.execute(
        "SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND end<=? AND globalTid=?",
        (scope["start"], scope["end"], scope["globalTid"]),
    ).fetchall()
    launches = [row for row in apis if "cudaGraphLaunch" in strings[row["nameId"]]]
    assert len(launches) == 1 and launches[0]["correlationId"] == formal["correlation"]
    actual = activity_rows(
        connection, strings, correlation=formal["correlation"], process=parents.process
    )
    for row in actual:
        row["capture_node"] = original_node(row["graphNodeId"], parents)
        row["owner"] = template["node_owners"][str(row["capture_node"])]
    actual.sort(key=lambda row: row["capture_node"])
    assert actual == formal["nodes"]
    connection.close()
    for row in actual:
        assert row["globalPid"] == parents.process
        validate_graph_id(row, template["executable_graph_id"])
        assert original_node(row["graphNodeId"], parents) == row["capture_node"]
        assert row["owner"] == template["node_owners"][str(row["capture_node"])]
    assert [signature(row) for row in formal["nodes"]] == formal["signatures"]
    return formal


def validate_graph_id(row, expected):
    # NSYS memory tables can omit graphId; native node membership still binds them.
    if row["kind"] == "kernel" or "graphId" in row:
        assert row["graphId"] == expected


def database(path):
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def activity_rows(connection, strings, *, correlation=None, process, window=None):
    rows = []
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
    for kind in ("kernel", "memcpy", "memset"):
        table = "CUPTI_ACTIVITY_KIND_" + kind.upper()
        if table not in tables:
            continue
        conditions, values = ["globalPid=?"], [process]
        if correlation is not None:
            conditions.append("correlationId=?")
            values.append(correlation)
        if window is not None:
            conditions.extend(("end>?", "start<?"))
            values.extend(window)
        for value in connection.execute(
            f"SELECT * FROM {table} WHERE {' AND '.join(conditions)}", values
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
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--formal-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    profile = args.profile_dir.resolve()
    result = validate_execution(profile)
    formal = validate_formal(args.formal_reference)
    paths = [profile / f"capture.{index}.sqlite" for index in range(1, 5)]
    parents = read_lineage(paths[:2])
    coverage, runs = {}, []
    expected_labels = {
        1: [LABEL + "construct=first5_outside"],
        2: [LABEL + "construct=first5_inside"],
        3: [LABEL + f"first_warmup={index}|arm=first5_inside" for index in range(5)],
        4: [
            LABEL + "warmup=first5_outside",
            LABEL + "warmup=first5_inside",
            LABEL + "pair=0|arm=first5_outside",
            LABEL + "pair=0|arm=first5_inside",
            LABEL + "pair=1|arm=first5_inside",
            LABEL + "pair=1|arm=first5_outside",
        ],
    }
    for index, path in enumerate(paths, 1):
        connection = database(path)
        strings = dict(connection.execute("SELECT id,value FROM StringIds"))
        scopes = connection.execute(
            "SELECT * FROM NVTX_EVENTS WHERE text LIKE ? ORDER BY start", (LABEL + "%",)
        ).fetchall()
        assert [scope["text"] for scope in scopes] == expected_labels[index]
        counts = {
            strings[name]: count
            for name, count in connection.execute(
                "SELECT nameId,count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME GROUP BY nameId"
            )
        }
        launches = sum(count for name, count in counts.items() if "cudaGraphLaunch" in name)
        assert launches == {1: 0, 2: 0, 3: 5, 4: 6}[index]
        assert sum(count for name, count in counts.items() if "cudaStreamBeginCapture" in name) == (
            1 if index in (1, 2) else 0
        )
        assert sum(count for name, count in counts.items() if "cudaStreamEndCapture" in name) == (
            1 if index in (1, 2) else 0
        )
        coverage[str(index)] = {"api_counts": counts, "labels": expected_labels[index]}
        if index in (1, 2):
            connection.close()
            continue
        for scope in scopes:
            fields = dict(item.split("=", 1) for item in scope["text"].split("|")[1:])
            arm = fields.get("arm", fields.get("warmup"))
            template = result["templates"][arm]
            process = scope["globalTid"] & PROCESS_MASK
            assert process == parents.process
            apis = connection.execute(
                "SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND end<=? AND globalTid=?",
                (scope["start"], scope["end"], scope["globalTid"]),
            ).fetchall()
            launches = [row for row in apis if "cudaGraphLaunch" in strings[row["nameId"]]]
            assert len(launches) == 1
            launch = launches[0]
            nodes = activity_rows(
                connection, strings, correlation=launch["correlationId"], process=process
            )
            for row in nodes:
                assert row["graphNodeId"]
                validate_graph_id(row, template["executable_graph_id"])
                row["capture_node"] = original_node(row["graphNodeId"], parents)
                row["owner"] = template["node_owners"][str(row["capture_node"])]
                assert scope["start"] <= row["start"] < row["end"] <= scope["end"]
            nodes.sort(key=lambda row: row["capture_node"])
            assert len(nodes) == len({row["capture_node"] for row in nodes}) == 197
            assert {row["capture_node"] for row in nodes} == set(template["gpu_node_ids"])
            assert [signature(row) for row in nodes] == formal["signatures"], scope["text"]
            assert [row["owner"] for row in nodes] == [row["owner"] for row in formal["nodes"]]
            selected = [row for row in nodes if row["owner"]["layer"] in LAYERS]
            assert len(selected) == 192
            stats = summary(selected)
            all_rows = activity_rows(
                connection,
                strings,
                process=process,
                window=(stats["start_ns"], stats["end_ns"]),
            )
            clipped = [
                {
                    **row,
                    "start": max(row["start"], stats["start_ns"]),
                    "end": min(row["end"], stats["end_ns"]),
                }
                for row in all_rows
            ]
            all_stats = summary(clipped)
            assert (stats["busy_us"], stats["idle_us"]) == (
                all_stats["busy_us"],
                all_stats["idle_us"],
            )
            selected_ids = {row["graphNodeId"] for row in selected}
            graph_ids = {row["graphNodeId"] for row in nodes}
            phase = (
                "first_warmup"
                if "first_warmup" in fields
                else "trace_warmup"
                if "warmup" in fields
                else "formal"
            )
            runs.append(
                {
                    "label": scope["text"],
                    "phase": phase,
                    "arm": arm,
                    "interval": index,
                    "ordinal": int(fields["first_warmup"]) + 1
                    if phase == "first_warmup"
                    else 6
                    if phase == "trace_warmup"
                    else int(fields["pair"]) + 7,
                    "process": process,
                    "correlation": launch["correlationId"],
                    "executable_graph_id": template["executable_graph_id"],
                    "nodes": nodes,
                    "statistics": stats,
                    "all_process_intersecting_activity_count": len(all_rows),
                    "non_graph_intersecting_activities": [
                        row for row in all_rows if row["graphNodeId"] not in graph_ids
                    ],
                    "non_layer_intersecting_activities": [
                        row for row in all_rows if row["graphNodeId"] not in selected_ids
                    ],
                    "all_process_gpu_clipped_union_matches_selected_union": True,
                    "full_gpu_signature_and_ownership_match_formal": True,
                }
            )
        connection.close()
    assert len(runs) == 11
    args.output_dir.mkdir(parents=True, exist_ok=False)
    helper_paths = [
        Path(__file__),
        Path(__file__).with_name("analyze_event_boundary.py"),
        Path(__file__).with_name("analyze_capture_boundary.py"),
        Path("evaluation/validation.py"),
        Path("experiments/deepseek_v32_motivation/src/graph_attribution.py"),
    ]
    for source in helper_paths:
        shutil.copy2(source, args.output_dir / source.name)
    shutil.copy2(args.formal_reference, args.output_dir / "formal_reference.json")
    output = {
        "inputs_sha256": {
            str(path): digest(path)
            for path in (*paths, profile / "result.json", args.formal_reference)
        },
        "analysis_sources_sha256": {str(path): digest(path) for path in helper_paths},
        "execution_receipt_and_archived_sources_verified": True,
        "formal_reference_evidence_and_native_ids_verified": True,
        "zero_internal_event_nodes": True,
        "coverage": coverage,
        "clone_edges": len(parents),
        "scopes": runs,
        "boundary": "Same zero-event current production graph and FullExtendGraphCapture; only collection during first five full graph replays differs. Exact four-interval coverage verifies no construction graph launches, five inside-arm first replays and final two warmups/four ABBA replays. All197 GPU signatures/all17 launch fields/copy bytes and native ownership match current formal. Each replay uses its own native capture ledger, same-process clone lineage and exact executable graph ID. All same-process GPU activities intersecting the192-node layer window are clipped and their union verified. This is not full DAG-edge equivalence or a hardware scheduling mechanism proof; no-NSYS timing is separate.",
    }
    (args.output_dir / "result.json").write_text(json.dumps(output, indent=2) + "\n")
    for run in runs:
        print(
            run["label"],
            {
                key: run["statistics"][key]
                for key in ("span_us", "busy_us", "idle_us", "gap_median_ns", "gap_max_ns")
            },
        )


if __name__ == "__main__":
    main()
