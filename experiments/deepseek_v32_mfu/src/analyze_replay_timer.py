"""Audit one-graph replay event timing, raw NSYS settings and native GPU activity."""

from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path

from evaluation.validation import require_receipt
from experiments.deepseek_v32_mfu.src.analyze_capture_boundary import (
    PROCESS_MASK,
    digest,
    summary,
)
from experiments.deepseek_v32_mfu.src.analyze_event_boundary import signature
from experiments.deepseek_v32_mfu.src.analyze_replay_boundary import (
    activity_rows,
    database,
    validate_formal,
    validate_graph_id,
)
from experiments.deepseek_v32_motivation.src.graph_attribution import (
    original_node,
    read_lineage,
)

LABEL = "q1_replay_timer|"
LAYERS = ("layer_0", "layer_1", "layer_2")


def metadata(connection):
    # Capture files include environment values; only export relevant profiler settings.
    rows = connection.execute(
        "SELECT name,value FROM META_DATA_CAPTURE WHERE "
        "name IN ('CAPTURE_EVENT_TYPE','RATE_HZ','SHOW_BACKTRACE',"
        "'COLLECT_THREAD_STATE_TRACE','COLLECT_GPU_CTX_SW_TRACE','USE_LINUX_PERF',"
        "'COLLECT_GPU_MEMORY_USAGE') OR name LIKE 'CUDA_%' OR name LIKE 'OS_RUNTIME_%'"
    ).fetchall()
    values = collections.defaultdict(list)
    for name, value in rows:
        values[name].append(value)
    return {key: sorted(value) for key, value in sorted(values.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--formal-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    profile = args.profile_dir.resolve()
    result = json.loads((profile / "result.json").read_text())
    assert result["mode"] == "profile" and result["accepted"] is True
    assert result["run_id"] == profile.name
    assert digest(result["receipt"]["path"]) == result["receipt"]["sha256"]
    require_receipt(
        result["receipt"]["path"], kind="deepseek-q1-replay-timer-v1", identity=result["identity"]
    )
    for relative, expected in result["identity"]["source"]["sources"].items():
        assert digest(profile / "source" / relative) == expected, relative
    assert digest(profile / "request.json") == result["identity"]["source"]["request"]["sha256"]
    template = result["template"]
    assert dict(collections.Counter(template["node_types"].values())) == {0: 181, 1: 15, 2: 1}
    assert result["identity"]["node_type_counts"] == {"0": 181, "1": 15, "2": 1}
    assert len(template["gpu_node_ids"]) == len(set(template["gpu_node_ids"])) == 197
    assert template["runtime"]["replays"] == 0 and result["memory"]["graph"]["replays"] == 5
    assert [(row["pair"], row["arm"]) for row in result["result"]["samples"]] == [
        (0, "plain"),
        (0, "timed"),
        (1, "timed"),
        (1, "plain"),
    ]
    formal = validate_formal(args.formal_reference)
    formal_path = next(
        Path(path) for path in formal["input_sha256"] if path.endswith("capture_4.sqlite")
    )
    connection = database(formal_path.resolve())
    formal_metadata = metadata(connection)
    connection.close()
    assert formal_metadata["CAPTURE_EVENT_TYPE"] == ["Cuda", "NvtxEvents"]
    assert formal_metadata["RATE_HZ"] == ["0"]
    assert formal_metadata["CUDA_GRAPH_TRACE_OPTIONS:MODE"] == ["Node"]
    assert formal_metadata["COLLECT_THREAD_STATE_TRACE"] == ["false"]
    paths = [profile / f"capture.{index}.sqlite" for index in (1, 2)]
    parents = read_lineage(paths[:1])
    coverage, runs = {}, []
    expected_labels = {
        1: [LABEL + "construct=shared"],
        2: [
            LABEL + "warmup=plain",
            LABEL + "warmup=timed",
            LABEL + "pair=0|arm=plain",
            LABEL + "pair=0|arm=timed",
            LABEL + "pair=1|arm=timed",
            LABEL + "pair=1|arm=plain",
        ],
    }
    for index, path in enumerate(paths, 1):
        connection = database(path)
        strings = dict(connection.execute("SELECT id,value FROM StringIds"))
        observed_metadata = metadata(connection)
        assert observed_metadata == formal_metadata, {
            key: (formal_metadata.get(key), observed_metadata.get(key))
            for key in formal_metadata.keys() | observed_metadata.keys()
            if formal_metadata.get(key) != observed_metadata.get(key)
        }
        scopes = connection.execute(
            "SELECT * FROM NVTX_EVENTS WHERE text LIKE ? ORDER BY start", (LABEL + "%",)
        ).fetchall()
        assert [row["text"] for row in scopes] == expected_labels[index]
        counts = {
            strings[name]: count
            for name, count in connection.execute(
                "SELECT nameId,count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME GROUP BY nameId"
            )
        }
        assert sum(count for name, count in counts.items() if "cudaGraphLaunch" in name) == (
            0 if index == 1 else 6
        )
        assert sum(count for name, count in counts.items() if "cudaStreamBeginCapture" in name) == (
            1 if index == 1 else 0
        )
        assert sum(count for name, count in counts.items() if "cudaStreamEndCapture" in name) == (
            1 if index == 1 else 0
        )
        coverage[str(index)] = {
            "api_counts": counts,
            "labels": expected_labels[index],
            "relevant_capture_metadata": observed_metadata,
            "relevant_capture_metadata_exactly_matches_formal": True,
        }
        if index == 1:
            connection.close()
            continue
        for scope in scopes:
            fields = dict(part.split("=", 1) for part in scope["text"].split("|")[1:])
            arm = fields.get("arm", fields.get("warmup"))
            process = scope["globalTid"] & PROCESS_MASK
            assert process == parents.process
            apis = connection.execute(
                "SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND end<=? AND globalTid=?",
                (scope["start"], scope["end"], scope["globalTid"]),
            ).fetchall()
            launches = [row for row in apis if "cudaGraphLaunch" in strings[row["nameId"]]]
            records = [row for row in apis if "cudaEventRecord" in strings[row["nameId"]]]
            assert len(launches) == 1 and len(records) == (2 if arm == "timed" else 0)
            launch = launches[0]
            if records:
                records.sort(key=lambda row: row["start"])
                assert records[0]["end"] <= launch["start"] < launch["end"] <= records[1]["start"]
            nodes = activity_rows(
                connection, strings, correlation=launch["correlationId"], process=process
            )
            for row in nodes:
                validate_graph_id(row, template["executable_graph_id"])
                row["capture_node"] = original_node(row["graphNodeId"], parents)
                row["owner"] = template["node_owners"][str(row["capture_node"])]
                assert scope["start"] <= row["start"] < row["end"] <= scope["end"]
            nodes.sort(key=lambda row: row["capture_node"])
            assert len(nodes) == len({row["capture_node"] for row in nodes}) == 197
            assert {row["capture_node"] for row in nodes} == set(template["gpu_node_ids"])
            assert [signature(row) for row in nodes] == formal["signatures"]
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
            runs.append(
                {
                    "label": scope["text"],
                    "arm": arm,
                    "phase": "warmup" if "warmup" in fields else "formal",
                    "correlation": launch["correlationId"],
                    "process": process,
                    "executable_graph_id": template["executable_graph_id"],
                    "event_record_apis": [
                        {**dict(row), "name": strings[row["nameId"]]} for row in records
                    ],
                    "graph_launch_api": {**dict(launch), "name": strings[launch["nameId"]]},
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
                }
            )
        connection.close()
    assert len(runs) == 6
    assert all(
        [row["graphNodeId"] for row in run["nodes"]]
        == [row["graphNodeId"] for row in runs[0]["nodes"]]
        for run in runs
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    helpers = [
        Path(__file__),
        Path(__file__).with_name("analyze_replay_boundary.py"),
        Path(__file__).with_name("analyze_event_boundary.py"),
        Path(__file__).with_name("analyze_capture_boundary.py"),
        Path("evaluation/validation.py"),
        Path("experiments/deepseek_v32_motivation/src/graph_attribution.py"),
    ]
    for path in helpers:
        shutil.copy2(path, args.output_dir / path.name)
    shutil.copy2(args.formal_reference, args.output_dir / "formal_reference.json")
    output = {
        "inputs_sha256": {
            str(path): digest(path)
            for path in (*paths, profile / "result.json", args.formal_reference)
        },
        "analysis_sources_sha256": {str(path): digest(path) for path in helpers},
        "execution_receipt_and_archived_sources_verified": True,
        "formal_reference_raw_rows_and_native_ids_verified": True,
        "same_graph_native_nodes_and_executable_id": True,
        "coverage": coverage,
        "clone_edges": len(parents),
        "scopes": runs,
        "boundary": "One model/cache/native graph, dynamic plain/timed replay scope. Raw runtime APIs require zero/two outside cudaEventRecord calls bracketing exactly one graph launch. All six warmup/formal calls share the exact executable graph/native197GPU nodes; signatures/all17kernelfields/copybytes/nativeowners match current formal. Relevant raw capture metadata exactly matches formal cuda+nvtx/node/no sampling/contextswitch, with no OSRuntime capture. All same-process activity clipped unions equal the192layer-node unions. Clean wall timing is separate; this is not a hardware mechanism proof.",
    }
    (args.output_dir / "result.json").write_text(json.dumps(output, indent=2) + "\n")
    for run in runs:
        print(
            run["label"],
            {
                key: run["statistics"][key]
                for key in ("span_us", "busy_us", "idle_us", "gap_median_ns", "gap_max_ns")
            },
            "event_records",
            len(run["event_record_apis"]),
        )


if __name__ == "__main__":
    main()
