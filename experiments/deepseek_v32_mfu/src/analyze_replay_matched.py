"""Audit the matched reduced timer with the FREE reference and its distinct receipt."""

from __future__ import annotations

import argparse
import collections
import json
import math
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

from evaluation.validation import require_receipt
from experiments.deepseek_v32_mfu.src import analyze_replay_timer as timer_audit
from experiments.deepseek_v32_mfu.src.analyze_replay_boundary import (
    activity_rows,
    database,
    validate_formal,
    validate_graph_id,
)
from experiments.deepseek_v32_mfu.src.q1_replay_matched import (
    EXPERIMENT,
    FORMAL_ID,
    KIND,
    ROOT,
    digest,
    match_runtime,
    read_environment,
    require,
)
from experiments.deepseek_v32_motivation.src.graph_attribution import (
    original_node,
    read_lineage,
)

WRAPPER = "experiments/deepseek_v32_mfu/src/q1_replay_matched.py"
LAYERS = ("layer_0", "layer_1", "layer_2")


def read(path):
    return json.loads(Path(path).read_text())


def cross_process_union(connection, strings, statistics_row):
    """Include every process/device row intersecting this recorded layer window."""
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
    processes = set()
    for kind in ("KERNEL", "MEMCPY", "MEMSET"):
        table = "CUPTI_ACTIVITY_KIND_" + kind
        if table in tables:
            processes.update(
                row[0]
                for row in connection.execute(
                    f"SELECT DISTINCT globalPid FROM {table} WHERE end>? AND start<?",
                    (statistics_row["start_ns"], statistics_row["end_ns"]),
                )
            )
    rows = []
    for process in sorted(processes):
        rows.extend(
            activity_rows(
                connection,
                strings,
                process=process,
                window=(statistics_row["start_ns"], statistics_row["end_ns"]),
            )
        )
    clipped = [
        {
            **row,
            "unclipped_start": row["start"],
            "unclipped_end": row["end"],
            "start": max(row["start"], statistics_row["start_ns"]),
            "end": min(row["end"], statistics_row["end_ns"]),
        }
        for row in rows
    ]
    observed = timer_audit.summary(clipped)
    require(
        all(observed[key] == statistics_row[key] for key in ("span_us", "busy_us", "idle_us")),
        "Cross-process GPU union differs",
    )
    return {
        "statistics": observed,
        "clipped_activities": clipped,
        "processes": sorted(processes),
        "matches_selected_layer_union": True,
    }


def build_formal_reference(formal_dir):
    require(formal_dir.name == FORMAL_ID, "Expected the current FREE profile directory")
    result = read(formal_dir / "result.json")
    require(result["accepted"] is True and result["run_id"] == FORMAL_ID, "Invalid formal run")
    paths = [
        formal_dir / name
        for name in (
            "capture_3.sqlite",
            "capture_4.sqlite",
            "full_graph_templates.json",
            "result.json",
        )
    ]
    template = next(row for row in read(paths[2]) if row["method"] == "hbm")
    require(
        dict(collections.Counter(template["node_types"].values())) == {0: 181, 1: 15, 2: 1},
        "Unexpected formal node types",
    )
    parents = read_lineage(paths[:1])
    connection = database(paths[1])
    try:
        strings = dict(connection.execute("SELECT id,value FROM StringIds"))
        scope_name = "echo/hbm/extend_annotated/shared/extend_graph_replay_q_1/call_3"
        scopes = connection.execute(
            "SELECT * FROM NVTX_EVENTS WHERE text=?", (scope_name,)
        ).fetchall()
        require(len(scopes) == 1, "Expected one formal measured replay scope")
        scope = scopes[0]
        require(
            scope["globalTid"] & timer_audit.PROCESS_MASK == parents.process,
            "Formal process/lineage differs",
        )
        apis = connection.execute(
            "SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND end<=? AND globalTid=?",
            (scope["start"], scope["end"], scope["globalTid"]),
        ).fetchall()
        launches = [row for row in apis if "cudaGraphLaunch" in strings[row["nameId"]]]
        require(len(launches) == 1, "Expected one formal graph launch")
        correlation = launches[0]["correlationId"]
        nodes = activity_rows(connection, strings, correlation=correlation, process=parents.process)
        for row in nodes:
            validate_graph_id(row, template["executable_graph_id"])
            row["capture_node"] = original_node(row["graphNodeId"], parents)
            row["owner"] = template["node_owners"][str(row["capture_node"])]
            # This formal NVTX range encloses replay submission, not completion.
            # Correlation and native lineage bind its asynchronous GPU activities.
        nodes.sort(key=lambda row: row["capture_node"])
        require(
            len(nodes) == len({row["capture_node"] for row in nodes}) == 197,
            "Unexpected formal GPU node count",
        )
        require(
            {row["capture_node"] for row in nodes} == set(template["gpu_node_ids"]),
            "Formal native node ledger differs",
        )
        layer = [row for row in nodes if row["owner"]["layer"] in LAYERS]
        require(len(layer) == 192, "Unexpected formal layer node count")
        stats = timer_audit.summary(layer)
        all_processes = cross_process_union(connection, strings, stats)
    finally:
        connection.close()
    return {
        "input_sha256": {str(path): digest(path) for path in paths},
        "nodes": nodes,
        "signatures": [timer_audit.signature(row) for row in nodes],
        "layer_statistics": stats,
        "gpu_node_count": 197,
        "layer_node_count": 192,
        "non_gpu_node_count": 0,
        "correlation": correlation,
        "process": parents.process,
        "scope": scope_name,
        "cross_process_union": all_processes,
        "boundary": "FREE formal HBM measured replay; original capture lineage, 197 GPU nodes, "
        "192 layer owners and clipped intersecting activity from every process.",
    }


def validate_execution(directory, mode, formal_dir):
    result = read(directory / "result.json")
    require(result["accepted"] is True and result["mode"] == mode, "Unexpected reduced run mode")
    require(result["run_id"] == directory.name, "Reduced run ID differs")
    identity = result["identity"]
    source = identity["source"]
    receipt = result["receipt"]
    require(digest(receipt["path"]) == receipt["sha256"], "Check receipt changed")
    require_receipt(receipt["path"], kind=KIND, identity=identity)
    require(source["contract"]["matched_environment"] == KIND, "Missing matched wrapper contract")
    require(source["sources"][WRAPPER] == digest(ROOT / WRAPPER), "Matched wrapper source changed")
    for relative, expected in source["sources"].items():
        require(digest(directory / "source" / relative) == expected, "Archived source differs")
    require(digest(directory / "request.json") == source["request"]["sha256"], "Request changed")
    formal = read(formal_dir / "result.json")
    for name, record in source["formal_reference"].items():
        expected_path = formal_dir / ("capture_4.sqlite" if name == "capture_environment" else name)
        require(Path(record["path"]) == expected_path, "Different formal reference path")
        require(digest(expected_path) == record["sha256"], "Formal reference changed")
    environment = source["formal_reference"]["capture_environment"]["selected_environment"]
    require(
        environment == read_environment(formal_dir / "capture_4.sqlite"),
        "Selected formal environment differs",
    )
    require(
        source["physical_device"] == 0 and source["affinity"] == list(range(8)),
        "Reduced hardware placement differs",
    )
    require(source["uuid"] == formal["hardware"]["torch_device_uuid"], "Reduced GPU UUID differs")
    require(source["request"]["sha256"] == formal["request_sha256"], "Formal input differs")
    require(
        all(
            source["sources"].get(name) == value for name, value in formal["source_sha256"].items()
        ),
        "Formal production sources differ",
    )
    missing = match_runtime(identity["runtime"], formal)
    recorded_match = identity["runtime"]["formal_match"]
    require(
        recorded_match["passed"] is True and recorded_match["unobserved_formal_entries"] == missing,
        "Recorded runtime matching result differs",
    )
    for relative, expected in recorded_match["archived_artifacts"].items():
        require(digest(directory / relative) == expected, "Archived runtime artifact differs")
    return result


def recompute_clean_summary(result):
    rows = result["result"]["samples"]
    require(len(rows) == 100, "Expected exactly 50 clean pairs")
    require(
        [(row["pair"], row["arm"], row["order"]) for row in rows]
        == [
            (pair, arm, "AB" if pair % 2 == 0 else "BA")
            for pair in range(50)
            for arm in (("plain", "timed") if pair % 2 == 0 else ("timed", "plain"))
        ],
        "Clean samples are not the assigned balanced AB/BA sequence",
    )
    require(
        all(math.isfinite(row["wall_ms"]) and row["wall_ms"] > 0 for row in rows),
        "Invalid clean wall sample",
    )
    lookup = {(row["pair"], row["arm"]): row["wall_ms"] for row in rows}
    deltas = [lookup[pair, "timed"] - lookup[pair, "plain"] for pair in range(50)]
    recomputed = {
        "wall_ms": {
            "median": {
                arm: statistics.median(lookup[pair, arm] for pair in range(50))
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
    require(recomputed == result["result"]["summary"], "Clean sample summary differs")
    return recomputed


def matched_receipt(path, *, kind, identity):
    """Adapt the old analyzer's call site, never the stored receipt or identity."""
    require(kind == "deepseek-q1-replay-timer-v1", "Unexpected delegated receipt call")
    require(
        identity["source"]["contract"]["matched_environment"] == KIND,
        "Only the matched wrapper identity is accepted",
    )
    return require_receipt(path, kind=KIND, identity=identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--bench-dir", type=Path, required=True)
    parser.add_argument(
        "--formal-profile", type=Path, default=EXPERIMENT / "output/data" / FORMAL_ID
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    formal_dir, profile, bench, output = (
        path.resolve()
        for path in (args.formal_profile, args.profile_dir, args.bench_dir, args.output_dir)
    )
    require(not output.exists(), "Analysis output already exists")
    profile_result = validate_execution(profile, "profile", formal_dir)
    bench_result = validate_execution(bench, "bench", formal_dir)
    require(
        profile_result["identity"] == bench_result["identity"], "Bench/profile identities differ"
    )
    require(profile_result["receipt"] == bench_result["receipt"], "Bench/profile checks differ")
    clean_summary = recompute_clean_summary(bench_result)
    reference = build_formal_reference(formal_dir)
    with tempfile.TemporaryDirectory(prefix="q1-replay-matched-analysis-") as temporary:
        temporary_reference = Path(temporary) / "formal_reference.json"
        temporary_reference.write_text(json.dumps(reference, indent=2) + "\n")
        validate_formal(temporary_reference)
        previous_arguments, previous_receipt = sys.argv, timer_audit.require_receipt
        try:
            timer_audit.require_receipt = matched_receipt
            sys.argv = [
                str(Path(timer_audit.__file__)),
                "--profile-dir",
                str(profile),
                "--formal-reference",
                str(temporary_reference),
                "--output-dir",
                str(output),
            ]
            timer_audit.main()
        finally:
            sys.argv, timer_audit.require_receipt = previous_arguments, previous_receipt
        result = read(output / "result.json")
        result["inputs_sha256"].pop(str(temporary_reference))
        result["inputs_sha256"][str(output / "formal_reference.json")] = digest(temporary_reference)
    connection = database(profile / "capture.2.sqlite")
    try:
        strings = dict(connection.execute("SELECT id,value FROM StringIds"))
        for scope in result["scopes"]:
            scope["cross_process_union"] = cross_process_union(
                connection, strings, scope["statistics"]
            )
    finally:
        connection.close()
    for path in (Path(__file__).resolve(), ROOT / WRAPPER):
        shutil.copy2(path, output / path.name)
        result["analysis_sources_sha256"][str(path)] = digest(path)
    result["inputs_sha256"][str(bench / "result.json")] = digest(bench / "result.json")
    result["clean_benchmark"] = {
        "run_id": bench.name,
        "pairs": 50,
        "summary": clean_summary,
        "independently_recomputed": True,
        "boundary": "Complete synchronized forward wall; restored prefix and archival outside timing.",
    }
    result["matched_control"] = {
        "receipt_kind": KIND,
        "formal_profile_run_id": FORMAL_ID,
        "wrapper_sha256": digest(ROOT / WRAPPER),
        "execution_identity_unchanged": True,
        "formal_reference_and_archived_runtime_verified": True,
        "receipt_adapter": "The delegated hard-coded timer call is checked and then validates ONLY "
        "the new matched receipt kind; no receipt fields or identities are modified.",
        "runtime_boundary": profile_result["identity"]["runtime"]["formal_match"]["boundary"],
    }
    result["accepted"] = True
    result["boundary"] += (
        " This entry additionally binds the FREE formal reference, requires the distinct matched "
        "receipt, independently recomputes 50 clean pairs, and verifies every-process clipped "
        "activity unions. Existing same-process field names are retained as original analyzer output."
    )
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(output / "result.json")


if __name__ == "__main__":
    main()
