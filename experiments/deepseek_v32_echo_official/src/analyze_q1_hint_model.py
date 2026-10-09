"""Read-only audit of exact-hint model acceptance, clean pairs and raw graph lineage."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
from collections import Counter
from contextlib import closing
from pathlib import Path

from evaluation.validation import identity_digest, require_receipt
from experiments.deepseek_v32_mfu.src.analyze_capture_boundary import (
    PROCESS_MASK,
    digest,
    summary,
)
from experiments.deepseek_v32_mfu.src.analyze_event_boundary import FIELDS, signature
from experiments.deepseek_v32_mfu.src.analyze_nsys import _union
from experiments.deepseek_v32_mfu.src.analyze_replay_boundary import (
    activity_rows,
    database,
    validate_graph_id,
)
from experiments.deepseek_v32_motivation.src.graph_attribution import original_node, read_lineage

ROOT = Path(__file__).resolve().parents[3]
KIND = "deepseek-q1-hint-exact-private-model-v1"
ARMS = ("baseline", "candidate")
LAYERS = ("layer_0", "layer_1", "layer_2")
SOURCE_CLOSURE = (
    "experiments/deepseek_v32_echo_official/src/analyze_q1_hint_model.py",
    "evaluation/validation.py",
    "experiments/deepseek_v32_mfu/src/analyze_capture_boundary.py",
    "experiments/deepseek_v32_mfu/src/analyze_event_boundary.py",
    "experiments/deepseek_v32_mfu/src/analyze_replay_boundary.py",
    "experiments/deepseek_v32_mfu/src/analyze_nsys.py",
    "experiments/deepseek_v32_motivation/src/graph_attribution.py",
    "experiments/deepseek_v32_motivation/src/graph_instrumentation.py",
)
CAPTURE_KEYS = (
    "CAPTURE_EVENT_TYPE",
    "RATE_HZ",
    "CUDA_GRAPH_TRACE_OPTIONS:MODE",
    "CUDA_TRACE_SCOPE",
    "COLLECT_THREAD_STATE_TRACE",
    "COLLECT_GPU_CTX_SW_TRACE",
    "PROCESS_0:COMMAND",
    "PROCESS_0:WORKING_DIR",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def file_identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def audit_run(directory, mode, identity, receipt):
    value = read(directory / "result.json")
    require(
        value.get("completed") is True
        and value.get("mode") == mode
        and value.get("run_id") == directory.name
        and value.get("result", {}).get("passed") is True,
        "Incomplete or mislabeled execution: " + str(directory),
    )
    require(value["identity"] == identity, "Execution identity differs: " + mode)
    if mode != "check":
        require(
            value["result"]["receipt_sha256"] == receipt["receipt_sha256"],
            "Receipt binding differs",
        )
    for relative, expected in identity["source"]["sources"].items():
        require(
            digest(directory / "source" / relative) == expected,
            "Archived source differs: " + relative,
        )
    native = identity["timing_runtime"]["mapped_native"]
    require(native, "Missing mapped native inventory")
    require(
        len({entry["name"] for entry in native}) == len(native), "Duplicate native archive name"
    )
    for entry in native:
        path = directory / "native" / entry["name"]
        require(
            digest(path) == entry["library"]["sha256"]
            and path.stat().st_size == entry["library"]["bytes"],
            "Archived native differs: " + entry["name"],
        )
    for arm in ARMS:
        graph = value["memory"][arm]["graph"]
        require(
            graph["history_tokens"] == 65536
            and graph["query_tokens"] == graph["graph_count"] == 1
            and graph["cache_method"] == "echo"
            and graph["return_hidden"] is True,
            "Execution graph does not match H64K/A1 ECHO",
        )
    return value, {
        "mode": mode,
        "result": file_identity(directory / "result.json"),
        "source_files_verified": len(identity["source"]["sources"]),
        "native_files_verified": len(native),
    }


def audit_bindings(args):
    identity = read(args.receipt)["identity"]
    receipt = require_receipt(args.receipt, kind=KIND, identity=identity)
    checks = receipt["checks"]
    require(
        checks.get("tokens") == [111090, 111091, 111092]
        and checks.get("prepared_cap_both_arms") == 64
        and all(
            checks.get(key) is True
            for key in (
                "eager_graph_both_arms",
                "all_offsets_bitwise",
                "actual_stage_proofs",
                "clean_graph_checked",
                "actual_q1_to_a2_consumer",
            )
        ),
        "Full model receipt lacks required correctness coverage",
    )
    contract = identity["source"]["contract"]
    require(
        contract["history"] == 65536
        and contract["append"] == 1
        and contract["capacity"] == 65537
        and contract["slots"] == 65600
        and contract["layers"] == [0, 1, 2],
        "Unexpected complete model contract",
    )
    component = identity["source"]["component"]
    component_path = Path(component["receipt"]["path"])
    require(
        digest(component_path) == component["receipt"]["sha256"], "Component receipt bytes changed"
    )
    component_receipt = require_receipt(
        component_path, kind="deepseek-q1-hint-exact-candidate-v1", identity=component["identity"]
    )
    require(component_receipt["checks"]["comparisons"] == 78, "Incomplete component acceptance")
    component_result = read(component_receipt["artifact_paths"]["result.json"])
    native = identity["timing_runtime"]["hint"]["native"]
    require(
        native == component["native"] == component_result["runtime"]["native"]
        and component["component_runtime"] == component_result["runtime"]
        and native["build_identity"]["source_identity"] == component["identity"]["candidate_build"],
        "Model and component use different private native builds",
    )
    for path, expected in component["component_runtime"]["loaded_libraries"].items():
        require(
            identity["timing_runtime"]["hint"]["loaded_libraries"].get(path) == expected,
            "Model changed a component runtime library",
        )
    component_bench = Path(component["benchmark"]["path"])
    require(
        digest(component_bench) == component["benchmark"]["sha256"], "Component benchmark changed"
    )
    component_timing = read(component_bench)
    require(
        component_timing["completed"] is True
        and component_timing["mode"] == "bench"
        and component_timing["identity"] == component["identity"]
        and component_timing["receipt"]["sha256"] == component["receipt"]["sha256"],
        "Component timing is not bound to its acceptance",
    )
    request = identity["source"]["request"]
    require(digest(request["path"]) == request["sha256"], "Request bytes changed")
    runs, records = {}, []
    for mode, directory in (
        ("check", args.receipt.parent),
        ("bench", args.bench_dir),
        ("profile", args.profile_dir),
    ):
        runs[mode], record = audit_run(directory, mode, identity, receipt)
        records.append(record)
    return runs, {
        "receipt": file_identity(args.receipt),
        "receipt_signature": receipt["receipt_sha256"],
        "identity_sha256": identity_digest(identity),
        "receipt_checks": checks,
        "receipt_artifacts_verified": len(receipt["artifacts"]),
        "component_receipt": file_identity(component_path),
        "component_benchmark": file_identity(component_bench),
        "private_native": native,
        "runs": records,
        "boundary": "Receipt signatures, every retained acceptance artifact, archived execution source/native bytes, exact check/bench/profile identity and accepted component DSO/runtime are verified. Numerical checks are receipt-bound here, not rerun. Checkpoint identity retains metadata hashes and shard stat inventory; it does not hash all weight payloads.",
    }


def traffic(rows):
    retained = []
    for sample in rows:
        metrics = sample["layer_metrics"]
        require(len(metrics) == 3, "Missing per-layer cache counters")
        for index, row in enumerate(metrics):
            require(
                row["record_bytes"] == 1152
                and row["selection_records"] == 2048
                and row["written_records"] == row["host_written_records"] == 1
                and row["evicted_records"]
                == row["capacity_splits"]
                == row["transient_written_records"]
                == 0
                and 0 <= row["prefetched_records"] <= 64
                and row["recalled_records"] + row["resident_selection_records"]
                == row["selection_records"]
                and row["host_to_device_bytes"]
                == (row["prefetched_records"] + row["recalled_records"]) * row["record_bytes"]
                and row["device_to_host_bytes"]
                == row["host_written_records"] * row["record_bytes"],
                "Cache counter/byte accounting differs",
            )
            retained.append(
                {
                    "arm": sample["arm"],
                    "pair": sample["pair"],
                    "order": sample["order"],
                    "layer": index,
                    **row,
                }
            )
    fields = (
        "prefetched_records",
        "prefetch_capacity_failures",
        "recalled_records",
        "resident_selection_records",
        "host_to_device_bytes",
        "device_to_host_bytes",
    )
    summaries = []
    for arm in ARMS:
        for layer in range(3):
            selected = [row for row in retained if row["arm"] == arm and row["layer"] == layer]
            require(selected, "Empty traffic stratum")
            summaries.append(
                {
                    "arm": arm,
                    "layer": layer,
                    "samples": len(selected),
                    "counters": {
                        field: {
                            "min": min(row[field] for row in selected),
                            "median": statistics.median(row[field] for row in selected),
                            "max": max(row[field] for row in selected),
                        }
                        for field in fields
                    },
                }
            )
    return {
        "actual_samples": retained,
        "summary": summaries,
        "boundary": "These are each recorded execution's actual cache counters. Mapped-host prefetch/recall traffic is kernel activity and is not inferred from CUDA memcpy rows. Scheduling may change prediction/recall splits; no profile sample is reused as a clean sample's traffic.",
    }


def clean_timing(bench):
    pairs, rows = bench["pairs"], bench["result"]["samples"]
    require(
        type(pairs) is int and pairs >= 100 and pairs % 2 == 0, "Insufficient balanced model pairs"
    )
    expected = [
        (pair, arm, "AB" if pair % 2 == 0 else "BA")
        for pair in range(pairs)
        for arm in (ARMS if pair % 2 == 0 else ARMS[::-1])
    ]
    require(
        [(row["pair"], row["arm"], row["order"]) for row in rows] == expected,
        "Missing, duplicated or unbalanced timing sample",
    )
    require(
        all(math.isfinite(row["wall_ms"]) and row["wall_ms"] > 0 for row in rows),
        "Invalid model latency",
    )
    by_pair = {(row["pair"], row["arm"]): row["wall_ms"] for row in rows}
    delta = [by_pair[pair, "candidate"] - by_pair[pair, "baseline"] for pair in range(pairs)]
    derived = {
        "median_wall_ms": {
            arm: statistics.median(row["wall_ms"] for row in rows if row["arm"] == arm)
            for arm in ARMS
        },
        "paired_delta_ms": delta,
        "median_paired_delta_ms": statistics.median(delta),
        "candidate_wins": sum(value < 0 for value in delta),
        "order_median_delta_ms": {
            order: statistics.median(delta[index] for index in range(pairs) if index % 2 == parity)
            for parity, order in enumerate(("AB", "BA"))
        },
    }
    require(derived == bench["result"]["summary"], "Saved clean summary differs from raw samples")
    return {
        "pairs": pairs,
        "summary": derived,
        "traffic": traffic(rows),
        "boundary": "Independent clean complete forward(return_hidden=True)+synchronize wall latency; restore, binding and counter collection are outside timing. Profile latency does not supply these numbers.",
    }


def all_activity(connection, strings, start, end):
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
    processes = set()
    for name in ("KERNEL", "MEMCPY", "MEMSET"):
        table = "CUPTI_ACTIVITY_KIND_" + name
        if table in tables:
            processes.update(
                row[0]
                for row in connection.execute(
                    f"SELECT DISTINCT globalPid FROM {table} WHERE end>? AND start<?", (start, end)
                )
            )
    return [
        row
        for process in sorted(processes)
        for row in activity_rows(connection, strings, process=process, window=(start, end))
    ]


def clipped_union(connection, strings, selected):
    statistics_ = summary(selected)
    start, end = statistics_["start_ns"], statistics_["end_ns"]
    rows = all_activity(connection, strings, start, end)
    clipped = [
        {
            **row,
            "unclipped_start": row["start"],
            "unclipped_end": row["end"],
            "start": max(start, row["start"]),
            "end": min(end, row["end"]),
        }
        for row in rows
    ]
    actual = _union((row["start"], row["end"]) for row in clipped)
    expected = _union((row["start"], row["end"]) for row in selected)
    require(
        actual == expected,
        "All-recorded-process/device clipped union differs from selected graph work",
    )
    groups = sorted({(row["globalPid"], row["deviceId"]) for row in clipped})
    return {
        "selected": statistics_,
        "all_recorded_processes_devices": summary(clipped),
        "exact_union_intervals_equal": True,
        "clipped_activities": clipped,
        "per_process_device": [
            {
                "process": process,
                "device": device,
                "statistics": summary(
                    [
                        row
                        for row in clipped
                        if (row["globalPid"], row["deviceId"]) == (process, device)
                    ]
                ),
            }
            for process, device in groups
        ],
        "boundary": "Every exported kernel/memcpy/memset row intersecting this window is included without filtering process or device. This describes recorded activities, not untraced processes or SM utilization.",
    }


def profile_coverage(path, index, invocation):
    with closing(database(path)) as connection:
        strings = dict(connection.execute("SELECT id,value FROM StringIds"))
        metadata = list(connection.execute("SELECT name,value FROM META_DATA_CAPTURE"))
        safe = {key: [row[1] for row in metadata if row[0] == key] for key in CAPTURE_KEYS}
        require(
            set(safe["CAPTURE_EVENT_TYPE"]) == {"Cuda", "NvtxEvents"}, "Unexpected trace providers"
        )
        require(
            safe["CUDA_GRAPH_TRACE_OPTIONS:MODE"] == ["Node"] and safe["RATE_HZ"] == ["0"],
            "Node trace or sampling configuration differs",
        )
        require(
            safe["COLLECT_THREAD_STATE_TRACE"] == ["false"]
            and safe["COLLECT_GPU_CTX_SW_TRACE"] == ["false"],
            "Unexpected context-switch trace",
        )
        argv = sorted(
            (int(row[0].rsplit("_", 1)[1]), row[1])
            for row in metadata
            if row[0].startswith("PROCESS_0:ARGUMENT_")
        )
        command = [*safe["PROCESS_0:COMMAND"], *(value for _, value in argv)]
        require(
            command == invocation["argv"][invocation["argv"].index("taskset") :],
            "Raw NSYS target command differs from invocation",
        )
        require(
            safe["PROCESS_0:WORKING_DIR"] == [invocation["cwd"]],
            "Raw NSYS working directory differs",
        )
        scopes = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM NVTX_EVENTS WHERE text LIKE 'q1_hint_model%' ORDER BY start"
            )
        ]
        expected = (
            ["q1_hint_model_capture/" + ARMS[index - 1]]
            if index < 3
            else ["q1_hint_model/" + arm for arm in ARMS]
        )
        require(
            [row["text"] for row in scopes] == expected, "Capture range labels or ordering differ"
        )
        counts = Counter(
            strings[row[0]]
            for row in connection.execute("SELECT nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        )

        def api_count(fragment):
            return sum(count for name, count in counts.items() if fragment in name)

        if index < 3:
            require(
                api_count("cudaStreamBeginCapture")
                == api_count("cudaStreamEndCapture")
                == api_count("cudaGraphInstantiate")
                == 1
                and api_count("cudaGraphLaunch") == 0,
                "Setup capture lineage coverage differs",
            )
        else:
            require(
                api_count("cudaGraphLaunch") == 2
                and api_count("cudaStreamBeginCapture")
                == api_count("cudaStreamEndCapture")
                == api_count("cudaGraphInstantiate")
                == 0,
                "Replay range contains wrong graph lifecycle calls",
            )
        return {
            "sqlite": file_identity(path),
            "scopes": scopes,
            "safe_capture_metadata": safe,
            "target_command": command,
            "graph_api_counts": {
                name: count
                for name, count in counts.items()
                if any(fragment in name for fragment in ("Graph", "Capture"))
            },
            "export_metadata": dict(connection.execute("SELECT name,value FROM META_DATA_EXPORT")),
        }


def hint_change(runs):
    removed, added, rows = [], [], []
    for layer in LAYERS:
        selected = {}
        for arm in ARMS:
            selected[arm] = sorted(
                (
                    row
                    for row in runs[arm]["nodes"]
                    if row["owner"]["layer"] == layer and row["owner"]["stage"] == "prefetch_hint"
                ),
                key=lambda row: row["start"],
            )
            require(all(row["kind"] == "kernel" for row in selected[arm]), "Non-kernel hint node")
        baseline, candidate = selected["baseline"], selected["candidate"]
        require(
            len(baseline) == 4 and len(candidate) == 2,
            "Expected three-versus-one mean kernels plus EMA",
        )
        require(
            baseline[0]["name"] == "mask_count"
            and "at::native::reduce_kernel" in baseline[1]["name"]
            and baseline[2]["name"] == "publish"
            and baseline[3]["name"] == candidate[1]["name"] == "update_decode_ema"
            and candidate[0]["name"] == "q1_hint_exact::mean_kernel(const float *, float *)",
            "Actual hint kernel sequence differs",
        )
        require(
            signature(baseline[3]) == signature(candidate[1])
            and baseline[3]["owner"] == candidate[1]["owner"],
            "Decode EMA changed",
        )
        removed.extend(baseline[:3])
        added.extend(candidate[:1])
        rows.append(
            {
                "layer": layer,
                "baseline": [
                    {
                        "signature": signature(row),
                        "owner": row["owner"],
                        "capture_node": row["capture_node"],
                        "duration_us": (row["end"] - row["start"]) / 1000,
                    }
                    for row in baseline
                ],
                "candidate": [
                    {
                        "signature": signature(row),
                        "owner": row["owner"],
                        "capture_node": row["capture_node"],
                        "duration_us": (row["end"] - row["start"]) / 1000,
                    }
                    for row in candidate
                ],
                "baseline_hint_kernel_sum_us": sum(row["end"] - row["start"] for row in baseline)
                / 1000,
                "candidate_hint_kernel_sum_us": sum(row["end"] - row["start"] for row in candidate)
                / 1000,
            }
        )
    counters = {}
    for arm, excluded in (("baseline", removed), ("candidate", added)):
        excluded_ids = {row["capture_node"] for row in excluded}
        counters[arm] = Counter(
            json.dumps({"signature": signature(row), "owner": row["owner"]}, sort_keys=True)
            for row in runs[arm]["nodes"]
            if row["capture_node"] not in excluded_ids
        )
    require(
        counters["baseline"] == counters["candidate"],
        "Non-mean kernel/copy signature and native ownership multiset changed",
    )
    common = [
        {**json.loads(key), "count": count} for key, count in sorted(counters["baseline"].items())
    ]
    return {
        "layers": rows,
        "removed_mean_nodes": len(removed),
        "added_mean_nodes": len(added),
        "unchanged_node_count": sum(counters["baseline"].values()),
        "unchanged_signature_owner_multiset": common,
        "kernel_signature_fields": list(FIELDS),
        "boundary": "Native capture lineage assigns each layer/stage before names identify the kernels within prefetch_hint. The unchanged multiset includes EMA, all other kernels, all memcpy bytes/kinds and memset bytes. This verifies observed work/signatures/owners, not complete graph edge equivalence or a hardware cause for the clean latency delta.",
    }


def analyze_profile(args, profile):
    templates = profile["result"]["capture_templates"]
    require(set(templates) == set(ARMS), "Missing graph template")
    invocation = read(args.profile_dir / "invocation.json")
    for flag in (
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--cuda-graph-trace=node",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=repeat",
    ):
        require(flag in invocation["argv"], "Required NSYS invocation flag missing: " + flag)
    paths = [args.profile_dir / f"capture.{index}.sqlite" for index in (1, 2, 3)]
    coverage = [profile_coverage(path, index, invocation) for index, path in enumerate(paths, 1)]
    parents = read_lineage(paths[:2])
    require(parents and parents.process is not None, "Missing same-process clone lineage")
    for index, arm in enumerate(ARMS):
        template = templates[arm]
        require(
            template == read(args.profile_dir / (arm + "_capture_template.json")),
            "Standalone graph template differs",
        )
        require(
            template["method"] == arm and template["runtime"]["replays"] == 0,
            "Wrong capture arm or preparation replay state",
        )
        require(
            set(template["node_types"].values()) == {0, 1, 2}, "Unexpected event/other graph node"
        )
        expected = set(template["gpu_node_ids"])
        require(
            len(expected) == len(template["gpu_node_ids"])
            and expected
            == set(map(int, template["node_owners"]))
            == set(map(int, template["node_types"])),
            "Incomplete graph node/type/owner inventory",
        )
        with closing(database(paths[index])) as connection:
            events = [
                dict(row) for row in connection.execute("SELECT * FROM CUDA_GRAPH_NODE_EVENTS")
            ]
            require(
                expected <= {row["graphNodeId"] for row in events},
                "Template node absent from its raw setup capture",
            )
            require(
                all(row["globalTid"] & PROCESS_MASK == parents.process for row in events),
                "Setup lineage process differs",
            )
    samples = profile["result"]["samples"]
    require(
        [(row["arm"], row["pair"], row["order"]) for row in samples]
        == [(arm, 0, "AB") for arm in ARMS],
        "Wrong profiled replay order",
    )
    runs = {}
    with closing(database(paths[2])) as connection:
        strings = dict(connection.execute("SELECT id,value FROM StringIds"))
        for scope in coverage[2]["scopes"]:
            arm = scope["text"].split("/")[1]
            template = templates[arm]
            require(
                scope["globalTid"] & PROCESS_MASK == parents.process,
                "Replay and capture processes differ",
            )
            runtime = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start<? AND end>?",
                    (scope["end"], scope["start"]),
                )
                if row["globalTid"] & PROCESS_MASK == parents.process
            ]
            launches = [row for row in runtime if "cudaGraphLaunch" in strings[row["nameId"]]]
            require(len(launches) == 1, "Replay does not contain exactly one runtime graph launch")
            launch = launches[0]
            require(
                launch["globalTid"] == scope["globalTid"]
                and scope["start"] <= launch["start"] < launch["end"] <= scope["end"],
                "Launch does not belong to replay scope/thread",
            )
            nodes = activity_rows(
                connection, strings, correlation=launch["correlationId"], process=parents.process
            )
            for row in nodes:
                require(row.get("graphNodeId"), "Replay activity lacks native graph node ID")
                row["capture_node"] = original_node(row["graphNodeId"], parents)
                require(str(row["capture_node"]) in template["node_owners"], "Unknown replay node")
                row["owner"] = template["node_owners"][str(row["capture_node"])]
                require(
                    row["owner"]["layer"] in (*LAYERS, "shared"), "Unexpected native layer owner"
                )
                require(
                    template["node_types"][str(row["capture_node"])]
                    == {"kernel": 0, "memcpy": 1, "memset": 2}[row["kind"]],
                    "Raw activity and native node types differ",
                )
                validate_graph_id(row, template["executable_graph_id"])
                require(
                    scope["start"] <= row["start"] < row["end"] <= scope["end"],
                    "Synchronized replay activity crosses its CPU scope",
                )
            require(
                len(nodes) == len({row["capture_node"] for row in nodes})
                and {row["capture_node"] for row in nodes} == set(template["gpu_node_ids"]),
                "Missing, duplicate or extra replay GPU node",
            )
            nodes.sort(key=lambda row: row["capture_node"])
            scope_activities = all_activity(connection, strings, scope["start"], scope["end"])
            outside = [
                row
                for row in scope_activities
                if not (
                    row["globalPid"] == parents.process
                    and row["correlationId"] == launch["correlationId"]
                )
            ]
            require(
                Counter((row["kind"], row.get("copyKind"), row.get("bytes")) for row in outside)
                == Counter({("memcpy", 1, 8): 1, ("memcpy", 8, 8): 1})
                and all(
                    not row.get("graphNodeId") and row["globalPid"] == parents.process
                    for row in outside
                ),
                "Unexpected activity outside graph in forward scope",
            )
            layer_nodes = [row for row in nodes if row["owner"]["layer"] in LAYERS]
            runs[arm] = {
                "scope": scope,
                "launch": {**launch, "name": strings[launch["nameId"]]},
                "capture_graph_id": template["capture_graph_id"],
                "executable_graph_id": template["executable_graph_id"],
                "nodes": nodes,
                "node_types": dict(Counter(row["kind"] for row in nodes)),
                "single_graph_launch": True,
                "every_native_gpu_node_verified_once": True,
                "complete_graph_union": clipped_union(connection, strings, nodes),
                "three_layer_union": clipped_union(connection, strings, layer_nodes),
                "layer_unions": {
                    layer: clipped_union(
                        connection,
                        strings,
                        [row for row in nodes if row["owner"]["layer"] == layer],
                    )
                    for layer in LAYERS
                },
                "outside_graph_input_copies": outside,
                "actual_cache_metrics": next(
                    row["layer_metrics"] for row in samples if row["arm"] == arm
                ),
            }
    return {
        "invocation": file_identity(args.profile_dir / "invocation.json"),
        "coverage": coverage,
        "raw_reports": [
            file_identity(args.raw_profile_dir / f"capture.{index}.nsys-rep") for index in (1, 2, 3)
        ],
        "process": parents.process,
        "clone_edges": len(parents),
        "runs": runs,
        "hint_change": hint_change(runs),
        "traffic": traffic(samples),
        "boundary": "Independent invasive profile, with setup captures 1/2 and one AB replay pair in capture 3. SQL and raw reports are retained by hash. Recorded GPU interval sums/unions are not clean model latency or the candidate promotion metric.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--bench-dir", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--raw-profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in vars(args):
        setattr(args, name, getattr(args, name).resolve())
    require(not args.output_dir.exists(), "Analysis output already exists")
    source = {relative: digest(ROOT / relative) for relative in SOURCE_CLOSURE}
    runs, bindings = audit_bindings(args)
    result = {
        "passed": True,
        "run_id": args.output_dir.name,
        "analysis_sources": source,
        "bindings": bindings,
        "clean_timing": clean_timing(runs["bench"]),
        "profile": analyze_profile(args, runs["profile"]),
    }
    require(
        source == {relative: digest(ROOT / relative) for relative in SOURCE_CLOSURE},
        "Analysis helper source changed while reading",
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for relative, expected in source.items():
        destination = args.output_dir / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        require(digest(destination) == expected, "Analyzer source archive differs")
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "passed": True,
                "analysis": str(args.output_dir / "result.json"),
                "clean_summary": result["clean_timing"]["summary"],
                "unchanged_nodes": result["profile"]["hint_change"]["unchanged_node_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
