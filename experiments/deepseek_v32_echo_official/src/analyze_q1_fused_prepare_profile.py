"""Audit the separate stateless-preparation profile using raw native graph lineage."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from contextlib import closing
from itertools import pairwise
from pathlib import Path

from experiments.deepseek_v32_echo_official.src import analyze_q1_fused_prepare_model as model
from experiments.deepseek_v32_echo_official.src.analyze_q1_hint_model import (
    ARMS,
    CAPTURE_KEYS,
    FIELDS,
    LAYERS,
    PROCESS_MASK,
    activity_rows,
    all_activity,
    clipped_union,
    database,
    file_identity,
    original_node,
    read,
    read_lineage,
    require,
    signature,
    traffic,
    validate_graph_id,
)

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
POLICY = "private-fused-prepare-model-immutable-flashinfer-v1"
SOURCE_CLOSURE = (
    *model.SOURCE_CLOSURE,
    str(Path(__file__).resolve().relative_to(ROOT)),
    "experiments/deepseek_v32_echo_official/scripts/profile_q1_fused_prepare_model.py",
)


def operation(row):
    require(row["kind"] == "kernel", "Non-kernel in indexer stage")
    name = row["name"]
    if "at::native::FillFunctor<int>" in name:
        return "bounds_zero"
    if "at::native::arange_cuda_out" in name:
        return "arange"
    if name == "pack_page64":
        return "pack_page64"
    if name.startswith("void deep_gemm::smxx_paged_mqa_logits_metadata<"):
        return "scheduler"
    if name.startswith("official_prefetch::prepare_kernel("):
        return "stage_and_tokens"
    if name.startswith("void q1_fused_prepare::prepare_kernel<"):
        return "fused_prepare"
    if name.startswith("void deep_gemm::sm90_fp8_paged_mqa_logits_fused_v2<"):
        return "official_core"
    if name.startswith("void deep_gemm::smxx_clean_logits<"):
        return "clean"
    if name.startswith("official_prefetch::validate_promotion_kernel("):
        return "validate_promotion"
    if name.startswith("official_prefetch::copy_publish_kernel("):
        return "copy_publish"
    raise ValueError("Unknown indexer-stage kernel: " + name)


def preparation_change(runs):
    expected = {
        "baseline": [
            "bounds_zero",
            "arange",
            "pack_page64",
            "arange",
            "scheduler",
            "stage_and_tokens",
            "official_core",
            "clean",
            "validate_promotion",
            "copy_publish",
        ],
        "candidate": [
            "bounds_zero",
            "arange",
            "fused_prepare",
            "scheduler",
            "official_core",
            "clean",
            "validate_promotion",
            "copy_publish",
        ],
    }
    positions = {"baseline": (2, 3, 5), "candidate": (2,)}
    removed = {arm: [] for arm in ARMS}
    layers = []
    for layer in LAYERS:
        comparison = {"layer": layer}
        for arm in ARMS:
            nodes = sorted(
                (
                    row
                    for row in runs[arm]["nodes"]
                    if row["owner"]["layer"] == layer and row["owner"]["stage"] == "indexer_fused"
                ),
                key=lambda row: row["start"],
            )
            require(
                [operation(row) for row in nodes] == expected[arm],
                "Unexpected indexer kernel sequence",
            )
            require(
                all(
                    row["owner"]
                    == {
                        "layer": layer,
                        "stage": "indexer_fused",
                        "source_stage": "indexer_prefetch",
                        "stage_path": ["extend_graph_body", layer, "indexer_prefetch"],
                    }
                    for row in nodes
                ),
                "Indexer native ownership differs",
            )
            require(
                all(left["end"] <= right["start"] for left, right in pairwise(nodes)),
                "Indexer kernels do not follow the declared sequential order",
            )
            selected = [nodes[index] for index in positions[arm]]
            removed[arm].extend(selected)
            comparison[arm] = [
                {
                    "signature": signature(row),
                    "owner": row["owner"],
                    "capture_node": row["capture_node"],
                    "operation": operation(row),
                    "duration_us": (row["end"] - row["start"]) / 1000,
                }
                for row in selected
            ]
            comparison[arm + "_preparation_sum_us"] = (
                sum(row["end"] - row["start"] for row in selected) / 1000
            )
        layers.append(comparison)
    counters = {}
    for arm in ARMS:
        excluded = {row["capture_node"] for row in removed[arm]}
        require(len(excluded) == len(removed[arm]), "Duplicate preparation capture node")
        counters[arm] = Counter(
            json.dumps({"signature": signature(row), "owner": row["owner"]}, sort_keys=True)
            for row in runs[arm]["nodes"]
            if row["capture_node"] not in excluded
        )
    require(
        counters["baseline"] == counters["candidate"],
        "Non-preparation kernel/copy signature and native ownership multiset changed",
    )
    return {
        "layers": layers,
        "baseline_preparation_nodes": len(removed["baseline"]),
        "candidate_preparation_nodes": len(removed["candidate"]),
        "baseline_preparation_sum_us": sum(row["end"] - row["start"] for row in removed["baseline"])
        / 1000,
        "candidate_preparation_sum_us": sum(
            row["end"] - row["start"] for row in removed["candidate"]
        )
        / 1000,
        "unchanged_node_count": sum(counters["baseline"].values()),
        "unchanged_signature_owner_multiset": [
            {**json.loads(key), "count": count}
            for key, count in sorted(counters["baseline"].items())
        ],
        "kernel_signature_fields": list(FIELDS),
        "boundary": "Native capture lineage assigns layer/stage before the complete within-stage sequence identifies preparation. Only packing, block-table arange, and token/stage reset are replaced. Bounds preparation, scheduler, official core, clean, promotion, all other kernels and all copy/memset bytes retain exact observed signature/owner multisets. This does not establish full graph edge equivalence or attribute the clean wall delta solely to preparation.",
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
                "SELECT * FROM NVTX_EVENTS WHERE text LIKE 'q1_fused_prepare_model%' ORDER BY start"
            )
        ]
        expected = (
            ["q1_fused_prepare_model_capture/" + ARMS[index - 1]]
            if index < 3
            else ["q1_fused_prepare_model/" + arm for arm in ARMS]
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
        "preparation_change": preparation_change(runs),
        "traffic": traffic(samples),
        "boundary": "Independent invasive profile, with setup captures 1/2 and one AB replay pair in capture 3. SQL and raw reports are retained by hash. Recorded GPU interval sums/unions are not clean model latency or the candidate promotion metric.",
    }


def audit_record_files(node, evidence):
    if isinstance(node, dict):
        if {"path", "sha256", "bytes"} <= node.keys():
            evidence.file(node["path"], node["sha256"], size=node["bytes"])
        for value in node.values():
            audit_record_files(value, evidence)
    elif isinstance(node, list):
        for value in node:
            audit_record_files(value, evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "bench-dir", "profile-dir", "raw-profile-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        setattr(args, name, value.resolve())
    require(
        args.output_dir.is_relative_to(EXPERIMENT / "output"), "Analysis output outside experiment"
    )
    require(not args.output_dir.exists(), "Use a fresh analysis run ID")
    evidence = model.Evidence()
    sources = {name: evidence.file(ROOT / name)["sha256"] for name in SOURCE_CLOSURE}
    bench, bindings = model.audit_bindings(args.receipt, args.bench_dir, evidence)
    receipt = evidence.read(args.receipt)
    identity = bench["identity"]
    profile = model.audit_run(args.profile_dir, "profile", identity, receipt, evidence)
    require(
        profile["result"]["receipt_sha256"] == receipt["receipt_sha256"], "Profile receipt differs"
    )
    envelope = evidence.read(args.profile_dir / "pre_gate_identity.json")
    require(
        envelope.get("schema") == POLICY
        and envelope.get("mode") == "profile"
        and envelope.get("identity") == identity
        and envelope.get("pairs") == profile["pairs"]
        and envelope.get("receipt_path") == str(args.receipt),
        "Profile pre-gate envelope differs",
    )
    invocation = evidence.read(args.profile_dir / "invocation.json")
    checked_environments = [
        row["build_identity"]["environment"]
        for row in identity["timing_runtime"]["private_flashinfer_native"].values()
    ]
    require(
        all(row == invocation["accepted_build_environment"] for row in checked_environments),
        "NSYS target did not preserve the checked compiler environment",
    )
    evidence.file(invocation["script"]["path"], invocation["script"]["sha256"])
    checked = evidence.read(args.receipt.parent / "result.json")
    memory = {}
    for arm in ARMS:
        graph = profile["memory"][arm]["graph"]
        expected = checked["memory"][arm]["graph"]
        require(
            all(
                graph[key] == expected[key] == bench["memory"][arm]["graph"][key]
                for key in (
                    "private_reserved_bytes",
                    "static_allocated_bytes",
                    "reservation_bytes",
                    "chosen_private_limit_bytes",
                )
            ),
            "Graph memory contract changed across check, bench and profile",
        )
        require(
            graph["reservation_bytes"]
            == graph["chosen_private_limit_bytes"] + graph["static_allocated_bytes"]
            and graph["private_reserved_bytes"] <= graph["chosen_private_limit_bytes"],
            "Graph reservation arithmetic differs",
        )
        row = profile["memory"][arm]
        require(
            0
            <= row["allocated"]
            <= row["reserved"]
            <= row["device_used"]
            <= graph["memory_at_allocation"]["device_total"],
            "Invalid process memory snapshot",
        )
        memory[arm] = row
    analyzed = analyze_profile(args, profile)
    audit_record_files(analyzed, evidence)
    result = {
        "schema": "deepseek-private-q1-fused-prepare-profile-analysis-v1",
        "run_id": args.output_dir.name,
        "passed": True,
        "passed_meaning": "Receipt/native/archive identity, raw trace ownership and complete work/traffic accounting passed. This is not a clean timing or production promotion verdict.",
        "bindings": {**bindings, "profile": evidence.file(args.profile_dir / "result.json")},
        "profile": analyzed,
        "memory": memory,
        "memory_boundary": "Per-arm private graph storage is compared exactly. Allocated/reserved/device-used are sequential process snapshots, not isolated per-model footprints or continuous peaks.",
        "analysis_sources": sources,
    }
    evidence.verify()
    args.output_dir.mkdir(parents=True)
    model.write_json(args.output_dir / "result.json", result)
    for name, expected in sources.items():
        target = args.output_dir / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        evidence.file(target, expected)
    model.write_json(args.output_dir / "input_hashes.json", evidence.files)
    model.write_json(
        args.output_dir / "publication_manifest.json",
        {
            "schema": "deepseek-private-q1-fused-prepare-profile-publication-v1",
            "artifacts": {
                str(path.relative_to(args.output_dir)): evidence.file(path)
                for path in sorted(args.output_dir.rglob("*"))
                if path.is_file()
            },
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "run_id": args.output_dir.name,
                "unchanged_nodes": analyzed["preparation_change"]["unchanged_node_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
