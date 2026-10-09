"""Generate verified four-method MFU artifacts for one full CUDA Graph per extend."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from evaluation.validation import identity_digest
from experiments.deepseek_v32_mfu.src.operator_report import analyze_captures, read_calls
from experiments.deepseek_v32_mfu.src.publish_layers import STAGES
from experiments.deepseek_v32_mfu.src.report_mfu import (
    _independent_runs,
    audit_local_native_artifacts,
    audit_work,
    plot_mfu,
    save_csv,
    save_json,
)
from experiments.deepseek_v32_mfu.src.run_contract import (
    METHODS,
    benchmark_view,
    digest,
    execution_identity,
)
from experiments.deepseek_v32_motivation.src.graph_instrumentation import REPLAY_PATTERN

PEAKS = {"FP8": 1979.0, "BF16": 989.5, "FP32": 67.0}
PHASES = ("prefill_annotated", "extend_annotated")
GRAPH_POLICY = "deepseek-full-extend-graph-v2-dense-late-wait"


def _read(path):
    return json.loads(path.read_text())


def audit_hardware(hardware):
    """Validate recorded NVML/torch identity; display names do not identify the SKU."""
    gpu = hardware["gpu"]
    pci = hardware["pci_identity"]
    if (
        hardware.get("is_sm90") is not True
        or gpu.get("compute_cap") != "9.0"
        or pci.get("vendor_id") != "10de"
        or pci.get("device_id") != "2335"
        or int(gpu["pci.device_id"], 16) != 0x233510DE
        or hardware.get("h200_sxm_reference_matches_pci") is not True
    ):
        raise ValueError("nominal H200 SXM peaks require matching SM90 and PCI 10de:2335 evidence")
    torch_uuid, nvml_uuid = hardware.get("torch_device_uuid"), gpu.get("uuid")
    if (
        not isinstance(torch_uuid, str)
        or not isinstance(nvml_uuid, str)
        or not torch_uuid.removeprefix("GPU-")
        or torch_uuid.removeprefix("GPU-") != nvml_uuid.removeprefix("GPU-")
    ):
        raise ValueError("recorded torch and NVML device UUIDs differ")
    reference = hardware.get("h200_sxm_reference", {})
    if reference.get("verified") and reference.get("dense_peaks_tflops") != PEAKS:
        raise ValueError("recorded verified H200 SXM reference differs from the MFU dense peaks")


def capture_paths(directory, result, *, methods=METHODS):
    """Require all setup captures and exactly one measured capture per phase/method."""
    order = result["nsys_capture_order"]
    expected = (["graph_setup"] if result["compute_graphs"] else []) + [
        label
        for method in methods
        for label in (
            f"{method}/prefill_annotated",
            f"{method}/extend_graph_setup",
            f"{method}/extend_annotated",
        )
    ]
    if order != expected:
        raise ValueError("capture order must include every independent setup and measured phase")
    setup, measured = [], []
    for index, label in enumerate(order, 1):
        (setup if label.endswith("graph_setup") else measured).append(
            directory / f"capture_{index}.sqlite"
        )
    return setup, measured


def _match_template(replay, template):
    if (
        replay["graph_capture_id"] != template["capture_graph_id"]
        or replay["graph_id"] != template["executable_graph_id"]
        or replay["graph_gpu_node_ids"] != template["gpu_node_ids"]
    ):
        raise ValueError("graph replay differs from its captured template")


def audit_graph_ledgers(result, calls, full_templates, compute_templates=None, *, methods=METHODS):
    """Bind actual capture-time matrix metadata and every replay to its template."""
    if full_templates != result["full_extend_graph_templates"]:
        raise ValueError("full graph template file differs from profile metadata")
    if len(full_templates) != len(methods) or {row["method"] for row in full_templates} != set(
        methods
    ):
        raise ValueError("full graph templates must cover the selected methods exactly")
    full_rows = []
    for template in full_templates:
        method = template["method"]
        selected = [call for call in calls if (call["mode"], call["phase"]) == (method, PHASES[1])]
        replays = [call for call in selected if call.get("graph_replay")]
        if len(replays) != 1 or not replays[0].get("full_extend_graph"):
            raise ValueError("each extend must contain exactly one full graph replay")
        replay = replays[0]
        if replay["stage"] != f"extend_graph_replay_q_{result['extend_tokens']}":
            raise ValueError("full graph query shape differs from complete extend")
        _match_template(replay, template)
        nodes = template["gpu_node_ids"]
        owners = template["node_owners"]
        if (
            not nodes
            or len(nodes) != len(set(nodes))
            or set(owners) != {str(node) for node in nodes}
            or replay["graph_node_owners"] != owners
        ):
            raise ValueError("full graph node ownership is incomplete or changed")
        operators = template.get("operators", [])
        apis = [call for call in selected if call.get("graph_api")]
        matrix = [call for call in selected if call["useful_flops"] is not None]
        if not operators or len(apis) != len(operators) or apis != matrix:
            raise ValueError("full extend matrix work must come from captured matrix APIs")
        expected_calls = Counter(
            (f"layer_{layer}", stage)
            for layer in range(3)
            for stage in STAGES
            if stage not in {"indexer_fused", "lm_head"}
        )
        expected_calls["shared", "lm_head"] = 1
        actual_calls = Counter(
            (call["layer"], "indexer_qk" if call["stage"] == "indexer_fused" else call["stage"])
            for call in apis
        )
        if actual_calls != expected_calls:
            raise ValueError(
                "full extend requires one of every matrix API per layer and one LM head"
            )
        claimed = []
        for operator, call in zip(operators, apis, strict=True):
            if (
                call.get("graph_replay_nvtx") != replay["nvtx"]
                or call.get("parent_call_id") != replay["call_id"]
                or any(call.get(key) != value for key, value in operator.items())
                or call.get("cpu_inclusive_ns") != 0
            ):
                raise ValueError("matrix API replay ledger differs from capture-time metadata")
            api_nodes = operator["graph_node_ids"]
            if not api_nodes or operator["useful_flops"] is None:
                raise ValueError("matrix API template has no GPU nodes or useful work")
            claimed.extend(api_nodes)
        if len(claimed) != len(set(claimed)) or not set(claimed) <= set(nodes):
            raise ValueError("matrix API GPU node ownership is ambiguous")
        full_rows.append(
            {
                "method": method,
                "replays": 1,
                "gpu_nodes": len(nodes),
                "matrix_api_calls": len(apis),
                "matrix_api_gpu_nodes": len(claimed),
            }
        )
    prefill = [call for call in calls if call["phase"] == PHASES[0]]
    if any(call.get("full_extend_graph") for call in prefill):
        raise ValueError("prefill unexpectedly uses an extend graph")
    replays = [call for call in prefill if call.get("graph_replay")]
    prefill_audit = None
    if result["compute_graphs"]:
        if compute_templates is None:
            raise ValueError("prefill compute graphs require their capture templates")
        templates = {
            (item["part"], item["layer"], item["queries"]): item
            for item in compute_templates["graphs"]
        }
        if len(templates) != len(compute_templates["graphs"]):
            raise ValueError("duplicate prefill compute graph template")
        sizes = [
            min(result["chunk_size"], result["prefix_tokens"] - start)
            for start in range(0, result["prefix_tokens"], result["chunk_size"])
        ]
        expected = Counter(
            (method, layer, part, size)
            for method in methods
            for size in sizes
            for layer in range(3)
            for part in ("projection", "finish")
        )
        observed = Counter()
        for replay in replays:
            match = REPLAY_PATTERN.fullmatch(replay["stage"])
            if match is None:
                raise ValueError("prefill graph has an unknown replay shape")
            part, layer, size = match[1], int(match[2]), int(match[3])
            template = templates.get((part, layer, size))
            if template is None or replay["layer"] != f"layer_{layer}":
                raise ValueError("prefill replay lacks its layer/shape template")
            _match_template(replay, template)
            observed[replay["mode"], layer, part, size] += 1
        if observed != expected:
            raise ValueError("prefill replay counts do not cover every layer and query chunk")
        prefill_audit = {
            "chunks": sizes,
            "total_replays": sum(observed.values()),
            "every_layer_and_chunk_verified": True,
        }
    elif replays:
        raise ValueError("prefill graph replays contradict compute_graphs=False")
    return {"full_extend": full_rows, "prefill_compute_graphs": prefill_audit}


def audit_analysis(result, analysis, graph_ledger, *, methods=METHODS):
    expected = {(method, phase) for method in methods for phase in PHASES}
    captures = analysis["captures"]
    if (
        len(captures) != len(expected)
        or {(row["mode"], row["phase"]) for row in captures} != expected
    ):
        raise ValueError("profile lacks a selected method/phase capture")
    if analysis["calls_outside_selected_captures"]:
        raise ValueError("operator ledger contains uncaptured calls")
    if analysis["dense_peaks_tflops"] != PEAKS:
        raise ValueError("full graph MFU requires the recorded H200 dense peaks")
    by_method = {row["method"]: row for row in graph_ledger["full_extend"]}
    for capture in captures:
        audit = capture["audit"]
        if not audit["kernel_count_and_time_conserved"] or not audit["metadata_call_counts_match"]:
            raise ValueError("operator attribution failed conservation or call coverage")
        if audit["layer_unscoped_kernel_count"] or any(
            row["unattributed_count"] for row in audit["activity_counts_and_ns"].values()
        ):
            raise ValueError("profile contains unattributed GPU work")
        graph = capture["graph_attribution"]
        if (result["compute_graphs"] or capture["phase"] == PHASES[1]) and (
            graph is None or not graph["every_replay_gpu_node_verified"]
        ):
            raise ValueError("graph node ownership was not independently verified")
        if capture["phase"] == PHASES[1]:
            expected_graph = by_method[capture["mode"]]
            if (
                not graph.get("full_extend_graph")
                or graph["replays"] != 1
                or graph["gpu_node_activities"] != expected_graph["gpu_nodes"]
                or graph["matrix_api_count"] != expected_graph["matrix_api_calls"]
                or graph["matrix_api_gpu_node_activities"] != expected_graph["matrix_api_gpu_nodes"]
            ):
                raise ValueError("observed full graph differs from its matrix API ledger")
    for rows in (analysis["operators"], analysis["operators_by_layer"]):
        for row in rows:
            if row["useful_flops"] is None:
                if row["kernel_mfu_percent"] is not None:
                    raise ValueError("nonmatrix work cannot have an invented MFU")
            elif row["kernel_mfu_percent"] is None or row["kernel_ns"] <= 0:
                raise ValueError("matrix API lacks exclusively attributed GPU kernel time")


def _source_hashes():
    repository = Path(__file__).resolve().parents[3]
    files = [
        Path(__file__).with_name(name)
        for name in (
            "report_full_graph_mfu.py",
            "report_mfu.py",
            "operator_report.py",
            "analyze_nsys.py",
            "full_graph_profile.py",
            "execution_utilization.py",
            "run_contract.py",
            "publish_layers.py",
        )
    ]
    files.extend(
        repository / "experiments/deepseek_v32_motivation/src" / name
        for name in ("graph_attribution.py", "graph_instrumentation.py")
    )
    return {str(path.relative_to(repository)): digest(path) for path in files}


def generate(directory, output):
    """Analyze immutable independent runs; write fresh CPU-only report artifacts."""
    directory = Path(directory).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"report destination already exists: {output}")
    result = _read(directory / "result.json")
    if (
        result.get("schema_version") != 3
        or result.get("mode") != "profile"
        or result.get("accepted") is not True
        or result.get("extend_graph") is not True
        or result.get("extend_graph_policy_revision") != GRAPH_POLICY
        or result.get("profile_detail") != "matrix_api_node_ownership"
        or result.get("num_layers") != 3
        or result.get("methods") != list(METHODS)
        or result.get("run_id") != directory.name
    ):
        raise ValueError(
            "report requires an accepted four-method three-layer full graph MFU profile"
        )
    audit_hardware(result["hardware"])
    wall = benchmark_view(directory, result)
    receipt, directories, runs = _independent_runs(directory, result)
    native = audit_local_native_artifacts(runs)
    for name, run in runs.items():
        audit_hardware(run["hardware"])
        if run.get("mode") != name or run.get("accepted") is not True:
            raise ValueError(f"independent {name} run is not accepted")
        if (
            identity_digest(execution_identity(run)) != identity_digest(run["execution_identity"])
            or run["execution_identity"] != result["execution_identity"]
        ):
            raise ValueError("check, bench and profile have different execution identities")
        if digest(directories[name] / "request.json") != run["request_sha256"]:
            raise ValueError(f"{name} request changed")
        for path, expected in run["source_sha256"].items():
            if digest(directories[name] / "source" / path) != expected:
                raise ValueError(f"{name} source snapshot changed: {path}")
    calls, metadata = read_calls(directory / "operator_calls.json")
    if metadata.get("run_id") != result["run_id"]:
        raise ValueError("operator ledger belongs to another run")
    setup, paths = capture_paths(directory, result)
    inputs = [
        *[path / "result.json" for path in directories.values()],
        directory / "operator_calls.json",
        directory / "full_graph_templates.json",
        Path(receipt["receipt_path"]),
        *setup,
        *paths,
    ]
    if result["compute_graphs"]:
        inputs.append(directory / "graph_templates.json")
    before = {str(path): digest(path) for path in inputs}
    sources = _source_hashes()
    graph_ledger = audit_graph_ledgers(
        result,
        calls,
        _read(directory / "full_graph_templates.json"),
        _read(directory / "graph_templates.json") if result["compute_graphs"] else None,
    )
    analysis = analyze_captures(
        paths, calls, metadata=metadata, graph_setup_paths=setup, peaks=PEAKS
    )
    audit_analysis(result, analysis, graph_ledger)
    final, coverage = audit_work(wall, calls, PEAKS)
    summary = {
        "schema": "deepseek-full-extend-graph-mfu-v1",
        "report_run_id": output.name,
        "run_ids": {name: run["run_id"] for name, run in runs.items()},
        "profile_run_id": result["run_id"],
        "benchmark": result["benchmark"],
        "validation_receipt": result["validation_receipt"],
        "execution_identity_sha256": identity_digest(result["execution_identity"]),
        "configuration": {
            key: wall[key]
            for key in (
                "scope",
                "num_layers",
                "prefix_tokens",
                "extend_tokens",
                "chunk_size",
                "extend_chunk_size",
                "sparse_pool_tokens",
                "extend_residency",
                "compute_graphs",
                "extend_graph",
                "extend_graph_policy_revision",
                "warmups",
                "repeats",
                "prefill_repeats",
                "hbm_cache_budget_bytes",
                "dram_cache_budget_bytes",
            )
        },
        "hardware": result["hardware"],
        "dependencies": result["dependencies"],
        "backend_provenance": result["backend_provenance"],
        "compute_precision": result["compute_precision"],
        "dense_peaks_tflops": PEAKS,
        "peak_reference": analysis["peak_reference"],
        "wall_time_denominator": wall["wall_time_denominator"],
        "final_mfu": final,
        "query_coverage": coverage,
        "graph_ledger": graph_ledger,
        "source_sha256": result["source_sha256"],
        "current_recorded_local_native_artifacts": native,
        "boundaries": {
            "final_mfu": "100 * sum(useful matrix FLOPs / precision-specific dense peak) / independent synchronized clean wall time; host/control, nonmatrix, IO and gaps remain in the denominator.",
            "operator_mfu": "Each matrix API uses its exclusively attributed GPU kernel-duration sum. Fused prefetch and quantization remain in that API denominator. Nonmatrix FLOPs and MFU are null.",
            "prefill": "All prefix chunks and three layers, including embedding, final norm and last-token LM head; not the cropped final-chunk timeline window.",
            "extend": "Complete fixed-prefix extend with one graph replay; graph preparation and prefix restoration excluded from clean timing, input staging/begin/synchronization/commit included.",
            "scope": "Checkpoint layers L0-L2 on one GPU; no full 61-layer or serving extrapolation. One intrusive profile per method/phase; no NCU hardware utilization measurement.",
            "figures": "MFU figures only; existing final prefill/extend timelines are not regenerated.",
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "analysis.json", analysis)
    save_json(output / "summary.json", summary)
    save_csv(
        output / "final_mfu.csv",
        [{k: v for k, v in row.items() if k != "details"} for row in final],
    )
    save_csv(output / "operator_mfu.csv", analysis["operators"])
    save_csv(output / "operator_mfu_by_layer.csv", analysis["operators_by_layer"])
    save_csv(output / "query_coverage.csv", coverage)
    save_csv(
        output / "timing_samples.csv",
        [
            {"method": method, "phase": phase, "sample": index, "wall_ms": value}
            for method in METHODS
            for phase in ("prefill", "extend")
            for index, value in enumerate(wall["measurements"][method][phase + "_samples_ms"])
        ],
    )
    plot_mfu(final, analysis["operators"], output)
    save_json(
        output / "provenance.json",
        {
            "report_run_id": output.name,
            "directories": {name: str(path) for name, path in directories.items()},
            "inputs_sha256": before,
            "analysis_source_sha256": sources,
            "execution_source_sha256": result["source_sha256"],
            "source_boundary": "Analysis helpers are hashed at generation; execution snapshots remain bound to all three independent runs.",
            "argv": sys.argv,
        },
    )
    if before != {str(path): digest(path) for path in inputs} or sources != _source_hashes():
        raise RuntimeError("report input or analysis source changed during generation")
    save_json(
        output / "publication_manifest.json",
        {
            "schema": "full-graph-mfu-artifacts-v1",
            "files_sha256": {
                path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()
            },
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "output"
    )
    args = parser.parse_args()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id) is None:
        parser.error("--run-id must be one nonempty directory name")
    output = args.output_root / "data" / args.run_id
    generate(args.profile_run, output)
    print(output / "publication_manifest.json")


if __name__ == "__main__":
    main()
