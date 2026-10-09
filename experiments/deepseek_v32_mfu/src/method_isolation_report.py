"""Audit and render a cohort whose four methods ran in separate processes.

Each trace keeps its original process, graph IDs, receipt and clean benchmark.
Only audited numerical rows and panel data are joined across children.
"""

from __future__ import annotations

import argparse
import datetime
import shlex
from collections import Counter
from pathlib import Path

from evaluation.validation import identity_digest, require_receipt
from experiments.deepseek_v32_mfu.src import audit_shape_matrix as audit
from experiments.deepseek_v32_mfu.src import compact_timeline as compact
from experiments.deepseek_v32_mfu.src import launch_gap
from experiments.deepseek_v32_mfu.src import report_full_graph_mfu as mfu
from experiments.deepseek_v32_mfu.src.method_process_audit import audit_capture_process
from experiments.deepseek_v32_mfu.src.operator_report import analyze_captures, read_calls
from experiments.deepseek_v32_mfu.src.report_mfu import (
    audit_local_native_artifacts,
    audit_work,
    plot_mfu,
    save_csv,
    save_json,
)
from experiments.deepseek_v32_mfu.src.run_contract import (
    METHODS,
    benchmark_view,
    common_workload_identity,
    execution_identity,
    receipt_kind,
    validate_completed_result,
)

CHILDREN = ("check", "bench", "profile", "operators")
KIND = "deepseek-v32-method-isolation"
CONFIGURATION = {
    "prefix_tokens": 65536,
    "extend_tokens": 1,
    "num_layers": 3,
    "chunk_size": 1024,
    "extend_chunk_size": 1,
    "sparse_pool_tokens": 65600,
    "host_arena_tokens": 65600,
    "workspace_query_tokens": 1024,
    "hbm_cache_budget_bytes": 24 * 2**30,
    "dram_cache_budget_bytes": 64 * 2**30,
    "extend_residency": "cold",
    "compute_graphs": True,
    "extend_graph": True,
    "warmups": 1,
    "prefill_repeats": 3,
    "repeats": 5,
    "seed": 42,
    "method_isolation": "fresh-process-one-method-v1",
}


def validate_configuration(manifest, results):
    expected = {
        **CONFIGURATION,
        "trace_warmups": 1,
        "physical_device": "0",
        "cpu_affinity": list(range(8)),
    }
    configuration = manifest["configuration"]
    audit.require(
        all(configuration.get(key) == value for key, value in expected.items()),
        "Cohort configuration differs from the declared fixed workload",
    )
    for method in METHODS:
        for phase, result in results[method].items():
            audit.require(
                all(result.get(key) == value for key, value in CONFIGURATION.items())
                and result["model"] == configuration["model"],
                "Child configuration differs from the cohort declaration",
            )
            if phase == "profile":
                audit.require(
                    result["measurement_identity"]["trace_warmups_per_capture"] == 1,
                    "Minimal profile trace warmup count differs",
                )


def manifest_children(manifest, evidence):
    """Validate manifest coverage and immutable child records before joining."""
    audit.require(
        manifest.get("schema_version") == 1 and manifest.get("kind") == KIND,
        "Unsupported method-isolation manifest",
    )
    children = manifest["methods"]
    audit.require(set(children) == set(METHODS), "Cohort must cover exactly four methods")
    directories, results, paths, ids, orders = {}, {}, set(), set(), []
    for method in METHODS:
        audit.require(set(children[method]) == set(CHILDREN), "Incomplete method child coverage")
        directories[method], results[method] = {}, {}
        for phase in CHILDREN:
            child = children[method][phase]
            directory = Path(child["directory"]).resolve(strict=True)
            run_id = child["run_id"]
            audit.require(directory not in paths and run_id not in ids, "Duplicate cohort child")
            paths.add(directory)
            ids.add(run_id)
            audit.require(directory.name == run_id, "Child directory differs from run ID")
            path = directory / "result.json"
            result = evidence.read(path)
            audit.require(evidence.digest(path) == child["result_sha256"], "Child result changed")
            audit.require(result["run_id"] == run_id, "Child result belongs to another run")
            audit.require(
                result.get("schema_version") == 4 and result.get("methods") == [method],
                "Child is not the selected isolated method",
            )
            detail = {
                "profile": "minimal_node_model_scopes",
                "operators": "matrix_api_node_ownership",
            }.get(phase)
            validate_completed_result(
                path,
                expected_mode="profile" if detail else phase,
                expected_profile_detail=detail,
            )
            runner = directory / "runner.json"
            audit.require(
                evidence.digest(runner) == child["runner_record_sha256"],
                "Child runner record changed",
            )
            evidence.read(runner)
            audit.require(
                isinstance(child["invocation"], list)
                and child["invocation"]
                and all(isinstance(value, str) for value in child["invocation"]),
                "Missing literal child invocation",
            )
            orders.append((child["order"], method, phase, run_id))
            directories[method][phase], results[method][phase] = directory, result
    audit.require(
        sorted(order for order, *_ in orders) == list(range(len(orders))),
        "Child execution order is incomplete or duplicated",
    )
    expected_order = [
        {"method": method, "phase": phase, "run_id": run_id}
        for _, method, phase, run_id in sorted(orders)
    ]
    audit.require(manifest["execution_order"] == expected_order, "Manifest execution order differs")
    audit.require(
        [(row["method"], row["phase"]) for row in expected_order]
        == [(method, "check") for method in METHODS]
        + [(method, phase) for method in METHODS for phase in CHILDREN[1:]],
        "All four checks must precede the ordered per-method measurements",
    )
    audit.require(manifest.get("completed_at_utc"), "Cohort has no completion record")
    return directories, results


def check_graph_acceptance(result):
    """Check the saved runtime graph/capacity contract without rerunning CUDA."""
    for bundle in result["extend_graph_checks"].values():
        runtime = bundle["runtime"]
        audit.require(
            runtime["allocated"]
            and runtime["graph_count"] == 1
            and (runtime["history_tokens"], runtime["query_tokens"])
            == (result["prefix_tokens"], result["extend_tokens"]),
            "Runtime graph shape/count differs",
        )
        audit.require(
            runtime["policy_revision"] == result["extend_graph_policy_revision"],
            "Runtime graph policy differs",
        )
        audit.require(
            0 < runtime["private_reserved_bytes"] <= runtime["chosen_private_limit_bytes"]
            and runtime["static_allocated_bytes"] + runtime["private_reserved_bytes"]
            <= runtime["reservation_bytes"],
            "Graph reservation exceeded",
        )
        for name, item in bundle["checks"].items():
            if name.endswith("_cache"):
                audit.require(item["equal"] and item["layers"] == 3, "Cache comparison failed")
            else:
                audit.require(
                    item["rtol"] == 0.01 and item["atol"] == 0.02,
                    "Numerical tolerance differs",
                )


def _single_option(arguments, option):
    """Read one literal option, rejecting overrides hidden later in an argv."""
    values, remaining = [], []
    cursor = 0
    while cursor < len(arguments):
        item = arguments[cursor]
        if item == option:
            audit.require(cursor + 1 < len(arguments), f"Missing value for recorded {option}")
            values.append(arguments[cursor + 1])
            cursor += 2
        elif item.startswith(option + "="):
            values.append(item[len(option) + 1 :])
            cursor += 1
        else:
            remaining.append(item)
            cursor += 1
    audit.require(len(values) == 1 and values[0], f"Missing or duplicate recorded {option}")
    return values[0], remaining


def validate_runner_command(runner, result, *, method, phase, root):
    """Bind the phase's exact wrapper transformation to its recorded child argv."""
    wrapper, module, temporary_prefix, trace_name = {
        "check": ("run.sh", "measure", "deepseek-echo", None),
        "bench": ("run.sh", "measure", "deepseek-echo", None),
        "profile": ("gap_profile.sh", "gap_profile", "deepseek-minimal-node", "minimal"),
        "operators": ("profile_layers.sh", "profile_layers", "deepseek-layers3", "layers3"),
    }[phase]
    source = "experiments/deepseek_v32_mfu/scripts/" + wrapper
    audit.require(
        runner.get("schema_version") == 1
        and runner.get("source") == source
        and runner.get("helper_source") == "experiments/deepseek_v32_mfu/scripts/runner_common.sh",
        "Recorded runner/helper does not match this child role",
    )
    invocation, command = runner.get("invocation"), runner.get("command")
    audit.require(
        all(
            isinstance(values, list) and values and all(isinstance(value, str) for value in values)
            for values in (invocation, command)
        ),
        "Runner requires literal invocation and command argument lists",
    )
    audit.require(
        Path(invocation[0]).is_absolute() and Path(invocation[0]).resolve() == root / source,
        "Runner invocation names a different wrapper",
    )
    selected, _ = _single_option(invocation[1:], "--method")
    audit.require(selected == method, "Runner invocation selects a different method")
    forwarded = invocation[1:]
    if phase in ("check", "bench"):
        mode, forwarded = _single_option(forwarded, "--mode")
        audit.require(mode == phase, "Runner invocation selects a different execution mode")
    else:
        audit.require(
            not any(value.split("=", 1)[0] == "--mode" for value in forwarded),
            "Profile invocation cannot override its execution mode",
        )
    audit.require(
        not any(
            value.split("=", 1)[0] in {"--run-id", "--output", "--nsys"} for value in forwarded
        ),
        "Runner invocation overrides a wrapper-owned execution option",
    )
    module = "experiments.deepseek_v32_mfu.src." + module
    if trace_name is None:
        audit.require(len(command) >= 9, "Incomplete recorded Python command")
        output = Path(command[8])
        expected = [
            "python",
            "-m",
            module,
            "--mode",
            phase,
            "--run-id",
            result["run_id"],
            "--output",
            str(output),
            *forwarded,
        ]
    else:
        audit.require(len(command) >= 18, "Incomplete recorded NSYS command")
        output = Path(command[16])
        expected = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=repeat",
            "--output",
            str(output.parent / "profile" / trace_name),
            "python",
            "-m",
            module,
            "--run-id",
            result["run_id"],
            "--output",
            str(output),
            "--nsys",
            *forwarded,
        ]
    audit.require(
        output.is_absolute()
        and output.name == "data"
        and output.parent.name.startswith(f"{temporary_prefix}-{result['run_id']}."),
        "Recorded command output is not this child's temporary run directory",
    )
    audit.require(command == expected, "Recorded child command differs from its wrapper invocation")
    audit.require(
        runner.get("command_text") == shlex.join(command),
        "Recorded command text differs from its literal arguments",
    )


def audit_runner_records(manifest, directories, results, evidence):
    """Check saved process and invocation records, not a claim of machine isolation."""
    root = Path(__file__).resolve().parents[3]
    runner_files = {
        "experiments/deepseek_v32_mfu/scripts/" + name
        for name in (
            "method_isolation.sh",
            "runner_common.sh",
            "run.sh",
            "gap_profile.sh",
            "profile_layers.sh",
        )
    }
    audit.require(
        set(manifest["runner"]["sources"]) == runner_files,
        "Incomplete cohort runner source inventory",
    )
    for source, expected in manifest["runner"]["sources"].items():
        audit.require(evidence.digest(root / source) == expected, "Cohort runner source changed")
    request = manifest["request"]
    audit.require(evidence.digest(request["path"]) == request["sha256"], "Cohort request changed")
    processes, process_ids, previous_end = [], set(), None
    for item in manifest["execution_order"]:
        method, phase = item["method"], item["phase"]
        child = manifest["methods"][method][phase]
        directory, result = directories[method][phase], results[method][phase]
        runner = evidence.read(directory / "runner.json")
        validate_runner_command(runner, result, method=method, phase=phase, root=root)
        audit.require(
            result["request_sha256"] == request["sha256"], "Child request differs from cohort"
        )
        invocation = child["invocation"]
        audit.require(
            invocation[:6]
            == ["env", "MFU_RUN_ID=" + result["run_id"], "taskset", "-c", "0-7", "bash"]
            and runner["invocation"] == invocation[6:]
            and runner["run_id"] == result["run_id"],
            "Recorded child command differs from the invoked isolated runner",
        )
        for field, sha_field in (("source", "sha256"), ("helper_source", "helper_sha256")):
            audit.require(
                evidence.digest(root / runner[field]) == runner[sha_field],
                "Child runner source changed",
            )
        audit.require(
            runner["target_process"] == result["process_provenance"]
            and runner["cpu_affinity"] == list(range(8))
            and runner["environment"]["CUDA_VISIBLE_DEVICES"] == "0",
            "Child runner and target process provenance differ",
        )
        process = runner["target_process"]
        identity = (runner["shell_process"]["boot_id"], process["pid"], process["start_ticks"])
        audit.require(identity not in process_ids, "Methods reused one execution process")
        process_ids.add(identity)
        start = datetime.datetime.fromisoformat(child["started_at_utc"])
        end = datetime.datetime.fromisoformat(child["completed_at_utc"])
        audit.require(
            start < end and (previous_end is None or previous_end <= start),
            "Children overlap or have invalid execution times",
        )
        previous_end = end
        audit.require(
            set(child["files"]) == {"data", "log", "profile"},
            "Incomplete child artifact categories",
        )
        for category, entries in child["files"].items():
            audit.require(category in {"data", "log", "profile"}, "Unknown child output category")
            base = directory.parent.parent / category / result["run_id"]
            actual = {str(path.relative_to(base)) for path in base.rglob("*") if path.is_file()}
            audit.require(actual == set(entries), "Child output inventory changed")
            for relative, expected in entries.items():
                path = (base / relative).resolve(strict=True)
                audit.require(
                    path.is_relative_to(base.resolve()), "Child artifact escapes its directory"
                )
                audit.require(evidence.digest(path) == expected, "Child artifact changed")
        processes.append({"method": method, "phase": phase, **process})
    return processes


def audit_method(method, directories, results, hbm_check, hbm_receipt_path, evidence):
    check = results["check"]
    receipt_path = directories["check"] / "receipt.json"
    receipt = require_receipt(
        receipt_path, kind=receipt_kind(check), identity=check["execution_identity"]
    )
    evidence.digest(receipt_path)
    if method != "hbm":
        reference = receipt["checks"]["hbm_reference"]
        hbm_receipt = require_receipt(
            hbm_receipt_path, kind=receipt_kind(hbm_check), identity=hbm_check["execution_identity"]
        )
        evidence.digest(hbm_receipt_path)
        audit.require(
            Path(reference["receipt_path"]).resolve() == hbm_receipt_path
            and reference["receipt_sha256"] == hbm_receipt["receipt_sha256"],
            "Offload reference does not bind this cohort's HBM check",
        )
    audit.require(
        receipt["checks"]["comparisons"] == check["correctness"]
        and receipt["checks"]["extend_graph"] == check["extend_graph_checks"],
        "Receipt differs from saved check results",
    )
    for item in receipt["artifacts"].values():
        audit.require(
            evidence.digest(directories["check"] / item["path"]) == item["sha256"],
            "Numerical artifact changed",
        )
    for phase, result in results.items():
        audit.require(
            result["execution_identity"]
            == execution_identity(result)
            == check["execution_identity"],
            "Method check/bench/profile identities differ",
        )
        audit.require(
            common_workload_identity(result) == common_workload_identity(hbm_check),
            "Cohort methods have different source, inputs or execution configuration",
        )
        audit.require(
            result["num_layers"] == 3
            and (result["prefix_tokens"], result["extend_tokens"], result["chunk_size"])
            == (65536, 1, 1024)
            and result["extend_chunk_size"] == 1
            and result["slots"]
            == result["sparse_pool_tokens"]
            == result["host_arena_tokens"]
            == 65600
            and result["compute_graphs"] is True
            and result["extend_graph"] is True
            and result["extend_residency"] == "cold",
            "This cohort requires the declared H64K/A1 cold full-graph workload",
        )
        mfu.audit_hardware(result["hardware"])
        audit.require(
            evidence.digest(directories[phase] / "request.json") == result["request_sha256"],
            "Child request changed",
        )
        if phase != "check":
            benchmark_view(directories[phase], result)
            audit.require(
                Path(result["validation_receipt"]["receipt_path"]).resolve() == receipt_path,
                "Child binds another method's check",
            )
        if phase in ("profile", "operators"):
            audit.require(
                Path(result["benchmark"]["directory"]).resolve() == directories["bench"],
                "Profile binds another benchmark",
            )
    check_graph_acceptance(check)
    sources = {
        phase: audit.source_audit(directories[phase], result, evidence)
        for phase, result in results.items()
    }
    runtimes = {
        phase: audit.runtime_audit(result, evidence, require_offload=method != "hbm")
        for phase, result in results.items()
    }
    native = {}
    for phase in ("profile", "operators"):
        native[phase] = audit_local_native_artifacts(
            {"check": check, "bench": results["bench"], "profile": results[phase]},
            require_offload=method != "hbm",
        )
    timings = []
    bench = results["bench"]
    for phase in ("prefill", "extend"):
        timings.append(
            {
                "method": method,
                "phase": phase,
                "benchmark_run_id": bench["run_id"],
                "samples_ms": bench["measurements"][method][phase + "_samples_ms"],
                "median_ms": bench["measurements"][method][phase + "_median_ms"],
                "cache_samples": audit.cache_sample_audit(bench, method, phase),
            }
        )
    return {
        "receipt_sha256": evidence.digest(receipt_path),
        "execution_identity_sha256": identity_digest(check["execution_identity"]),
        "sources": sources,
        "runtimes": runtimes,
        "native_artifacts": native,
        "timing": timings,
        "bounded_prefetch": (
            audit.prefetch_audit(directories["check"], check, receipt, evidence)
            if method == "echo"
            else None
        ),
    }


def matrix_work(calls):
    """Compare semantic useful work without comparing process-local graph IDs."""
    work = Counter()
    for call in calls:
        if call["useful_flops"] is not None:
            stage = "indexer" if call["stage"] in {"indexer_qk", "indexer_fused"} else call["stage"]
            work[call["phase"], call["layer"], stage, call["precision"]] += call["useful_flops"]
    return work


def operator_evidence(method, directory, result, evidence):
    ledger = directory / "operator_calls.json"
    evidence.digest(ledger)
    calls, metadata = read_calls(ledger)
    audit.require(metadata["run_id"] == result["run_id"], "Operator ledger run differs")
    setup, paths = mfu.capture_paths(directory, result, methods=(method,))
    for path in (*setup, *paths):
        evidence.digest(path)
    processes = [
        audit_capture_process(
            path,
            target_pid=result["process_provenance"]["pid"],
            method=method,
            phase=phase,
            require_single_graph=phase == "extend",
        )
        for path, phase in zip(paths, ("prefill", "extend"), strict=True)
    ]
    graph = mfu.audit_graph_ledgers(
        result,
        calls,
        evidence.read(directory / "full_graph_templates.json"),
        evidence.read(directory / "graph_templates.json"),
        methods=(method,),
    )
    analysis = analyze_captures(
        paths, calls, metadata=metadata, graph_setup_paths=setup, peaks=mfu.PEAKS
    )
    mfu.audit_analysis(result, analysis, graph, methods=(method,))
    graph["capture_processes"] = processes
    rows, coverage = audit_work(
        benchmark_view(directory, result), calls, mfu.PEAKS, methods=(method,)
    )
    for row in rows:
        row.update(profile_run_id=result["run_id"], benchmark_run_id=result["benchmark"]["run_id"])
    return analysis, graph, rows, coverage, matrix_work(calls)


def timeline_evidence(method, directory, result, evidence):
    stored = evidence.read(directory / "gap_audit.json")
    fresh = launch_gap.analyze_run(directory)
    audit.require(stored == fresh, "Stored gap analysis differs from a fresh raw-trace reread")
    for index in range(1, len(result["nsys_capture_order"]) + 1):
        evidence.digest(directory / f"capture_{index}.sqlite")
    _, panels = compact.extract_profile_panels(directory, fresh, methods=(method,))
    cropped, raw_audits = {}, {}
    for phase in ("prefill", "extend"):
        panel = compact.three_layer_panel(panels[phase][0], phase)
        index = result["nsys_capture_order"].index(f"{method}/{phase}_annotated") + 1
        sqlite = directory / f"capture_{index}.sqlite"
        native = audit.read_native(sqlite, evidence)
        raw_audits[phase] = audit.window_inventory(native, panel)
        raw_audits[phase]["processes"] = audit_capture_process(
            sqlite,
            target_pid=result["process_provenance"]["pid"],
            method=method,
            phase=phase,
            require_single_graph=phase == "extend",
        )
        if phase == "extend":
            raw_audits["graph"] = audit.graph_inventory(directory, result, method, native, evidence)
        panel["provenance"] = {
            "profile_run_id": result["run_id"],
            "profile_directory": str(directory),
            "sqlite": str(sqlite),
            "sqlite_sha256": evidence.digest(sqlite),
            "result_sha256": evidence.digest(directory / "result.json"),
            "benchmark": result["benchmark"],
            "validation_receipt": result["validation_receipt"],
        }
        cropped[phase] = panel
        if phase == "extend":
            startup = compact.extend_startup_panel(panels[phase][0])
            startup["provenance"] = panel["provenance"]
            raw_audits["startup"] = audit.window_inventory(native, startup)
            cropped["startup"] = startup
    return cropped, raw_audits, fresh["prefill_methods"][0]


def report_text(summary, panels):
    lookup = {(row["method"], row["phase"]): row for row in summary["final_mfu"]}
    windows = {panel["method"]: panel["window"] for panel in panels["extend"]}
    lines = [
        "# H64K + A1 with independent method processes",
        "",
        (
            "Each method has a separate numerical check, clean benchmark, minimal node profile "
            "and operator profile. Every process performs only its selected method's normal "
            "warmup. HBM outputs are loaded on CPU as the offload correctness reference."
        ),
        "",
        (
            "The workload uses checkpoint layers L0–L2, H=65,536, A=1, history chunk=1,024, "
            "P=NH=65,600, FP8 weights and BF16 main KV on H200 SXM / SM90, GPU0, CPU0–7. "
            "Offload restores a cold prefix before every step. HBM keeps resident KV. "
            "One warmup precedes three prefill and five step wall samples per method."
        ),
        "",
        "| Method | Prefill wall ms | Step wall ms | Step MFU % | L0–L2 GPU ms | GPU idle µs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        prefill, extend = lookup[method, "prefill"], lookup[method, "extend"]
        window = windows[method]
        lines.append(
            f"| `{method}` | {prefill['wall_median_ms']:.6f} | {extend['wall_median_ms']:.6f} "
            f"| {extend['mfu_percent']:.6f} | {window['window_ms']:.6f} "
            f"| {window['gpu_idle_ms'] * 1000:.3f} |"
        )
    lines.extend(
        [
            "",
            "![Three-layer single-token timeline](extend.svg)",
            "",
            (
                "Wall medians cover complete synchronized execution, including required input, "
                "transaction and commit work. Loading, graph preparation and prefix restoration "
                "remain outside timing. The intrusive GPU timeline runs separately and covers "
                "L0's first compute through L2's last compute; dense includes L0 history H2D. "
                "The full graph also contains embedding, final norm and the last-token LM head."
            ),
            "",
            (
                "MFU is 100 × the sum of useful matrix FLOPs divided by each precision's nominal "
                "dense peak, divided by the independent complete wall time. It is not measured "
                "Tensor Core occupancy. Nominal FP8/BF16/FP32 peaks are 1979/989.5/67 TFLOP/s."
            ),
            "",
            (
                "13 standard output comparisons and 44 full-graph checks cover all four methods. "
                "The report independently rereads saved outputs, bounded ECHO transitions, actual "
                "traffic, original source/native artifacts, and raw GPU process/graph/window "
                "inventories. Unsaved scores and KV payloads retain their runtime correctness "
                "boundary. Process observations do not prove continuous machine isolation."
            ),
            "",
            (
                "Fresh method processes remove another method's same-process preparation from "
                "the measurement boundary. This is not a change to model computation or evidence "
                "of a particular hardware mechanism. Different-batch wall times are not paired "
                "speedup estimates. The official SGLang comparison retains different inputs, "
                "natural residency and framework behavior."
            ),
            "",
            (
                f"Cohort: `{summary['run_id']}`. [Summary and child identities](summary.json), "
                "[all wall samples](timing_samples.csv), [raw-window summaries](windows.csv), "
                "[complete-stage MFU](final_mfu.csv), [operator MFU](operator_mfu.csv), "
                "[input hashes](input_hashes.json), [publication manifest](publication_manifest.json)."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def generate(manifest_path, output):
    output = Path(output).resolve()
    audit.require(not output.exists(), "Report destination already exists")
    evidence = audit.Evidence()
    analysis_sources = {
        str(path): evidence.digest(path)
        for path in (
            Path(__file__).resolve(),
            *[
                Path(__file__).with_name(name).resolve()
                for name in (
                    "audit_shape_matrix.py",
                    "compact_timeline.py",
                    "launch_gap.py",
                    "report_full_graph_mfu.py",
                    "report_mfu.py",
                    "run_contract.py",
                    "operator_report.py",
                    "method_process_audit.py",
                    "execution_utilization.py",
                    "shape_matrix_sources.py",
                    "redraw_gap_timeline.py",
                    "full_graph_profile.py",
                    "analyze_nsys.py",
                    "timeline.py",
                )
            ],
        )
    }
    manifest = evidence.read(manifest_path)
    directories, results = manifest_children(manifest, evidence)
    validate_configuration(manifest, results)
    processes = audit_runner_records(manifest, directories, results, evidence)
    checks = {method: directories[method]["check"] for method in METHODS}
    method_audits, operator_analyses, graph_audits = {}, {}, {}
    final, coverage, operators, by_layer = [], [], [], []
    panels, raw_windows, reference_work = {"prefill": [], "extend": []}, {}, None
    prefill_gaps = []
    startup_panels = []
    for method in METHODS:
        method_audits[method] = audit_method(
            method,
            directories[method],
            results[method],
            results["hbm"]["check"],
            directories["hbm"]["check"] / "receipt.json",
            evidence,
        )
        analysis, graph, rows, query_rows, work = operator_evidence(
            method, directories[method]["operators"], results[method]["operators"], evidence
        )
        if reference_work is None:
            reference_work = work
        audit.require(work == reference_work, "Cohort useful matrix work differs across methods")
        operator_analyses[method], graph_audits[method] = analysis, graph
        final.extend(rows)
        coverage.extend(query_rows)
        for key, target in (("operators", operators), ("operators_by_layer", by_layer)):
            target.extend(
                {**row, "profile_run_id": results[method]["operators"]["run_id"]}
                for row in analysis[key]
            )
        cropped, raw_windows[method], prefill_gap = timeline_evidence(
            method, directories[method]["profile"], results[method]["profile"], evidence
        )
        prefill_gaps.append(prefill_gap)
        startup_panels.append(cropped["startup"])
        for phase, phase_panels in panels.items():
            phase_panels.append(cropped[phase])
    launch_gap.compare_prefill_gaps(prefill_gaps)
    tensors = {
        phase: audit.saved_tensors(
            checks,
            {method: directories[method][phase] for method in METHODS},
            1,
            evidence,
        )
        for phase in ("profile", "operators")
    }
    evidence.verify()
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": "deepseek-v32-isolated-method-report-v1",
        "run_id": manifest["run_id"],
        "passed": True,
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": evidence.digest(manifest_path),
        "configuration": manifest["configuration"],
        "children": {
            method: {
                phase: {
                    key: value
                    for key, value in child.items()
                    if key
                    in {
                        "run_id",
                        "directory",
                        "result_sha256",
                        "runner_record_sha256",
                        "receipt_path",
                        "receipt_sha256",
                    }
                }
                for phase, child in manifest["methods"][method].items()
            }
            for method in METHODS
        },
        "processes": processes,
        "methods": method_audits,
        "tensors": tensors,
        "raw_windows": raw_windows,
        "prefill_gap_comparison": prefill_gaps,
        "operator_graph_audits": graph_audits,
        "final_mfu": final,
        "analysis_source_sha256": analysis_sources,
        "boundary": "Fresh process per method and per check/bench/profile role, with each method's own normal warmup. Shared disk JIT caches and device are retained. No claim of cleared device state or continuous machine isolation. GPU layer windows and clean synchronized full-step wall samples are distinct metrics.",
    }
    save_json(output / "summary.json", summary)
    save_json(output / "operator_analysis.json", operator_analyses)
    save_json(output / "window_rows.json", panels)
    (output / "results.md").write_text(report_text(summary, panels))
    save_csv(
        output / "final_mfu.csv",
        [{k: v for k, v in row.items() if k != "details"} for row in final],
    )
    save_csv(output / "operator_mfu.csv", operators)
    save_csv(output / "operator_mfu_by_layer.csv", by_layer)
    save_csv(output / "query_coverage.csv", coverage)
    save_csv(
        output / "windows.csv",
        [
            {"phase": phase, "method": panel["method"], **panel["window"], **panel["provenance"]}
            for phase, phase_panels in panels.items()
            for panel in phase_panels
        ],
    )
    save_csv(
        output / "timing_samples.csv",
        [
            {
                "method": method,
                "phase": row["phase"],
                "sample": index,
                "wall_ms": value,
                "benchmark_run_id": row["benchmark_run_id"],
            }
            for method in METHODS
            for row in method_audits[method]["timing"]
            for index, value in enumerate(row["samples_ms"])
        ],
    )
    plot_mfu(final, operators, output)
    compact.draw(
        panels["prefill"],
        panels["extend"],
        output,
        layout="separate",
        window_kind="three-layers",
        io_layout="directions",
        annotations="idle-echo",
    )
    startup_output = output / "with_startup"
    startup_output.mkdir()
    save_json(startup_output / "window_rows.json", {"extend": startup_panels})
    compact.draw(
        panels["prefill"],
        startup_panels,
        startup_output,
        layout="separate",
        window_kind="extend-startup",
        io_layout="directions",
        annotations="idle-echo",
    )
    save_json(output / "input_hashes.json", evidence.hashes)
    evidence.verify()
    save_json(
        output / "publication_manifest.json",
        {
            "schema": "isolated-method-artifacts-v1",
            "files_sha256": {
                str(path.relative_to(output)): audit.Evidence().digest(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.manifest, args.output_dir)
    print(args.output_dir / "publication_manifest.json")


if __name__ == "__main__":
    main()
