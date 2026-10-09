"""Export checked Q1 hint NCU evidence without executing CUDA work."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import shlex
import shutil
import sys
from pathlib import Path

from experiments.deepseek_v32_echo_official.src import q1_hint_baseline as baseline

ROOT = baseline.ROOT
EXPERIMENT = baseline.EXPERIMENT
NCU_PYTHON = Path("/opt/nvidia/nsight-compute/2026.1.1/extras/python")
SASS_STRIDE = 16
SASS_LIMIT = 1 << 20
EXECUTED_METRICS = ("inst_executed", "thread_inst_executed", "thread_inst_executed_true")
require = baseline.require
digest = baseline.digest


def json_value(value):
    """Preserve nonfinite API values explicitly instead of emitting invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")


def record(path):
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def option(arguments, name):
    require(arguments.count(name) == 1, f"Expected one {name} in the saved command")
    return arguments[arguments.index(name) + 1]


def verify_run(directory, collection):
    require(directory.parent == EXPERIMENT / "output/data", "Expected a top-level data run")
    component = directory / "component"
    result_path = component / "result.json"
    result = json.loads(result_path.read_text())
    receipt = Path(result["receipt"]["path"]).resolve()
    checked = baseline.verify_profile_result(component, receipt, "layer_0")
    require(
        json.loads((directory / "completion.json").read_text()) == checked,
        "Saved child completion differs from the revalidated result",
    )
    sources = result["identity"]["source"]["sources"]
    for relative, expected in sources.items():
        require(
            digest(component / "source" / relative) == expected, "Profile source archive changed"
        )
    helper = Path(baseline.__file__).resolve()
    require(sources[str(helper.relative_to(ROOT))] == digest(helper), "Validation helper changed")
    script = directory / "profile_q1_hint.sh"
    require(
        digest(script)
        == sources["experiments/deepseek_v32_echo_official/scripts/profile_q1_hint.sh"],
        "Profile launcher archive changed",
    )
    arguments = shlex.split((directory / "command.txt").read_text())
    report = EXPERIMENT / "output/profile" / directory.name / f"{collection}.ncu-rep"
    expected_options = {
        "--set": collection,
        "--target-processes": "all",
        "--profile-from-start": "off",
        "--replay-mode": "kernel",
        "--cache-control": "all",
        "--clock-control": "base",
        "--nvtx-include": "q1_hint_baseline_layer_0/",
        "--kernel-name": "regex:.*reduce_kernel.*",
        "--launch-count": "1",
        "--layer": "layer_0",
    }
    for key, expected in expected_options.items():
        require(option(arguments, key) == expected, f"Profile command changed: {key}")
    for key, expected in (
        ("--receipt", receipt),
        ("--output-dir", component),
        ("--export", report.with_suffix("")),
    ):
        require((ROOT / option(arguments, key)).resolve() == expected, f"Wrong command path: {key}")
    sections = [arguments[i + 1] for i, value in enumerate(arguments) if value == "--section"]
    require(
        sections
        == (
            ["PmSampling", "PmSampling_WarpStates"] if collection == "full" else ["SourceCounters"]
        ),
        "Unexpected NCU sections",
    )
    logs = EXPERIMENT / "output/log" / directory.name
    stdout = (logs / "stdout.log").read_text()
    process_ids = re.findall(r"Connected to process (\d+)", stdout)
    require(len(process_ids) == 1, "Expected one recorded profile process")
    warnings = [line for line in stdout.splitlines() if "==WARNING==" in line]
    missing = []
    for warning in warnings:
        if "Unable to access the following" in warning:
            missing.extend(
                item.strip().rstrip(".") for item in warning.split(": ", 1)[1].split(",")
            )
    paths = [result_path, receipt, report]
    paths.extend(
        directory / name
        for name in (
            "command.txt",
            "completion.json",
            "ncu_version.txt",
            "details.txt",
            "profile_q1_hint.sh",
        )
    )
    paths.extend(logs / name for name in ("stdout.log", "stderr.log", "completion.stderr.log"))
    return result, {
        "collection": collection,
        "run_id": directory.name,
        "profile_report": str(report),
        "component_verification": checked,
        "process_id_from_collector_log": int(process_ids[0]),
        "collector_warnings": warnings,
        "collector_unavailable_metrics": missing,
        "inputs": [record(path) for path in paths],
    }


def metric_value(metric, index=None):
    args = () if index is None else (index,)
    return {
        "kind": metric.kind(*args),
        "has_value": metric.has_value(*args),
        "value": metric.value(index),
    }


def export_metrics(action):
    scalars, instances = {}, {}
    for name in sorted(action.metric_names()):
        metric = action[name]
        count = metric.num_instances()
        scalars[name] = {
            **metric_value(metric),
            "unit": metric.unit(),
            "description": metric.description(),
            "metric_type": metric.metric_type(),
            "metric_subtype": metric.metric_subtype(),
            "rollup_operation": metric.rollup_operation(),
            "num_instances": count,
            "has_correlation_ids": metric.has_correlation_ids(),
        }
        if count:
            correlation = metric.correlation_ids() if metric.has_correlation_ids() else None
            require(
                correlation is None or correlation.num_instances() == count,
                "Correlation count differs",
            )
            instances[name] = [
                {
                    "index": index,
                    **metric_value(metric, index),
                    "correlation": None
                    if correlation is None
                    else metric_value(correlation, index),
                }
                for index in range(count)
            ]
    return scalars, instances


def source_info(action, address):
    info = action.source_info(address)
    return None if info is None else {"file_name": info.file_name(), "line": info.line()}


def scan_sass(action, base, limit=SASS_LIMIT):
    rows = []
    for offset in range(0, limit, SASS_STRIDE):
        address = base + offset
        sass = action.sass_by_pc(address)
        if not sass:
            require(rows, "SASS is absent at the requested start address")
            return {
                "base_address": base,
                "stride_bytes": SASS_STRIDE,
                "limit_bytes": limit,
                "first_empty_address": address,
                "first_empty_offset": offset,
                "instructions": rows,
            }
        rows.append(
            {
                "address": address,
                "address_hex": hex(address),
                "offset": offset,
                "sass": sass,
                "source_info": source_info(action, address),
            }
        )
    raise RuntimeError("SASS scan did not terminate before its finite bound")


def pc_evidence(action, instances, base):
    names = [
        name
        for name in instances
        if name.startswith("smsp__pcsamp_warps_issue_stalled") or name in EXECUTED_METRICS
    ]
    require("inst_executed" in names, "Missing per-PC executed instruction counts")
    locations, metrics = {}, {}
    for name in names:
        rows = []
        for instance in instances[name]:
            correlation = instance["correlation"]
            require(correlation is not None, "Missing PC correlation")
            pc = correlation["value"]
            require(
                type(pc) is int and base <= pc < base + SASS_LIMIT,
                "PC outside the selected function bound",
            )
            require((pc - base) % SASS_STRIDE == 0, "Unaligned instruction PC")
            if pc not in locations:
                offset = pc - base
                absolute = action.sass_by_pc(pc)
                require(bool(absolute), "Per-PC metric has no absolute SASS")
                relative = action.sass_by_pc(offset)
                locations[pc] = {
                    "absolute_pc": pc,
                    "absolute_pc_hex": hex(pc),
                    "function_offset": offset,
                    "function_offset_hex": hex(offset),
                    "sass_absolute": absolute,
                    "source_info_absolute": source_info(action, pc),
                    "sass_relative_diagnostic": relative,
                    "source_info_relative_diagnostic": source_info(action, offset),
                    "absolute_relative_sass_equal": absolute == relative,
                }
            rows.append({**instance, "absolute_pc": pc, "function_offset": pc - base})
        metrics[name] = rows
    executed = metrics["inst_executed"]
    require(
        all(type(row["value"]) is int and row["value"] >= 0 for row in executed),
        "Invalid executed instruction counter",
    )
    return {
        "metrics": metrics,
        "locations": [locations[pc] for pc in sorted(locations)],
        "executed_metric_names": [name for name in names if name in EXECUTED_METRICS],
        "inst_executed_nonzero_rows": sum(row["value"] > 0 for row in executed),
        "inst_executed_instance_sum": sum(row["value"] for row in executed),
        "absolute_relative_sass_mismatches": sum(
            not row["absolute_relative_sass_equal"] for row in locations.values()
        ),
        "boundary": "Execution counts use absolute correlation PCs. Relative offsets and their differing API SASS lookups are diagnostic only. Static disassembly is not proof that an instruction executed; PC sampling and executed instruction counters are distinct collections.",
    }


def export_report(ncu, run):
    context = ncu.load_report(run["profile_report"])
    require(context.num_ranges() == 1, "Expected one NCU range")
    report_range = context.range_by_idx(0)
    require(report_range.num_actions() == 1, "Expected one selected reduction launch")
    require(
        tuple(report_range.actions_by_nvtx(["q1_hint_baseline_layer_0/"], [])) == (0,),
        "NCU action is outside the expected NVTX range",
    )
    action = report_range.action_by_idx(0)
    require(action.workload_type() == action.WorkloadType_KERNEL, "Expected a kernel action")
    name = action.name(action.NameBase_DEMANGLED)
    require("reduce_kernel<" in name, "Unexpected selected kernel")
    scalars, instances = export_metrics(action)
    base_rows = instances["launch__function_pcs"]
    require(
        len(base_rows) == 1 and type(base_rows[0]["value"]) is int,
        "Expected one absolute function base",
    )
    base = base_rows[0]["value"]
    launch = {
        "api_range_index": 0,
        "api_action_index": 0,
        "process_id_from_collector_log": run["process_id_from_collector_log"],
        "name_demangled": name,
        "name_mangled": action.name(action.NameBase_MANGLED),
        "name_function": action.name(action.NameBase_FUNCTION),
        "function_base_address": base,
        "geometry_and_identifiers": {
            key: value["value"] for key, value in scalars.items() if key.startswith("launch__")
        },
        "cuda_launch_correlation_id": None,
        "identifier_boundary": "NCU range/action indices and recorded context/device/stream IDs are retained. This public API does not expose a CUDA launch correlation ID; the process ID is from the archived collector log.",
    }
    for axis, block in zip("xyz", (512, 1, 1), strict=True):
        require(
            scalars[f"launch__block_dim_{axis}"]["value"] == block, "Unexpected reduction block"
        )
        require(scalars[f"launch__grid_dim_{axis}"]["value"] == 1, "Unexpected reduction grid")
    absolute = scan_sass(action, base)
    relative = scan_sass(action, 0)
    pc = pc_evidence(action, instances, base)
    absolute_addresses = {row["address"] for row in absolute["instructions"]}
    require(
        all(row["absolute_pc"] in absolute_addresses for row in pc["locations"]),
        "Per-PC data extends beyond the absolute disassembly",
    )
    rules = action.rule_results_as_dicts()
    pm = {name: rows for name, rows in instances.items() if name.startswith("pmsampling:")}
    require(bool(pm) == (run["collection"] == "full"), "Unexpected PM collection coverage")
    summary = {
        "scalar_metric_count": len(scalars),
        "instanced_metric_count": len(instances),
        "pm_metric_count": len(pm),
        "rule_count": len(rules),
        "pcsamp_sample_count": scalars["smsp__pcsamp_sample_count"]["value"],
        "inst_executed_nonzero_rows": pc["inst_executed_nonzero_rows"],
        "inst_executed_instance_sum": pc["inst_executed_instance_sum"],
        "absolute_sass_instruction_count": len(absolute["instructions"]),
        "relative_diagnostic_instruction_count": len(relative["instructions"]),
        "absolute_relative_sass_mismatches": pc["absolute_relative_sass_mismatches"],
        "source_correlated_pc_count": sum(
            row["source_info_absolute"] is not None for row in pc["locations"]
        ),
    }
    return (
        launch,
        summary,
        {
            "scalar_metrics.json": scalars,
            "metric_instances.json": instances,
            "pm_samples.json": pm,
            "rule_results.json": rules,
            "pc_metrics.json": pc,
            "sass_absolute.json": absolute,
            "sass_relative_diagnostic.json": relative,
            "launch.json": launch,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(
        output.parent == EXPERIMENT / "output/data", "Analysis must use a new experiment data run"
    )
    require(not output.exists(), "Analysis output already exists")
    results, runs = {}, {}
    for collection in ("full", "source"):
        directory = getattr(args, collection + "_run").resolve()
        results[collection], runs[collection] = verify_run(directory, collection)
    require(
        results["full"]["identity"] == results["source"]["identity"],
        "Full/source execution identities differ",
    )
    require(
        results["full"]["receipt"] == results["source"]["receipt"],
        "Full/source check receipts differ",
    )
    sys.path.insert(0, str(NCU_PYTHON))
    ncu = importlib.import_module("ncu_report")
    native = importlib.import_module("_ncu_report")
    require(Path(ncu.__file__).resolve() == NCU_PYTHON / "ncu_report.py", "Unexpected NCU parser")
    require(
        Path(native.__file__).resolve() == NCU_PYTHON / "_ncu_report.so",
        "Unexpected NCU parser binary",
    )
    artifacts, summaries, launches = {}, {}, {}
    for collection, run in runs.items():
        launches[collection], summaries[collection], artifacts[collection] = export_report(ncu, run)
    for key in ("name_demangled", "name_mangled", "name_function", "geometry_and_identifiers"):
        require(
            launches["full"][key] == launches["source"][key], f"Full/source launch differs: {key}"
        )
    output.mkdir(parents=True)
    for collection, files in artifacts.items():
        for name, value in files.items():
            write(output / collection / name, value)
    parser_sources = [
        Path(__file__).resolve(),
        Path(baseline.__file__).resolve(),
        ROOT / "evaluation/validation.py",
        Path(ncu.__file__).resolve(),
        Path(native.__file__).resolve(),
    ]
    archives = []
    for path in parser_sources:
        target = output / "analysis_sources" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
        require(digest(target) == digest(path), "Parser/helper archive differs")
        archives.append({**record(path), "copy": str(target.relative_to(output))})
    for collection, run in runs.items():
        for item in run["inputs"]:
            require(digest(Path(item["path"])) == item["sha256"], "Input changed while parsing")
            if Path(item["path"]).suffix != ".ncu-rep":
                target = (
                    output
                    / "inputs"
                    / collection
                    / (
                        "component_result.json"
                        if Path(item["path"]).name == "result.json"
                        else Path(item["path"]).name
                    )
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item["path"], target)
                item["copy"] = str(target.relative_to(output))
    exported = {
        str(path.relative_to(output)): record(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    write(
        output / "result.json",
        {
            "accepted": True,
            "schema": "q1-hint-ncu-evidence-v1",
            "run_id": output.name,
            "component_kind": baseline.KIND,
            "same_execution_identity_and_receipt": True,
            "runs": runs,
            "summary": summaries,
            "analysis_sources": archives,
            "exported_artifacts": exported,
            "boundary": "CPU extraction of two existing invasive NCU reports for one isolated real-input L0 PyTorch reduction, not a complete model or clean latency measurement. All metric values and API has_value flags are preserved without blanket validity claims; unavailable CTC warnings remain attached. Instance values/correlation IDs can exist while has_value is false. Source PC sampling is sparse (11 samples in the supplied source run), not a stable stall distribution. Per-PC execution counts and absolute function SASS identify observed instructions; differing relative SASS lookups are API diagnostics, not actual-function evidence. No GPU execution, production change or inference that every disassembled instruction executed.",
        },
    )
    print(output / "result.json")


if __name__ == "__main__":
    main()
