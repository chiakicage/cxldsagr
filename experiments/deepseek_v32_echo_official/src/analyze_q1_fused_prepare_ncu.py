"""Extract bounded preparation NCU evidence, preserving metric validity flags."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from experiments.deepseek_v32_echo_official.src import analyze_q1_fused_prepare_model as model
from experiments.deepseek_v32_echo_official.src import analyze_q1_fused_prepare_profile as profile

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
require = model.require
SELECTED_METRICS = (
    "gpu__time_duration.sum",
    "sm__cycles_active.sum",
    "sm__cycles_active.max",
    "sm__ctas_launched.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
)
PACKING_SOURCE = "operators/deepseek_v32/indexer/page64.py"
PACKING_DRIVER = "experiments/deepseek_v32_echo_official/src/q1_packing.py"
PACKING_INPUT_DRIVER = "experiments/deepseek_v32_echo_official/src/q1_indexer.py"


def packing_bridge(completed, identity):
    """Bind earlier isolated packing to the current accepted specialization/input."""
    historical = completed["identity"]
    require(
        completed["mode"] == "profile"
        and completed["profile"] == {"method": "fused", "output_shape": [1025, 8448]},
        "Historical profile is not the complete isolated packing call",
    )
    require(
        completed["compiled_kernels"] == identity["packing_kernels"],
        "Historical packing specialization or assembly differs",
    )
    first_input = min(identity["inputs"])
    require(
        historical["inputs_sha256"] == {first_input: identity["inputs"][first_input]},
        "Historical packing is not the exact saved L0 input",
    )
    require(
        all(
            historical[key] == identity[key]
            for key in ("torch", "cuda", "triton", "device", "capability")
        )
        and historical["precision"] == {"matmul_tf32": False}
        and identity["precision"] == {"tf32": False},
        "Historical packing runtime/precision/architecture differs",
    )


def python_ast(path):
    return ast.dump(ast.parse(Path(path).read_text()), include_attributes=False)


def input_assignments(path):
    tree = ast.parse(Path(path).read_text())
    case = next(
        item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "Case"
    )
    init = next(
        item for item in case.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"
    )
    return {
        target.attr: ast.unparse(node.value)
        for node in init.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute) and target.attr in ("k", "scales")
    }


def normalize_sass(text, base=0):
    # NCU renders absolute branch targets and omits cuobjdump's register-reuse annotation.
    value = " ".join(text.strip().rstrip(";").replace(".reuse", "").split())
    if base and re.search(r"\b(?:BRA|BSSY)\b", value):
        value = re.sub(r"0x[0-9a-f]+", lambda match: hex(int(match[0], 16) - base), value)
    return value


def sass_binding(action, cubin, evidence, pending, tag):
    executable = Path("/usr/local/cuda/bin/cuobjdump")
    command = [str(executable), "--dump-sass", str(cubin)]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    require(
        result.stdout.count("Function :") == 1 and "Function : pack_page64" in result.stdout,
        "Unexpected packing CUBIN functions",
    )
    assembly = {
        int(offset, 16): normalize_sass(instruction)
        for offset, instruction in re.findall(r"/\*([0-9a-f]+)\*/\s+(.*?)\s*;", result.stdout)
    }
    metric = action["inst_executed"]
    require(metric.has_correlation_ids(), "Packing report has no instruction PCs")
    correlations = metric.correlation_ids()
    pcs = sorted(correlations.value(index) for index in range(correlations.num_instances()))
    require(
        len(pcs) == 64 and pcs == list(range(pcs[0], pcs[0] + 1024, 16)),
        "Packing instruction coverage differs",
    )
    for pc in pcs:
        require(
            normalize_sass(action.sass_by_pc(pc), pcs[0]) == assembly.get(pc - pcs[0]),
            "NCU instruction differs from the accepted packing CUBIN",
        )
    pending[tag + ".cubin_disassembly.json"] = {
        "argv": command,
        "tool": evidence.file(executable),
        "cubin": evidence.file(cubin),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    return {
        "matched_instruction_rows": len(pcs),
        "cubin_instruction_rows_including_padding": len(assembly),
        "normalization": "Whitespace/semicolon and omitted register-reuse annotations; absolute branch targets rebased to the recorded function entry. This corroborates instructions, not a byte comparison of the profiled code object. Exact assembly hashes are independently bound by both producers.",
    }


def metric_value(metric, index=None):
    arguments = () if index is None else (index,)
    value = metric.value(index)
    result = {"has_value": metric.has_value(*arguments), "value": value}
    if isinstance(value, float) and not math.isfinite(value):
        result.update(value=None, nonfinite_value=repr(value))
    return result


def extract(action):
    scalars, instances, locations = {}, {}, defaultdict(dict)
    for name in sorted(action.metric_names()):
        metric = action[name]
        scalars[name] = {
            **metric_value(metric),
            "unit": metric.unit(),
            "description": metric.description(),
            "instances": metric.num_instances(),
        }
        if not metric.num_instances():
            continue
        correlation = metric.correlation_ids() if metric.has_correlation_ids() else None
        require(
            correlation is None or correlation.num_instances() == metric.num_instances(),
            "Metric correlation length differs",
        )
        rows = []
        for index in range(metric.num_instances()):
            row = metric_value(metric, index)
            row["correlation"] = None if correlation is None else correlation.value(index)
            rows.append(row)
            if (
                correlation is not None
                and row["value"]
                and (
                    name.startswith("smsp__pcsamp_warps_issue_stalled")
                    or name
                    in ("inst_executed", "thread_inst_executed", "thread_inst_executed_true")
                )
            ):
                locations[row["correlation"]][name] = row
        instances[name] = rows
    pcs = []
    for pc, counters in sorted(locations.items()):
        source = action.source_info(pc)
        pcs.append(
            {
                "absolute_pc": pc,
                "sass": action.sass_by_pc(pc),
                "source": None
                if source is None
                else {"file": source.file_name(), "line": source.line()},
                "counters": counters,
            }
        )
    return scalars, instances, pcs


def action_evidence(action, row, tag, evidence, pending, *, required=SELECTED_METRICS):
    kernel_name = action.name(action.NameBase_DEMANGLED)
    signature = row["signature"]
    require(kernel_name == signature["name"], "NCU and native-owned NSYS kernel names differ")
    scalars, instances, pcs = extract(action)
    geometry = {
        "launch__grid_size": signature["gridX"] * signature["gridY"] * signature["gridZ"],
        "launch__block_size": signature["blockX"] * signature["blockY"] * signature["blockZ"],
        "launch__registers_per_thread": signature["registersPerThread"],
    }
    for metric, value in geometry.items():
        require(
            scalars[metric]["has_value"] is True and scalars[metric]["value"] == value,
            "NCU launch geometry differs from the bound native-owned preparation",
        )
    source_files = {item["source"]["file"] for item in pcs if item["source"]}
    embedded = dict(action.source_files())
    source_records = []
    for name in sorted(source_files | set(embedded)):
        path = Path(name)
        source_records.append(
            {
                "recorded_path": name,
                "current_file": evidence.file(path) if path.is_file() else None,
                "embedded_sha256": hashlib.sha256(embedded[name].encode()).hexdigest()
                if name in embedded
                else None,
            }
        )
    pending[tag + ".metrics.json"] = scalars
    pending[tag + ".instances.json"] = instances
    pending[tag + ".pc_evidence.json"] = pcs
    pending[tag + ".rules.json"] = action.rule_results_as_dicts()
    pending[tag + ".source_markers.json"] = action.source_markers()
    pending[tag + ".embedded_sources.json"] = embedded
    selected = {name: scalars.get(name) for name in SELECTED_METRICS}
    require(
        all(selected[name] is not None and selected[name]["has_value"] for name in required),
        "A required aggregate or launch metric is unavailable",
    )
    return {
        "tag": tag,
        "kernel": kernel_name,
        "operation": row["operation"],
        "native_owner": row["owner"],
        "native_signature": signature,
        "metrics": selected,
        "source_files": source_records,
        "pc_locations": len(pcs),
        "invalid_nonzero_correlated_values": sum(
            not value["has_value"] and bool(value["value"])
            for item in pcs
            for value in item["counters"].values()
        ),
    }


def log_evidence(run_dir, tag, evidence):
    result = {}
    for stream in ("stdout", "stderr"):
        path = EXPERIMENT / "output/log" / run_dir.name / (tag + "." + stream + ".log")
        result[stream] = {
            "file": evidence.file(path),
            "warnings": [
                line
                for line in path.read_text().splitlines()
                if any(word in line for word in ("WARNING", "WARN", "ERR", "Warning"))
            ],
        }
    return result


def historical_packing(run_dir, identity, row, evidence, pending, ncu_report, archive):
    summaries = {}
    for tag in ("fused_full", "fused_source"):
        directory = run_dir / tag
        completed = evidence.read(directory / "result.json")
        packing_bridge(completed, identity)
        historical = completed["identity"]
        receipt_record = completed["receipt"]
        evidence.file(receipt_record["path"], receipt_record["sha256"])
        receipt = evidence.read(receipt_record["path"])
        require(receipt["accepted"] is True, "Historical packing acceptance is missing")
        old_identity = {
            key: value for key, value in receipt["identity"].items() if key != "inputs_sha256"
        }
        actual_identity = {
            key: value for key, value in historical.items() if key != "inputs_sha256"
        }
        require(old_identity == actual_identity, "Historical packing receipt identity differs")
        require(
            all(
                receipt["identity"]["inputs_sha256"].get(path) == digest
                for path, digest in historical["inputs_sha256"].items()
            ),
            "Historical packing input is outside its acceptance",
        )
        require(
            all(kernel in receipt["compiled_kernels"] for kernel in completed["compiled_kernels"]),
            "Historical packing specialization is outside its acceptance",
        )
        for name, digest in historical["source_sha256"].items():
            evidence.file(directory / "source" / model.safe_relative(name), digest)
        for name in (PACKING_SOURCE, PACKING_INPUT_DRIVER):
            evidence.file(ROOT / name, historical["source_sha256"][name])
        old_driver = directory / "source" / PACKING_DRIVER
        require(
            python_ast(old_driver) == python_ast(ROOT / PACKING_DRIVER),
            "Historical packing driver differs beyond formatting",
        )
        inputs = input_assignments(directory / "source" / PACKING_INPUT_DRIVER)
        require(
            inputs
            == input_assignments(EXPERIMENT / "src/q1_official_prefetch.py")
            == {"k": "data['index_keys'].cuda()", "scales": "data['index_scales'].cuda()"},
            "Historical and accepted input copies differ",
        )
        kernel = completed["compiled_kernels"][0]
        require(len(completed["compiled_kernels"]) == 1, "Unexpected packing specializations")
        for extension, digest in kernel["asm_sha256"].items():
            evidence.file(archive / f"{kernel['hash']}.{extension}", digest)
        command_path = run_dir / (tag + ".command")
        evidence.file(command_path)
        command = shlex.split(command_path.read_text())
        expected_options = {
            "--profile-from-start": "off",
            "--replay-mode": "kernel",
            "--cache-control": "all",
            "--clock-control": "base",
            "--launch-count": "1",
            "--mode": "profile",
            "--method": "fused",
            "--physical-device": "0",
        }
        require(
            all(
                command[command.index(option) + 1] == value
                for option, value in expected_options.items()
            ),
            "Historical packing collection boundary differs",
        )
        evidence.file(run_dir / "profile_q1_packing.sh")
        report_path = EXPERIMENT / "output/profile" / run_dir.name / (tag + ".ncu-rep")
        report = ncu_report.load_report(str(report_path))
        require(
            report.num_ranges() == 1 and report.range_by_idx(0).num_actions() == 1,
            "Historical packing action coverage differs",
        )
        action = report.range_by_idx(0).action_by_idx(0)
        name = "historical_packing_" + tag.removeprefix("fused_")
        required = SELECTED_METRICS[-3:] + ("gpu__time_duration.sum",)
        if tag == "fused_full":
            required += tuple(
                metric for metric in SELECTED_METRICS[:8] if metric != "sm__ctas_launched.sum"
            )
        record = action_evidence(action, row, name, evidence, pending, required=required)
        record["sass_binding"] = sass_binding(
            action, archive / f"{kernel['hash']}.cubin", evidence, pending, name
        )
        summaries[tag] = {
            "report": evidence.file(report_path),
            "producer": evidence.file(directory / "result.json"),
            "receipt": receipt_record,
            "historical_identity": historical,
            "input_copy_expressions": inputs,
            "driver_ast_equal": True,
            "archived_driver": evidence.file(old_driver),
            "current_driver": evidence.file(ROOT / PACKING_DRIVER),
            "command": command,
            "command_file": evidence.file(command_path),
            "action": record,
            "logs": log_evidence(run_dir, tag, evidence),
        }
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "run-dir",
        "baseline-run-dir",
        "packing-run-dir",
        "receipt",
        "profile-analysis",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        setattr(args, name, value.resolve())
    require(args.run_dir.parent == EXPERIMENT / "output/data", "Unexpected NCU run directory")
    require(
        args.baseline_run_dir.parent == EXPERIMENT / "output/data", "Unexpected baseline directory"
    )
    require(
        args.packing_run_dir.parent == EXPERIMENT / "output/data", "Unexpected packing directory"
    )
    require(
        args.output_dir.is_relative_to(EXPERIMENT / "output") and not args.output_dir.exists(),
        "Use a new analysis directory under the experiment",
    )
    evidence = model.Evidence()
    raw_receipt = evidence.read(args.receipt)
    identity = raw_receipt["identity"]
    receipt = evidence.receipt(args.receipt, kind=model.COMPONENT_KIND, identity=identity)
    require(
        receipt["checks"] == {"passed": True, "complete_cases": 210, "byte_cases": 100},
        "Incomplete component acceptance",
    )
    for group in ("sources", "runtime_files", "inputs"):
        for name, expected in identity[group].items():
            evidence.file(name, expected)
    model.verify_runtime_files(identity["candidate_native"], evidence)
    model.verify_runtime_files(identity["private_generic_native"], evidence)
    model.verify_runtime_files(identity["official_native"], evidence)
    model.verify_runtime_files(identity["mapped_local_native"], evidence)
    ns = evidence.read(args.profile_analysis)
    require(
        ns["passed"] is True
        and ns["schema"] == "deepseek-private-q1-fused-prepare-profile-analysis-v1",
        "Missing accepted native-owned preparation profile",
    )
    source_paths = (
        Path(__file__),
        Path(model.__file__),
        Path(profile.__file__),
        EXPERIMENT / "scripts/profile_q1_fused_prepare_ncu.py",
    )
    source_hashes = {
        str(path.resolve().relative_to(ROOT)): evidence.file(path)["sha256"]
        for path in source_paths
    }
    for relative in (*model.SOURCE_CLOSURE, *profile.SOURCE_CLOSURE):
        source_hashes[relative] = evidence.file(ROOT / relative)["sha256"]
    for relative, expected in ns["analysis_sources"].items():
        evidence.file(ROOT / relative, expected)
    sys.path.insert(0, "/opt/nvidia/nsight-compute/2026.1.1/extras/python")
    import ncu_report

    pending = {}
    summaries = {}
    for variant in ("baseline", "candidate"):
        run_dir = args.baseline_run_dir if variant == "baseline" else args.run_dir
        directory = run_dir / variant
        completed = evidence.read(directory / "summary.json")
        require(
            completed["mode"] == "profile"
            and completed["variant"] == variant
            and completed["policy"] == "zero"
            and completed["identity"] == identity
            and completed["receipt_sha256"] == receipt["receipt_sha256"],
            "Component profile/receipt/native identity differs",
        )
        require(
            Path(completed["input"]).resolve() == Path(min(identity["inputs"])),
            "Expected the real layer 0 input",
        )
        require(
            evidence.read(directory / "identity.json") == identity,
            "Attempted component identity differs",
        )
        for relative, digest in completed["runtime_archives"].items():
            evidence.file(directory / model.safe_relative(relative), digest)
        original_sources = sorted(identity["sources"].items())
        for index, (name, digest) in enumerate(original_sources):
            evidence.file(directory / "source" / f"{index:04d}_{Path(name).name}", digest)
        invocation = evidence.read(run_dir / (variant + "_invocation.json"))
        collector = source_paths[-1]
        evidence.file(collector, invocation["script_sha256"])
        count = 2 if variant == "baseline" else 1
        require(
            invocation["expected_actions"] == (3 if variant == "baseline" else 1)
            and invocation["skipped_matching_bounds_arange"] == (1 if variant == "baseline" else 0),
            "Unexpected NCU action selection",
        )
        report_path = EXPERIMENT / "output/profile" / run_dir.name / (variant + ".ncu-rep")
        report = ncu_report.load_report(str(report_path))
        require(report.num_ranges() == 1, "Expected exactly one NCU range")
        region = report.range_by_idx(0)
        require(
            region.num_actions() == count
            and tuple(region.actions_by_nvtx(["q1_fused_prepare_complete_" + variant + "/"], []))
            == tuple(range(count)),
            "Profile action count or accepted NVTX owner differs",
        )
        expected = ns["profile"]["preparation_change"]["layers"][0][variant]
        if variant == "baseline":
            require(
                expected[0]["signature"]["name"] == "pack_page64", "Unexpected first preparation"
            )
            expected = expected[1:]
        require(len(expected) == count, "Native-owned preparation action count differs")
        actions = [
            action_evidence(
                region.action_by_idx(index), row, variant + "_" + str(index), evidence, pending
            )
            for index, row in enumerate(expected)
        ]
        logs = log_evidence(run_dir, variant, evidence)
        summaries[variant] = {
            "report": evidence.file(report_path),
            "component": evidence.file(directory / "summary.json"),
            "invocation": invocation,
            "actions": actions,
            "logs": logs,
            "component_completion_attempts": completed["attempts"],
            "actual_actions": count,
            "complete_preparation_capture": variant == "candidate",
            "missing_preparation": ["pack_page64"] if variant == "baseline" else [],
        }
    packing = historical_packing(
        args.packing_run_dir,
        identity,
        ns["profile"]["preparation_change"]["layers"][0]["baseline"][0],
        evidence,
        pending,
        ncu_report,
        args.baseline_run_dir / "baseline/packing_assembly",
    )
    evidence.verify()
    result = {
        "schema": "deepseek-private-q1-fused-prepare-ncu-analysis-v1",
        "passed": True,
        "receipt_signature": receipt["receipt_sha256"],
        "identity": identity,
        "profile_analysis": evidence.file(args.profile_analysis),
        "reports": summaries,
        "historical_packing": packing,
        "complete_same_run_baseline": False,
        "cross_report_latency_sum": None,
        "parser": evidence.file(ncu_report.__file__),
        "analysis_sources": source_hashes,
        "boundary": "Individually verified kernel diagnostics: historical GPU0 isolated pack_page64 (full and source collections), GPU1 page arange/stage preparation subset, and GPU1 fused preparation. Historical packing has the exact accepted CUBIN/PTX/metadata, current packing source, saved L0 input and CUDA copy expressions; its driver differs only in formatting. The GPU1 baseline capture omitted packing for an unresolved reason; it is not a complete same-run baseline. NCU kernel replay uses cache flush and base clocks. Different calls, collections and GPUs are never summed as paired latency. Every scalar and per-instance validity flag is retained, including nonzero invalid values. Missing metrics/source content remain unavailable; profiler rule text is preserved without endorsing invalid sampling percentages.",
    }
    args.output_dir.mkdir(parents=True)
    for name, value in pending.items():
        model.write_json(args.output_dir / name, value)
    model.write_json(args.output_dir / "result.json", result)
    for name, digest in source_hashes.items():
        target = args.output_dir / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        evidence.file(target, digest)
    model.write_json(args.output_dir / "input_hashes.json", evidence.files)
    model.write_json(
        args.output_dir / "publication_manifest.json",
        {
            "artifacts": {
                str(path.relative_to(args.output_dir)): evidence.file(path)
                for path in sorted(args.output_dir.rglob("*"))
                if path.is_file()
            }
        },
    )
    print(json.dumps({"passed": True, "run_id": args.output_dir.name, "actions": 5}))


if __name__ == "__main__":
    main()
