"""Audit a four-process warm Q1 cohort without publishing MFU or replacing cold results."""

from __future__ import annotations

import argparse
import datetime
import math
import shutil
import statistics
from pathlib import Path

from evaluation.validation import identity_digest, require_receipt
from experiments.deepseek_v32_mfu.src import audit_shape_matrix as audit
from experiments.deepseek_v32_mfu.src import method_isolation_report as isolated
from experiments.deepseek_v32_mfu.src import report_full_graph_mfu as hardware
from experiments.deepseek_v32_mfu.src.report_mfu import (
    audit_local_native_artifacts,
    save_csv,
    save_json,
)
from experiments.deepseek_v32_mfu.src.run_contract import (
    benchmark_view,
    common_workload_identity,
    execution_identity,
    receipt_kind,
    validate_completed_result,
)

KIND = "deepseek-v32-q1-warm-alignment"
ORDER = (("hbm", "check"), ("echo", "check"), ("echo", "bench"), ("echo", "profile"))
CONFIGURATION = {**isolated.CONFIGURATION, "extend_residency": "warm"}


def manifest_children(manifest, evidence):
    audit.require(
        manifest.get("schema_version") == 1
        and manifest.get("kind") == KIND
        and manifest.get("completed_at_utc"),
        "Incomplete or unsupported warm Q1 manifest",
    )
    children = manifest["methods"]
    audit.require(
        set(children) == {"hbm", "echo"}
        and set(children["hbm"]) == {"check"}
        and set(children["echo"]) == {"check", "bench", "profile"},
        "Warm Q1 requires exactly four declared child roles",
    )
    audit.require(
        [(row["method"], row["phase"]) for row in manifest["execution_order"]] == list(ORDER),
        "Warm Q1 child execution order differs",
    )
    directories, results, seen = {"hbm": {}, "echo": {}}, {"hbm": {}, "echo": {}}, set()
    for index, (method, phase) in enumerate(ORDER):
        child = children[method][phase]
        directory = Path(child["directory"]).resolve(strict=True)
        run_id = child["run_id"]
        audit.require(directory not in seen and directory.name == run_id, "Aliased child directory")
        seen.add(directory)
        audit.require(child["order"] == index, "Child order differs from manifest")
        audit.require(
            manifest["execution_order"][index]["run_id"] == run_id, "Child run ID differs"
        )
        path = directory / "result.json"
        audit.require(evidence.digest(path) == child["result_sha256"], "Child result changed")
        result = validate_completed_result(
            path,
            expected_mode=phase if phase != "profile" else "profile",
            expected_profile_detail="minimal_node_model_scopes" if phase == "profile" else None,
        )
        audit.require(
            result["schema_version"] == 4
            and result["run_id"] == run_id
            and result["methods"] == [method],
            "Child method or run differs",
        )
        audit.require(
            evidence.digest(directory / "runner.json") == child["runner_record_sha256"],
            "Child runner changed",
        )
        directories[method][phase], results[method][phase] = directory, result
    return directories, results


def audit_cohort(manifest, directories, results, evidence):
    root = Path(__file__).resolve().parents[3]
    expected = {
        **CONFIGURATION,
        "trace_warmups": 1,
        "physical_device": "0",
        "cpu_affinity": list(range(8)),
    }
    configuration = manifest["configuration"]
    audit.require(
        all(configuration.get(k) == v for k, v in expected.items()), "Warm configuration differs"
    )
    request = evidence.read(manifest["request"]["path"])
    audit.require(
        evidence.digest(manifest["request"]["path"]) == manifest["request"]["sha256"]
        and request["stable_prefix_tokens"] == 65536
        and request["candidate_suffix_tokens"] == 1
        and len(request["input_ids"]) == 65537
        and request["input_ids"][-1] == configuration["append_token"],
        "Warm request binding differs",
    )
    expected_sources = {
        "experiments/deepseek_v32_mfu/scripts/" + name
        for name in ("q1_warm_alignment.sh", "runner_common.sh", "run.sh", "gap_profile.sh")
    }
    audit.require(
        set(manifest["runner"]["sources"]) == expected_sources, "Runner source inventory differs"
    )
    for name, digest in manifest["runner"]["sources"].items():
        audit.require(evidence.digest(root / name) == digest, "Warm runner source changed")
    processes, seen, previous_end = [], set(), None
    reference = results["hbm"]["check"]
    for method, phase in ORDER:
        directory, result = directories[method][phase], results[method][phase]
        audit.require(
            all(result.get(k) == v for k, v in CONFIGURATION.items())
            and result["model"] == configuration["model"]
            and result["request_sha256"] == manifest["request"]["sha256"]
            and result["execution_identity"]
            == execution_identity(result)
            == results[method]["check"]["execution_identity"]
            and common_workload_identity(result) == common_workload_identity(reference),
            "Warm child execution identity differs",
        )
        hardware.audit_hardware(result["hardware"])
        runner = evidence.read(directory / "runner.json")
        isolated.validate_runner_command(runner, result, method=method, phase=phase, root=root)
        for field, digest in (("source", "sha256"), ("helper_source", "helper_sha256")):
            audit.require(
                evidence.digest(root / runner[field]) == runner[digest],
                "Child runner source identity changed",
            )
        child = manifest["methods"][method][phase]
        audit.require(
            child["invocation"][:6]
            == ["env", "MFU_RUN_ID=" + result["run_id"], "taskset", "-c", "0-7", "bash"]
            and runner["invocation"] == child["invocation"][6:]
            and runner["target_process"] == result["process_provenance"]
            and runner["run_id"] == result["run_id"]
            and runner["cpu_affinity"] == list(range(8))
            and runner["environment"]["CUDA_VISIBLE_DEVICES"] == "0",
            "Child process or literal invocation differs",
        )
        process = runner["target_process"]
        identity = (runner["shell_process"]["boot_id"], process["pid"], process["start_ticks"])
        audit.require(identity not in seen, "Warm child reused an execution process")
        seen.add(identity)
        start, end = (
            datetime.datetime.fromisoformat(child[k])
            for k in ("started_at_utc", "completed_at_utc")
        )
        audit.require(
            start < end and (previous_end is None or previous_end <= start), "Children overlap"
        )
        previous_end = end
        audit.require(
            set(child["files"]) == {"data", "log", "profile"}, "Artifact categories differ"
        )
        for category, entries in child["files"].items():
            base = directory.parent.parent / category / result["run_id"]
            actual = {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()}
            audit.require(actual == set(entries), "Child artifact inventory changed")
            for relative, digest in entries.items():
                path = (base / relative).resolve(strict=True)
                audit.require(
                    path.is_relative_to(base.resolve()) and evidence.digest(path) == digest,
                    "Child artifact changed",
                )
        audit.require(
            evidence.digest(directory / "request.json") == result["request_sha256"],
            "Saved request changed",
        )
        processes.append({"method": method, "phase": phase, **process})
    return processes


def audit_receipts(directories, results, evidence):
    receipts = {}
    for method in ("hbm", "echo"):
        check = results[method]["check"]
        path = directories[method]["check"] / "receipt.json"
        receipt = require_receipt(
            path, kind=receipt_kind(check), identity=check["execution_identity"]
        )
        evidence.digest(path)
        audit.require(
            receipt["checks"]["comparisons"] == check["correctness"]
            and receipt["checks"]["extend_graph"] == check["extend_graph_checks"],
            "Receipt differs from saved correctness",
        )
        for artifact in receipt["artifacts"].values():
            audit.require(
                evidence.digest(path.parent / artifact["path"]) == artifact["sha256"],
                "Check output changed",
            )
        isolated.check_graph_acceptance(check)
        receipts[method] = receipt
    reference = receipts["echo"]["checks"]["hbm_reference"]
    audit.require(
        Path(reference["receipt_path"]).resolve() == directories["hbm"]["check"] / "receipt.json"
        and reference["receipt_sha256"] == receipts["hbm"]["receipt_sha256"],
        "ECHO binds a different HBM reference",
    )
    for phase in ("bench", "profile"):
        result = results["echo"][phase]
        benchmark_view(directories["echo"][phase], result)
        audit.require(
            Path(result["validation_receipt"]["receipt_path"]).resolve()
            == directories["echo"]["check"] / "receipt.json",
            "Measurement binds a different check",
        )
    profile = results["echo"]["profile"]
    audit.require(
        Path(profile["benchmark"]["directory"]).resolve() == directories["echo"]["bench"],
        "Profile binds a different benchmark",
    )
    audit.require(
        profile["measurement_identity"]["trace_warmups_per_capture"] == 1,
        "Trace warmup count differs",
    )
    return receipts


def saved_outputs(directories, evidence):
    import torch

    loaded, inventory, comparisons = {}, [], []
    for method in ("hbm", "echo"):
        directory = directories[method]["check"]
        bundles = {}
        for name in ("control", "prefix_logits", "extend_graph_check"):
            path = directory / f"{method}_{name}.pt"
            evidence.digest(path)
            bundles[name] = torch.load(path, map_location="cpu", weights_only=True)
        audit.require(
            set(bundles["control"]) == {"hidden", "logits"}, "Control output schema differs"
        )
        graph = bundles.pop("extend_graph_check")
        audit.require(
            set(graph) == {"baseline", "changed_baseline", "changed_graph"},
            "Graph output schema differs",
        )
        bundles["prefix_logits"] = {"logits": bundles["prefix_logits"]}
        bundles.update(graph)
        for group, bundle in bundles.items():
            audit.require(
                set(bundle) == ({"logits"} if group == "prefix_logits" else {"hidden", "logits"}),
                "Saved output schema differs",
            )
            for key, value in bundle.items():
                inventory.append(audit.tensor_record(value, f"{method}/{group}/{key}", 1))
        for key in ("hidden", "logits"):
            comparisons.append(
                audit.compare(
                    bundles["control"][key], graph["baseline"][key], f"{method}/baseline/{key}"
                )
            )
            comparisons.append(
                audit.compare(
                    graph["changed_graph"][key],
                    graph["changed_baseline"][key],
                    f"{method}/changed/{key}",
                )
            )
            audit.require(
                not torch.equal(bundles["control"][key], graph["changed_graph"][key]),
                "Changed input did not affect output",
            )
        loaded[method] = bundles
    for group in ("control", "prefix_logits", "changed_graph"):
        for key in loaded["echo"][group]:
            comparisons.append(
                audit.compare(
                    loaded["echo"][group][key],
                    loaded["hbm"][group][key],
                    f"hbm_vs_echo/{group}/{key}",
                )
            )
    path = directories["echo"]["profile"] / "echo_profile_output.pt"
    evidence.digest(path)
    profile = torch.load(path, map_location="cpu", weights_only=True)
    audit.require(set(profile) == {"hidden", "logits"}, "Profile output schema differs")
    for key, value in profile.items():
        inventory.append(audit.tensor_record(value, f"echo/profile/{key}", 1))
        comparisons.append(
            audit.compare(value, loaded["echo"]["control"][key], f"echo/profile/{key}")
        )
    return {
        "inventory": inventory,
        "comparisons": comparisons,
        "boundary": "Saved output CPU reread; cache transitions retain their independently bound runtime acceptance.",
    }


def draw(panel, output):
    shape = panel["shape"]
    metrics = isolated.compact.draw(
        [],
        [panel],
        output,
        layout="extend-only",
        window_kind="three-layers",
        io_layout="directions",
        annotations="idle-echo",
        figure_title=(
            f"Warm ECHO | H={shape['prefix_tokens']:,}, A={shape['extend_tokens']:,}, L0-L2"
        ),
    )
    for suffix in ("svg", "png"):
        (output / f"extend.{suffix}").rename(output / f"timeline.{suffix}")
    return metrics


def generate(manifest_path, output):
    output = Path(output).resolve()
    audit.require(not output.exists(), "Warm report output already exists")
    evidence = audit.Evidence()
    manifest = evidence.read(manifest_path)
    directories, results = manifest_children(manifest, evidence)
    processes = audit_cohort(manifest, directories, results, evidence)
    receipts = audit_receipts(directories, results, evidence)
    sources, runtimes = {}, {}
    for method, phase in ORDER:
        key = method + "/" + phase
        sources[key] = audit.source_audit(
            directories[method][phase], results[method][phase], evidence
        )
        runtimes[key] = audit.runtime_audit(
            results[method][phase], evidence, require_offload=method == "echo"
        )
    tensors = saved_outputs(directories, evidence)
    bench, profile = results["echo"]["bench"], results["echo"]["profile"]
    samples = audit.cache_sample_audit(bench, "echo", "extend")
    wall = bench["measurements"]["echo"]["extend_samples_ms"]
    audit.require(
        len(wall) == CONFIGURATION["repeats"]
        and all(
            type(value) in (int, float) and math.isfinite(value) and value > 0 for value in wall
        )
        and statistics.median(wall) == bench["measurements"]["echo"]["extend_median_ms"],
        "Formal wall samples or median differ",
    )
    native = audit_local_native_artifacts(results["echo"], require_offload=True)
    panels, raw, _ = isolated.timeline_evidence(
        "echo", directories["echo"]["profile"], profile, evidence
    )
    panel = panels["extend"]
    cache_rows = [
        {"sample": sample, "layer": layer, **metrics}
        for sample, values in enumerate(samples)
        for layer, metrics in enumerate(values)
    ]
    wall_rows = [
        {"sample": index, "wall_ms": value, "benchmark_run_id": bench["run_id"]}
        for index, value in enumerate(bench["measurements"]["echo"]["extend_samples_ms"])
    ]
    summary = {
        "schema": "deepseek-v32-q1-warm-report-v1",
        "run_id": manifest["run_id"],
        "passed": True,
        "configuration": manifest["configuration"],
        "children": manifest["methods"],
        "processes": processes,
        "sources": sources,
        "runtimes": runtimes,
        "native_artifacts": native,
        "receipts": {method: receipt["receipt_sha256"] for method, receipt in receipts.items()},
        "execution_identity_sha256": identity_digest(bench["execution_identity"]),
        "tensors": tensors,
        "step_wall_median_ms": bench["measurements"]["echo"]["extend_median_ms"],
        "step_samples": wall_rows,
        "cache_samples": cache_rows,
        "profile_cache_per_layer": profile["measurements"]["echo"]["extend_cache_per_layer"],
        "window": panel["window"],
        "raw_windows": raw,
        "boundary": "Warm retains the prepared prefill main-KV state before each independent step. This is not a claim of full official SGLang policy equivalence. Complete synchronized wall time and intrusive L0-L2 GPU window are separate. No MFU or cross-batch speedup is reported.",
    }
    for name in (
        "q1_warm_report.py",
        "method_isolation_report.py",
        "audit_shape_matrix.py",
        "launch_gap.py",
        "compact_timeline.py",
        "run_contract.py",
        "report_mfu.py",
        "report_full_graph_mfu.py",
        "method_process_audit.py",
    ):
        evidence.digest(Path(__file__).with_name(name))
    evidence.verify()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "summary.json", summary)
    save_json(output / "window_rows.json", {"extend": [panel]})
    save_csv(output / "cache_samples.csv", cache_rows)
    save_csv(output / "timing_samples.csv", wall_rows)
    save_json(
        output / "provenance.json",
        {
            "manifest": str(Path(manifest_path).resolve()),
            "manifest_sha256": evidence.digest(manifest_path),
            "profile": panel["provenance"],
            "input_sha256": evidence.hashes,
        },
    )
    draw(panel, output)
    (output / "results.md").write_text(
        f"# 本地 ECHO warm Q1 对照\n\n运行 `{manifest['run_id']}` 保留 prefill 后的主 KV 驻留，每次单步前恢复相同 prefix。输入追加 token 为 {manifest['configuration']['append_token']}。真实 L0–L2，H=65,536、A=1、P=NH=65,600，GPU0 / CPU0–7；四个独立进程分别完成 HBM 检查、ECHO 检查、ECHO 正式计时和最小节点 profile。\n\n完整同步单步 wall 中位数为 {summary['step_wall_median_ms']:.6f} ms；独立 NSYS 的 L0–L2 窗口为 {panel['window']['window_ms']:.6f} ms，GPU idle 为 {panel['window']['gpu_idle_ms'] * 1000:.3f} µs。Wall 包含输入、事务、同步和提交，加载、编译、图准备和 prefix 恢复在计时外。Profile 窗口不等于完整单步延迟。\n\n![warm 三层 timeline](timeline.svg)\n\n[逐层流量与召回计数](cache_samples.csv)、[全部 wall 样本](timing_samples.csv)、[来源及身份](provenance.json)、[完整审计](summary.json)分别保留正式计时和 profile 的证据。已重读保存输出、receipt、源码、native、原始 GPU 活动和 graph 归属。warm 对齐驻留条件，不自动证明与官方 SGLang 的缓存策略、输入或环境完全一致；本报告不计算 MFU，也不以不同批次推断加速。\n"
    )
    evidence.verify()
    save_json(
        output / "publication_manifest.json",
        {
            "schema": "q1-warm-artifacts-v1",
            "files_sha256": {
                str(path.relative_to(output)): audit.Evidence().digest(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        },
    )
    return summary


def redraw(source, output, run_id):
    """Restyle an immutable accepted report without rerunning GPU measurement."""
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    audit.require(not output.exists(), "Warm display output already exists")
    audit.require(bool(run_id) and Path(run_id).name == run_id, "Invalid display run ID")
    evidence = audit.Evidence()
    original_manifest = evidence.read(source / "publication_manifest.json")
    for relative, digest in original_manifest["files_sha256"].items():
        path = (source / relative).resolve(strict=True)
        audit.require(
            path.is_relative_to(source) and evidence.digest(path) == digest,
            "Accepted warm report changed",
        )
    summary = evidence.read(source / "summary.json")
    panels = evidence.read(source / "window_rows.json")["extend"]
    audit.require(
        summary["passed"]
        and summary["configuration"]["extend_residency"] == "warm"
        and len(panels) == 1
        and panels[0]["method"] == "echo"
        and panels[0]["window"] == summary["window"],
        "Display requires one accepted warm ECHO window",
    )
    renderer = {
        str(path.relative_to(Path(__file__).resolve().parents[3])): evidence.digest(path)
        for path in (Path(__file__), Path(isolated.compact.__file__))
    }
    evidence.verify()
    shutil.copytree(source, output)
    metrics = draw(panels[0], output)
    (output / "original_publication_manifest.json").write_bytes(
        (source / "publication_manifest.json").read_bytes()
    )
    unchanged = {
        relative: digest
        for relative, digest in original_manifest["files_sha256"].items()
        if relative not in {"timeline.svg", "timeline.png", "results.md"}
    }
    audit.require(
        all(audit.Evidence().digest(output / name) == digest for name, digest in unchanged.items()),
        "Restyling changed measured data",
    )
    save_json(
        output / "display_provenance.json",
        {
            "schema": "q1-warm-timeline-display-v1",
            "run_id": run_id,
            "measurement_run_id": summary["run_id"],
            "source_report": str(source),
            "source_publication_sha256": evidence.digest(source / "publication_manifest.json"),
            "renderer_source_sha256": renderer,
            "unchanged_data_sha256": unchanged,
            "rendered_metrics": metrics,
            "boundary": "CPU-only redraw from the accepted window. Measurement, activity timestamps, layer boundaries and traffic counters are unchanged. The shared cold/warm renderer highlights ECHO prepare/finalize/hint; other control remains active and is never relabeled as idle.",
        },
    )
    report = output / "results.md"
    report.write_text(
        report.read_text()
        + f"\n展示更新 `{run_id}` 复用四方案 timeline 的泳道、计算阶段配色、L0–L2 分界和图例；仅从已验收的 profile 数据重绘，未新增 GPU 测量。原始时间戳、计数和延迟保持不变；来源见 [展示记录](display_provenance.json)。\n"
    )
    evidence.verify()
    save_json(
        output / "publication_manifest.json",
        {
            "schema": "q1-warm-display-artifacts-v1",
            "run_id": run_id,
            "files_sha256": {
                str(path.relative_to(output)): audit.Evidence().digest(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path != output / "publication_manifest.json"
            },
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--source-report", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source_report:
        if not args.run_id:
            parser.error("--source-report requires a fresh --run-id")
        redraw(args.source_report, args.output, args.run_id)
    else:
        if args.run_id:
            parser.error("--run-id is only used with --source-report")
        generate(args.manifest, args.output)
    print(args.output / "publication_manifest.json")


if __name__ == "__main__":
    main()
