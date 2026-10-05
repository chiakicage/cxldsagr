"""Compare two accepted reports on CPU without rerunning either implementation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from shlex import quote

from evaluation.validation import require_receipt
from experiments.deepseek_v32_echo_official.src.measure import normalize_compute_graphs

STEM = "existing_implementation_comparison"
GIB = 2**30
CONFIG_FIELDS = (
    "num_users",
    "rounds",
    "history_tokens",
    "candidate_tokens",
    "chunk_size",
    "sparse_pool_tokens",
    "host_arena_tokens",
    "seed",
    "requests_per_scheme",
    "layers",
    "warmup_requests_per_scheme",
    "warmup_request_indices",
    "warmup_policy",
    "sampling",
    "workspace_query_tokens",
    "padded_history_tokens",
    "byte_subbudgets",
    "noncache_headroom_bytes",
)
MODEL_FIELDS = (
    "model",
    "checkpoint",
    "physical_layers",
    "source_layers",
    "input_semantics",
    "model_scope",
    "linear_backend",
    "chunk_size",
    "sparse_pool_tokens",
    "output",
    "source_layer_parameters",
    "backbone_parameters",
    "endpoint_parameters",
    "total_parameters",
)
HARDWARE_FIELDS = ("name", "total_memory", "sm_count", "python", "torch", "cuda")
GRAPH_PRECISION_FIELDS = {
    "float32_matmul_precision",
    "cuda_matmul_allow_tf32",
    "bf16_reduced_precision_reduction",
    "fp16_reduced_precision_reduction",
    "cudnn_allow_tf32",
}
LABELS = {
    ("motivation", "hbm"): "HBM-only（本地对照）",
    ("motivation", "echo"): "我们的 ECHO（已有验收版本）",
    ("official", "hbm"): "HBM-only（官方对照）",
    ("official", "echo"): "官方 ECHO 适配路径",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path):
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def numerical_evidence(metadata, audit, kind, *, receipt_override=None):
    numerical_key = (
        "all_candidate_hidden_and_logits_exact"
        if kind == "motivation"
        else "all_candidate_hidden_and_logits_numerical_pass"
    )
    if metadata["schema"].endswith("-bench-v1"):
        expected_kind = f"deepseek-v32-{'motivation' if kind == 'motivation' else 'echo-official'}-full-trace-v1"
        evidence = metadata["correctness_receipt"]
        path = Path(evidence["path"] if receipt_override is None else receipt_override)
        require(metadata.get("mode") == "bench", f"{kind} benchmark mode differs")
        require(evidence["kind"] == expected_kind, f"{kind} receipt kind differs")
        require(digest(path) == evidence["sha256"], f"{kind} correctness receipt changed")
        require(
            evidence["identity"] == metadata["validation_identity"],
            f"{kind} correctness identity differs",
        )
        receipt = require_receipt(
            path, kind=expected_kind, identity=metadata["validation_identity"]
        )
        checked = receipt["checks"]["numerical_and_lifecycle"]
        require(checked["status"] == "passed", f"{kind} independent numerical audit failed")
        require(checked[numerical_key] is True, f"{kind} independent numerical audit did not pass")
        if kind == "motivation":
            reported = audit["independent_correctness_receipt"]
            require(reported["checks"] == receipt["checks"], "motivation receipt audit differs")
        else:
            reported = audit["correctness_receipt"]
            require(audit[numerical_key] is True, "official report numerical audit differs")
            require(
                audit["numerical_summary"] == checked["numerical_summary"],
                "official report numerical summary differs",
            )
        require(
            reported["kind"] == expected_kind and reported["sha256"] == evidence["sha256"],
            f"{kind} report receipt provenance differs",
        )
        return {
            "mode": "independent_check_receipt",
            "path": display_path(path),
            "recorded_path": evidence["path"],
            "sha256": evidence["sha256"],
            "kind": expected_kind,
            "checked_requests": checked["checked_requests"],
        }
    require(receipt_override is None, "receipt override requires an independent benchmark")
    require(audit[numerical_key] is True, f"{kind} numerical audit did not pass")
    return {"mode": "combined_measurement_and_check"}


def read_report(path, kind, *, receipt_override=None):
    if path.is_dir():
        path = path / "summary.json"
    report = json.loads(path.read_text())
    metadata, audit = report["metadata"], report["audit"]
    expected_schema = f"deepseek-v32-{'motivation' if kind == 'motivation' else 'echo-official'}-v1"
    require(
        metadata["schema"] in {expected_schema, expected_schema.removesuffix("-v1") + "-bench-v1"},
        f"unexpected {kind} schema",
    )
    require(metadata["status"] == "accepted", f"{kind} report is not accepted")
    require(audit["status"] == "passed", f"{kind} audit did not pass")
    require(audit["source_sha256"] == metadata["source_sha256"], f"{kind} source mismatch")
    provenance_path = path.with_name("report_provenance.json")
    provenance = json.loads(provenance_path.read_text())
    require(provenance["run_id"] == metadata["run_id"], f"{kind} provenance run mismatch")
    require(provenance["audit"] == audit, f"{kind} provenance audit mismatch")
    numerical = numerical_evidence(metadata, audit, kind, receipt_override=receipt_override)
    source = {
        "summary_path": display_path(path),
        "summary_sha256": digest(path),
        "provenance_path": display_path(provenance_path),
        "provenance_sha256": digest(provenance_path),
        "run_id": metadata["run_id"],
        "source_sha256": metadata["source_sha256"],
        "git_revision": metadata["git_revision"],
        "workload_sha256": metadata["workload_sha256"],
        "status": metadata["status"],
        "audit_status": audit["status"],
        "original_input_sha256": provenance["input_sha256"],
        "original_report_generator_sha256": provenance["report_generator_sha256"],
        "hardware": metadata["hardware"],
        "measurement_boundary": metadata["measurement_boundary"],
        "memory_boundary": metadata["memory_boundary"],
        "precision_policy": metadata.get("precision_policy"),
        "precision_settings": metadata.get("precision_settings"),
        "numerical_policy": metadata.get("numerical_policy"),
        "numerical_evidence": numerical,
        "numerical_audit": {k: v for k, v in audit.items() if k != "numerical_sha256"},
    }
    return report, source


def case_for(report, scheme):
    cases = [case for case in report["metadata"]["cases"] if case["scheme"] == scheme]
    require(len(cases) == 1, f"expected one {scheme} case")
    return cases[0]


def compare_identity(motivation, official):
    left, right = motivation["metadata"], official["metadata"]
    config = {}
    for field in CONFIG_FIELDS:
        require(left["config"][field] == right["config"][field], f"config differs: {field}")
        config[field] = left["config"][field]
    graph_modes = [normalize_compute_graphs(metadata["config"]) for metadata in (left, right)]
    require(graph_modes[0] == graph_modes[1], "config differs: enable_compute_graphs")
    config["enable_compute_graphs"] = graph_modes[0]
    require(config["byte_subbudgets"] is None, "expected fixed P/NH without byte subbudgets")
    require(config["noncache_headroom_bytes"] is None, "unexpected headroom deduction")
    require(left["workload_sha256"] == right["workload_sha256"], "workload differs")
    require(left["checkpoint"] == right["checkpoint"], "checkpoint metadata differs")
    require(left["model_dimensions"] == right["model_dimensions"], "model dimensions differ")
    model = {key: case_for(motivation, "hbm")["backend"][key] for key in MODEL_FIELDS}
    for report in (motivation, official):
        for scheme in ("hbm", "echo"):
            case = case_for(report, scheme)
            require(case["requests"] == config["requests_per_scheme"], "incomplete case")
            require(case["started_empty"] is True, "case did not start empty")
            require(
                case["warmup_request_ids"] == config["warmup_request_indices"], "warmup differs"
            )
            require(
                case["warmup_requests"] == config["warmup_requests_per_scheme"], "warmup differs"
            )
            for key, value in model.items():
                require(case["backend"][key] == value, f"model differs: {key}")
    hardware = {}
    for field in HARDWARE_FIELDS:
        require(left["hardware"][field] == right["hardware"][field], f"hardware differs: {field}")
        hardware[field] = left["hardware"][field]
    dependencies = {}
    for name in ("deep_gemm", "flash_mla", "flashinfer"):
        a = left["backend_provenance"]["installed"][name]["distribution_version"]
        b = right["backend_provenance"]["installed"][name]["distribution_version"]
        require(a == b, f"installed dependency version differs: {name}")
        dependencies[name] = a
    settings = [metadata.get("precision_settings") for metadata in (left, right)]
    common_precision = {}
    full_precision_equal = False
    require(
        all(value is None or isinstance(value, dict) for value in settings),
        "invalid precision settings",
    )
    if all(value is not None for value in settings):
        common = settings[0].keys() & settings[1].keys()
        require(bool(common), "no common precision settings")
        if graph_modes[0]:
            require(GRAPH_PRECISION_FIELDS <= common, "graph comparison lacks precision flags")
        for name in sorted(common):
            require(
                type(settings[0][name]) is type(settings[1][name])
                and settings[0][name] == settings[1][name],
                f"precision differs: {name}",
            )
            common_precision[name] = settings[0][name]
        full_precision_equal = (
            bool(common_precision)
            and settings[0] == settings[1]
            and left.get("precision_policy") is not None
            and left.get("precision_policy") == right.get("precision_policy")
        )
        precision_boundary = (
            "Common recorded effective precision flags match. Fields recorded by only one run "
            "and experiment-specific policy labels do not establish full policy equality."
            if not full_precision_equal
            else "All recorded effective precision flags and policy identifiers match."
        )
    else:
        require(not graph_modes[0], "graph comparison lacks precision flags")
        precision_boundary = (
            "At least one legacy eager run lacks precision settings; full matmul/TF32 policy "
            "equality is not established."
        )
    return {
        "status": "passed_for_recorded_fields",
        "config": config,
        "model": model,
        "model_dimensions": left["model_dimensions"],
        "hardware_and_runtime": hardware,
        "installed_dependency_versions": dependencies,
        "workload_sha256": left["workload_sha256"],
        "checkpoint_metadata_matches": True,
        "checkpoint_identity_boundary": left["checkpoint"]["identity_boundary"],
        "same_physical_gpu": left["hardware"]["uuid"] == right["hardware"]["uuid"],
        "same_source_snapshot": left["source_sha256"] == right["source_sha256"],
        "recorded_common_precision_settings": common_precision,
        "recorded_common_precision_settings_equality_verified": bool(common_precision),
        "full_precision_policy_equality_verified": full_precision_equal,
        "precision_comparison_boundary": precision_boundary,
    }


def extract_rows(report, kind):
    metadata = report["metadata"]
    rows = []
    for scheme in ("hbm", "echo"):
        groups = [row for row in report["summary"] if row["scheme"] == scheme]
        require(len(groups) == 2, f"expected first/revisit summaries for {kind}/{scheme}")
        visits = {group["visit_kind"]: group for group in groups}
        require(set(visits) == {"first", "revisit"}, "unexpected visit groups")
        first, revisit = visits["first"], visits["revisit"]
        require(first["requests"] == metadata["config"]["num_users"], "first count differs")
        require(
            sum(group["requests"] for group in groups) == metadata["config"]["requests_per_scheme"],
            "summary request count differs",
        )
        for group in groups:
            require(
                math.isclose(
                    group["latency_mean_ms"] * group["requests"],
                    group["latency_sum_ms"],
                    rel_tol=1e-12,
                ),
                "mean/sum disagree",
            )
        case = case_for(report, scheme)
        allocated, reserved = case["torch_peak_allocated_bytes"], case["torch_peak_reserved_bytes"]
        require(0 < allocated <= reserved, "invalid allocator peaks")
        rows.append(
            {
                "group": kind,
                "scheme": scheme,
                "implementation": LABELS[kind, scheme],
                "run_id": metadata["run_id"],
                "source_sha256": metadata["source_sha256"],
                "requests": sum(group["requests"] for group in groups),
                "first_requests": first["requests"],
                "revisit_requests": revisit["requests"],
                "first_mean_ms": first["latency_mean_ms"],
                "first_p95_ms": first["latency_p95_ms"],
                "revisit_mean_ms": revisit["latency_mean_ms"],
                "revisit_p95_ms": revisit["latency_p95_ms"],
                "total_seconds": sum(group["latency_sum_ms"] for group in groups) / 1000,
                "torch_peak_allocated_bytes": allocated,
                "torch_peak_reserved_bytes": reserved,
                "torch_peak_allocated_gib": allocated / GIB,
                "torch_peak_reserved_gib": reserved / GIB,
                "gpu_uuid": metadata["hardware"]["uuid"],
            }
        )
    return rows


def render_markdown(result):
    identity = result["identity_check"]
    config, model = identity["config"], identity["model"]
    hardware = identity["hardware_and_runtime"]
    rows, differences = result["rows"], result["observed_our_echo_vs_official_echo"]
    gpu_relation = "同一物理 GPU" if identity["same_physical_gpu"] else "不同物理 GPU"
    graph_mode = "启用" if config["enable_compute_graphs"] else "关闭"
    independent = all(
        source["numerical_evidence"]["mode"] == "independent_check_receipt"
        for source in result["inputs"].values()
    )
    numerical_boundary = (
        "两组数值验收均来自独立 check，正式计时不保存或比较完整输出。"
        if independent
        else "数值验收分别对应各来源报告记录的执行边界。"
    )
    if identity["full_precision_policy_equality_verified"]:
        precision_boundary = "两组记录的全部有效精度设置及 policy ID 一致。"
    elif identity["recorded_common_precision_settings_equality_verified"]:
        precision_boundary = (
            "两组共同记录的有效精度设置一致；仅一组记录的字段及各实验的 policy ID "
            "不能用于确认完整精度策略相同。"
        )
    else:
        precision_boundary = (
            "至少一组旧 eager 运行未记录 precision settings，"
            "不能确认两次运行的完整 matmul/TF32 策略相同。"
        )
    lines = [
        "# 已有实现结果对照",
        "",
        (
            "本表只整理已发布且通过验收的结果，没有新增 GPU 运行或计时。"
            f"本地列对应 `{result['inputs']['motivation']['run_id']}`，"
            f"官方列对应 `{result['inputs']['official']['run_id']}`。"
            "结果仅代表这两次运行记录的实现。"
        ),
        "",
        (
            f"两组均使用 {model['physical_layers']} 个独立 checkpoint dense block 副本"
            f"（{model['total_parameters']:,} 参数），采用 FP8 linear；"
            "各副本复用对应源层的 hidden/residual 输入。该负载是 checkpoint 工作负载替身。"
            f"P={config['sparse_pool_tokens']:,}，NH={config['host_arena_tokens']:,}，"
            f"H={config['history_tokens']:,}，A={config['candidate_tokens']}，"
            f"history chunk={config['chunk_size']:,}；"
            f"{config['num_users']} 用户按顺序访问 {config['rounds']} 轮，seed={config['seed']}。"
            f"两组均{graph_mode}公共计算 CUDA Graph。"
            "工作负载 SHA256 相同，P/NH 不扣除字节子预算或经验预留。"
        ),
        "",
        (
            "| 实现 | 首访均值 ms | 首访 p95 ms | 复访均值 ms | 复访 p95 ms | 总耗时 s |"
            " allocated 峰值 GiB | reserved 峰值 GiB |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = (
            "first_mean_ms",
            "first_p95_ms",
            "revisit_mean_ms",
            "revisit_p95_ms",
            "total_seconds",
            "torch_peak_allocated_gib",
            "torch_peak_reserved_gib",
        )
        lines.append(
            f"| {row['implementation']} | " + " | ".join(f"{row[k]:.3f}" for k in values) + " |"
        )
    lines += [
        "",
        (
            f"首访、复访分别含 {rows[0]['first_requests']}、{rows[0]['revisit_requests']} 个请求；"
            "p95 直接引用各组已发布的线性插值分位数。总耗时为全部请求延迟之和，"
            "包含 miss 时的 history 重建和全部 candidate hidden、末 token LM head。"
            "allocated/reserved 是各方案完整正式轨迹的 PyTorch 峰值；两者分别报告，"
            "不作为设备已用 HBM 的峰值。"
        ),
        "",
        (
            f"以官方 ECHO 为分母，本地 ECHO 的首访均值差为 "
            f"{differences['first_mean_ms']['relative_percent']:+.3f}%，"
            f"复访均值差为 {differences['revisit_mean_ms']['relative_percent']:+.3f}%，"
            f"总耗时差为 {differences['total_seconds']['relative_percent']:+.3f}%。"
            "正值表示本地耗时较高，负值表示本地耗时较低。这些是两次独立轨迹的观测差异，"
            "不足以判断优势是否稳定。"
        ),
        "",
        (
            f"两次运行使用{gpu_relation}，PyTorch 记录的设备名称均为 "
            f"{hardware['name']}（{hardware['sm_count']} SM、{hardware['total_memory'] / GIB:.3f} GiB HBM）。"
            "本地 HBM 使用 mainline DeepGEMM logits，本地 ECHO 使用项目融合 prefetch 路径，"
            "两者使用 FlashInfer top-k；官方组使用 ECHO 原始 resident/fused logits 和原始 top-k。"
            "两组选择路径和 HBM 基线有差异，不能把差额全部归因于缓存或搬运实现。"
        ),
        "",
        (
            "两组记录的 FP8 linear 设置一致，checkpoint 身份核对依赖路径和 shard stat 清单，"
            "没有权重内容哈希。"
            f"{precision_boundary}"
            f"{numerical_boundary}"
            "本地输出与其 HBM 对照逐位一致；官方输出及独立 resident 重跑"
            "均通过该实验预先固定的数值门槛，官方 top-k 顺序及并列值可能变化。"
            "本次整理未新增跨运行数值比较。"
        ),
        "",
        "## 来源",
        "",
    ]
    for kind, source in result["inputs"].items():
        lines += [
            f"- {kind}：`{source['run_id']}`；源码 SHA256 `{source['source_sha256']}`。",
            f"  输入 `{source['summary_path']}`，SHA256 `{source['summary_sha256']}`。",
            f"  GPU UUID `{source['hardware']['uuid']}`。",
        ]
    lines += [
        "",
        f"共同 workload SHA256：`{result['identity_check']['workload_sha256']}`。",
        "",
        (
            "CSV 保存未四舍五入的数据；JSON 保存输入及生成器哈希、配置核对、"
            "原验收状态、完整 run ID 和测量边界。生成入口："
        ),
        "",
        "```bash",
        "python -m experiments.deepseek_v32_echo_official.src.compare_existing \\",
        f"  --motivation-report {quote(result['inputs']['motivation']['summary_path'])} \\",
        f"  --official-report {quote(result['inputs']['official']['summary_path'])} \\",
        *[
            f"  --{kind}-receipt {quote(source['numerical_evidence']['path'])} \\"
            for kind, source in result["inputs"].items()
            if source["numerical_evidence"]["mode"] == "independent_check_receipt"
        ],
        f"  --output-dir {quote(result['output_directory'])}",
        "```",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motivation-report", type=Path, required=True)
    parser.add_argument("--official-report", type=Path, required=True)
    parser.add_argument("--motivation-receipt", type=Path)
    parser.add_argument("--official-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    motivation, motivation_source = read_report(
        args.motivation_report, "motivation", receipt_override=args.motivation_receipt
    )
    official, official_source = read_report(
        args.official_report, "official", receipt_override=args.official_receipt
    )
    identity = compare_identity(motivation, official)
    for filename in ("workload/requests.jsonl", "workload/workload.json"):
        require(
            motivation_source["original_input_sha256"][filename]
            == official_source["original_input_sha256"][filename],
            f"published workload file hashes differ: {filename}",
        )
    rows = extract_rows(motivation, "motivation") + extract_rows(official, "official")
    own, upstream = rows[1], rows[3]
    differences = {
        key: {
            "difference": own[key] - upstream[key],
            "relative_percent": (own[key] / upstream[key] - 1) * 100,
        }
        for key in ("first_mean_ms", "revisit_mean_ms", "total_seconds")
    }
    result = {
        "schema": "deepseek-v32-echo-existing-comparison-v1",
        "analysis_only": True,
        "new_gpu_measurements": False,
        "original_numerical_audits_reused_without_rerun": True,
        "comparison_scope": "independent accepted runs; not a sole-cache-implementation speedup",
        "inputs": {"motivation": motivation_source, "official": official_source},
        "generator": {"path": display_path(Path(__file__)), "sha256": digest(Path(__file__))},
        "output_directory": display_path(args.output_dir),
        "identity_check": identity,
        "rows": rows,
        "official_upstream_revision": official["metadata"]["official_provenance"][
            "upstream_revision"
        ],
        "difference_definition": "our_echo - official_echo; relative percent uses official_echo denominator",
        "observed_our_echo_vs_official_echo": differences,
        "cross_run_numerical_comparison_performed": False,
        "statistical_stability_established": False,
    }
    markdown = render_markdown(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{STEM}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    with (args.output_dir / f"{STEM}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / f"{STEM}.md").write_text(markdown)
    print(
        json.dumps(
            {
                "output_directory": str(args.output_dir),
                "rows": len(rows),
                "identity_check": identity["status"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
