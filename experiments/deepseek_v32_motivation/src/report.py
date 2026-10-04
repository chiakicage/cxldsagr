"""Audit saved tensors and publish the four-policy finite-cache comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from experiments.deepseek_v32_motivation.src.measure import (
    SCHEMA,
    SCHEMES,
    WARMUP_POLICY,
    check_compute_graph_replays,
    check_request,
    validate_warmup_trace,
    write_json,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def audit_run(directory):
    """Recheck every saved request and output; no accepted row is trusted by itself."""
    import torch

    from experiments.gr_serving.src.measure import numerical_comparison
    from experiments.gr_serving.src.workload import token_sha256

    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    require(metadata.get("schema") == SCHEMA, "unknown measurement schema")
    require(metadata.get("status") == "accepted", "run is incomplete or failed")
    config = metadata["config"]
    require(config["schemes"] == list(SCHEMES), "all four schemes must be present")
    require(config["byte_subbudgets"] is None, "byte subbudgets are outside this experiment")
    require(config["noncache_headroom_bytes"] is None, "headroom assumptions are not measurements")
    cases = metadata["cases"]
    require([case["scheme"] for case in cases] == list(SCHEMES), "incomplete scheme matrix")
    expected_count = config["requests_per_scheme"]
    for case in cases:
        require(case["requests"] == expected_count, "incomplete measured trace")
        require(
            case["warmup_requests"] == 3
            and case["warmup_request_ids"] == [0, 1, config["num_users"]]
            and case["started_empty"],
            "invalid cache initialization",
        )
    source_manifest = json.loads((directory / "source_manifest.json").read_text())
    source_id = hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode()).hexdigest()
    require(source_id == metadata["source_sha256"], "source identity mismatch")
    for name, expected_digest in source_manifest.items():
        require(digest(directory / "source" / name) == expected_digest, f"source changed: {name}")
    requests = read_jsonl(directory / "workload/requests.jsonl")
    workload = json.loads((directory / "workload/workload.json").read_text())
    require(len(requests) == expected_count, "incomplete workload")
    require(
        workload["workload_sha256"] == metadata["workload_sha256"], "workload identity mismatch"
    )
    identity = {
        key: workload[key] for key in ("config", "heat_sha256", "tokenizer_sha256", "requests")
    }
    identity_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    require(identity_digest == metadata["workload_sha256"], "workload manifest changed")
    require(len(workload["requests"]) == len(requests), "incomplete workload identity inventory")
    first_round = []
    for index, request in enumerate(requests):
        require(
            all(request.get(key) == value for key, value in workload["requests"][index].items()),
            "request differs from workload identity inventory",
        )
        require(request["request_id"] == index, "request order changed")
        require(
            len(request["input_ids"]) == config["history_tokens"] + config["candidate_tokens"],
            "request token count changed",
        )
        require(
            token_sha256(request["input_ids"]) == request["input_sha256"], "input tokens changed"
        )
        if index < config["num_users"]:
            first_round.append(request["user_id"])
        require(
            request["user_id"] == first_round[index % config["num_users"]],
            "trace is not sequential complete rounds",
        )
    require(len(set(first_round)) == config["num_users"], "first round repeats a user")
    traces = metadata.get("warmup_traces", {})
    require(set(traces) == set(SCHEMES), "missing actual warmup paths")
    for scheme in SCHEMES:
        validate_warmup_trace(scheme, traces[scheme], config, requests)
    rows = read_jsonl(directory / "measurements.jsonl")
    require(len(rows) == expected_count * len(SCHEMES), "incomplete measurement matrix")
    by_key = {(row["scheme"], row["request_id"]): row for row in rows}
    require(len(by_key) == len(rows), "duplicate request measurement")
    tensor_hashes = {}
    for index, request in enumerate(requests):
        reference = None
        for scheme in SCHEMES:
            row = by_key.get((scheme, index))
            require(row is not None, "missing scheme/request")
            require(row["run_id"] == metadata["run_id"], "run ID mismatch")
            require(row["workload_sha256"] == metadata["workload_sha256"], "row workload mismatch")
            require(row["input_sha256"] == request["input_sha256"], "row input mismatch")
            require(row["diagnostics_scope"] == "candidate_only", "transfer scope is ambiguous")
            check_request(request, row, config)
            if config.get("enable_compute_graphs", False):
                graphs = row.get("compute_graphs", {})
                check_compute_graph_replays(graphs.get("before"), graphs.get("after"), config, row)
            for name in ("latency_ms", "admission_ms", "prefix_ms", "extend_ms", "cleanup_ms"):
                require(math.isfinite(row[name]) and row[name] >= 0, f"invalid {name}")
            require(row["latency_ms"] > 0, "request latency must be positive")
            stages = sum(
                row[name] for name in ("admission_ms", "prefix_ms", "extend_ms", "cleanup_ms")
            )
            require(
                math.isclose(stages, row["latency_ms"], abs_tol=1e-5), "timing stages do not sum"
            )
            relative = row["output_file"]
            require(relative == f"numerical/{scheme}/{index:06d}.pt", "unexpected output path")
            require(digest(directory / relative) == row["output_sha256"], "output bytes changed")
            tensor_hashes[relative] = row["output_sha256"]
            payload = torch.load(directory / relative, weights_only=True, map_location="cpu")
            require(payload["request_id"] == index, "output request identity changed")
            require(
                payload["input_sha256"] == request["input_sha256"], "output input identity changed"
            )
            dims = metadata["model_dimensions"]
            require(
                tuple(payload["hidden"].shape) == (config["candidate_tokens"], dims["hidden"]),
                "output does not cover every candidate hidden state",
            )
            require(
                tuple(payload["logits"].shape) == (1, dims["vocabulary"]),
                "output does not cover last-token vocabulary logits",
            )
            if reference is None:
                reference = payload
            for name in ("hidden", "logits"):
                result = numerical_comparison(payload[name], reference[name], atol=0, rtol=0)
                require(
                    result["exact"] and row["numerical"][name]["exact"],
                    "numerical evidence is not bitwise equal to HBM",
                )
    for case in cases:
        final = by_key[(case["scheme"], expected_count - 1)]["memory_after"]
        for name in ("torch_peak_allocated_bytes", "torch_peak_reserved_bytes"):
            require(case[name] == final[name], "case memory peak differs from final request sample")
    return (
        metadata,
        rows,
        {
            "status": "passed",
            "checked_requests": len(rows),
            "compared_offload_requests": expected_count * (len(SCHEMES) - 1),
            "all_candidate_hidden_and_logits_exact": True,
            "source_sha256": source_id,
            "warmup_policy": WARMUP_POLICY,
            "checked_warmup_requests": 3 * len(SCHEMES),
            "host_recall_warmed_schemes": list(SCHEMES[1:]),
            "numerical_sha256": tensor_hashes,
        },
    )


def percentile(values, quantile):
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def optional_total(rows, name):
    values = [row["cache_diagnostics"].get(name) for row in rows]
    return sum(values) if all(value is not None for value in values) else None


def summarize(rows):
    summary = []
    for scheme in SCHEMES:
        for revisit in (False, True):
            group = [
                row for row in rows if row["scheme"] == scheme and row["is_revisit"] == revisit
            ]
            require(bool(group), f"missing first/revisit group for {scheme}")
            latency = [row["latency_ms"] for row in group]
            selected = optional_total(group, "selection_records")
            resident = optional_total(group, "resident_selection_records")
            dense_requested = optional_total(group, "dense_requested_records")
            dense_resident = optional_total(group, "dense_resident_records")
            item = {
                "scheme": scheme,
                "visit_kind": "revisit" if revisit else "first",
                "requests": len(group),
                "prefix_hits": sum(row["prefix_cache_hit"] for row in group),
                "evicted_users": sum(len(row["evicted_users"]) for row in group),
                "latency_mean_ms": statistics.mean(latency),
                "latency_median_ms": statistics.median(latency),
                "latency_p95_ms": percentile(latency, 0.95),
                "latency_sum_ms": sum(latency),
                "prefix_mean_ms": statistics.mean(row["prefix_ms"] for row in group),
                "extend_mean_ms": statistics.mean(row["extend_ms"] for row in group),
                "candidate_host_to_device_bytes": optional_total(group, "host_to_device_bytes"),
                "candidate_device_to_host_bytes": optional_total(group, "device_to_host_bytes"),
                "candidate_selection_records": selected,
                "candidate_resident_selection_records": resident,
                "candidate_consumer_union_resident_ratio": (
                    resident / selected if selected and resident is not None else None
                ),
                "candidate_dense_requested_records": dense_requested,
                "candidate_dense_resident_records": dense_resident,
                "candidate_dense_fetched_records": optional_total(group, "dense_fetched_records"),
                "candidate_dense_history_hit_ratio": (
                    dense_resident / dense_requested
                    if dense_requested and dense_resident is not None
                    else None
                ),
                "max_cache_hbm_bytes": max(row["cache_hbm_bytes"] for row in group),
                "max_cache_dram_bytes": max(row["cache_dram_bytes"] for row in group),
            }
            summary.append(item)
    baseline = {
        row["visit_kind"]: row["latency_mean_ms"] for row in summary if row["scheme"] == "hbm"
    }
    for row in summary:
        row["mean_latency_speedup_vs_hbm"] = baseline[row["visit_kind"]] / row["latency_mean_ms"]
    return summary


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flattened(row):
    names = (
        "scheme",
        "request_id",
        "user_id",
        "visit_index",
        "round_index",
        "is_revisit",
        "prefix_cache_hit",
        "prefix_hit_tier",
        "cached_users",
        "latency_ms",
        "admission_ms",
        "prefix_ms",
        "extend_ms",
        "cleanup_ms",
        "cache_hbm_bytes",
        "cache_dram_bytes",
        "shared_cache_hbm_bytes",
        "shared_cache_dram_bytes",
    )
    result = {name: row[name] for name in names}
    result["evicted_users"] = json.dumps(row["evicted_users"])
    for name in (
        "host_to_device_bytes",
        "device_to_host_bytes",
        "prefetched_records",
        "recalled_records",
        "evicted_records",
        "selection_records",
        "resident_selection_records",
        "hbm_token_hit_ratio_before_recall",
        "hit_ratio_stage",
        "dense_requested_records",
        "dense_resident_records",
        "dense_fetched_records",
    ):
        result[f"candidate_{name}"] = row["cache_diagnostics"].get(name)
    for stage in ("before", "after"):
        for name in (
            "torch_allocated_bytes",
            "torch_reserved_bytes",
            "torch_peak_allocated_bytes",
            "torch_peak_reserved_bytes",
            "cuda_free_bytes",
            "cuda_total_bytes",
        ):
            result[f"{stage}_{name}"] = row[f"memory_{stage}"][name]
    return result


def markdown(metadata, summary):
    config, hardware = metadata["config"], metadata["hardware"]
    lines = [
        f"# DeepSeek V3.2 有限缓存对照（{metadata['run_id']}）",
        "",
        (
            f"同一条顺序访问轨迹包含 {config['num_users']} 个用户、{config['rounds']} 轮；"
            f"H={config['history_tokens']:,}，A={config['candidate_tokens']:,}，"
            f"C={config['chunk_size']:,}，P={config['sparse_pool_tokens']:,}，"
            f"NH={config['host_arena_tokens']:,}。只执行这 {config['num_users']} 个用户，未填满 NH。"
        ),
        "",
        (
            "四种方案使用相同的有限 P 历史主 KV 容量。HBM-only 按 P 保留用户，容量不足时"
            "淘汰并在复访时重建；ECHO、serial_sparse 和 dense_prefetch 在 NH 内保留历史，"
            "共享有限 P 历史槽。候选整批执行。未设置 cache 字节子预算或额外 headroom。"
        ),
        "",
        (
            f"硬件为 {hardware['name']}，PyTorch {hardware['torch']} / CUDA {hardware['cuda']}。"
            "十个独立 dense block 使用真实 checkpoint 前三层及对应 source hidden/residual 输入；"
            "这不是训练得到的十层模型或完整 DeepSeek。GR 输入由共享生成器产生，属于合成内容。"
        ),
        "",
        (
            "每种方案依次预热首位用户首访、第二位用户首访、首位用户复访，共三次请求。"
            "三种 offload 方案的最后一次预热均核验 prefix hit、candidate H2D>0 和 "
            "recalled_records>0，确认通用 host gather 路径已执行。随后释放全部用户和共享缓存，从空 cache "
            "执行完整轨迹。端到端 wall latency 包含准入、淘汰、miss 时的历史构建、全部候选 "
            "hidden、末 token LM head 和历史清理。输入生成、模型加载、预热、输出复制、"
            "诊断统计和数值比较不计时；每请求仅测一次，p95 使用线性插值。"
        ),
        "",
        "## 首次访问与复访",
        "",
        "| 方案 | 访问 | 请求数 | prefix hit | E2E 均值 ms | 中位数 ms | p95 ms | 均值相对 HBM 加速 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        visit = "复访" if row["visit_kind"] == "revisit" else "首次"
        lines.append(
            f"| {row['scheme']} | {visit} | {row['requests']} | {row['prefix_hits']}/{row['requests']} "
            f"| {row['latency_mean_ms']:.3f} | {row['latency_median_ms']:.3f} "
            f"| {row['latency_p95_ms']:.3f} | {row['mean_latency_speedup_vs_hbm']:.3f}× |"
        )
    lines.extend(
        [
            "",
            (
                "被淘汰后的请求仍计为复访。prefix hit 表示用户固定历史被保留，不表示选中的"
                "全部主 KV 已驻留 HBM。"
            ),
            "",
            "## 候选阶段搬运与缓存",
            "",
            (
                "下表的传输与 token 计数只覆盖 candidate forward。后端在该阶段开始时重置计数，"
                "因此它们不包含首次构建或重建 history 的流量，不能当作整请求传输总量。"
            ),
            "",
            "| 方案 | 访问 | candidate H2D 总 GiB | candidate D2H 总 GiB | 精确消费并集驻留率 | dense 历史命中率 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        ratio = row["candidate_consumer_union_resident_ratio"]
        dense = row["candidate_dense_history_hit_ratio"]
        ratio_text = "—" if ratio is None else f"{ratio:.3%}"
        dense_text = "—" if dense is None else f"{dense:.3%}"
        visit = "复访" if row["visit_kind"] == "revisit" else "首次"
        lines.append(
            f"| {row['scheme']} | {visit} | {row['candidate_host_to_device_bytes'] / 2**30:.6f} "
            f"| {row['candidate_device_to_host_bytes'] / 2**30:.6f} | {ratio_text} | {dense_text} |"
        )
    lines.extend(
        [
            "",
            (
                "精确消费并集驻留率按 resident_selection_records / selection_records 汇总，"
                "采样位于融合 prefetch 和 append 之后、精确 recall 之前；不表示请求开始时的命中率。"
                "dense 历史命中率单独使用 dense_resident_records / dense_requested_records。"
                "各原始计数与 hit_ratio_stage 保存在逐请求表中。"
            ),
            "",
            "## 内存与数值验收",
            "",
            "| 方案 | CUDA allocated 峰值 GiB | reserved 峰值 GiB |",
            "|---|---:|---:|",
        ]
    )
    for case in metadata["cases"]:
        lines.append(
            f"| {case['scheme']} | {case['torch_peak_allocated_bytes'] / 2**30:.6f} "
            f"| {case['torch_peak_reserved_bytes'] / 2**30:.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "每种方案在释放预热资源后、重新分配共享缓存前重置 CUDA allocator 峰值；"
                "峰值覆盖已加载权重、共享分配和完整请求轨迹。allocated 为活跃分配，reserved "
                "还含 allocator 保留的空闲缓存，两者不能相加。device free-memory 只在边界采样，"
                "不等于连续进程峰值。实际 cache 拥有的 HBM/DRAM 字节与这些峰值分开保存。"
            ),
            "",
            (
                "每条请求的全部 candidate hidden 和末 token logits 均保存到 CPU，在计时外与 "
                "HBM-only 相同请求的输出逐位比较；报告生成时再次核验所有保存输出及来源 hash。"
                "所有对照均通过。"
            ),
            "",
            (
                "逐请求数据见 [per_request.csv](per_request.csv)，分组数据见 [summary.csv](summary.csv)，"
                "完整汇总与边界见 [summary.json](summary.json)，验收与来源见 "
                "[report_provenance.json](report_provenance.json)。完整源数据与输出 tensor 保留在"
                f" `output/data/{metadata['run_id']}/`。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_report(directory, output):
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(output)
    metadata, rows, audit = audit_run(directory)
    summary = summarize(rows)
    output.mkdir(parents=True)
    write_csv(output / "per_request.csv", [flattened(row) for row in rows])
    write_csv(output / "summary.csv", summary)
    write_json(output / "summary.json", {"metadata": metadata, "summary": summary, "audit": audit})
    (output / "results.md").write_text(markdown(metadata, summary))
    write_json(
        output / "report_provenance.json",
        {
            "run_id": metadata["run_id"],
            "audit": audit,
            "input_sha256": {
                name: digest(directory / name)
                for name in (
                    "metadata.json",
                    "measurements.jsonl",
                    "source_manifest.json",
                    "memory.json",
                    "workload/requests.jsonl",
                    "workload/workload.json",
                )
            },
            "report_generator_sha256": digest(__file__),
        },
    )
    return summary


def main(argv=None):
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    args = command.parse_args(argv)
    write_report(args.run_dir, args.output_dir)
    print(f"verified report: {args.output_dir}")


if __name__ == "__main__":
    main()
