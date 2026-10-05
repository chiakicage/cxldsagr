"""Publish verified fixed ECHO capacity runs and separate static capacity plans."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
import time
from decimal import Decimal
from pathlib import Path

import torch

from experiments.cache_management.src.capacity_probe import (
    REFERENCE_SCHEMA,
    check_lifecycle,
    compare_reference,
    load_reference,
    reference_binding,
    tensor_summary,
    write_json,
)

ROOT = Path(__file__).resolve().parents[3]
MEMORY_BOUNDARY = (
    "PyTorch allocated/reserved 峰值从模型加载前开始统计；设备 free memory 只在调用边界采样。"
    "这些值均不是 NVML 进程峰值，设备已用显存还可能包含其他进程。"
    "host allocator counters 和进程 RSS 单独记录，不能替代精确的 cache 所有权统计。"
)
BUDGET_METRICS = (
    "torch_peak_allocated_bytes",
    "torch_peak_reserved_bytes",
    "maximum_sampled_device_used_bytes",
)
BUDGET_LABELS = {
    "observed_exceeded": "已观察到超额",
    "no_observed_excess": "未观察到超额",
    "unmeasured": "未测量",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def aggregate(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def candidate_policy(value):
    policy = value.get("candidate_persistence", "host_backed")
    require(policy in ("host_backed", "gpu_transient"), "unknown candidate persistence policy")
    return policy


def checked_relative(name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "invalid evidence relative path")
    return path


def verify_archive(directory, state):
    manifest = read_json(directory / "source_manifest.json")
    require(bool(manifest), "source snapshot is empty")
    require(aggregate(manifest) == state["source_sha256"], "source snapshot aggregate differs")
    for name, expected in manifest.items():
        require(
            digest(directory / "source" / checked_relative(name)) == expected,
            f"archived source differs: {name}",
        )
    return manifest


def summarize_memory(samples, requests):
    required_stages = {
        "before_model_loading",
        "after_model_loading",
        "after_shared_pool_allocation",
        "after_cleanup",
        *(f"{side}_request_{index}" for index in range(requests) for side in ("before", "after")),
    }
    require(
        required_stages <= {row["stage"] for row in samples},
        "memory boundary samples are incomplete",
    )
    fields = (
        "torch_peak_allocated_bytes",
        "torch_peak_reserved_bytes",
        "cuda_free_bytes",
        "cuda_total_bytes",
    )
    require(
        all(all(name in row for name in fields) for row in samples),
        "CUDA memory observations are incomplete",
    )
    require(
        all(0 <= row["cuda_free_bytes"] <= row["cuda_total_bytes"] for row in samples),
        "invalid device free-memory sample",
    )
    return {
        "torch_peak_allocated_bytes": max(row["torch_peak_allocated_bytes"] for row in samples),
        "torch_peak_reserved_bytes": max(row["torch_peak_reserved_bytes"] for row in samples),
        "minimum_sampled_cuda_free_bytes": min(row["cuda_free_bytes"] for row in samples),
        "maximum_sampled_device_used_bytes": max(
            row["cuda_total_bytes"] - row["cuda_free_bytes"] for row in samples
        ),
        "host_allocated_bytes_peak_counter": max(
            (
                row.get("host_allocator_counters", {}).get("allocated_bytes.peak", 0)
                for row in samples
            ),
            default=0,
        ),
        "process_max_rss_bytes": max(
            (row.get("process_max_rss_bytes", 0) for row in samples), default=0
        ),
        "memory_samples": len(samples),
    }


def read_case(directory):
    state = read_json(directory / "status.json")
    require(state.get("schema") == "echo-capacity-probe-v1", "unknown capacity run schema")
    require(
        state.get("status") == "complete" and state.get("capacity_passed") is True,
        "capacity run has not passed",
    )
    require(state.get("numerical_status") == "passed", "capacity run lacks numerical acceptance")
    config = state["config"]
    require(
        config["scheme"] == "echo" and config["resource_mode"] == "fixed_pools",
        "run is not fixed-pools ECHO",
    )
    require(
        config["hbm_budget_bytes"] is None and config["dram_budget_bytes"] is None,
        "run uses byte sub-budgets",
    )
    persistence = candidate_policy(config)
    retained = config["history_tokens"]
    if persistence == "host_backed":
        retained += config["candidate_tokens"]
    else:
        require(
            config.get("retained_capacity_tokens") == retained, "retained history capacity differs"
        )
    padded = (retained + 63) // 64 * 64
    require(config["padded_session_tokens"] == padded, "session padding differs")
    users = config["host_arena_tokens"] // padded
    require(
        users == config["users"] >= 1 and config["rounds"] >= 2, "user capacity or rounds differ"
    )
    require(
        config["requests"] == users * config["rounds"] == state["completed_requests"],
        "completed request count differs",
    )
    require(
        state["max_retained_users"] == users and state["cleanup"]["passed"],
        "retention or cleanup failed",
    )
    source_manifest = verify_archive(directory, state)
    numerical = read_json(directory / "numerical/manifest.json")
    require(
        numerical.get("schema") == REFERENCE_SCHEMA
        and numerical.get("scheme") == "echo"
        and numerical.get("status") == "complete",
        "ECHO numerical manifest is incomplete",
    )
    binding = reference_binding(
        config,
        state["source_sha256"],
        state["checkpoint"],
        numerical["binding"]["tokenizer_sha256"],
    )
    require(numerical["binding"] == binding, "ECHO numerical binding differs")
    audit = read_json(directory / "numerical_audit.json")
    require(
        audit.get("schema") == "echo-capacity-numerical-audit-v1", "unknown numerical audit schema"
    )
    require(
        audit.get("status") == "passed"
        and audit.get("run_id") == state["run_id"]
        and audit.get("binding") == binding,
        "numerical audit does not bind this run",
    )
    require(len(audit["requests"]) == config["requests"], "numerical audit request count differs")
    require(
        audit["requests_file_sha256"] == digest(directory / "requests.jsonl"),
        "audited request ledger differs",
    )
    reference_directory = Path(audit["reference_directory"])
    reference = load_reference(reference_directory, binding)
    require(reference is not None, "independent HBM reference is unavailable")
    require(
        digest(reference_directory / "manifest.json") == audit["reference_manifest_sha256"],
        "audited HBM reference manifest differs",
    )
    rows = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
    require(len(rows) == config["requests"], "saved request count differs")
    files = {
        name: digest(directory / name)
        for name in (
            "status.json",
            "source_manifest.json",
            "memory.json",
            "requests.jsonl",
            "numerical/manifest.json",
            "numerical_audit.json",
        )
    }
    reference_files = {"manifest.json": digest(reference_directory / "manifest.json")}
    for index, (row, checked) in enumerate(zip(rows, audit["requests"], strict=True)):
        require(
            row["request_id"] == checked["request_id"] == index, "numerical request order differs"
        )
        require(
            checked.get("status") == "passed"
            and checked.get("atol") == checked.get("rtol") == 0
            and all(checked.get(name, {}).get("exact") is True for name in ("hidden", "logits")),
            "numerical audit is not exact for every output",
        )
        check_lifecycle(
            {"request_id": index, "user_id": index % users}, row["metrics"], config, index
        )
        relative = f"numerical/{index:06d}.pt"
        require(row["output_file"] == relative, "saved numerical path differs")
        files[relative] = digest(directory / relative)
        require(files[relative] == row["output_file_sha256"], "saved numerical file digest differs")
        payload = torch.load(directory / relative, weights_only=True, map_location="cpu")
        require(
            payload["request_id"] == index and payload["input_sha256"] == row["input_sha256"],
            "saved output input identity differs",
        )
        for name in ("hidden", "logits"):
            _, summary = tensor_summary(payload[name])
            require(summary == row[name], f"saved {name} tensor digest differs")
        require(
            payload["hidden"].ndim == 2
            and payload["hidden"].shape[0] == config["candidate_tokens"],
            "candidate hidden scope differs",
        )
        require(
            payload["logits"].ndim == 2 and payload["logits"].shape[0] == 1,
            "last-token logits scope differs",
        )
        comparison = compare_reference(reference_directory, reference, index, payload)
        require(comparison["status"] == "passed", "HBM numerical output is unavailable")
        reference_files[f"{index:06d}.pt"] = digest(reference_directory / f"{index:06d}.pt")
    memory = summarize_memory(read_json(directory / "memory.json"), config["requests"])
    summary = {
        "run_id": state["run_id"],
        "status": "measured_passed",
        "P": config["sparse_pool_tokens"],
        "NH": config["host_arena_tokens"],
        "U": users,
        "H": config["history_tokens"],
        "A": config["candidate_tokens"],
        "C": config["chunk_size"],
        "candidate_persistence": persistence,
        "retained_session_tokens": retained,
        "candidate_device_to_host_bytes": 0 if persistence == "gpu_transient" else None,
        "layers": config["num_layers"],
        "rounds": config["rounds"],
        "requests": config["requests"],
        "first_visits": users,
        "revisits": config["requests"] - users,
        "revisit_hits": config["requests"] - users,
        "evictions": 0,
        "numerical_status": "passed",
        "source_sha256": state["source_sha256"],
        "checkpoint_path": state["checkpoint"]["path"],
        "checkpoint_inventory_sha256": aggregate(state["checkpoint"]),
        "gpu": state["hardware"]["name"],
        "total_hbm_bytes": state["hardware"]["total_memory"],
        "model_weights_hbm_bytes": state["backend"]["weights"]["hbm"],
        "torch_version": state["hardware"].get("torch"),
        "cuda_version": state["hardware"].get("cuda"),
        "hidden_dtype": rows[0]["hidden"]["dtype"],
        "logits_dtype": rows[0]["logits"]["dtype"],
        **memory,
    }
    provenance = {
        "run_id": state["run_id"],
        "directory": str(directory.resolve()),
        "files_sha256": files,
        "source_manifest": source_manifest,
        "checkpoint": state["checkpoint"],
        "reference_directory": str(reference_directory.resolve()),
        "reference_files_sha256": reference_files,
    }
    return summary, provenance


def verify_budgets(budgets):
    fraction = Decimal(str(budgets["hbm_fraction"]))
    require(fraction.is_finite() and 0 < fraction <= 1, "invalid HBM fraction")
    limit = int(budgets["total_hbm_bytes"] * fraction)
    raw = limit - budgets["model_hbm_bytes"]
    expected = {
        "fraction_hbm_limit_bytes": limit,
        "raw_hbm_envelope_bytes": raw,
        "hbm_envelope_bytes": max(0, raw),
        "fraction_reserve_bytes": budgets["total_hbm_bytes"] - limit,
        "cache_hbm_budget_bytes": max(0, raw) - budgets["noncache_headroom_bytes"],
    }
    require(
        all(budgets[name] == value for name, value in expected.items()),
        "plan does not use floor(total * fraction) - model",
    )


def verify_ledger(point, budgets):
    require(point["budgets"] == budgets, "point and plan budgets differ")
    hbm, dram = point["hbm"], point["dram"]
    require(
        hbm["reservation_bytes"] == sum(hbm["components"].values()),
        "HBM reservation components do not sum",
    )
    require(
        hbm["allocator_allowance_bytes"]
        == hbm["persistent_allocator_allowance_bytes"] + hbm["scratch_allocator_headroom_bytes"],
        "allocator allowances do not sum",
    )
    require(
        hbm["cache_total_bytes"] == hbm["reservation_bytes"] + hbm["allocator_allowance_bytes"],
        "cache HBM ledger does not sum",
    )
    require(
        hbm["total_bytes"] == hbm["cache_total_bytes"] + budgets["noncache_headroom_bytes"],
        "total HBM ledger differs",
    )
    require(dram["total_bytes"] == sum(dram["components"].values()), "DRAM components do not sum")
    for name, ledger in (("hbm", hbm), ("dram", dram)):
        budget = budgets["hbm_envelope_bytes" if name == "hbm" else "dram_budget_bytes"]
        require(
            ledger["budget_bytes"] == budget
            and ledger["remaining_bytes"] == budget - ledger["total_bytes"],
            f"{name} remaining budget differs",
        )
    persistence = candidate_policy(point)
    context = point["history_tokens"] + point["candidate_tokens"]
    capacity = point["history_tokens"] if persistence == "gpu_transient" else context
    padded = (capacity + 63) // 64 * 64
    require(
        point["session_capacity_tokens"] == capacity and point["session_page_tokens"] == padded,
        "plan session padding differs",
    )
    require(
        point["full_sessions"] == point["host_arena_tokens"] // padded, "plan user capacity differs"
    )
    maximum_slots = (1 << 31) - 1 - 128 * 256
    if persistence == "gpu_transient":
        maximum_slots -= point["candidate_tokens"]
        require(point["execution_context_tokens"] == context, "candidate execution context differs")
        require(
            point["candidate_slots"] == point["candidate_tokens"],
            "candidate GPU tail capacity differs",
        )
        require(
            point["candidate_host_tokens"] == point["candidate_device_to_host_bytes"] == 0,
            "transient candidates reserve host storage or copies",
        )
        require(
            point["maximum_device_slots"] == maximum_slots, "candidate tail indexing limit differs"
        )
        layers, candidate, dimension = (
            point["num_layers"],
            point["candidate_tokens"],
            point["index_head_dim"],
        )
        expected = {
            "session_index_keys": point["full_sessions"] * layers * capacity * dimension,
            "session_index_scales": point["full_sessions"] * layers * capacity * 4,
            "candidate_kv_records": layers * candidate * point["record_width"] * 2,
            "merged_index_keys": context * dimension,
            "merged_index_scales": context * 4,
        }
        require(
            all(hbm["components"].get(name) == size for name, size in expected.items()),
            "history/candidate indexer or KV ledger differs",
        )
    require(
        point["useful_device_slots"]
        == min(point["sparse_pool_tokens"], point["host_arena_tokens"]),
        "useful P differs",
    )
    constraints = []
    if hbm["total_bytes"] > budgets["hbm_envelope_bytes"]:
        constraints.append("hbm_budget")
    if dram["total_bytes"] > budgets["dram_budget_bytes"]:
        constraints.append("dram_budget")
    if point["sparse_pool_tokens"] > maximum_slots:
        constraints.append("native_device_index_limit")
    if point["host_arena_tokens"] > ((1 << 31) - 2) // 64 * 64:
        constraints.append("native_host_index_limit")
    if point["sparse_pool_tokens"] < point["minimum_device_slots"]:
        constraints.append("minimum_device_pool")
    if point["full_sessions"] < 1:
        constraints.append("full_session_fit")
    require(
        point["violated_constraints"] == constraints and point["feasible"] == (not constraints),
        "point feasibility does not match its constraints",
    )


def read_plan(path, run_provenance):
    path = path / "plan.json" if path.is_dir() else path
    data = read_json(path)
    require(
        data.get("schema") == "echo-capacity-plan-v1"
        and data.get("status") == "static_plan_not_execution",
        "unknown static plan schema",
    )
    require(bool(data["sources"]), "plan source identities are empty")
    if "checkpoint_config_sha256" in data:
        require(
            digest(path.parent / "checkpoint_config.json") == data["checkpoint_config_sha256"],
            "static plan checkpoint config differs",
        )
    verified = {}
    for name, expected in data["sources"].items():
        relative = checked_relative(name)
        candidates = [Path(run["directory"]) / "source" / relative for run in run_provenance] + [
            path.parent / "source" / relative,
            ROOT / relative,
        ]
        found = next(
            (
                candidate
                for candidate in candidates
                if candidate.is_file() and digest(candidate) == expected
            ),
            None,
        )
        require(found is not None, f"plan source identity is unavailable: {name}")
        verified[name] = {"sha256": expected, "verified_from": str(found)}
    plan, parameters = data["plan"], data["parameters"]
    budgets = plan["budgets"]
    verify_budgets(budgets)
    require(
        data["hardware"]["total_hbm_bytes"] == budgets["total_hbm_bytes"]
        and data["hardware"]["model_hbm_bytes"] == budgets["model_hbm_bytes"],
        "hardware and budget memory inputs differ",
    )
    require(
        plan["feasible"] is True and plan["selected"] is not None,
        "plan has no feasible selected point",
    )
    selected, following = plan["selected"], plan["next_candidate"]
    verify_ledger(selected, budgets)
    verify_ledger(following, budgets)
    require(
        selected["feasible"] and not following["feasible"],
        "selected point or next candidate violates the search boundary",
    )
    require(
        plan["limiting_constraints"] == following["violated_constraints"],
        "plan limiting constraints differ",
    )
    require(
        plan["validation"]["gpu_execution_performed"] is False
        and plan["validation"]["physical_peak_proven"] is False,
        "static plan claims execution or a measured maximum",
    )
    variable = plan["search_variable"]
    if variable == "NH":
        require(
            selected["sparse_pool_tokens"]
            == following["sparse_pool_tokens"]
            == plan["fixed_P"]
            == parameters["fixed_p"],
            "fixed P changed during NH search",
        )
        require(
            plan["step_tokens"] == selected["session_page_tokens"]
            and following["host_arena_tokens"]
            == selected["host_arena_tokens"] + plan["step_tokens"],
            "next NH is not the next full-user candidate",
        )
        require(
            plan["max_host_arena_tokens"] == selected["host_arena_tokens"]
            and plan["max_full_sessions"] == selected["full_sessions"],
            "selected NH maximum differs",
        )
    elif variable == "P":
        require(
            selected["host_arena_tokens"]
            == following["host_arena_tokens"]
            == plan["fixed_NH"]
            == parameters["fixed_nh"],
            "fixed NH changed during P search",
        )
        require(
            plan["step_tokens"] == 1
            and following["sparse_pool_tokens"] == selected["sparse_pool_tokens"] + 1,
            "next P is not the next allocation candidate",
        )
        require(
            plan["raw_alloc_max_P"] == selected["sparse_pool_tokens"]
            and plan["useful_max_P"] == selected["useful_device_slots"],
            "raw or useful P maximum differs",
        )
        if plan.get("useful_selected") is not None:
            verify_ledger(plan["useful_selected"], budgets)
            require(plan["useful_selected"]["feasible"], "useful P point is not feasible")
    else:
        raise ValueError("unknown plan search variable")
    summary = {
        "run_id": data["run_id"],
        "status": "static_plan_not_execution",
        "search_variable": variable,
        "fixed_P": plan.get("fixed_P"),
        "fixed_NH": plan.get("fixed_NH"),
        "P": selected["sparse_pool_tokens"],
        "NH": selected["host_arena_tokens"],
        "U": selected["full_sessions"],
        "useful_P": selected["useful_device_slots"],
        "next_P": following["sparse_pool_tokens"],
        "next_NH": following["host_arena_tokens"],
        "next_violated_constraints": ",".join(following["violated_constraints"]),
        "H": selected["history_tokens"],
        "A": selected["candidate_tokens"],
        "C": selected["chunk_size"],
        "layers": selected["num_layers"],
        "candidate_persistence": candidate_policy(selected),
        "retained_session_tokens": selected["session_capacity_tokens"],
        "noncache_headroom_note": parameters.get("noncache_headroom_note"),
        "checkpoint_path": parameters["model_path"],
        "sources_sha256": aggregate(data["sources"]),
        "hbm_total_bytes": selected["hbm"]["total_bytes"],
        "dram_total_bytes": selected["dram"]["total_bytes"],
        **budgets,
    }
    return summary, {
        "run_id": data["run_id"],
        "path": str(path.resolve()),
        "sha256": digest(path),
        "sources": verified,
        "source_hashes": data["sources"],
        "plan": plan,
    }


def compatible_budget_case(case, plan, provenance, runs):
    run = next(item for item in runs if item["run_id"] == case["run_id"])
    return (
        candidate_policy(case) == candidate_policy(plan)
        and all(
            case[name] == plan[name]
            for name in ("H", "A", "C", "layers", "checkpoint_path", "total_hbm_bytes")
        )
        and all(
            run["source_manifest"].get(name) == expected
            for name, expected in provenance["source_hashes"].items()
        )
    )


def observe_budget(case, plan):
    """Compare whole-run observations to total HBM times the fraction once."""
    limit = int(plan["total_hbm_bytes"] * Decimal(str(plan["hbm_fraction"])))
    require(limit == plan["fraction_hbm_limit_bytes"], "fraction observation limit differs")
    metrics = {}
    for name in BUDGET_METRICS:
        observed = case[name]
        require(type(observed) is int and observed >= 0, "invalid observed memory value")
        metrics[name] = {
            "observed_bytes": observed,
            "excess_bytes": max(0, observed - limit),
            "remaining_bytes": limit - observed,
            "status": "observed_exceeded" if observed > limit else "no_observed_excess",
        }
    exceeded = [name for name in BUDGET_METRICS if metrics[name]["excess_bytes"] > 0]
    return {
        "run_id": case["run_id"],
        "hbm_fraction": plan["hbm_fraction"],
        "total_hbm_bytes": plan["total_hbm_bytes"],
        "fraction_hbm_limit_bytes": limit,
        "status": "observed_exceeded" if exceeded else "no_observed_excess",
        "exceeded_metrics": exceeded,
        "metrics": metrics,
        "continuous_process_peak_verified": False,
        "comparison_scope": "whole-run observations including model memory against floor(total_hbm_bytes * hbm_fraction); model memory is not subtracted again",
    }


def observation_status(observations):
    if not observations:
        return "unmeasured"
    return (
        "observed_exceeded"
        if any(row["status"] == "observed_exceeded" for row in observations)
        else "no_observed_excess"
    )


def match_runs(plan, provenance, cases, runs):
    def matches(case, p):
        return (
            case["P"] == p
            and case["NH"] == plan["NH"]
            and compatible_budget_case(case, plan, provenance, runs)
        )

    selected = [case for case in cases if matches(case, plan["P"])]
    useful = [case for case in cases if matches(case, plan["useful_P"])]
    plan["measured_selected_run_ids"] = ",".join(case["run_id"] for case in selected)
    plan["measured_useful_run_ids"] = ",".join(case["run_id"] for case in useful)
    selected_observations = [observe_budget(case, plan) for case in selected]
    useful_observations = [observe_budget(case, plan) for case in useful]
    plan["budget_observation_status"] = observation_status(selected_observations)
    plan["useful_budget_observation_status"] = observation_status(useful_observations)
    plan["budget_observations"] = {
        "selected": selected_observations,
        "useful": useful_observations,
    }


def observe_cases(cases, plans, plan_sources, runs):
    """Apply shared fraction policies without claiming a selected-plan match."""
    for case in cases:
        policies = {}
        for plan, provenance in zip(plans, plan_sources, strict=True):
            if not compatible_budget_case(case, plan, provenance, runs):
                continue
            key = (plan["total_hbm_bytes"], plan["hbm_fraction"])
            if key not in policies:
                policies[key] = {
                    **observe_budget(case, plan),
                    "compatible_policy_plan_ids": [],
                    "selected_capacity_plan_ids": [],
                    "useful_capacity_plan_ids": [],
                }
            row = policies[key]
            row["compatible_policy_plan_ids"].append(plan["run_id"])
            if case["run_id"] in plan["measured_selected_run_ids"].split(","):
                row["selected_capacity_plan_ids"].append(plan["run_id"])
            if case["run_id"] in plan["measured_useful_run_ids"].split(","):
                row["useful_capacity_plan_ids"].append(plan["run_id"])
        case["budget_observations"] = list(policies.values())
        case["budget_observation_status"] = observation_status(case["budget_observations"])


def markdown(cases, plans, run_id):
    lines = [
        f"# ECHO 固定容量结果（{run_id}）",
        "",
        "P 是每层 HBM history token pool 容量，NH 是全局 host arena 的逻辑 token 容量。gpu_transient 按 H 保留用户历史，U=floor(NH/padded(H))；候选整批执行，各层 KV 尾部保留 A 行，indexer 只使用当前层 H+A 合并 scratch。旧 host_backed 结果按 H+A 预留，不与新语义混用。",
        "",
    ]
    if cases:
        lines.extend(
            [
                "## 固定 P/NH 的实测",
                "",
                "已列入下表的配置完成全部轮次，保留 U 个用户；全部复访命中，没有用户淘汰。全部候选 hidden 和末 token logits 与独立 HBM 参考精确一致。gpu_transient 还逐请求核对保留长度 H 和候选 D2H=0。",
                "",
                "| Run ID | 候选语义 | P | NH | U | H / A / C | 请求数 | 复访命中 | PyTorch allocated 峰值 GiB | reserved 峰值 GiB |",
                "|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
            ]
        )
        for case in cases:
            lines.append(
                f"| `{case['run_id']}` | {candidate_policy(case)} | {case['P']:,} | {case['NH']:,} | {case['U']:,} | {case['H']} / {case['A']} / {case['C']} | {case['requests']:,} | {case['revisit_hits']:,}/{case['revisits']:,} | {case['torch_peak_allocated_bytes'] / 2**30:.3f} | {case['torch_peak_reserved_bytes'] / 2**30:.3f} |"
            )
        lines.extend(
            [
                "",
                MEMORY_BOUNDARY,
                "",
                "## HBM fraction 预算的实测观察",
                "",
                "下表将完成运行的三项内存观测与 floor(设备总 HBM × fraction) 比较，模型占用已经包含在观测值中，不再扣除。任一指标超额都记为已观察到超额；allocated 未超额不能抵消 reserved 或设备已用显存的超额。未观察到超额只描述现有观测，不保证连续的进程峰值始终满足预算。",
                "",
                "| Run ID | fraction | 总量阈值 GiB | allocated 峰值 GiB | reserved 峰值 GiB | 边界设备已用最大值 GiB | 预算观察 |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for case in cases:
            for observation in case.get("budget_observations", []):
                values = []
                for name in BUDGET_METRICS:
                    metric = observation["metrics"][name]
                    value = f"{metric['observed_bytes'] / 2**30:.3f}"
                    if metric["excess_bytes"]:
                        value += f"（超 {metric['excess_bytes'] / 2**30:.3f}）"
                    values.append(value)
                lines.append(
                    f"| `{case['run_id']}` | {observation['hbm_fraction']} | {observation['fraction_hbm_limit_bytes'] / 2**30:.3f} | {' | '.join(values)} | {BUDGET_LABELS[observation['status']]} |"
                )
        lines.extend(
            [
                "",
                "此处按兼容计划的 fraction 政策比较已完成运行。各计划选中容量是否实测及其预算状态另列在下表，不能把已有运行视为其他容量或新增 headroom 方案的验证。",
            ]
        )
    else:
        lines.append(
            "本次发布只包含静态规划，未包含完整容量运行，也没有新实现的 allocated/reserved 峰值观测。"
        )
    lines.extend(
        [
            "",
            "## 固定一个容量后的规划上界",
            "",
            "HBM 规划额度为 floor(设备总 HBM × fraction) − 模型加载占用；显式 noncache headroom 另扣一次。规划包含 cache 预留、allocator allowance 和 CPU DRAM 预算。每行只改变一个容量，独立求得的 P 与 NH 上界不能拼成已验证组合。",
            "",
        ]
    )
    seen_budgets = set()
    for plan in plans:
        budget = tuple(
            plan[name]
            for name in (
                "total_hbm_bytes",
                "hbm_fraction",
                "model_hbm_bytes",
                "hbm_envelope_bytes",
                "noncache_headroom_bytes",
                "dram_budget_bytes",
            )
        )
        if budget not in seen_budgets:
            total, fraction, model, envelope, headroom, dram = budget
            lines.extend(
                [
                    f"预算输入：设备总 HBM {total / 2**30:.3f} GiB，fraction={fraction}，模型加载占用 {model / 2**30:.3f} GiB；HBM 规划额度 {envelope / 2**30:.3f} GiB，另留 noncache headroom {headroom / 2**30:.3f} GiB；CPU DRAM 预算 {dram / 2**30:.3f} GiB。",
                    "",
                ]
            )
            if plan.get("noncache_headroom_note"):
                lines.extend([f"余量来源：{plan['noncache_headroom_note']}", ""])
            seen_budgets.add(budget)
    lines.extend(
        [
            "| Plan ID | 候选语义 | 固定条件 | 规划 P | 规划 NH | U | 有用 P=min(P,NH) | 下一候选违反 | 所选点实测 Run ID | 所选点预算观察 |",
            "|---|---|---|---:|---:|---:|---:|---|---|---|",
        ]
    )
    notes = []
    for plan in plans:
        fixed = (
            f"P={plan['fixed_P']:,}"
            if plan["search_variable"] == "NH"
            else f"NH={plan['fixed_NH']:,}"
        )
        lines.append(
            f"| `{plan['run_id']}` | {candidate_policy(plan)} | {fixed} | {plan['P']:,} | {plan['NH']:,} | {plan['U']:,} | {plan['useful_P']:,} | {plan['next_violated_constraints']} | {plan['measured_selected_run_ids'] or '未实测'} | {BUDGET_LABELS[plan['budget_observation_status']]} |"
        )
        if plan["P"] > plan["useful_P"]:
            notes.append(
                f"\n`{plan['run_id']}` 的原始 P 规划上界超过 NH，多出的 slots 不增加可驻留的独立 host token 数。有用容量点的实测为：{plan['measured_useful_run_ids'] or '未实测'}；预算观察为{BUDGET_LABELS[plan['useful_budget_observation_status']]}。"
            )
    lines.extend(notes)
    lines.extend(
        [
            "",
            "下一候选违反的是规划约束，不代表实测 OOM。静态规划未证明物理容量最大值。完整执行与数值验收也需和 fraction 预算的实际观测分别判断。",
            "",
            "## 测量范围与来源",
            "",
            "模型是 checkpoint 前三层独立复制出的 10 个 dense block 工作负载替身；副本使用对应源层的 hidden/residual 输入。结果不代表经过训练的 DeepSeek 8B 或完整 61 层模型。",
            "",
            "实测记录的 checkpoint 身份包含路径和 shard 文件 stat，未计算全部权重内容的 hash。HBM 参考仅用于数值验收，不作为 serving 时延对照。"
            if cases
            else "本次使用给定的模型加载占用作为规划输入，未重新加载模型或执行用户轨迹。",
            "",
            "完整字节数、源码身份、checkpoint 路径和测量边界见 [summary.json](summary.json)；表格见 [cases.csv](cases.csv) 与 [plans.csv](plans.csv)。输入文件、源码快照及本报告生成器的 hash 见 [report_provenance.json](report_provenance.json)。",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["run_id", "status"])
        writer.writeheader()
        writer.writerows(
            {
                name: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for name, value in row.items()
            }
            for row in rows
        )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        default=[],
        help="optional completed capacity runs; omit for a static-only report",
    )
    result.add_argument("--plans", type=Path, nargs="+", required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/cache_management/report/capacity",
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    generator = Path(__file__).resolve()
    generator_sha256 = digest(generator)
    require(
        bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id)), "invalid publication run ID"
    )
    require(not args.output_dir.exists(), "report output already exists")
    cases, runs, plans, plan_sources = [], [], [], []
    for directory in args.runs:
        case, provenance = read_case(directory)
        require(all(item["run_id"] != case["run_id"] for item in cases), "duplicate source run ID")
        cases.append(case)
        runs.append(provenance)
    for path in args.plans:
        plan, provenance = read_plan(path, runs)
        require(all(item["run_id"] != plan["run_id"] for item in plans), "duplicate plan run ID")
        match_runs(plan, provenance, cases, runs)
        plans.append(plan)
        plan_sources.append(provenance)
    observe_cases(cases, plans, plan_sources, runs)
    summary = {
        "schema": "echo-capacity-report-v1",
        "run_id": args.run_id,
        "cases": cases,
        "plans": plans,
        "memory_boundary": MEMORY_BOUNDARY,
        "maximum_scope": "independent static one-variable maxima; matched measured points do not establish physical maxima",
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="echo-capacity-report-") as temporary:
        output = Path(temporary) / "report"
        output.mkdir()
        write_json(output / "summary.json", summary)
        write_csv(output / "cases.csv", cases)
        write_csv(output / "plans.csv", plans)
        (output / "results.md").write_text(markdown(cases, plans, args.run_id))
        require(digest(generator) == generator_sha256, "report generator changed during validation")
        write_json(
            output / "report_provenance.json",
            {
                "schema": "echo-capacity-report-provenance-v1",
                "run_id": args.run_id,
                "created_unix": time.time(),
                "generator": {
                    "path": str(generator),
                    "sha256": generator_sha256,
                },
                "source_run_ids": [row["run_id"] for row in cases],
                "plan_run_ids": [row["run_id"] for row in plans],
                "runs": runs,
                "plans": plan_sources,
                "output_files_sha256": {
                    path.name: digest(path) for path in sorted(output.iterdir())
                },
            },
        )
        require(not args.output_dir.exists(), "report output appeared during validation")
        shutil.move(str(output), str(args.output_dir))
    print(f"Published {len(cases)} verified runs and {len(plans)} static plans: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
