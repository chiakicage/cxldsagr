"""Simulate one layer from measured compute and KV transfer service times."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from itertools import pairwise
from pathlib import Path

from experiments.deepseek_v32_motivation.src.simulation_inputs import (
    WITHDRAWN_PROFILE_RUN_ID,
    digest,
)

STAGES = ("projection", "index", "topk", "attention", "finish")
LABELS = {
    "projection": "Norm + projection",
    "index": "Indexer",
    "topk": "Exact top-k",
    "attention": "Sparse MLA",
    "finish": "Output + norm + MLP",
}


def _event(events, lane, stage, start, duration, label=None):
    if duration < 0:
        raise ValueError("negative service time")
    if duration:
        events.append(
            {
                "lane": lane,
                "stage": stage,
                "start_ns": start,
                "end_ns": start + duration,
                "label": label or LABELS.get(stage, stage),
            }
        )
    return start + duration


def _compute(events, costs, stages, start=0):
    for stage in stages:
        start = _event(events, "compute", stage, start, costs[stage])
    return start


def _scenario(identifier, phase, label, events, compute_end, io_end, boundary):
    return {
        "id": identifier,
        "phase": phase,
        "label": label,
        "events": events,
        "completion_ns": max(compute_end, io_end),
        "compute_ready_ns": compute_end,
        "io_ready_ns": io_end,
        "boundary": boundary,
    }


def _dense_prefetch(extend):
    """Chain DMA across L0/L1/L2, then crop the view at L1 P and L1 output."""
    context = extend["dense_pipeline_context"]
    previous = context["compute"]["stages_ns"]
    previous_dma = context["h2d"]["duration_ns"]
    current_dma = extend["dense_h2d"]["duration_ns"]
    next_dma = extend["dense_next_h2d"]["duration_ns"]
    if any(type(value) is not int or value < 0 for value in (previous_dma, current_dma, next_dma)):
        raise ValueError("negative or invalid dense service time")
    if set(previous) != set(STAGES) or any(type(v) is not int or v <= 0 for v in previous.values()):
        raise ValueError("expected five positive previous-layer compute costs")
    if sum(previous.values()) != context["compute"]["total_ns"]:
        raise ValueError("previous-layer costs do not conserve total compute")
    previous_prefix = sum(previous[stage] for stage in STAGES[:3])
    previous_tail = previous["attention"] + previous["finish"]
    # L0 P and DMA start together. L1 P starts after L0's dependent compute.
    previous_completion = max(previous_prefix, previous_dma) + previous_tail
    lead = previous_completion - previous_dma
    current_start = -lead
    current_end = current_start + current_dma
    full_transfers = [
        {
            "transfer_role": role,
            "layer": context["previous_layer"] + offset,
            "start_ns": start,
            "end_ns": start + duration,
            "duration_ns": duration,
            "label": label,
        }
        for role, offset, start, duration, label in (
            ("current_layer", 1, current_start, current_dma, "L1 fetch remainder"),
            ("next_layer_context", 2, current_end, next_dma, "L2 fetch beginning"),
        )
    ]
    events = []
    costs = extend["compute"]["stages_ns"]
    prefix_end = _compute(events, costs, STAGES[:3])
    wait = max(0, current_end - prefix_end)
    attention_start = _event(events, "wait", "wait", prefix_end, wait, "Wait for KV")
    compute_end = _compute(events, costs, STAGES[3:], attention_start)
    for transfer in full_transfers:
        start = max(0, transfer["start_ns"])
        end = min(compute_end, transfer["end_ns"])
        if end <= start:
            continue
        _event(events, "io", "h2d", start, end - start, transfer["label"])
        events[-1].update(
            {
                "transfer_role": transfer["transfer_role"],
                "layer": transfer["layer"],
                "full_start_ns": transfer["start_ns"],
                "full_end_ns": transfer["end_ns"],
                "full_duration_ns": transfer["duration_ns"],
            }
        )
    row = _scenario(
        "extend_dense_prefetch",
        "extend",
        "Dense prefetch: chained across layers",
        events,
        compute_end,
        max(0, current_end),
        "L1 P starts at t=0; L1 DMA follows L0 DMA. L2 DMA is context only, cropped at L1 output.",
    )
    row.update(
        {
            "full_transfers": full_transfers,
            "previous_layer": context["previous_layer"],
            "previous_projection_start_ns": -previous_completion,
            "previous_fetch_end_ns": -lead,
            "prefetch_lead_ns": lead,
            "fetch_service_before_zero_ns": min(lead, current_dma),
            "wait_ns": wait,
        }
    )
    return row


def simulate(inputs):
    """Pack measured costs, preserving the declared data dependencies.

    There is no CPU/cache work or inherited gap. Only a modeled data dependency
    can stall compute. The oracle scenario alone relaxes selection availability.
    """
    scenarios = []
    prefill, extend = inputs["prefill"], inputs["extend"]
    for phase in (prefill, extend):
        costs = phase["compute"]["stages_ns"]
        if set(costs) != set(STAGES) or any(type(v) is not int or v <= 0 for v in costs.values()):
            raise ValueError("expected five positive integer compute costs in ns")
        if sum(costs.values()) != phase["compute"]["total_ns"]:
            raise ValueError("stage costs do not conserve total compute")

    costs = prefill["compute"]["stages_ns"]
    writeback = prefill["d2h"]["duration_ns"]
    for identifier, label in (
        ("prefill_hbm", "HBM resident"),
        ("prefill_serial", "Offload: serial writeback"),
        ("prefill_overlap", "Offload: overlapped writeback"),
    ):
        events = []
        time = _compute(events, costs, ("projection",))
        io_end = 0
        if identifier != "prefill_hbm":
            io_end = _event(events, "io", "d2h", time, writeback, "Main KV D2H")
        if identifier == "prefill_serial":
            time = _event(events, "wait", "wait", time, writeback, "Wait for D2H")
        compute_end = _compute(events, costs, STAGES[1:], time)
        scenarios.append(
            _scenario(
                identifier,
                "prefill",
                label,
                events,
                compute_end,
                io_end,
                "Selected chunk L1 output and its persistent KV writeback are complete.",
            )
        )

    costs = extend["compute"]["stages_ns"]
    sparse = extend["sparse_h2d"]["duration_ns"]
    for identifier, label in (
        ("extend_hbm", "HBM resident"),
        ("extend_serial_sparse", "Serial sparse fetch"),
        ("extend_oracle_sparse", "Ideal sparse overlap (oracle)"),
    ):
        events = []
        io_end = 0
        time = _compute(events, costs, ("projection",))
        if identifier == "extend_oracle_sparse":
            io_end = _event(events, "io", "h2d", time, sparse, "Exact-set H2D (oracle)")
        time = _compute(events, costs, ("index",), time)
        if identifier == "extend_oracle_sparse":
            time = _event(events, "wait", "wait", time, max(0, io_end - time), "IO tail")
        time = _compute(events, costs, ("topk",), time)
        if identifier == "extend_serial_sparse":
            io_end = _event(events, "io", "h2d", time, sparse, "Selected KV H2D")
        if identifier == "extend_serial_sparse":
            time = _event(events, "wait", "wait", time, max(0, io_end - time), "Wait for KV")
        compute_end = _compute(events, costs, ("attention", "finish"), time)
        scenarios.append(
            _scenario(
                identifier,
                "extend",
                label,
                events,
                compute_end,
                io_end,
                "L1 input available at t=0 through L1 output ready; candidate KV is discarded.",
            )
        )

    if "echo" in extend:
        echo = extend["echo"]
        events = []
        time = _compute(events, costs, ("projection",))
        time = _event(
            events, "fused", "fused", time, echo["fused"]["duration_ns"], "Indexer + fused prefetch"
        )
        time = _event(
            events, "compute", "cleanup", time, echo["cleanup"]["duration_ns"], "Causal score mask"
        )
        time = _compute(events, costs, ("topk",), time)
        io_end = _event(
            events, "io", "h2d", time, echo["recall"]["duration_ns"], "Residual exact recall"
        )
        time = _event(events, "wait", "wait", time, io_end - time, "Wait for residual KV")
        compute_end = _compute(events, costs, ("attention", "finish"), time)
        scenarios.append(
            _scenario(
                "extend_echo_measured",
                "extend",
                "ECHO: measured fused + recall",
                events,
                compute_end,
                io_end,
                "Common HBM P/K/A/F plus measured ECHO fused kernel, causal mask and residual recall; internal fusion overlap is unknown.",
            )
        )

    scenarios.append(_dense_prefetch(extend))
    result = {"schema_version": 2, "run_id": inputs["run_id"], "scenarios": scenarios}
    if "source" in inputs:
        result["plot_source_note"] = inputs["source"]["profile_run_id"]
    result["audit"] = audit_simulation(result, inputs)
    if all("work" in inputs[phase]["compute"] for phase in ("prefill", "extend")):
        add_metrics(result, inputs)
    return result


def audit_simulation(document, inputs):
    transfer_costs = {
        "prefill_serial": inputs["prefill"]["d2h"]["duration_ns"],
        "prefill_overlap": inputs["prefill"]["d2h"]["duration_ns"],
        "extend_serial_sparse": inputs["extend"]["sparse_h2d"]["duration_ns"],
        "extend_oracle_sparse": inputs["extend"]["sparse_h2d"]["duration_ns"],
    }
    if "echo" in inputs["extend"]:
        transfer_costs["extend_echo_measured"] = inputs["extend"]["echo"]["recall"]["duration_ns"]
    checks = []
    for scenario in document["scenarios"]:
        events = scenario["events"]
        costs = inputs[scenario["phase"]]["compute"]["stages_ns"]
        compute = [event for event in events if event["lane"] in {"compute", "fused"}]
        expected_stages = list(STAGES)
        if scenario["id"] == "extend_echo_measured":
            echo = inputs["extend"]["echo"]
            costs = {
                **costs,
                "fused": echo["fused"]["duration_ns"],
                "cleanup": echo["cleanup"]["duration_ns"],
            }
            expected_stages = ["projection", "fused", "cleanup", "topk", "attention", "finish"]
        if [event["stage"] for event in compute] != expected_stages:
            raise ValueError("compute order or coverage changed")
        for event in compute:
            if event["end_ns"] - event["start_ns"] != costs[event["stage"]]:
                raise ValueError("simulation changed a measured compute cost")
        if any(a["end_ns"] > b["start_ns"] for a, b in pairwise(compute)):
            raise ValueError("dependent compute stages overlap")
        if max(event["end_ns"] for event in events) != scenario["completion_ns"]:
            raise ValueError("completion truncates an event")
        if any(event["start_ns"] < 0 for event in events):
            raise ValueError("simulation begins before inputs are available")
        if compute[0]["start_ns"] != 0:
            raise ValueError("time origin must remain the current layer projection start")
        io = [event for event in events if event["lane"] == "io"]
        if scenario["id"] == "extend_dense_prefetch":
            _audit_dense_prefetch(scenario, inputs["extend"])
        elif sum(e["end_ns"] - e["start_ns"] for e in io) != transfer_costs.get(scenario["id"], 0):
            raise ValueError("simulation changed a measured transfer cost")
        stages = {event["stage"]: event for event in compute}
        if io:
            transfer = io[0]
            identifier = scenario["id"]
            if (
                identifier.startswith("prefill")
                and transfer["start_ns"] < stages["projection"]["end_ns"]
            ):
                raise ValueError("writeback precedes KV production")
            if (
                identifier in {"extend_serial_sparse", "extend_echo_measured"}
                and transfer["start_ns"] < stages["topk"]["end_ns"]
            ):
                raise ValueError("serial transfer precedes exact selection")
            if (
                identifier in {"extend_serial_sparse", "extend_echo_measured"}
                and stages["attention"]["start_ns"] < transfer["end_ns"]
            ):
                raise ValueError("attention consumes unavailable KV")
            if (
                identifier == "extend_oracle_sparse"
                and stages["topk"]["start_ns"] < transfer["end_ns"]
            ):
                raise ValueError("oracle scenario does not join indexer and transfer")
        checks.append(
            {
                "scenario": scenario["id"],
                "compute_and_io_costs_conserved": True,
                "dependencies_checked": True,
            }
        )
    return {"status": "passed", "scenarios": checks}


def _audit_dense_prefetch(scenario, extend):
    context = extend["dense_pipeline_context"]
    previous = context["compute"]["stages_ns"]
    previous_dma = context["h2d"]["duration_ns"]
    previous_prefix = sum(previous[stage] for stage in STAGES[:3])
    previous_tail = sum(previous[stage] for stage in STAGES[3:])
    origin = max(previous_prefix, previous_dma) + previous_tail
    if scenario["previous_projection_start_ns"] != -origin:
        raise ValueError("previous layer does not establish the L1 P origin")
    lead = origin - previous_dma
    if scenario["prefetch_lead_ns"] != lead or scenario["previous_fetch_end_ns"] != -lead:
        raise ValueError("dense prefetch lead differs from preceding fetch completion")
    transfers = scenario["full_transfers"]
    if len(transfers) != 2:
        raise ValueError("dense pipeline must retain current and next full transfer metadata")
    current, following = transfers
    if current["start_ns"] != -lead or following["start_ns"] != current["end_ns"]:
        raise ValueError("dense transfers are not chained after the previous fetch")
    expected_io = []
    for transfer, role, key, offset in (
        (current, "current_layer", "dense_h2d", 1),
        (following, "next_layer_context", "dense_next_h2d", 2),
    ):
        duration = extend[key]["duration_ns"]
        if (
            transfer["end_ns"] - transfer["start_ns"] != duration
            or transfer["duration_ns"] != duration
            or transfer["transfer_role"] != role
            or transfer["layer"] != context["previous_layer"] + offset
        ):
            raise ValueError("dense full-transfer duration or ownership changed")
        start = max(0, transfer["start_ns"])
        end = min(scenario["completion_ns"], transfer["end_ns"])
        if end > start:
            expected_io.append(
                (role, start, end, transfer["start_ns"], transfer["end_ns"], duration)
            )
    actual_io = [
        (
            e["transfer_role"],
            e["start_ns"],
            e["end_ns"],
            e["full_start_ns"],
            e["full_end_ns"],
            e["full_duration_ns"],
        )
        for e in scenario["events"]
        if e["lane"] == "io"
    ]
    if actual_io != expected_io:
        raise ValueError("visible dense IO does not match full transfers cropped to the L1 window")
    costs = extend["compute"]["stages_ns"]
    prefix = sum(costs[stage] for stage in STAGES[:3])
    tail = sum(costs[stage] for stage in STAGES[3:])
    attention = next(e for e in scenario["events"] if e["stage"] == "attention")
    if (
        attention["start_ns"] != max(prefix, current["end_ns"])
        or scenario["io_ready_ns"] != max(0, current["end_ns"])
        or scenario["wait_ns"] != max(0, current["end_ns"] - prefix)
        or scenario["compute_ready_ns"] != attention["start_ns"] + tail
        or scenario["completion_ns"] != scenario["compute_ready_ns"]
        or scenario["fetch_service_before_zero_ns"] != min(lead, current["duration_ns"])
    ):
        raise ValueError("dense L1 readiness or completion boundary is inconsistent")


def compute_metrics(compute):
    """Normalize work at each precision before combining it, never average MFU."""
    work = compute["work"]
    rows = {}
    for stage in (*STAGES, "total"):
        entry = work["total"] if stage == "total" else work["stages"][stage]
        duration = compute["total_ns"] if stage == "total" else compute["stages_ns"][stage]
        by_precision = entry["flops_by_precision"]
        for precision, flops in by_precision.items():
            peak = work["dense_peaks_tflops"][precision]
            if not math.isfinite(peak) or peak <= 0 or type(flops) is not int or flops <= 0:
                raise ValueError("invalid precision-specific peak or matrix FLOPs")
        ideal = (
            sum(
                flops / (work["dense_peaks_tflops"][precision] * 1000)
                for precision, flops in by_precision.items()
            )
            if by_precision
            else None
        )
        if ideal is not None and not math.isclose(ideal, entry["ideal_compute_ns"], rel_tol=1e-12):
            raise ValueError("normalized matrix work does not match the input ledger")
        rows[stage] = {
            "stage": stage,
            "duration_ns": duration,
            "useful_flops": sum(by_precision.values()) if by_precision else None,
            "flops_by_precision": by_precision,
            "ideal_compute_ns": ideal,
            "compute_mfu_percent": ideal / duration * 100 if ideal is not None else None,
        }
    total_flops = Counter()
    for stage in STAGES:
        total_flops.update(rows[stage]["flops_by_precision"])
    if dict(total_flops) != rows["total"]["flops_by_precision"]:
        raise ValueError("total FLOPs differ from stage work")
    return rows


def transfer_metrics(transfer):
    """Decimal GB/s equals bytes/ns; never divide by the unhidden IO tail."""
    duration, byte_count = transfer["duration_ns"], transfer["bytes"]
    if (
        type(byte_count) is not int
        or byte_count < 0
        or duration < 0
        or byte_count > 0
        and duration == 0
    ):
        raise ValueError("invalid transfer bytes/duration")
    return {
        "bytes": byte_count,
        "MiB": byte_count / 2**20,
        "duration_ns": duration,
        "bandwidth_GBps": byte_count / duration if duration > 0 and byte_count > 0 else None,
    }


def add_metrics(document, inputs):
    phase_metrics = {
        phase: compute_metrics(inputs[phase]["compute"]) for phase in ("prefill", "extend")
    }
    transfers = {
        "prefill_d2h": ("prefill", inputs["prefill"]["d2h"]),
        "sparse_h2d": ("extend", inputs["extend"]["sparse_h2d"]),
        "dense_h2d": ("extend", inputs["extend"]["dense_h2d"]),
        "dense_next_h2d": ("extend", inputs["extend"]["dense_next_h2d"]),
    }
    if "echo" in inputs["extend"]:
        transfers["echo_recall"] = ("extend", inputs["extend"]["echo"]["recall"])
    io_metrics = {
        key: {
            "phase": phase,
            "source_scheme": transfer["source_scheme"],
            "layer": transfer["layer"],
            "records": transfer["records"],
            **transfer_metrics(transfer),
        }
        for key, (phase, transfer) in transfers.items()
    }
    by_scenario = {
        "prefill_serial": "prefill_d2h",
        "prefill_overlap": "prefill_d2h",
        "extend_serial_sparse": "sparse_h2d",
        "extend_oracle_sparse": "sparse_h2d",
        "extend_dense_prefetch": "dense_h2d",
        "extend_echo_measured": "echo_recall",
    }
    for row in document["scenarios"]:
        metrics = phase_metrics[row["phase"]]
        is_echo = row["id"] == "extend_echo_measured"
        row["compute_mfu_percent"] = None if is_echo else metrics["total"]["compute_mfu_percent"]
        row["schedule_mfu_percent"] = (
            metrics["total"]["ideal_compute_ns"] / row["completion_ns"] * 100
        )
        io = io_metrics.get(by_scenario.get(row["id"]))
        row["io_bytes"] = io["bytes"] if io else 0
        row["standalone_io_bandwidth_GBps"] = io["bandwidth_GBps"] if io else None
        for transfer in row.get("full_transfers", []):
            key = "dense_h2d" if transfer["transfer_role"] == "current_layer" else "dense_next_h2d"
            transfer.update(transfer_metrics(inputs["extend"][key]))
        for event in row["events"]:
            if event["lane"] == "compute":
                event["compute_mfu_percent"] = metrics.get(event["stage"], {}).get(
                    "compute_mfu_percent"
                )
            elif event["lane"] == "io":
                if "transfer_role" in event:
                    key = (
                        "dense_h2d"
                        if event["transfer_role"] == "current_layer"
                        else "dense_next_h2d"
                    )
                    full = io_metrics[key]
                    event.update(
                        {"full_bytes": full["bytes"], "full_bandwidth_GBps": full["bandwidth_GBps"]}
                    )
                else:
                    event.update({"bytes": io["bytes"], "bandwidth_GBps": io["bandwidth_GBps"]})
            elif event["lane"] == "fused":
                echo = inputs["extend"]["echo"]
                fused = echo["fused"]
                event.update(
                    {
                        "bytes": fused["bytes"],
                        "fused_payload_GBps": fused["bytes"] / fused["duration_ns"],
                        "compute_mfu_percent": fused["ideal_compute_ns"]
                        / fused["duration_ns"]
                        * 100,
                    }
                )
                row["io_bytes"] += fused["bytes"]
                row["fused_prefetch_fraction"] = echo["coverage_fraction"]
                row["fused_payload_GBps"] = event["fused_payload_GBps"]
    if "echo" in inputs["extend"]:
        fused = inputs["extend"]["echo"]["fused"]
        io_metrics["echo_fused"] = {
            "phase": "extend",
            "source_scheme": "echo",
            "layer": 1,
            "records": fused["records"],
            "bytes": fused["bytes"],
            "MiB": fused["bytes"] / 2**20,
            "duration_ns": fused["duration_ns"],
            "bandwidth_GBps": None,
            "fused_payload_GBps": fused["bytes"] / fused["duration_ns"],
            "boundary": "Whole fused compute+IO kernel; independent IO duration and bandwidth unknown.",
        }
    document["metrics"] = {"compute": phase_metrics, "io": io_metrics}


def _save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row))
        )
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def metrics_report(inputs, simulation):
    compute = simulation["metrics"]["compute"]
    percent = lambda value: "N/A" if value is None else f"{value:.2f}%"
    lines = [
        "## 计算 MFU 与通信带宽",
        "",
        "FLOPs 来自与图中 L1/chunk 对应的调用账本，每个 prefill/extend 样本匹配 14 个矩阵 API，FMA 计 2 FLOPs。Graph API 通过 replay 归属匹配；prefill 使用 chunk 63 的实际 causal pair 数，没有把完整 prefill 工作量除以 64。",
        "混合精度先分别归一化：T_ideal=Σ(FLOPs_d/Peak_d)，纯计算 MFU=T_ideal/计算 kernel 时长之和。FP8、BF16、FP32 的 H200 dense peak 分别为 1,979、989.5、67 TFLOP/s，来源为 [NVIDIA H200 规格](https://www.nvidia.com/en-us/data-center/h200/)。不把混合 FLOPs 全部除以 FP8 峰值，也不对逐算子 MFU 求算术平均。",
        "",
        "| 计算阶段 | Prefill GFLOPs | Prefill 计算 MFU | Extend GFLOPs | Extend 计算 MFU |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for stage in (*STAGES, "total"):
        p, e = compute["prefill"][stage], compute["extend"][stage]
        pf = "N/A" if p["useful_flops"] is None else f"{p['useful_flops'] / 1e9:.6f}"
        ef = "N/A" if e["useful_flops"] is None else f"{e['useful_flops'] / 1e9:.6f}"
        lines.append(
            f"| {LABELS.get(stage, '全部计算')} | {pf} | {percent(p['compute_mfu_percent'])} | {ef} | {percent(e['compute_mfu_percent'])} |"
        )
    lines.extend(
        [
            "",
            "Top-k 没有矩阵 FLOPs，MFU 为 N/A；它的时间仍计入全部计算分母。Norm、RoPE、量化、softmax 等非矩阵工作也不增加 useful matrix FLOPs，但其 kernel 时间保留。这里的 MFU 不等于 NCU 的硬件利用率。",
            "",
            "通信量统计各层完整搬运的主 KV payload；indexer 保持驻留，控制传输、D2D 和 candidate D2H 不计入。有效带宽=字节数/完整搬运区间，GB/s 使用十进制，MiB 使用 2²⁰ B。即使搬运被隐藏或在图中截断，带宽分母仍是完整搬运时长，不能改用暴露的等待时间或可见片段。它不是链路含协议开销的物理吞吐。",
            "",
            "| 搬运 | Records | Payload B | MiB | 完整区间 ms | 有效带宽 GB/s |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    names = {
        "prefill_d2h": "Prefill L1 D2H",
        "sparse_h2d": "Extend sparse L1 H2D",
        "dense_h2d": "Extend dense L1 H2D",
        "dense_next_h2d": "Dense L2 H2D（相邻层上下文）",
        "echo_recall": "ECHO residual recall H2D",
        "echo_fused": "ECHO fused prefetch（整个融合区间）",
    }
    for key, row in simulation["metrics"]["io"].items():
        bandwidth = "N/A" if row["bandwidth_GBps"] is None else f"{row['bandwidth_GBps']:.3f}"
        lines.append(
            f"| {names[key]} | {row['records']:,} | {row['bytes']:,} | {row['MiB']:.6f} | {row['duration_ns'] / 1e6:.6f} | {bandwidth} |"
        )
    lines.extend(
        [
            "",
            "图中另外标出调度 MFU=T_ideal/模拟完成时间，包含必要的 I/O 等待，与纯计算 MFU 分开。Dense 的分子和完成时间均对应 L1；L2 搬运仅用于说明相邻层的衔接，不计入 L1 的通信量或 MFU。截断片段不按时间比例估算字节数。完整精度分量和数值见 [计算指标](compute_metrics.csv)及[通信指标](io_metrics.csv)。",
            "",
        ]
    )
    return lines


def echo_report(inputs, simulation):
    echo = inputs["extend"]["echo"]
    row = next(r for r in simulation["scenarios"] if r["id"] == "extend_echo_measured")
    fused = echo["fused"]
    recall = echo["recall"]
    return [
        "## 已测 ECHO fused prefetch 与 recall",
        "",
        "图中的 ECHO 行保留本次 profile 的融合 indexer/prefetch、causal mask 和剩余精确 recall 时长，其他 P/K/A/F 阶段沿用公共 HBM 成本。图外的 cache 管理及 CPU launch 已排除；融合 kernel 内部的管理和同步无法从现有计时中剥离。因此整行是按已测组件重排的 simulation，不是 ECHO 完整层延迟。这里的“已测”指上述 run ID；不替代后续代码版本的补测。",
        "",
        f"融合 kernel 耗时 {fused['duration_ns'] / 1e6:.6f} ms，预取 {fused['records']:,} records（{fused['bytes']:,} B），覆盖本层 {echo['coverage_fraction'] * 100:.2f}% 的历史 miss；后续 recall 补齐 {recall['records']:,} records（{recall['bytes']:,} B），耗时 {recall['duration_ns'] / 1e6:.6f} ms。两部分合计 {row['io_bytes']:,} B，与串行 sparse 的历史 H2D 一致。Candidate 的 128 records 已在 GPU，不计入历史预取覆盖率。",
        "",
        f"融合区间的 MFU 为 {fused['ideal_compute_ns'] / fused['duration_ns'] * 100:.2f}%，分母包含融合计算与搬运。其平均载荷率为 {fused['bytes'] / fused['duration_ns']:.3f} GB/s，分母同样是完整 fused kernel；内部 IO 时间不可分解，故不能称为独立通信带宽。图上使用一个跨计算/I/O 两条 lane 的区间，不虚构内部 overlap 比例。",
        "",
        f"融合 kernel 后的 causal score mask 是必要计算，单独保留 {echo['cleanup']['duration_ns'] / 1e3:.3f} µs，不增加矩阵 FLOPs。此行模拟完成时间为 {row['completion_ns'] / 1e6:.6f} ms，调度 MFU 为 {row['schedule_mfu_percent']:.2f}%；纯计算 MFU 因融合区间无法拆分而记为 N/A。Prefetch 减少了后续 recall 的通信量，但不能仅凭覆盖率推出整层加速。",
        "",
    ]


def dense_report(inputs, simulation):
    row = next(r for r in simulation["scenarios"] if r["id"] == "extend_dense_prefetch")
    previous = inputs["extend"]["dense_pipeline_context"]
    previous_costs = previous["compute"]["stages_ns"]
    prefix0 = sum(previous_costs[stage] for stage in STAGES[:3])
    tail0 = sum(previous_costs[stage] for stage in STAGES[3:])
    current, following = row["full_transfers"]
    ms = lambda value: f"{value / 1e6:.6f}"
    return [
        "## Dense 跨层预取的时间基准",
        "",
        "横轴零点始终是本层 L1 的 P 开始。Dense 在前一层 fetch 结束后立即启动本层 fetch；L1 fetch 结束后又立即启动 L2 fetch。图只显示 L1 P 开始到 L1 输出就绪的窗口，因此左侧为 L1 fetch 的剩余部分，右侧为 L2 fetch 的开头。",
        "",
        f"使用同一 MFU profile 的 L0 实测成本确定零点前的进度：P₀+I₀+K₀={ms(prefix0)} ms，A₀+F₀={ms(tail0)} ms，完整 DMA D₀={ms(previous['h2d']['duration_ns'])} ms。令 L0 P 与 L0 DMA 同时开始，则 L0 完成于 max(P₀+I₀+K₀,D₀)+A₀+F₀；L1 fetch 比 L1 P 提前 Δ=max(P₀+I₀+K₀−D₀,0)+A₀+F₀={ms(row['prefetch_lead_ns'])} ms 启动。L0 的逐算子独占时长、调用数和 kernel 汇总已交叉核对，选定源行随 inputs.json 保存。",
        "",
        "L1 的计算仍按 P→I→K→A→F 执行，A 等待本层 fetch。因此 L1 完成时间 T=max(p+i+k,D₁−Δ)+a+f，等待时间为 max(D₁−Δ−p−i−k,0)。这里 Δ 来自相邻实测层，没有假定各层计算时间相同。",
        "",
        f"L1 完整 fetch 区间为 [{ms(current['start_ns'])}, {ms(current['end_ns'])}] ms，图中可见 [{ms(max(0, current['start_ns']))}, {ms(current['end_ns'])}] ms；KV 等待为 {ms(row['wait_ns'])} ms，L1 输出在 {ms(row['completion_ns'])} ms 就绪。L2 fetch 从 {ms(following['start_ns'])} ms 开始，图中仅保留至 L1 输出的 {ms(row['completion_ns'] - following['start_ns'])} ms；其完整结束时刻 {ms(following['end_ns'])} ms 不延长 L1 窗口。",
        "",
        f"L1 通信量仍为完整 {current['MiB']:g} MiB，带宽仍用完整 D₁={ms(current['duration_ns'])} ms 计算。本层 fetch 提前启动，减少了本层等待。图中从零点开始显示 L1 fetch 的剩余部分，末尾接上 L2 fetch 的开头。L0 和 L1 的 A+F 时长略有差异，因此该窗口不等于假定同构层的稳态周期，也不代表完整模型吞吐。",
        "",
    ]


def write_report(output, inputs, simulation):
    by_id = {row["id"]: row for row in simulation["scenarios"]}
    pc = inputs["prefill"]["compute"]["stages_ns"]
    ec = inputs["extend"]["compute"]["stages_ns"]
    ms = lambda value: f"{value / 1e6:.6f}"
    sparse_bytes = inputs["extend"]["sparse_h2d"]["bytes"]
    dense_bytes = inputs["extend"]["dense_h2d"]["bytes"]
    text = [
        "# 单层计算与 I/O simulation",
        "",
        f"Simulation run ID：`{inputs['run_id']}`。测量输入来自 `{inputs['source']['profile_run_id']}`，平台为 H200 / SM90。",
        "源实验的环境、依赖与独立验收见 [MFU 说明](../../../deepseek_v32_mfu/README.md)；选定区间、来源文件及哈希保存在本报告的 `inputs.json` 中。",
        "",
        "按 motivation 的 H=65,536、A=128、history chunk=1,024 设置，选从 0 编号的 L1。Prefill 只取最后一个 chunk（63），此前已有 64,512 tokens；extend 使用完整 128-token batch。",
        "计时输入来自 MFU 实验中真实 checkpoint 前三层连续执行的 L1，同一形状不表示与 C10 的某个 GR 请求有相同 token 或激活。Motivation 的十 block、P=65,536、NH=16,777,216 作为目标背景；本次只模拟一层，不运行准入或容量轨迹，也不推算完整十层或全部 64 个 chunk。",
        "",
        "## 实测成本与排除范围",
        "",
        "解析方案共用 HBM 路径的计算 kernel 时长；ECHO 行单独替换其已测 fused indexer、causal mask 和 recall，其他阶段仍用公共成本。计算按阶段累加 kernel duration，保留 norm、RoPE、量化、top-k 等计算辅助操作。去掉 CPU launch、CPU scope、独立的 cache 管理、映射、D2D、memset 和原时间线的空隙；融合 kernel 内部操作随整体时长保留。因 KV 尚未到齐产生的依赖等待仍保留。",
        "",
        "| 成本 | Prefill L1 / chunk 63 ms | Extend L1 ms |",
        "| --- | ---: | ---: |",
    ]
    for stage in STAGES:
        text.append(f"| {LABELS[stage]} | {ms(pc[stage])} | {ms(ec[stage])} |")
    text.extend(
        [
            f"| 计算合计 C | {ms(sum(pc.values()))} | {ms(sum(ec.values()))} |",
            f"| 主 KV D2H W | {ms(inputs['prefill']['d2h']['duration_ns'])} | 0（candidate discard） |",
            f"| 稀疏 H2D S | 0 | {ms(inputs['extend']['sparse_h2d']['duration_ns'])} |",
            f"| 当前层完整历史 H2D D₁ | 0 | {ms(inputs['extend']['dense_h2d']['duration_ns'])} |",
            "",
            f"Prefill 的 D2H 为 1,179,648 B（1.125 MiB）；sparse extend 为 {sparse_bytes:,} B，dense 为 {dense_bytes:,} B（72 MiB），本层流量比为 {dense_bytes / sparse_bytes:.2f}×。Sparse 使用已测 mapped-host gather 时长，dense 使用已测 `cudaMemcpyAsync` DMA 时长；两者均不是按名义带宽估计。主 KV 为 BF16 512 latent + 64 RoPE，1,152 B/token；线性层采用 FP8，indexer K/scales 留在 HBM。",
            "",
            "MFU 的实际 pool=65,664，extend 是持久 append，每层另有 147,456 B candidate D2H。这里按 motivation 的 candidate discard 语义排除该写回，只保留历史 H2D；也不把源 pool 的分配或命中行为移植成 P=65,536 的容量结论。每种方法的每个阶段只有一次侵入式 profile，时长没有重复采样置信区间。",
            "",
            "## Prefill",
            "",
            "令 p、i、k、a、f 分别为 projection、indexer、top-k、attention、finish 的时长，C=p+i+k+a+f。该 chunk 的历史已驻留，没有历史 H2D。串行写回为 C+W；理想异步写回在 projection 产出主 KV 后启动，结束时间为 max(C,p+W)。假设写回源能保持有效、传输与计算互不减速，不计管理它们的开销。",
            "",
            "| 模拟方案 | L1 输出及必要写回完成 ms | 调度 MFU |",
            "| --- | ---: | ---: |",
        ]
    )
    for identifier in ("prefill_hbm", "prefill_serial", "prefill_overlap"):
        row = by_id[identifier]
        text.append(
            f"| {row['label']} | {ms(row['completion_ns'])} | {row['schedule_mfu_percent']:.2f}% |"
        )
    text.extend(
        [
            "",
            "![Prefill 模拟时间线](simulation_prefill.svg)",
            "",
            "## Extend",
            "",
            "主表从 L1 P 开始计到 L1 输出就绪。HBM 行假设主 KV 已驻留；sparse 与 ECHO 行从本层尚未搬入历史开始，dense 的历史搬运则接在上一层 fetch 后面，部分发生在零点之前。",
            "",
            "- 串行 sparse：精确 top-k 后搬入选择集，再计算 attention，T=C+S。",
            "- 理想 sparse overlap：假设 indexer 开始时已提前获知需要搬运的精确集合，H2D 与 indexer 重叠，T=p+max(i,S)+k+a+f。这是显式的重叠收益上界假设；当前 indexer 完成前通常没有完整精确集合，因此不能当作 ECHO 的已实现时序或预测性能。",
            "- Dense 跨层预取：本层完整历史 H2D 接在上一层 fetch 后，P 仍从 t=0 开始，attention 等待本层 H2D 与 top-k 完成。令 fetch 提前量为 Δ，则 T=max(p+i+k,D₁−Δ)+a+f。提前量由上一层实测成本确定，详见下文。",
            "",
            "| 模拟方案 | L1 输出就绪 ms | 调度 MFU |",
            "| --- | ---: | ---: |",
        ]
    )
    for identifier in (
        "extend_hbm",
        "extend_serial_sparse",
        "extend_oracle_sparse",
        "extend_echo_measured",
        "extend_dense_prefetch",
    ):
        row = by_id[identifier]
        text.append(
            f"| {row['label']} | {ms(row['completion_ns'])} | {row['schedule_mfu_percent']:.2f}% |"
        )
    gain = (
        by_id["extend_serial_sparse"]["completion_ns"]
        / by_id["extend_oracle_sparse"]["completion_ns"]
    )
    text.extend(
        [
            "",
            f"本层 S<i，理想情况下可隐藏全部稀疏搬运时间，较串行 sparse 的收益上限为 {gain:.3f}×。这组成本下，完整历史传输仍长于可用于隐藏它的计算窗口。资源竞争和调度可行性不在模型内，不能据此宣布实际 ECHO 或 dense 实现的加速比。",
            "",
            "![Extend 模拟时间线](simulation_extend.svg)",
            "",
            "## 核验与复现",
            "",
            "模拟使用整数 ns，核验每个方案的计算阶段顺序、P 的零点、成本守恒、依赖等待和结束边界。Dense 的完整 DMA 元数据保留负起点及窗口外终点，图示区间和 timeline.csv 仅保存与 L1 窗口的交集；核验二者一致，并将下一层通信与本层记账分开。模型计算与 I/O 时长固定，不建模带宽竞争、SM 争用、融合 kernel 的内部时序或未测输入。原始 fused ECHO kernel 未被拆成虚构的实测分量。",
            "",
            "[输入及来源哈希](inputs.json)、[模拟区间](timeline.csv)、[数值汇总](summary.csv)、[模拟与核验](simulation.json)、[发布清单](publication_manifest.json)随报告保存。输入提取读取已发布数据及与之绑定的原始 FLOPs 调用账本，不启动 GPU；选定调用随 inputs.json 保存。本次结果是 CPU simulation，不是新的硬件计时或正确性验收。",
            "",
            "从仓库根目录运行（使用新的 run ID 与输出目录）：",
            "",
            "```bash",
            ".venv/bin/python -m experiments.deepseek_v32_motivation.src.simulation_inputs \\",
            "  --run-id simulation_new \\",
            "  --output experiments/deepseek_v32_motivation/output/data/simulation_new/inputs.json",
            ".venv/bin/python -m experiments.deepseek_v32_motivation.src.simulate \\",
            "  --inputs experiments/deepseek_v32_motivation/output/data/simulation_new/inputs.json \\",
            "  --output-dir experiments/deepseek_v32_motivation/output/data/simulation_new/report",
            "```",
            "",
            f"本次完整产物保留在 `experiments/deepseek_v32_motivation/output/data/{inputs['run_id']}/`。未替换原 motivation 硬件实测报告。",
            "",
        ]
    )
    position = text.index("## Prefill")
    text[position:position] = metrics_report(inputs, simulation)
    position = text.index("## 核验与复现")
    text[position:position] = dense_report(inputs, simulation) + echo_report(inputs, simulation)
    (output / "results.md").write_text("\n".join(text))


def publish(inputs_path, output):
    inputs = json.loads(inputs_path.read_text())
    if inputs.get("source", {}).get("profile_run_id") == WITHDRAWN_PROFILE_RUN_ID:
        raise ValueError("source profile is withdrawn; its simulation cannot be republished")

    from experiments.deepseek_v32_motivation.src.plot_simulation import draw_simulation

    document = simulate(inputs)
    if "metrics" not in document or "echo" not in inputs["extend"]:
        raise ValueError("publication requires selected-call FLOPs and measured ECHO inputs")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(inputs_path, output / "inputs.json")
    _save_json(output / "simulation.json", document)
    _save_csv(
        output / "compute_metrics.csv",
        [
            {
                "phase": phase,
                **{key: value for key, value in row.items() if key != "flops_by_precision"},
                **{
                    f"{precision.lower()}_flops": row["flops_by_precision"].get(precision)
                    for precision in ("FP8", "BF16", "FP32")
                },
            }
            for phase, stages in document["metrics"]["compute"].items()
            for row in stages.values()
        ],
    )
    _save_csv(
        output / "io_metrics.csv",
        [{"transfer": key, **value} for key, value in document["metrics"]["io"].items()],
    )
    _save_csv(
        output / "summary.csv",
        [
            {key: value for key, value in row.items() if key not in {"events", "full_transfers"}}
            for row in document["scenarios"]
        ],
    )
    _save_csv(
        output / "timeline.csv",
        [
            {"scenario": row["id"], "phase": row["phase"], **event}
            for row in document["scenarios"]
            for event in row["events"]
        ],
    )
    draw_simulation(document, output)
    write_report(output, inputs, document)
    sources = tuple(
        Path(__file__).with_name(name)
        for name in ("simulate.py", "simulation_inputs.py", "plot_simulation.py")
    )
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshots / path.name)
    _save_json(
        output / "publication_manifest.json",
        {
            "run_id": inputs["run_id"],
            "kind": "CPU simulation from measured component costs; no new GPU measurement",
            "artifacts": {
                path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()
            },
            "generator_sources": {
                str(path.relative_to(Path(__file__).resolve().parents[3])): digest(path)
                for path in sources
            },
        },
    )
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    document = publish(args.inputs, args.output_dir)
    print(
        json.dumps(
            {
                "run_id": document["run_id"],
                "output": str(args.output_dir),
                "audit": document["audit"]["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
