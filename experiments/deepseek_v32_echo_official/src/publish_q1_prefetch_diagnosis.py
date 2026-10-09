"""Publish the bounded cold/resident diagnosis from verified original-core captures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.deepseek_v32_echo_official.src.analyze_q1_official_core import digest, require

ROOT = Path(__file__).resolve().parents[3]
TAGS = ("layer_2_zero_full", "layer_2_warm_full")
ROWS = (
    ("NCU core duration (us)", "gpu__time_duration.sum", 0.001),
    ("Mean SM active cycles", "sm__cycles_active.avg", 1),
    ("Maximum SM active cycles", "sm__cycles_active.max", 1),
    ("Atomic ALU instructions", "smsp__inst_executed_op_generic_atom_dot_alu.sum", 1),
    ("L1 global atomic requests", "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum", 1),
    (
        "TEX sysmem read miss sectors",
        "lts__t_sectors_srcunit_tex_aperture_sysmem_op_read_lookup_miss.sum",
        1,
    ),
    ("DRAM read bytes", "dram__bytes_read.sum", 1),
    ("Register spill local memory requests", "sass__inst_executed_register_spilling_mem_local", 1),
)


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.run_dir.name == "q1_official_core_ncu_20261008_01",
        "This report describes the reviewed run",
    )
    source = args.run_dir / "comparison_verified.json"
    comparison = load(source)
    require(comparison["run_id"] == args.run_dir.name, "Run ID differs")
    sources = {str(source.resolve()): digest(source)}
    identities = []
    for tag in TAGS:
        capture = comparison["captures"][tag]
        directory = args.run_dir / tag
        for name, expected in capture["analysis_source_hashes"].items():
            path = directory / "analysis" / name
            require(digest(path) == expected, "Analysis evidence changed: " + str(path))
            sources[str(path.resolve())] = expected
        require(digest(Path(capture["report"])) == capture["report_sha256"], "NCU bytes changed")
        sources[capture["report"]] = capture["report_sha256"]
        passes = list((directory / "passes").glob("pass-*"))
        require(len(passes) == 1, "Expected one application per capture")
        component_path = passes[0] / "component" / "summary.json"
        component = load(component_path)
        require(component["observation"] == capture["observation"], "Readback differs")
        require(component["variant"] == "baseline", "Expected unchanged baseline")
        require(component["receipt_sha256"] == capture["receipt_sha256"], "Receipt differs")
        sources[str(component_path.resolve())] = digest(component_path)
        identities.append(component["identity"])
        for label, metric, _ in ROWS:
            require(capture["valid_aggregate_metrics"][metric]["has_value"], label + " unavailable")
    require(identities[0] == identities[1], "Cold/resident execution identities differ")
    identity = identities[0]
    cold, warm = (comparison["captures"][tag] for tag in TAGS)
    require(cold["observation"]["threshold_fp32_bits"] == 0, "Cold threshold differs")
    require(warm["observation"]["threshold_fp32_bits"] == 0, "Warm threshold differs")
    require(cold["observation"]["initial_resident_history"] == 0, "Expected cold history")
    require(warm["observation"]["initial_resident_history"] == 65536, "Expected resident history")
    require(identity["capability"] == [9, 0], "Expected SM90")
    require(identity["physical_device"] == 1, "Unexpected GPU")
    require(
        [item["valid_aggregate_metrics"]["dram__bytes_read.sum"]["value"] for item in (cold, warm)]
        == [8996352, 9251072],
        "Reviewed DRAM figures differ",
    )
    require(
        [item["observation"]["attempted_records"] for item in (cold, warm)] == [33464, 0]
        and [item["observation"]["staged_records"] for item in (cold, warm)] == [64, 0]
        and [item["observation"]["logical_host_copy_bytes"] for item in (cold, warm)] == [73728, 0],
        "Reviewed application-visible traffic differs",
    )
    pilot_path = next(
        (args.run_dir / "layer_2_zero_pilot" / "passes").glob("pass-*/observation.json")
    )
    pilot = load(pilot_path)
    require(
        pilot["staged_host_ids"] == list(range(14944, 14976)) + list(range(14912, 14944)),
        "Reviewed pilot IDs differ",
    )
    require(
        cold["observation"]["staged_host_ids"]
        == list(range(28736, 28768)) + list(range(28672, 28704)),
        "Reviewed cold IDs differ",
    )
    sources[str(pilot_path.resolve())] = digest(pilot_path)
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for label, metric, scale in ROWS:
        rows.append(
            {
                "metric": label,
                "cold": cold["valid_aggregate_metrics"][metric]["value"] * scale,
                "resident": warm["valid_aggregate_metrics"][metric]["value"] * scale,
                "ncu_metric": metric,
            }
        )
    for name in (
        "initial_resident_history",
        "attempted_records",
        "staged_records",
        "logical_host_copy_bytes",
    ):
        rows.append(
            {
                "metric": name,
                "cold": cold["observation"][name],
                "resident": warm["observation"][name],
                "ncu_metric": "application-visible readback",
            }
        )
    with (destination / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "cold", "resident", "ncu_metric"))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "run_id": args.run_dir.name,
        "captures": comparison["captures"],
        "execution": {
            key: identity[key]
            for key in (
                "device",
                "capability",
                "physical_device",
                "gpu_uuid",
                "cpu_affinity",
                "torch",
                "cuda",
                "triton",
                "precision",
                "inputs",
            )
        },
        "boundary": comparison["boundary"],
        "clean_timing": False,
        "atomic_dominance_established": False,
        "validated_pm_timeline": False,
        "per_full_capture_replay_passes": 45,
    }
    write(destination / "evidence.json", summary)
    table = "\n".join(
        f"| {row['metric']} | {row['cold']:,.3f} | {row['resident']:,.3f} |" for row in rows[:6]
    )
    report = f"""# ECHO Q1 冷态预取的尾部开销

相同 L2 输入和正零阈值下，完全驻留对照消除了冷态融合核的大部分耗时与最慢 SM 的长尾。
额外开销位于 miss、额度预约和预取搬运这条路径；本次对照同时去掉预约与搬运，不能
单独归因于原子操作，也不能把差值称为可获得的优化收益。

| 指标 | 冷态 | 完全驻留 |
| --- | ---: | ---: |
{table}

冷态在应用可见的完成状态中尝试预约 33,464 条记录，实际暂存 64 条，共 73,728 B；
完全驻留时三者均为 0。两侧 DRAM 读取量分别为 8,996,352 / 9,251,072 B，
寄存器 spill 的 local-memory 请求均为 9,884。平均 SM 活跃周期接近，而最大值相差明显，
支持预取额外工作存在长尾的判断。表中 NCU 聚合指标均明确返回 `has_value=true`。
完整数值与指标名称见 [metrics.csv](metrics.csv)，身份、来源哈希和可用性见
[evidence.json](evidence.json) 与 [provenance.json](provenance.json)。

## 官方源码与可观测行为

固定版本 ECHO 已在 warp 内聚合 miss mask，再由一个线程调用 `atomicAdd`。
预约的记录数不能当成原子指令数。成功取得额度后，每个 warp 在循环中逐条搬运其记录。
单 pass pilot 的实际 ID 为 14944–14975 和 14912–14943；冷态 full 的最终可见 ID 为
28736–28767 和 28672–28703。两次都是同一 256-token task 的两个完整 32-record
组，结合源码可知每个获胜 warp 串行处理 32 条记录。这是集中搬运工作的证据，
尚不足以证明它独自造成全部尾部延迟。

源码固定为 `bc1b75c1000010d0ac6f032ebaac283255c050b1`：
[warp 预约与搬运循环](https://github.com/sjtu-zhao-lab/ECHO/blob/bc1b75c1000010d0ac6f032ebaac283255c050b1/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh#L788)，
[1152 B record 搬运](https://github.com/sjtu-zhao-lab/ECHO/blob/bc1b75c1000010d0ac6f032ebaac283255c050b1/DeepGEMM/deep_gemm/include/deep_gemm/common/utils.cuh#L335)。
本次没有修改官方 kernel。

## 测量与验证边界

Run ID：`{args.run_dir.name}`。GPU 为 {identity["device"]} / SM90，物理 GPU1，
CPU affinity 为 `{identity["cpu_affinity"]}`；Torch `{identity["torch"]}`、
CUDA `{identity["cuda"]}`、Triton `{identity["triton"]}`。使用真实 checkpoint L2 的
Q1/H64/D128 FP8 Q/K、FP32 scales/weights 和 BF16 D576 host records，H=65,536，
N=65,537。两侧 query、key、scale、weight 的保存文件身份相同，阈值 bits 均为 0。
输入属于独立组件数据，不等同于官方 SGLang 的实际 decode 输入与自然驻留状态。

NCU 2026.1.1 使用 kernel replay、cache flush 和 base clock control。先运行一个
冷态单硬件 pass pilot，再分别采集冷态和完全驻留的 full/source，每项 45 passes。
未采集 L0/L1。该时长仅覆盖官方融合 core，是侵入式 profile；打包、promotion、
top-k、精确 recall 和完整模型延迟均不在此时长内。

实际官方 ELF SHA256 为 `{cold["official_elf_sha256"]}`。
独立组件验收签名为 `{cold["receipt_sha256"]}`，验收路径为
`/tmp/cxldsagr-checks/q1-fused-prepare/check_20261008_02/receipt.json`。
采集委托给已验收 baseline，核验原输入、源码/native、NVTX、唯一 kernel 和 launch
geometry；读回在 ProfilerStop 后完成。NCU 在 replay 间恢复设备写入，但合法预约
次序仍可变化，最终读回不能代表每个内部 pass 的实际预取集合。

原始 per-PC 与 PM 数组虽有非零数值，其实例可用性标志为 false；因此未用这些数值
推算源码行的耗时比例或绘制有效 PM 时间线。warm 的 1.5 µs 采样间隔还触发了超过
workload 时长 10% 的警告。六项 CTC 指标不可用，不能记为 0；全部警告保留在原日志。
原始数据、日志和 profiler 文件分别位于本实验 `output/data/`、`output/log/`、
`output/profile/` 下的同名 run ID 目录。

采集入口为 `scripts/profile_q1_official_core.sh`，解析入口为
`src/analyze_q1_official_core.py` 和 `src/report_q1_official_core.py`。
本报告由 `src/publish_q1_prefetch_diagnosis.py --run-dir <原 run 目录>
--output-dir <新报告目录>` 生成；按实验约定用 `python -m` 运行。
"""
    (destination / "report.md").write_text(report)
    archive = destination / Path(__file__).name
    archive.write_bytes(Path(__file__).read_bytes())
    write(
        destination / "provenance.json",
        {
            "source_run_id": args.run_dir.name,
            "publisher": str(Path(__file__).relative_to(ROOT)),
            "publisher_sha256": digest(Path(__file__)),
            "publisher_archive": str(archive.resolve()),
            "command": {
                "module": "experiments.deepseek_v32_echo_official.src.publish_q1_prefetch_diagnosis",
                "run_dir": str(args.run_dir),
                "output_dir": str(args.output_dir),
            },
            "source_files": sources,
            "artifacts": {
                name: digest(destination / name)
                for name in ("report.md", "evidence.json", "metrics.csv")
            },
        },
    )
    print(destination)


if __name__ == "__main__":
    main()
