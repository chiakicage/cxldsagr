"""Publish selected Q1 API timings and NCU evidence from immutable run records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import sys
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT.parents[1]
OUTPUT = EXPERIMENT / "output"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def match_identity(accepted, actual):
    accepted, actual = dict(accepted), dict(actual)
    accepted_inputs = accepted.pop("inputs_sha256")
    actual_inputs = actual.pop("inputs_sha256")
    require(accepted == actual, "Source/native/environment identity mismatch")
    require(
        all(accepted_inputs.get(path) == value for path, value in actual_inputs.items()),
        "Requested inputs are not an accepted exact-content subset",
    )


class Evidence:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.files = {}
        self.identities = {}

    def bind(self, path):
        path = path.resolve()
        key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        record = {"sha256": digest(path), "bytes": path.stat().st_size}
        require(key not in self.files or self.files[key] == record, f"Changed evidence: {path}")
        self.files[key] = record
        return record["sha256"]

    def read(self, name, path):
        self.bind(path)
        data = json.loads(path.read_text())
        destination = self.output_dir / "inputs" / f"{name}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        return data

    def snapshots(self, path, identity):
        for relative, expected in identity["source_sha256"].items():
            snapshot = path.parent / "source" / relative
            require(self.bind(snapshot) == expected, f"Source snapshot mismatch: {snapshot}")

    def pair(self, name, check_path, bench_path):
        check = self.read(f"{name}_check", check_path)
        bench = self.read(f"{name}_bench", bench_path)
        require(check["accepted"] is True, f"Unaccepted check: {check_path}")
        require(bench["mode"] == "bench", f"Not a benchmark: {bench_path}")
        require(bench["receipt"]["sha256"] == digest(check_path), "Receipt hash mismatch")
        require(Path(bench["receipt"]["path"]).resolve() == check_path.resolve(), "Wrong receipt")
        match_identity(check["identity"], bench["identity"])
        for path, data in ((check_path, check), (bench_path, bench)):
            self.snapshots(path, data["identity"])
        self.identities[name] = {
            "check_run_id": check["run_id"],
            "bench_run_id": bench["run_id"],
            "original_identity": bench["identity"],
        }
        return check, bench


def timing_rows(bench, operator, real_only=False):
    rows = []
    for sample in bench["samples"]:
        case = sample.get("case", f"kernel_inputs_layer_{sample.get('layer')}.pt")
        match = re.fullmatch(r"kernel_inputs_layer_(\d+)\.pt", case)
        if real_only and match is None:
            continue
        values = sample["gpu_us_samples"]
        require(len(values) == 7, "Expected seven independent timing batches")
        require(all(math.isfinite(value) and value > 0 for value in values), "Invalid timing")
        require(statistics.median(values) == sample["gpu_us_median"], "Median mismatch")
        require(sample["iterations_per_sample"] == 100, "Expected 100 calls per batch")
        rows.append(
            {
                "operator": operator,
                "run_id": bench["run_id"],
                "case": case,
                "layer": int(match[1]) if match else 0,
                "context_tokens": sample.get("logical_k", 65537),
                "method": sample["method"],
                "execution": sample["execution"],
                "gpu_us_median": sample["gpu_us_median"],
                "gpu_us_min": min(values),
                "gpu_us_max": max(values),
                "wall_us_median": sample["wall_us_median"],
                "batches": len(values),
                "calls_per_batch": sample["iterations_per_sample"],
                "wrapper_warmups": sample["warmups"],
            }
        )
    return rows


def plot_bars(ax, rows, methods, labels, cases):
    positions = list(range(len(cases)))
    for method_index, (method, label) in enumerate(zip(methods, labels, strict=True)):
        selected = [
            next(row for row in rows if row["case"] == case and row["method"] == method)
            for case in cases
        ]
        values = [row["gpu_us_median"] for row in selected]
        offsets = [position + (method_index - 0.5) * 0.36 for position in positions]
        bars = ax.bar(
            offsets,
            values,
            width=0.34,
            label=label,
            color="white" if method_index == 0 else "#0072B2",
            edgecolor="#555555" if method_index == 0 else "#004F7C",
            linewidth=1.2,
            yerr=[
                [row["gpu_us_median"] - row["gpu_us_min"] for row in selected],
                [row["gpu_us_max"] - row["gpu_us_median"] for row in selected],
            ],
            error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#333333"},
        )
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=4, fontsize=9)
    ax.set_xticks(positions)
    ax.set_ylabel("Complete wrapper latency (µs)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def workspace_rows(indexer, attention):
    rows = []
    for operator, bench in (("indexer", indexer), ("attention", attention)):
        for sample in bench["samples"]:
            case = sample.get("case", f"kernel_inputs_layer_{sample.get('layer')}.pt")
            if (
                not re.fullmatch(r"kernel_inputs_layer_\d+\.pt", case)
                or sample["execution"] != "graph"
            ):
                continue
            if operator == "indexer":
                metrics = dict(sample["graph_memory"])
                metrics["packed_k_and_scales_storage_bytes"] = sample["paged_packed_storage_bytes"]
            else:
                memory = sample["memory"]
                before, after = memory["before_capture"], memory["after_capture"]
                metrics = {f"capture_{key}_delta": after[key] - before[key] for key in before}
                metrics["capture_peak_allocated_delta"] = (
                    memory["capture_peak_allocated"] - before["allocated"]
                )
            for metric, value in metrics.items():
                rows.append(
                    {
                        "operator": operator,
                        "run_id": bench["run_id"],
                        "case": case,
                        "method": sample["method"],
                        "metric": metric,
                        "bytes": value,
                        "kind": "storage_shape"
                        if metric.startswith("packed_")
                        else "observed_delta",
                    }
                )
    return rows


def ncu_rows(evidence, operator, summary_path, check):
    summary = evidence.read(f"{operator}_ncu_summary", summary_path)
    rows = []
    for tag, report in summary["reports"].items():
        require(evidence.bind(Path(report["path"])) == report["sha256"], "NCU report hash mismatch")
        profile_path = summary_path.parent.parent / tag / "result.json"
        profile = evidence.read(f"{operator}_ncu_{tag}", profile_path)
        require(profile["profile_calls"] == 1, "NCU profile must contain one API call")
        match_identity(check["identity"], profile["identity"])
        metrics = evidence.read(
            f"{operator}_ncu_{tag}_metrics", summary_path.parent / f"{tag}.metrics.json"
        )
        keys = set(report["metrics"]) | {
            "launch__registers_per_thread",
            "launch__shared_mem_per_block",
            "launch__occupancy_limit_registers",
            "launch__occupancy_limit_shared_mem",
        }
        for metric in sorted(keys):
            value = metrics.get(metric)
            if value is None or value["value"] is None:
                continue
            rows.append(
                {
                    "operator": operator,
                    "report": tag,
                    "kernel": report["kernel"],
                    "metric": metric,
                    "value": value["value"],
                    "unit": value["unit"],
                    "ncu_report_path": report["path"],
                    "ncu_report_sha256": report["sha256"],
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(args.output_dir.resolve().is_relative_to(OUTPUT / "data"), "Use output/data")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(args.output_dir / "matplotlib")
    evidence = Evidence(args.output_dir)
    check, bench = evidence.pair(
        "attention",
        OUTPUT / "acceptance/q1_attention_check_20261008_01/result.json",
        OUTPUT / "data/q1_attention_bench_20261008_01/result.json",
    )
    require(
        check["identity_stable_after_execution"] and bench["identity_stable_after_execution"],
        "Attention identity changed during execution",
    )
    for path, expected in bench["identity"]["inputs_sha256"].items():
        require(evidence.bind(Path(path)) == expected, "Input bytes differ")
    api = timing_rows(bench, "attention", real_only=True)
    require(len(api) == 12, "Expected three layers, two APIs and two modes")
    rounding = [
        {"layer": row["case"]["layer"], "comparison": name, **row[name]}
        for row in check["checks"]
        for name in ("prefill_vs_oracle", "decode_vs_oracle", "decode_vs_prefill")
    ]
    ncu = ncu_rows(
        evidence,
        "attention",
        OUTPUT / "data/q1_attention_ncu_20261008_01/analysis/summary.json",
        check,
    )
    write_csv(args.output_dir / "api_timing.csv", api)
    write_csv(args.output_dir / "attention_rounding.csv", rounding)
    write_csv(args.output_dir / "workspace.csv", workspace_rows({"samples": []}, bench))
    write_csv(args.output_dir / "ncu_metrics.csv", ncu)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), layout="constrained")
    for ax, mode in zip(axes, ("eager", "graph"), strict=True):
        rows = [row for row in api if row["execution"] == mode]
        plot_bars(
            ax,
            rows,
            ("prefill", "decode"),
            ("Sparse prefill", "Split16 decode"),
            [f"kernel_inputs_layer_{i}.pt" for i in range(3)],
        )
        ax.set_xticklabels([f"Layer {i}" for i in range(3)])
        ax.set_title("Complete attention API · " + mode)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="outside lower center", ncol=2)
    fig.suptitle("Q1 · 2,048 selected records · BF16 · H200")
    for extension in ("svg", "png", "pdf"):
        fig.savefig(args.output_dir / f"api_latency.{extension}", dpi=180)
    plt.close(fig)
    summary = {
        "schema": "q1-attention-report-v2",
        "run_id": args.output_dir.name,
        "check_run_id": check["run_id"],
        "bench_run_id": bench["run_id"],
        "cases": len(check["checks"]),
        "receipt_verified": True,
        "source_snapshots_verified": True,
        "scope": "Unchanged production attention APIs only; current MQA and official ECHO "
        "adapter results are in q1_official_path and full-model MFU reports.",
    }
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "report.md").write_text("""# Q1 sparse attention 算子诊断

16 分片 decode 的完整 Graph API 约为 15.4 µs，相对官方 sparse prefill API
加速 3.53–3.57 倍。图中为 7 个批次的中位数，每批 100 次调用；误差线为批次全范围。

![完整 attention API](api_latency.svg)

数据来自仍有效的 `q1_attention_bench_20261008_01`；本轮只重新整理图表，未将其
记作新的 GPU 测量。当前 MQA 打包与官方 ECHO 融合路径见
[官方路径报告](../q1_official_path/report.md)，完整模型见
[MFU 报告](../../../deepseek_v32_mfu/README.md)。

输入为真实 L0–L2 的 BF16 Q=[1,128,576]、KV=[65537,576]，selection=[1,2048]，
value dimension=512；GPU2 H200/SM90，CPU16–23，Torch 2.12.1+cu130、Triton 3.7.1。
计时包含 selection/layout 适配、Q repeat、官方 sparse MLA、FP32 LSE combine 和输出
分配。两种 API 启动时预热20次，每种 eager/Graph 方式又预热20次；验收、计时和
NCU 分进程执行。输入来自一次额外 eager 诊断 forward，并由该验收独立绑定。

两条 API 按原 atol=4e-3、rtol=2e-2 与 FP32 oracle 比较，也直接相互比较。每层每种
API 验证4次固定输入 Graph replay，并验证空 selection 输出全零及恢复后逐位一致。
Decode 与 prefill 的最大绝对差异为4.8828125e-4/2.44140625e-4/1.220703125e-4；
BF16 partial 与 FP32 LSE 合并改变舍入路径，不能表述为与 prefill 逐位一致。
详见[误差表](attention_rounding.csv)、[完整计时](api_timing.csv)。

独立 NCU 中，核心 grid 从2增到32，SM throughput 从0.98%增到6.74%，DRAM 读取
峰值比例从0.79%增到9.34%。每 CTA 使用384 threads、168 registers/thread 和
231,888 B shared memory，限制每 SM 驻留一个 CTA；原小 grid 留下大量空闲 SM。
NCU 的71.520→10.976 µs 只含核心 kernel，使用 cache flush 和 base clocks，不能
替代完整 API 计时。Source sampling 较少，不据此精确排序 stall 原因。

[workspace](workspace.csv) 是该次运行的 allocated/reserved/device-used 观测差值，
不能代替完整 cache 预算或 graph private reserved；decode capture 的 allocated 峰值
增量为4,604,928 B，prefill为133,120 B。当前模型的预算与完整请求验收另行报告。
[来源记录](provenance.json)绑定原 receipt、源码快照、输入和 NCU 文件，
[汇总](summary.json)记录本次报告整理的 run ID。
""")
    for path in (Path(__file__),):
        evidence.bind(path)
        shutil.copy2(path, args.output_dir / path.name)
    for path, expected in evidence.files.items():
        require(
            digest(ROOT / path) == expected["sha256"], "Evidence changed while generating report"
        )
    selected = [
        p
        for p in args.output_dir.iterdir()
        if p.is_file() and p.suffix in {".csv", ".json", ".md", ".svg", ".png", ".pdf"}
    ]
    write_json(
        args.output_dir / "provenance.json",
        {
            "schema": "q1-attention-provenance-v2",
            "run_id": args.output_dir.name,
            "argv": sys.argv,
            "files": evidence.files,
            "original_identities": evidence.identities,
            "artifacts_sha256": {p.name: digest(p) for p in selected},
            "all_bound_files_stable_after_generation": True,
        },
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
