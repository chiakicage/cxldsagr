"""Measure or summarize GR layer-0 MLA cache unions and ideal offload costs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HISTORIES = (4096, 8192, 16384, 32768, 65536, 524288, 1048576)
NEW_TOKENS = (1024, 2048, 4096)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true", help="Run selected GPU cases afresh")
    parser.add_argument("--root", type=Path, default=Path("GR/generated/cache_union_sweep"))
    parser.add_argument(
        "--report", type=Path, default=Path("docs/extend_step_profile/gr_cache_offload_sweep.md")
    )
    parser.add_argument("--histories", type=int, nargs="+", choices=HISTORIES, default=HISTORIES)
    args = parser.parse_args()
    rows = []
    for h in args.histories:
        for n in NEW_TOKENS:
            out = args.root / f"h{h}_n{n}"
            if args.measure:
                out.mkdir(parents=True, exist_ok=True)
                with (out / "run.log").open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "model_run.measure_gr_mla_cache_union",
                            "--history",
                            str(h),
                            "--new",
                            str(n),
                            "--output",
                            str(out),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            r = json.loads((out / "result.json").read_text())
            assert r["history_tokens"] == h and r["new_tokens"] == n
            assert r.get("query_union_tokens", r["queries_computed_together"]) == n
            assert r["total_references"] == n * 2048
            assert (
                r["unique_history_tokens"] + r["unique_new_tokens"] == r["unique_mla_cache_tokens"]
            )
            b = r["unique_token_bytes"]
            bh = r["unique_history_tokens"] * 656
            bp = r["unique_history_pages_64_tokens"] * 64 * 656
            base = 2 * n * 128 * 2048
            f = base * (512 + 64 + 512)
            # Nine projections incl. BF16 index head weights; dimensions in extend docs.
            proj = (
                2
                * n
                * (
                    7168 * 1536
                    + 1536 * 128 * 192
                    + 7168 * 576
                    + 128 * 128 * 512
                    + 1536 * 64 * 128
                    + 7168 * 128
                    + 7168 * 64
                    + 128 * 512 * 128
                    + 16384 * 7168
                )
            )
            index = 2 * 64 * 128 * (n * h + n * (n + 1) // 2)
            r.update(
                mla_flops=f,
                extend_matrix_flops=proj + index + f,
                historical_token_bytes=bh,
                historical_page_bytes=bp,
                mla_ai_unique_all=f / b,
                mla_ai_history=f / bh,
                mla_ai_history_pages=f / bp,
                extend_ai_mla_cache_only=(proj + index + f) / b,
                pcie_history_ms=bh / 50e9 * 1000,
                pcie_history_pages_ms=bp / 50e9 * 1000,
                gddr_unique_all_ms=b / 800e9 * 1000,
                mla_all_mxfp8_ideal_ms=f / 487e12 * 1000,
                mla_mixed_ideal_ms=base * (512 / 487e12 + 64 / 123e12 + 512 / 247e12) * 1000,
            )
            rows.append(r)
            print(
                f"{h // 1024}K+{n // 1024}K: history {r['unique_history_tokens']}, new {r['unique_new_tokens']}",
                flush=True,
            )
    (args.root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")

    def table(headers, data):
        return (
            "| "
            + " | ".join(headers)
            + " |\n|"
            + "|".join(["---"] * len(headers))
            + "|\n"
            + "".join("| " + " | ".join(map(str, row)) + " |\n" for row in data)
        )

    def shape(r):
        return f"{r['history_tokens'] // 1024}K + {r['new_tokens'] // 1024}K"

    report = """# GR 输入下的 MLA cache 加载量与 offload 分析

## 结论与统计范围

本次比较 4K–1024K history、1K–4K new 下，按 token 和按 64-token page 去重搬运的理想预算。长 history 的并集与 page 覆盖可能改变 offload 判断，具体以各行的搬运 / 计算比例为准；比例小于 100% 才表示该理想搬运时间短于混合精度 MLA 计算时间。这尚不等于现有 kernel 已经实现了搬运与计算重叠。

每组使用 GR input generator 生成一份商品文本，seed=42，真实 DeepSeek-V3.2 checkpoint 的 embedding、输入 RMSNorm 和第 0 层 indexer 权重。Indexer 不做 Hadamard，top-k=2048。所有 new query 按整批统计；512K/1024K history 的 logits 按 128 query 分批计算，每个 query 搜索全部因果可见历史；历史 K 的 4096-token 分批准备仅用于生成第 0 层 token-local 输入。每组检查 token 边界、有效 logits 有限、所选位置满足因果约束，以及每个 query 的 top-k 无重复。

这是每个形状一份输入、第 0 层的统计，不是多层、多请求分布。各形状由同一个生成器配置方法生成，不保证互为同一文本的截断。沿用当前去掉 Hadamard 的近似精度路径，统计不代表原模型所有层的选择分布。

512K/1024K history 的 indexer 按 128 个 query 分批计算 logits，每个 query 仍搜索全部因果可见 KV，命中并集按完整 1K/2K/4K new 统计。多层测量中 MLA 的 new 仍整批执行，层间 hidden states 暂存在 CPU。长序列超出 checkpoint 配置的 163840 位置上限，仅扩展 RoPE 缓存，沿用原 YaRN 参数；这些命中数据不证明长上下文任务质量。

## 去重 cache 加载量

每个 MLA cache token 按当前实现占 656 B：512 B FP8 latent、16 B FP32 scales、128 B BF16 RoPE。以下“总量”包含选中的历史与新增 token；“历史量”才是本轮需要从 CPU 搬入的量，假设新增 cache 已在 GPU。MiB=2²⁰ B，K=1024 token。

"""
    report += table(
        [
            "History + new",
            "历史去重 token",
            "新增去重 token",
            "总去重 token",
            "历史覆盖率",
            "总量 MiB",
            "历史量 MiB",
            "历史 page / 总历史 page",
            "整页历史量 MiB",
        ],
        [
            [
                shape(r),
                f"{r['unique_history_tokens']:,}",
                f"{r['unique_new_tokens']:,}",
                f"{r['unique_mla_cache_tokens']:,}",
                f"{r['history_coverage_pct']:.1f}%",
                f"{r['unique_token_bytes'] / 2**20:.3f}",
                f"{r['historical_token_bytes'] / 2**20:.3f}",
                f"{r['unique_history_pages_64_tokens']} / {r['history_tokens'] // 64}",
                f"{r['historical_page_bytes'] / 2**20:.3f}",
            ]
            for r in rows
        ],
    )
    report += """
Token 去重表示每个选中 token 在本轮只搬一次；整页去重表示每个涉及的 64-token page 只搬一次，即使页内只有少数 token 被选中也搬整页。这两种量都没有计入再次加载和跨层 cache。Top-k 位置在 128 个 MLA head 间共享，cache 容量不乘 head 数。

## 计算强度

对 T 个新增 query，128 个 head、每个 query 2048 个位置，计算整个 MLA 的 QK 与 PV：

```text
F_MLA = 2 × T × 128 × 2048 × (512 + 64 + 512) FLOPs
B_all = (历史去重 token + 新增去重 token) × 656
B_host = 历史去重 token × 656
B_page = 历史去重 page × 64 × 656
I_all = F_MLA / B_all
I_host = F_MLA / B_host
I_page = F_MLA / B_page
```

T=1K、2K、4K 时，F_MLA 分别为 0.584、1.168、2.336 TFLOP，与 history 长度无关。以下强度均为 FLOP/B，分母只包括指定的 MLA cache 字节量，表示理想复用后的有效强度，不是实际 GDDR 流量测量。Softmax、量化等非矩阵运算未计入 FLOPs。

"""
    report += table(
        [
            "History + new",
            "MLA / 总去重 KV",
            "MLA / 历史 token",
            "MLA / 历史 page",
            "整次 attention / 总去重 MLA KV",
        ],
        [
            [
                shape(r),
                f"{r['mla_ai_unique_all']:,.0f}",
                f"{r['mla_ai_history']:,.0f}",
                f"{r['mla_ai_history_pages']:,.0f}",
                f"{r['extend_ai_mla_cache_only']:,.0f}",
            ]
            for r in rows
        ],
    )
    report += """
最后一列延续此前“整次 extend”的口径，计入 9 项投影、indexer logits、MLA 的有效矩阵 FLOPs，不含 FFN。投影为 `402128896 × T` FLOPs，indexer 为 `2 × 64 × 128 × (T × H + T × (T+1)/2)` FLOPs。其分母仍仅为 MLA cache，**不能据此判断整次 attention 的实际 GDDR roofline**：权重、activation、indexer K、logits、top-k 中间结果等访存尚未计入。Indexer 的计算随 history 增长，不会增加 MLA 自身的复用强度。

## Roofline 与传输预算

采用项目确认的峰值和指定带宽，带宽用十进制 GB/s。

| 计算精度 | 峰值 TFLOPS | GDDR 800 GB/s 拐点 FLOP/B | PCIe 50 GB/s 拐点 FLOP/B |
|---|---:|---:|---:|
| MXFP8 | 487 | 608.75 | 9,740 |
| FP8 | 247 | 308.75 | 4,940 |
| BF16 | 123 | 153.75 | 2,460 |

实际 MLA 是混合精度：latent QK 用 MXFP8、RoPE QK 用 BF16、PV 用 FP8。为避免把整个算子都按 MXFP8 算，另算各部分在对应峰值下的理想时间之和：

```text
t_MX = F_MLA / (487 × 10¹²)
t_mixed = 2 × T × 128 × 2048 ×
          (512 / (487 × 10¹²) + 64 / (123 × 10¹²) + 512 / (247 × 10¹²))
t_PCIe = B_host / (50 × 10⁹)
t_page = B_page / (50 × 10⁹)
```

`t_mixed` 仍不含 softmax、量化、同步等开销。下表所有时间均为带宽或峰值算力推导值，**不是此次实测 kernel 耗时**。

"""
    report += table(
        [
            "History + new",
            "总去重 KV / GDDR ms",
            "历史 token / PCIe ms",
            "历史 page / PCIe ms",
            "全 MXFP8 理想 ms",
            "混合精度理想 ms",
            "token 搬运 / 混合计算",
            "page 搬运 / 混合计算",
        ],
        [
            [
                shape(r),
                f"{r['gddr_unique_all_ms']:.4f}",
                f"{r['pcie_history_ms']:.4f}",
                f"{r['pcie_history_pages_ms']:.4f}",
                f"{r['mla_all_mxfp8_ideal_ms']:.3f}",
                f"{r['mla_mixed_ideal_ms']:.3f}",
                f"{100 * r['pcie_history_ms'] / r['mla_mixed_ideal_ms']:.1f}%",
                f"{100 * r['pcie_history_pages_ms'] / r['mla_mixed_ideal_ms']:.1f}%",
            ]
            for r in rows
        ],
    )
    worst_token = max(rows, key=lambda r: r["pcie_history_ms"] / r["mla_mixed_ideal_ms"])
    worst_page = max(rows, key=lambda r: r["pcie_history_pages_ms"] / r["mla_mixed_ideal_ms"])
    report += f"""
按 token 搬运时，最不利的一组为 {shape(worst_token)}，搬运预算占混合精度理想计算时间的 {100 * worst_token["pcie_history_ms"] / worst_token["mla_mixed_ideal_ms"]:.1f}%；按整页搬运时，最不利的一组为 {shape(worst_page)}，占 {100 * worst_page["pcie_history_pages_ms"] / worst_page["mla_mixed_ideal_ms"]:.1f}%。整页搬运的最低历史 KV 计算强度为 {min(r["mla_ai_history_pages"] for r in rows):,.0f} FLOP/B，可与全 MXFP8 的 PCIe 拐点 9,740 FLOP/B 对照。按混合精度预算，token 去重有 {sum(r["pcie_history_ms"] < r["mla_mixed_ideal_ms"] for r in rows)}/{len(rows)} 组搬运时间小于计算时间，page 去重有 {sum(r["pcie_history_pages_ms"] < r["mla_mixed_ideal_ms"] for r in rows)}/{len(rows)} 组；不能把短 history 的整页结论外推到长 history。

## 实现 offload 时需要保留的前提

必须先计算 indexer 和 top-k，才能知道本批 MLA 要用哪些历史 token。Indexer K 与 MLA cache 是两份数据；本分析假设较小的 indexer K cache 常驻 GPU，历史部分每 token 128 B FP8 加 4 B scale，在 64K/512K/1024K 时分别为 8.25/66/132 MiB。若它也 offload，需要另计全历史扫描所需传输，不能只使用这里的 MLA 字节量。

去重列表需要映射到 GPU 暂存 cache，再把所有 query 的索引重映射到这份 cache，供整批 MLA 复用。50 GB/s 假设数据可以批量高效搬运；逐 token 发起 656 B 拷贝通常无法实现这个带宽。CPU gather、暂存区写入、索引回传与重映射、DMA 发起、GPU 再加载均可能增加时间，当前测量没有实现这些操作。

整页方案能减少 gather 的碎片化，但从表中可见其搬运量可能接近整个历史。若搬运和 MLA 串行执行，PCIe 时间会直接相加；要隐藏它，需要可行的流水线和 buffer 调度。本文的 compute-bound 判断只针对理想去重 cache 的 roofline，现有 MLA kernel 的实际 GDDR 重读、访存延迟和计算利用率仍需独立验证。

## 复现

在仓库根目录运行，模型默认位于 `models/DeepSeek-V3.2`：

```bash
env PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m model_run.sweep_gr_mla_cache --measure
# 已有 21 组结果时，只重新汇总：
.venv/bin/python -m model_run.sweep_gr_mla_cache
```

[测量脚本](../../model_run/measure_gr_mla_cache_union.py)负责生成输入和执行真实 indexer；[汇总脚本](../../model_run/sweep_gr_mla_cache.py)计算字节量、FLOPs 和传输预算。输入、top-k 索引、日志与结果 JSON 保存在 `GR/generated/cache_union_sweep/`，由 Git 忽略；仓库只保留脚本和此 Markdown 汇总。此前的 [64K + 4K 单组报告](gr_cache_union_64k_4k.md)可用于交叉核对。
"""
    report = report.replace("21 组", f"{len(rows)} 组")
    args.report.write_text(report)


if __name__ == "__main__":
    main()
