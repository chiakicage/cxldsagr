"""Export token replay inputs and compare three recorded GR indexer layers."""

import json
import shutil
import tempfile
from pathlib import Path

from model_run.export_gr_kv_hits import export_case, plot_cases
from model_run.sweep_gr_mla_cache import HISTORIES, NEW_TOKENS


def main():
    source = Path("GR/generated/multilayer_hits_validated")
    report_dir = Path("docs/extend_step_profile")
    all_rows = []
    for layer in range(3):
        cases = []
        out = source / f"layer{layer}_replay"
        for h in HISTORIES:
            for n in NEW_TOKENS:
                name = f"h{h}_n{n}"
                case = export_case(source / f"layer{layer}" / name, out / name, h, n)
                cases.append(case)
                result = json.loads((source / f"layer{layer}" / name / "result.json").read_text())
                all_rows.append(result)
        (out / "summary.json").write_text(json.dumps([c[0] for c in cases], indent=2) + "\n")
        with tempfile.TemporaryDirectory() as temp:
            plot_cases(cases, Path(temp))
            for p in Path(temp).iterdir():
                shutil.copyfile(p, report_dir / f"layer{layer}_{p.name}")
    (source / "summary.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    report = """# 第 0、1、2 层的 GR KV 命中位置

本页已更新为数值修正后重跑的 revision 2；历史搬入量、全部 63 组计算强度和数值校验见 [offload 与正确性报告](gr_multilayer_offload_validation.md)。旧版数据另保留在 `GR/generated/multilayer_hits/`。

对同一批 21 份输入执行真实 checkpoint 的 embedding → 第 0 层 → 第 1 层 → 第 2 层 indexer。每层分别统计 new query 的 top-k 并集，包含历史与 new KV；top-k=2048，不做 Hadamard。0/1/2 均为 dense 层，第 2 层不计算 FFN；其 attention 额外用于数值与搬运回放校验。

512K/1024K history 的 indexer 按 128 个 query 分批计算 logits，每个 query 仍搜索全部因果可见 KV，命中并集按完整 1K/2K/4K new 统计；MLA 的 new 仍整批执行。层间 hidden states 暂存在 CPU。第 2 层历史只准备完整 K/MLA cache，并执行首末 chunk 的抽样校验；历史 query 的未使用 top-k 不计算。长序列超出 checkpoint 配置的 163840 位置上限，仅扩展 RoPE 缓存，沿用原 YaRN 参数；这些命中数据不证明长上下文任务质量。

## 执行路径与精度

每个形状直接复用 `GR/generated/cache_union_sweep/` 的原始 token IDs，并记录输入 SHA-256。每层按 512-token chunk 执行历史部分，new 的 1K/2K/4K query 按整批统计（长 history 的 logits 分批），因果边界精确到每个 query。第 0、1 层历史和 new 均执行 attention、输出投影、残差、post-attention RMSNorm、dense SwiGLU（gate/up/down）与第二次残差，所得 hidden states 才作为下一层输入。不是把 embedding 分别喂给三个 indexer。

Q/K 投影、indexer 投影、dense FFN 使用 checkpoint FP8 权重和 MXFP8 activation；Norm affine 保留 checkpoint FP32，残差相加后的 Norm、SwiGLU 中间运算和 indexer head-weight 投影均使用 FP32。KV_B 权重按原 block scale 反量化后拆分为 per-head K/V，用 BF16 bmm，避免转置后重新量化引入误差。O 投影 scale 不是二次幂，按 checkpoint 原 scale 反量化为 BF16 做 linear。MLA 沿用项目压缩 cache 与混合精度 sparse kernel；沿用现有 MLA/indexer RoPE 和无 Hadamard 路径。这是项目当前近似计算路径的三层扩展，未与官方完整模型逐层数值对齐。

历史初期可见位置少于 2048 时，只选全部有效位置，余项为无效索引。每个 chunk 检查有效索引数量、无重复、因果约束、有效 logits 与 hidden states 有限；导出时检查并集、引用总数、重映射及连续段重建一致性。第 0 层逐 query 的 top-k 集合还与之前的独立 indexer 实验交叉核对。

所有 63 组均增加独立 FP32 index/MLA/FFN 抽样参考；完整 new 批次还经过 CPU token gather、GPU staging 与索引重映射，MLA 输出与 resident cache 逐位一致。误差、精度边界和测试结果见专项正确性报告。

## 覆盖率对比

每格为“历史覆盖率 / new 覆盖率”，均按整批 query 去重。层间使用完全相同的输入，形状之间仍不保证是同一文本的截断。

| History + new | Layer 0 | Layer 1 | Layer 2 |
|---|---:|---:|---:|
"""
    for h in HISTORIES:
        for n in NEW_TOKENS:
            rs = [r for r in all_rows if r["history_tokens"] == h and r["new_tokens"] == n]
            report += (
                f"| {h // 1024}K + {n // 1024}K | "
                + " | ".join(
                    f"{r['unique_history_tokens'] / h:.2%} / {r['unique_new_tokens'] / n:.2%}"
                    for r in rs
                )
                + " |\n"
            )
    report += "\n## 去重命中总量\n\n每格为“历史 + new = 总 token 数（MiB）”，每条 MLA KV 656 B；history 和 new 均计入搬入量。\n\n| History + new | Layer 0 | Layer 1 | Layer 2 |\n|---|---:|---:|---:|\n"
    for h in HISTORIES:
        for n in NEW_TOKENS:
            rs = [r for r in all_rows if r["history_tokens"] == h and r["new_tokens"] == n]
            report += (
                f"| {h // 1024}K + {n // 1024}K | "
                + " | ".join(
                    f"{r['unique_history_tokens']} + {r['unique_new_tokens']} = {r['unique_mla_cache_tokens']} ({r['unique_token_bytes'] / 2**20:.3f})"
                    for r in rs
                )
                + " |\n"
            )
    overlap = min(r["layer0_reference_overlap"] for r in all_rows if r["layer"] == 0)
    peak = json.loads((source / "validation_summary.json").read_text())["peak_allocated_mib"]
    report += f"\n第 0 层与旧实验的逐 query top-k 集合一致比例最低为 **{overlap:.6%}**。全部形状峰值 PyTorch allocated 显存（含数值校验与 staging）为 **{peak / 1024:.3f} GiB**（逐层加载权重；不等同于驱动进程占用或三层权重同时驻留）。\n"
    report += "\n## 密度热力图\n\n每格 64 token，色标为其中被选中过的 token 比例，0%–100%；不累计重复引用。分箱只用于展示，回放数据精确到单 token。\n"
    for layer in range(3):
        report += f"\n### Layer {layer}\n"
        for n in (1, 2, 4):
            report += f"\n{n}K new：\n\n![Layer {layer}, {n}K new](layer{layer}_gr_kv_hit_distribution_{n}k.png)\n\n[SVG](layer{layer}_gr_kv_hit_distribution_{n}k.svg)\n"
    report += """
## 输出与复现

- 原始输入、逐 query INT32 top-k 和测量信息：`GR/generated/multilayer_hits_validated/layer{0,1,2}/h{H}_n{T}/`。
- NPY 回放输入：`GR/generated/multilayer_hits_validated/layer{0,1,2}_replay/h{H}_n{T}/`，包括原始索引、去重 token、源字节偏移、紧凑索引、引用数、连续段和 manifest；格式同原 [位置报告](gr_kv_hit_distribution.md)。
- 汇总：`GR/generated/multilayer_hits_validated/summary.json`。KV 内容及中间 hidden states 不落盘；这些是地址搬运实验输入。

```bash
env PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m model_run.sweep_gr_mla_cache --measure --histories 524288 1048576
env PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m model_run.sweep_gr_multilayer_hits --histories 524288 1048576 --skip-completed
.venv/bin/python -m model_run.sweep_gr_mla_cache
.venv/bin/python -m model_run.export_gr_kv_hits
.venv/bin/python -m model_run.report_gr_offload_validation
.venv/bin/python -m model_run.report_gr_multilayer_hits
```

只跑一个形状：`.venv/bin/python -m model_run.measure_gr_multilayer_hits --history 65536 --new 4096 --validate`。显存测量包含实际层间 forward，生成图表和回放 NPY 仅需 CPU。原始产物由 Git 忽略，脚本、报告与图片保留。
"""
    (report_dir / "gr_multilayer_kv_hits.md").write_text(report)
    print(
        f"Exported {len(all_rows)} cases; peak {peak / 1024:.3f} GiB; layer-0 overlap {overlap:.6%}"
    )


if __name__ == "__main__":
    main()
