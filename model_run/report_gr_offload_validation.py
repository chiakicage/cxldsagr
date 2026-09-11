"""Summarize all 63 corrected cases, numerical checks and ideal MLA offload budgets."""

import json
from pathlib import Path

from model_run.sweep_gr_mla_cache import HISTORIES, NEW_TOKENS


def main():
    root = Path("GR/generated/multilayer_hits_validated")
    rows = []
    validations = []
    peaks = []
    for h in HISTORIES:
        for n in NEW_TOKENS:
            val = json.loads((root / f"h{h}_n{n}_validation.json").read_text())
            assert len(val) == 9 and len([v for v in val if "ffn" in v]) == 6
            validations.extend(val)
            for v in val:
                if v["start"] == h:
                    assert v["attention"]["cpu_staged_output_bitwise_equal"]
            log = (root / f"h{h}_n{n}.log").read_text()
            assert "Completed; peak allocated MiB:" in log
            peaks.append(float(log.split("Completed; peak allocated MiB:")[-1].strip()))
            for layer in range(3):
                r = json.loads((root / f"layer{layer}" / f"h{h}_n{n}" / "result.json").read_text())
                assert r["numerics_revision"] == 2
                base = 2 * n * 128 * 2048
                f = base * (512 + 64 + 512)
                t = base * (512 / 487e12 + 64 / 123e12 + 512 / 247e12)
                bh = r["unique_history_tokens"] * 656
                ba = r["unique_mla_cache_tokens"] * 656
                r.update(
                    history_bytes=bh,
                    all_bytes=ba,
                    mla_flops=f,
                    mla_mixed_ideal_ms=t * 1000,
                    history_ai=f / bh,
                    all_ai=f / ba,
                    history_pcie50_ms=bh / 50e9 * 1000,
                    all_pcie50_ms=ba / 50e9 * 1000,
                    history_transfer_fraction=bh / 50e9 / t,
                    all_transfer_fraction=ba / 50e9 / t,
                    history_required_GBs=bh / t / 1e9,
                    all_required_GBs=ba / t / 1e9,
                )
                rows.append(r)
    summary = {
        k: {
            "samples": sum(k in v for v in validations),
            "max_nrmse": max(v[k]["nrmse"] for v in validations if k in v),
            "max_abs": max(v[k]["max_abs"] for v in validations if k in v),
            "min_cosine": min(v[k]["cosine"] for v in validations if k in v),
        }
        for k in ("index", "attention", "ffn")
    }
    summary["offload_bitwise_cases"] = sum(
        v["attention"].get("cpu_staged_output_bitwise_equal", False) for v in validations
    )
    summary["max_topk_objective_gap"] = max(
        v["index"]["topk_objective_relative_gap"] for v in validations
    )
    summary["peak_allocated_mib"] = max(peaks)
    (root / "offload_budgets.json").write_text(json.dumps(rows, indent=2) + "\n")
    (root / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = """# 三层 MLA 历史 offload：数值检查与计算强度

此次覆盖全部 **4K/8K/16K/32K/64K/512K/1024K history × 1K/2K/4K new × 第 0/1/2 层，共 63 组**。重点是历史搬入量及其能否被计算覆盖，而不是 new 是否全命中。表中同时保留“仅历史 offload”和“历史 + new 都 offload”的预算。

512K/1024K history 的 indexer 按 128 个 query 分批计算 logits，每个 query 仍搜索全部因果可见 KV，命中并集按完整 1K/2K/4K new 统计；MLA 的 new 仍整批执行。层间 hidden states 暂存在 CPU。第 2 层历史只准备完整 K/MLA cache，并执行首末 chunk 的抽样校验；历史 query 的未使用 top-k 不计算。长序列超出 checkpoint 配置的 163840 位置上限，仅扩展 RoPE 缓存，沿用原 YaRN 参数；这些命中数据不证明长上下文任务质量。

## 修正与数值校验

对照 [DeepSeek 官方 inference/model.py](https://github.com/deepseek-ai/DeepSeek-V3.2/blob/main/inference/model.py) 的 RMSNorm、Indexer、MLP 与 Block，发现并修正前版三处精度差异：

- 残差相加用 FP32；Norm 在尚未舍入的和上计算，残差流另存 BF16。下一层也保留 FFN 输出与残差的独立舍入边界。
- SwiGLU 的 SiLU 和乘法使用 FP32，再将结果转 BF16 输入 down projection。
- Indexer head-weight 投影使用 FP32 输入、权重与结果。

修正后重跑所有 63 组。旧版数据保留在 `GR/generated/multilayer_hits/`；当前数据在 `GR/generated/multilayer_hits_validated/`，`numerics_revision=2`。不能再把旧版“第 0 层与原实验 100% 一致”当成本版检查结论。

保留已有无 Hadamard 设定；MLA 使用量化 latent cache、FP8 QK/PV，K/V 吸收投影与 O 使用反量化 BF16。因此下述验证针对本项目指定计算路径，**不等于与官方全模型逐位一致，也没有验证最终任务质量**。

每组每层抽查历史首 chunk、历史末 chunk、整批 new；每处取首、中、末三个 query。Index logits 用独立 PyTorch FP32 点积、ReLU 和 head 加权参考，top-k 检查参考目标值差距以容许临界并列。MLA 参考逐条解码 656 B cache 后，用 FP32 QK、softmax、PV 计算。FFN 参考独立量化 activation、反量化 checkpoint block，再做 FP32 GEMM。参考计算关闭 TF32。

NRMSE 定义为 `||actual-reference||₂ / ||reference||₂`，表中取所有抽样组最差值。MLA 参考不模拟 kernel 内额外的 Q/PV FP8 量化，二者允许有小量数值误差。

| 检查项 | 抽样组数（每组 3 query） | 最大 NRMSE | 最低余弦相似度 | 最大绝对误差 |
|---|---:|---:|---:|---:|
"""
    for key, label in [
        ("index", "Indexer logits"),
        ("attention", "MLA latent 输出"),
        ("ffn", "第 0/1 层 FFN 输出"),
    ]:
        s = summary[key]
        report += f"| {label} | {s['samples']} | {s['max_nrmse']:.6%} | {s['min_cosine']:.8f} | {s['max_abs']:.6g} |\n"
    report += f"\nTop-k 的最大相对目标值差为 {summary['max_topk_objective_gap']:.3g}。**全部 {summary['offload_bitwise_cases']} 组完整 new 批次**另外执行实际 packed KV → CPU → CPU 按 token gather → GPU staging → 索引重映射 → 同一 MLA kernel，结果与原 GPU cache 的 MLA 输出逐位一致，最大绝对误差为 0。这部分覆盖所有 new query 和全部 heads，并非只检查抽样行。显存峰值（含验证）为 {max(peaks) / 1024:.3f} GiB。\n"
    report += """
Checkpoint 的 K/V 拆分、O 非二次幂 scale、三层五种 FP8 投影、Norm、RoPE 另由 `model_run/tests/test_gr_checkpoint_math.py` 独立测试（3 层参数化测试全部通过）；RoPE 参考使用 float64 频率和复数旋转，覆盖到位置 69631。另增两个长位置 RoPE 测试，覆盖 512K/1024K history 与 4K new 的末端，仍采用独立 float64 参考；现有 FP32 相位构造在长位置误差更大，因此明确要求 NRMSE < 1%、cosine > 0.9999，不沿用短位置的 0.4% 误差界。

## 预算公式

```text
F = 2 × new × 128 × 2048 × (512 + 64 + 512)
B_hist = unique_history × 656
B_all  = (unique_history + unique_new) × 656
I_hist = F / B_hist; I_all = F / B_all
T_compute = 2 × new × 128 × 2048 × (512/487e12 + 64/123e12 + 512/247e12)
T_transfer = B / 50e9
所需有效带宽 = B / T_compute
```

F 只计 MLA QK+PV，不用 FFN/投影等额外 FLOPs 放大计算强度。T_compute 是混合精度峰值推导的理想时间（1K/2K/4K new 分别约 1.957/3.913/7.827 ms）。这要求整批 query 复用每层去重 cache，每条 KV 理想上只搬一次。Indexer K 仍驻 GPU，索引生成的全历史扫描不在这份稀疏搬入量内。

## 全部 63 组历史覆盖率与预算

AI 单位 FLOP/B，MiB=2²⁰ B。“预算占比”是 50 GB/s 理想搬运时间 / 理想 MLA 计算时间，H 为仅历史，All 包含历史与 new。

| 层 | History + new | 历史覆盖率 | 历史 MiB | All MiB | AI H | AI All | 预算占比 H / All | 所需带宽 H / All (GB/s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for r in sorted(rows, key=lambda r: (r["layer"], r["history_tokens"], r["new_tokens"])):
        report += f"| {r['layer']} | {r['history_tokens'] // 1024}K + {r['new_tokens'] // 1024}K | {r['unique_history_tokens'] / r['history_tokens']:.2%} | {r['history_bytes'] / 2**20:.3f} | {r['all_bytes'] / 2**20:.3f} | {r['history_ai']:,.0f} | {r['all_ai']:,.0f} | {r['history_transfer_fraction']:.2%} / {r['all_transfer_fraction']:.2%} | {r['history_required_GBs']:.3f} / {r['all_required_GBs']:.3f} |\n"
    worst = max(rows, key=lambda r: r["all_transfer_fraction"])
    max_hist = max(rows, key=lambda r: r["history_transfer_fraction"])
    report += f"\n历史 AI 最低 **{min(r['history_ai'] for r in rows):,.0f} FLOP/B**，All AI 最低 **{min(r['all_ai'] for r in rows):,.0f} FLOP/B**；487 TFLOPS / 50 GB/s 对应的拐点为 9740 FLOP/B。最不利 All 场景是第 {worst['layer']} 层 {worst['history_tokens'] // 1024}K + {worst['new_tokens'] // 1024}K：搬运 {worst['all_bytes'] / 2**20:.3f} MiB，50 GB/s 理想时间 {worst['all_pcie50_ms']:.4f} ms，占混合 MLA 理想时间 {worst['all_transfer_fraction']:.2%}；隐藏搬运所需有效带宽约 {worst['all_required_GBs']:.3f} GB/s。仅历史的最大预算占比为 {max_hist['history_transfer_fraction']:.2%}。\n"
    report += """
## 能支持什么结论

每组能否覆盖理想搬运时间，以表中的预算占比判断：低于 100% 表示搬运预算小于混合精度 MLA 理想时间。历史覆盖率本身不足以作判断；长 history 即使覆盖比例很低，去重字节量也可能较大。

**数值无损搬运已在 63 组验证；端到端性能无损尚未证明。** CPU staging 在本实验中是串行的正确性回放，没有实现异步重叠，也没有测量 offload 延迟。50 GB/s 是容量模型的带宽假设，不是碎片访问实测值；应看上表所需有效带宽，并把 CPU gather、索引回传/去重/重映射、调度与同步纳入总预算。当前 profile 中常驻 GPU 的计算耗时不能直接替代这份真实 checkpoint 三层实现的 offload 计时。

即使传输时间小于 MLA 计算，也不自动意味着本层传输可全部被隐藏：top-k 生成后才知道地址，搬入后才能计算依赖它们的 MLA，需要分块流水线或其他可重叠工作。后续实际性能实验应以同一输入的 GPU resident 与 offload 端到端时间对比，并报告差值。

当 history 和 new 都 offload 时，All 只计读取选中记录；new cache 首次写回 CPU 还要单独计入（每层 new×656 B），没有混入上表。以上也不包含重新搬入次数或跨请求缓存复用。

## 复现

```bash
env PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m model_run.sweep_gr_multilayer_hits
env PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest model_run/tests/test_gr_checkpoint_math.py -q
.venv/bin/python -m model_run.report_gr_offload_validation
.venv/bin/python -m model_run.report_gr_multilayer_hits
```

原始结果、每处数值误差、搬运回放检查和完整预算 JSON 位于 `GR/generated/multilayer_hits_validated/`。具体命中位置、热力图和 NPY 回放格式见 [三层位置报告](gr_multilayer_kv_hits.md)。
"""
    Path("docs/extend_step_profile/gr_multilayer_offload_validation.md").write_text(report)
    print(json.dumps(summary, indent=2))
    print(
        "Worst all-cache budget:",
        worst["layer"],
        worst["history_tokens"],
        worst["new_tokens"],
        worst["all_transfer_fraction"],
        worst["all_required_GBs"],
    )


if __name__ == "__main__":
    main()
