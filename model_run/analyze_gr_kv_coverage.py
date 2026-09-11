"""Describe empirical history coverage versus length in the 21 recorded GR cases."""

import json
from pathlib import Path

import numpy as np

from model_run.export_gr_kv_hits import plt
from model_run.sweep_gr_mla_cache import HISTORIES, NEW_TOKENS


def main():
    root = Path("docs/extend_step_profile")
    rows = json.loads(Path("GR/generated/kv_hit_replay/summary.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    fits = []
    for n, color in zip(NEW_TOKENS, ("#2563eb", "#d97706", "#059669")):
        group = sorted((r for r in rows if r["new"] == n), key=lambda r: r["history"])
        h = np.array([r["history"] for r in group])
        u = np.array([r["unique_history"] for r in group])
        p = u / h
        x = np.log(h / 4096)
        beta, log_a = np.polyfit(x, np.log(p), 1)
        pred = np.exp(log_a + beta * x)
        r2 = 1 - np.sum((np.log(p) - np.log(pred)) ** 2) / np.sum(
            (np.log(p) - np.log(p).mean()) ** 2
        )
        err = np.max(np.abs(pred / p - 1))
        fits.append((n // 1024, np.exp(log_a), beta, r2, err))
        axes[0].plot(h / 1024, p * 100, "o-", color=color, label=f"{n // 1024}K new")
        axes[0].plot(h / 1024, pred * 100, "--", color=color, alpha=0.5)
        axes[1].plot(h / 1024, u / 1024, "o-", color=color, label=f"{n // 1024}K new")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([h / 1024 for h in HISTORIES], [f"{h // 1024}K" for h in HISTORIES])
        ax.set_xlabel("History length")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set(
        ylabel="Selected history tokens / history (%)",
        title="History coverage (dashed: empirical power fit)",
        ylim=(0, 105),
    )
    axes[1].set(
        ylabel="Unique selected history tokens (K)", title="Absolute history demand still grows"
    )
    for ext in ("png", "svg"):
        fig.savefig(root / f"gr_kv_coverage_relation.{ext}", dpi=160)
    plt.close(fig)
    report = """# 历史 KV 命中率与历史长度的经验关系

这里的命中率指整批 new query 的 top-k 并集覆盖率：`p_history = unique_history / history`，不是单个 query 的 top-k 比例，也不是硬件 cache hit rate。数据来自已有 21 组 GR 第 0 层实验，top-k=2048。

![历史覆盖率及绝对需求](gr_kv_coverage_relation.png)

## 实测覆盖率

| History | 1K new：历史 / new | 2K new：历史 / new | 4K new：历史 / new |
|---|---:|---:|---:|
"""
    for h in HISTORIES:
        cells = []
        for n in NEW_TOKENS:
            r = next(r for r in rows if r["history"] == h and r["new"] == n)
            cells.append(f"{r['unique_history'] / h:.2%} / {r['unique_new'] / n:.2%}")
        report += f"| {h // 1024}K | " + " | ".join(cells) + " |\n"
    report += """
历史覆盖率随 history 增长而下降，但命中的历史 token 绝对数仍在增加。新增 512K/1024K 数据后，不能继续把原先 4K–64K 的拟合直接外推；这里重新拟合全部七个 history 档位。

## 幂律近似

分别固定 new 长度，对七个点做自然对数空间的普通最小二乘：

`p_history ≈ A × (history / 4096)^β`

p 使用 0–1 比例。这是同一批数据上的描述性拟合，没有独立验证集；4K 点接近覆盖率上限，也参与拟合，没有剔除。

| New | A | β | log 空间 R² | 最大相对拟合误差 |
|---|---:|---:|---:|---:|
"""
    for n, a, beta, r2, err in fits:
        report += f"| {n}K | {a:.4f} | {beta:.4f} | {r2:.4f} | {err:.1%} |\n"
    report += """
这是 4K–1024K 当前样本内的描述性拟合。各条曲线的 β、R² 与最大误差见表；512K 和 1024K 仍然各只有一份输入，拟合不能替代实测，也不能证明覆盖率按固定幂律变化。

## New 长度与复用

固定 history 时，增加 new 通常会增加历史 KV 并集，但其增幅不等于 query 数的增幅。不同 new 长度对应不同完整输入，不能把这些点当成同一次请求追加 query 的严格增量过程。New 是否全命中也须逐组读取表格，不能把短 history 的近全覆盖结论沿用到长 history。

这些经验关系可用于当前输入的搬运容量估算：历史字节量为 `656 × history × p_history`。密度不包含具体地址间隙，碎片访问性能仍应使用已导出的精确地址测量。

512K/1024K 沿用 checkpoint 的 YaRN 参数扩展 RoPE 缓存，超出原配置的 163840 位置上限；indexer logits 每 128 query 分批，统计仍是完整 new 的并集。

## 适用范围与复现

每个形状只有一份 seed=42 的 GR 合成商品文本，使用真实 checkpoint 第 0 层 indexer、无 Hadamard。不同形状不保证是同一文本的截断，history 长度与文本内容的影响尚未分离。这里不能给出总体置信区间，也不能把拟合当作多层或真实业务的通用规律；验证需要多文本、多 seed，并用同一长前缀的不同截断控制内容差异。

```bash
.venv/bin/python -m model_run.analyze_gr_kv_coverage
```

输入为 `GR/generated/kv_hit_replay/summary.json`，只重新统计和绘图，不运行模型。
"""
    (root / "gr_kv_coverage_relation.md").write_text(report)


if __name__ == "__main__":
    main()
