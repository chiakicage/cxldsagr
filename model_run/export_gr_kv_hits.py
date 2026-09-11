"""Export recorded GR top-k positions for token-granular offload replay and plots."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gr-kv-hit-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model_run.sweep_gr_mla_cache import HISTORIES, NEW_TOKENS


def export_case(source, target, history, new):
    target.mkdir(parents=True, exist_ok=True)
    raw = source / "selected_indices.pt"
    result = json.loads((source / "result.json").read_text())
    indices = torch.load(raw, map_location="cpu", weights_only=True).numpy()
    assert indices.dtype == np.int32 and indices.shape == (new, result["topk"])
    assert result["history_tokens"] == history and result["new_tokens"] == new
    assert np.all(indices >= 0)
    assert np.all(indices <= history + np.arange(new)[:, None])
    assert np.all(np.diff(np.sort(indices, axis=1), axis=1) > 0)
    counts = np.bincount(indices.ravel(), minlength=history + new)
    unique = np.flatnonzero(counts).astype(np.int32)
    assert unique.tolist() == json.loads((source / "unique_token_ids.json").read_text())
    assert len(unique) == result["unique_mla_cache_tokens"]
    assert np.count_nonzero(counts[:history]) == result["unique_history_tokens"]
    assert np.count_nonzero(counts[history:]) == result["unique_new_tokens"]
    assert int(counts.sum()) == result["total_references"] == indices.size
    compact = np.searchsorted(unique, indices).astype(np.int32)
    assert np.array_equal(unique[compact], indices)
    # Runs are exact adjacent selected records, never expanded to cache pages.
    starts = np.r_[0, np.flatnonzero(np.diff(unique) != 1) + 1]
    lengths = np.diff(np.r_[starts, len(unique)]).astype(np.int32)
    runs = np.column_stack((unique[starts], lengths))
    assert np.array_equal(np.concatenate([np.arange(s, s + n) for s, n in runs]), unique)
    arrays = {
        "query_positions": np.arange(history, history + new, dtype=np.int32),
        "selected_indices": indices,
        "unique_token_ids": unique,
        "reference_counts": counts,
        "source_byte_offsets": unique.astype(np.int64) * 656,
        "compact_indices": compact,
        "contiguous_runs": runs,
    }
    for name, array in arrays.items():
        np.save(target / f"{name}.npy", array, allow_pickle=False)
    gaps = np.diff(unique)
    stats = {
        "case": source.name,
        "history": history,
        "new": new,
        "unique_history": int(np.count_nonzero(counts[:history])),
        "unique_new": int(np.count_nonzero(counts[history:])),
        "unique_total": len(unique),
        "references_history": int(counts[:history].sum()),
        "references_new": int(counts[history:].sum()),
        "read_bytes": len(unique) * 656,
        "run_count": len(runs),
        "singleton_run_pct": float(100 * np.mean(lengths == 1)),
        "run_length_mean": float(lengths.mean()),
        "run_length_p50": float(np.median(lengths)),
        "run_length_p95": float(np.percentile(lengths, 95)),
        "run_length_max": int(lengths.max()),
        "selected_gap_p95": float(np.percentile(gaps, 95)),
    }
    manifest = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "source_indices_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "source_input_sha256": hashlib.sha256((source / "input.jsonl").read_bytes()).hexdigest(),
        "source_result": result,
        "stats": stats,
        "offload_scope": "historical and new MLA cache; indexer cache is separate",
        "record_bytes": 656,
        "layout": "one contiguous host allocation, absolute token j starts at j * 656",
        "arrays": {k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in arrays.items()},
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return stats, counts, lengths


def plot_cases(cases, report_dir):
    cmap = plt.get_cmap("Blues")
    for col, n in enumerate(NEW_TOKENS):
        fig, axes = plt.subplots(
            len(HISTORIES), 1, figsize=(16, 1.6 * len(HISTORIES)), layout="constrained"
        )
        for row, (h, ax) in enumerate(zip(HISTORIES, axes)):
            counts = cases[row * len(NEW_TOKENS) + col][1]
            selected_per_bin = (counts > 0).reshape(-1, 64).sum(axis=1)
            assert int(selected_per_bin.sum()) == int(np.count_nonzero(counts))
            im = ax.imshow(
                selected_per_bin[None, :] / 64,
                aspect="auto",
                interpolation="none",
                cmap=cmap,
                vmin=0,
                vmax=1,
                extent=(0, len(counts) / 1024, 0.5, -0.5),
            )
            ax.axvline(h / 1024, color="#22d3ee", ls="--", lw=1)
            ax.set(yticks=[], title=f"{h // 1024}K history", xlabel="Absolute KV token position")
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=10, integer=True))
            ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.0f}K"))
        fig.colorbar(
            im,
            ax=axes,
            label="Hit density",
            ticks=[0, 0.25, 0.5, 0.75, 1],
            format=matplotlib.ticker.PercentFormatter(xmax=1),
            shrink=0.85,
        )
        fig.suptitle(
            f"{n // 1024}K new | KV hit density | 64 tokens per cell",
            fontsize=13,
        )
        for extension in ("png", "svg"):
            fig.savefig(report_dir / f"gr_kv_hit_distribution_{n // 1024}k.{extension}", dpi=160)
        plt.close(fig)
    frag, fax = plt.subplots(
        len(HISTORIES), len(NEW_TOKENS), figsize=(16, 2.8 * len(HISTORIES)), layout="constrained"
    )
    for (stats, counts, lengths), rx in zip(cases, fax.flat):
        h, n = stats["history"], stats["new"]
        values, freq = np.unique(lengths, return_counts=True)
        rx.step(values, np.cumsum(freq) / len(lengths) * 100, where="post", color="#2563eb")
        rx.set(
            xscale="log",
            xlim=(1, max(2, lengths.max())),
            ylim=(0, 102),
            title=f"{h // 1024}K + {n // 1024}K: {len(lengths):,} runs, {stats['singleton_run_pct']:.1f}% singletons",
            xlabel="Contiguous selected run length (tokens)",
            ylabel="Runs CDF (%)",
        )
        rx.grid(alpha=0.15)
    frag.suptitle(
        "Fragmentation of the sorted union (history + new) | exact adjacent tokens only",
        fontsize=15,
    )
    for figure, name in [(frag, "gr_kv_hit_runs")]:
        figure.savefig(report_dir / f"{name}.png", dpi=160)
        figure.savefig(report_dir / f"{name}.svg")
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("GR/generated/cache_union_sweep"))
    parser.add_argument("--output", type=Path, default=Path("GR/generated/kv_hit_replay"))
    parser.add_argument("--report-dir", type=Path, default=Path("docs/extend_step_profile"))
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for h in HISTORIES:
        for n in NEW_TOKENS:
            name = f"h{h}_n{n}"
            case = export_case(args.source / name, args.output / name, h, n)
            cases.append(case)
            print(json.dumps(case[0]), flush=True)
    (args.output / "summary.json").write_text(json.dumps([c[0] for c in cases], indent=2) + "\n")
    plot_cases(cases, args.report_dir)
    report = """# GR KV 命中位置与碎片分布

第 1、2 层已补测，见 [第 0、1、2 层对比](gr_multilayer_kv_hits.md)。本页仍保留原第 0 层数据。

复用已有 21 组实验的 `selected_indices.pt`，没有重新采样或重跑模型。每组一份 GR 商品文本，seed=42，真实 checkpoint 第 0 层 indexer、无 Hadamard、top-k=2048，new 的命中并集按整批统计。这是现有实验的实际选择位置，不是随机模拟地址，也不是硬件 cache hit；不代表多层或多请求分布。

本次将 **历史和 new 的 MLA cache 都纳入 CPU 搬入列表**，按整批 query 的并集去重，每条记录 656 B。Indexer cache 是另一份数据，未包含在该稀疏搬运列表中。这里只生成后续实验输入，没有实现 offload 或测量碎片访问带宽。

512K/1024K history 的 indexer 按 128 个 query 分批计算 logits，每个 query 仍搜索全部因果可见 KV，命中并集按完整 1K/2K/4K new 统计。多层报告中 MLA 的 new 仍整批执行，层间 hidden states 暂存在 CPU。长序列超出 checkpoint 配置的 163840 位置上限，仅扩展 RoPE 缓存，沿用原 YaRN 参数；这些命中数据不证明长上下文任务质量。

## 位置分布

### 1K new

![1K new 命中位置](gr_kv_hit_distribution_1k.png)

[SVG 原图](gr_kv_hit_distribution_1k.svg)

### 2K new

![2K new 命中位置](gr_kv_hit_distribution_2k.png)

[SVG 原图](gr_kv_hit_distribution_2k.svg)

### 4K new

![4K new 命中位置](gr_kv_hit_distribution_4k.png)

[SVG 原图](gr_kv_hit_distribution_4k.svg)

横轴是绝对 KV token 位置；1K、2K、4K new 分别绘成三张独立图，每张图的 7 个面板对应 4K、8K、16K、32K、64K、512K、1024K history。**每格覆盖连续 64 个 token，颜色表示其中至少被本批 query 选中过一次的 token 比例，即命中密度，范围为 0%–100%。** 例如 50% 表示该区间有 32 个 KV 需要搬入；同一个 token 被多个 query 选中仍只计 1。全部图共用色标，颜色越深表示命中越密集。青色虚线为 history/new 边界。

密度计算方法为 `(reference_counts > 0).reshape(-1, 64).mean(axis=1)`，从绝对位置 0 开始划分 `[0,64)、[64,128)、…`，不做插值平滑。64K history + 4K new 对应 1088 格，而非 69632 个逐 token 色条。分箱只用于展示密度，不意味着按整页搬运；逐 token 的原始位置和实验输入保持不变。图不能展示格内具体命中位置，缩到小窗口时仍可能损失格间细节，可打开原图放大查看。

历史覆盖率随 history 长度的关系、幂律拟合和适用范围见 [历史 KV 覆盖率分析](gr_kv_coverage_relation.md)。

## 连续片段分布

![连续片段长度分布](gr_kv_hit_runs.png)

按绝对位置排序后，相邻且全部命中的 token 合并为一段；不跨未命中位置，也不按页补齐。CDF 以片段数为权重，横轴为对数刻度。单 token 段比例越高，表示若逐段搬运，小请求占比越高；实际效率仍需要实验测量。

| History + new | 历史命中 | new 命中 | 总搬入 MiB | 连续段数 | 单 token 段占比 | 平均段长 | P95 段长 | 最长段 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for s, _, _ in cases:
        report += (
            f"| {s['history'] // 1024}K + {s['new'] // 1024}K | {s['unique_history']:,} | {s['unique_new']:,} | "
            f"{s['read_bytes'] / 2**20:.3f} | {s['run_count']:,} | {s['singleton_run_pct']:.1f}% | "
            f"{s['run_length_mean']:.2f} | {s['run_length_p95']:.1f} | {s['run_length_max']:,} |\n"
        )
    report += """
## 后续实验输入

输出目录：`GR/generated/kv_hit_replay/h{history}_n{new}/`，共 21 组。所有 NPY 数组可由 NumPy 直接读取或 mmap；整数使用当前主机的 little-endian 格式。

| 文件 | 内容 |
|---|---|
| `selected_indices.npy` | INT32 `[new,2048]`，逐 query 的绝对 KV 位置，保留原 top-k 输出顺序；不保证按分数排序 |
| `query_positions.npy` | INT32 `[new]`，query 绝对位置 |
| `unique_token_ids.npy` | INT32 `[U]`，历史 + new 的并集，按地址升序；用于每条 KV 只搬一次 |
| `source_byte_offsets.npy` | INT64 `[U]`，连续 host cache 中的源偏移 `token_id × 656`，每项读取 656 B |
| `compact_indices.npy` | INT32 `[new,2048]`，指向按 `unique_token_ids` 顺序打包的 GPU cache；满足 `unique_token_ids[compact_indices] == selected_indices` |
| `reference_counts.npy` | INT64 `[history+new]`，每个 KV 被多少个 query 选中，保留零命中位置 |
| `contiguous_runs.npy` | INT32 `[R,2]`，每行是 `(起始绝对 token, 连续 token 数)`，可精确合并相邻记录 |
| `manifest.json` | 来源、原始索引与输入 SHA-256、形状、dtype、统计与地址布局假设 |

源偏移假设 history 和 new 连续存放；若拆成两个 host buffer，new 的局部偏移为 `(token_id-history)×656`。这里没有导出 KV 内容；地址微基准可填充可校验的合成记录，数值模型回放则需另外生成对应 MLA cache。new cache 的写入/回传开销也不在此处的搬入字节量内。

```python
from pathlib import Path
import numpy as np

p = Path("GR/generated/kv_hit_replay/h65536_n4096")
offsets = np.load(p / "source_byte_offsets.npy", mmap_mode="r")
remap = np.load(p / "compact_indices.npy", mmap_mode="r")
# 将 host[offsets[i]:offsets[i]+656] 搬到 staging[i*656:(i+1)*656]。
# MLA 使用 remap 访问 staging 中的记录；物理页布局/尾页 padding 由后续实现处理。
```

复现（只用 CPU）：

```bash
.venv/bin/python -m model_run.export_gr_kv_hits
```

导出时检查全部 21 组的索引 dtype/形状、因果边界、逐 query 无重复、命中并集与原实验一致、引用总数守恒、重映射可逆，以及连续段能精确重建并集。原始实验与导出 NPY 留在被 Git 忽略的 `GR/generated/`；脚本、报告及分布图随仓库保存。
"""
    (args.report_dir / "gr_kv_hit_distribution.md").write_text(report)


if __name__ == "__main__":
    main()
