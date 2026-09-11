"""Audit and report all 315 independent inputs and 945 three-layer KV hit results."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from GR.input_generator import DEFAULT_ITEM_LENGTHS as NEWS
from GR.input_generator import DEFAULT_USER_LENGTHS as HISTORIES
from model_run.export_gr_kv_hits import export_case, plt
from model_run.measure_gr_content_matrix import ROOT, case_name
from model_run.sweep_gr_content_matrix import complete


def label(n):
    return f"{n // 1024}K" if n >= 1024 else str(n)


def load_rows(root):
    rows, validations = [], []
    history_hashes, item_hashes, input_hashes = {}, {}, set()
    peaks = []
    for h in HISTORIES:
        for u in range(3):
            assert complete(root, h, u), f"Incomplete group: h{h} u{u}"
            peaks.append(
                json.loads((root / f"h{h}_u{u}/complete.json").read_text())["peak_allocated_mib"]
            )
            for n in NEWS:
                for i in range(3):
                    name = case_name(h, u, n, i)
                    checks = json.loads((root / f"{name}_validation.json").read_text())
                    assert len(checks) == 9
                    assert sum("ffn" in c for c in checks) == 6
                    assert (
                        sum(
                            c.get("attention", {}).get("cpu_staged_output_bitwise_equal", False)
                            for c in checks
                        )
                        == 3
                    )
                    validations.extend(checks)
                    source_hash = None
                    for layer in range(3):
                        r = json.loads((root / f"layer{layer}" / name / "result.json").read_text())
                        assert (
                            r["history_tokens"],
                            r["new_tokens"],
                            r["history_variant"],
                            r["item_variant"],
                            r["layer"],
                        ) == (h, n, u, i, layer)
                        assert r["total_references"] == n * 2048
                        assert (
                            r["unique_history_tokens"] + r["unique_new_tokens"]
                            == r["unique_mla_cache_tokens"]
                        )
                        assert r["unique_token_bytes"] == 656 * r["unique_mla_cache_tokens"]
                        assert 0 <= r["unique_new_tokens"] <= n
                        assert 0 < r["unique_history_tokens"] <= h
                        if source_hash is None:
                            source_hash = r["source_input_sha256"]
                        assert source_hash == r["source_input_sha256"]
                        assert (
                            history_hashes.setdefault((h, u), r["history_sha256"])
                            == r["history_sha256"]
                        )
                        assert (
                            item_hashes.setdefault((n, i), r["candidate_suffix_sha256"])
                            == r["candidate_suffix_sha256"]
                        )
                        r.update(
                            case=name,
                            history_coverage_pct=100 * r["unique_history_tokens"] / h,
                            new_coverage_pct=100 * r["unique_new_tokens"] / n,
                            history_mib=656 * r["unique_history_tokens"] / 2**20,
                            all_mib=r["unique_token_bytes"] / 2**20,
                        )
                        rows.append(r)
                    assert (
                        source_hash
                        == hashlib.sha256(
                            (root / "inputs" / name / "input.jsonl").read_bytes()
                        ).hexdigest()
                    )
                    assert source_hash not in input_hashes
                    input_hashes.add(source_hash)
    assert len(input_hashes) == 315 and len(rows) == 945
    for h in HISTORIES:
        assert len({history_hashes[h, u] for u in range(3)}) == 3
    for n in NEWS:
        assert len({item_hashes[n, i] for i in range(3)}) == 3
    metrics = {
        name: {
            "max_nrmse": max(c[name]["nrmse"] for c in validations if name in c),
            "min_cosine": min(c[name]["cosine"] for c in validations if name in c),
        }
        for name in ("index", "attention", "ffn")
    }
    metrics.update(
        inputs=315,
        layer_results=945,
        bitwise_staging_cases=945,
        peak_allocated_mib=max(peaks),
        independent_history_item_hashes=True,
    )
    return rows, metrics


def plot_overviews(rows, target):
    for layer in range(3):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
        for ax, key, title in zip(
            axes, ("history_coverage_pct", "new_coverage_pct"), ("History coverage", "New coverage")
        ):
            values = np.array(
                [
                    [
                        np.mean(
                            [
                                r[key]
                                for r in rows
                                if r["layer"] == layer
                                and r["history_tokens"] == h
                                and r["new_tokens"] == n
                            ]
                        )
                        for n in NEWS
                    ]
                    for h in HISTORIES
                ]
            )
            im = ax.imshow(values, vmin=0, vmax=100, cmap="Blues", aspect="auto")
            ax.set(
                xticks=range(len(NEWS)),
                xticklabels=[label(n) for n in NEWS],
                yticks=range(len(HISTORIES)),
                yticklabels=[label(h) for h in HISTORIES],
                xlabel="New tokens",
                ylabel="History tokens",
                title=title,
            )
            for y in range(len(HISTORIES)):
                for x in range(len(NEWS)):
                    ax.text(
                        x,
                        y,
                        f"{values[y, x]:.1f}%",
                        ha="center",
                        va="center",
                        color="white" if values[y, x] > 55 else "black",
                        fontsize=9,
                    )
            fig.colorbar(im, ax=ax, label="Mean of 9 content pairs (%)")
        fig.suptitle(f"Layer {layer} | 3 histories x 3 item variants per shape")
        for ext in ("png", "svg"):
            fig.savefig(target / f"gr_content_matrix_layer{layer}.{ext}", dpi=160)
        plt.close(fig)


def plot_positions(rows, root, target):
    from matplotlib.ticker import PercentFormatter

    for layer in range(3):
        for n in NEWS:
            fig, axes = plt.subplots(len(HISTORIES), 1, figsize=(16, 8), layout="constrained")
            for h, ax in zip(HISTORIES, axes):
                density = np.zeros((h + n) // 64)
                group = [
                    r
                    for r in rows
                    if r["layer"] == layer and r["history_tokens"] == h and r["new_tokens"] == n
                ]
                assert len(group) == 9
                for r in group:
                    ids = np.asarray(
                        json.loads(
                            (
                                root / f"layer{layer}" / r["case"] / "unique_token_ids.json"
                            ).read_text()
                        )
                    )
                    assert len(ids) == r["unique_mla_cache_tokens"] and np.all(np.diff(ids) > 0)
                    density += np.bincount(ids // 64, minlength=len(density))
                density /= 9 * 64
                im = ax.imshow(
                    density[None, :],
                    aspect="auto",
                    interpolation="none",
                    cmap="Blues",
                    vmin=0,
                    vmax=1,
                    extent=(0, (h + n) / 1024, 0.5, -0.5),
                )
                ax.axvline(h / 1024, color="#22d3ee", ls="--", lw=1)
                ax.set(
                    yticks=[], title=f"{label(h)} history", xlabel="Absolute KV position (K tokens)"
                )
            fig.colorbar(
                im,
                ax=axes,
                label="Mean hit density over 9 independent content pairs",
                format=PercentFormatter(xmax=1),
                ticks=[0, 0.25, 0.5, 0.75, 1],
                shrink=0.85,
            )
            fig.suptitle(f"Layer {layer} | {label(n)} new | 64 tokens per bin")
            for ext in ("png", "svg"):
                fig.savefig(target / f"gr_content_positions_layer{layer}_n{n}.{ext}", dpi=160)
            plt.close(fig)


def write_viewer(rows, target):
    keys = (
        "layer",
        "history_tokens",
        "new_tokens",
        "history_variant",
        "item_variant",
        "history_coverage_pct",
        "new_coverage_pct",
        "unique_history_tokens",
        "unique_new_tokens",
        "all_mib",
    )
    data = [{k: r[k] for k in keys} for r in rows]
    html = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>三层 KV 命中 · 九种内容组合</title>
<style>body{font:16px system-ui;margin:32px;color:#17243b;background:#f8fafc}select{font:inherit;margin:0 24px 0 8px;padding:6px}.layers{display:grid;grid-template-columns:repeat(3,minmax(270px,1fr));gap:20px}table{width:100%;border-collapse:collapse;background:white}th,td{border:1px solid #d5deea;padding:12px;text-align:center}td strong{display:block;font-size:20px}td small{display:block;margin-top:8px;color:#31435c}h1{font-size:24px}p{line-height:1.7}@media(max-width:1200px){.layers{grid-template-columns:1fr}}</style>
<h1>三层 KV 命中：3 份 history × 3 份 item</h1><p>315 份输入，945 组层级结果。每格显示历史覆盖率 / new 覆盖率，括号内为去重 token 数。行是 history 内容样本，列是 item 内容样本。</p>
<p>History <select id="h"></select>New <select id="n"></select></p><div class="layers" id="layers"></div>
<script>const data=DATA;const hs=HISTORIES,ns=NEWS;const label=n=>n>=1024?(n/1024)+'K':String(n);
for(const [id,values] of [['h',hs],['n',ns]]){const s=document.getElementById(id);values.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=label(v);s.append(o)});s.onchange=render;}
function render(){const h=+document.getElementById('h').value,n=+document.getElementById('n').value;const container=document.getElementById('layers');container.replaceChildren();for(let l=0;l<3;l++){const section=document.createElement('section');const heading=document.createElement('h2');heading.textContent='Layer '+l;section.append(heading);const table=document.createElement('table');table.innerHTML='<tr><th></th><th>Item 0</th><th>Item 1</th><th>Item 2</th></tr>';for(let u=0;u<3;u++){const tr=document.createElement('tr');const th=document.createElement('th');th.textContent='History '+u;tr.append(th);for(let i=0;i<3;i++){const r=data.find(r=>r.layer===l&&r.history_tokens===h&&r.new_tokens===n&&r.history_variant===u&&r.item_variant===i);const td=document.createElement('td');td.style.background=`rgba(37,99,235,${r.history_coverage_pct/500})`;td.innerHTML='<strong>'+r.history_coverage_pct.toFixed(2)+'%</strong> / '+r.new_coverage_pct.toFixed(2)+'%<small>('+r.unique_history_tokens.toLocaleString()+' / '+r.unique_new_tokens.toLocaleString()+')<br>'+r.all_mib.toFixed(3)+' MiB</small>';tr.append(td)}table.append(tr)}section.append(table);container.append(section)}}render();</script></html>"""
    html = (
        html.replace("DATA", json.dumps(data))
        .replace("HISTORIES", json.dumps(HISTORIES))
        .replace("NEWS", json.dumps(NEWS))
    )
    (target / "gr_content_matrix.html").write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report-dir", type=Path, default=Path("docs/extend_step_profile"))
    parser.add_argument("--export-replay", action="store_true")
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    rows, metrics = load_rows(args.root)
    (args.root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.root / "validation_summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    columns = [
        "layer",
        "history_tokens",
        "new_tokens",
        "history_variant",
        "item_variant",
        "unique_history_tokens",
        "unique_new_tokens",
        "history_coverage_pct",
        "new_coverage_pct",
        "history_mib",
        "all_mib",
        "history_sha256",
        "candidate_suffix_sha256",
        "source_input_sha256",
    ]
    with (args.report_dir / "gr_content_matrix.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot_overviews(rows, args.report_dir)
    plot_positions(rows, args.root, args.report_dir)
    write_viewer(rows, args.report_dir)
    report = """# 第 0、1、2 层：九种内容组合的 KV 命中

**315 份输入 × 3 层 = 945 组层级结果**：history = 4K/16K/64K/256K/1024K；new = 64/128/256/512/1K/2K/4K。每个形状使用 3 份 history 和 3 份独立 item 内容，完整交叉成九种组合。

[交互对比九种组合](gr_content_matrix.html) · [全部 945 行 CSV](gr_content_matrix.csv) · [位置密度图](gr_kv_hit_distribution.md)

## 输入与执行口径

同一 history 长度下，history 样本 0/1/2 的文本分别固定，不随 new 长度或 item 样本变化。同一 new 长度下，item 样本 0/1/2 的候选后缀分别固定，在所有 history 长度与内容样本间复用。SHA-256 检查确认 history 与 item 的独立组合，以及每种内容确实有三份不同文本。固定 seed=42；这里的三份样本不提供总体置信区间。

History 包含固定指令，new 是其后的候选区块，KV 边界精确为 H 与 H+N。Generator 对完整 prompt 重新编码并检查精确 token 数。短 new 自动减少完整候选条目，最多保留 20 个；没有通过随机 token 或截断半句凑预算。

真实 checkpoint embedding → 第 0 层 → 第 1 层 → 第 2 层。第 0、1 层执行 attention、输出投影、残差、RMSNorm 和 dense SwiGLU；第 2 层记录 indexer top-k，并执行 attention 数值与搬运校验。共享同一 history 的 21 个分支复用其因果 prefill，各分支只覆盖自己的 new cache，保留独立的 new hidden states。已将 new 顺序反转（先 4K 后 64）复跑，18 组层级 top-k 集合完全一致，检查分支间无后缀污染。

Top-k=2048，不做 Hadamard。Indexer 每 128 个 query 分批，但每个 query 都搜索完整因果可见 KV；命中覆盖率按整批 new query 的 top-k 并集计算，MLA 的 new 仍整批执行。历史以 512-token chunk 前向，层间 hidden states 暂存在 CPU。第 2 层历史仅准备完整 K/MLA cache，并执行首末 chunk 的抽样校验。

沿用 revision 2 的混合精度路径：checkpoint FP8 权重 / MXFP8 activation，FP32 Norm affine、残差归一化及 SwiGLU 中间计算；K/V 拆分和 O 使用 checkpoint 反量化 BF16。256K/1024K 超出 checkpoint 的 163840 位置上限，只扩展 RoPE 缓存，沿用原 YaRN 参数。长位置 RoPE 通过独立 float64 参考检查（NRMSE < 1%、cosine > 0.9999）；不代表长上下文任务质量或与官方完整模型逐位一致。

## 九种组合的覆盖率范围

每格为“历史均值 [最小–最大] / new 均值 [最小–最大]”，单位为百分比；九个样本等权。完整单组合值见交互页或 CSV。

| History + new | Layer 0 | Layer 1 | Layer 2 |
|---|---:|---:|---:|
"""
    for h in HISTORIES:
        for n in NEWS:
            cells = []
            for layer in range(3):
                group = [
                    r
                    for r in rows
                    if r["layer"] == layer and r["history_tokens"] == h and r["new_tokens"] == n
                ]
                parts = []
                for key in ("history_coverage_pct", "new_coverage_pct"):
                    v = [r[key] for r in group]
                    parts.append(f"{np.mean(v):.2f} [{min(v):.2f}–{max(v):.2f}]")
                cells.append(" / ".join(parts))
            report += f"| {label(h)} + {label(n)} | " + " | ".join(cells) + " |\n"
    report += "\n## 三层均值热力图\n"
    for layer in range(3):
        report += f"\n![Layer {layer}](gr_content_matrix_layer{layer}.png)\n"
    report += f"\n## 校验\n\n945 组 new 的 CPU token gather、GPU staging 与索引重映射回放均与 resident MLA 输出逐位一致。每份输入每层抽查历史首末 chunk 和 new，检查 index logits、MLA 与 FFN 的独立 FP32 参考；复用的历史校验不会算作新的独立样本。全部组的峰值 PyTorch allocated 显存为 **{metrics['peak_allocated_mib'] / 1024:.3f} GiB**，不等于驱动进程占用。\n\n| 参考项 | 最大 NRMSE | 最小 cosine |\n|---|---:|---:|\n"
    for name in ("index", "attention", "ffn"):
        report += (
            f"| {name} | {metrics[name]['max_nrmse']:.6f} | {metrics[name]['min_cosine']:.6f} |\n"
        )
    report += """
## 输出与复现

仓库保留报告、引用图片、CSV 和交互页。原始输入、top-k/NPY 回放及日志属于可再生成的大型输出，清理后需先运行测量命令重建，再重新生成报告；小型统计与校验 JSON 留在本地 `GR/generated/`。以下路径描述复现时生成的产物。

- `GR/generated/content_matrix/inputs/h{H}_u{U}_n{N}_i{I}/`：315 份精确输入。
- `GR/generated/content_matrix/layer{L}/h{H}_u{U}_n{N}_i{I}/`：945 份原始 INT32 top-k、历史与 new 并集、统计和输入哈希。
- `summary.json` / `validation_summary.json`：完整统计与校验摘要；每份输入另有 `_validation.json`。
- 每条压缩 MLA KV 为 656 B；CSV 的 `history_mib` 只计历史，`all_mib` 包含历史和 new，均按本批 query 去重。

```bash
env PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m model_run.sweep_gr_content_matrix
.venv/bin/python -m model_run.report_gr_content_matrix
# 可选：导出全部精确 NPY 搬运回放数组
.venv/bin/python -m model_run.report_gr_content_matrix --export-replay
```

调度按 15 个 history 内容组保存完成标记，重跑会跳过完整组。旧版单内容实验使用 `GR/generated/cache_union_sweep/`、`GR/generated/multilayer_hits_validated/`，不混入本次内容矩阵。
"""
    (args.report_dir / "gr_multilayer_kv_hits.md").write_text(report)
    positions = """# 三层 KV 命中位置：九种内容组合

本页对应 **315 份输入、945 组层级结果**，详见 [三层覆盖率与校验](gr_multilayer_kv_hits.md)；[交互页](gr_content_matrix.html) 和 [CSV](gr_content_matrix.csv) 保留九种组合的独立数值。

每格覆盖连续 64 个 KV token，颜色表示九种内容组合的平均命中密度：对每份输入先按整批 new 的 top-k 去重，再计算该格的命中比例，最后对九份输入等权平均。**没有把不同文本的命中位置合成一次请求的并集。** 复现时，每份输入精确位置写入各层 `selected_indices.pt` 和 `unique_token_ids.json`。横轴是绝对 KV 位置，青色虚线是 history/new 边界；长 history 下的短 new 区域在总览中很窄，具体 new 覆盖率请看交互页。

Top-k=2048；new 为 64/128/256/512/1K/2K/4K，history 为 4K/16K/64K/256K/1024K。覆盖率是选中 KV 的并集比例，不是硬件 cache hit rate。分箱只用于展示，不表示必须按 64-token 整页搬运。
"""
    for layer in range(3):
        positions += f"\n## Layer {layer}\n"
        for n in NEWS:
            filename = f"gr_content_positions_layer{layer}_n{n}"
            positions += f"\n### {label(n)} new\n\n![Layer {layer}, {label(n)} new]({filename}.png)\n\n[SVG]({filename}.svg)\n"
    (args.report_dir / "gr_kv_hit_distribution.md").write_text(positions)
    if args.export_replay:
        for r in rows:
            name = r["case"]
            export_case(
                args.root / f"layer{r['layer']}" / name,
                args.root / f"layer{r['layer']}_replay" / name,
                r["history_tokens"],
                r["new_tokens"],
            )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
