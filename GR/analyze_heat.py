"""Measure user popularity from archived reference data; export figures and tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import ijson
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dataset import DEFAULT_DATA_ROOT


def summarize(weights: np.ndarray, dataset: str, field: str) -> tuple[dict, dict]:
    values = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{dataset}/{field}: expected finite, positive user weights")
    values = np.sort(values)[::-1]
    n = len(values)
    cumulative = np.cumsum(values)
    total = float(cumulative[-1])
    shares = cumulative / total
    # Equivalent to the standard ascending-rank Gini formula.
    rank = np.arange(1, n + 1, dtype=np.float64)
    gini = (n + 1 - 2 * float(np.dot(rank, values)) / total) / n
    percentiles = np.quantile(values, [0, 0.5, 0.9, 0.99, 1])
    row = {
        "dataset": dataset,
        "field": field,
        "users": n,
        "total_weight": total,
        "mean_weight": total / n,
        "min_weight": float(percentiles[0]),
        "median_weight": float(percentiles[1]),
        "p90_weight": float(percentiles[2]),
        "p99_weight": float(percentiles[3]),
        "max_weight": float(percentiles[4]),
        "gini": gini,
        "effective_users": total**2 / float(np.dot(values, values)),
        "users_for_50pct_traffic_pct": 100 * (int(np.searchsorted(shares, 0.5)) + 1) / n,
    }
    for pct in (1, 5, 10, 20, 50):
        row[f"top_{pct}pct_traffic_pct"] = 100 * float(shares[math.ceil(n * pct / 100) - 1])
    # Curves are downsampled for artifacts; all statistics use the complete population.
    points = np.unique(np.concatenate(([1, n], np.geomspace(1, n, min(n, 800)).astype(int))))
    curve = {
        "user_fraction": points / n,
        "traffic_fraction": shares[points - 1],
        "relative_weight": values[points - 1] / (total / n),
    }
    return row, curve


def fingerprint(path: Path) -> dict:
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def load_interaction_weights(path: Path) -> np.ndarray:
    users: Counter = Counter()
    with path.open("rb") as source:
        for _, pairs in ijson.kvitems(source, ""):
            for uid, count in pairs:
                if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                    raise ValueError(f"{path}: invalid frequency {count!r}")
                users[int(uid)] += count
    return np.fromiter(users.values(), dtype=np.float64, count=len(users))


def plot_main(rows: list[dict], curves: dict, output: Path) -> None:
    selected = [r for r in rows if r["field"] in ("interaction_count", "pv_int")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    colors = ["#2463a6", "#ed8936", "#3d9967", "#9754aa", "#cf5263", "#6b7280"]
    for row, color in zip(selected, colors, strict=True):
        curve = curves[(row["dataset"], row["field"])]
        label = row["dataset"].replace("industrial_", "Industrial ")
        if row["field"] == "pv_int":
            label += " (synthetic)"
        x, y = curve["user_fraction"], curve["traffic_fraction"]
        axes[0].plot(np.r_[0, x * 100], np.r_[0, y * 100], color=color, label=label, lw=2)
        axes[1].loglog(x * 100, curve["relative_weight"], color=color, lw=1.8)
    axes[0].plot([0, 100], [0, 100], "--", color="#aaa", lw=1, label="Uniform")
    axes[0].set(
        xlabel="Hottest users (% of users)",
        ylabel="Cumulative traffic (%)",
        title="Traffic concentration",
        xlim=(0, 100),
        ylim=(0, 100),
    )
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].axhline(1, color="#aaa", ls="--", lw=1)
    axes[1].set(
        xlabel="User rank percentile (hot to cold, log scale)",
        ylabel="User weight / mean weight (log scale)",
        title="Relative popularity by rank",
    )
    positions = np.arange(len(selected))
    axes[2].barh(
        positions - 0.18,
        [r["top_1pct_traffic_pct"] for r in selected],
        height=0.36,
        color="#2463a6",
        label="Top 1% users",
    )
    axes[2].barh(
        positions + 0.18,
        [r["top_10pct_traffic_pct"] for r in selected],
        height=0.36,
        color="#ed8936",
        label="Top 10% users",
    )
    axes[2].set_yticks(positions, [r["dataset"] for r in selected], fontsize=8)
    axes[2].invert_yaxis()
    axes[2].set(xlabel="Share of traffic (%)", title="Head traffic share")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.18)
        axis.set_axisbelow(True)
    fig.suptitle(
        "User popularity: interaction-count proxies vs synthetic industrial profiles", fontsize=13
    )
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"user_heat_distribution.{extension}", dpi=180)
    plt.close(fig)


def plot_industrial(curves: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for axis, dataset in zip(axes, ("industrial_100k", "industrial_10M"), strict=True):
        for field, color, style in [
            ("pv_share", "#2463a6", "-"),
            ("pv_int", "#3d9967", "--"),
            ("pv_scaled_1_100", "#cf5263", "-"),
        ]:
            curve = curves[(dataset, field)]
            axis.plot(
                np.r_[0, curve["user_fraction"] * 100],
                np.r_[0, curve["traffic_fraction"] * 100],
                label=field,
                color=color,
                ls=style,
                lw=2,
            )
        axis.plot([0, 100], [0, 100], ":", color="#999", label="Uniform")
        axis.set(
            title=dataset,
            xlabel="Hottest users (% of users)",
            ylabel="Cumulative traffic (%)",
            xlim=(0, 100),
            ylim=(0, 100),
        )
        axis.legend(fontsize=9)
        axis.grid(alpha=0.18)
    fig.suptitle("Industrial field choice changes the simulated request distribution", fontsize=13)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"industrial_heat_fields.{extension}", dpi=180)
    plt.close(fig)


def report(rows: list[dict], output: Path) -> None:
    by_key = {(r["dataset"], r["field"]): r for r in rows}
    text = [
        "# 用户访问热度分布",
        "",
        "统计对象是输入用户抽样的相对权重，所有数据均为全量统计；曲线绘制最多取 800 个排名点。",
        "",
        "![用户热度分布](user_heat_distribution.png)",
        "",
        "## 数据口径",
        "",
        (
            "- Beauty、Games、Books、Clothing：累加各自 `timestep_map.json` 中每个用户的交互次数。"
            "这是历史交互活跃度代理，不是线上请求日志；时间步是用户内部交互时间的排名，不是全局秒级时间。"
        ),
        (
            "- industrial：CSV 来自归档中的合成数据生成流程。`pv_share` 是构造的访问占比，"
            "`pv_int` 是整数化访问次数，`pv_scaled_1_100` 是再线性缩放、取整到 1–100 的值。"
            "原 input generator 使用最后一个字段。生成分布参数是否来自真实工业统计，归档脚本未说明。"
        ),
        "- 主图使用 industrial 的 `pv_int`，下方单独比较全部三个字段。每个数据集内部将权重归一化后比较。",
        "",
        "## 全量统计",
        "",
        (
            "头部用户按权重从高到低排列，人数向上取整。Gini 越大，热度越集中；"
            "“50% 流量所需用户”越少，集中程度越高。"
        ),
        "",
        "| 数据集 | 热度字段 | 用户数 | Top 1% 流量 | Top 10% 流量 | Top 20% 流量 | Gini | 50% 流量所需用户 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        text.append(
            f"| {row['dataset']} | {row['field']} | {row['users']:,} | "
            f"{row['top_1pct_traffic_pct']:.2f}% | {row['top_10pct_traffic_pct']:.2f}% | "
            f"{row['top_20pct_traffic_pct']:.2f}% | {row['gini']:.3f} | "
            f"{row['users_for_50pct_traffic_pct']:.2f}% |"
        )
    real = [r for r in rows if r["field"] == "interaction_count"]
    ordered = sorted(real, key=lambda r: r["gini"], reverse=True)
    text += [
        "",
        "## 对 KV cache 实验的含义",
        "",
        "按 Gini 排列，四个交互数据集的热度集中程度从高到低为："
        + " → ".join(r["dataset"] for r in ordered)
        + "。",
        "",
    ]
    for dataset in ("industrial_100k", "industrial_10M"):
        original = by_key[(dataset, "pv_int")]
        scaled = by_key[(dataset, "pv_scaled_1_100")]
        text.append(
            f"- {dataset}：采用 `pv_int` 时 Top 10% 用户占 {original['top_10pct_traffic_pct']:.2f}% 流量；"
            f"采用原脚本的 `pv_scaled_1_100` 时为 {scaled['top_10pct_traffic_pct']:.2f}%。"
            "这些字段不能当作等价的热度权重。"
        )
    text += [
        "",
        (
            "`pv_scaled_1_100` 的 min-max 缩放依赖样本最大值，并带有向下取整和最小值 1。"
            "因此增加合成用户数量时，该字段归一化后的集中程度也可能改变；"
            "对比缓存策略时应固定热度字段，避免把缩放效应误认为用户规模效应。"
        ),
        "",
        (
            "这些图只描述用户边际热度。实际缓存命中还取决于复访时间、其他请求造成的复用距离、"
            "历史 KV 大小、缓存容量和淘汰策略。这里的用户热度排序不能直接解释为命中率。"
        ),
        "",
        (
            "对于“历史稳定、候选更新”的实验，可将热度权重归一化后分配每个用户的访问次数，"
            "再单独安排复访时间。若只选用户子集，应在子集内重新归一化，且记录选取方法。"
        ),
        "",
        "## Industrial 字段对比",
        "",
        "![Industrial 字段对比](industrial_heat_fields.png)",
        "",
        "## 复现与产物",
        "",
        "```bash",
        "uv sync --group analysis  # 新环境安装分析依赖",
        ".venv/bin/python -m GR.analyze_heat --output GR/analysis",
        "```",
        "",
        "- `heat_summary.csv` / `heat_summary.json`：完整统计，含频次分位数、Top 5%/50% 和有效用户数。",
        "- `heat_curves.csv`：图中使用的排名曲线采样点。",
        "- `heat_provenance.json`：输入路径、字节数、SHA-256、统计时间和工具版本。",
        "- 两张图各提供 PNG、SVG、PDF。",
        "",
        (
            "有效用户数定义为 `1 / sum(p_u**2)`，衡量边际分布的集中程度，不代表实际去重用户数。"
            "归一化概率为 `p_u = w_u / sum(w)`；本次没有使用额外的热度指数变换（相当于 α=1）。"
        ),
        "",
    ]
    (output / "README.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("GR/analysis"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, curves, sources = [], {}, []
    for dataset in ("beauty", "games", "books", "clothing"):
        path = args.data_root / dataset / "timestep_map.json"
        print(f"Reading {dataset}: {path}", flush=True)
        weights = load_interaction_weights(path)
        row, curve = summarize(weights, dataset, "interaction_count")
        rows.append(row)
        curves[(dataset, "interaction_count")] = curve
        sources.append(fingerprint(path))
        print(json.dumps(row), flush=True)
        del weights
    for size in ("100k", "10M"):
        path = args.data_root / "industrial" / f"users_{size}.csv"
        print(f"Reading industrial_{size}: {path}", flush=True)
        data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2, 3), ndmin=2)
        for index, field in enumerate(("pv_share", "pv_int", "pv_scaled_1_100")):
            row, curve = summarize(data[:, index], f"industrial_{size}", field)
            rows.append(row)
            curves[(f"industrial_{size}", field)] = curve
            print(json.dumps(row), flush=True)
        sources.append(fingerprint(path))
        del data
    with (args.output / "heat_summary.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "heat_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.output / "heat_curves.csv").open("w", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(
            ["dataset", "field", "user_fraction", "traffic_fraction", "relative_weight"]
        )
        for (dataset, field), curve in curves.items():
            for x, y, z in zip(*curve.values(), strict=True):
                writer.writerow([dataset, field, x, y, z])
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "sources": sources,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "ijson": ijson.__version__,
        "population": "all users in each input file",
        "heat_exponent": 1,
        "quantile_method": "numpy linear",
        "top_fraction_rounding": "ceil",
        "industrial_primary_field": "pv_int",
    }
    (args.output / "heat_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_main(rows, curves, args.output)
    plot_industrial(curves, args.output)
    report(rows, args.output)
    print(f"Artifacts written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
