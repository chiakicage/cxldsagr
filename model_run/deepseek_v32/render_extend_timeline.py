"""Render an extend CUPTI trace as an SVG with exclusive operation lanes."""

import argparse
import json
import math
from html import escape
from pathlib import Path

LANES = [
    ("投影 GEMM", "#3979b7"),
    ("MQA logits", "#dc8c2d"),
    ("Sparse MLA（融合）", "#b34c66"),
    ("布局转换 / 拼接", "#8565b0"),
    ("独立 activation 量化", "#419e9b"),
    ("Top-k", "#754c36"),
    ("Norm / RoPE", "#568f4c"),
    ("Cache 追加", "#6b7786"),
    ("输出复制 / 其余辅助", "#8993a3"),
]


def lane(step):
    if step.endswith("/gemm") or step == "index_weights":
        return 0
    if step == "index/mqa_logits":
        return 1
    if step == "mla/prefill":
        return 2
    if step.endswith("/layout"):
        return 3
    if step.endswith("/quantize") or step == "index/q_quantize":
        return 4
    if step == "index/topk":
        return 5
    if step.endswith(("_norm", "/rope")):
        return 6
    if step == "cache/append":
        return 7
    return 8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report-index", type=int, default=0)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    report = summary["reports"][args.report_index]
    events = json.loads(Path(report["trace"]).read_text())["traceEvents"]
    scopes = [
        e for e in events if e.get("cat") == "user_annotation" and e["name"].startswith("step::")
    ]
    launches = {
        e["args"]["correlation"]: e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})
    }
    activities = []
    for e in events:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        launch = launches[e["args"]["correlation"]]
        parents = [
            s
            for s in scopes
            if (s["pid"], s["tid"]) == (launch["pid"], launch["tid"])
            and s["ts"] <= launch["ts"] <= s["ts"] + s["dur"]
        ]
        step = min(parents, key=lambda s: s["dur"])["name"][6:]
        activities.append((e, step, lane(step)))
    total = sum(e["dur"] for e, _, _ in activities) / 1000
    assert math.isclose(total, report["gpu_total_ms"], abs_tol=1e-6)
    start = min(e["ts"] for e, _, _ in activities)
    span = (max(e["ts"] + e["dur"] for e, _, _ in activities) - start) / 1000
    tick = max(1, math.ceil(span / 14))
    limit = math.ceil(span / tick) * tick
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="760" viewBox="0 0 1600 760">',
        '<rect width="100%" height="100%" fill="#fafbfd"/>',
        (
            '<style>text{font-family:system-ui,"Noto Sans CJK SC","Microsoft YaHei",'
            "sans-serif;fill:#223047;font-size:14px}</style>"
        ),
    ]

    def text(x, y, value):
        svg.append(f'<text x="{x}" y="{y}">{escape(value)}</text>')

    text(28, 38, "DeepSeek V3.2 extend · GPU 时间轴")
    text(
        28,
        65,
        f"{report['history'] // 1024}K history + "
        f"{report['new_tokens'] // 1024}K new · chunk={report['chunk']} · "
        f"{summary['mode']} · 实际 CUDA kernel 起止时间",
    )
    text(
        28,
        91,
        f"本次 trace 跨度 {span:.3f} ms；GPU 活动累计 {total:.3f} ms；"
        f"独立测量端到端 {report['e2e_ms']:.3f} ms。",
    )
    for t in range(0, limit + 1, tick):
        x = 290 + t / limit * 1265
        svg.append(f'<path d="M{x:.3f},136 v450" stroke="#e0e5ec"/>')
        text(round(x - 5, 3), 127, str(t))
    text(1480, 615, "时间（ms）")
    text(28, 163, "全部 GPU 活动")
    for i, (label, _) in enumerate(LANES):
        text(28, 208 + i * 44, label)
        subtotal = sum(e["dur"] for e, _, group in activities if group == i) / 1000
        text(220, 208 + i * 44, f"{subtotal:.2f} ms")
    for e, step, group in activities:
        begin = (e["ts"] - start) / 1000
        duration = e["dur"] / 1000
        title = escape(
            f"{step} | {begin:.6f}–{begin + duration:.6f} ms | {duration:.6f} ms\n{e['name']}"
        )
        for y in (147, 192 + group * 44):
            svg.append(
                f'<rect x="{290 + begin / limit * 1265:.4f}" y="{y}" '
                f'width="{duration / limit * 1265:.4f}" height="22" '
                f'fill="{LANES[group][1]}"><title>{title}</title></rect>'
            )
    text(
        28,
        677,
        "横轴以首个 GPU 活动为 0；保留真实间隙，细小 kernel 未人为加宽。悬停查看步骤、时间及 kernel 名。",
    )
    text(
        28, 703, "Sparse MLA 内含 QK、softmax、PV；投影 GEMM 含 API 内 scale 打包、归约等辅助操作。"
    )
    text(
        28,
        729,
        f"时间轴来自一次预热后的 profile（{len(activities)} 个 GPU 活动）；"
        f"端到端为另测的无插桩 {summary['repeats']} 次均值。",
    )
    svg.append("</svg>")
    args.output.write_text("\n".join(svg) + "\n")
    print(f"{len(activities)} activities; span={span:.6f} ms; GPU={total:.6f} ms")


if __name__ == "__main__":
    main()
