"""Plot analytical schedules using measured compute and KV transfer durations."""

from __future__ import annotations

from pathlib import Path

STAGES = ("projection", "index", "topk", "attention", "finish")
STAGE_CODES = dict(zip(STAGES, ("P", "I", "K", "A", "F"), strict=True))
STAGE_NAMES = {
    "projection": "P  Norm + projections",
    "index": "I  Indexer logits",
    "topk": "K  Exact top-k",
    "attention": "A  Sparse attention",
    "finish": "F  V expansion + output + norm + MLP",
}
COLORS = {
    "projection": "#0072B2",
    "index": "#009E73",
    "topk": "#CC79A7",
    "attention": "#56B4E9",
    "finish": "#626A73",
    "h2d": "#E69F00",
    "d2h": "#E69F00",
    "wait": "#F0F1F3",
    "fused": "#E0EFE9",
    "cleanup": "#D9DEE3",
    "next_layer_context": "#FCEABD",
}
INK = "#172B3A"
MUTED = "#566573"


def _milliseconds(nanoseconds):
    return nanoseconds / 1e6


def _percent(value):
    return "N/A" if value is None else f"{value:.1f}%"


def _draw_fused(axis, event, scenario):
    from matplotlib.patches import Rectangle

    start = _milliseconds(event["start_ns"])
    duration = _milliseconds(event["end_ns"] - event["start_ns"])
    axis.add_patch(
        Rectangle(
            (start, -0.28),
            duration,
            1.56,
            facecolor=COLORS["fused"],
            edgecolor="#ABD0C0",
            linewidth=0.8,
            hatch="//",
            zorder=3,
        )
    )
    lines = [
        f"Indexer + prefetch · {duration:.3f} ms",
        f"Fused MFU {_percent(event.get('compute_mfu_percent'))} (includes IO)",
    ]
    if "bytes" in event:
        payload = f"{event['bytes'] / 2**20:.3f} MiB"
        if "fused_prefetch_fraction" in scenario:
            payload += f" · {scenario['fused_prefetch_fraction'] * 100:.2f}% prefetched"
        lines.append(payload)
    if "fused_payload_GBps" in event:
        lines.append(f"Payload / fused time: {event['fused_payload_GBps']:.2f} GB/s")
    axis.text(
        start + duration / 2,
        0.5,
        "\n".join(lines),
        ha="center",
        va="center",
        fontsize=9.3,
        color="#164B3C",
        linespacing=1.35,
        bbox={"facecolor": COLORS["fused"], "edgecolor": "none", "pad": 3},
        zorder=4,
    )


def _draw_dense_transfer(axis, event, scenario):
    from matplotlib.patches import Rectangle

    start = _milliseconds(event["start_ns"])
    duration = _milliseconds(event["end_ns"] - event["start_ns"])
    is_next = event["transfer_role"] == "next_layer_context"
    axis.add_patch(
        Rectangle(
            (start, -0.28),
            duration,
            0.56,
            facecolor=COLORS["next_layer_context"] if is_next else COLORS["h2d"],
            edgecolor="#C79220" if is_next else "white",
            linewidth=0.7,
            hatch="//" if is_next else None,
            zorder=3,
        )
    )
    if is_next:
        label = f"L2 fetch beginning\n{duration:.3f} ms shown · context"
    else:
        label = (
            f"L1 fetch remainder · {duration:.3f} ms shown\n"
            f"Full L1 copy: {event['full_bytes'] / 2**20:.3f} MiB · "
            f"{_milliseconds(event['full_duration_ns']):.6f} ms · "
            f"{event['full_bandwidth_GBps']:.3f} GB/s"
        )
        axis.text(
            start + duration / 2,
            0.50,
            f"L1 copy began {_milliseconds(scenario['prefetch_lead_ns']):.3f} ms before t=0",
            ha="center",
            va="center",
            fontsize=8.7,
            color="#8B5900",
            zorder=4,
        )
    axis.text(
        start + duration / 2,
        0,
        label,
        ha="center",
        va="center",
        fontsize=8.7,
        linespacing=1.15,
        color=INK,
        zorder=4,
    )


def _axis_style(axis, xmax):
    from matplotlib.ticker import MaxNLocator

    axis.set_xlim(0, xmax)
    axis.set_ylim(-0.37, 1.32)
    axis.set_yticks([1, 0], ["L1 compute", "KV transfer"])
    axis.tick_params(axis="y", length=0, pad=9, labelsize=10)
    axis.tick_params(axis="x", color="#C3CBD1", labelcolor=MUTED, labelsize=9)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=4))
    axis.grid(axis="x", color="#E6EAEE", linewidth=0.65)
    axis.set_axisbelow(True)
    for y in (0, 1):
        axis.axhline(y, color="#D8DFE5", linewidth=0.7, zorder=0)
    axis.spines["bottom"].set_color("#C3CBD1")
    for side in ("left", "right", "top"):
        axis.spines[side].set_visible(False)


def _draw_events(axis, scenario, xmax):
    from matplotlib.patches import Rectangle

    has_io = False
    for event in scenario["events"]:
        start = _milliseconds(event["start_ns"])
        duration = _milliseconds(event["end_ns"] - event["start_ns"])
        if duration <= 0:
            continue
        stage = event["stage"]
        lane = event["lane"]
        if lane == "fused":
            has_io = True
            _draw_fused(axis, event, scenario)
            continue
        if lane == "io" and "transfer_role" in event:
            has_io = True
            _draw_dense_transfer(axis, event, scenario)
            continue
        y = 0 if lane == "io" else 1
        is_wait = lane == "wait" or stage == "wait"
        is_cleanup = stage == "cleanup"
        axis.add_patch(
            Rectangle(
                (start, y - 0.28),
                duration,
                0.56,
                facecolor=COLORS[stage],
                edgecolor="#A0A9B2" if is_wait or is_cleanup else "white",
                linewidth=0.7,
                hatch="////" if is_wait else ".." if is_cleanup else None,
                zorder=3,
            )
        )
        if stage in STAGES and duration / xmax >= 0.012:
            has_mfu = "compute_mfu_percent" in event and duration / xmax >= 0.04
            axis.text(
                start + duration / 2,
                y + 0.11 if has_mfu else y,
                STAGE_CODES[stage],
                ha="center",
                va="center",
                color=INK if stage in ("topk", "attention") else "white",
                fontsize=9.5 if has_mfu else 10,
                fontweight="bold",
                zorder=4,
            )
            if has_mfu:
                axis.text(
                    start + duration / 2,
                    y - 0.11,
                    _percent(event["compute_mfu_percent"]),
                    ha="center",
                    va="center",
                    color=INK if stage in ("topk", "attention") else "white",
                    fontsize=8.5,
                    zorder=4,
                )
        elif is_wait and duration / xmax >= 0.065:
            axis.text(
                start + duration / 2,
                y,
                "KV wait",
                ha="center",
                va="center",
                fontsize=9,
                color=MUTED,
                zorder=4,
            )
        if lane == "io":
            has_io = True
            label = f"{stage.upper()}  {duration:.3f} ms"
            if scenario["id"] == "extend_echo_measured":
                residual = 1 - scenario["fused_prefetch_fraction"]
                label = f"Residual {label} · {residual * 100:.2f}%"
            if "bytes" in event:
                label += f"\n{event['bytes'] / 2**20:.3f} MiB"
                if "bandwidth_GBps" in event:
                    label += f" · {event['bandwidth_GBps']:.2f} GB/s"
            midpoint = start + duration / 2
            if duration / xmax >= 0.25:
                axis.text(
                    midpoint,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    linespacing=1.15,
                    color=INK,
                    zorder=4,
                )
            else:
                text_x = min(max(midpoint + 0.03 * xmax, 0.12 * xmax), 0.96 * xmax)
                alignment = "right" if text_x > 0.76 * xmax else "left"
                axis.annotate(
                    label,
                    xy=(midpoint, 0.26),
                    xytext=(text_x, 0.50),
                    ha=alignment,
                    va="center",
                    fontsize=8.7,
                    linespacing=1.1,
                    color="#8B5900",
                    arrowprops={"arrowstyle": "-", "lw": 0.75, "color": "#A8781D"},
                    zorder=5,
                )
    if not has_io:
        axis.text(
            0.018 * xmax,
            0,
            "No KV transfer",
            ha="left",
            va="center",
            fontsize=9,
            color=MUTED,
        )


def _legend(figure, y, *, include_wait, include_fused=False, include_context=False):
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=COLORS[key], label=STAGE_NAMES[key]) for key in STAGES]
    handles.append(Patch(facecolor=COLORS["h2d"], label="Host ↔ GPU KV transfer"))
    if include_wait:
        handles.append(
            Patch(
                facecolor=COLORS["wait"],
                edgecolor="#A0A9B2",
                hatch="////",
                label="Data dependency wait",
            )
        )
    if include_context:
        handles.append(
            Patch(
                facecolor=COLORS["next_layer_context"],
                edgecolor="#C79220",
                hatch="//",
                label="Next-layer IO context",
            )
        )
    if include_fused:
        handles.extend(
            [
                Patch(
                    facecolor=COLORS["fused"],
                    edgecolor="#ABD0C0",
                    hatch="//",
                    label="Fused compute + IO",
                ),
                Patch(
                    facecolor=COLORS["cleanup"],
                    edgecolor="#A0A9B2",
                    hatch="..",
                    label="Fused cleanup",
                ),
            ]
        )
    figure.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.11, y),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=1.3,
        columnspacing=1.75,
        borderaxespad=0,
    )


def _header(figure, title, subtitle):
    figure.text(0.04, 0.955, title, fontsize=16, fontweight="bold", color=INK)
    figure.text(0.04, 0.922, subtitle, fontsize=10, color=MUTED)


def _save(figure, output, stem):
    import matplotlib.pyplot as plt

    for extension in ("svg", "png"):
        metadata = {"Creator": "plot_simulation.py"}
        if extension == "svg":
            metadata["Date"] = None
        figure.savefig(
            output / f"{stem}.{extension}",
            dpi=200,
            facecolor="white",
            metadata=metadata,
        )
    plt.close(figure)


def _source_note(document):
    return document.get("plot_source_note", "Measured stage durations; rescheduled analytically.")


def _draw_primary(document, scenarios, output, *, phase):
    import matplotlib.pyplot as plt

    is_extend = phase == "extend"
    height = 11.8 if is_extend else 8.6
    figure, axes = plt.subplots(len(scenarios), 1, figsize=(12.6, height), sharex=True)
    figure.subplots_adjust(
        left=0.135,
        right=0.97,
        top=0.85,
        bottom=0.245,
        hspace=0.8 if is_extend else 0.75,
    )
    title = "SIMULATION | Extend: one-layer compute and KV transfer"
    subtitle = "DeepSeek V3.2 · H=65,536 · A=128 · L1 (zero-based) · every row starts at L1 P"
    if not is_extend:
        title = "SIMULATION | Prefill: one-layer compute and KV writeback"
        subtitle = "DeepSeek V3.2 · H=65,536 · Q=1,024 · final history chunk 63 · L1 (zero-based)"
    _header(figure, title, subtitle)
    xmax = max(_milliseconds(row["completion_ns"]) for row in scenarios) * 1.035
    for axis, scenario in zip(axes, scenarios, strict=True):
        _axis_style(axis, xmax)
        _draw_events(axis, scenario, xmax)
        completion = _milliseconds(scenario["completion_ns"])
        axis.set_title(
            scenario["label"], loc="left", fontsize=11, fontweight="bold", color=INK, pad=13
        )
        metrics = f"Completion  {completion:.6f} ms"
        if "compute_mfu_percent" in scenario:
            mfu = f"Compute MFU {_percent(scenario['compute_mfu_percent'])}"
            if scenario.get("schedule_mfu_percent") is not None:
                mfu += f" · Schedule MFU {_percent(scenario['schedule_mfu_percent'])}"
            metrics = mfu + "\n" + metrics
        axis.set_title(metrics, loc="right", fontsize=9.5, color=MUTED, pad=13)
        axis.axvline(completion, color=INK, linestyle=(0, (3, 3)), linewidth=0.8, zorder=5)
    axes[-1].set_xlabel(
        "Time since L1 P (norm + projections) starts (ms)", fontsize=10, labelpad=10
    )
    has_fused = any(event["lane"] == "fused" for row in scenarios for event in row["events"])
    has_context = any(
        event.get("transfer_role") == "next_layer_context"
        for row in scenarios
        for event in row["events"]
    )
    _legend(figure, 0.183, include_wait=True, include_fused=has_fused, include_context=has_context)
    if is_extend:
        footnotes = [
            "Block percentages are stage MFU; top-k is N/A. Compute MFU excludes IO; schedule MFU includes modeled IO waits.",
            "Fused MFU includes IO. Payload / fused time is not an independent transfer bandwidth; the kernel is indivisible.",
            "Dense bars show transfer fragments; bytes/rate describe the full L1 copy. The hatched L2 beginning is context only.",
            "Oracle sparse assumes early selection. Separate cache/control and CPU launch gaps are excluded; fused internal work is retained.",
        ]
        start_y, step = 0.099, 0.020
    else:
        serial = next(row for row in scenarios if row["id"] == "prefill_serial")
        overlap = next(row for row in scenarios if row["id"] == "prefill_overlap")
        saved_us = (serial["completion_ns"] - overlap["completion_ns"]) / 1e3
        footnotes = [
            f"Overlapping writeback saves {saved_us:.3f} µs for this chunk and layer. Completion includes IO; this is not full-history latency.",
            "Block percentages are stage MFU; top-k is N/A. Compute MFU excludes IO; schedule MFU includes modeled IO waits.",
            "IO labels: payload MiB and decimal GB/s. Cache management, CPU launch gaps and resource contention are excluded.",
        ]
        start_y, step = 0.093, 0.023
    for index, line in enumerate(footnotes):
        figure.text(0.04, start_y - index * step, line, fontsize=9, color=MUTED)
    figure.text(0.04, 0.019, _source_note(document), fontsize=8.5, color=MUTED)
    _save(figure, output, f"simulation_{phase}")


def draw_simulation(document, output: Path):
    """Write extend and prefill simulations as SVG and PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    scenarios = {row["id"]: row for row in document["scenarios"]}
    groups = {
        "prefill": ("prefill_hbm", "prefill_serial", "prefill_overlap"),
        "extend": (
            "extend_hbm",
            "extend_serial_sparse",
            "extend_oracle_sparse",
            "extend_echo_measured",
            "extend_dense_prefetch",
        ),
    }
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "svg.fonttype": "none",
            "svg.hashsalt": "deepseek-motivation-simulation",
        }
    ):
        for phase, ids in groups.items():
            _draw_primary(document, [scenarios[key] for key in ids], output, phase=phase)
