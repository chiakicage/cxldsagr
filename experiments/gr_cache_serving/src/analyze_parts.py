"""Attribute CUPTI GPU work to innermost runtime NVTX scopes, without double counting."""

import argparse
import csv
import hashlib
import heapq
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def duration(intervals):
    return sum(end - start for start, end in union(intervals)) / 1e6


def intersection(a, b):
    a, b = union(a), union(b)
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if end > start:
            result.append((start, end))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return result


def complement(intervals, start, end):
    cursor, gaps = start, []
    for left, right in intersection(intervals, [(start, end)]):
        if left > cursor:
            gaps.append((cursor, left))
        cursor = right
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


def parse_label(text):
    if not text or not text.startswith("gr/"):
        return None
    fields = dict(item.split("=", 1) for item in text.split("/")[1:])
    for key in ("r", "c", "l"):
        fields[key] = int(fields[key])
    return fields


def attribute_apis(ranges, apis):
    """API launch thread/time, not GPU execution time, determines owning scope."""
    by_thread = defaultdict(list)
    for item in ranges:
        by_thread[item["globalTid"]].append(item)
    result = {}
    api_threads = defaultdict(list)
    for api in apis:
        api_threads[api["globalTid"]].append(api)
    for thread, entries in api_threads.items():
        scopes = sorted(by_thread[thread], key=lambda r: r["start"])
        active = []
        index = 0
        for api in sorted(entries, key=lambda r: r["start"]):
            while index < len(scopes) and scopes[index]["start"] <= api["start"]:
                scope = scopes[index]
                heapq.heappush(active, (-scope["start"], scope["end"], index))
                index += 1
            while active and active[0][1] <= api["start"]:
                heapq.heappop(active)
            if active:
                result[api["correlationId"]] = scopes[active[0][2]]["label"]
    return result


def read_trace(path):
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    ranges = []
    for row in db.execute("SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        item = dict(row)
        label = parse_label(item.get("text") or strings.get(item.get("textId")))
        if label:
            item["label"] = label
            ranges.append(item)
    apis = []
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table in tables:
            apis.extend(dict(row) for row in db.execute(f"SELECT * FROM {table}"))
    owners = attribute_apis(ranges, apis)
    events = []
    for kind in ("KERNEL", "MEMCPY", "MEMSET"):
        table = f"CUPTI_ACTIVITY_KIND_{kind}"
        if table not in tables:
            continue
        for row in db.execute(f"SELECT * FROM {table}"):
            item = dict(row)
            item["kind"] = kind
            item["label"] = owners.get(item["correlationId"])
            item["name"] = strings.get(item.get("shortName"), kind)
            events.append(item)
    db.close()
    if not ranges or not any(e["kind"] == "KERNEL" and e["label"] for e in events):
        raise ValueError("trace has no attributed NVTX/CUDA work")
    return ranges, events


COMPUTE = {
    "qkv_rope",
    "indexer",
    "main_attention",
    "attention_output",
    "attention_other",
    "ffn_dense",
    "ffn_moe",
    "norm_comm",
    "layer",
}


def analyze(path, summary):
    ranges, events = read_trace(path)
    phase_rows, part_rows, window_rows, kernel_rows, io_rows = [], [], [], [], []
    captured = [
        r
        for r in summary["rows"]
        if r["repetition"] == "profile"
        and summary["capture_start"] <= r["ordinal"] < summary["capture_end"]
    ]
    for row in captured:
        ordinal = row["ordinal"]
        for phase in ("prefill", "candidate"):
            scopes = [r for r in ranges if r["label"]["r"] == ordinal and r["label"]["p"] == phase]
            outer = [r for r in scopes if r["label"]["s"] == "phase"]
            if not outer:
                continue
            if len(outer) != 1:
                raise ValueError("phase must have exactly one top-level scope")
            outer = outer[0]
            window = [(outer["start"], outer["end"])]
            work = [
                e
                for e in events
                if e["label"] and e["label"]["r"] == ordinal and e["label"]["p"] == phase
            ]
            interval = lambda seq: [(e["start"], e["end"]) for e in seq]
            kernels = [e for e in work if e["kind"] == "KERNEL"]
            compute = [e for e in kernels if e["label"]["s"] in COMPUTE]
            dense_copy = [
                e for e in work if e["kind"] == "MEMCPY" and e["label"]["s"] == "dense_producer"
            ]
            gathers = [r for r in scopes if r["label"]["s"] == "cpu_gather"]
            waits = [r for r in scopes if r["label"]["s"] == "dense_consumer_wait"]
            all_active = intersection(interval(events), window)
            idle = complement(all_active, outer["start"], outer["end"])
            base = {
                "mode": summary["mode"],
                "ordinal": ordinal,
                "phase": phase,
                "first_visit": row["first_visit"],
            }
            phase_rows.append(
                {
                    **base,
                    "wall_ms": duration(window),
                    "all_gpu_active_ms": duration(all_active),
                    "no_traced_gpu_activity_ms": duration(window) - duration(all_active),
                    "gpu_kernel_sum_ms": sum(e["end"] - e["start"] for e in kernels) / 1e6,
                    "dense_copy_union_ms": duration(interval(dense_copy)),
                    "dense_copy_compute_overlap_ms": duration(
                        intersection(interval(dense_copy), interval(compute))
                    ),
                    "dense_copy_bytes": sum(e.get("bytes", 0) for e in dense_copy),
                    "cpu_gather_sum_ms": sum(e["end"] - e["start"] for e in gathers) / 1e6,
                    "gather_compute_overlap_ms": duration(
                        intersection(interval(gathers), interval(compute))
                    ),
                    "consumer_cpu_wait_sum_ms": sum(e["end"] - e["start"] for e in waits) / 1e6,
                    "gpu_idle_during_cpu_gather_ms": duration(
                        intersection(idle, interval(gathers))
                    ),
                }
            )
            for layer in range(5):
                layer_gather = [e for e in gathers if e["label"]["l"] == layer]
                layer_copy = [e for e in dense_copy if e["label"]["l"] == layer]
                layer_wait = [e for e in waits if e["label"]["l"] == layer]
                io_rows.append(
                    {
                        **base,
                        "layer": layer,
                        "cpu_gather_ms": duration(interval(layer_gather)),
                        "prefix_h2d_ms": duration(interval(layer_copy)),
                        "prefix_h2d_bytes": sum(e.get("bytes", 0) for e in layer_copy),
                        "consumer_cpu_wait_ms": duration(interval(layer_wait)),
                    }
                )
            totals = defaultdict(float)
            kernel_totals = defaultdict(float)
            for event in kernels:
                label = event["label"]
                ms = (event["end"] - event["start"]) / 1e6
                totals[label["l"], label["s"]] += ms
                kernel_totals[label["s"], event["name"]] += ms
            for (layer, part), ms in sorted(totals.items()):
                part_rows.append({**base, "layer": layer, "part": part, "gpu_sum_ms": ms})
            for (part, name), ms in sorted(kernel_totals.items()):
                kernel_rows.append({**base, "part": part, "kernel": name, "gpu_sum_ms": ms})
            if phase == "candidate" and summary["mode"] == "dense_prefetch":
                for layer in range(4):
                    main = [
                        e
                        for e in kernels
                        if e["label"]["l"] == layer and e["label"]["s"] == "main_attention"
                    ]
                    idx = [
                        e
                        for e in kernels
                        if e["label"]["l"] == layer + 1 and e["label"]["s"] == "indexer"
                    ]
                    copy = [e for e in dense_copy if e["label"]["l"] == layer + 1]
                    if main and idx and copy:
                        start, end = min(e["start"] for e in main), max(e["end"] for e in idx)
                        copy_start, copy_end = (
                            min(e["start"] for e in copy),
                            max(e["end"] for e in copy),
                        )
                        window_rows.append(
                            {
                                **base,
                                "producer_layer": layer,
                                "attention_to_next_indexer_ms": (end - start) / 1e6,
                                "copy_start_after_attention_ms": (copy_start - start) / 1e6,
                                "copy_end_after_indexer_ms": (copy_end - end) / 1e6,
                            }
                        )
    unowned = [e for e in events if e["label"] is None]
    return {
        "phases": phase_rows,
        "parts": part_rows,
        "windows": window_rows,
        "kernels": kernel_rows,
        "io_layers": io_rows,
        "unattributed_gpu_sum_ms": sum(e["end"] - e["start"] for e in unowned) / 1e6,
        "unattributed_gpu_events": len(unowned),
        "total_gpu_events": len(events),
    }


def render(output, results, summaries, paths):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"sparse_sync": "#65757e", "dense_prefetch": "#168d82"}
    names = {"sparse_sync": "DSA blocking", "dense_prefetch": "Dense window"}
    categories = [
        "ffn_dense",
        "ffn_moe",
        "indexer",
        "main_attention",
        "qkv_rope",
        "attention_output",
        "sparse_recall",
        "cache_layout",
        "kv_write",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for i, (result, summary) in enumerate(zip(results, summaries, strict=True)):
        mode = summary["mode"]
        records = [r for r in result["parts"] if r["phase"] == "candidate" and not r["first_visit"]]
        n = len({r["ordinal"] for r in records})
        values = [sum(r["gpu_sum_ms"] for r in records if r["part"] == c) / n for c in categories]
        axes[0].barh(
            [j + (i - 0.5) * 0.36 for j in range(len(categories))],
            values,
            height=0.36,
            color=colors[mode],
            label=names[mode],
        )
        controls = []
        for rep in ("control_before", "profile", "control_after"):
            rows = [
                r
                for r in summary["rows"]
                if r["repetition"] == rep
                and not r["first_visit"]
                and summary["capture_start"] <= r["ordinal"] < summary["capture_end"]
            ]
            controls.append(sum(r["service_ms"] for r in rows) / len(rows))
        axes[1].bar(
            [j + (i - 0.5) * 0.36 for j in range(3)],
            controls,
            width=0.36,
            color=colors[mode],
            label=names[mode],
        )
    axes[0].set(
        yticks=range(len(categories)),
        yticklabels=categories,
        xlabel="GPU kernel duration sum per revisit (ms)",
        title="Operator work, not wall-time shares",
    )
    axes[0].invert_yaxis()
    axes[1].set(
        xticks=range(3),
        xticklabels=["Before", "Nsight", "After"],
        ylabel="Request wall time (ms)",
        title="Same four revisits, separate replay passes",
    )
    axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.22)
    for ax in axes:
        ax.legend()
    fig.suptitle("Five layers | 64K + 1K | fixed budgets | four sampled revisits")
    fig.savefig(output / "breakdown.png", dpi=170)
    plt.close(fig)

    part_colors = {
        "ffn_dense": "#aa5b78",
        "ffn_moe": "#c87639",
        "indexer": "#6e83ac",
        "main_attention": "#168d82",
        "sparse_recall": "#d1a62b",
    }
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), layout="constrained", sharex=True)
    for ax, path, summary in zip(axes, paths, summaries, strict=True):
        ranges, events = read_trace(path.parents[1] / "profile" / path.name / "trace.sqlite")
        ordinal = next(
            r["ordinal"]
            for r in summary["rows"]
            if r["repetition"] == "profile"
            and not r["first_visit"]
            and r["ordinal"] >= summary["capture_start"]
        )
        scopes = [
            r for r in ranges if r["label"]["r"] == ordinal and r["label"]["p"] == "candidate"
        ]
        begin = next(r["start"] for r in scopes if r["label"]["s"] == "phase")
        for e in events:
            label = e["label"]
            if not label or label["r"] != ordinal or label["p"] != "candidate":
                continue
            if e["kind"] == "KERNEL":
                y, color = 2, part_colors.get(label["s"], "#a7adae")
            elif e["kind"] == "MEMCPY" and label["s"] == "dense_producer":
                y, color = 1, "#455c88"
            else:
                continue
            ax.broken_barh(
                [((e["start"] - begin) / 1e6, (e["end"] - e["start"]) / 1e6)],
                (y - 0.3, 0.6),
                facecolors=color,
            )
        for r in scopes:
            if r["label"]["s"] == "cpu_gather":
                ax.broken_barh(
                    [((r["start"] - begin) / 1e6, (r["end"] - r["start"]) / 1e6)],
                    (-0.3, 0.6),
                    facecolors="#168d82",
                )
        ax.set(
            yticks=[0, 1, 2],
            yticklabels=["CPU gather", "Prefix H2D", "GPU kernels"],
            title=f"{names[summary['mode']]} | request ordinal {ordinal} | candidate phase",
        )
        ax.grid(axis="x", alpha=0.2)
    axes[-1].set_xlabel("Time from candidate entry (ms)")
    fig.legend(
        handles=[Patch(color=c, label=p) for p, c in part_colors.items()],
        loc="outside upper center",
        ncol=5,
        fontsize=8,
    )
    fig.savefig(output / "timeline.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("refusing to overwrite report")
    summaries = [json.loads((p / "summary.json").read_text()) for p in args.runs]
    if {s["mode"] for s in summaries} != {"dense_prefetch", "sparse_sync"}:
        raise ValueError("requires Dense and DSA")
    for field in ("source_sha256", "trace_metadata", "capture_start", "capture_end"):
        if summaries[0][field] != summaries[1][field]:
            raise ValueError(f"paired {field} differs")
    outcomes = lambda s: [
        (r["repetition"], r["ordinal"], r["user_id"], r["first_visit"], r["prefix_reused"])
        for r in s["rows"]
    ]
    if outcomes(summaries[0]) != outcomes(summaries[1]):
        raise ValueError("paired cache outcomes differ")
    for field in ("checkpoint_metadata_sha256", "echo_revision", "echo_patch_sha256"):
        if summaries[0]["runner_provenance"][field] != summaries[1]["runner_provenance"][field]:
            raise ValueError(f"paired model provenance differs: {field}")
    results = []
    for path, summary in zip(args.runs, summaries, strict=True):
        database = path.parents[1] / "profile" / path.name / "trace.sqlite"
        results.append(analyze(database, summary))
    args.output_dir.mkdir(parents=True)
    for kind in ("phases", "parts", "windows", "kernels", "io_layers"):
        rows = [r for result in results for r in result[kind]]
        if rows:
            with (args.output_dir / f"{kind}.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    manifest = {
        "run_ids": [p.name for p in args.runs],
        "summary_sha256": {
            p.name: hashlib.sha256((p / "summary.json").read_bytes()).hexdigest() for p in args.runs
        },
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "coverage": [
            {k: v for k, v in result.items() if not isinstance(v, list)} for result in results
        ],
        "limits": "Diagnostic sample: 2 first visits, 4 revisits; profiler overhead; CPU/GPU durations are not additive; no-traced-GPU-activity is not a stall root-cause attribution.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    render(args.output_dir, results, summaries, args.runs)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
