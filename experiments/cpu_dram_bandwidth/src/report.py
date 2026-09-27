"""Plot recorded DRAM bandwidth summaries without running new measurements.

Example, from the repository root::

    python -m experiments.cpu_dram_bandwidth.src.report \
        --data-dir experiments/cpu_dram_bandwidth/output/data/RUN_ID \
        --output-dir experiments/cpu_dram_bandwidth/output/data/RUN_ID/analysis

Multiple data directories produce one three-panel row per run. Error bars show
the minimum and maximum recorded repetitions, with the median as the line.
Optional native case records supply a separate IMC hardware-counter crosscheck.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

NODE_SETS = ("0", "1", "0,1")
NODE_TITLES = ("Socket 0 / NUMA 0", "Socket 1 / NUMA 1", "Both sockets, simultaneous")
OPERATIONS = ("read", "nt-write", "nt-copy", "nt-triad")
COLORS = ("#0072B2", "#D55B00", "#009E73", "#CC79A7")
MARKERS = ("o", "s", "^", "D")
SPEC_GBPS_PER_SOCKET = 8 * 8 * 4400 / 1000
IMC_CROSSCHECK_FIELDS = (
    "run_id",
    "node",
    "op",
    "label",
    "threads_per_socket",
    "threads",
    "payload_median_GBps",
    "imc_read_median_GBps",
    "imc_write_median_GBps",
    "imc_total_median_GBps",
    "imc_min_running_fraction",
    "imc_capture_median_seconds",
    "payload_median_seconds",
    "case_local_throttled_time_delta_ns_including_setup",
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, rows, fieldnames=None):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_runs(data_dirs):
    runs = []
    seen = set()
    for directory in data_dirs:
        directory = directory.resolve()
        if directory in seen:
            raise ValueError(f"Duplicate data directory: {directory}")
        seen.add(directory)
        path = directory / "summary.json"
        raw = path.read_bytes()
        records = json.loads(raw)
        if not isinstance(records, list) or not records:
            raise ValueError(f"Expected a nonempty summary list: {path}")
        rows = []
        for record in records:
            if not record["validation_passed"]:
                raise ValueError(f"Invalid measurement in {path}: {record['label']}")
            nodes = ",".join(str(node) for node in sorted(map(int, record["nodes"].split(","))))
            if nodes not in NODE_SETS or record["op"] not in OPERATIONS:
                raise ValueError(f"Unsupported node set or operation in {path}: {record}")
            for key in ("median_GBps", "best_GBps", "min_GBps", "mean_GBps"):
                if not math.isfinite(record[key]) or record[key] <= 0:
                    raise ValueError(f"Invalid {key} in {path}: {record['label']}")
            if not record["min_GBps"] <= record["median_GBps"] <= record["best_GBps"]:
                raise ValueError(f"Inconsistent repetition statistics in {path}: {record['label']}")
            if record["threads"] != record["threads_per_node"] * len(nodes.split(",")):
                raise ValueError(f"Inconsistent thread counts in {path}: {record['label']}")
            rows.append({"run_id": directory.name, **record, "nodes": nodes})
        runs.append(
            {
                "run_id": directory.name,
                "summary_path": str(path),
                "summary_sha256": hashlib.sha256(raw).hexdigest(),
                "hardware_evidence_files": [
                    {
                        "path": str(evidence),
                        "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                    }
                    for name in ("clock_counters.json", "hardware_evidence.json")
                    if (evidence := directory / name).is_file()
                ],
                "rows": rows,
            }
        )
    return runs


def select_peaks(runs):
    selected = []
    for run in runs:
        for nodes in NODE_SETS:
            for operation in OPERATIONS:
                candidates = [
                    row for row in run["rows"] if row["nodes"] == nodes and row["op"] == operation
                ]
                if not candidates:
                    continue
                peak = max(candidates, key=lambda row: (row["median_GBps"], -row["threads"]))
                selected.append(
                    {
                        "run_id": run["run_id"],
                        "node": nodes,
                        "op": operation,
                        "best_median_GBps": peak["median_GBps"],
                        "threads_per_socket": peak["threads_per_node"],
                        "threads": peak["threads"],
                        "best_sample_GBps": peak["best_GBps"],
                        "min_sample_GBps": peak["min_GBps"],
                        "mean_GBps": peak["mean_GBps"],
                        "mib_per_thread": peak["mib_per_thread"],
                        "active_working_set_bytes": peak["active_working_set_bytes"],
                        "reps": peak["reps"],
                        "passes": peak["passes"],
                        "label": peak["label"],
                        "copy_one_direction_median_GBps": (
                            peak["median_GBps"] / 2 if operation == "nt-copy" else None
                        ),
                    }
                )
    return selected


def local_throttled_time_ns(text):
    for line in (text or "").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == "throttled_time":
            return int(fields[1])
    return None


def load_imc_crosscheck(runs):
    """Keep hardware-counter rates separate from the existing payload tables."""
    rows = []
    imc_fields = (
        "imc_read_GBps",
        "imc_write_GBps",
        "imc_total_GBps",
        "imc_min_running_fraction",
        "imc_capture_seconds",
    )
    for run in runs:
        cases = {}
        directory = Path(run["summary_path"]).parent
        for summary in run["rows"]:
            label = summary["label"]
            if label not in cases:
                path = directory / "cases" / f"{label}.json"
                cases[label] = json.loads(path.read_text()) if path.is_file() else None
            case = cases[label]
            if case is None:
                continue
            results = [
                result for result in case["measurement"]["results"] if result["op"] == summary["op"]
            ]
            if len(results) != 1:
                raise ValueError(
                    f"Expected one native result for {run['run_id']}/{label}/{summary['op']}"
                )
            result = results[0]
            if not any(field in result for field in imc_fields):
                continue
            if not result["validation"]["passed"]:
                raise ValueError(
                    f"Native validation failed: {run['run_id']}/{label}/{summary['op']}"
                )
            for field in (*imc_fields, "seconds", "payload_GBps"):
                values = result.get(field)
                if not isinstance(values, list) or len(values) != summary["reps"]:
                    raise ValueError(f"Missing or inconsistent {field}: {run['run_id']}/{label}")
                if any(not math.isfinite(value) or value < 0 for value in values):
                    raise ValueError(f"Invalid {field}: {run['run_id']}/{label}")
            if min(result["seconds"]) <= 0 or min(result["imc_capture_seconds"]) <= 0:
                raise ValueError(f"Nonpositive capture interval: {run['run_id']}/{label}")
            if (
                not 0
                < min(result["imc_min_running_fraction"])
                <= max(result["imc_min_running_fraction"])
                <= 1
            ):
                raise ValueError(f"Invalid counter running fraction: {run['run_id']}/{label}")
            payload_median = statistics.median(result["payload_GBps"])
            if not math.isclose(payload_median, summary["median_GBps"], rel_tol=1e-9):
                raise ValueError(f"Native/summary payload mismatch: {run['run_id']}/{label}")
            before = local_throttled_time_ns(case.get("cpu_local_stat_before"))
            after = local_throttled_time_ns(case.get("cpu_local_stat_after"))
            throttle_delta = after - before if before is not None and after is not None else None
            if throttle_delta is not None and throttle_delta < 0:
                raise ValueError(f"Local throttling counter reset: {run['run_id']}/{label}")
            rows.append(
                {
                    "run_id": run["run_id"],
                    "node": summary["nodes"],
                    "op": summary["op"],
                    "label": label,
                    "threads_per_socket": summary["threads_per_node"],
                    "threads": summary["threads"],
                    "payload_median_GBps": payload_median,
                    "imc_read_median_GBps": statistics.median(result["imc_read_GBps"]),
                    "imc_write_median_GBps": statistics.median(result["imc_write_GBps"]),
                    "imc_total_median_GBps": statistics.median(result["imc_total_GBps"]),
                    "imc_min_running_fraction": min(result["imc_min_running_fraction"]),
                    "imc_capture_median_seconds": statistics.median(result["imc_capture_seconds"]),
                    "payload_median_seconds": statistics.median(result["seconds"]),
                    "case_local_throttled_time_delta_ns_including_setup": throttle_delta,
                }
            )
    return rows


def plot_runs(runs, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "cpu-dram-bandwidth",
        }
    )
    fig, axes = plt.subplots(len(runs), 3, figsize=(14, 3.7 * len(runs) + 1.3), squeeze=False)
    for run_index, run in enumerate(runs):
        for node_index, nodes in enumerate(NODE_SETS):
            ax = axes[run_index, node_index]
            records = [row for row in run["rows"] if row["nodes"] == nodes]
            ceiling = SPEC_GBPS_PER_SOCKET * len(nodes.split(","))
            ax.axhline(ceiling, color="0.45", linestyle="--", linewidth=1)
            ax.text(
                0.97,
                ceiling,
                f"Specification ceiling: {ceiling:g} GB/s",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=9,
                color="0.35",
            )
            for operation, color, marker in zip(OPERATIONS, COLORS, MARKERS, strict=True):
                points = sorted(
                    (row for row in records if row["op"] == operation),
                    key=lambda row: row["threads_per_node"],
                )
                if not points:
                    continue
                medians = [row["median_GBps"] for row in points]
                ax.errorbar(
                    [row["threads_per_node"] for row in points],
                    medians,
                    yerr=[
                        [row["median_GBps"] - row["min_GBps"] for row in points],
                        [row["best_GBps"] - row["median_GBps"] for row in points],
                    ],
                    color=color,
                    marker=marker,
                    markersize=4.5,
                    linewidth=1.5,
                    elinewidth=0.8,
                    capsize=2,
                    label=operation,
                )
            if records:
                ticks = sorted({row["threads_per_node"] for row in records})
                ax.set_xticks(ticks)
                ax.set_xlim(max(0, min(ticks) - 4), max(ticks) + 4)
            else:
                ax.text(0.5, 0.5, "No measurements", transform=ax.transAxes, ha="center")
            highest = max([ceiling, *[row["best_GBps"] for row in records]])
            ax.set_ylim(0, highest * 1.14)
            ax.set_title(f"{NODE_TITLES[node_index]}\n{run['run_id']}", fontsize=10)
            ax.set_xlabel("Worker threads per socket")
            ax.set_ylabel("Payload bandwidth (GB/s, decimal)")
            ax.grid(axis="y", color="0.9", linewidth=0.7)
            ax.spines[["right", "top"]].set_visible(False)
    handles = [
        Line2D([], [], color=color, marker=marker, linewidth=1.5, label=operation)
        for operation, color, marker in zip(OPERATIONS, COLORS, MARKERS, strict=True)
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955)
    )
    fig.suptitle("CPU DRAM bandwidth: median and min–max across repetitions", y=0.997, fontsize=13)
    fig.text(
        0.5,
        0.012,
        "Dashed lines: 8 channels/socket × 8 bytes × 4400 MT/s; "
        "DDR5-4400 verified using DCLK counters (SMBIOS ConfiguredSpeed not read).\n"
        "Payload: read/write = 1 stream; NT copy = read + write; NT triad = 2 reads + 1 write. "
        "Dual-socket values are measured simultaneously.\n"
        "Error bars retain all recorded repetitions. Rates count application bytes, "
        "not memory-controller hardware counters.\n"
        "High-thread counts may be CPU-throttled in this container; "
        "their lower rates do not establish a lower DRAM hardware ceiling.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.16 / len(runs), 1, 0.90 + 0.035 * (len(runs) - 1) / len(runs)))
    fig.savefig(output / "bandwidth_scaling.svg", metadata={"Date": None})
    fig.savefig(output / "bandwidth_scaling.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        nargs="+",
        action="extend",
        required=True,
        help="One or more run directories containing summary.json; may be repeated",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination for SVG/CSV/JSON (default: first data directory / analysis)",
    )
    args = parser.parse_args()
    runs = load_runs(args.data_dir)
    output = args.output_dir or args.data_dir[0] / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    selected = select_peaks(runs)
    imc_crosscheck = load_imc_crosscheck(runs)
    all_rows = [row for run in runs for row in run["rows"]]
    write_csv(output / "selected_peaks.csv", selected)
    write_json(output / "selected_peaks.json", selected)
    write_csv(output / "all_measurements.csv", all_rows)
    write_json(output / "all_measurements.json", all_rows)
    write_csv(output / "imc_crosscheck.csv", imc_crosscheck, IMC_CROSSCHECK_FIELDS)
    write_json(output / "imc_crosscheck.json", imc_crosscheck)
    write_json(
        output / "provenance.json",
        {
            "sources": [
                {key: value for key, value in run.items() if key != "rows"} for run in runs
            ],
            "selection": "Maximum configuration median within each run/node/op; samples from that configuration",
            "bandwidth_unit": "decimal GB/s",
            "specification_GBps_per_socket": SPEC_GBPS_PER_SOCKET,
            "specification_assumption": "8 channels/socket, 8 data bytes/transfer, 4400 MT/s at 2DPC",
            "effective_memory_rate_MTps": 4400,
            "effective_memory_rate_counter_verified": True,
            "smbios_configured_speed_read": False,
            "clock_counter_evidence": {
                "files": [evidence for run in runs for evidence in run["hardware_evidence_files"]],
                "DCLK": "Intel EMR UNC_M_CLOCKTICKS event=0x01, umask=0x01: approximately 2.2 GHz",
                "HCLK": "Intel EMR UNC_M_HCLOCKTICKS event=0x01, umask=0x00: approximately 1.1 GHz",
                "coverage": "8 memory-controller PMUs per socket, both sockets",
                "inference": "DDR transfers twice per DCLK cycle: 2.2 GHz × 2 = 4400 MT/s",
            },
            "container_throttling": (
                "cpu.stat.local throttled_time increased during the initial sweep; "
                "ancestor cgroup limits are not visible in this mount. High-thread counts "
                "may be CPU-throttled and cannot establish a lower DRAM hardware ceiling."
            ),
            "error_bars": "Minimum to maximum recorded repetition within each configuration",
            "dual_socket": "One simultaneous native measurement; never sum independent socket measurements",
            "traffic": "Application payload bytes, not IMC hardware-counter traffic",
            "imc_crosscheck": {
                "source": "Optional cases/<label>.json native result arrays; cases without IMC are omitted",
                "aggregation": "Each rate and interval column is the median of its own repetitions",
                "running_fraction": "Minimum IMC running fraction across all recorded repetitions and counters",
                "capture_interval": "IMC rates use the outer counter-snapshot interval; payload uses the timed kernel interval",
                "local_throttling_delta": (
                    "Nanoseconds from case-level cpu.stat.local before/after; includes process setup, "
                    "initialization, warmup, all timed operations, validation, and teardown. "
                    "Repeated for each operation in the case: do not sum across operations. "
                    "Null means the counter was unavailable."
                ),
            },
        },
    )
    plot_runs(runs, output)
    print(
        f"Wrote bandwidth_scaling.svg, selected_peaks.*, all_measurements.*, "
        f"imc_crosscheck.*, provenance.json to {output}"
    )


if __name__ == "__main__":
    main()
