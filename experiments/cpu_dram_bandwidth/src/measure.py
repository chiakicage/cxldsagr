"""Metadata, one native measurement, and analysis for the DRAM bandwidth experiment."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def command_output(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        return {
            "argv": argv,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except OSError as exc:
        return {"argv": argv, "error": str(exc)}


def topology():
    rows = []
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
        nodes = list(base.glob("node[0-9]*"))
        rows.append(
            {
                "cpu": cpu,
                "node": int(nodes[0].name[4:]),
                "socket": int((base / "topology/physical_package_id").read_text()),
                "core": int((base / "topology/core_id").read_text()),
            }
        )
    return rows


def selected_cpus(nodes, threads_per_node):
    selected = []
    rows = topology()
    for node in nodes:
        siblings = {}
        for row in rows:
            if row["node"] == node:
                siblings.setdefault((row["socket"], row["core"]), []).append(row["cpu"])
        cores = list(siblings.values())
        ordered = [
            core[level]
            for level in range(max(map(len, cores), default=0))
            for core in cores
            if len(core) > level
        ]
        if not 1 <= threads_per_node <= len(ordered):
            raise ValueError(f"node {node}: requested {threads_per_node}, available {len(ordered)}")
        # Spread physical cores over the socket; add SMT siblings only above the core count.
        if threads_per_node <= len(cores):
            selected.extend(
                ordered[i * len(cores) // threads_per_node] for i in range(threads_per_node)
            )
        else:
            selected.extend(ordered[:threads_per_node])
    return selected


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def metadata(args):
    paths = [
        "/proc/meminfo",
        "/proc/self/status",
        "/proc/self/cgroup",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        "/sys/fs/cgroup/cpu/cpu.stat",
        "/sys/fs/cgroup/cpu/cpu.stat.local",
        "/proc/pressure/cpu",
        "/proc/self/mountinfo",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/kernel/mm/transparent_hugepage/enabled",
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
        "/sys/class/dmi/id/product_name",
        "/sys/class/dmi/id/board_name",
    ]
    paths += [
        str(path)
        for path in Path("/sys/devices/system/edac/mc").glob("mc*/dimm*/*")
        if path.name in {"dimm_label", "size", "dimm_mem_type", "dimm_dev_type"}
    ]
    commands = [
        ["lscpu"],
        ["numactl", "--hardware"],
        ["gcc", "--version"],
        ["uname", "-a"],
        ["ps", "-eo", "pid,comm,pcpu,pmem", "--sort=-pcpu"],
        ["git", "rev-parse", "HEAD"],
    ]
    save(
        args.output,
        {
            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
            "topology": topology(),
            "files": {p: read_optional(p) for p in paths},
            "commands": [command_output(cmd) for cmd in commands],
        },
    )


def run(args):
    nodes = [int(node) for node in args.nodes.split(",")]
    cpus = selected_cpus(nodes, args.threads_per_node)
    argv = [
        "numactl",
        "--membind=" + args.nodes,
        str(Path(args.binary).resolve()),
        "--cpus",
        ",".join(map(str, cpus)),
        "--mib-per-thread",
        str(args.mib_per_thread),
        "--warmup",
        str(args.warmup),
        "--reps",
        str(args.reps),
        "--passes",
        str(args.passes),
        "--ops",
        args.ops,
    ]
    if args.imc:
        argv.append("--imc")
    start = dt.datetime.now(dt.UTC).isoformat()
    cpu_before = read_optional("/sys/fs/cgroup/cpu/cpu.stat")
    local_before = read_optional("/sys/fs/cgroup/cpu/cpu.stat.local")
    pressure_before = read_optional("/proc/pressure/cpu")
    load_before = read_optional("/proc/loadavg")
    elapsed_start = time.perf_counter()
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - elapsed_start
    sys.stderr.write(result.stderr)
    if result.returncode:
        sys.stdout.write(result.stdout)
        raise SystemExit(result.returncode)
    native = json.loads(result.stdout)
    record = {
        "label": args.label,
        "timestamp_utc": start,
        "nodes": nodes,
        "cpus": cpus,
        "threads_per_node": args.threads_per_node,
        "command": argv,
        "binary_sha256": hashlib.sha256(Path(args.binary).read_bytes()).hexdigest(),
        "load_before": load_before,
        "load_after": read_optional("/proc/loadavg"),
        "cpu_stat_before": cpu_before,
        "cpu_stat_after": read_optional("/sys/fs/cgroup/cpu/cpu.stat"),
        "cpu_local_stat_before": local_before,
        "cpu_local_stat_after": read_optional("/sys/fs/cgroup/cpu/cpu.stat.local"),
        "cpu_pressure_before": pressure_before,
        "cpu_pressure_after": read_optional("/proc/pressure/cpu"),
        "process_elapsed_seconds_including_setup": elapsed,
        "measurement": native,
    }
    save(args.output, record)
    print(json.dumps(record), flush=True)


def summarize(args):
    rows = []
    for path in sorted(Path(args.data_dir).glob("cases/*.json")):
        case = json.loads(path.read_text())
        native = case["measurement"]
        for result in native["results"]:
            rates = result["payload_GBps"]
            row = {
                "label": case["label"],
                "nodes": ",".join(map(str, case["nodes"])),
                "threads_per_node": case["threads_per_node"],
                "threads": len(case["cpus"]),
                "mib_per_thread": native["mib_per_thread"],
                "op": result["op"],
                "median_GBps": statistics.median(rates),
                "best_GBps": max(rates),
                "min_GBps": min(rates),
                "mean_GBps": statistics.mean(rates),
                "active_working_set_bytes": result["active_working_set_bytes"],
                "reps": native["reps"],
                "passes": native["passes"],
                "validation_passed": result["validation"]["passed"],
            }
            if not row["validation_passed"]:
                raise RuntimeError(f"Validation failed: {path}, {result['op']}")
            rows.append(row)
    if not rows:
        raise ValueError("No measurement cases found")
    save(Path(args.data_dir) / "summary.json", rows)
    with (Path(args.data_dir) / "summary.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['label']:30s} {row['op']:10s} median {row['median_GBps']:8.2f} GB/s "
            f"best {row['best_GBps']:8.2f} GB/s"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    meta = sub.add_parser("metadata", help="Save hardware and process environment")
    meta.add_argument("--output", required=True)
    meta.set_defaults(function=metadata)
    bench = sub.add_parser("run", help="Measure one pinned NUMA configuration")
    bench.add_argument("--binary", required=True)
    bench.add_argument("--label", required=True)
    bench.add_argument("--nodes", required=True)
    bench.add_argument("--threads-per-node", type=int, required=True)
    bench.add_argument("--mib-per-thread", type=int, default=128)
    bench.add_argument("--warmup", type=int, default=2)
    bench.add_argument("--reps", type=int, default=7)
    bench.add_argument("--passes", type=int, default=4)
    bench.add_argument("--ops", default="read,nt-write,nt-copy,nt-triad")
    bench.add_argument("--imc", action="store_true", help="Require Intel IMC hardware counters")
    bench.add_argument("--output", required=True)
    bench.set_defaults(function=run)
    summary = sub.add_parser("summarize", help="Create CSV/JSON from successful measurement cases")
    summary.add_argument("--data-dir", required=True)
    summary.set_defaults(function=summarize)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
