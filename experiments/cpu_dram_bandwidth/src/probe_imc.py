"""Read Emerald Rapids IMC clock counters without generating a memory workload."""

import argparse
import ctypes
import datetime as dt
import fcntl
import json
import os
import platform
import statistics
import struct
import time
from pathlib import Path

EVENT_SOURCE = (
    "https://raw.githubusercontent.com/intel/perfmon/main/EMR/events/emeraldrapids_uncore.json"
)
EVENTS = {
    "UNC_M_CLOCKTICKS": {
        "config": 0x0101,
        "event": "0x01",
        "umask": "0x01",
        "description": "Number of DRAM DCLK clock cycles while the event is enabled",
    },
    "UNC_M_HCLOCKTICKS": {
        "config": 0x0001,
        "event": "0x01",
        "umask": "0x00",
        "description": "Number of DRAM HCLK clock cycles while the event is enabled",
    },
}


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def expand_cpu_list(value):
    cpus = []
    for part in value.strip().split(","):
        ends = part.split("-")
        cpus.extend(range(int(ends[0]), int(ends[-1]) + 1))
    return cpus


def hardware_evidence():
    controllers = []
    for path in sorted(Path("/sys/devices/system/edac/mc").glob("mc[0-9]*")):
        controllers.append(
            {
                "path": str(path),
                "name": read_optional(path / "mc_name"),
                "size_mib": read_optional(path / "size_mb"),
                "dimms": [
                    {
                        "path": str(dimm),
                        **{
                            key: read_optional(dimm / key)
                            for key in (
                                "dimm_label",
                                "dimm_location",
                                "size",
                                "dimm_mem_type",
                                "dimm_dev_type",
                                "dimm_edac_mode",
                            )
                        },
                    }
                    for dimm in sorted(path.glob("dimm*"))
                ],
            }
        )
    return {
        "product_name": read_optional("/sys/class/dmi/id/product_name"),
        "board_name": read_optional("/sys/class/dmi/id/board_name"),
        "edac_controllers": controllers,
        "cpu_cfs_quota_us": read_optional("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
        "cpu_cfs_period_us": read_optional("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
        "cpu_stat": read_optional("/sys/fs/cgroup/cpu/cpu.stat"),
        "cpu_stat_local": read_optional("/sys/fs/cgroup/cpu/cpu.stat.local"),
        "cgroup_mounts": [
            line
            for line in Path("/proc/self/mountinfo").read_text().splitlines()
            if " - cgroup " in line
        ],
    }


def probe(duration):
    if platform.machine() != "x86_64":
        raise RuntimeError("This probe uses the x86_64 perf_event_open syscall number")
    cpuinfo = dict(
        line.split(":", 1)
        for line in Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0].splitlines()
        if ":" in line
    )
    cpuinfo = {key.strip(): value.strip() for key, value in cpuinfo.items()}
    if (cpuinfo.get("vendor_id"), cpuinfo.get("cpu family"), cpuinfo.get("model")) != (
        "GenuineIntel",
        "6",
        "207",
    ):
        raise RuntimeError("Event definitions in this probe are for Emerald Rapids model 207")
    devices = sorted(
        Path("/sys/bus/event_source/devices").glob("uncore_imc_[0-9]*"),
        key=lambda path: int(path.name.rsplit("_", 1)[1]),
    )
    if not devices:
        raise RuntimeError("No uncore IMC performance monitoring units are exposed")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    handles = []
    samples = []
    try:
        for device in devices:
            pmu_type = int((device / "type").read_text())
            for cpu in expand_cpu_list((device / "cpumask").read_text()):
                socket = int(
                    Path(
                        f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id"
                    ).read_text()
                )
                for name, event in EVENTS.items():
                    attr = ctypes.create_string_buffer(128)
                    # read_format: TOTAL_TIME_ENABLED | TOTAL_TIME_RUNNING; initially disabled.
                    struct.pack_into("IIQQQQQ", attr, 0, pmu_type, 128, event["config"], 0, 0, 3, 1)
                    fd = libc.syscall(298, ctypes.byref(attr), -1, cpu, -1, 0)
                    if fd < 0:
                        error = ctypes.get_errno()
                        raise OSError(error, f"{device.name} CPU {cpu}: {os.strerror(error)}")
                    handles.append((fd, device.name, cpu, socket, name))
        for fd, *_ in handles:
            fcntl.ioctl(fd, 0x2403, 0)  # PERF_EVENT_IOC_RESET
            fcntl.ioctl(fd, 0x2400, 0)  # PERF_EVENT_IOC_ENABLE
        time.sleep(duration)
        for fd, device, cpu, socket, name in handles:
            fcntl.ioctl(fd, 0x2401, 0)  # PERF_EVENT_IOC_DISABLE
            count, enabled, running = struct.unpack("QQQ", os.read(fd, 24))
            if not running:
                raise RuntimeError(f"{device} {name} on CPU {cpu} did not run")
            samples.append(
                {
                    "pmu": device,
                    "cpu": cpu,
                    "socket": socket,
                    "event_name": name,
                    "count": count,
                    "time_enabled_ns": enabled,
                    "time_running_ns": running,
                    "running_fraction": running / enabled,
                    "rate_hz": count * 1e9 / running,
                }
            )
    finally:
        for fd, *_ in handles:
            os.close(fd)
    summaries = []
    for socket in sorted({row["socket"] for row in samples}):
        rates = [
            row["rate_hz"]
            for row in samples
            if row["socket"] == socket and row["event_name"] == "UNC_M_CLOCKTICKS"
        ]
        summaries.append(
            {
                "socket": socket,
                "dclk_hz_mean": statistics.mean(rates),
                "dclk_hz_min": min(rates),
                "dclk_hz_max": max(rates),
                "ddr_data_rate_mtps_mean": 2e-6 * statistics.mean(rates),
            }
        )
    return {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "cpu_model": cpuinfo["model name"],
        "requested_duration_s": duration,
        "event_source": EVENT_SOURCE,
        "event_definitions": EVENTS,
        "interpretation": (
            "DDR transfer rate = 2 * measured DCLK frequency. The sysfs clockticks alias "
            "has umask 0x00 (HCLK), so using it as DCLK underestimates data rate by 2x. "
            "Counters are socket-wide; this probe does not generate a memory workload."
        ),
        "samples": samples,
        "sockets": summaries,
        "hardware": hardware_evidence(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON result path")
    parser.add_argument(
        "--duration", type=float, default=0.2, help="Sampling seconds (default: 0.2)"
    )
    args = parser.parse_args()
    if not 0.05 <= args.duration <= 60:
        parser.error("--duration must be between 0.05 and 60 seconds")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    result = probe(args.duration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result["sockets"]))


if __name__ == "__main__":
    main()
