"""Actual Nsight Compute DRAM traffic; independent of clean API MFU timing."""

from __future__ import annotations

import math
import re

METRICS = (
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "gpu__time_duration.sum",
    "dram__bytes.sum.peak_sustained_elapsed.per_second",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
)
HBM_FIELDS = (
    "hbm_read_bytes_per_call",
    "hbm_write_bytes_per_call",
    "hbm_bytes_per_call",
    "hbm_bandwidth_gbps",
    "hbm_util_pct",
    "hbm_peak_gbps",
    "hbm_profile_time_us",
    "hbm_profile_inner",
    "hbm_metric_name",
    "hbm_cache_control",
    "hbm_time_boundary",
    "hbm_profile_path",
    "hbm_profile_sha256",
    "hbm_replay_mode",
    "hbm_ncu_throughput_pct",
    "hbm_profiler_passes",
    "hbm_metrics",
)


def _number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid HBM metric {name}")
    if positive and value == 0:
        raise ValueError(f"HBM metric {name} must be positive")
    return value


def from_metrics(metrics, *, inner, profile_path, profile_sha256, replay_mode, time_boundary):
    """Normalize a whole profiled graph/range; never average kernel percentages."""
    if type(inner) is not int or inner < 1:
        raise ValueError("HBM profile requires a positive call count")
    for name in METRICS:
        if name not in metrics:
            raise ValueError(f"Missing actual DRAM metric {name}")
    read = _number(metrics[METRICS[0]], METRICS[0])
    write = _number(metrics[METRICS[1]], METRICS[1])
    duration_ns = _number(metrics[METRICS[2]], METRICS[2], positive=True)
    peak_bps = _number(metrics[METRICS[3]], METRICS[3], positive=True)
    ncu_pct = _number(metrics[METRICS[4]], METRICS[4])
    passes = metrics.get("profiler__replayer_passes")
    if passes != 1:
        raise ValueError("This warm-cache traffic contract requires one profiling pass")
    bw_gbps = (read + write) / duration_ns
    result = {
        "hbm_read_bytes_per_call": read / inner,
        "hbm_write_bytes_per_call": write / inner,
        "hbm_bytes_per_call": (read + write) / inner,
        "hbm_bandwidth_gbps": bw_gbps,
        "hbm_util_pct": bw_gbps / (peak_bps / 1e9) * 100,
        "hbm_peak_gbps": peak_bps / 1e9,
        "hbm_profile_time_us": duration_ns / 1000 / inner,
        "hbm_profile_inner": inner,
        "hbm_metric_name": "dram__bytes_read.sum + dram__bytes_write.sum",
        "hbm_cache_control": "none",
        "hbm_time_boundary": time_boundary,
        "hbm_profile_path": str(profile_path),
        "hbm_profile_sha256": profile_sha256,
        "hbm_replay_mode": replay_mode,
        "hbm_ncu_throughput_pct": ncu_pct,
        "hbm_profiler_passes": passes,
        "hbm_metrics": metrics,
    }
    validate_bandwidth(result)
    return result


def validate_bandwidth(operator):
    missing = set(HBM_FIELDS) - operator.keys()
    if missing:
        raise ValueError(f"Incomplete actual HBM observation: {sorted(missing)}")
    for name in HBM_FIELDS[:7]:
        _number(operator[name], name, positive=name in {"hbm_peak_gbps", "hbm_profile_time_us"})
    if type(operator["hbm_profile_inner"]) is not int or operator["hbm_profile_inner"] < 1:
        raise ValueError("Invalid HBM profile call count")
    if operator["hbm_cache_control"] != "none" or operator["hbm_profiler_passes"] != 1:
        raise ValueError("HBM traffic requires warm caches and one profiling pass")
    for name in (
        "hbm_metric_name",
        "hbm_time_boundary",
        "hbm_profile_path",
        "hbm_profile_sha256",
        "hbm_replay_mode",
    ):
        if not isinstance(operator[name], str) or not operator[name]:
            raise ValueError(f"Missing HBM provenance {name}")
    if not re.fullmatch(r"[a-f0-9]{64}", operator["hbm_profile_sha256"]):
        raise ValueError("Invalid HBM profile content SHA-256")
    if operator["hbm_metric_name"] != "dram__bytes_read.sum + dram__bytes_write.sum":
        raise ValueError("Actual HBM metric must count physical DRAM reads and writes")
    if operator["hbm_replay_mode"] != "kernel-graph":
        raise ValueError("Actual HBM metric requires profiling a complete CUDA Graph")
    metrics = operator["hbm_metrics"]
    if not isinstance(metrics, dict) or not set(METRICS) <= metrics.keys():
        raise ValueError("Missing raw HBM hardware metrics")
    for name in METRICS:
        _number(metrics[name], name, positive=name in METRICS[2:4])
    if metrics.get("profiler__replayer_passes") != 1:
        raise ValueError("Raw HBM hardware metrics require one profiling pass")
    _number(operator["hbm_ncu_throughput_pct"], "hbm_ncu_throughput_pct")
    read, write = operator["hbm_read_bytes_per_call"], operator["hbm_write_bytes_per_call"]
    total = operator["hbm_bytes_per_call"]
    bw = operator["hbm_bandwidth_gbps"]
    expected = {
        "hbm_bytes_per_call": read + write,
        "hbm_bandwidth_gbps": total / (operator["hbm_profile_time_us"] * 1000),
        "hbm_util_pct": bw / operator["hbm_peak_gbps"] * 100,
        "hbm_read_bytes_per_call": metrics[METRICS[0]] / operator["hbm_profile_inner"],
        "hbm_write_bytes_per_call": metrics[METRICS[1]] / operator["hbm_profile_inner"],
        "hbm_profile_time_us": metrics[METRICS[2]] / 1000 / operator["hbm_profile_inner"],
        "hbm_peak_gbps": metrics[METRICS[3]] / 1e9,
        "hbm_ncu_throughput_pct": metrics[METRICS[4]],
    }
    for name, value in expected.items():
        if not math.isclose(operator[name], value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"HBM arithmetic differs for {name}")
    return {name: operator[name] for name in HBM_FIELDS}
