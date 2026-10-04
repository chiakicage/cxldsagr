"""Read process CPU, thread, and NUMA policy without changing them."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path


def numa_policy():
    library = ctypes.util.find_library("numa")
    if library is None:
        return {"available": False, "reason": "libnuma not found"}
    try:
        lib = ctypes.CDLL(library, use_errno=True)
        get = lib.get_mempolicy
        get.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        get.restype = ctypes.c_int
        max_nodes = 1024
        word_bits = ctypes.sizeof(ctypes.c_ulong) * 8
        mask = (ctypes.c_ulong * (max_nodes // word_bits))()
        mode = ctypes.c_int()
        if get(ctypes.byref(mode), mask, max_nodes, None, 0) != 0:
            error = ctypes.get_errno()
            return {"available": False, "errno": error, "reason": os.strerror(error)}
        names = {
            0: "default",
            1: "preferred",
            2: "bind",
            3: "interleave",
            4: "local",
            5: "preferred_many",
            6: "weighted_interleave",
        }
        return {
            "available": True,
            "mode_raw": mode.value,
            "mode": names.get(mode.value & 0x7FFF, "unknown"),
            "nodes": [n for n in range(max_nodes) if mask[n // word_bits] & (1 << (n % word_bits))],
            "scope": "calling-thread default memory policy; not a per-allocation placement proof",
        }
    except (AttributeError, OSError) as error:
        return {"available": False, "reason": str(error)}


def cpu_environment(torch):
    status = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("Cpus_allowed_list", "Mems_allowed_list"):
            status[key] = value.strip()
    return {
        "schema": "nosa-cpu-environment-v1",
        "affinity": sorted(os.sched_getaffinity(0)),
        "allowed": status,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "numa_policy": numa_policy(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "OMP_PROC_BIND",
                "OMP_PLACES",
                "KMP_AFFINITY",
                "GOMP_CPU_AFFINITY",
                "PYTORCH_CUDA_ALLOC_CONF",
                "PYTORCH_ALLOC_CONF",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }


def require_matching_cpu_environment(actual, reference):
    """Legacy pairs remain identifiable; new-versus-legacy cannot claim a match."""
    current, previous = actual.get("cpu_environment"), reference.get("cpu_environment")
    if current is None and previous is None:
        return {"status": "not_recorded_in_legacy_pair"}
    if current is None or previous is None or current != previous:
        raise ValueError(
            "CPU affinity, thread configuration, or NUMA policy differs or is unrecorded"
        )
    return {"status": "recorded_and_equal"}


def finish_cpu_environment(metadata, torch):
    """Record completion evidence and reject a changed execution environment."""
    final = cpu_environment(torch)
    metadata["cpu_environment_final"] = final
    try:
        audit = require_matching_cpu_environment({"cpu_environment": final}, metadata["hardware"])
    except ValueError as error:
        metadata["cpu_environment_audit"] = {"status": "rejected", "reason": str(error)}
        raise
    metadata["cpu_environment_audit"] = audit
    return audit


def validate_cpu_environment_record(metadata):
    """Reopen saved start/end evidence; never invent fields for legacy runs."""
    initial = metadata.get("hardware", {}).get("cpu_environment")
    final = metadata.get("cpu_environment_final")
    if initial is None and final is None:
        if "cpu_environment_audit" in metadata:
            raise ValueError("legacy CPU environment cannot claim a recorded audit")
        return {"status": "not_recorded_in_legacy_pair"}
    audit = require_matching_cpu_environment(
        {"cpu_environment": final}, {"cpu_environment": initial}
    )
    if metadata.get("cpu_environment_audit") != audit:
        raise ValueError("saved CPU environment audit differs from start/end evidence")
    return audit
