"""Collect one baseline and one candidate preparation call with bounded NCU filters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/deepseek_v32_echo_official"
NCU = "/opt/nvidia/nsight-compute/2026.1.1/ncu"
METRICS = (
    "gpu__time_duration.sum",
    "sm__cycles_active.sum",
    "sm__cycles_active.max",
    "sm__ctas_launched.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError("Invalid run ID")
    receipt = args.receipt.resolve(strict=True)
    folders = {
        kind: EXPERIMENT / "output" / kind / args.run_id for kind in ("data", "log", "profile")
    }
    if any(path.exists() for path in folders.values()):
        raise FileExistsError("Use a new run ID")
    for path in folders.values():
        path.mkdir(parents=True)
    overrides = {
        "CUDA_VISIBLE_DEVICES": "1",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": str(ROOT / ".venv/bin") + os.pathsep + os.environ["PATH"],
    }
    environment = {**os.environ, **overrides}
    application_env = ["/usr/bin/env"]
    if "LD_LIBRARY_PATH" in os.environ:
        application_env.append("LD_LIBRARY_PATH=" + os.environ["LD_LIBRARY_PATH"])
    else:
        application_env.extend(["-u", "LD_LIBRARY_PATH"])
    for variant in ("baseline", "candidate"):
        if variant == "baseline":
            pattern = (
                "regex:pack_page64|official_prefetch::prepare_kernel|at::native::arange_cuda_out"
            )
            # The first matching arange builds query bounds. The next three
            # matches are exactly packing, block-table arange and stage/token prep.
            skip, count = 1, 3
        else:
            pattern, skip, count = "regex:q1_fused_prepare::prepare_kernel", 0, 1
        command = [
            NCU,
            "--target-processes",
            "all",
            "--profile-from-start",
            "off",
            "--replay-mode",
            "kernel",
            "--cache-control",
            "all",
            "--clock-control",
            "base",
            "--section",
            "LaunchStats",
            "--section",
            "SourceCounters",
            "--metrics",
            ",".join(METRICS),
            "--import-source",
            "yes",
            "--nvtx",
            "--nvtx-include",
            "q1_fused_prepare_complete_" + variant + "/",
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            pattern,
            "--launch-skip",
            str(skip),
            "--launch-count",
            str(count),
            "--export",
            str(folders["profile"] / variant),
            "taskset",
            "-c",
            "8-15",
            *application_env,
            ".venv/bin/python",
            "-B",
            "-m",
            "experiments.deepseek_v32_echo_official.src.q1_fused_prepare_pinned_run",
            "--mode",
            "profile",
            "--physical-device",
            "1",
            "--warmups",
            "20",
            "--variant",
            variant,
            "--policy",
            "zero",
            "--receipt",
            str(receipt),
            "--output-dir",
            str(folders["data"] / variant),
        ]
        invocation = {
            "argv": command,
            "cwd": str(ROOT),
            "environment_overrides": overrides,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "expected_actions": count,
            "skipped_matching_bounds_arange": skip,
            "boundary": "One real L0 complete call under the accepted component identity. Only selected preparation kernels are profiled. Kernel replay with cache flush/base clocks is not clean API timing; unsupported source or sampling fields remain unavailable.",
        }
        (folders["data"] / (variant + "_invocation.json")).write_text(
            json.dumps(invocation, indent=2) + "\n"
        )
        with (
            (folders["log"] / (variant + ".stdout.log")).open("x") as stdout,
            (folders["log"] / (variant + ".stderr.log")).open("x") as stderr,
        ):
            subprocess.run(
                command, cwd=ROOT, env=environment, check=True, stdout=stdout, stderr=stderr
            )
    print(
        json.dumps(
            {
                "completed": True,
                "run_id": args.run_id,
                "baseline_actions": 3,
                "candidate_actions": 1,
            }
        )
    )


if __name__ == "__main__":
    main()
