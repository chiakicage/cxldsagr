"""Run one receipt-bound paired preparation profile and export all three ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/deepseek_v32_echo_official"


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--cpus", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError("Invalid run ID")
    receipt = args.receipt.resolve(strict=True)
    request = args.request.resolve(strict=True)
    accepted = json.loads(receipt.read_text())["identity"]
    environments = [
        row["build_identity"]["environment"]
        for row in accepted["timing_runtime"]["private_flashinfer_native"].values()
    ]
    if len(environments) != 3 or any(row != environments[0] for row in environments):
        raise ValueError("Inconsistent accepted FlashInfer environment")
    checked = environments[0]
    folders = {
        kind: EXPERIMENT / "output" / kind / args.run_id for kind in ("data", "log", "profile")
    }
    if any(path.exists() for path in folders.values()):
        raise FileExistsError("Use a fresh run ID in every output category")
    folders["log"].mkdir(parents=True)
    folders["profile"].mkdir(parents=True)

    # Restore only the build variables covered by the accepted receipt after
    # profiler injection; keep NSYS's separate instrumentation variables intact.
    application_env = ["/usr/bin/env"]
    for name, value in sorted(checked.items()):
        if value is None:
            application_env.extend(["-u", name])
    application_env.extend(
        f"{name}={value}" for name, value in sorted(checked.items()) if value is not None
    )
    overrides = {
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "PATH": checked["PATH"],
    }
    environment = {**os.environ, **overrides}
    command = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--cuda-graph-trace=node",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=repeat",
        "--output",
        str(folders["profile"] / "capture"),
        "taskset",
        "-c",
        args.cpus,
        *application_env,
        ".venv/bin/python",
        "-B",
        "-m",
        "experiments.deepseek_v32_echo_official.src.q1_fused_prepare_model_pinned_run",
        "profile",
        "--request",
        str(request),
        "--receipt",
        str(receipt),
        "--output-dir",
        str(folders["data"]),
    ]
    invocation = {
        "argv": command,
        "cwd": str(ROOT),
        "environment_overrides": overrides,
        "accepted_build_environment": checked,
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "note": "One independent process: baseline setup, candidate setup, then one AB replay pair. Profile durations are not clean timing samples.",
    }
    write(folders["log"] / "invocation.json", invocation)
    with (folders["log"] / "nsys_version.txt").open("x") as stream:
        subprocess.run(["nsys", "--version"], cwd=ROOT, env=environment, check=True, stdout=stream)
    with (
        (folders["log"] / "stdout.log").open("x") as stdout,
        (folders["log"] / "stderr.log").open("x") as stderr,
    ):
        subprocess.run(command, cwd=ROOT, env=environment, check=True, stdout=stdout, stderr=stderr)
    shutil.copyfile(folders["log"] / "invocation.json", folders["data"] / "invocation.json")
    reports = sorted(folders["profile"].glob("capture.*.nsys-rep"))
    expected = [folders["profile"] / f"capture.{index}.nsys-rep" for index in (1, 2, 3)]
    if reports != expected:
        raise RuntimeError("Expected exactly two setup ranges and one paired replay range")
    for index, report in enumerate(reports, 1):
        with (
            (folders["log"] / f"export.{index}.stdout.log").open("x") as stdout,
            (folders["log"] / f"export.{index}.stderr.log").open("x") as stderr,
        ):
            subprocess.run(
                [
                    "nsys",
                    "export",
                    "--type",
                    "sqlite",
                    "--output",
                    str(folders["data"] / f"capture.{index}.sqlite"),
                    str(report),
                ],
                cwd=ROOT,
                env=environment,
                check=True,
                stdout=stdout,
                stderr=stderr,
            )
    print(json.dumps({"completed": True, "run_id": args.run_id, "ranges": 3}))


if __name__ == "__main__":
    main()
