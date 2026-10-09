"""Own independent normal-decode checks, request timing and NSYS capture."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.deepseek_v32_echo_official.scripts.run_engine import (
    ECHO_ROOT,
    REPRO_ROOT,
    ROOT,
    assert_gpu_idle,
    audit_capacity,
    capture_runtime_manifest,
    environment_identity,
    file_sha256,
    load_numerical_diagnostic,
    run_owned,
    scoped_runtime_environment,
    server_environment,
    source_identity,
    validate_environment,
    write_json,
)
from experiments.deepseek_v32_echo_official.src.decode_run import PROTOCOL, engine_arguments
from experiments.deepseek_v32_echo_official.src.engine_run import load_workload, token_hash


def sources():
    result = source_identity()
    for name in [
        "src/decode_run.py",
        "src/decode_profile_hooks.py",
        "src/decode_graph_inspector.py",
        "scripts/run_decode.py",
        "scripts/run_decode.sh",
    ]:
        path = ROOT / name
        result[str(path.relative_to(REPO_ROOT))] = file_sha256(path)
    path = REPO_ROOT / "experiments/deepseek_v32_motivation/src/graph_instrumentation.py"
    result[str(path.relative_to(REPO_ROOT))] = file_sha256(path)
    return result


def compatibility(identity):
    core = json.loads(json.dumps(identity))
    required = (
        "src/decode_run.py",
        "src/engine_run.py",
        "src/preflight.py",
        "src/capacity.py",
        "src/workload_client.py",
    )
    core["sources"] = {
        name: digest for name, digest in core["sources"].items() if name.endswith(required)
    }
    return core


def verify_worker(path, mode):
    result = json.loads((path / "complete.json").read_text())
    if (
        result["status"] != "completed"
        or result["mode"] != mode
        or result["protocol"] != PROTOCOL
        or result["row"]["completion_tokens"] != 2
        or result["row"]["cached_tokens"] != 0
        or file_sha256(path / "warmup.json") != result["warmup_sha256"]
    ):
        raise ValueError("Incomplete normal-decode worker")
    if mode == "check":
        artifact = result["row"]["output_artifact"]
        if not artifact["finite"] or file_sha256(Path(artifact["path"])) != artifact["sha256"]:
            raise ValueError("Independent output check changed")
    return result


def main():
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case", choices=("echo", "resident_reference"), required=True)
    parser.add_argument("--mode", choices=("check", "bench", "profile"), required=True)
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--check-run", type=Path)
    parser.add_argument("--performance-only", action="store_true", required=True)
    parser.add_argument("--numerical-diagnostic", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=REPRO_ROOT / "env/.venv/bin/python")
    parser.add_argument(
        "--model", type=Path, default=REPRO_ROOT / "weights/deepseek_v32_first3_fp8"
    )
    parser.add_argument("--workload", type=Path, default=REPRO_ROOT / "inputs/fixed_history")
    parser.add_argument("--build-receipt", type=Path, default=REPRO_ROOT / "env/build_receipt.json")
    parser.add_argument("--nsys", type=Path, default=Path(shutil.which("nsys") or "nsys"))
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--shutdown-timeout", type=float, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id) or args.repeats < 1:
        parser.error("Use a new safe run ID and positive repeats")
    if (args.mode == "check") != (args.check_run is None):
        parser.error("bench/profile require --check-run; check forbids it")
    for name in ("python", "model", "workload", "build_receipt", "numerical_diagnostic", "nsys"):
        setattr(args, name, getattr(args, name).absolute())
    diagnostic = load_numerical_diagnostic(args.numerical_diagnostic)
    workload_manifest, requests = load_workload(args.workload)
    prompt_sha256 = token_hash(requests[0]["input_ids"][:65536])
    validate_environment(dict(os.environ))
    environment = server_environment(
        scoped_runtime_environment(dict(os.environ)), args.case, args.gpu
    )
    if any(name.startswith("ECHO_ENGINE_PROFILE_") for name in environment):
        raise ValueError("Profile environment must be configured only by the profile worker")
    data = ROOT / "output/data" / args.run_id
    logs = ROOT / "output/log" / args.run_id
    profiles = ROOT / "output/profile" / args.run_id
    if any(path.exists() for path in (data, logs, profiles)):
        raise FileExistsError(args.run_id)
    assert_gpu_idle(args.gpu)
    data.mkdir(parents=True)
    logs.mkdir(parents=True)
    source_map = sources()
    record = {
        "schema": "echo-normal-decode-run-v1",
        "run_id": args.run_id,
        "mode": args.mode,
        "case": args.case,
        "status": "running",
        "protocol": PROTOCOL,
        "workers": [],
        "numerical_acceptance": False,
        "numerical_diagnostic": diagnostic,
        "sources": source_map,
        "workload_sha256": workload_manifest["workload_sha256"],
        "prompt_sha256": prompt_sha256,
        "started_ns": time.time_ns(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    write_json(data / "run.json", record)
    try:
        preflight_command = [
            str(args.python),
            "-B",
            "-m",
            "src.preflight",
            "--echo-root",
            str(ECHO_ROOT),
            "--model",
            str(args.model),
            "--model-provenance",
            str(args.model / "export_manifest.json"),
            "--expected-python",
            str(args.python),
            "--gpu-count",
            "1",
            "--build-receipt",
            str(args.build_receipt),
            "--output",
            str(data / "preflight.json"),
        ]
        run_owned(
            args,
            preflight_command,
            data=data / "preflight",
            logs=logs / "preflight",
            environment=environment,
            timeout=600,
        )
        preflight = json.loads((data / "preflight.json").read_text())
        if not preflight["runtime_validated"]:
            raise ValueError("Official runtime preflight did not pass")
        identity = {
            "case": args.case,
            "protocol": PROTOCOL,
            "preflight": preflight,
            "sources": source_map,
            "engine_arguments": engine_arguments(args.model, args.case),
            "environment": environment_identity(environment)["execution_values"],
            "diagnostic": diagnostic,
        }
        record["identity"] = identity
        if args.check_run:
            check_path = args.check_run / "run.json"
            check = json.loads(check_path.read_text())
            if (
                check["status"] != "completed"
                or check["mode"] != "check"
                or compatibility(check["identity"]) != compatibility(identity)
            ):
                raise ValueError("Independent check identity differs from this run")
            checked_worker = verify_worker(args.check_run / "sample_00/engine", "check")
            if not (
                checked_worker["prefix_sha256"]
                == checked_worker["row"]["input_sha256"]
                == prompt_sha256
            ):
                raise ValueError("Independent check did not use the current exact history tokens")
            record["check"] = {
                "path": str(check_path),
                "sha256": file_sha256(check_path),
                "run_id": check["run_id"],
            }
        write_json(data / "run.json", record)
        for index in range(args.repeats if args.mode == "bench" else 1):
            name = f"sample_{index:02d}"
            worker = data / name
            command = [
                str(args.python),
                "-B",
                "-m",
                "experiments.deepseek_v32_echo_official.src.decode_run",
                "--model",
                str(args.model),
                "--workload",
                str(args.workload),
                "--case",
                args.case,
                "--mode",
                args.mode,
                "--output",
                str(worker / "engine"),
                "--startup-timeout",
                str(args.startup_timeout),
                "--request-timeout",
                str(args.request_timeout),
            ]
            if args.mode == "profile":
                profiles.mkdir(parents=True)
                command = [
                    str(args.nsys),
                    "profile",
                    "--trace=cuda,nvtx",
                    "--sample=none",
                    "--cpuctxsw=none",
                    "--cuda-graph-trace=node",
                    "--capture-range=cudaProfilerApi",
                    "--capture-range-end=stop",
                    "--wait=all",
                    "--force-overwrite=false",
                    "--output",
                    str(profiles / "decode"),
                    *command,
                ]
            run_owned(
                args,
                command,
                data=worker,
                logs=logs / name,
                environment=environment,
                timeout=args.startup_timeout + 3 * args.request_timeout + 300,
                profile_ownership=args.mode == "profile",
            )
            completion = verify_worker(worker / "engine", args.mode)
            if not (
                completion["prefix_sha256"] == completion["row"]["input_sha256"] == prompt_sha256
            ):
                raise ValueError("Worker did not use the qualified exact history tokens")
            capacity = audit_capacity(
                json.loads((worker / "engine/server_info.json").read_text()),
                {str(path): path.read_text() for path in (logs / name).glob("worker.*.log")},
                case=args.case,
            )
            write_json(worker / "capacity.json", capacity)
            row = {
                "name": name,
                "command": command,
                "completion": completion,
                "capacity": capacity,
                "gpu_release_sha256": file_sha256(worker / "gpu_release_drain.json"),
                "jit": capture_runtime_manifest(worker / "jit_artifacts.json"),
            }
            if args.mode == "profile":
                path = profiles / "decode.nsys-rep"
                if not path.exists() or not path.stat().st_size:
                    raise ValueError("NSYS did not produce a report")
                row["nsys"] = {"path": str(path), "sha256": file_sha256(path)}
                row["hooks"] = {
                    path.name: file_sha256(path)
                    for path in (worker / "engine/hooks").glob("*.json")
                }
            record["workers"].append(row)
            write_json(data / "run.json", record)
        if sources() != source_map:
            raise ValueError("Source changed during execution")
        for name, digest in source_map.items():
            target = data / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / name, target)
            if file_sha256(target) != digest:
                raise ValueError("Source snapshot differs")
        record.update(status="completed", finished_ns=time.time_ns())
        write_json(data / "run.json", record)
    except BaseException as error:
        record.update(status="failed", error=repr(error), finished_ns=time.time_ns())
        try:
            write_json(data / "run.json", record)
        except BaseException as cleanup:  # noqa: BLE001
            raise BaseExceptionGroup(
                "Decode execution and failure-record persistence failed", [error, cleanup]
            )
        raise
    print(json.dumps({"run_id": args.run_id, "status": "completed"}))


if __name__ == "__main__":
    main()
