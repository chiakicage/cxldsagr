"""Run independent official SGLang Engine checks, timing, and NSYS profiling.

No HTTP server or client is launched. Each pair owns a fresh Engine, warms it
with two synthetic disjoint-prefix pairs, then executes formal H with a logical
prefix miss and H+A with exactly H cached. Official cache is never flushed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
ECHO_ROOT = REPO_ROOT / "3rdparty/ECHO"
REPRO_ROOT = ECHO_ROOT / "reproduction/cxldsagr"
RUNTIME_ROOT = ROOT / "output/runtime"
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPRO_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from scripts.lifecycle import environment_identity, managed_process
from scripts.run import (
    GPUProcessMonitor,
    assert_gpu_idle,
    drain_owned_gpu_processes,
    load_numerical_diagnostic,
    server_environment,
    write_json,
)
from src.capacity import audit_capacity
from src.preflight import file_sha256, validate_environment
from src.workload_client import canonical, load_workload

from experiments.deepseek_v32_echo_official.scripts.source_identity import (
    acceptance_identity,
    archive_sources,
    source_identity,
    verify_source_archive,
)
from experiments.deepseek_v32_echo_official.src.engine_run import WARMUP_PROTOCOL, engine_arguments


def scoped_runtime_environment(base: dict[str, str]) -> dict[str, str]:
    """Keep every writable runtime/cache/temp location inside this experiment."""
    result = dict(base)
    paths = {
        "UV_CACHE_DIR": "uv-cache",
        "XDG_CACHE_HOME": "cache",
        "XDG_CONFIG_HOME": "config",
        "XDG_DATA_HOME": "data",
        "DG_JIT_CACHE_DIR": "cache/deep-gemm",
        "SGL_DG_CACHE_DIR": "cache/deep-gemm",
        "SGLANG_DG_CACHE_DIR": "cache/deep-gemm",
        "SGLANG_CACHE_DIR": "cache/sglang",
        "SGLANG_TORCH_PROFILER_DIR": "profile",
        "FLASHINFER_WORKSPACE_BASE": "cache/flashinfer-workspace",
        "TRITON_CACHE_DIR": "cache/triton",
        "TRITON_DUMP_DIR": "cache/triton-dump",
        "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
        "TORCH_EXTENSIONS_DIR": "cache/torch-extensions",
        "TORCH_HOME": "cache/torch",
        "CUDA_CACHE_PATH": "cache/cuda-driver",
        "NUMBA_CACHE_DIR": "cache/numba",
        "MPLCONFIGDIR": "cache/matplotlib",
        "HF_HOME": "cache/huggingface",
        "HF_HUB_CACHE": "cache/huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": "cache/huggingface/hub",
        "HF_MODULES_CACHE": "cache/huggingface/modules",
        "PROMETHEUS_MULTIPROC_DIR": "prometheus",
        "PYTHONPYCACHEPREFIX": "pycache",
        "TMPDIR": "tmp",
        "TMP": "tmp",
        "TEMP": "tmp",
    }
    runtime = RUNTIME_ROOT.resolve()
    if not runtime.is_relative_to((ROOT / "output").resolve()):
        raise ValueError("runtime root must remain inside experiment output")
    for variable, suffix in paths.items():
        path = runtime / suffix
        if not path.resolve().is_relative_to(runtime):
            raise ValueError(f"runtime path escapes the experiment: {path}")
        path.mkdir(parents=True, exist_ok=True)
        result[variable] = str(path)
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    result["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT), str(REPRO_ROOT)))
    # These optional writers would otherwise retain a caller's unrelated path.
    for variable, suffix in (
        ("TRANSFORMERS_CACHE", "cache/huggingface/hub"),
        ("FLASHINFER_CUBIN_DIR", "cache/flashinfer-cubins"),
        ("NCCL_DEBUG_FILE", "nccl.%h.%p.log"),
        ("CUDNN_LOGDEST_DBG", "cudnn.log"),
    ):
        if variable in result:
            result[variable] = str(runtime / suffix)
    return result


def capture_runtime_manifest(output_path: Path) -> dict:
    """Inventory experiment-owned JIT artifacts after the owned worker exits."""
    cache_root = (RUNTIME_ROOT / "cache").resolve(strict=True)
    files = {}
    for path in sorted(cache_root.rglob("*")):
        relative = path.relative_to(cache_root)
        if (
            any(part in ("__pycache__", "tmp", "temp", "locks") for part in relative.parts)
            or path.suffix in (".pyc", ".lock", ".tmp")
            or path.name in ("lock", ".ninja_log", ".ninja_deps", "flashinfer_jit.log")
            or path.name.startswith("__grp__")
            or not path.is_file()
        ):
            continue
        if not path.resolve().is_relative_to(cache_root):
            raise ValueError(f"runtime cache symlink escapes its experiment directory: {path}")
        before = path.stat()
        digest = file_sha256(path)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"runtime artifact changed during snapshot: {path}")
        files[relative.as_posix()] = {
            "path": str(path),
            "size_bytes": after.st_size,
            "sha256": digest,
        }
    manifest = {
        "schema": "echo-sglang-engine-runtime-snapshot-v1",
        "cache_root": str(cache_root),
        "captured_ns": time.time_ns(),
        "files": files,
        "file_count": len(files),
        "total_file_bytes": sum(row["size_bytes"] for row in files.values()),
        "scope": (
            "Artifacts present in the experiment-owned cache after process release; "
            "not evidence that each artifact was loaded or executed. Fresh cache tree "
            "is populated by independent checks/warmup; packaged native inputs stay read-only."
        ),
    }
    if output_path.exists():
        raise FileExistsError(output_path)
    write_json(output_path, manifest)
    return {"path": str(output_path), "sha256": file_sha256(output_path), "file_count": len(files)}


def run_owned(args, command, *, data, logs, environment, timeout, profile_ownership=False) -> None:
    """Run one owned process tree and confirm GPU release, including on failure."""
    assert_gpu_idle(args.gpu)
    data.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    if profile_ownership:
        from experiments.deepseek_v32_echo_official.scripts.profile_lifecycle import (
            run_profile_owned,
        )

        run_profile_owned(
            args,
            command,
            cwd=REPO_ROOT,
            data=data,
            logs=logs,
            environment=environment,
            timeout=timeout,
        )
        return
    monitor = None
    primary = None
    try:
        with (
            managed_process(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout_path=logs / "worker.stdout.log",
                stderr_path=logs / "worker.stderr.log",
                event_path=data / "process_events.jsonl",
                shutdown_timeout=args.shutdown_timeout,
            ) as process,
            GPUProcessMonitor(
                args.gpu, process.process.pid, data / "gpu_processes.jsonl"
            ) as monitor,
        ):
            process.wait(timeout, dependencies=(monitor,))
    except BaseException as error:
        primary = error
        raise
    finally:
        if monitor is not None:
            try:
                drain_owned_gpu_processes(
                    args.gpu,
                    pgid=monitor.pgid,
                    owned_processes=monitor.owned_processes,
                    path=data / "gpu_release_drain.json",
                )
            except BaseException as cleanup:
                if primary is not None:
                    raise BaseExceptionGroup(
                        "Engine run and GPU release failed", [primary, cleanup]
                    )
                raise


def verify_pair(path: Path, mode: str) -> dict:
    complete = json.loads((path / "complete.json").read_text())
    rows = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()]
    if (
        complete.get("schema") != "echo-sglang-engine-pair-v1"
        or complete.get("status") != "completed"
        or complete.get("completed") != 2
        or complete.get("mode") != mode
        or complete.get("requests_sha256") != file_sha256(path / "requests.jsonl")
        or [row.get("phase") for row in rows] != ["prefill", "extend"]
    ):
        raise ValueError("incomplete or inconsistent Engine pair")
    if complete.get("warmup_protocol") != WARMUP_PROTOCOL:
        raise ValueError("Engine pair lacks the required in-process warmup protocol")
    warmup = json.loads((path / "in_engine_warmup.json").read_text())
    if (
        file_sha256(path / "in_engine_warmup.json") != complete.get("in_engine_warmup_sha256")
        or warmup.get("schema") != "echo-engine-in-process-warmup-v1"
        or warmup.get("status") != "completed"
        or warmup.get("warmup_protocol") != WARMUP_PROTOCOL
        or [
            (row["first_token_offset"], row["phase"], row["cached_tokens"])
            for row in warmup["rows"]
        ]
        != [(1, "prefill", 0), (1, "extend", 65536), (2, "prefill", 0), (2, "extend", 65536)]
    ):
        raise ValueError("Engine pair has incomplete or inconsistent warmup evidence")
    for row, prompt, cached in zip(rows, (65536, 65664), (0, 65536), strict=True):
        if (
            row.get("prompt_tokens") != prompt
            or row.get("cached_tokens") != cached
            or row.get("computed_prompt_tokens") != prompt - cached
            or row.get("completion_tokens") != 1
            or row.get("finish_reason") != {"type": "length", "length": 1}
            or len(row.get("output_ids", [])) != 1
        ):
            raise ValueError("Engine pair has the wrong request/cache boundary")
        if mode == "check":
            artifact = row["output_artifact"]
            if file_sha256(Path(artifact["path"])) != artifact["sha256"]:
                raise ValueError("independent Engine output artifact changed")
        elif (
            type(row.get("wall_ms")) not in (int, float)
            or not math.isfinite(row["wall_ms"])
            or row["wall_ms"] <= 0
        ):
            raise ValueError("Engine pair lacks finite positive wall timing")
    return {
        "path": str(path),
        "complete_sha256": file_sha256(path / "complete.json"),
        "requests_sha256": file_sha256(path / "requests.jsonl"),
        "cached_tokens": [0, 65536],
        "rows": rows,
    }


def run_pair(args, name, mode, data, logs, profiles, environment) -> dict:
    pair_data, pair_logs = data / name, logs / name
    pair_data.mkdir()
    pair_logs.mkdir()
    command = [
        str(args.python),
        "-B",
        "-m",
        "experiments.deepseek_v32_echo_official.src.engine_run",
        "--model",
        str(args.model),
        "--workload",
        str(args.workload),
        "--case",
        args.case,
        "--mode",
        mode,
        "--output",
        str(pair_data / "engine"),
        "--startup-timeout",
        str(args.startup_timeout),
        "--request-timeout",
        str(args.request_timeout),
    ]
    if args.enable_piecewise_cuda_graph:
        command.append("--enable-piecewise-cuda-graph")
    if mode == "profile":
        command += ["--profile-dir", str(pair_data / "profile_hooks")]
        profiles.mkdir(parents=True, exist_ok=False)
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
            str(profiles / "engine_pair"),
            *command,
        ]
    started_ns = time.time_ns()
    run_owned(
        args,
        command,
        data=pair_data,
        logs=pair_logs,
        environment=environment,
        timeout=args.startup_timeout + 6 * args.request_timeout + 300,
        profile_ownership=mode == "profile",
    )
    pair = verify_pair(pair_data / "engine", mode)
    capacity = audit_capacity(
        json.loads((pair_data / "engine/server_info.json").read_text()),
        {
            str(path): path.read_text()
            for path in (pair_logs / "worker.stdout.log", pair_logs / "worker.stderr.log")
        },
        case=args.case,
    )
    write_json(pair_data / "capacity.json", capacity)
    jit = capture_runtime_manifest(pair_data / "jit_artifacts.json")
    result = {
        "name": name,
        "mode": mode,
        "command": command,
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
        "capacity": capacity,
        "completion": pair,
        "jit_manifest": jit,
        "gpu_release_sha256": file_sha256(pair_data / "gpu_release_drain.json"),
    }
    if mode == "profile":
        report = profiles / "engine_pair.nsys-rep"
        if not report.is_file() or report.stat().st_size == 0:
            raise ValueError("NSYS returned zero without a nonempty report")
        result["nsys_report"] = {
            "path": str(report),
            "sha256": file_sha256(report),
            "bytes": report.stat().st_size,
        }
    return result


def audit_check(path: Path, identity: dict) -> dict:
    record_path = path / "run.json"
    record = json.loads(record_path.read_text())
    if (
        record.get("schema") != "echo-sglang-engine-run-v1"
        or record.get("mode") != "check"
        or record.get("status") != "completed"
        or record.get("exitcode") != 0
        or record.get("numerical_acceptance") is not False
        or record.get("structure_check_passed") is not True
        or len(record.get("pairs", [])) != 1
    ):
        raise ValueError("a matching completed independent Engine structure check is required")
    archive = verify_source_archive(path, record)
    core = acceptance_identity(identity)
    if acceptance_identity(record["execution_identity"]) != core:
        raise ValueError("Engine worker/input/runtime acceptance identity differs from the check")
    pair = record["pairs"][0]
    current = verify_pair(path / "check/engine", "check")
    if pair["completion"] != current:
        raise ValueError("independent Engine check evidence changed")
    capacity = json.loads((path / "check/capacity.json").read_text())
    if capacity != pair["capacity"] or capacity.get("passed") is not True:
        raise ValueError("independent Engine capacity check changed or did not pass")
    return {
        "run_id": record["run_id"],
        "path": str(record_path),
        "sha256": file_sha256(record_path),
        "structure_check_passed": True,
        "numerical_acceptance": False,
        "source_archive": archive,
        "comparison_rule": core["comparison_rule"],
        "acceptance_identity_sha256": hashlib.sha256(canonical(core).encode()).hexdigest(),
        "check_execution_identity_sha256": record["execution_identity_sha256"],
        "orchestration_source_differences": {
            key: {
                "check": record["reproduction_source_sha256"].get(key),
                "run": identity["reproduction_source_sha256"].get(key),
            }
            for key in sorted(
                set(record["reproduction_source_sha256"])
                | set(identity["reproduction_source_sha256"])
            )
            if record["reproduction_source_sha256"].get(key)
            != identity["reproduction_source_sha256"].get(key)
        },
        "scope": "Exact Engine worker, input, qualified runtime/config/environment and required execution helpers; archived observer/launcher/profile-only source differences are disclosed separately. No numerical equivalence pass is claimed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case", choices=("echo", "resident_reference"), required=True)
    parser.add_argument("--mode", choices=("check", "bench", "profile"), required=True)
    parser.add_argument(
        "--enable-piecewise-cuda-graph",
        action="store_true",
        help="Request the official prefill/extend graph path; verify runtime support before timing.",
    )
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--python", type=Path, default=REPRO_ROOT / "env/.venv/bin/python")
    parser.add_argument(
        "--model", type=Path, default=REPRO_ROOT / "weights/deepseek_v32_first3_fp8"
    )
    parser.add_argument("--workload", type=Path, default=REPRO_ROOT / "inputs/fixed_history")
    parser.add_argument("--build-receipt", type=Path, default=REPRO_ROOT / "env/build_receipt.json")
    parser.add_argument("--numerical-diagnostic", type=Path, required=True)
    parser.add_argument("--performance-only", action="store_true", required=True)
    parser.add_argument(
        "--check-run",
        type=Path,
        help="Matching independent Engine check; required for bench/profile.",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "output")
    parser.add_argument("--nsys", type=Path, default=Path(shutil.which("nsys") or "nsys"))
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--shutdown-timeout", type=float, default=30)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("run-id must be a safe new directory name")
    if not re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)", args.gpu):
        parser.error("--gpu must identify exactly one physical GPU")
    if min(args.startup_timeout, args.request_timeout, args.shutdown_timeout) <= 0:
        parser.error("all lifecycle deadlines must be positive")
    if (args.mode == "check") != (args.check_run is None):
        parser.error("--check-run is required for bench/profile and forbidden for check")
    for name in ("python", "model", "workload", "build_receipt", "numerical_diagnostic"):
        setattr(args, name, getattr(args, name).absolute())
        if not getattr(args, name).exists():
            raise FileNotFoundError(getattr(args, name))
    args.output_root = args.output_root.absolute()
    if not args.output_root.resolve().is_relative_to((ROOT / "output").resolve()):
        parser.error(
            "--output-root must remain inside experiments/deepseek_v32_echo_official/output"
        )
    if args.check_run is not None:
        args.check_run = args.check_run.resolve(strict=True)
    if args.mode == "profile":
        args.nsys = args.nsys.resolve(strict=True)
    diagnostic = load_numerical_diagnostic(args.numerical_diagnostic)
    manifest, requests = load_workload(args.workload)
    validate_environment(dict(os.environ))
    environment = server_environment(
        scoped_runtime_environment(dict(os.environ)), args.case, args.gpu
    )
    # A caller cannot silently enable profiling hooks in a check or timing run.
    if any(key.startswith("ECHO_ENGINE_PROFILE_") for key in environment):
        raise ValueError("Engine profiling environment must be set only by the profile worker")
    data, logs = args.output_root / "data" / args.run_id, args.output_root / "log" / args.run_id
    profiles = args.output_root / "profile" / args.run_id
    diagnostics = args.output_root / "diagnostics" / args.run_id
    if any(path.exists() for path in (data, logs, profiles, diagnostics)):
        raise FileExistsError(f"run ID already exists: {args.run_id}")
    assert_gpu_idle(args.gpu)
    data.mkdir(parents=True)
    logs.mkdir(parents=True)
    sources = source_identity()
    record = {
        "schema": "echo-sglang-engine-run-v1",
        "run_id": args.run_id,
        "case": args.case,
        "mode": args.mode,
        "status": "running",
        "exitcode": None,
        "started_ns": time.time_ns(),
        "pairs": [],
        "numerical_acceptance": False,
        "user_requested_performance_only": True,
        "numerical_diagnostic": diagnostic,
        "workload_sha256": manifest["workload_sha256"],
        "source_request_id": 0,
        "warmup_protocol": WARMUP_PROTOCOL,
        "source_input_sha256": requests[0]["input_sha256"],
        "reproduction_source_sha256": sources,
        "environment": environment_identity(environment),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "protocol": (
            "Check: one fresh H->H+A Engine pair. Bench: one separate warmup pair then "
            "five fresh Engine pairs; prefill samples are pairs 0-2, pairs 3-4 prefill is "
            "setup only, and all five extends are samples. Profile: one separate warmup "
            "pair then one fresh instrumented pair. Same workload request 0 in every "
            "pair; prefill cached=0, extend cached=65536 must hold exactly. Every worker, "
            "including outer warmup, first runs two synthetic H->H+A warm pairs (+1/+2 "
            "to token0 modulo129280). Formal pairs use the original input. No flush."
        ),
        "measurement_boundary": (
            "Engine.generate wall time, retaining scheduler/IPC/sampling/detokenization. "
            "No HTTP. In-process synthetic warmup is excluded. Formal prefill is a logical "
            "prefix miss in a warmed Engine with normal official eviction, not an empty "
            "allocator. Extend retains post-prefill HBM residency. No claim of "
            "equivalence to MFU warmed/cold timing, equal capacity, or isolated HTTP overhead."
        ),
        "monitor_scope": "Discrete GPU ownership observations; continuous isolation is not proven.",
    }
    write_json(data / "run.json", record)
    try:

        def terminate(signum, frame):
            raise InterruptedError(f"Engine driver received signal {signum}")

        signal.signal(signal.SIGTERM, terminate)
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
        record["preflight_command"] = preflight_command
        write_json(data / "run.json", record)
        run_owned(
            args,
            preflight_command,
            data=data / "preflight",
            logs=logs / "preflight",
            environment=environment,
            timeout=600,
        )
        preflight = json.loads((data / "preflight.json").read_text())
        if not preflight.get("runtime_validated"):
            raise ValueError("preflight did not qualify the official runtime")
        identity = {
            "case": args.case,
            "engine_arguments": engine_arguments(
                args.model, args.case, piecewise_cuda_graph=args.enable_piecewise_cuda_graph
            ),
            "workload_sha256": manifest["workload_sha256"],
            "source_input_sha256": requests[0]["input_sha256"],
            "warmup_protocol": WARMUP_PROTOCOL,
            "reproduction_source_sha256": sources,
            "preflight": preflight,
            "numerical_diagnostic_sha256": diagnostic["sha256"],
            "execution_environment_values": environment_identity(environment)["execution_values"],
        }
        record["execution_identity"] = identity
        record["execution_identity_sha256"] = hashlib.sha256(
            canonical(identity).encode()
        ).hexdigest()
        record["acceptance_identity_sha256"] = hashlib.sha256(
            canonical(acceptance_identity(identity)).encode()
        ).hexdigest()
        record["acceptance_identity"] = acceptance_identity(identity)
        if args.check_run is not None:
            record["check_audit"] = audit_check(args.check_run, identity)
        if args.mode == "profile":
            record["nsys_version"] = subprocess.check_output(
                [str(args.nsys), "--version"], text=True
            ).strip()
        write_json(data / "run.json", record)
        if args.mode != "check":
            record["warmup"] = run_pair(args, "warmup", "warmup", data, logs, profiles, environment)
            write_json(data / "run.json", record)
        for index in range(5 if args.mode == "bench" else 1):
            name = f"pair_{index:02d}" if args.mode == "bench" else args.mode
            pair = run_pair(args, name, args.mode, data, logs, profiles, environment)
            if args.mode == "bench":
                for row in pair["completion"]["rows"]:
                    row["sample_index"] = index
                    row["measured"] = row["phase"] == "extend" or index < 3
                    row["pair_name"] = name
            record["pairs"].append(pair)
            write_json(data / "run.json", record)
        if source_identity() != sources:
            raise ValueError("reproduction source changed during the Engine run")
        if load_numerical_diagnostic(args.numerical_diagnostic) != diagnostic:
            raise ValueError("retained numerical diagnostic changed during the Engine run")
        if args.mode == "bench":
            samples = [
                row
                for pair in record["pairs"]
                for row in pair["completion"]["rows"]
                if row["measured"]
            ]
            write_json(data / "timing_samples.json", samples)
            record["timing_samples_sha256"] = file_sha256(data / "timing_samples.json")
            record["sample_counts"] = {
                phase: sum(row["phase"] == phase for row in samples)
                for phase in ("prefill", "extend")
            }
        record.update(
            status="completed",
            exitcode=0,
            finished_ns=time.time_ns(),
            structure_check_passed=True,
        )
        write_json(data / "run.json", record)
        archive_sources(data, record)
    except BaseException as primary:
        record.update(
            status="failed_diagnostic_only",
            exitcode=1,
            finished_ns=time.time_ns(),
            error=repr(primary),
        )
        try:
            write_json(data / "run.json", record)
            diagnostics.mkdir(parents=True)
            shutil.move(str(data), str(diagnostics / "data"))
            shutil.move(str(logs), str(diagnostics / "log"))
            if profiles.exists():
                shutil.move(str(profiles), str(diagnostics / "profile"))
        except BaseException as cleanup:  # noqa: BLE001
            raise BaseExceptionGroup(
                "Engine run and diagnostic archival failed", [primary, cleanup]
            )
        raise
    print(json.dumps({"run_id": args.run_id, "data": str(data), "status": record["status"]}))


if __name__ == "__main__":
    main()
