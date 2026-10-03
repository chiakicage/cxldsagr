"""Single-GPU paired GR service measurements with every-request numerical checks."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

import torch

from experiments.gr_serving.src.report import write_report
from experiments.gr_serving.src.workload import WorkloadConfig, build_workload
from GR.input_generator import MODEL_FORMATS
from serving.persistent import PersistentGRRunner

SCHEMES = {
    "deepseek_v32": ("hbm", "echo", "serial_sparse", "dense_prefetch"),
    "nosa": ("hbm", "serial_sparse", "dense_prefetch", "overlap"),
}
ROOT = Path(__file__).resolve().parents[3]


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_snapshot(output):
    """Save actual source bytes, including uncommitted and untracked implementation."""
    extensions = {".py", ".sh", ".cu", ".cuh", ".cpp", ".h", ".hpp"}
    manifest = {}
    for directory in ("models", "layers", "operators", "cache", "executor", "serving", "GR"):
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file() or path.suffix not in extensions:
                continue
            if any(part in ("__pycache__", "output", "generated", "build") for part in path.parts):
                continue
            relative = path.relative_to(ROOT)
            destination = output / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((ROOT / "experiments/gr_serving").rglob("*")):
        if not path.is_file() or path.suffix not in (".py", ".sh") or "output" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(output / "source_manifest.json", manifest)
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def numerical_comparison(actual, expected, *, atol, rtol):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("all candidate hidden states must match shape and dtype")
    a, b = actual.float(), expected.float()
    if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise AssertionError("nonfinite output in serving correctness gate")
    difference = a - b
    metrics = {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "exact": bool(torch.equal(actual, expected)),
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(difference.norm() / b.norm().clamp_min(1e-20)),
        "atol": atol,
        "rtol": rtol,
    }
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return metrics


def verify_source_snapshot(output):
    manifest = json.loads((output / "source_manifest.json").read_text())
    changed = [
        name
        for name, digest in manifest.items()
        if not (ROOT / name).is_file()
        or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
    ]
    if changed:
        raise RuntimeError(f"implementation changed while measuring: {changed}")


def verify_access_trace(workload, path):
    """Bind generated model inputs to an already selected complete access trace."""
    with Path(path).open(newline="") as source:
        expected = list(csv.DictReader(source))
    if len(expected) != len(workload.requests):
        raise ValueError("saved access trace and generated workload have different lengths")
    for request, row in zip(workload.requests, expected, strict=True):
        identity = {name: int(row[name]) for name in ("request_id", "user_id", "visit_index")}
        identity.update(
            is_revisit=bool(int(row["is_revisit"])),
            previous_request_id=(
                int(row["previous_request_id"]) if row["previous_request_id"] else None
            ),
            timestamp=float(row["synthetic_timestamp"]),
        )
        if any(request[name] != value for name, value in identity.items()):
            raise ValueError(f"request {request['request_id']}: differs from saved access trace")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _backend(model, args):
    if model == "deepseek_v32":
        from models.deepseek_v32.serving_backend import DeepSeekServingBackend

        return DeepSeekServingBackend(
            args.deepseek_path,
            scheme="hbm",
            device=args.device,
            num_layers=args.deepseek_layers,
            chunk_size=args.chunk_size,
            slots=args.deepseek_slots,
        )
    from models.nosa.serving import NosaServingBackend

    return NosaServingBackend.from_pretrained(
        args.nosa_path,
        scheme="hbm",
        device=args.device,
        chunk_size=args.chunk_size,
        max_seq_len=args.history_tokens + args.candidate_tokens,
    )


def _select_scheme(backend, model, scheme, args):
    if model == "deepseek_v32":
        backend.scheme = scheme
        return backend
    from models.nosa.serving import NosaServingBackend

    return NosaServingBackend(backend.model, scheme, chunk_size=args.chunk_size)


def _warmup(backend, request, args):
    # Distinct cache ownership; none of these sessions survive into measurement.
    with PersistentGRRunner(
        backend, hbm_budget_bytes=args.hbm_budget_bytes, dram_budget_bytes=args.dram_budget_bytes
    ) as runner:
        for _ in range(args.warmup):
            result = runner.execute(request)
            del result
    backend.synchronize()


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--run-id", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--models", nargs="+", choices=tuple(SCHEMES), default=list(SCHEMES))
    result.add_argument("--users", nargs="+", type=int, default=[1, 8, 32, 64, 128, 256, 512])
    result.add_argument(
        "--requests",
        type=int,
        default=32,
        help="total draws per population; an explicit legacy cap may end the trace early",
    )
    result.add_argument(
        "--max-revisits",
        type=int,
        help="optional legacy per-user revisit cap; default is uncapped",
    )
    result.add_argument(
        "--allow-empty-revisits",
        action="store_true",
        help="retain a bounded trace with no revisits and report null revisit latency",
    )
    result.add_argument("--history-tokens", type=int, default=16384)
    result.add_argument("--candidate-tokens", type=int, default=1024)
    result.add_argument("--hbm-budget-gib", type=float, default=4.0)
    result.add_argument("--dram-budget-gib", type=float, default=16.0)
    result.add_argument("--chunk-size", type=int, default=1024)
    result.add_argument("--deepseek-slots", type=int, default=4096)
    result.add_argument("--deepseek-layers", type=int, default=10)
    result.add_argument("--deepseek-path", type=Path, default=Path("/preset-models"))
    result.add_argument("--nosa-path", type=Path, default=Path("/mnt/ssd-wlcb/chenkaiqi/NOSA-8B"))
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--sampling", choices=("weighted", "sequential"), default="weighted")
    result.add_argument("--heat-dataset", default="beauty")
    result.add_argument(
        "--heat-field", help="curve field; defaults to the selected dataset's native heat field"
    )
    result.add_argument(
        "--access-trace", type=Path, help="existing requests_N.csv; check every draw before loading"
    )
    result.add_argument("--warmup", type=int, default=2)
    result.add_argument("--atol", type=float, default=0.0)
    result.add_argument("--rtol", type=float, default=0.0)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.sampling == "sequential":
        if args.max_revisits is not None:
            raise ValueError("sequential traces do not support a revisit cap")
        if args.access_trace is not None:
            raise ValueError("sequential traces do not use an external heat trace")
        args.heat_dataset = None
        args.heat_field = None
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run ID must contain only letters, digits, underscores or hyphens")
    if args.warmup < 2 or min(args.users) < 1 or len(set(args.users)) != len(args.users):
        raise ValueError("use warmup >= 2 and distinct positive user populations")
    if args.requests < 2:
        raise ValueError("at least two requests are needed to measure revisits")
    if args.max_revisits is not None and args.max_revisits < 1:
        raise ValueError("max-revisits must permit at least one repeat visit")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("comparison tolerances must be nonnegative")
    args.hbm_budget_bytes = int(args.hbm_budget_gib * 2**30)
    args.dram_budget_bytes = int(args.dram_budget_gib * 2**30)
    if args.hbm_budget_bytes <= 0 or args.dram_budget_bytes <= 0:
        raise ValueError("both cache budget caps must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
        raise RuntimeError("serving comparison requires one SM90/Hopper GPU")
    torch.cuda.set_device(device)
    os.environ.setdefault("CXLDSAGR_SM90_BACKEND", "native")
    # Keep model order part of the experiment contract.
    models = [name for name in SCHEMES if name in args.models]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    source_id = source_snapshot(output)
    props = torch.cuda.get_device_properties(device)
    metadata = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "running",
        "started_unix": time.time(),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source_sha256": source_id,
        "git_revision": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "submodules": _git("submodule", "status"),
        "hardware": {
            "gpu_name": props.name,
            "gpu_uuid": str(props.uuid),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": props.total_memory,
            "sm_count": props.multi_processor_count,
            "device": str(device),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "host": platform.node(),
            "python": platform.python_version(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "torch_cpu_threads": torch.get_num_threads(),
            "host_memory_placement": "default process NUMA policy; pinned local CPU DRAM",
        },
        "dependencies": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            **{
                name: importlib.metadata.version(name)
                for name in (
                    "triton",
                    "flashinfer-python",
                    "apache-tvm-ffi",
                    "safetensors",
                    "tokenizers",
                )
            },
        },
        "measurement": "synchronized serial wall latency; includes cache admission, cold prefix, all candidate tokens, and cleanup; excludes token generation, loading, JIT warmup and numerical comparison",
        "cache_policy": "whole-user LRU; prefix identity checked; candidate suffix truncated; equal HBM/DRAM caps include derived records, cache scratch and staging",
        "numerical_gate": "every request's complete candidate hidden compared to HBM on identical trace; each scheme begins with empty caches",
        "models": {},
        "cases": [],
    }
    _write(output / "metadata.json", metadata)
    rows = []
    with (
        (output / "measurements.jsonl").open("x") as measurements,
        (output / "correctness.jsonl").open("x") as correctness,
    ):
        for model in models:
            tokenizer = args.deepseek_path if model == "deepseek_v32" else args.nosa_path
            workloads = {}
            for users in args.users:
                print(
                    json.dumps(
                        {
                            "event": "build_workload",
                            "model": model,
                            "users": users,
                            "requests": args.requests,
                        }
                    ),
                    flush=True,
                )
                config = WorkloadConfig(
                    model,
                    users,
                    args.requests,
                    history_tokens=args.history_tokens,
                    candidate_tokens=args.candidate_tokens,
                    seed=args.seed,
                    sampling=args.sampling,
                    heat_dataset=args.heat_dataset,
                    heat_field=args.heat_field,
                    max_revisits=args.max_revisits,
                    context_limit=(
                        args.history_tokens + args.candidate_tokens
                        if args.history_tokens + args.candidate_tokens
                        > MODEL_FORMATS[model].MAX_INPUT_TOKENS
                        else None
                    ),
                )
                workloads[users] = build_workload(config, tokenizer=tokenizer)
                if args.access_trace is not None:
                    trace_hash = verify_access_trace(workloads[users], args.access_trace)
                    if metadata.get("access_trace_sha256", trace_hash) != trace_hash:
                        raise RuntimeError("selected access trace changed between workloads")
                    metadata["access_trace_sha256"] = trace_hash
                    shutil.copyfile(args.access_trace, output / "access_trace.csv")
                workloads[users].write(output / "workloads" / model / str(users))
                print(
                    json.dumps(
                        {
                            "event": "workload_ready",
                            "model": model,
                            "users": users,
                            **workloads[users].manifest["observed"],
                        }
                    ),
                    flush=True,
                )
                if not args.allow_empty_revisits and not any(
                    request["is_revisit"] for request in workloads[users].requests
                ):
                    raise ValueError("sampled trace has no revisits; increase request count")
            print(json.dumps({"event": "load_model", "model": model}), flush=True)
            backend = _backend(model, args)
            metadata["models"][model] = backend.describe()
            _write(output / "metadata.json", metadata)
            for scheme in SCHEMES[model]:
                backend = _select_scheme(backend, model, scheme, args)
                print(json.dumps({"event": "warmup", "model": model, "scheme": scheme}), flush=True)
                _warmup(backend, workloads[args.users[0]].requests[0], args)
                for users, workload in workloads.items():
                    reference = output / "reference" / model / str(users)
                    reference.mkdir(parents=True, exist_ok=True)
                    torch.cuda.reset_peak_memory_stats(device)
                    start = time.time()
                    max_error = 0.0
                    exact = True
                    with PersistentGRRunner(
                        backend,
                        hbm_budget_bytes=args.hbm_budget_bytes,
                        dram_budget_bytes=args.dram_budget_bytes,
                    ) as runner:
                        for request in workload.requests:
                            result = runner.execute(request)
                            actual = result.hidden.detach().cpu()
                            del result.hidden
                            request_id = request["request_id"]
                            reference_file = reference / f"{request_id:06d}.pt"
                            if scheme == "hbm":
                                torch.save(actual, reference_file)
                                expected = actual
                            else:
                                expected = torch.load(
                                    reference_file, weights_only=True, map_location="cpu"
                                )
                            comparison = numerical_comparison(
                                actual, expected, atol=args.atol, rtol=args.rtol
                            )
                            max_error = max(max_error, comparison["max_abs"])
                            exact &= comparison["exact"]
                            correctness.write(
                                json.dumps(
                                    {
                                        "model": model,
                                        "scheme": scheme,
                                        "num_users": users,
                                        "request_id": request_id,
                                        **comparison,
                                    }
                                )
                                + "\n"
                            )
                            correctness.flush()
                            row = {
                                **result.metrics,
                                "run_id": args.run_id,
                                "model": model,
                                "num_users": users,
                                "workload_sha256": workload.manifest["workload_sha256"],
                                "history_tokens": args.history_tokens,
                                "candidate_tokens": args.candidate_tokens,
                                "seed": args.seed,
                                "correctness_max_abs": comparison["max_abs"],
                                "correctness_exact": comparison["exact"],
                            }
                            measurements.write(json.dumps(row) + "\n")
                            measurements.flush()
                            rows.append(row)
                            if request_id % 8 == 0 or request_id + 1 == len(workload.requests):
                                print(
                                    json.dumps(
                                        {
                                            "event": "request",
                                            "model": model,
                                            "scheme": scheme,
                                            "users": users,
                                            "request": request_id,
                                            "latency_ms": row["latency_ms"],
                                            "hit": row["prefix_cache_hit"],
                                            "cache_users": row["cached_users"],
                                        }
                                    ),
                                    flush=True,
                                )
                            del result, actual, expected
                    case = {
                        "model": model,
                        "scheme": scheme,
                        "num_users": users,
                        "requests": len(workload.requests),
                        "observed_users": workload.manifest["observed"]["unique_users"],
                        "revisits": workload.manifest["observed"]["revisits"],
                        "max_revisits_observed": workload.manifest["observed"]["max_revisits"],
                        "duration_seconds": time.time() - start,
                        "all_hidden_exact": exact,
                        "max_abs": max_error,
                        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                        "session_reservation": backend.estimate_session_bytes(
                            args.history_tokens + args.candidate_tokens, args.history_tokens
                        ),
                    }
                    metadata["cases"].append(case)
                    _write(output / "metadata.json", metadata)
            del backend, workloads
            gc.collect()
            torch.cuda.empty_cache()
    # Publishable status requires every requested model/population/scheme trace.
    expected_cases = sum(len(SCHEMES[model]) * len(args.users) for model in models)
    if len(metadata["cases"]) != expected_cases:
        raise RuntimeError("incomplete comparison matrix")
    verify_source_snapshot(output)
    write_report(rows, output / "analysis")
    metadata.update(status="accepted", completed_unix=time.time(), measured_requests=len(rows))
    _write(output / "metadata.json", metadata)
    print(
        json.dumps(
            {
                "event": "accepted",
                "run_id": args.run_id,
                "cases": expected_cases,
                "measured_requests": len(rows),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
