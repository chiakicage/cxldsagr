"""Independent exact-recall acceptance and clean per-layer cache timings."""

import argparse
import importlib.metadata
import json
import os
import shutil
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from evaluation.local_native import collect_local_native_artifacts
from evaluation.validation import identity_digest, require_receipt, write_receipt
from experiments.cache_manager_performance.src.recall_workload import (
    STATES,
    RecallFixture,
    compare_states,
    load_candidate,
    validate_selection,
)
from experiments.cache_manager_performance.src.workload import Config, file_sha256, load_workload

ROOT = Path(__file__).resolve().parents[3]
KIND = "deepseek-state-exact-recall-v1"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def execution_sources(candidate_dir=None):
    """Executed fixture/cache sources only; report/analyzer edits are independent."""
    relative = (
        "cache/sparse_token_cache.py",
        "cache/sparse_token_pool.py",
        "cache/host_allocation.py",
        "operators/common/kv_transfer.py",
        "operators/common/csrc/kv_transfer.cu",
        "operators/deepseek_v32/indexer/echo.py",
        "operators/deepseek_v32/indexer/cache_ops.py",
        "operators/deepseek_v32/indexer/recall_dispatch.py",
        "evaluation/validation.py",
        "evaluation/local_native.py",
        "experiments/deepseek_v32_mfu/src/kernel_profile.py",
        "experiments/deepseek_v32_mfu/src/backend_provenance.py",
        "experiments/cache_manager_performance/src/workload.py",
        "experiments/cache_manager_performance/src/recall_workload.py",
        "experiments/cache_manager_performance/src/recall.py",
        "experiments/cache_manager_performance/scripts/recall.sh",
        "pyproject.toml",
        "uv.lock",
    )
    files = {"repo/" + name: ROOT / name for name in relative}
    for path in (ROOT / "operators/deepseek_v32/indexer/csrc").glob("echo_*"):
        if path.suffix in {".cu", ".cuh", ".cpp", ".h", ".hpp"}:
            files["repo/" + str(path.relative_to(ROOT))] = path
    if candidate_dir is not None:
        for path in candidate_dir.rglob("*"):
            if (
                path.is_file()
                and path.suffix in {".py", ".cu", ".cuh", ".cpp", ".h", ".hpp", ".json"}
                and "__pycache__" not in path.parts
            ):
                files["candidate/" + str(path.relative_to(candidate_dir))] = path
    return files


def source_hashes(files):
    return {name: file_sha256(path) for name, path in sorted(files.items())}


def runtime_identity():
    from experiments.deepseek_v32_mfu.src.backend_provenance import _recall_dispatch_identity
    from operators.deepseek_v32.indexer.echo import build_info

    local_native = collect_local_native_artifacts()
    libraries = {row["library"]["path"]: row["library"]["sha256"] for row in local_native}
    properties = torch.cuda.get_device_properties(0)
    return {
        "echo_build": build_info(),
        "recall_dispatch_build": _recall_dispatch_identity(),
        "native_library_sha256": libraries,
        "local_native_jit": local_native,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tvm_ffi": importlib.metadata.version("apache-tvm-ffi"),
        "python": sys.version,
        "gpu_uuid": str(properties.uuid),
        "gpu_name": properties.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("CXLDSAGR_", "TVM_", "OMP_", "MKL_", "OPENBLAS_", "PYTORCH_"))
            or key in {"CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING", "CUDA_LAUNCH_BLOCKING"}
        },
    }


def close_all(fixtures):
    errors = []
    for fixture in fixtures.values():
        try:
            fixture.close()
        except BaseException as error:  # noqa: BLE001 -- retain every cleanup failure.
            errors.append(error)
    if errors:
        original = sys.exception()
        raise BaseExceptionGroup(
            "recall execution/cleanup failed", ([original] if original else []) + errors
        )


def variants(capture, config, state, candidate, reference=False):
    result = {}
    try:
        result["baseline"] = RecallFixture(capture, config, state)
        if candidate is not None:
            result["candidate"] = RecallFixture(capture, config, state, candidate=candidate)
        if reference:
            result["reference"] = RecallFixture(capture, config, state)
        return result
    except BaseException:
        close_all(result)
        raise


def timed_call(fixture):
    """Only the production private dispatch and completion fence are measured."""
    start = time.perf_counter_ns()
    physical = fixture.invoke()
    enqueued = time.perf_counter_ns()
    torch.cuda.synchronize(fixture.device)
    complete = time.perf_counter_ns()
    return physical, {"enqueue_ms": (enqueued - start) / 1e6, "wall_ms": (complete - start) / 1e6}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "bench"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--validation-receipt", type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=31)
    args = parser.parse_args(argv)
    if not args.run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for c in args.run_id
    ):
        parser.error("run ID must contain only letters, digits, dot, underscore or hyphen")
    if args.warmup < 1 or args.repeats < 31:
        parser.error("warmup must be positive and repeats must be at least 31")
    if (args.mode == "bench") != (args.validation_receipt is not None):
        parser.error("bench requires a receipt; check creates its own receipt")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("exact recall requires an available Hopper GPU")
    config = Config()
    captures, capture_identity = load_workload(args.capture_dir, config)
    for capture in captures:
        validate_selection(capture, config)
    candidate_dir = args.candidate_dir.resolve(strict=True) if args.candidate_dir else None
    files = execution_sources(candidate_dir)
    sources = source_hashes(files)
    candidate = load_candidate(candidate_dir)
    output = args.output_root / "data" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode():
        for state in STATES:
            for capture in captures:
                fixtures = variants(capture, config, state, candidate)
                try:
                    for fixture in fixtures.values():
                        for _ in range(args.warmup):
                            fixture.reset()
                            fixture.invoke()
                            torch.cuda.synchronize(fixture.device)
                            fixture.finish()
                finally:
                    close_all(fixtures)
        identity = {
            "kind": KIND,
            "config": asdict(config),
            "states": STATES,
            "captures": capture_identity,
            "sources": sources,
            "runtime": runtime_identity(),
            "scope": "one actual layer; production exact recall plus device completion fence",
            "reference": "public ensure plus logical KV/map/free/protection/priority/counter oracle",
            "allocation_ties": "equal-priority slots may differ; exact top-k remains unchanged",
        }
        receipt = (
            require_receipt(args.validation_receipt, kind=KIND, identity=identity)
            if args.mode == "bench"
            else None
        )
        for name, path in files.items():
            destination = output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        write_json(output / "identity.json", identity)
        samples, checks, evidence = [], [], {}
        for state in STATES:
            for capture in captures:
                fixtures = variants(
                    capture, config, state, candidate, reference=args.mode == "check"
                )
                try:
                    names = [name for name in fixtures if name != "reference"]
                    for sample in range(2 if args.mode == "check" else args.repeats):
                        reference = None
                        if args.mode == "check":
                            fixture = fixtures["reference"]
                            fixture.reset()
                            before = fixture.before_recall()
                            physical = fixture.invoke(reference=True)
                            torch.cuda.synchronize(fixture.device)
                            reference = fixture.inspect(physical, before)
                            fixture.finish()
                        for name in names if sample % 2 == 0 else names[::-1]:
                            fixture = fixtures[name]
                            fixture.reset()
                            before = fixture.before_recall() if args.mode == "check" else None
                            physical, timing = timed_call(fixture)
                            if reference is not None:
                                actual = fixture.inspect(physical, before)
                                compare_states(actual, reference)
                                key = f"{state}_layer{capture['layer']}_{name}_{sample}"
                                evidence[key] = {
                                    k: v for k, v in actual.items() if k != "live_records"
                                }
                                checks.append({"case": key, "passed": True, **actual["allocation"]})
                            fixture.finish()
                            samples.append(
                                {
                                    "state": state,
                                    "layer": capture["layer"],
                                    "variant": name,
                                    "sample": sample,
                                    **timing,
                                    "metrics": fixture.cache.metrics(),
                                }
                            )
                finally:
                    close_all(fixtures)
        if source_hashes(files) != sources or runtime_identity() != identity["runtime"]:
            raise RuntimeError("execution source or native identity changed during the run")
        clock = time.get_clock_info("perf_counter")
        result = {
            "kind": KIND,
            "mode": args.mode,
            "run_id": args.run_id,
            "passed": True,
            "identity_sha256": identity_digest(identity),
            "validation_receipt": receipt["receipt_sha256"] if receipt else None,
            "invocation": sys.argv,
            "warmup": args.warmup,
            "samples": samples,
            "checks": checks,
            "clock": {
                "implementation": clock.implementation,
                "resolution_seconds": clock.resolution,
                "monotonic": clock.monotonic,
                "adjustable": clock.adjustable,
            },
            "boundary": "prefix/reset/begin/append/drain/stats reset/commit/oracles excluded; private recall and device fence included; wall minus enqueue is not pure GPU compute; no IO subtraction or model claim",
            "candidate_archive": "source/candidate" if candidate is not None else None,
        }
        write_json(output / "result.json", result)
        if args.mode == "check":
            torch.save(evidence, output / "state_evidence.pt")
            write_receipt(
                output / "receipt.json",
                kind=KIND,
                identity=identity,
                checks={"passed": True, "cases": checks},
                artifacts={
                    "result": output / "result.json",
                    "states": output / "state_evidence.pt",
                },
            )
        for state in STATES:
            for layer in range(config.layers):
                for name in ("baseline", "candidate") if candidate is not None else ("baseline",):
                    selected = [
                        row["wall_ms"]
                        for row in samples
                        if (row["state"], row["layer"], row["variant"]) == (state, layer, name)
                    ]
                    print(
                        f"{state} layer={layer} {name} median={statistics.median(selected):.6f} ms"
                    )
        print(output, flush=True)


if __name__ == "__main__":
    main()
