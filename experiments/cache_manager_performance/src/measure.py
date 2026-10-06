"""Independent acceptance and NSYS measurement of the post-top-k cache window."""

import argparse
import importlib.metadata
import json
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch

from evaluation.local_native import collect_local_native_artifacts
from evaluation.validation import identity_digest, require_receipt, write_receipt
from experiments.cache_manager_performance.src.workload import (
    Config,
    Replay,
    file_sha256,
    load_workload,
)
from experiments.deepseek_v32_mfu.src.backend_provenance import (
    collect_backend_provenance,
    collect_flashinfer_runtime_artifacts,
)

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/cache_manager_performance"
KIND = "deepseek-cache-manager-transition-v2"
MEASURED_SCHEMES = ("echo", "serial_sparse")
MEASURED_PHASES = ("extend_cold", "extend_warm")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_identity():
    files = {ROOT / "pyproject.toml", ROOT / "uv.lock"}
    for directory in (
        "cache",
        "operators/deepseek_v32/indexer",
        "operators/deepseek_v32/attention",
        "operators/common",
        "models/deepseek_v32/cache",
        "experiments/cache_manager_performance",
    ):
        files.update(
            path
            for path in (ROOT / directory).rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".cu", ".cuh", ".cpp", ".h", ".sh"}
            and not {"tests", "output", "report", "__pycache__"}.intersection(path.parts)
        )
    files.update(
        ROOT / name
        for name in (
            "models/deepseek_v32/attention.py",
            "models/attention_contracts.py",
            "evaluation/validation.py",
            "evaluation/local_native.py",
            "experiments/deepseek_v32_mfu/src/kernel_profile.py",
            "experiments/deepseek_v32_mfu/src/analyze_nsys.py",
            "experiments/deepseek_v32_mfu/src/launch_gap.py",
            "experiments/deepseek_v32_mfu/src/timeline.py",
            "experiments/deepseek_v32_mfu/src/backend_provenance.py",
        )
    )
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(files)}


def execution_identity(config, captures):
    import deep_gemm

    from operators.deepseek_v32.attention.device_only.mla import build_info as mla_build_info
    from operators.deepseek_v32.indexer.echo import build_info

    local_native = collect_local_native_artifacts()
    libraries = {row["library"]["path"]: row["library"]["sha256"] for row in local_native}
    properties = torch.cuda.get_device_properties(0)
    deepgemm = Path(deep_gemm.__file__).resolve().parent
    return {
        "kind": KIND,
        "config": asdict(config),
        "pool_tokens": config.slots,
        "capture_files": captures,
        "sources": source_identity(),
        "echo_build": build_info(),
        "mla_build": mla_build_info(),
        "native_libraries": libraries,
        "local_native_jit": local_native,
        "deepgemm_native": {str(path): file_sha256(path) for path in deepgemm.glob("*.so")},
        "backend_provenance": collect_backend_provenance(),
        "flashinfer_runtime": collect_flashinfer_runtime_artifacts(require_local_native=True),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("triton", "apache-tvm-ffi", "flashinfer-python")
            },
            "device": properties.name,
            "uuid": str(properties.uuid),
            "capability": list(torch.cuda.get_device_capability(0)),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "torch_threads": torch.get_num_threads(),
            "python": sys.version,
            "allocator": os.environ.get("PYTORCH_ALLOC_CONF"),
        },
        "boundary": {
            "model_projection": False,
            "mla": True,
            "mlp": False,
            "actual_production_runner_cache_order": True,
            "actual_indexer_and_topk": True,
            "measured_start": "last exact-top-k GPU activity end, including invalid-ID mask",
            "measured_end": "first actual sparse_attn_fwd_kernel GPU start",
            "indexer_and_topk_compute_in_metric": False,
            "mla_compute_in_metric": False,
            "measurement": "NSYS timestamp interval; profiler overhead remains; no whole-replay wall-time metric",
            "measured_schemes": list(MEASURED_SCHEMES),
            "measured_phases": list(MEASURED_PHASES),
            "prefill_queries": "captured extend queries repeated; synthetic final chunk",
            "prefill_prefix": "real uninterrupted appends rebuilt before every sample",
            "extend_queries": "captured model layer queries",
            "extend_prefix": "production snapshot restore; cold additionally releases GPU IDs",
            "initial_hint": "zero for every independent replay",
            "whole_model_gap_gate": False,
            "mfu": None,
        },
    }


class Scopes:
    def __init__(self, scheme, phase):
        self.prefix = f"manager/{scheme}/{phase}"
        self.stack = []

    @contextmanager
    def __call__(self, stage):
        self.stack.append(stage)
        torch.cuda.nvtx.range_push(self.prefix + "/" + "/".join(self.stack))
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
            self.stack.pop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "profile"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--validation-receipt", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--history", type=int, default=65536)
    parser.add_argument("--append", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        raise ValueError("run ID must be a plain nonempty name")
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("warmup and repeats must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("cache manager replay requires a Hopper GPU")
    config = Config(args.history, args.append, args.chunk, args.layers)
    captures, capture_identity = load_workload(args.capture_dir, config)
    default_root = (
        Path("/tmp/cxldsagr-checks/cache_manager_performance")
        if args.mode == "check"
        else EXPERIMENT / "output"
    )
    output = (args.output_root or default_root) / "data" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    replays = {}
    profile_started = False
    cleanup_attempted = False
    with torch.inference_mode():
        try:
            for phase in MEASURED_PHASES:
                for scheme in ("hbm", *MEASURED_SCHEMES):
                    replay = Replay(captures, scheme, phase, config)
                    replays[(phase, scheme)] = replay
                    for _ in range(args.warmup):
                        replay.reset()
                        replay.run()
            identity = execution_identity(config, capture_identity)
            for relative in identity["sources"]:
                destination = output / "source" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((ROOT / relative).read_bytes())
            write_json(output / "identity.json", identity)
            receipt = None
            if args.mode != "check":
                receipt = require_receipt(args.validation_receipt, kind=KIND, identity=identity)
            result = {
                "schema": KIND,
                "run_id": args.run_id,
                "mode": args.mode,
                "identity_sha256": identity_digest(identity),
                "warmup": args.warmup,
                "repeats": args.repeats if args.mode == "profile" else 1,
                "validation_receipt": receipt["receipt_sha256"] if receipt else None,
                "invocation": list(sys.argv),
                "samples": [],
                "mfu": None,
                "complete_model_gate": "not evaluated by this standalone experiment",
                "measurement_boundary": identity["boundary"],
            }
            reference = {}
            # The logical resident consumer is an independent mapping/output
            # oracle. Its setup and comparisons are outside profiled windows.
            for phase in MEASURED_PHASES:
                replay = replays[(phase, "hbm")]
                replay.reset()
                replay.run()
                _, reference[phase] = replay.verify()
            if args.mode == "profile":
                torch.cuda.cudart().cudaProfilerStart()
                profile_started = True
            for phase in MEASURED_PHASES:
                for scheme in (
                    ("hbm", *MEASURED_SCHEMES) if args.mode == "check" else MEASURED_SCHEMES
                ):
                    replay = replays[(phase, scheme)]
                    for sample in range(result["repeats"]):
                        replay.reset()
                        scope = Scopes(scheme, phase) if args.mode == "profile" else None
                        with scope("window") if scope else torch.no_grad():
                            replay.run(scope)
                        row = {
                            "phase": phase,
                            "scheme": scheme,
                            "sample": sample,
                            "metrics": [runner.cache.metrics() for runner in replay.runners],
                        }
                        if args.mode in ("check", "profile"):
                            checks, evidence_tensors = replay.verify(reference[phase])
                            row["checks"] = checks
                            evidence = output / f"{phase}_{scheme}_{sample}_evidence.pt"
                            torch.save(evidence_tensors, evidence)
                            row["evidence"] = {
                                "path": evidence.name,
                                "sha256": file_sha256(evidence),
                            }
                        result["samples"].append(row)
                    print(f"{scheme:14s} {phase:20s} samples={result['repeats']}", flush=True)
            if profile_started:
                torch.cuda.cudart().cudaProfilerStop()
                profile_started = False
            if execution_identity(config, capture_identity) != identity:
                raise RuntimeError("execution source or native identity changed during measurement")
            cleanup_attempted = True
            cleanup_errors = []
            for replay in replays.values():
                try:
                    replay.close()
                except BaseException as error:  # noqa: BLE001 -- finish all cleanup before acceptance.
                    cleanup_errors.append(error)
            if cleanup_errors:
                raise BaseExceptionGroup("replay cleanup failed before acceptance", cleanup_errors)
            result["passed"] = True
            write_json(output / "result.json", result)
            if args.mode == "check":
                artifacts = {path.name: path for path in output.glob("*_evidence.pt")}
                artifacts["result"] = output / "result.json"
                write_receipt(
                    output / "receipt.json",
                    kind=KIND,
                    identity=identity,
                    checks={
                        "passed": True,
                        "cases": len(result["samples"]),
                        "scope": "exact selections, consumed KV, maps, free bitmap, clocks and real MLA outputs vs resident logical KV",
                    },
                    artifacts=artifacts,
                )
            print(output, flush=True)
        finally:
            errors = []
            if profile_started:
                try:
                    torch.cuda.cudart().cudaProfilerStop()
                except BaseException as error:  # noqa: BLE001 -- retain profiler cleanup failure.
                    errors.append(error)
            for replay in () if cleanup_attempted else replays.values():
                try:
                    replay.close()
                except BaseException as error:  # noqa: BLE001 -- preserve all cleanup failures.
                    errors.append(error)
            if errors:
                original = sys.exception()
                raise BaseExceptionGroup(
                    "replay resource cleanup failed", ([original] if original else []) + errors
                )


if __name__ == "__main__":
    main()
