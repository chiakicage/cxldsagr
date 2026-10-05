"""Independent Nsight stage captures of the accepted official ECHO pipeline.

Capture one cold request and the first revisit after the complete first-round
user population. Report diagnostic stages and real kernels; graph-node traces
are not presented as standalone operator timings or calibrated MFU.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from unittest.mock import patch

from evaluation.validation import require_receipt
from experiments.deepseek_v32_echo_official.src import measure
from experiments.deepseek_v32_motivation.src.profile import Scopes, capture_range
from experiments.nosa_motivation.src.validation import (
    begin_validation,
    execution_environment,
    record_runtime,
    reference_directory,
)

SCHEMA = "deepseek-v32-echo-official-profile-v1"


def profile_environment():
    effective = execution_environment()
    # Nsight injects this library into the profiled child; it is diagnostic
    # instrumentation, recorded separately from the checked model environment.
    model = {key: value for key, value in effective.items() if key != "CUDA_INJECTION64_PATH"}
    injected = effective.get("CUDA_INJECTION64_PATH")
    injection = None if injected is None else {"path": injected, "sha256": measure.digest(injected)}
    return model, effective, injection


def finish_evidence(output, metadata, calls, comparisons):
    measure.write_json(output / "calls.json", calls)
    measure.write_json(output / "correctness.json", comparisons)
    metadata["profile_artifacts"] = {
        str(path.relative_to(output)): measure.digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path not in (output / "metadata.json", output / "result.json")
    }
    measure.write_json(output / "metadata.json", metadata)
    measure.write_json(
        output / "result.json",
        {
            "accepted": True,
            "captures": len(metadata["captures"]),
            "checked_requests": len(comparisons),
            "source_sha256": metadata["source_sha256"],
            "metadata_sha256": measure.digest(output / "metadata.json"),
            "reference_receipt_sha256": metadata["reference_receipt_sha256"],
        },
    )


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-id", required=True)
    command.add_argument("--reference-run", type=Path, required=True)
    command.add_argument("--output-dir", type=Path)
    command.add_argument("--receipt-override", type=Path)
    command.add_argument("--scheme", choices=measure.SCHEMES)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--nsys", action="store_true")
    return command


def load_reference(directory, *, receipt_override=None):
    original = json.loads((Path(directory) / "metadata.json").read_text())
    receipt_path = (
        receipt_override or original["correctness_receipt"]["path"]
        if original.get("schema") == measure.BENCH_SCHEMA
        else Path(directory) / "receipt.json"
    )
    directory = reference_directory(
        directory,
        bench_schema=measure.BENCH_SCHEMA,
        kind=measure.RECEIPT_KIND,
        receipt_override=receipt_override,
    ).resolve(strict=True)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("schema") != measure.CHECK_SCHEMA or metadata.get("status") != "accepted":
        raise ValueError("official profile requires an accepted independent check")
    receipt = require_receipt(
        receipt_path,
        kind=measure.RECEIPT_KIND,
        identity=metadata["validation_identity"],
    )
    return directory, metadata, receipt


@contextmanager
def instrument_stages(backend, scopes):
    """Add host scopes to the model's existing callback without tensor operations."""
    forward = backend._forward

    def annotated(session, ids, **kwargs):
        if kwargs.get("scope") is not None:
            raise ValueError("official profile owns the model scope callback")
        scopes._chunk_index = -1
        scopes._forward_start, scopes._forward_tokens = session.length, len(ids)
        scopes._forward_chunk = kwargs.get("chunk_size") or len(ids)
        segment = "history" if session.length == 0 else "candidate"
        with scopes.context(segment=segment), scopes("forward"):
            try:
                return forward(session, ids, **{**kwargs, "scope": scopes.model_scope})
            finally:
                scopes.close_chunk()

    with patch.object(backend, "_forward", annotated):
        yield


def main(argv=None):
    args = parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("invalid run ID")
    if not args.nsys:
        raise ValueError("official profile requires --nsys and CUDA profiler capture ranges")
    reference, checked, receipt = load_reference(
        args.reference_run, receipt_override=args.receipt_override
    )
    receipt_path = Path(receipt["receipt_path"])
    config = checked["config"]
    target = args.output_dir or measure.EXPERIMENT / "output/data" / args.run_id
    if target.exists() or args.run_id == checked["run_id"]:
        raise FileExistsError(target)
    output = Path(tempfile.mkdtemp(prefix=f"official-profile-{args.run_id}-"))
    print(f"temporary output: {output}", flush=True)
    shutil.copytree(reference / "workload", output / "workload")
    requests = [
        json.loads(line) for line in (output / "workload/requests.jsonl").read_text().splitlines()
    ]
    workload_manifest = json.loads((output / "workload/workload.json").read_text())
    from types import SimpleNamespace

    workload = SimpleNamespace(requests=requests, manifest=workload_manifest)
    metadata = {
        "schema": SCHEMA,
        "mode": "bench",
        "run_id": args.run_id,
        "status": "running",
        "config": config,
        "checkpoint": checked["checkpoint"],
        "workload_sha256": checked["workload_sha256"],
        "reference_run": str(reference),
        "reference_run_id": checked["run_id"],
        "reference_receipt_sha256": measure.digest(receipt_path),
        "captures": [],
        "warmup_traces": {},
        "cases": [],
        "identity_verifications": [],
        "measurement_boundary": "intrusive profile only; no formal latency samples",
        "instrumentation_boundary": "existing model stage callbacks plus NVTX/host timestamps; no added CUDA events, reductions or transfer",
        "graph_boundary": "node-level request tracing; no standalone operator timing or graph matrix FLOP claim",
        "started_unix": time.time(),
    }
    backend = None
    failure = None
    try:
        import torch

        from evaluation.provenance import backend_provenance
        from experiments.nosa_motivation.src.cpu_environment import (
            cpu_environment,
            finish_cpu_environment,
        )
        from experiments.nosa_motivation.src.provenance import (
            loaded_native_artifacts,
            verify_native_artifacts,
        )
        from models.deepseek_v32.execution.official import build_official_backend
        from serving.persistent import PersistentGRRunner

        device = torch.device(args.device)
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
            raise RuntimeError("official profiling requires Hopper")
        torch.cuda.set_device(device)
        os.environ.setdefault("CXLDSAGR_SM90_BACKEND", "native")
        metadata["precision_settings"] = measure.configure_precision(torch)
        metadata["precision_policy"] = measure.PRECISION_POLICY
        (
            metadata["execution_environment"],
            metadata["profiling_environment"],
            metadata["nsys_injection"],
        ) = profile_environment()
        metadata["source_sha256"] = measure.snapshot_sources(output)
        metadata["backend_provenance"] = backend_provenance()
        props = torch.cuda.get_device_properties(device)
        metadata["hardware"] = {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "name": props.name,
            "uuid": str(props.uuid),
            "total_memory": props.total_memory,
            "sm_count": props.multi_processor_count,
            "cpu_environment": cpu_environment(torch),
        }
        model_path = Path(checked["checkpoint"]["path"])
        current_checkpoint = {
            "path": str(model_path.resolve()),
            "files": {
                path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                for path in sorted(model_path.glob("*.safetensors"))
            },
            "identity_boundary": "checkpoint path and shard stat inventory, not weights hashes",
        }
        if current_checkpoint != checked["checkpoint"]:
            raise ValueError("profile checkpoint differs from check")
        backend = build_official_backend(
            model_path,
            scheme="hbm",
            device=device,
            num_layers=config["layers"],
            chunk_size=config["chunk_size"],
            sparse_pool_tokens=config["sparse_pool_tokens"],
            host_arena_tokens=config["host_arena_tokens"],
            workspace_query_tokens=config["workspace_query_tokens"],
            enable_compute_graphs=measure.normalize_compute_graphs(config),
        )
        metadata["official_provenance"] = backend.pipeline.provenance()
        metadata["official_artifact_manifest_sha256"] = measure.snapshot_official(
            metadata["official_provenance"], output
        )
        begin_validation(metadata, output, receipt_path, measure.RECEIPT_KIND)
        runner_type = partial(PersistentGRRunner, native_token_validation=True)
        all_calls, comparisons = [], []
        schemes = [args.scheme] if args.scheme else list(measure.SCHEMES)
        for scheme in schemes:
            backend.configure_scheme(scheme)
            metadata["warmup_traces"][scheme] = measure.motivation.warmup(
                backend, workload, config, runner_type
            )
            measure.verify_identities(backend, output, metadata, f"after_{scheme}_warmup")
            gc.collect()
            torch.cuda.empty_cache()
            with runner_type(
                backend, resource_limits=measure.motivation.resource_limits(config)
            ) as runner:
                native = loaded_native_artifacts()
                record_runtime(metadata, scheme, backend, native, runner.token_validation_identity)
                for request in requests[: config["num_users"] + 1]:
                    index = request["request_id"]
                    phase = (
                        "cold"
                        if index == 0
                        else "revisit"
                        if index == config["num_users"]
                        else "prepare"
                    )
                    before = backend.describe().get("compute_graphs")
                    if phase == "prepare":
                        result = runner.execute(request)
                    else:
                        capture = len(metadata["captures"]) + 1
                        scopes = Scopes(scheme, phase, capture)
                        backend.synchronize()
                        with (
                            capture_range(torch.cuda.cudart()),
                            instrument_stages(backend, scopes),
                            scopes("request", request_id=index),
                        ):
                            result = runner.execute(request)
                        if scopes.active:
                            raise AssertionError("unclosed official profile scope")
                        all_calls.extend(scopes.calls)
                        metadata["captures"].append(
                            {
                                "capture_index": capture,
                                "scheme": scheme,
                                "phase": phase,
                                "request_id": index,
                                "sqlite": f"capture_{capture}.sqlite",
                                "diagnostic_runner_latency_ms": result.metrics["latency_ms"],
                            }
                        )
                    measure.motivation.check_request(request, result.metrics, config)
                    measure.motivation.check_compute_graph_replays(
                        before, backend.describe().get("compute_graphs"), config, result.metrics
                    )
                    expected = torch.load(
                        reference / f"numerical/hbm/{index:06d}.pt",
                        weights_only=True,
                        map_location="cpu",
                    )
                    payload = measure.motivation.tensor_payload(backend, result, request, config)
                    if payload["input_sha256"] != expected["input_sha256"]:
                        raise ValueError("profile request differs from checked input")
                    comparison = {
                        name: measure.numerical_comparison(payload[name], expected[name], name)
                        for name in ("hidden", "logits")
                    }
                    measure.require_numerical({"request_id": index, "numerical": comparison})
                    comparisons.append(
                        {"scheme": scheme, "request_id": index, "numerical": comparison}
                    )
                    del result, payload, expected
                verify_native_artifacts(native)
                metadata["cases"].append(
                    {
                        "scheme": scheme,
                        "backend": backend.describe(),
                        "requests": config["num_users"] + 1,
                    }
                )
            backend.close()
        measure.verify_identities(backend, output, metadata, "final")
        finish_cpu_environment(metadata, torch)
        if profile_environment() != (
            metadata["execution_environment"],
            metadata["profiling_environment"],
            metadata["nsys_injection"],
        ):
            raise RuntimeError("execution environment or Nsight injection changed during profile")
        require_receipt(
            receipt_path,
            kind=measure.RECEIPT_KIND,
            identity=checked["validation_identity"],
        )
        metadata.update(status="accepted", completed_unix=time.time())
        finish_evidence(output, metadata, all_calls, comparisons)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output), str(target))
        print(f"accepted diagnostic official profile: {target}", flush=True)
    except BaseException as error:
        failure = error
        metadata.update(status="failed", error=repr(error))
        measure.write_json(output / "metadata.json", metadata)
        print(f"failed profile retained outside experiments: {output}", flush=True)
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
            except BaseException as cleanup_error:
                if failure is not None:
                    raise BaseExceptionGroup(
                        "official profile and resource cleanup failed", [failure, cleanup_error]
                    ) from None
                raise


if __name__ == "__main__":
    main()
