"""Replay accepted first/revisit requests for separate CUDA timelines and internal work."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cache.allocator.snapshot import runtime_info as allocator_snapshot_runtime_info
from evaluation import pool_scan_provenance as pool_scan
from experiments.nosa_motivation.src.config import (
    BENCH_SCHEMA,
    EXPERIMENT,
    METHODS,
    resource_limits,
)
from experiments.nosa_motivation.src.cpu_environment import (
    finish_cpu_environment,
    require_matching_cpu_environment,
)
from experiments.nosa_motivation.src.matrix_baseline import benchmark_matrix_apis
from experiments.nosa_motivation.src.measure import (
    RECEIPT_KIND,
    backend_factory,
    check_graph_replays,
    check_request,
    check_resource_plan,
    compare_hidden,
    precision_settings,
    warmup,
)
from experiments.nosa_motivation.src.profile_hbm import analyze_trace as analyze_api_trace
from experiments.nosa_motivation.src.profile_hbm import api_scopes
from experiments.nosa_motivation.src.provenance import (
    checkpoint_inventory,
    digest,
    loaded_native_artifacts,
    native_build_identity,
    publish_directories,
    runtime_environment,
    snapshot_sources,
    verify_native_artifacts,
    verify_native_build_identity,
    verify_source_snapshot,
    write_json,
)
from experiments.nosa_motivation.src.report import audit_run, read_jsonl
from experiments.nosa_motivation.src.timeline import analyze_trace
from experiments.nosa_motivation.src.validation import reference_directory
from experiments.nosa_motivation.src.work_intervals import capture_candidate_work


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--run-id", required=True)
    command.add_argument(
        "--reference-run",
        required=True,
        type=Path,
        help="independent check directory, clean bench with receipt, or legacy run",
    )
    command.add_argument("--model-path", type=Path)
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--method", choices=METHODS)
    command.add_argument("--repeats", default=1, type=int)
    command.add_argument("--output-root", type=Path, default=EXPERIMENT / "output")
    return command


def runtime_sources(manifest):
    orchestration = {
        "experiments/nosa_motivation/src/config.py",
        "experiments/nosa_motivation/src/measure.py",
        # Historical snapshots retain these names; their saved reader must keep
        # auditing them. Current GR/evaluation files are included below as non-experiment code.
        "experiments/gr_serving/src/workload.py",
        "experiments/gr_serving/src/pool_scan_provenance.py",
    }
    return {
        key: value
        for key, value in manifest.items()
        if not key.startswith("experiments/") or key in orchestration
    }


@contextmanager
def module_scopes(model, backend):
    import torch

    from models.nosa import scoring
    from models.nosa.attention import NosaFixedAttention
    from models.nosa.indexer import NosaIndexer

    def wrap(function, name):
        def call(*args, **kwargs):
            with torch.profiler.record_function(name):
                return function(*args, **kwargs)

        return call

    with ExitStack() as stack:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                stack.enter_context(
                    patch.object(module, "forward", wrap(module.forward, f"nosa::matrix/{name}"))
                )
        for cls, name in ((NosaIndexer, "indexer"), (NosaFixedAttention, "sparse_attention")):
            stack.enter_context(patch.object(cls, "__call__", wrap(cls.__call__, f"nosa::{name}")))
        stack.enter_context(
            patch.object(
                backend, "session_metrics", wrap(backend.session_metrics, "nosa::diagnostics")
            )
        )
        stack.enter_context(
            patch.object(
                scoring, "cis_scores", wrap(scoring.cis_scores, "nosa::matrix/cis_projection")
            )
        )
        graphs = getattr(backend, "compute_graphs", None)
        if graphs is not None:
            for pair in graphs.pairs.values():
                for name in ("project", "finish"):
                    graph = getattr(pair, f"{name}_graph")
                    stack.enter_context(
                        patch.object(
                            graph,
                            "replay",
                            wrap(
                                graph.replay,
                                f"nosa::compute_graph_{name}/{pair.layer}/{pair.queries}",
                            ),
                        )
                    )
        yield


def compare_replay(result, request, method, config, reference_dir):
    import torch

    row = {**result.metrics, "method": method}
    check_request(request, row, config)
    expected = torch.load(
        reference_dir / "numerical" / method / f"{request['request_id']:06d}.pt",
        map_location="cpu",
        weights_only=True,
    )
    if expected["input_sha256"] != request["input_sha256"]:
        raise ValueError("profile replay input differs from reference")
    comparison = compare_hidden(result.hidden.detach().cpu(), expected["hidden"], config)
    if not comparison["exact"]:
        raise ValueError("profile replay must exactly match the same-method accepted output")
    return comparison


def replay(model, method, config, workload, reference, data, profiles, sample, *, internal):
    import torch

    from operators.nosa.attention.offload.api import estimate_trace_rows
    from serving.persistent import PersistentGRRunner

    backend = backend_factory(model, method, config)
    limits = resource_limits(config)
    if internal:
        limits["trace_capacity"] = estimate_trace_rows(
            config["history_tokens"] + config["candidate_tokens"],
            config["workspace_query_tokens"],
            model.config.num_key_value_heads,
        )
    selected = (0, config["num_users"])
    records = []
    native_before = loaded_native_artifacts()
    pool_scan_before = pool_scan.snapshot()
    try:
        with PersistentGRRunner(
            backend, resource_limits=limits, native_token_validation=True
        ) as runner:
            check_resource_plan(method, runner.resource_plan, config)
            for request in workload.requests[: selected[-1] + 1]:
                index = request["request_id"]
                graph_before = backend.describe().get("compute_graphs")
                if index not in selected:
                    result = runner.execute(request)
                    check_graph_replays(
                        graph_before,
                        backend.describe().get("compute_graphs"),
                        config,
                        result.metrics,
                    )
                    compare_replay(result, request, method, config, reference)
                    del result
                    continue
                label = f"{method}_{index:06d}_{sample}"
                if internal:
                    work = []
                    with capture_candidate_work(
                        backend, config, work, evidence_dir=data, label=label
                    ):
                        result = runner.execute(request)
                    if len(work) != config["layers"]:
                        raise ValueError("internal trace did not cover every candidate layer")
                    record = {
                        "request_id": index,
                        "method": method,
                        "sample": sample,
                        "mode": "internal_work",
                        "layers": work,
                        "extra_shared_trace_bytes": limits["trace_capacity"] * 4 * 8,
                        "boundary": (
                            "Separate diagnostic replay with per-layer synchronization and CPU "
                            "copies; its wall time is not a latency result. Page and stripe "
                            "globaltimer windows intersect softmax updates, a subset of math."
                        ),
                    }
                else:
                    root_name = f"nosa_request::{label}"
                    with (
                        api_scopes(backend),
                        module_scopes(model, backend),
                        torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as profiler,
                        torch.profiler.record_function(root_name),
                    ):
                        result = runner.execute(request)
                    path = profiles / f"{label}.json"
                    profiler.export_chrome_trace(str(path))
                    record = {
                        "request_id": index,
                        "method": method,
                        "sample": sample,
                        "mode": "timeline",
                        "trace_file": path.name,
                        "trace_sha256": digest(path),
                        "root_name": root_name,
                        "analysis": analyze_trace(json.loads(path.read_text()), root_name),
                        "api_analysis": analyze_api_trace(path),
                        "instrumented_runner_metrics": result.metrics,
                        "boundary": (
                            "The outer root covers runner.execute including post-latency "
                            "diagnostic reads; those reads have a dedicated diagnostics scope. "
                            "Instrumented wall time does not replace reference request latency."
                        ),
                    }
                record["numerical"] = compare_replay(result, request, method, config, reference)
                record["instrumented_runner_metrics"] = result.metrics
                output_path = data / f"{label}_{record['mode']}_hidden.pt"
                torch.save(result.hidden.detach().cpu(), output_path)
                record.update(output_file=output_path.name, output_sha256=digest(output_path))
                graph_after = backend.describe().get("compute_graphs")
                check_graph_replays(graph_before, graph_after, config, result.metrics)
                record["compute_graphs"] = {"before": graph_before, "after": graph_after}
                record["record_file"] = f"{label}_{record['mode']}.json"
                records.append(record)
                write_json(data / record["record_file"], record)
                print(f"profiled {label} {record['mode']}", flush=True)
                del result
            case = {
                "method": method,
                "sample": sample,
                "mode": "internal_work" if internal else "timeline",
                "token_validation": runner.token_validation_identity,
                "allocator_snapshot": allocator_snapshot_runtime_info(),
                "native_artifacts_before": native_before,
                "native_artifacts_after": verify_native_artifacts(native_before),
                "resource_plan": dict(runner.resource_plan.metadata),
                "admission_hbm_tokens": runner.resource_plan.hbm_tokens,
                "admission_host_pages": runner.resource_plan.host_pages,
                "shared_reservation": {
                    "hbm": runner.resource_plan.shared.hbm,
                    "dram": runner.resource_plan.shared.dram,
                },
                "backend": backend.describe(),
            }
        case["pool_referrers"] = pool_scan.finish_case(pool_scan_before)
        return records, case
    finally:
        backend.close()


def main(argv=None):
    args = parser().parse_args(argv)
    args.reference_run = reference_directory(
        args.reference_run, bench_schema=BENCH_SCHEMA, kind=RECEIPT_KIND
    )
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id) or args.repeats <= 0:
        raise ValueError("safe run ID and positive profile repeats required")
    targets = {name: args.output_root / name / args.run_id for name in ("data", "profile")}
    if any(path.exists() for path in targets.values()):
        raise FileExistsError("profile run ID already exists")
    reference, _, audit = audit_run(args.reference_run)
    if reference["run_id"] == args.run_id:
        raise ValueError("profiling requires a separate run ID")
    staging = Path(tempfile.mkdtemp(prefix=f"nosa-motivation-profile-{args.run_id}-"))
    data, profiles = staging / "data", staging / "profile"
    data.mkdir()
    profiles.mkdir()
    metadata = {
        "schema": "nosa-motivation-profile-v2",
        "run_id": args.run_id,
        "status": "running",
        "reference_run_id": reference["run_id"],
    }
    try:
        import torch

        from models.nosa.model import NosaForCausalLM
        from serving.persistent import PersistentGRRunner

        config = reference["config"]
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        if torch.cuda.get_device_capability(device) != (9, 0):
            raise RuntimeError("profile requires SM90/Hopper")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        os.environ.setdefault("CXLDSAGR_SM90_BACKEND", "native")
        source_id = snapshot_sources(data)
        current_manifest = json.loads((data / "source_manifest.json").read_text())
        original_manifest = json.loads((args.reference_run / "source_manifest.json").read_text())
        if runtime_sources(current_manifest) != runtime_sources(original_manifest):
            raise ValueError("profile runtime or orchestration differs from accepted measurement")
        checkpoint = args.model_path or Path(reference["checkpoint"]["path"])
        if checkpoint_inventory(checkpoint) != reference["checkpoint"]:
            raise ValueError("profile checkpoint differs from accepted measurement")
        if precision_settings(torch) != reference["precision_settings"]:
            raise ValueError("profile precision differs from accepted measurement")
        metadata.update(
            source_sha256=source_id,
            reference_source_sha256=reference["source_sha256"],
            reference_audit=audit,
            config=config,
            hardware=runtime_environment(torch, device),
            workload_sha256=reference["workload_sha256"],
            repeats=args.repeats,
            checkpoint=checkpoint_inventory(checkpoint),
            precision_settings=precision_settings(torch),
            native_provenance={"build_before": native_build_identity()},
        )
        require_matching_cpu_environment(metadata["hardware"], reference["hardware"])
        if metadata["hardware"]["uuid"] != reference["hardware"]["uuid"]:
            raise ValueError("profile must use the same GPU UUID as the accepted measurement")
        if (
            metadata["native_provenance"]["build_before"]
            != reference["native_provenance"]["build_after"]
        ):
            raise ValueError("profile native build/dependency identity differs from reference")
        workload = SimpleNamespace(
            requests=read_jsonl(args.reference_run / "workload/requests.jsonl")
        )
        model = NosaForCausalLM.from_pretrained(
            checkpoint,
            device=device,
            dtype=torch.bfloat16,
            attention_mode="sparse",
            sparse_backend="auto",
        )
        model.config = replace(
            model.config,
            max_position_embeddings=config["history_tokens"] + config["candidate_tokens"],
        )
        methods = [args.method] if args.method else list(METHODS)
        records, cases = [], []
        metadata["methods"] = methods
        for method in methods:
            warmup(
                backend_factory(model, method, config),
                method,
                workload,
                config,
                partial(PersistentGRRunner, native_token_validation=True),
            )
            for sample in range(args.repeats):
                for internal in (
                    (False, True) if method in ("sync_sparse", "async_sparse") else (False,)
                ):
                    result, case = replay(
                        model,
                        method,
                        config,
                        workload,
                        args.reference_run,
                        data,
                        profiles,
                        sample,
                        internal=internal,
                    )
                    records.extend(result)
                    cases.append(case)
        verify_source_snapshot(data, check_current=True)
        if checkpoint_inventory(checkpoint) != metadata["checkpoint"]:
            raise ValueError("checkpoint changed during profile replay")
        if precision_settings(torch) != metadata["precision_settings"]:
            raise ValueError("matmul precision changed during profile replay")
        metadata["native_provenance"].update(
            build_after=verify_native_build_identity(metadata["native_provenance"]["build_before"]),
            artifacts_final=loaded_native_artifacts(),
        )
        # The independent raw APIs may load additional compute libraries. Keep
        # their inventory separate from the accepted model replay's binaries.
        native = metadata["native_provenance"]
        raw_path = data / "raw_matrix_api.json"
        raw_reference = benchmark_matrix_apis(
            model,
            config["history_tokens"],
            config["candidate_tokens"],
            config["chunk_size"],
            peak_tflops=config["peak_bf16_tflops"],
        )
        write_json(raw_path, raw_reference)
        metadata["matrix_reference"] = {
            "file": raw_path.name,
            "sha256": digest(raw_path),
            "native_provenance": {
                "build_before": native["build_after"],
                "build_after": verify_native_build_identity(native["build_after"]),
                "artifacts_before": native["artifacts_final"],
                "artifacts_after": verify_native_artifacts(
                    native["artifacts_final"], allow_additions=True
                ),
            },
        }
        verify_source_snapshot(data, check_current=True)
        if checkpoint_inventory(checkpoint) != metadata["checkpoint"]:
            raise ValueError("checkpoint changed during raw matrix benchmark")
        if precision_settings(torch) != metadata["precision_settings"]:
            raise ValueError("matmul precision changed during raw matrix benchmark")
        finish_cpu_environment(metadata, torch)
        metadata.update(status="validating", records=records, cases=cases)
        write_json(data / "metadata.json", metadata)
        from experiments.nosa_motivation.src.profile_audit import audit_profile

        metadata["acceptance"] = audit_profile(data, profiles, args.reference_run)
        metadata["status"] = "diagnostic_valid"
        write_json(data / "metadata.json", metadata)
        publish_directories({"data": data, "profile": profiles}, targets)
        shutil.rmtree(staging, ignore_errors=True)
        print(
            f"valid diagnostic profile; performance gates reported separately: {targets['data']}",
            flush=True,
        )
    except BaseException as error:
        metadata.update(status="failed", error=repr(error))
        if data.exists():
            write_json(data / "metadata.json", metadata)
        print(f"failed profile diagnostics retained outside experiments: {staging}", flush=True)
        raise


if __name__ == "__main__":
    main()
