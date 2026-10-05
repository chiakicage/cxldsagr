"""Run four fixed P/NH NOSA methods on independent caches and one cyclic GR trace."""

from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

from cache.allocator.snapshot import runtime_info as allocator_snapshot_runtime_info
from evaluation import pool_scan_provenance as pool_scan
from experiments.nosa_motivation.src.config import (
    BACKEND_SCHEMES,
    BENCH_SCHEMA,
    CHECK_SCHEMA,
    EXPERIMENT,
    METHODS,
    WARMUP_POLICY,
    build_workload,
    configuration,
    parser,
    resource_limits,
)
from experiments.nosa_motivation.src.cpu_environment import finish_cpu_environment
from experiments.nosa_motivation.src.provenance import (
    checkpoint_inventory,
    digest,
    loaded_native_artifacts,
    memory_sample,
    native_build_identity,
    publish_directories,
    runtime_environment,
    snapshot_sources,
    verify_native_artifacts,
    verify_native_build_identity,
    verify_source_snapshot,
    write_json,
)
from experiments.nosa_motivation.src.validation import (
    begin_validation,
    execution_environment,
    publish_receipt,
    record_runtime,
    validate_mode_arguments,
    without_performance,
)

RECEIPT_KIND = "nosa-motivation-full-trace-v1"


def precision_settings(torch):
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
    }


def compare_hidden(actual, expected, config):
    import torch

    if actual.ndim != 2 or actual.shape != expected.shape or actual.dtype != torch.bfloat16:
        raise AssertionError("hidden comparison requires matching full BF16 candidate tensors")
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise AssertionError("candidate hidden states must be finite")
    torch.testing.assert_close(
        actual,
        expected,
        atol=config["numerical_atol"],
        rtol=config["numerical_rtol"],
    )
    difference = (actual.float() - expected.float()).abs()
    return {
        "passed": True,
        "exact": torch.equal(actual, expected),
        "elements": actual.numel(),
        "max_abs": difference.max().item(),
        "mean_abs": difference.mean().item(),
        "atol": config["numerical_atol"],
        "rtol": config["numerical_rtol"],
    }


def check_resource_plan(method, plan, config):
    metadata = plan.metadata
    if metadata.get("sparse_pool_tokens") != config["sparse_pool_tokens"]:
        raise AssertionError("backend did not apply the requested finite P capacity")
    if metadata.get("candidate_persistence") != "gpu_transient":
        raise AssertionError("fixed P/NH candidates must execute in transient GPU storage")
    if method == "hbm":
        if plan.hbm_tokens != config["sparse_pool_tokens"] or plan.host_pages:
            raise AssertionError("HBM-only admission must use exactly P retained history tokens")
    elif (
        metadata.get("host_arena_tokens") != config["host_arena_tokens"]
        or plan.host_pages != config["host_arena_tokens"] // 64
        or plan.hbm_tokens
    ):
        raise AssertionError("offload admission must use the exact NH history-page quota")


def check_diagnostics(method, diagnostics, config):
    if diagnostics.get("candidate_persistence") != "gpu_transient":
        raise AssertionError("candidate lifecycle was not GPU transient")
    if diagnostics.get("retained_length") != config["history_tokens"]:
        raise AssertionError("request did not retain exactly the stable history")
    if diagnostics.get("transfer_metrics_status") != "complete":
        raise AssertionError("request transfer counters are incomplete")
    for name in ("host_to_device_bytes", "device_to_host_bytes"):
        values = [diagnostics.get(f"{prefix}{name}") for prefix in ("", "prefix_", "candidate_")]
        if any(type(value) is not int or value < 0 for value in values):
            raise AssertionError(f"missing or invalid transfer accounting: {name}")
        if values[0] != values[1] + values[2]:
            raise AssertionError(f"prefix and candidate transfers do not sum: {name}")
        if method == "hbm" and any(values):
            raise AssertionError("HBM-only must not transfer historical KV to or from host")
    if diagnostics["candidate_device_to_host_bytes"]:
        raise AssertionError("candidate KV must not be persisted to DRAM")


def check_request(request, row, config):
    method = row["method"]
    if method not in METHODS:
        raise AssertionError("unknown measured method")
    index, users = request["request_id"], config["num_users"]
    visit = index // users
    resident_users = (
        config["sparse_pool_tokens"] if method == "hbm" else config["host_arena_tokens"]
    ) // config["padded_history_tokens"]
    expected = {
        "scheme": BACKEND_SCHEMES[method],
        "request_id": index,
        "user_id": request["user_id"],
        "input_sha256": request["input_sha256"],
        "visit_index": visit,
        "is_revisit": visit > 0,
        "prefix_cache_hit": visit > 0 and resident_users >= users,
        "stable_prefix_tokens": config["history_tokens"],
        "candidate_suffix_tokens": config["candidate_tokens"],
        "resource_mode": "fixed_pools",
        "hbm_budget_bytes": None,
        "dram_budget_bytes": None,
        "cached_users": min(index + 1, users, resident_users),
    }
    for name, value in expected.items():
        if row.get(name) != value:
            raise AssertionError(f"request {index}: {name}={row.get(name)!r}; expected {value!r}")
    check_diagnostics(method, row.get("cache_diagnostics", {}), config)


def validate_warmup(method, rows, config, requests):
    indices = config["warmup_request_indices"]
    if config["warmup_policy"] != WARMUP_POLICY or indices != [0, 1, config["num_users"]]:
        raise AssertionError("invalid warmup policy")
    if len(rows) != len(indices):
        raise AssertionError("incomplete warmup")
    for order, (row, index) in enumerate(zip(rows, indices, strict=True)):
        for field in ("request_id", "user_id", "input_sha256"):
            if row.get(field) != requests[index][field]:
                raise AssertionError("warmup request identity differs from the workload")
        if row.get("visit_index") != int(order == 2):
            raise AssertionError("warmup confused visit identity with residency")
        expected_hit = order == 2 and (
            method != "hbm" or config["sparse_pool_tokens"] >= 2 * config["padded_history_tokens"]
        )
        if row.get("prefix_cache_hit") != expected_hit:
            raise AssertionError("warmup history admission differs from fixed pool semantics")
        check_diagnostics(method, row["cache_diagnostics"], config)
    if method != "hbm" and config["sparse_pool_tokens"] == config["padded_history_tokens"]:
        recalled = rows[-1]["cache_diagnostics"].get("candidate_main_kv_host_to_device_bytes")
        if type(recalled) is not int or recalled <= 0:
            raise AssertionError(
                "offload warmup did not exercise main-KV host recall after another user"
            )


def backend_factory(model, method, config):
    from models.nosa.execution.fixed import NosaFixedServingBackend

    backend = NosaFixedServingBackend(
        model,
        BACKEND_SCHEMES[method],
        chunk_size=config["chunk_size"],
        sparse_pool_tokens=config["sparse_pool_tokens"],
        host_arena_tokens=config["host_arena_tokens"],
    )
    if config["enable_compute_graphs"]:
        backend.enable_compute_graphs(graph_query_sizes(config))
    return backend


def graph_query_sizes(config):
    history, chunk = config["history_tokens"], config["chunk_size"]
    return tuple(sorted({config["candidate_tokens"], min(history, chunk), history % chunk} - {0}))


def check_graph_replays(before, after, config, metrics):
    if not config["enable_compute_graphs"]:
        return
    expected_graphs = 2 * config["layers"] * len(graph_query_sizes(config))
    for state in (before, after):
        if (
            not state
            or not state.get("enabled")
            or state.get("eager_fallbacks") != 0
            or state.get("graphs") != expected_graphs
            or tuple(state.get("query_sizes", ())) != graph_query_sizes(config)
            or not state.get("policy_revision")
            or not state.get("finite_validation")
        ):
            raise AssertionError("compute graphs do not cover the complete measured workload")
    stable_fields = (
        "policy_revision",
        "finite_validation",
        "static_storage_bytes",
        "static_allocated_bytes",
        "private_reserved_bytes",
        "chosen_private_limit_bytes",
        "static_allocation_limit_bytes",
    )
    if any(before.get(name) != after.get(name) for name in stable_fields):
        raise AssertionError("compute graph policy or reserved storage changed during a request")
    batches = 1
    if not metrics["prefix_cache_hit"]:
        batches += (config["history_tokens"] + config["chunk_size"] - 1) // config["chunk_size"]
    for key in ("project_replays", "finish_replays"):
        if after[key] - before[key] != config["layers"] * batches:
            raise AssertionError(f"incomplete compute graph replay coverage: {key}")


def effective_work(model_config, config, metrics):
    from experiments.nosa_motivation.src.flops import matrix_flops, mfu_percent

    candidate = matrix_flops(
        model_config,
        config["history_tokens"],
        config["candidate_tokens"],
        config["candidate_tokens"],
    )
    prefix = (
        {name: 0 for name in candidate}
        if metrics["prefix_cache_hit"]
        else matrix_flops(model_config, 0, config["history_tokens"], config["chunk_size"])
    )
    total = {name: candidate[name] + prefix[name] for name in candidate}
    return {
        "prefix_flops": prefix,
        "candidate_flops": candidate,
        "request_flops": total,
        "request_wall_mfu_pct": mfu_percent(
            total, metrics["latency_ms"], config["peak_bf16_tflops"]
        ),
        "candidate_wall_mfu_pct": mfu_percent(
            candidate, metrics["extend_ms"], config["peak_bf16_tflops"]
        ),
        "peak_bf16_tflops": config["peak_bf16_tflops"],
        "definition": (
            "Useful forward matrix FLOPs: all projections/CIS delta, causal compressed-score "
            "QK, and full NOSA sparse causal QK/PV. No LM head, backward pass, padded work, "
            "repair, or recomputation is added to the numerator. Denominator is synchronized "
            "runner wall time. API proximity remains a separate measured comparison."
        ),
    }


def warmup(backend, method, workload, config, runner_type):
    rows = []
    try:
        with runner_type(backend, resource_limits=resource_limits(config)) as runner:
            check_resource_plan(method, runner.resource_plan, config)
            for index in config["warmup_request_indices"]:
                result = runner.execute(workload.requests[index])
                rows.append(result.metrics)
                del result
        validate_warmup(method, rows, config, workload.requests)
        return rows
    finally:
        backend.close()


def run_case(backend, method, workload, config, output, metadata, observe, runner_type):
    import torch

    checking = metadata.get("mode", "check") == "check"
    numerical_dir = output / "numerical" / method
    if checking:
        numerical_dir.mkdir(parents=True)
    observe(f"{method}/before_shared_allocation")
    # Each method has already warmed its complete prefix/candidate paths. New
    # libraries loaded by a later method's warmup are legitimate; this snapshot
    # freezes only the following measured method, not the pre-warmup process.
    native_before = loaded_native_artifacts()
    pool_scan_before = pool_scan.snapshot()
    try:
        with runner_type(backend, resource_limits=resource_limits(config)) as runner:
            if len(runner.pool):
                raise AssertionError(
                    "each measured method must start from an independent empty cache"
                )
            plan = runner.resource_plan
            check_resource_plan(method, plan, config)
            if "validation_identity" in metadata:
                record_runtime(
                    metadata,
                    method,
                    backend,
                    native_before,
                    runner.token_validation_identity,
                    allocator_snapshot_runtime_info(),
                )
            observe(f"{method}/after_shared_allocation")
            for request in workload.requests:
                index = request["request_id"]
                metadata.update(stage="measuring", active_method=method, active_request=index)
                write_json(output / "metadata.json", metadata)
                graph_before = backend.describe().get("compute_graphs")
                result = runner.execute(request)
                graph_after = backend.describe().get("compute_graphs")
                row = {
                    **result.metrics,
                    "method": method,
                    "run_id": metadata["run_id"],
                    "workload_sha256": metadata["workload_sha256"],
                    "memory_after": observe(f"{method}/after_request_{index}"),
                }
                check_request(request, row, config)
                check_graph_replays(graph_before, graph_after, config, row)
                row["compute_graphs"] = {"before": graph_before, "after": graph_after}
                if tuple(result.hidden.shape) != (
                    config["candidate_tokens"],
                    model_hidden_size(backend),
                ):
                    raise AssertionError("model did not return all candidate hidden states")
                if checking:
                    hidden = result.hidden.detach().cpu()
                    expected = (
                        {"hidden": hidden, "input_sha256": request["input_sha256"]}
                        if method == "hbm"
                        else torch.load(
                            output / "numerical/hbm" / f"{index:06d}.pt",
                            map_location="cpu",
                            weights_only=True,
                        )
                    )
                    if expected["input_sha256"] != request["input_sha256"]:
                        raise AssertionError("reference output uses different request tokens")
                    row["numerical"] = compare_hidden(hidden, expected["hidden"], config)
                    path = numerical_dir / f"{index:06d}.pt"
                    torch.save(
                        {
                            "request_id": index,
                            "input_sha256": request["input_sha256"],
                            "hidden": hidden,
                        },
                        path,
                    )
                    row.update(
                        output_file=str(path.relative_to(output)), output_sha256=digest(path)
                    )
                    row = without_performance(row)
                    detail = f"hidden_max_abs={row['numerical']['max_abs']:.6g}"
                    del hidden, expected
                else:
                    row["effective_work"] = effective_work(backend.model.config, config, row)
                    detail = f"latency_ms={row['latency_ms']:.3f}"
                del result
                with (output / "measurements.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(
                    f"{method} {index + 1}/{config['requests_per_method']}: "
                    f"revisit={row['is_revisit']} hit={row['prefix_cache_hit']} "
                    f"{detail}",
                    flush=True,
                )
            case = {
                "method": method,
                "scheme": backend.scheme,
                "requests": config["requests_per_method"],
                "started_empty": True,
                "warmup_requests": len(metadata["warmup_traces"][method]),
                "resource_plan": dict(plan.metadata),
                "shared_reservation": {"hbm": plan.shared.hbm, "dram": plan.shared.dram},
                "admission_host_pages": plan.host_pages,
                "admission_hbm_tokens": plan.hbm_tokens,
                "token_validation": runner.token_validation_identity,
                "allocator_snapshot": allocator_snapshot_runtime_info(),
                "backend": backend.describe(),
                "native_artifacts_before": native_before,
                "native_artifacts_after": verify_native_artifacts(native_before),
                "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(backend.device),
                "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(backend.device),
            }
        case["pool_referrers"] = pool_scan.finish_case(pool_scan_before)
        return case
    finally:
        backend.close()


def model_hidden_size(backend):
    return backend.model.config.hidden_size


def main(argv=None):
    args = parser().parse_args(argv)
    validate_mode_arguments(args)
    config = configuration(args)
    target = args.output_dir or (
        Path(tempfile.gettempdir()) / "cxldsagr-checks/nosa_motivation" / args.run_id
        if args.mode == "check"
        else EXPERIMENT / "output/data" / args.run_id
    )
    if args.mode == "check" and EXPERIMENT.parent.resolve() in target.resolve().parents:
        raise ValueError("correctness evidence belongs outside experiments; choose --output-dir")
    if target.exists():
        raise FileExistsError(target)
    output = Path(tempfile.mkdtemp(prefix=f"nosa-motivation-{args.run_id}-"))
    print(f"temporary output: {output}", flush=True)
    metadata = {
        "schema": CHECK_SCHEMA if args.mode == "check" else BENCH_SCHEMA,
        "mode": args.mode,
        "execution_environment": execution_environment(),
        "run_id": args.run_id,
        "status": "running",
        "config": config,
        "cases": [],
        "warmup_traces": {},
        "started_unix": time.time(),
        "measurement_boundary": (
            "Synchronized runner wall latency includes token validation/transfer, admission, "
            "eviction, independent sparse history construction on miss, all candidate hidden "
            "states, and candidate discard. Model loading, request generation, warmup, counters "
            "and reports are outside latency. Bench performs no full-output CPU copies, "
            "numerical comparison or output saving between requests. Check publishes no timings."
        ),
        "diagnostics_boundary": (
            "Transfer counters report prefix and candidate KV/indexer payload separately; "
            "they are not physical PCIe traffic. Whole-user prefix hits differ from HBM page hits."
        ),
        "memory_boundary": (
            "Peaks reset after warmup cleanup and before each fresh shared allocation. "
            "Allocated/reserved peaks include weights and the full measured trace. "
            "Device-used observations are boundary samples, not device or process peaks. "
            "P/NH are capacities; no byte subbudget or observed allocator gap is subtracted."
        ),
        "model_boundary": (
            "Complete 32-layer NOSA checkpoint with full NOSA sparse policy and normalized "
            "candidate hidden output. No LM head; the DeepSeek motivation experiment also "
            "executes its last-token LM head and uses a checkpoint workload substitute. "
            "Synthetic cyclic GR input does not establish task quality or representativeness."
        ),
        "performance_acceptance": {
            "raw_matrix_api_mfu_close": (
                "pending independent matching-shape matrix API timing for complete useful work"
            ),
            "compute_api_mfu": (
                "pending separate whole-compute API timing including norm, activation and helpers; "
                "this broader denominator cannot establish raw-matrix API proximity"
            ),
            "compute_io_dominance": "pending complete CPU/CUDA timeline attribution",
            "async_internal_overlap": "pending actual copy/compute work intervals",
        },
    }
    try:
        from functools import partial

        import torch

        from models.nosa.model import NosaForCausalLM
        from serving.persistent import PersistentGRRunner

        runner_type = partial(PersistentGRRunner, native_token_validation=True)
        device = torch.device(args.device)
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
            raise RuntimeError("formal NOSA motivation measurements require SM90/Hopper")
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        os.environ.setdefault("CXLDSAGR_SM90_BACKEND", "native")
        metadata["execution_environment"] = execution_environment()
        metadata.update(
            precision_settings=precision_settings(torch),
            source_sha256=snapshot_sources(output),
            hardware=runtime_environment(torch, device),
            checkpoint=checkpoint_inventory(args.model_path),
            native_provenance={"build_before": native_build_identity()},
        )
        samples = []

        def observe(stage):
            sample = memory_sample(torch, device, stage)
            samples.append(sample)
            write_json(output / "memory.json", samples)
            return sample

        workload = build_workload(config, args.model_path)
        workload.write(output / "workload")
        metadata["workload_sha256"] = workload.manifest["workload_sha256"]
        observe("before_model_loading")
        model = NosaForCausalLM.from_pretrained(
            args.model_path,
            device=device,
            dtype=torch.bfloat16,
            attention_mode="sparse",
            sparse_backend="auto",
        )
        if (
            model.config.num_hidden_layers,
            len(model.model.layers),
            model.config.num_attention_heads,
            model.config.num_key_value_heads,
            model.config.head_dim,
        ) != (32, 32, 32, 2, 128) or model.ignored_checkpoint_keys:
            raise AssertionError("measurement requires all NOSA checkpoint layers and parameters")
        model.config = replace(
            model.config,
            max_position_embeddings=config["history_tokens"] + config["candidate_tokens"],
        )
        metadata["model_config"] = asdict(model.config)
        metadata["loaded_parameters"] = sum(value.numel() for value in model.parameters())
        torch.cuda.synchronize(device)
        metadata["model_loaded_memory"] = observe("after_model_loading")
        begin_validation(metadata, output, args.validation_receipt, RECEIPT_KIND)
        for method in METHODS:
            metadata.update(stage="warmup", active_method=method)
            write_json(output / "metadata.json", metadata)
            metadata["warmup_traces"][method] = warmup(
                backend_factory(model, method, config), method, workload, config, runner_type
            )
            if args.mode == "check":
                metadata["warmup_traces"][method] = without_performance(
                    metadata["warmup_traces"][method]
                )
            verify_native_build_identity(metadata["native_provenance"]["build_before"])
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            observe(f"{method}/before_graph_preparation_and_shared_allocation")
            metadata["cases"].append(
                run_case(
                    backend_factory(model, method, config),
                    method,
                    workload,
                    config,
                    output,
                    metadata,
                    observe,
                    runner_type,
                )
            )
            observe(f"{method}/after_cleanup")
            write_json(output / "metadata.json", metadata)
        verify_source_snapshot(output, check_current=True)
        if precision_settings(torch) != metadata["precision_settings"]:
            raise RuntimeError("matmul precision changed during measurement")
        if checkpoint_inventory(args.model_path) != metadata["checkpoint"]:
            raise RuntimeError("checkpoint identity changed during measurement")
        metadata["native_provenance"].update(
            build_after=verify_native_build_identity(metadata["native_provenance"]["build_before"]),
            artifacts_final=loaded_native_artifacts(),
        )
        finish_cpu_environment(metadata, torch)
        if execution_environment() != metadata["execution_environment"]:
            raise RuntimeError("execution environment changed during the run")
        metadata.update(status="accepted", stage="complete", completed_unix=time.time())
        write_json(output / "metadata.json", metadata)
        from experiments.nosa_motivation.src.report import audit_run, write_report

        if args.mode == "check":
            _, _, audit = audit_run(output)
            publish_receipt(metadata, output, RECEIPT_KIND, audit)
        else:
            write_report(output, output / "report")
        publish_directories({"data": output}, {"data": target})
        shutil.rmtree(output, ignore_errors=True)
        print(f"accepted four-method {args.mode} run: {target}", flush=True)
    except BaseException as error:
        metadata.update(status="failed", error=repr(error))
        write_json(output / "metadata.json", metadata)
        print(f"failed run diagnostics retained outside experiments: {output}", flush=True)
        raise


if __name__ == "__main__":
    main()
