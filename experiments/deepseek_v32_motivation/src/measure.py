"""Four fixed-pool DeepSeek GR policies on the same two-round user trace."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
import time
from functools import partial
from pathlib import Path

from experiments.nosa_motivation.src.validation import (
    add_mode_arguments,
    begin_validation,
    execution_environment,
    publish_receipt,
    record_runtime,
    validate_mode_arguments,
    without_performance,
)

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/deepseek_v32_motivation"
SCHEMES = ("hbm", "echo", "serial_sparse", "dense_prefetch")
SCHEMA = "deepseek-v32-motivation-v1"
CHECK_SCHEMA = "deepseek-v32-motivation-check-v1"
BENCH_SCHEMA = "deepseek-v32-motivation-bench-v1"
RECEIPT_KIND = "deepseek-v32-motivation-full-trace-v1"
WARMUP_POLICY = "two_users_then_first_user_host_recall_v1"
INDEXER_DISPATCH_POLICY = "echo_fused_only_if_history_residency_unproven_v1"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def checkpoint_inventory(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "files": {
            item.name: {"size": item.stat().st_size, "mtime_ns": item.stat().st_mtime_ns}
            for item in sorted(path.glob("*.safetensors"))
        },
        "identity_boundary": "checkpoint path and shard stat inventory, not weights hashes",
    }


def precision_settings(torch_module=None):
    """Record effective matmul policy without requiring CUDA initialization."""
    if torch_module is None:
        import torch as torch_module

    matmul = torch_module.backends.cuda.matmul
    settings = {"float32_matmul_precision": torch_module.get_float32_matmul_precision()}
    for label, owner, name in (
        ("cuda_matmul_allow_tf32", matmul, "allow_tf32"),
        ("cuda_matmul_fp32_precision", matmul, "fp32_precision"),
        ("backends_fp32_precision", torch_module.backends, "fp32_precision"),
        ("cudnn_allow_tf32", torch_module.backends.cudnn, "allow_tf32"),
        ("cudnn_fp32_precision", torch_module.backends.cudnn, "fp32_precision"),
        ("bf16_reduced_precision_reduction", matmul, "allow_bf16_reduced_precision_reduction"),
        ("fp16_reduced_precision_reduction", matmul, "allow_fp16_reduced_precision_reduction"),
    ):
        try:
            settings[label] = getattr(owner, name)
        except (AttributeError, RuntimeError) as error:
            settings[label] = {"unavailable": str(error)}
    return settings


def configure_precision(torch_module):
    torch_module.set_float32_matmul_precision("highest")
    torch_module.backends.cuda.matmul.allow_tf32 = False
    settings = precision_settings(torch_module)
    if settings["cuda_matmul_allow_tf32"] is not False:
        raise RuntimeError("formal measurements require CUDA matmul TF32 to be disabled")
    return settings


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    add_mode_arguments(command)
    command.add_argument("--run-id", required=True)
    command.add_argument("--output-dir", type=Path)
    command.add_argument("--model-path", type=Path, default=Path("/preset-models"))
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--num-users", type=int, default=16)
    command.add_argument("--rounds", type=int, default=2)
    command.add_argument("--history-tokens", type=int, default=65536)
    command.add_argument("--candidate-tokens", type=int, default=128)
    command.add_argument("--chunk-size", type=int, default=1024)
    command.add_argument("--sparse-pool-tokens", type=int, default=65536)
    command.add_argument("--host-arena-tokens", type=int, default=16777216)
    command.add_argument(
        "--compute-graphs",
        action="store_true",
        help="prepare pure-compute CUDA Graphs before requests; require complete replay coverage",
    )
    command.add_argument("--seed", type=int, default=42)
    return command


def configuration(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run ID must contain letters, digits, underscores or hyphens")
    names = (
        "num_users",
        "rounds",
        "history_tokens",
        "candidate_tokens",
        "chunk_size",
        "sparse_pool_tokens",
        "host_arena_tokens",
    )
    if any(getattr(args, key) <= 0 for key in names) or args.seed < 0 or args.rounds < 2:
        raise ValueError(
            "positive capacities, at least two rounds, and a nonnegative seed required"
        )
    if args.num_users < 2:
        raise ValueError("host-miss warmup requires at least two users")
    padded_history = (args.history_tokens + 63) // 64 * 64
    if args.sparse_pool_tokens % 64 or args.host_arena_tokens % 64:
        raise ValueError("P and NH must be multiples of 64")
    if args.sparse_pool_tokens < max(padded_history, args.chunk_size, args.candidate_tokens, 2048):
        raise ValueError("P must hold one history and the complete query/top-k working set")
    if args.host_arena_tokens < padded_history * args.num_users:
        raise ValueError("NH must retain the requested users; this experiment does not fill NH")
    return {
        **{name: getattr(args, name) for name in names},
        "seed": args.seed,
        "requests_per_scheme": args.num_users * args.rounds,
        "schemes": list(SCHEMES),
        "layers": 10,
        "warmup_requests_per_scheme": 3,
        "warmup_request_indices": [0, 1, args.num_users],
        "warmup_policy": WARMUP_POLICY,
        "indexer_dispatch_policy": INDEXER_DISPATCH_POLICY,
        "sampling": "sequential",
        "workspace_query_tokens": max(args.chunk_size, args.candidate_tokens),
        "padded_history_tokens": padded_history,
        "byte_subbudgets": None,
        "noncache_headroom_bytes": None,
        "enable_compute_graphs": args.compute_graphs,
    }


def resource_limits(config):
    return {
        "max_session_capacity": config["history_tokens"] + config["candidate_tokens"],
        "max_history_tokens": config["history_tokens"],
        "max_candidate_tokens": config["candidate_tokens"],
    }


def check_compute_graph_replays(before, after, config, metrics):
    """Require the complete actual history/candidate workload in both compute islands."""
    if not config.get("enable_compute_graphs", False):
        return
    for state in (before, after):
        if not state or not state.get("enabled") or not state.get("allocated"):
            raise ValueError("compute graph bank was not prepared before the request")
        if state.get("eager_fallbacks") != 0:
            raise ValueError("formal compute graph measurements cannot use eager fallback")
    history, chunk = config["history_tokens"], config["chunk_size"]
    history_batches = [chunk] * (history // chunk)
    if history % chunk:
        history_batches.append(history % chunk)
    candidates = [config["candidate_tokens"]]
    query_sizes = set(history_batches + candidates)
    executed = candidates if metrics["prefix_cache_hit"] else history_batches + candidates
    expected_keys = {(layer, count) for layer in range(config["layers"]) for count in query_sizes}
    states = []
    for state in (before, after):
        entries = state.get("graphs", [])
        keyed = {(entry["layer"], entry["queries"]): entry for entry in entries}
        if len(keyed) != len(entries) or set(keyed) != expected_keys:
            raise ValueError("compute graph bank does not cover every independent layer and batch")
        states.append(keyed)
    for key in expected_keys:
        expected = executed.count(key[1])
        for name in ("projection_replays", "finish_replays"):
            counts = [state[key].get(name) for state in states]
            if any(type(count) is not int or count < 0 for count in counts):
                raise ValueError("invalid compute graph replay counter")
            if counts[1] - counts[0] != expected:
                raise ValueError(f"incomplete compute graph replay coverage: {key} {name}")


def check_request(request, metrics, config):
    """Validate visit identity separately from cache residency on cyclic accesses."""
    index, users = request["request_id"], config["num_users"]
    visit = index // users
    resident_users = (
        config["sparse_pool_tokens"] // config["padded_history_tokens"]
        if metrics["scheme"] == "hbm"
        else config["host_arena_tokens"] // config["padded_history_tokens"]
    )
    expected_hit = visit > 0 and resident_users >= users
    expected = {
        "request_id": index,
        "user_id": request["user_id"],
        "visit_index": visit,
        "is_revisit": visit > 0,
        "prefix_cache_hit": expected_hit,
        "stable_prefix_tokens": config["history_tokens"],
        "candidate_suffix_tokens": config["candidate_tokens"],
        "resource_mode": "fixed_pools",
        "hbm_budget_bytes": None,
        "dram_budget_bytes": None,
        "cached_users": min(index + 1, users, resident_users),
    }
    for name, value in expected.items():
        if metrics.get(name) != value:
            raise AssertionError(
                f"request {index}: {name}={metrics.get(name)!r}, expected {value!r}"
            )
    diagnostics = metrics.get("cache_diagnostics", {})
    for name in ("host_to_device_bytes", "device_to_host_bytes"):
        value = diagnostics.get(name)
        if type(value) is not int or value < 0:
            raise AssertionError(f"request {index}: missing or invalid candidate {name}")
    if metrics["scheme"] == "hbm" and (
        diagnostics["host_to_device_bytes"] or diagnostics["device_to_host_bytes"]
    ):
        raise AssertionError("HBM-only baseline must not transfer cache records to/from host")


def snapshot_sources(output):
    from evaluation.provenance import source_snapshot

    source_snapshot(output, include_official=True)
    manifest = json.loads((output / "source_manifest.json").read_text())
    experiment_helpers = {
        "evaluation/provenance.py",
        "evaluation/validation.py",
        "GR/workload.py",
        "experiments/nosa_motivation/src/validation.py",
        "experiments/nosa_motivation/src/provenance.py",
        "experiments/nosa_motivation/src/cpu_environment.py",
        "experiments/deepseek_v32_echo_cache/src/capacity_probe.py",
        "experiments/deepseek_v32_echo_prefill/src/backend_provenance.py",
    }

    def executed_source(name):
        parts = Path(name).parts
        if "tests" in parts:
            return False
        if parts[0] == "models":
            return parts[1] == "deepseek_v32"
        if parts[0] == "operators":
            return len(parts) == 2 or parts[1] in ("common", "deepseek_v32")
        if parts[0] == "experiments":
            return name in experiment_helpers
        return True

    manifest = {name: value for name, value in manifest.items() if executed_source(name)}
    paths = [ROOT / "pyproject.toml", ROOT / "uv.lock"]
    paths.extend(ROOT / name for name in experiment_helpers)
    for directory in (EXPERIMENT / "src", EXPERIMENT / "scripts"):
        paths.extend(path for path in directory.rglob("*") if path.suffix in (".py", ".sh"))
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output / "source_manifest.json", manifest)
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def validate_warmup_trace(scheme, trace, config, requests):
    """Require a real host-recall path before a fixed-pool offload measurement."""
    indices = [0, 1, config["num_users"]]
    if (
        config["num_users"] < 2
        or config.get("warmup_policy") != WARMUP_POLICY
        or config.get("warmup_requests_per_scheme") != 3
        or config.get("warmup_request_indices") != indices
        or len(trace) != 3
    ):
        raise ValueError("warmup must execute two users then the first user's revisit")
    stages = ("first_user_first_visit", "second_user_first_visit", "first_user_revisit")
    for order, (row, index, stage) in enumerate(zip(trace, indices, stages, strict=True)):
        expected = requests[index]
        for name in ("request_id", "user_id", "input_sha256"):
            if row.get(name) != expected[name]:
                raise ValueError(f"warmup request identity differs: {name}")
        if (
            row.get("stage") != stage
            or row.get("visit_index") != (1 if order == 2 else 0)
            or row.get("is_revisit") != (order == 2)
            or row.get("retained_length") != config["history_tokens"]
            or row.get("candidate_persistence") != "gpu_transient"
        ):
            raise ValueError("warmup visit or candidate lifecycle is invalid")
        for name in (
            "host_to_device_bytes",
            "device_to_host_bytes",
            "prefetched_records",
            "recalled_records",
        ):
            if type(row.get(name)) is not int or row[name] < 0:
                raise ValueError(f"warmup is missing a valid candidate counter: {name}")
        if row["device_to_host_bytes"] != 0:
            raise ValueError("warmup candidate must remain GPU-only")
        if order < 2 and row.get("prefix_cache_hit") is not False:
            raise ValueError("the first two warmup requests must construct independent histories")
    if trace[0]["user_id"] == trace[1]["user_id"] or trace[0]["user_id"] != trace[2]["user_id"]:
        raise ValueError("warmup requires two distinct users and a first-user revisit")
    if scheme == "hbm":
        if any(row["host_to_device_bytes"] or row["recalled_records"] for row in trace):
            raise ValueError("HBM-only warmup must not read host KV")
        expected_hit = config["sparse_pool_tokens"] // config["padded_history_tokens"] >= 2
        if trace[2].get("prefix_cache_hit") != expected_hit:
            raise ValueError("HBM warmup revisit does not match the finite history quota")
    elif (
        trace[2].get("prefix_cache_hit") is not True
        or trace[2]["host_to_device_bytes"] <= 0
        or trace[2]["recalled_records"] <= 0
    ):
        raise ValueError("offload warmup did not exercise host recall before measurement")


def warmup(backend, workload, config, runner_type):
    """Exercise a retained-history host miss, then release all warmup cache resources."""
    trace = []
    stages = ("first_user_first_visit", "second_user_first_visit", "first_user_revisit")
    with runner_type(backend, resource_limits=resource_limits(config)) as runner:
        for index, stage in zip(config["warmup_request_indices"], stages, strict=True):
            result = runner.execute(workload.requests[index])
            metrics = result.metrics
            diagnostics = metrics["cache_diagnostics"]
            trace.append(
                {
                    "stage": stage,
                    **{
                        name: metrics[name]
                        for name in (
                            "request_id",
                            "user_id",
                            "visit_index",
                            "is_revisit",
                            "input_sha256",
                            "prefix_cache_hit",
                        )
                    },
                    **{
                        name: diagnostics[name]
                        for name in (
                            "candidate_persistence",
                            "retained_length",
                            "host_to_device_bytes",
                            "device_to_host_bytes",
                            "prefetched_records",
                            "recalled_records",
                        )
                    },
                }
            )
            del result
        validate_warmup_trace(backend.scheme, trace, config, workload.requests)
    backend.synchronize()
    backend.close()
    return trace


def tensor_payload(backend, result, request, config):
    import torch

    hidden = result.hidden.detach().cpu()
    logits = backend.last_logits.detach().cpu()
    if hidden.shape != (config["candidate_tokens"], backend.cfg.dim):
        raise AssertionError("missing candidate hidden states")
    if logits.shape != (1, backend.cfg.vocab_size):
        raise AssertionError("missing full-vocabulary last-token logits")
    if not bool(torch.isfinite(hidden).all() and torch.isfinite(logits).all()):
        raise AssertionError("nonfinite model output")
    return {
        "request_id": request["request_id"],
        "input_sha256": request["input_sha256"],
        "hidden": hidden,
        "logits": logits,
    }


def run_case(backend, workload, config, output, metadata, observe, runner_type):
    import torch

    from evaluation.provenance import numerical_comparison

    scheme = backend.scheme
    checking = metadata.get("mode", "check") == "check"
    numerical = output / "numerical" / scheme
    if checking:
        numerical.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats(backend.device)
    before_allocation = observe(f"{scheme}/before_cache_allocation")
    rows = []
    native_before = None
    with runner_type(backend, resource_limits=resource_limits(config)) as runner:
        if len(runner.pool):
            raise AssertionError("each measured scheme must start from an empty user cache")
        plan = runner.resource_plan
        if plan.metadata.get("sparse_pool_tokens") != config["sparse_pool_tokens"]:
            raise AssertionError("backend did not apply the requested finite P capacity")
        if (
            scheme != "hbm"
            and plan.metadata.get("host_arena_tokens") != config["host_arena_tokens"]
        ):
            raise AssertionError("backend did not apply the requested NH capacity")
        observe(f"{scheme}/after_cache_allocation")
        if "validation_identity" in metadata:
            from experiments.nosa_motivation.src.provenance import loaded_native_artifacts

            native_before = loaded_native_artifacts()
            record_runtime(
                metadata,
                scheme,
                backend,
                native_before,
                getattr(runner, "token_validation_identity", None),
            )
        for request in workload.requests:
            request_id = request["request_id"]
            metadata.update(stage="measuring", active_scheme=scheme, active_request=request_id)
            write_json(output / "metadata.json", metadata)
            before = observe(f"{scheme}/before_request_{request_id}")
            graphs_before = backend.describe().get("compute_graphs")
            result = runner.execute(request)
            graphs_after = backend.describe().get("compute_graphs")
            after = observe(f"{scheme}/after_request_{request_id}")
            check_request(request, result.metrics, config)
            check_compute_graph_replays(graphs_before, graphs_after, config, result.metrics)
            row = {
                **result.metrics,
                "run_id": metadata["run_id"],
                "round_index": request_id // config["num_users"],
                "workload_sha256": workload.manifest["workload_sha256"],
                "diagnostics_scope": "candidate_only",
                "memory_before": before,
                "memory_after": after,
                "compute_graphs": {"before": graphs_before, "after": graphs_after},
            }
            if checking:
                payload = tensor_payload(backend, result, request, config)
                expected = (
                    payload
                    if scheme == "hbm"
                    else torch.load(
                        output / "numerical/hbm" / f"{request_id:06d}.pt",
                        weights_only=True,
                        map_location="cpu",
                    )
                )
                if expected["input_sha256"] != payload["input_sha256"]:
                    raise AssertionError("HBM reference uses different request tokens")
                comparison = {
                    name: numerical_comparison(payload[name], expected[name], atol=0, rtol=0)
                    for name in ("hidden", "logits")
                }
                destination = numerical / f"{request_id:06d}.pt"
                torch.save(payload, destination)
                row.update(
                    numerical=comparison,
                    output_file=str(destination.relative_to(output)),
                    output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                )
                row = without_performance(row)
                detail = "numerical=passed"
                del payload, expected
            else:
                if result.hidden.shape != (config["candidate_tokens"], backend.cfg.dim):
                    raise AssertionError("missing candidate hidden states")
                if backend.last_logits.shape != (1, backend.cfg.vocab_size):
                    raise AssertionError("missing full-vocabulary last-token logits")
                detail = f"latency_ms={row['latency_ms']:.3f}"
            with (output / "measurements.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            rows.append(row)
            print(
                f"{scheme} {request_id + 1}/{config['requests_per_scheme']}: "
                f"user={request['user_id']} revisit={row['is_revisit']} "
                f"hit={row['prefix_cache_hit']} {detail}",
                flush=True,
            )
            del result
        case = {
            "scheme": scheme,
            "requests": len(rows),
            "warmup_requests": len(metadata["warmup_traces"][scheme]),
            "warmup_request_ids": [row["request_id"] for row in metadata["warmup_traces"][scheme]],
            "started_empty": True,
            "resource_plan": dict(plan.metadata),
            "token_validation": getattr(runner, "token_validation_identity", None),
            "shared_reservation": {"hbm": plan.shared.hbm, "dram": plan.shared.dram},
            "admission_page_capacity": plan.host_pages,
            "admission_hbm_token_capacity": plan.hbm_tokens,
            "backend": backend.describe(),
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(backend.device),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(backend.device),
            "before_cache_allocation": before_allocation,
        }
    backend.close()
    if native_before is not None:
        from experiments.nosa_motivation.src.provenance import verify_native_artifacts

        case["native_artifacts_before"] = native_before
        case["native_artifacts_after"] = verify_native_artifacts(native_before)
    observe(f"{scheme}/after_cleanup")
    return case


def main(argv=None):
    args = parser().parse_args(argv)
    validate_mode_arguments(args)
    config = configuration(args)
    target = args.output_dir or (
        Path(tempfile.gettempdir()) / "cxldsagr-checks/deepseek_v32_motivation" / args.run_id
        if args.mode == "check"
        else EXPERIMENT / "output/data" / args.run_id
    )
    if args.mode == "check" and EXPERIMENT.parent.resolve() in target.resolve().parents:
        raise ValueError("correctness evidence belongs outside experiments; choose --output-dir")
    if target.exists():
        raise FileExistsError(target)
    output = Path(tempfile.mkdtemp(prefix=f"deepseek-motivation-{args.run_id}-"))
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
            "Synchronized runner wall latency includes admission/eviction, history construction "
            "on a miss, all candidate hidden, last-token LM head and history cleanup. "
            "Loading, request generation, three warmup requests including host recall, "
            "compute graph preparation, "
            "diagnostics and report generation are outside latency. Bench performs no "
            "full-output CPU copies, numerical comparison or output saving between requests. "
            "Check publishes no timings."
        ),
        "diagnostics_boundary": (
            "Cache transfer/selection counters cover the candidate forward only; history "
            "prefill counters reset before candidate execution. Prefix cache hit is a separate "
            "whole-user history-retention metric."
        ),
        "memory_boundary": (
            "Each scheme resets CUDA allocator peaks after warmup resources are released, "
            "before fresh shared allocation. Peaks include loaded weights, shared allocation "
            "and the complete measured request trace. Per-request peak fields are cumulative "
            "within that scheme. Device free-memory samples are boundaries, not process peaks."
        ),
        "model_boundary": (
            "Ten independent dense checkpoint block copies replay source 0,1,2 inputs; "
            "not a trained ten-layer model or full DeepSeek V3.2. Synthetic GR inputs do not "
            "establish task quality or representativeness."
        ),
    }
    backend = None
    samples = []
    try:
        import torch

        from evaluation.provenance import (
            _git,
            backend_provenance,
            verify_source_snapshot,
        )
        from experiments.deepseek_v32_echo_cache.src.capacity_probe import memory_sample
        from GR.workload import WorkloadConfig, build_workload
        from models.deepseek_v32.serving_backend import DeepSeekServingBackend
        from serving.persistent import PersistentGRRunner

        runner_type = partial(PersistentGRRunner, native_token_validation=True)

        metadata["precision_settings"] = configure_precision(torch)
        metadata["precision_policy"] = "cuda_matmul_fp32_tf32_disabled_v1"
        device = torch.device(args.device)
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
            raise RuntimeError("this experiment requires one SM90/Hopper GPU")
        torch.cuda.set_device(device)
        os.environ.setdefault("CXLDSAGR_SM90_BACKEND", "native")
        metadata["execution_environment"] = execution_environment()
        metadata.update(
            source_sha256=snapshot_sources(output),
            git_revision=_git("rev-parse", "HEAD"),
            git_status=_git("status", "--short"),
            backend_provenance=backend_provenance(),
        )
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
        }
        from experiments.nosa_motivation.src.cpu_environment import cpu_environment

        metadata["hardware"]["cpu_environment"] = cpu_environment(torch)
        metadata["checkpoint"] = checkpoint_inventory(args.model_path)

        def observe(stage):
            sample = memory_sample(device, stage)
            if "cuda_observation_error" in sample:
                raise RuntimeError(sample["cuda_observation_error"])
            samples.append(sample)
            write_json(output / "memory.json", samples)
            return sample

        workload = build_workload(
            WorkloadConfig(
                model="deepseek_v32",
                num_users=config["num_users"],
                requests=config["requests_per_scheme"],
                history_tokens=config["history_tokens"],
                candidate_tokens=config["candidate_tokens"],
                seed=config["seed"],
                sampling="sequential",
                context_limit=config["history_tokens"] + config["candidate_tokens"],
            ),
            tokenizer=args.model_path,
        )
        workload.write(output / "workload")
        metadata["workload_sha256"] = workload.manifest["workload_sha256"]
        torch.cuda.reset_peak_memory_stats(device)
        observe("before_model_loading")
        backend = DeepSeekServingBackend(
            args.model_path,
            scheme="hbm",
            device=device,
            num_layers=config["layers"],
            chunk_size=config["chunk_size"],
            sparse_pool_tokens=config["sparse_pool_tokens"],
            host_arena_tokens=config["host_arena_tokens"],
            workspace_query_tokens=config["workspace_query_tokens"],
            enable_compute_graphs=config["enable_compute_graphs"],
        )
        backend.synchronize()
        metadata["model_dimensions"] = {
            "hidden": backend.cfg.dim,
            "vocabulary": backend.cfg.vocab_size,
        }
        metadata["model_loaded_memory"] = observe("after_model_loading")
        begin_validation(metadata, output, args.validation_receipt, RECEIPT_KIND)
        for scheme in SCHEMES:
            backend.configure_scheme(scheme)
            metadata.update(stage="warmup", active_scheme=scheme)
            write_json(output / "metadata.json", metadata)
            print(
                f"warmup {scheme}: user0 first, user1 first, user0 revisit; "
                "verify host recall before releasing all caches",
                flush=True,
            )
            metadata["warmup_traces"][scheme] = warmup(backend, workload, config, runner_type)
            write_json(output / "metadata.json", metadata)
            gc.collect()
            torch.cuda.empty_cache()
            case = run_case(backend, workload, config, output, metadata, observe, runner_type)
            metadata["cases"].append(case)
            write_json(output / "metadata.json", metadata)
        verify_source_snapshot(output)
        if backend_provenance() != metadata["backend_provenance"]:
            raise RuntimeError("installed backend identity changed during measurement")
        if precision_settings(torch) != metadata["precision_settings"]:
            raise RuntimeError("precision settings changed during measurement")
        if checkpoint_inventory(args.model_path) != metadata["checkpoint"]:
            raise RuntimeError("checkpoint identity changed during the run")
        if execution_environment() != metadata["execution_environment"]:
            raise RuntimeError("execution environment changed during the run")
        from experiments.nosa_motivation.src.cpu_environment import finish_cpu_environment

        finish_cpu_environment(metadata, torch)
        from experiments.deepseek_v32_echo_prefill.src.backend_provenance import (
            collect_flashinfer_runtime_artifacts,
        )
        from experiments.deepseek_v32_motivation.src.report import audit_run, write_report

        metadata.update(
            status="accepted",
            stage="complete",
            completed_unix=time.time(),
            precision_settings_verified_after_execution=True,
            flashinfer_runtime_artifacts=collect_flashinfer_runtime_artifacts(),
        )
        write_json(output / "metadata.json", metadata)
        if args.mode == "check":
            _, _, audit = audit_run(output)
            publish_receipt(metadata, output, RECEIPT_KIND, audit)
        else:
            write_report(output, output / "report")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output), str(target))
        print(
            f"accepted {args.run_id}: four schemes, {config['requests_per_scheme']} each; {target}"
        )
    except BaseException as error:
        metadata.update(status="failed", error=repr(error))
        write_json(output / "metadata.json", metadata)
        print(f"failed run retained outside experiments: {output}", flush=True)
        raise
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    main()
