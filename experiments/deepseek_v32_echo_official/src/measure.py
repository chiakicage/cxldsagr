"""Compare the adapted official ECHO offload pipeline with a fresh HBM baseline."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

from experiments.deepseek_v32_motivation.src import measure as motivation

ROOT = motivation.ROOT
EXPERIMENT = ROOT / "experiments/deepseek_v32_echo_official"
SCHEMA = "deepseek-v32-echo-official-v1"
SCHEMES = ("hbm", "echo")
LABELS = {"hbm": "HBM-only", "echo": "Official ECHO (adapted offload pipeline)"}
INDEXER_DISPATCH_POLICY = "official_fused_every_offload_call_v1"
PRECISION_POLICY = "fp32_highest_cuda_matmul_tf32_disabled_v1"
VERIFICATION_STAGES = ("after_hbm_warmup", "after_hbm_repeat", "after_echo_warmup", "final")
NUMERICAL_POLICY = {
    "policy_id": "official-resident-calibrated-fp64-v1",
    "atol": 1 / 32,
    "rtol": 1 / 64,
    "relative_l2_limits": {"hidden": 0.005, "logits": 0.01},
    "minimum_elementwise_pass_fraction": 0.999,
    "metrics_dtype": "torch.float64",
    "reference": "primary_hbm_same_run",
}
NUMERICAL_POLICY_BASIS = {
    "scope": "independent 64K resident calibration; 4 requests, 2 repeats",
    "frozen_before": "formal 64K offload output measurement",
    "calibration_metrics_sha256": "8f3e39e902febe1d6dfd2bd7cdf617de1cf6fd6d1a7bdca0f0adaa1ec26119e7",
    "hidden": {
        "max_relative_l2": 0.0025213068990911232,
        "minimum_elementwise_pass_fraction": 0.9999280657087054,
        "max_abs": 0.1796875,
    },
    "logits": {
        "max_relative_l2": 0.002260235285441428,
        "minimum_elementwise_pass_fraction": 1.0,
        "max_abs": 0.0625,
    },
    "interpretation": (
        "Upstream unordered top-k can vary ordering and ties; observed differences are not "
        "all proven to be rounding. Criteria are not fitted to formal offload outputs."
    ),
}
write_json = motivation.write_json


def parser():
    command = motivation.parser()
    command.description = __doc__
    return command


def normalize_compute_graphs(values):
    """Accept explicit Boolean aliases while preserving legacy absent-field semantics."""
    flags = [values[name] for name in ("compute_graphs", "enable_compute_graphs") if name in values]
    if any(type(value) is not bool for value in flags):
        raise ValueError("compute graph flags must be bool")
    if len(set(flags)) > 1:
        raise ValueError("conflicting compute graph flags")
    return flags[0] if flags else False


def configuration(args):
    values = vars(args)
    normalized = SimpleNamespace(**{**values, "compute_graphs": normalize_compute_graphs(values)})
    config = motivation.configuration(normalized)
    # Preserve exact re-audits of saved eager configurations from before this field existed.
    if not any(name in values for name in ("compute_graphs", "enable_compute_graphs")):
        config.pop("enable_compute_graphs", None)
    config["schemes"] = list(SCHEMES)
    config["indexer_dispatch_policy"] = INDEXER_DISPATCH_POLICY
    return config


def precision_settings(torch_module):
    """Own this snapshot so frozen older motivation helpers remain sufficient."""
    matmul = torch_module.backends.cuda.matmul
    return {
        "float32_matmul_precision": torch_module.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": matmul.allow_tf32,
        "bf16_reduced_precision_reduction": matmul.allow_bf16_reduced_precision_reduction,
        "fp16_reduced_precision_reduction": matmul.allow_fp16_reduced_precision_reduction,
        "cudnn_allow_tf32": torch_module.backends.cudnn.allow_tf32,
    }


def configure_precision(torch_module):
    torch_module.set_float32_matmul_precision("highest")
    torch_module.backends.cuda.matmul.allow_tf32 = False
    settings = precision_settings(torch_module)
    if settings["float32_matmul_precision"] != "highest" or settings["cuda_matmul_allow_tf32"]:
        raise RuntimeError(
            "official measurement requires highest FP32 precision without matmul TF32"
        )
    return settings


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def numerical_comparison(actual, expected, name):
    """Use frozen elementwise and global error limits, with FP64 CPU reductions."""
    import torch

    limit = NUMERICAL_POLICY["relative_l2_limits"][name]
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("numerical output shape or dtype differs from HBM")
    a, b = (
        actual.detach().to(device="cpu", dtype=torch.float64),
        expected.detach().to(device="cpu", dtype=torch.float64),
    )
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise AssertionError("nonfinite output in official ECHO numerical gate")
    difference = a - b
    relative_l2 = float(difference.norm() / b.norm().clamp_min(1e-20))
    allclose = bool(
        torch.allclose(a, b, atol=NUMERICAL_POLICY["atol"], rtol=NUMERICAL_POLICY["rtol"])
    )
    tolerance = NUMERICAL_POLICY["atol"] + NUMERICAL_POLICY["rtol"] * b.abs()
    passing = int((difference.abs() <= tolerance).sum())
    pass_fraction = passing / a.numel()
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "metrics_dtype": NUMERICAL_POLICY["metrics_dtype"],
        "finite": True,
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "max_abs": float(difference.abs().max()),
        "relative_l2": relative_l2,
        "allclose": allclose,
        "elementwise_pass_fraction": pass_fraction,
        "elementwise_pass_count": passing,
        "elementwise_outliers": a.numel() - passing,
        "elements": a.numel(),
        "atol": NUMERICAL_POLICY["atol"],
        "rtol": NUMERICAL_POLICY["rtol"],
        "relative_l2_limit": limit,
        "minimum_elementwise_pass_fraction": NUMERICAL_POLICY["minimum_elementwise_pass_fraction"],
        "numerical_pass": pass_fraction >= NUMERICAL_POLICY["minimum_elementwise_pass_fraction"]
        and relative_l2 <= limit,
    }


def save_comparison(backend, result, request, config, output, label):
    import torch

    payload = motivation.tensor_payload(backend, result, request, config)
    expected = (
        payload
        if label == "hbm"
        else torch.load(
            output / "numerical/hbm" / f"{request['request_id']:06d}.pt",
            weights_only=True,
            map_location="cpu",
        )
    )
    if any(payload[name] != expected[name] for name in ("request_id", "input_sha256")):
        raise AssertionError("HBM reference uses a different request")
    comparison = {
        name: numerical_comparison(payload[name], expected[name], name)
        for name in ("hidden", "logits")
    }
    destination = output / "numerical" / label / f"{request['request_id']:06d}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    torch.save(payload, destination)
    return {
        "numerical": comparison,
        "output_file": str(destination.relative_to(output)),
        "output_sha256": digest(destination),
    }


def require_numerical(row):
    failures = {
        name: value for name, value in row["numerical"].items() if not value["numerical_pass"]
    }
    if failures:
        raise AssertionError(
            f"fixed numerical gate failed for request {row['request_id']}: {failures}"
        )


def append_row(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def run_case(backend, workload, config, output, metadata, observe, runner_type):
    """Preserve motivation's timing boundary; compare saved output outside that interval."""
    import torch

    scheme = backend.scheme
    torch.cuda.reset_peak_memory_stats(backend.device)
    before_allocation = observe(f"{scheme}/before_cache_allocation")
    rows = []
    with runner_type(backend, resource_limits=motivation.resource_limits(config)) as runner:
        if len(runner.pool):
            raise AssertionError("each measured scheme must start from an empty user cache")
        plan = runner.resource_plan
        if plan.metadata.get("sparse_pool_tokens") != config["sparse_pool_tokens"]:
            raise AssertionError("backend did not apply the requested P capacity")
        if (
            scheme != "hbm"
            and plan.metadata.get("host_arena_tokens") != config["host_arena_tokens"]
        ):
            raise AssertionError("backend did not apply the requested NH capacity")
        observe(f"{scheme}/after_cache_allocation")
        for request in workload.requests:
            request_id = request["request_id"]
            metadata.update(stage="measuring", active_scheme=scheme, active_request=request_id)
            write_json(output / "metadata.json", metadata)
            before = observe(f"{scheme}/before_request_{request_id}")
            graphs_before = backend.describe().get("compute_graphs")
            result = runner.execute(request)
            graphs_after = backend.describe().get("compute_graphs")
            after = observe(f"{scheme}/after_request_{request_id}")
            motivation.check_request(request, result.metrics, config)
            motivation.check_compute_graph_replays(
                graphs_before, graphs_after, config, result.metrics
            )
            evidence = save_comparison(backend, result, request, config, output, scheme)
            row = {
                **result.metrics,
                "run_id": metadata["run_id"],
                "round_index": request_id // config["num_users"],
                "workload_sha256": workload.manifest["workload_sha256"],
                "diagnostics_scope": "candidate_only",
                "memory_before": before,
                "memory_after": after,
                "compute_graphs": {"before": graphs_before, "after": graphs_after},
                **evidence,
            }
            append_row(output / "measurements.jsonl", row)
            require_numerical(row)
            rows.append(row)
            print(
                f"{scheme} {request_id + 1}/{config['requests_per_scheme']}: "
                f"user={request['user_id']} revisit={row['is_revisit']} "
                f"hit={row['prefix_cache_hit']} latency_ms={row['latency_ms']:.3f}",
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
    observe(f"{scheme}/after_cleanup")
    return case


def run_resident_repeat(backend, workload, config, output, metadata, runner_type):
    """Repeat the full resident trace from empty sessions without publishing its latency."""
    if backend.scheme != "hbm":
        raise ValueError("resident repeat requires the HBM backend")
    backend.close()
    with runner_type(backend, resource_limits=motivation.resource_limits(config)) as runner:
        if len(runner.pool):
            raise AssertionError("resident repeat must start from an empty user cache")
        for request in workload.requests:
            metadata.update(stage="resident_repeat", active_request=request["request_id"])
            write_json(output / "metadata.json", metadata)
            graphs_before = backend.describe().get("compute_graphs")
            result = runner.execute(request)
            graphs_after = backend.describe().get("compute_graphs")
            motivation.check_request(request, result.metrics, config)
            motivation.check_compute_graph_replays(
                graphs_before, graphs_after, config, result.metrics
            )
            row = {
                **{key: value for key, value in result.metrics.items() if not key.endswith("_ms")},
                "validation_label": "hbm_repeat",
                "run_id": metadata["run_id"],
                "workload_sha256": workload.manifest["workload_sha256"],
                "compute_graphs": {"before": graphs_before, "after": graphs_after},
                **save_comparison(backend, result, request, config, output, "hbm_repeat"),
            }
            append_row(output / "validation.jsonl", row)
            require_numerical(row)
            print(
                f"hbm_repeat {request['request_id'] + 1}/{config['requests_per_scheme']}: numerical passed",
                flush=True,
            )
            del result
        repeat = {
            "label": "hbm_repeat",
            "scheme": "hbm",
            "requests": len(workload.requests),
            "started_empty": True,
            "timing_published": False,
            "resource_plan": dict(runner.resource_plan.metadata),
            "token_validation": getattr(runner, "token_validation_identity", None),
            "shared_reservation": {
                "hbm": runner.resource_plan.shared.hbm,
                "dram": runner.resource_plan.shared.dram,
            },
            "admission_hbm_token_capacity": runner.resource_plan.hbm_tokens,
            "backend": backend.describe(),
        }
    backend.close()
    return repeat


def snapshot_sources(output):
    """Include the reused harness and the new experiment in the production snapshot."""
    motivation.snapshot_sources(output)
    manifest = json.loads((output / "source_manifest.json").read_text())
    paths = [Path(motivation.__file__)]
    for directory in (EXPERIMENT / "src", EXPERIMENT / "scripts"):
        paths.extend(path for path in directory.rglob("*") if path.suffix in (".py", ".sh"))
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        manifest[str(relative)] = digest(path)
    write_json(output / "source_manifest.json", manifest)
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def provenance_files(provenance):
    """Require identities for both the upstream sources and the loaded extension."""
    files = []
    for kind, field in (("source", "source_files"), ("native", "native_files")):
        values = provenance.get(field)
        if not isinstance(values, dict) or not values:
            raise ValueError(f"official provenance requires nonempty {field}")
        for name, sha in sorted(values.items()):
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(sha, str)
                or len(sha) != 64
                or any(char not in "0123456789abcdef" for char in sha)
            ):
                raise ValueError(f"invalid official {field} identity")
            files.append((kind, name, sha))
    return files


def snapshot_official(provenance, output):
    """Copy external runtime dependencies and native binaries without path ambiguity."""
    manifest = []
    for kind, original, expected in provenance_files(provenance):
        path = Path(original)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"official {kind} changed before snapshot: {original}")
        path_id = hashlib.sha256(original.encode()).hexdigest()
        relative = Path("official_artifacts") / kind / path_id / path.name
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(path, destination)
        if digest(destination) != expected:
            raise RuntimeError(f"official {kind} changed while copying: {original}")
        manifest.append(
            {"kind": kind, "original": original, "snapshot": str(relative), "sha256": expected}
        )
    write_json(output / "official_artifact_manifest.json", manifest)
    return digest(output / "official_artifact_manifest.json")


def reconcile_official(previous, current, *, allow_new_native):
    """Only warmup may append shape-specific JIT files; existing bytes stay frozen."""
    provenance_files(current)
    if {key: value for key, value in previous.items() if key != "native_files"} != {
        key: value for key, value in current.items() if key != "native_files"
    }:
        raise RuntimeError("official source or build identity changed during measurement")
    old_native, new_native = previous["native_files"], current["native_files"]
    if any(new_native.get(name) != expected for name, expected in old_native.items()):
        raise RuntimeError("an existing official native artifact changed during measurement")
    if not allow_new_native and old_native != new_native:
        raise RuntimeError("official native artifact inventory changed after warmup")


def verify_identities(backend, output, metadata, stage):
    import torch

    from experiments.gr_serving.src.measure import backend_provenance, verify_source_snapshot

    precision = precision_settings(torch)
    if precision != metadata["precision_settings"]:
        raise RuntimeError("matmul precision settings changed during measurement")
    if stage == "final":
        metadata["precision_settings_final"] = precision
    verify_source_snapshot(output)
    if backend_provenance() != metadata["backend_provenance"]:
        raise RuntimeError("installed baseline backend identity changed during measurement")
    current = backend.official_provenance()
    reconcile_official(
        metadata["official_provenance"], current, allow_new_native=stage.endswith("_warmup")
    )
    for _, name, expected in provenance_files(current):
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"official artifact changed during measurement: {name}")
    metadata["official_provenance"] = current
    metadata["official_artifact_manifest_sha256"] = snapshot_official(current, output)
    metadata["identity_verifications"].append({"stage": stage, "status": "passed"})
    write_json(output / "metadata.json", metadata)


def main(argv=None):
    args = parser().parse_args(argv)
    config = configuration(args)
    target = args.output_dir or EXPERIMENT / "output/data" / args.run_id
    if target.exists():
        raise FileExistsError(target)
    output = Path(tempfile.mkdtemp(prefix=f"deepseek-echo-official-{args.run_id}-"))
    print(f"temporary output: {output}", flush=True)
    metadata = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "status": "running",
        "config": config,
        "cases": [],
        "warmup_traces": {},
        "identity_verifications": [],
        "validation_repeats": [],
        "numerical_policy": NUMERICAL_POLICY,
        "numerical_policy_basis": NUMERICAL_POLICY_BASIS,
        "scheme_labels": LABELS,
        "precision_policy": PRECISION_POLICY,
        "started_unix": time.time(),
        "implementation_boundary": (
            "Adapted official ECHO offload pipeline: upstream allocator, fused indexer/prefetch "
            "and exact recall. HBM uses upstream resident logits; both schemes use the same "
            "original upstream top-k semantics. Checkpoint weights, model projection, no-Hadamard indexer inputs, "
            "FlashMLA and fixed-history GR serving follow motivation. The HBM control therefore "
            "also replaces motivation's mainline DeepGEMM logits and FlashInfer top-k. "
            "This is not the unmodified upstream server."
        ),
        "measurement_boundary": (
            "Synchronized runner wall latency includes input transfer, admission/eviction, "
            "history construction on a miss, all candidate hidden, last-token LM head and "
            "history cleanup. GPU counter accumulation and reductions inside forward are included. "
            "Loading, generation, compute-graph preparation, three warmup requests, host counter readout, output copies, "
            "numerical comparison and reporting are outside latency. Each request "
            "is measured once. The HBM reference is executed afresh in this same run. A separate "
            "full resident validation repeat publishes outputs and errors, without latency."
        ),
        "diagnostics_boundary": (
            "Cache transfer counters cover candidate forward only; history prefill counters "
            "reset before the candidate. Device counter accumulation/reductions are timed inside "
            "forward; reading counters to the host occurs outside request latency. "
            "Prefix hit measures retained whole-user history."
        ),
        "memory_boundary": (
            "Each scheme resets CUDA allocated/reserved peaks after releasing warmup caches "
            "and before fresh shared allocation. Peaks include loaded weights and the complete "
            "formal trace. Device free memory is sampled only at request boundaries. P/NH are "
            "token capacities; no cache byte subbudget or empirical headroom is subtracted."
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

        metadata["precision_settings"] = configure_precision(torch)

        from experiments.deepseek_v32_echo_cache.src.capacity_probe import memory_sample
        from experiments.gr_serving.src.measure import _git, backend_provenance
        from experiments.gr_serving.src.workload import WorkloadConfig, build_workload
        from models.deepseek_v32.official_serving import OfficialDeepSeekServingBackend
        from serving.persistent import PersistentGRRunner

        runner_type = partial(PersistentGRRunner, native_token_validation=True)

        device = torch.device(args.device)
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
            raise RuntimeError("this experiment requires one SM90/Hopper GPU")
        torch.cuda.set_device(device)
        os.environ.setdefault("CXLDSAGR_SM90_BACKEND", "native")
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
        metadata["checkpoint"] = {
            "path": str(args.model_path.resolve()),
            "files": {
                path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                for path in sorted(args.model_path.glob("*.safetensors"))
            },
            "identity_boundary": "checkpoint path and shard stat inventory, not weights hashes",
        }

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
        observe("before_model_loading")
        backend = OfficialDeepSeekServingBackend(
            args.model_path,
            scheme="hbm",
            device=device,
            num_layers=config["layers"],
            chunk_size=config["chunk_size"],
            sparse_pool_tokens=config["sparse_pool_tokens"],
            host_arena_tokens=config["host_arena_tokens"],
            workspace_query_tokens=config["workspace_query_tokens"],
            enable_compute_graphs=normalize_compute_graphs(config),
        )
        backend.synchronize()
        metadata["model_dimensions"] = {
            "hidden": backend.cfg.dim,
            "vocabulary": backend.cfg.vocab_size,
        }
        metadata["model_loaded_memory"] = observe("after_model_loading")
        metadata["official_provenance"] = backend.official_provenance()
        metadata["official_artifact_manifest_sha256"] = snapshot_official(
            metadata["official_provenance"], output
        )
        for scheme in SCHEMES:
            backend.configure_scheme(scheme)
            metadata.update(stage="warmup", active_scheme=scheme)
            write_json(output / "metadata.json", metadata)
            print(f"warmup {LABELS[scheme]}: user0 first, user1 first, user0 revisit", flush=True)
            metadata["warmup_traces"][scheme] = motivation.warmup(
                backend, workload, config, runner_type
            )
            verify_identities(backend, output, metadata, f"after_{scheme}_warmup")
            gc.collect()
            torch.cuda.empty_cache()
            case = run_case(backend, workload, config, output, metadata, observe, runner_type)
            metadata["cases"].append(case)
            write_json(output / "metadata.json", metadata)
            if scheme == "hbm":
                gc.collect()
                torch.cuda.empty_cache()
                metadata["validation_repeats"].append(
                    run_resident_repeat(backend, workload, config, output, metadata, runner_type)
                )
                verify_identities(backend, output, metadata, "after_hbm_repeat")
        verify_identities(backend, output, metadata, "final")
        from experiments.deepseek_v32_echo_official.src.report import write_report
        from experiments.deepseek_v32_echo_prefill.src.backend_provenance import (
            collect_flashinfer_runtime_artifacts,
        )

        metadata.update(
            status="accepted",
            stage="complete",
            completed_unix=time.time(),
            flashinfer_runtime_artifacts=collect_flashinfer_runtime_artifacts(),
        )
        write_json(output / "metadata.json", metadata)
        write_report(output, output / "report")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output), str(target))
        print(
            f"accepted {args.run_id}: two schemes, {config['requests_per_scheme']} each; {target}"
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
