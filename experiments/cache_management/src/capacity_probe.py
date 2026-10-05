"""Fill fixed ECHO host/pool capacities with real sequential GR user histories.

This is one executable capacity point, not a search or a latency benchmark.
CUDA allocator peaks and sampled device free memory are not NVML process peaks.
Failed runs remain in a system temporary directory for inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_SCHEMA = "echo-capacity-reference-v1"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def capacity_config(args):
    for name in ("history_tokens", "candidate_tokens", "chunk_size", "sparse_pool_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.host_arena_tokens < 64 or args.host_arena_tokens % 64:
        raise ValueError("host_arena_tokens must be a positive multiple of 64")
    if args.rounds < 2:
        raise ValueError("at least two complete rounds are required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        raise ValueError("run_id must be a plain nonempty identifier")
    capacity = args.history_tokens + args.candidate_tokens
    padded = (args.history_tokens + 63) // 64 * 64
    users = args.host_arena_tokens // padded
    if users < 1:
        raise ValueError("host arena cannot hold one complete history")
    if max(args.chunk_size, args.candidate_tokens) > args.sparse_pool_tokens:
        raise ValueError("actual query batch must fit the fixed sparse pool")
    return {
        "scheme": "echo",
        "num_layers": 10,
        "history_tokens": args.history_tokens,
        "candidate_tokens": args.candidate_tokens,
        "candidate_persistence": "gpu_transient",
        "capacity_tokens": capacity,
        "retained_capacity_tokens": args.history_tokens,
        "padded_session_tokens": padded,
        "chunk_size": args.chunk_size,
        "sparse_pool_tokens": args.sparse_pool_tokens,
        "host_arena_tokens": args.host_arena_tokens,
        "users": users,
        "rounds": args.rounds,
        "requests": users * args.rounds,
        "unused_host_tokens_at_full_admission": args.host_arena_tokens - users * padded,
        "seed": 42,
        "resource_mode": "fixed_pools",
        "hbm_budget_bytes": None,
        "dram_budget_bytes": None,
    }


def memory_sample(device, stage):
    """Sample boundaries; do not turn free-memory observations into process peaks."""
    row = {"stage": stage, "unix_time": time.time()}
    try:
        free, total = torch.cuda.mem_get_info(device)
        row.update(
            cuda_free_bytes=free,
            cuda_total_bytes=total,
            torch_allocated_bytes=torch.cuda.memory_allocated(device),
            torch_reserved_bytes=torch.cuda.memory_reserved(device),
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    except (RuntimeError, AttributeError) as error:
        row["cuda_observation_error"] = repr(error)
    try:
        row["host_allocator_counters"] = dict(torch.cuda.memory.host_memory_stats())
    except (RuntimeError, AttributeError) as error:
        row["host_observation_error"] = repr(error)
    row["process_max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return row


def tensor_summary(tensor):
    if not isinstance(tensor, torch.Tensor) or not tensor.numel():
        raise AssertionError("candidate output must be a nonempty tensor")
    value = tensor.detach().cpu().contiguous()
    if not bool(torch.isfinite(value).all()):
        raise AssertionError("candidate output contains nonfinite values")
    return value, {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": True,
        "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def check_lifecycle(request, metrics, config, index):
    users = config["users"]
    visit, user = divmod(index, users)
    expected = {
        "user_id": user,
        "visit_index": visit,
        "is_revisit": visit > 0,
        "prefix_cache_hit": visit > 0,
        "evicted_users": [],
        "cached_users": min(index + 1, users),
        "cache_host_pages": min(index + 1, users) * config["padded_session_tokens"] // 64,
        "host_page_capacity": config["host_arena_tokens"] // 64,
        "stable_prefix_tokens": config["history_tokens"],
        "candidate_suffix_tokens": config["candidate_tokens"],
        "hbm_budget_bytes": None,
        "dram_budget_bytes": None,
        "resource_mode": "fixed_pools",
    }
    if request["request_id"] != index or request["user_id"] != user:
        raise AssertionError("workload does not follow complete ordered user traversals")
    for name, value in expected.items():
        if name not in metrics or metrics[name] != value:
            raise AssertionError(f"request {index}: unexpected {name}: {metrics.get(name)!r}")
    if config.get("candidate_persistence") == "gpu_transient":
        diagnostics = metrics.get("cache_diagnostics", {})
        for name, expected_value in (
            ("candidate_persistence", "gpu_transient"),
            ("retained_length", config["history_tokens"]),
            ("candidate_device_to_host_bytes", 0),
        ):
            if diagnostics.get(name) != expected_value:
                raise AssertionError(f"request {index}: candidate diagnostic {name} differs")


def reference_binding(config, source_id, checkpoint, tokenizer_sha256):
    binding = {
        "source_sha256": source_id,
        "checkpoint": checkpoint,
        "tokenizer_sha256": tokenizer_sha256,
        "num_layers": config["num_layers"],
        "history_tokens": config["history_tokens"],
        "candidate_tokens": config["candidate_tokens"],
        "chunk_size": config["chunk_size"],
        "output_scope": "all_candidate_normalized_hidden_and_last_token_lm_head",
    }
    if "candidate_persistence" in config:
        binding.update(
            candidate_persistence=config["candidate_persistence"],
            retained_capacity_tokens=config["retained_capacity_tokens"],
        )
    return binding


def load_reference(directory, binding):
    if directory is None or not (directory / "manifest.json").is_file():
        return None
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != REFERENCE_SCHEMA or manifest.get("scheme") != "hbm":
        raise ValueError("numerical acceptance requires an independent HBM reference manifest")
    if manifest.get("status") != "complete" or manifest.get("binding") != binding:
        raise ValueError("HBM reference is incomplete or has different source/input semantics")
    return manifest


def check_plan(plan, config):
    for name, requested in (
        ("host_arena_tokens", config["host_arena_tokens"]),
        ("sparse_pool_tokens", config["sparse_pool_tokens"]),
        ("max_session_capacity", config["capacity_tokens"]),
        ("max_history_tokens", config["retained_capacity_tokens"]),
        ("max_candidate_tokens", config["candidate_tokens"]),
        ("candidate_slots", config["candidate_tokens"]),
        ("candidate_persistence", config["candidate_persistence"]),
    ):
        if plan.metadata.get(name) != requested:
            raise AssertionError(f"backend changed the requested fixed capacity: {name}")
    if plan.host_pages != config["host_arena_tokens"] // 64:
        raise AssertionError("backend host-page quota differs from the fixed host arena")


def compare_reference(directory, manifest, index, payload):
    if manifest is None:
        return {"status": "unverified", "reason": "independent HBM reference unavailable"}
    path = directory / f"{index:06d}.pt"
    if not path.is_file():
        return {"status": "unverified", "reason": "reference request unavailable"}
    expected = torch.load(path, weights_only=True, map_location="cpu")
    if expected.get("input_sha256") != payload["input_sha256"]:
        raise AssertionError("reference request token identity differs")
    result = {"status": "passed", "atol": 0, "rtol": 0}
    for name in ("hidden", "logits"):
        actual, reference = payload[name], expected[name]
        tensor_summary(reference)
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        result[name] = {"exact": bool(torch.equal(actual, reference))}
    return result


def run_requests(runner, workload, config, output, metadata, observe, reference=None):
    numerical = output / "numerical"
    numerical.mkdir()
    all_compared = reference is not None
    with (output / "requests.jsonl").open("x") as stream:
        for index, request in enumerate(workload.requests):
            metadata["stage"] = (
                "first_visit_prefill_extend" if index < config["users"] else "revisit_extend"
            )
            metadata["active_request_id"] = index
            write_json(output / "status.json", metadata)
            observe(f"before_request_{index}")
            result = runner.execute(request)
            check_lifecycle(request, result.metrics, config, index)
            hidden, hidden_summary = tensor_summary(result.hidden)
            logits, logits_summary = tensor_summary(runner.backend.last_logits)
            if hidden.shape != (config["candidate_tokens"], runner.backend.cfg.dim):
                raise AssertionError("all candidate hidden states must have the model dimension")
            if logits.shape != (1, runner.backend.cfg.vocab_size):
                raise AssertionError("last-token logits must cover the complete vocabulary")
            del result.hidden
            payload = {
                "request_id": index,
                "input_sha256": request["input_sha256"],
                "hidden": hidden,
                "logits": logits,
            }
            path = numerical / f"{index:06d}.pt"
            torch.save(payload, path)
            metadata["stage"] = "numerical_comparison"
            comparison = compare_reference(
                None if reference is None else reference[0],
                None if reference is None else reference[1],
                index,
                payload,
            )
            all_compared &= comparison["status"] == "passed"
            row = {
                "request_id": index,
                "input_sha256": request["input_sha256"],
                "metrics": result.metrics,
                "hidden": hidden_summary,
                "logits": logits_summary,
                "numerical": comparison,
                "output_file": str(path.relative_to(output)),
                "output_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            metadata["completed_requests"] = index + 1
            metadata["max_retained_users"] = result.metrics["cached_users"]
            observe(f"after_request_{index}")
            if config["users"] <= 16 or (index + 1) % 16 == 0 or (index + 1) % config["users"] == 0:
                print(
                    f"request {index + 1}/{config['requests']}: user={request['user_id']} "
                    f"hit={result.metrics['prefix_cache_hit']} retained={result.metrics['cached_users']}",
                    flush=True,
                )
    if metadata["completed_requests"] != config["requests"]:
        raise AssertionError("workload ended before all complete traversals")
    if len(runner.pool) != config["users"]:
        raise AssertionError("full host user capacity was not retained")
    return "passed" if all_compared else "unverified"


def audit_existing(output, reference_directory):
    """Compare saved full outputs on CPU without repeating the model execution."""
    metadata = json.loads((output / "status.json").read_text())
    audit = {
        "schema": "echo-capacity-numerical-audit-v1",
        "run_id": metadata.get("run_id"),
        "reference_directory": str(reference_directory.resolve()),
        "started_unix": time.time(),
        "status": "running",
        "requests": [],
    }
    try:
        if metadata.get("status") != "complete" or not metadata.get("capacity_passed"):
            raise ValueError("only a completed capacity run can be audited")
        source_manifest = json.loads((output / "source_manifest.json").read_text())
        source_digest = hashlib.sha256(
            json.dumps(source_manifest, sort_keys=True).encode()
        ).hexdigest()
        if source_digest != metadata["source_sha256"]:
            raise AssertionError("saved source manifest identity differs")
        for name, digest in source_manifest.items():
            path = output / "source" / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise AssertionError(f"saved source snapshot differs: {name}")
        recorded = json.loads((output / "numerical/manifest.json").read_text())
        if (
            recorded.get("schema") != REFERENCE_SCHEMA
            or recorded.get("scheme") != "echo"
            or recorded.get("status") != "complete"
        ):
            raise ValueError("saved ECHO numerical manifest is incomplete")
        binding = recorded["binding"]
        expected_binding = reference_binding(
            metadata["config"],
            source_digest,
            metadata["checkpoint"],
            binding["tokenizer_sha256"],
        )
        if binding != expected_binding:
            raise AssertionError("saved numerical binding differs from the capacity run")
        manifest = load_reference(reference_directory, binding)
        if manifest is None:
            raise ValueError("independent HBM reference manifest is unavailable")
        audit["binding"] = binding
        audit["reference_manifest_sha256"] = hashlib.sha256(
            (reference_directory / "manifest.json").read_bytes()
        ).hexdigest()
        rows = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
        audit["requests_file_sha256"] = hashlib.sha256(
            (output / "requests.jsonl").read_bytes()
        ).hexdigest()
        if len(rows) != metadata["config"]["requests"]:
            raise AssertionError("saved request count differs from the complete workload")
        for index, row in enumerate(rows):
            if row["request_id"] != index:
                raise AssertionError("saved request order differs")
            if row["output_file"] != f"numerical/{index:06d}.pt":
                raise AssertionError("saved output path differs from the request identity")
            path = output / row["output_file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["output_file_sha256"]:
                raise AssertionError(f"request {index}: saved output file digest differs")
            payload = torch.load(path, weights_only=True, map_location="cpu")
            if payload["request_id"] != index or payload["input_sha256"] != row["input_sha256"]:
                raise AssertionError(f"request {index}: saved input identity differs")
            for name in ("hidden", "logits"):
                _, summary = tensor_summary(payload[name])
                if summary != row[name]:
                    raise AssertionError(f"request {index}: saved {name} digest differs")
            comparison = compare_reference(reference_directory, manifest, index, payload)
            audit["requests"].append({"request_id": index, **comparison})
            if comparison["status"] != "passed":
                raise AssertionError(f"request {index}: independent HBM output unavailable")
        audit["status"] = "passed"
    except (OSError, ValueError, AssertionError, RuntimeError, KeyError, TypeError) as error:
        audit["status"] = "failed"
        audit["failure"] = {"type": type(error).__name__, "message": str(error)}
    audit["finished_unix"] = time.time()
    write_json(output / "numerical_audit.json", audit)
    metadata["numerical_status"] = audit["status"]
    metadata["numerical_audit_file"] = "numerical_audit.json"
    write_json(output / "status.json", metadata)
    print(f"NUMERICAL {audit['status'].upper()}; audit={output / 'numerical_audit.json'}")
    return 0 if audit["status"] == "passed" else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    action = result.add_mutually_exclusive_group(required=True)
    action.add_argument("--run-id")
    action.add_argument("--audit-existing", type=Path, help="CPU audit of a completed run")
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--model-path", type=Path, default=Path("/preset-models"))
    result.add_argument("--sparse-pool-tokens", type=int)
    result.add_argument("--host-arena-tokens", type=int)
    result.add_argument("--chunk-size", type=int, choices=(1024,), default=1024)
    result.add_argument("--history-tokens", type=int, default=65536)
    result.add_argument("--candidate-tokens", type=int, default=128)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--rounds", type=int, default=2)
    result.add_argument("--reference-dir", type=Path)
    result.add_argument("--save-reference-dir", type=Path)
    return result


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.audit_existing is not None:
        if args.reference_dir is None:
            command.error("--audit-existing requires --reference-dir")
        return audit_existing(args.audit_existing, args.reference_dir)
    if args.sparse_pool_tokens is None or args.host_arena_tokens is None:
        command.error("--run-id requires --sparse-pool-tokens and --host-arena-tokens")
    config = capacity_config(args)
    target = args.output_dir or ROOT / "experiments/cache_management/output/data" / args.run_id
    if target.exists() or args.save_reference_dir is not None and args.save_reference_dir.exists():
        raise FileExistsError("output and exported-reference directories must be new")
    output = Path(tempfile.mkdtemp(prefix=f"echo-capacity-{args.run_id}-"))
    print(f"temporary output: {output}", flush=True)
    metadata = {
        "schema": "echo-capacity-probe-v1",
        "run_id": args.run_id,
        "status": "running",
        "stage": "source_snapshot",
        "config": config,
        "started_unix": time.time(),
        "completed_requests": 0,
        "max_retained_users": 0,
        "capacity_passed": False,
        "numerical_status": "unverified",
        "measurement_boundary": "PyTorch CUDA allocator peaks since before model loading; device free memory sampled at boundaries; neither is an NVML whole-process peak. Host allocator counters do not prove exact concurrent ownership.",
        "scope": "ten independent checkpoint dense blocks; complete real GR history construction, history-only retained sessions and GPU-transient candidates; one fixed P/NH point; no byte sub-budgets, warmup, or capacity extrapolation",
    }
    write_json(output / "status.json", metadata)
    backend = runner = None
    samples = []

    def observe(stage):
        samples.append(memory_sample(args.device, stage))
        write_json(output / "memory.json", samples)

    try:
        from evaluation.provenance import (
            _git,
            backend_provenance,
            source_snapshot,
            verify_source_snapshot,
        )
        from GR.workload import WorkloadConfig, build_workload
        from serving.persistent import PersistentGRRunner

        metadata["source_sha256"] = source_snapshot(output, include_official=True)
        metadata["git_revision"] = _git("rev-parse", "HEAD")
        metadata["git_status"] = _git("status", "--short")
        metadata["backend_provenance"] = backend_provenance()
        checkpoint = {
            "path": str(args.model_path.resolve()),
            "files": {
                path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                for path in sorted(args.model_path.glob("*.safetensors"))
            },
            "identity_boundary": "checkpoint path and file stat inventory; not a full weights digest",
        }
        metadata["checkpoint"] = checkpoint
        metadata["stage"] = "hardware_initialization"
        device = torch.device(args.device)
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
            raise RuntimeError("capacity probe requires one SM90/Hopper CUDA device")
        torch.cuda.set_device(device)
        os.environ.setdefault("CXLDSAGR_SM90_BACKEND", "native")
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
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        }
        torch.cuda.reset_peak_memory_stats(device)
        observe("before_model_loading")
        metadata["stage"] = "workload_generation"
        workload = build_workload(
            WorkloadConfig(
                model="deepseek_v32",
                num_users=config["users"],
                requests=config["requests"],
                history_tokens=args.history_tokens,
                candidate_tokens=args.candidate_tokens,
                seed=config["seed"],
                sampling="sequential",
            ),
            tokenizer=args.model_path,
        )
        workload.write(output / "workload")
        binding = reference_binding(
            config, metadata["source_sha256"], checkpoint, workload.manifest["tokenizer_sha256"]
        )
        reference = load_reference(args.reference_dir, binding)
        metadata["workload_sha256"] = workload.manifest["workload_sha256"]
        metadata["stage"] = "model_loading"
        write_json(output / "status.json", metadata)
        from models.deepseek_v32.execution.adapter import DeepSeekServingBackend

        backend = DeepSeekServingBackend(
            args.model_path,
            scheme="echo",
            device=args.device,
            num_layers=10,
            chunk_size=args.chunk_size,
            sparse_pool_tokens=args.sparse_pool_tokens,
            host_arena_tokens=args.host_arena_tokens,
            workspace_query_tokens=max(args.chunk_size, args.candidate_tokens),
        )
        observe("after_model_loading")
        metadata["stage"] = "shared_pool_allocation"
        write_json(output / "status.json", metadata)
        runner = PersistentGRRunner(
            backend,
            hbm_budget_bytes=None,
            dram_budget_bytes=None,
            resource_limits={
                "max_session_capacity": config["capacity_tokens"],
                "max_history_tokens": args.history_tokens,
                "max_candidate_tokens": args.candidate_tokens,
            },
        )
        check_plan(runner.resource_plan, config)
        metadata["backend"] = backend.describe()
        observe("after_shared_pool_allocation")
        metadata["numerical_status"] = run_requests(
            runner,
            workload,
            config,
            output,
            metadata,
            observe,
            None if reference is None else (args.reference_dir, reference),
        )
        write_json(
            output / "numerical/manifest.json",
            {
                "schema": REFERENCE_SCHEMA,
                "status": "complete",
                "scheme": "echo",
                "binding": binding,
            },
        )
        metadata["stage"] = "source_verification"
        verify_source_snapshot(output)
        metadata["capacity_passed"] = True
    except BaseException as error:  # noqa: BLE001 - Preserve diagnostics, including interruption.
        metadata["status"] = "failed"
        metadata["failure"] = {
            "stage": metadata["stage"],
            "type": type(error).__name__,
            "message": str(error),
        }
        (output / "error.txt").write_text(traceback.format_exc())
        observe("failure")
    finally:
        cleanup_errors = []
        for name, obj in (("runner", runner), ("backend", backend)):
            if obj is not None:
                try:
                    obj.close()
                except BaseException as error:  # noqa: BLE001 - Attempt every owner cleanup.
                    cleanup_errors.append({"object": name, "error": repr(error)})
        metadata["cleanup"] = {"passed": not cleanup_errors, "errors": cleanup_errors}
        if cleanup_errors:
            metadata["status"] = "failed"
            metadata["capacity_passed"] = False
        observe("after_cleanup")
        metadata["finished_unix"] = time.time()
        if metadata["status"] != "failed":
            metadata["status"] = "complete"
            metadata["stage"] = "complete"
        write_json(output / "status.json", metadata)
    if metadata["status"] == "failed":
        print(f"FAILED; artifacts retained at {output}", flush=True)
        return 1
    try:
        if args.save_reference_dir is not None:
            shutil.copytree(output / "numerical", args.save_reference_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        shutil.move(str(output), str(target))
    except OSError as error:
        metadata["status"] = "failed"
        metadata["failure"] = {
            "stage": "publication",
            "type": type(error).__name__,
            "message": str(error),
        }
        write_json(output / "status.json", metadata)
        print(f"FAILED to publish; artifacts retained at {output}: {error}", flush=True)
        return 1
    print(f"COMPLETE; numerical={metadata['numerical_status']}; output={target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
