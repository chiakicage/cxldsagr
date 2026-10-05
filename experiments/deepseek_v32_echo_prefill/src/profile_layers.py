"""Profile the real first three DeepSeek layers, separately for prefix and extend."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import statistics
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from unittest.mock import patch

import torch

from evaluation.validation import require_receipt, write_receipt
from experiments.deepseek_v32_echo_prefill.src.measure import make_request, source_manifest
from experiments.deepseek_v32_echo_prefill.src.operator_instrumentation import (
    InstrumentOperators,
    OperatorScopes,
)
from experiments.deepseek_v32_echo_prefill.src.profile_hardware import gather_hardware
from experiments.deepseek_v32_echo_prefill.src.run_contract import (
    RECEIPT_KIND,
    bind_benchmark,
    checkpoint_identity,
    execution_identity,
    receipt_binding,
)
from models.deepseek_v32.model import DeepSeekEchoModel
from models.deepseek_v32.nonmatrix import rms_norm
from operators.deepseek_v32.indexer.echo import build_info

ROOT = Path(__file__).resolve().parents[3]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sources():
    result = source_manifest()
    for name in (
        "profile_layers.py",
        "operator_instrumentation.py",
        "operator_flops.py",
        "profile_hardware.py",
        "run_contract.py",
    ):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("scripts/profile_layers.sh", "scripts/run.sh"):
        path = Path(__file__).parents[1] / name
        result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def tensor_storage_bytes(value):
    """Count diagnostic snapshot storage by identity, including nested pool state."""
    seen = {}

    def visit(item):
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            seen[(str(item.device), storage.data_ptr())] = storage.nbytes()
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return sum(seen.values())


def comparison(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise RuntimeError("Nonfinite checkpoint output")
    error = actual.float() - expected.float()
    return {
        "shape": list(actual.shape),
        "bitwise_equal": torch.equal(actual, expected),
        "max_abs": error.abs().max().item(),
        "relative_l2": (error.norm() / expected.float().norm().clamp_min(1e-20)).item(),
        "rtol": 0.01,
        "atol": 0.02,
    }


def timed(model, ids):
    model.synchronize()
    begin = time.perf_counter()
    output = model.forward(ids)
    return output, (time.perf_counter() - begin) * 1000


@torch.inference_mode()
def annotate(model, ids, mode, phase, nsys):
    scopes = OperatorScopes(mode, phase)
    block_outputs = []
    original = model.blocks[-1].forward

    def remember(*args, **kwargs):
        output = original(*args, **kwargs)
        if phase == "extend_annotated":
            # Retain existing tensors, with no additional CUDA work inside capture.
            block_outputs.append(output)
        return output

    model.synchronize()
    if nsys:
        torch.cuda.cudart().cudaProfilerStart()
    with InstrumentOperators(model, scopes), patch.object(model.blocks[-1], "forward", remember):
        begin = time.perf_counter()
        with scopes("forward_misc"):
            output = model.forward(ids, scope=scopes)
        wall = (time.perf_counter() - begin) * 1000
    if nsys:
        torch.cuda.cudart().cudaProfilerStop()
    normalized = None
    if block_outputs:
        normalized = torch.cat(
            [
                rms_norm(
                    hidden.float() + residual.float(), model.final_norm, model.cfg.norm_eps
                ).bfloat16()
                for hidden, residual in block_outputs
            ]
        ).cpu()
    return output.cpu(), normalized, scopes.calls, wall


def cache_metrics(model, snapshot, prefix_metrics):
    return {
        "snapshot_cpu_tensor_bytes": tensor_storage_bytes(snapshot),
        "cache_resource_plan": getattr(model, "_cache_resource_plan", None),
        "prefix_cache_per_layer": prefix_metrics,
        "extend_cache_per_layer": [block.cache.metrics() for block in model.blocks],
    }


def run_benchmark(model, ids, args, result):
    """Only default model outputs and wall timers; no output copies or comparisons."""
    for offload in (False, True):
        mode = "offload" if offload else "resident"
        prefix_times = []
        for _ in range(args.prefill_repeats):
            model.set_cache_mode(offload)
            _, elapsed = timed(model, ids[: args.prefix])
            prefix_times.append(elapsed)
        prefix_metrics = [block.cache.metrics() for block in model.blocks]
        snapshot = model.snapshot_prefix()
        extend_times = []
        for _ in range(args.repeats):
            model.restore_prefix(snapshot)
            _, elapsed = timed(model, ids[args.prefix :])
            extend_times.append(elapsed)
        result["measurements"][mode] = {
            **cache_metrics(model, snapshot, prefix_metrics),
            "prefix_samples_ms": prefix_times,
            "extend_samples_ms": extend_times,
            "prefix_median_ms": statistics.median(prefix_times),
            "extend_median_ms": statistics.median(extend_times),
        }
        del snapshot


def run_check(model, ids, args, result):
    controls, prefixes = {}, {}
    for offload in (False, True):
        mode = "offload" if offload else "resident"
        model.set_cache_mode(offload)
        prefixes[mode] = model.forward(ids[: args.prefix]).cpu()
        snapshot = model.snapshot_prefix()
        model.restore_prefix(snapshot)
        default = model.forward(ids[args.prefix :]).cpu()
        model.restore_prefix(snapshot)
        control = {
            key: value.cpu()
            for key, value in model.forward(ids[args.prefix :], return_hidden=True).items()
        }
        if control["hidden"].ndim != 2 or control["hidden"].shape[0] != args.extend:
            raise ValueError("Check must cover every extend hidden row")
        if control["logits"].ndim != 2 or control["logits"].shape[0] != 1:
            raise ValueError("Check must cover the last-token logits")
        result["correctness"][mode + "_default_extend_logits"] = comparison(
            control["logits"], default
        )
        controls[mode] = control
        torch.save(control, args.output / f"{mode}_control.pt")
        torch.save(prefixes[mode], args.output / f"{mode}_prefix_logits.pt")
        del snapshot
    for key in ("hidden", "logits"):
        result["correctness"]["resident_vs_offload_" + key] = comparison(
            controls["offload"][key], controls["resident"][key]
        )
    result["correctness"]["resident_vs_offload_prefix_logits"] = comparison(
        prefixes["offload"], prefixes["resident"]
    )


def run_profile(model, ids, args, result, receipt):
    calls = []
    for key in ("hidden", "logits"):
        result["correctness"]["resident_vs_offload_" + key] = receipt["checks"]["comparisons"][
            "resident_vs_offload_" + key
        ]
    for offload in (False, True):
        mode = "offload" if offload else "resident"
        control = torch.load(
            receipt["artifact_paths"][mode + "_control.pt"], map_location="cpu", weights_only=True
        )
        prefix_control = torch.load(
            receipt["artifact_paths"][mode + "_prefix_logits.pt"],
            map_location="cpu",
            weights_only=True,
        )
        model.set_cache_mode(offload)
        annotated_prefix, _, records, prefix_wall = annotate(
            model, ids[: args.prefix], mode, "prefill_annotated", args.nsys
        )
        calls.extend(records)
        result["correctness"][mode + "_profile_prefix_logits"] = comparison(
            annotated_prefix, prefix_control
        )
        prefix_metrics = [block.cache.metrics() for block in model.blocks]
        snapshot = model.snapshot_prefix()
        model.restore_prefix(snapshot)
        annotated, hidden, records, extend_wall = annotate(
            model, ids[args.prefix :], mode, "extend_annotated", args.nsys
        )
        calls.extend(records)
        for key, actual in (("logits", annotated), ("hidden", hidden)):
            result["correctness"][mode + "_profile_extend_" + key] = comparison(
                actual, control[key]
            )
        torch.save(
            {"hidden": hidden, "logits": annotated}, args.output / f"{mode}_profile_output.pt"
        )
        result["measurements"][mode] = {
            **cache_metrics(model, snapshot, prefix_metrics),
            "annotated_prefix_wall_ms": prefix_wall,
            "annotated_extend_wall_ms": extend_wall,
        }
        write_json(
            args.output / "operator_calls.json",
            {"schema_version": 2, "run_id": args.run_id, "calls": calls},
        )
        if offload:
            capture_kernel_inputs(model, ids[args.prefix :], snapshot, args)
        del snapshot


def capture_kernel_inputs(model, ids, snapshot, args):
    model.restore_prefix(snapshot)
    for layer, block in enumerate(model.blocks):

        def capture(p, keys, scales, indices, cache, position, layer=layer):
            torch.save(
                {
                    "layer": layer,
                    "query_start": position,
                    "attention_scale": model.cfg.attention_scale,
                    "q": p.q.cpu(),
                    "index_q": p.index_q.cpu(),
                    "index_weights": p.index_weights.cpu(),
                    "index_keys": keys.cpu(),
                    "index_scales": scales.cpu(),
                    "indices": indices.cpu(),
                    "kv": cache.host_records(),
                    "source_run_id": args.run_id,
                },
                args.output / f"kernel_inputs_layer_{layer}.pt",
            )

        block.attention.capture_hook = capture
    try:
        model.forward(ids)
    finally:
        for block in model.blocks:
            block.attention.capture_hook = None
    if not all(
        (args.output / f"kernel_inputs_layer_{layer}.pt").is_file()
        for layer in range(model.num_layers)
    ):
        raise RuntimeError("Profile did not capture every checkpoint layer's kernel inputs")


def run(mode=None, argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__
        if mode == "profile"
        else "Independent numerical check or clean three-layer checkpoint benchmark.",
        allow_abbrev=False,
    )
    parser.add_argument("--model", type=Path, default=Path("/preset-models"))
    parser.add_argument("--request", type=Path)
    parser.add_argument("--prefix", type=int, default=65536)
    parser.add_argument("--extend", type=int, default=1024)
    parser.add_argument(
        "--sparse-pool-tokens",
        "--slots",
        dest="slots",
        type=int,
        default=16384,
        help="usable tokens per shared model layer pool; excludes sentinel row",
    )
    parser.add_argument("--host-arena-tokens", type=int)
    parser.add_argument("--workspace-query-tokens", type=int)
    parser.add_argument("--hbm-cache-budget-gib", type=float, default=5)
    parser.add_argument("--dram-cache-budget-gib", type=float, default=64)
    parser.add_argument(
        "--extend-chunk-size",
        type=int,
        help="explicit extend chunk; default is the complete extend batch",
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prefill-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--physical-device", default="1", help="nvidia-smi index or UUID; CUDA device is 0"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    if mode == "profile":
        parser.add_argument("--nsys", action="store_true")
        parser.add_argument("--benchmark-run", type=Path)
    else:
        parser.add_argument("--mode", choices=("check", "bench"), default="bench")
    parser.add_argument("--validation-receipt", type=Path)
    args = parser.parse_args(argv)
    mode = mode or args.mode
    if mode != "check" and args.validation_receipt is None:
        parser.error("--validation-receipt is required for bench/profile")
    if mode == "check" and args.validation_receipt is not None:
        parser.error("check writes its own receipt; do not supply --validation-receipt")
    if (ROOT / "experiments").resolve() in args.output.resolve().parents:
        parser.error("Execute outside experiments; scripts publish accepted bench/profile staging")
    if args.hbm_cache_budget_gib <= 0 or args.dram_cache_budget_gib <= 0:
        parser.error("cache budgets must be positive")
    required_queries = max(args.chunk_size, args.extend_chunk_size or args.extend)
    if args.workspace_query_tokens is None:
        args.workspace_query_tokens = required_queries
    if args.workspace_query_tokens < required_queries:
        parser.error("workspace-query-tokens must cover prefill and extend query batches")
    if args.extend_chunk_size is not None and args.extend_chunk_size < 1:
        parser.error("extend-chunk-size must be positive")
    if args.host_arena_tokens is not None and (
        args.host_arena_tokens % 64
        or args.host_arena_tokens < (args.prefix + args.extend + 63) // 64 * 64
    ):
        parser.error("host-arena-tokens must contain whole pages covering prefix plus extend")
    if (
        min(
            args.prefix,
            args.extend,
            args.slots,
            args.chunk_size,
            args.warmups,
            args.repeats,
            args.prefill_repeats,
        )
        < 1
    ):
        parser.error("All dimensions and repetition counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    request = (
        json.loads(args.request.read_text())
        if args.request
        else make_request(args.model, args.prefix, args.extend, args.seed)
    )
    if (
        request["stable_prefix_tokens"] != args.prefix
        or request["candidate_suffix_tokens"] != args.extend
        or len(request["input_ids"]) != args.prefix + args.extend
    ):
        raise ValueError("Saved request does not match the exact prefix/extend boundary")
    write_json(args.output / "request.json", request)
    manifest = sources()
    write_json(args.output / "sources.json", manifest)
    for relative in manifest:
        target = args.output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    hardware = gather_hardware(args.physical_device)
    hardware["torch_properties"] = str(torch.cuda.get_device_properties(0))
    hardware["torch_device_uuid"] = str(torch.cuda.get_device_properties(0).uuid)
    if hardware["torch_device_uuid"].removeprefix("GPU-") != hardware["gpu"]["uuid"].removeprefix(
        "GPU-"
    ):
        raise ValueError(
            "CUDA device 0 differs from --physical-device; set CUDA_VISIBLE_DEVICES explicitly"
        )
    if not hardware["is_sm90"] or hardware["pci_identity"]["device_id"] != "2335":
        raise ValueError("MFU reference peaks require the verified H200 SXM device")
    write_json(args.output / "hardware.json", hardware)
    print(f"Loading real checkpoint layers 0–2; input {args.prefix}+{args.extend}", flush=True)
    model = DeepSeekEchoModel(
        args.model,
        devices=[0],
        num_layers=3,
        capacity=args.prefix + args.extend,
        slots=args.slots,
        chunk_size=args.chunk_size,
        extend_chunk_size=args.extend_chunk_size,
        host_arena_tokens=args.host_arena_tokens,
        workspace_query_tokens=args.workspace_query_tokens,
        hbm_cache_budget_bytes=int(args.hbm_cache_budget_gib * 2**30),
        dram_cache_budget_bytes=int(args.dram_cache_budget_gib * 2**30),
    )
    if any(block.is_moe for block in model.blocks):
        raise ValueError("This diagnostic requires the first three dense checkpoint blocks")
    from experiments.deepseek_v32_echo_prefill.src.backend_provenance import (
        collect_backend_provenance,
        collect_flashinfer_runtime_artifacts,
    )

    backend_identity = collect_backend_provenance()
    ids = request["input_ids"]
    result = {
        "schema_version": 2,
        "mode": mode,
        "run_id": args.run_id,
        "accepted": False,
        "scope": "checkpoint_layers_0_1_2_embedding_final_norm_last_token_lm_head",
        "model": str(args.model.resolve()),
        "num_layers": model.num_layers,
        "checkpoint_num_layers": model.cfg.num_hidden_layers,
        "seed": args.seed,
        "prefix_tokens": args.prefix,
        "extend_tokens": args.extend,
        "chunk_size": args.chunk_size,
        "slots": args.slots,
        "cache_policy_revision": "echo-global-pages-fifo-v1",
        "pool_scope": "model_device_per_layer",
        "sparse_pool_tokens": args.slots,
        "host_arena_tokens": args.host_arena_tokens or (args.prefix + args.extend + 63) // 64 * 64,
        "workspace_query_tokens": args.workspace_query_tokens,
        "hbm_cache_budget_bytes": int(args.hbm_cache_budget_gib * 2**30),
        "dram_cache_budget_bytes": int(args.dram_cache_budget_gib * 2**30),
        "extend_chunk_size": args.extend_chunk_size,
        "snapshot_schema": "echo-shared-prefix-v1",
        "snapshot_scope": "shared pools once plus session metadata; diagnostic CPU storage excluded from serving capacity",
        "prefetch_cap": "min(8192, sparse_pool_tokens - actual query batch)",
        "prefetch_flags": {"fused_extend_equivalent": True, "early_evict": False},
        "warmups": args.warmups,
        "repeats": args.repeats,
        "prefill_repeats": args.prefill_repeats,
        "timed_output": "last-token logits; all extend hidden verified by independent check",
        "nsys_capture_order": [
            "resident/prefill_annotated",
            "resident/extend_annotated",
            "offload/prefill_annotated",
            "offload/extend_annotated",
        ]
        if mode == "profile"
        else [],
        "timing": "synchronized wall; independent check, clean bench, and NVTX profile processes",
        "source_sha256": manifest,
        "backend_provenance": backend_identity,
        "compute_precision": dict(model.blocks[0].attention.attention.precision),
        "torch_precision": {
            "matmul_fp32_precision": torch.backends.cuda.matmul.fp32_precision,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "indexer_build": build_info(),
        "request_sha256": hashlib.sha256((args.output / "request.json").read_bytes()).hexdigest(),
        "checkpoint_metadata_sha256": {
            name: hashlib.sha256((args.model / name).read_bytes()).hexdigest()
            for name in ("config.json", "tokenizer.json", "model.safetensors.index.json")
            if (args.model / name).exists()
        },
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "triton", "safetensors", "apache-tvm-ffi")
        },
        "hardware": hardware,
        "measurements": {},
        "correctness": {},
    }
    result["checkpoint_identity"] = checkpoint_identity(args.model)
    result["execution_environment"] = {
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_num_threads": torch.get_num_threads(),
        "variables": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(
                (
                    "CXLDSAGR_",
                    "DG_",
                    "DJ_",
                    "FLASHINFER_",
                    "CUTE_DSL_",
                    "TRITON_",
                    "PYTORCH_",
                    "OMP_",
                    "MKL_",
                    "OPENBLAS_",
                )
            )
            or key in {"CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING", "CUDA_LAUNCH_BLOCKING"}
        },
    }
    # Exercise exactly the default execution path before binding loaded JIT identities.
    # This also completes compilation before any formal sample or profiler capture.
    for offload in (False, True):
        for _ in range(args.warmups):
            model.set_cache_mode(offload)
            model.forward(ids[: args.prefix])
            snapshot = model.snapshot_prefix()
            model.restore_prefix(snapshot)
            model.forward(ids[args.prefix :])
            del snapshot
    result["execution_runtime_artifacts"] = collect_flashinfer_runtime_artifacts()
    result["execution_identity"] = execution_identity(result)
    receipt = None
    if mode != "check":
        receipt = require_receipt(
            args.validation_receipt, kind=RECEIPT_KIND, identity=result["execution_identity"]
        )
        result["validation_receipt"] = receipt_binding(receipt)
    if mode == "profile" and args.benchmark_run is not None:
        result["benchmark"] = bind_benchmark(args.benchmark_run, result)
    if mode == "bench":
        run_benchmark(model, ids, args, result)
    elif mode == "check":
        run_check(model, ids, args, result)
    else:
        run_profile(model, ids, args, result, receipt)
    if checkpoint_identity(args.model) != result["checkpoint_identity"]:
        raise RuntimeError("Checkpoint identity changed during execution")
    if sources() != manifest:
        raise RuntimeError("Implementation changed during measurement")
    if collect_backend_provenance() != backend_identity:
        raise RuntimeError("Official backend libraries changed during measurement")
    result["flashinfer_runtime_artifacts"] = collect_flashinfer_runtime_artifacts()
    result["accepted"] = True
    write_json(args.output / "result.json", result)
    if mode == "check":
        write_receipt(
            args.output / "receipt.json",
            kind=RECEIPT_KIND,
            identity=result["execution_identity"],
            checks={"passed": True, "comparisons": result["correctness"]},
            artifacts={path.name: path for path in args.output.iterdir() if path.is_file()},
        )
    print(
        json.dumps({"run_id": args.run_id, "accepted": True, "correctness": result["correctness"]}),
        flush=True,
    )


def main(argv=None):
    run("profile", argv)


if __name__ == "__main__":
    main()
