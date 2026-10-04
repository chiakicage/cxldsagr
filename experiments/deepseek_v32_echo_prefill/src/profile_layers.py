"""Profile the real first three DeepSeek layers, separately for prefix and extend."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import statistics
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.deepseek_v32_echo_prefill.src.measure import make_request, source_manifest
from experiments.deepseek_v32_echo_prefill.src.operator_instrumentation import (
    InstrumentOperators,
    OperatorScopes,
)
from experiments.deepseek_v32_echo_prefill.src.profile_hardware import gather_hardware
from models.deepseek_v32.echo_infer import DeepSeekEchoModel
from models.deepseek_v32.echo_model import rms_norm
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
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
    parser.add_argument("--nsys", action="store_true")
    args = parser.parse_args()
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
        "schema_version": 1,
        "run_id": args.run_id,
        "accepted": False,
        "scope": "checkpoint_layers_0_1_2_embedding_final_norm_last_token_lm_head",
        "model": str(args.model.resolve()),
        "num_layers": model.num_layers,
        "checkpoint_num_layers": model.cfg.num_hidden_layers,
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
        "timed_output": "last-token logits; all extend hidden verified outside timed ranges",
        "nsys_capture_order": [
            "resident/prefill_annotated",
            "resident/extend_annotated",
            "offload/prefill_annotated",
            "offload/extend_annotated",
        ],
        "timing": "synchronized wall; no wrappers in formal timings; separate NVTX captures",
        "source_sha256": manifest,
        "backend_provenance": backend_identity,
        "compute_precision": dict(model.blocks[0].attention.attention.precision),
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
    calls, controls = [], {}
    for offload in (False, True):
        mode = "offload" if offload else "resident"
        print(f"{mode}: prefix warmup", flush=True)
        for _ in range(args.warmups):
            model.set_cache_mode(offload)
            timed(model, ids[: args.prefix])
        prefix_times = []
        for iteration in range(args.prefill_repeats):
            model.set_cache_mode(offload)
            prefix_output, elapsed = timed(model, ids[: args.prefix])
            prefix_times.append(elapsed)
            print(f"{mode} prefix {iteration}: {elapsed:.3f} ms", flush=True)
        prefix_control = prefix_output.cpu()
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
        snapshot_bytes = tensor_storage_bytes(snapshot)
        for _ in range(args.warmups):
            model.restore_prefix(snapshot)
            timed(model, ids[args.prefix :])
        extend_times = []
        for iteration in range(args.repeats):
            model.restore_prefix(snapshot)
            output, elapsed = timed(model, ids[args.prefix :])
            extend_times.append(elapsed)
            print(f"{mode} extend {iteration}: {elapsed:.3f} ms", flush=True)
        extend_control = output.cpu()
        model.restore_prefix(snapshot)
        control = {
            key: value.cpu()
            for key, value in model.forward(ids[args.prefix :], return_hidden=True).items()
        }
        comparison(control["logits"], extend_control)
        controls[mode] = control
        torch.save(control, args.output / f"{mode}_control.pt")
        model.restore_prefix(snapshot)
        annotated, hidden, records, extend_wall = annotate(
            model, ids[args.prefix :], mode, "extend_annotated", args.nsys
        )
        calls.extend(records)
        result["correctness"][mode + "_profile_extend_logits"] = comparison(
            annotated, extend_control
        )
        result["correctness"][mode + "_profile_extend_hidden"] = comparison(
            hidden, control["hidden"]
        )
        torch.save(
            {"hidden": hidden, "logits": annotated}, args.output / f"{mode}_profile_output.pt"
        )
        result["measurements"][mode] = {
            "snapshot_cpu_tensor_bytes": snapshot_bytes,
            "cache_resource_plan": getattr(model, "_cache_resource_plan", None),
            "prefix_samples_ms": prefix_times,
            "extend_samples_ms": extend_times,
            "prefix_median_ms": statistics.median(prefix_times),
            "extend_median_ms": statistics.median(extend_times),
            "annotated_prefix_wall_ms": prefix_wall,
            "annotated_extend_wall_ms": extend_wall,
            "prefix_cache_per_layer": prefix_metrics,
            "extend_cache_per_layer": [block.cache.metrics() for block in model.blocks],
        }
        write_json(
            args.output / "operator_calls.json",
            {"schema_version": 1, "run_id": args.run_id, "calls": calls},
        )
        if offload:
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
            model.forward(ids[args.prefix :])
            for block in model.blocks:
                block.attention.capture_hook = None
    for key in ("hidden", "logits"):
        result["correctness"]["resident_vs_offload_" + key] = comparison(
            controls["offload"][key], controls["resident"][key]
        )
    if sources() != manifest:
        raise RuntimeError("Implementation changed during measurement")
    if collect_backend_provenance() != backend_identity:
        raise RuntimeError("Official backend libraries changed during measurement")
    result["flashinfer_runtime_artifacts"] = collect_flashinfer_runtime_artifacts()
    result["accepted"] = True
    write_json(args.output / "result.json", result)
    print(
        json.dumps({"run_id": args.run_id, "accepted": True, "correctness": result["correctness"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
