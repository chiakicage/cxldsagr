"""Replay captured full-model activations for one selected Nsight Compute kernel.

This is an isolated kernel diagnostic. Offload uses a freshly reset cold HBM
pool; recall copies the captured selection's entire historical-token union.
Neither replay reconstructs the original model execution's cache-hit state or
measures its end-to-end residual-transfer overhead.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

import torch

from experiments.deepseek_v32_echo_prefill.src.backend_provenance import source_files

KERNELS = ("indexer-resident", "indexer-offload", "mla", "recall")
KERNEL_REGEX = {
    "indexer-resident": r"deep_gemm::sm90_fp8_mqa_logits<",
    "indexer-offload": r"echo_native::sm90_fp8_mqa_logits_fuse_prefetch<",
    "mla": r"sparse_attn_fwd_kernel",
    "recall": r"gather_records",
}
_ROOT = Path(__file__).resolve().parents[3]


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_inputs(path):
    """Validate captured CPU tensors without importing or initializing CUDA backends."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise TypeError("kernel input must be a tensor/metadata dictionary")
    required = {
        "q",
        "index_q",
        "index_weights",
        "index_keys",
        "index_scales",
        "indices",
        "kv",
        "query_start",
        "attention_scale",
        "source_run_id",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"capture is missing fields: {sorted(missing)}")
    q, keys, indices = data["q"], data["index_keys"], data["indices"]
    if not isinstance(q, torch.Tensor) or q.ndim != 3 or q.shape[1:] != (128, 576):
        raise ValueError("captured MLA q must have shape [queries,128,576]")
    if not isinstance(keys, torch.Tensor) or keys.ndim != 2 or keys.shape[1] != 128:
        raise ValueError("captured index keys must have shape [tokens,128]")
    rows, columns = len(q), len(keys)
    if not rows or not columns:
        raise ValueError("captured query and key sequences must be nonempty")
    shapes = {
        "q": ((rows, 128, 576), torch.bfloat16),
        "index_q": ((rows, 64, 128), torch.float8_e4m3fn),
        "index_weights": ((rows, 64), torch.float32),
        "index_keys": ((columns, 128), torch.float8_e4m3fn),
        "index_scales": ((columns,), torch.float32),
        "kv": ((columns, 576), torch.bfloat16),
    }
    for name, (shape, dtype) in shapes.items():
        tensor = data[name]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype} {shape}")
        data[name] = tensor.contiguous()
    if (
        not isinstance(indices, torch.Tensor)
        or indices.ndim != 2
        or indices.shape[0] != rows
        or not indices.shape[1]
        or indices.dtype != torch.int32
    ):
        raise ValueError("captured indices must be int32 [queries,selected_tokens]")
    data["indices"] = indices.contiguous()
    start = data["query_start"]
    if type(start) is not int or not 0 <= start <= columns - rows:
        raise ValueError("query_start must place all captured query rows inside the KV sequence")
    if not isinstance(data["source_run_id"], str) or not data["source_run_id"]:
        raise ValueError("source_run_id must identify the full-model capture")
    if not math.isfinite(float(data["attention_scale"])):
        raise ValueError("attention_scale must be finite")
    causal_ends = torch.arange(start + 1, start + rows + 1)[:, None]
    if ((indices < -1) | (indices >= columns) | ((indices >= 0) & (indices >= causal_ends))).any():
        raise ValueError("captured indices must contain causal logical token IDs or -1 padding")
    return data


def source_manifest():
    files = {
        *source_files(),
        Path(__file__),
        Path(__file__).parents[1] / "scripts/ncu.sh",
        *(
            path
            for directory in ("operators/deepseek_v32", "operators/common")
            for path in (_ROOT / directory).rglob("*")
            if path.is_file()
            and "tests" not in path.relative_to(_ROOT / directory).parts
            and path.suffix in {".py", ".cu", ".cuh", ".cpp", ".h", ".hpp"}
        ),
    }
    return {str(path.relative_to(_ROOT)): file_sha256(path) for path in sorted(files)}


def prepare_replay(data, kernel, pool_slots, prefetch_limit):
    """Return reset/launch/verify callables; all setup is outside the profile range."""
    device = torch.device("cuda", 0)
    rows, columns = len(data["q"]), len(data["kv"])
    start = data["query_start"]
    boundary = {"model_cache_state_reconstructed": False}

    def reset():
        pass

    if kernel.startswith("indexer"):
        from operators.deepseek_v32.indexer.echo import logits

        q, k, weights, scales = (
            data[name].to(device)
            for name in ("index_q", "index_keys", "index_weights", "index_scales")
        )
        state = None
        if kernel == "indexer-offload":
            if pool_slots < rows:
                raise ValueError(
                    "pool_slots must hold the pending query chunk, excluding sentinel zero"
                )
            limit = min(prefetch_limit, pool_slots - rows, 8192)
            host_capacity = (columns + 63) // 64 * 64
            host = torch.empty((host_capacity, 576), dtype=torch.bfloat16, pin_memory=True)
            host[:columns].copy_(data["kv"])
            pool = torch.empty((pool_slots + 1, 576), device=device, dtype=torch.bfloat16)
            h2d = torch.empty(host_capacity, device=device, dtype=torch.int32)
            d2h = torch.empty(pool_slots + 1, device=device, dtype=torch.int64)
            slots = torch.arange(1, pool_slots + 1, device=device, dtype=torch.int32)
            state = {
                "host": host,
                "device": pool,
                "host_to_device": h2d,
                "device_to_host": d2h,
                "free_slots": slots,
                "page_table": torch.arange(host_capacity // 64, device=device, dtype=torch.int32),
                "history_length": start,
                "allocation_log": torch.full(
                    (pool_slots + 1,), 2**31 - 1, device=device, dtype=torch.int64
                ),
                "prefetch_stats": torch.zeros(3, device=device, dtype=torch.int64),
                "counter": torch.zeros(1, device=device, dtype=torch.uint32),
                "offset": torch.zeros(16, device=device, dtype=torch.float32),
                "max_prefetch": limit,
            }

            def reset():
                h2d.fill_(2**31 - 1)
                d2h.fill_(2**31 - 1)
                pool[0].zero_()
                state["counter"].zero_()

            boundary.update(
                replay="actual_activations_with_cold_historical_pool",
                pool_slots=pool_slots,
                pending_current_records=rows,
                current_records_installed=False,
                padding_slots=1,
                page_size=64,
                cache_policy_revision="echo-global-pages-fifo-v1",
                historical_resident_records_before_launch=0,
                max_prefetch=limit,
                coarse_bin_offset=0.0,
                record_bytes=1152,
                excludes="exact_topk_residual_recall_attention_and_original_cache_hit_state",
            )
        else:
            boundary.update(
                replay="actual_activations_with_resident_index_keys", excludes="exact_topk"
            )

        def launch():
            return logits(q, k, weights, scales, start, prefetch=state)

        def verify(output):
            if torch.isnan(output).any() or torch.isposinf(output).any():
                raise RuntimeError("indexer replay produced invalid logits")
            result = {"logits_shape": list(output.shape)}
            if state is not None:
                fetched = torch.nonzero(h2d[:start] != 2**31 - 1).flatten()
                count = len(fetched)
                attempts = int(state["counter"].item())
                if (
                    count != state["prefetch_stats"][0].item()
                    or count != (state["allocation_log"] != 2**31 - 1).sum().item()
                ):
                    raise RuntimeError("prefetch counter and published cache mappings disagree")
                torch.testing.assert_close(
                    pool[h2d[fetched].long()].cpu(), host[fetched.cpu()], rtol=0, atol=0
                )
                if not h2d[start:columns].eq(2**31 - 1).all() or not pool[0].eq(0).all():
                    raise RuntimeError("pending suffix or padding sentinel was overwritten")
                result.update(
                    prefetched_records=count,
                    reservation_attempts=attempts,
                    actual_evictions=int(state["prefetch_stats"][1].item()),
                    rejected_reservations=int(state["prefetch_stats"][2].item()),
                    copied_bytes=1152 * count,
                )
            return result

    elif kernel == "mla":
        from operators.deepseek_v32.attention.device_only.mla import sparse_mla

        q, kv, indices = (data[name].to(device) for name in ("q", "kv", "indices"))

        def launch():
            return sparse_mla(q, kv, indices, float(data["attention_scale"]))

        def verify(output):
            if not torch.isfinite(output).all():
                raise RuntimeError("MLA replay produced nonfinite output")
            return {"output_shape": list(output.shape)}

        boundary.update(
            replay="actual_queries_and_logical_selection_with_resident_full_kv",
            excludes="indexer_topk_transfers_and_original_physical_pool_layout",
        )
    else:
        from operators.common.kv_transfer import gather_host_records

        selected = data["indices"]
        host_ids_cpu = selected[(selected >= 0) & (selected < start)].unique(sorted=True).long()
        if not len(host_ids_cpu):
            raise ValueError("capture has no selected historical records to profile")
        host = torch.empty_like(data["kv"], pin_memory=True).copy_(data["kv"])
        pool = torch.empty((len(host_ids_cpu), 576), dtype=torch.bfloat16, device=device)
        host_ids = host_ids_cpu.to(device)
        device_ids = torch.arange(len(host_ids_cpu), device=device)

        def launch():
            gather_host_records(host, pool, host_ids, device_ids)
            return pool

        def verify(output):
            torch.testing.assert_close(output.cpu(), host[host_ids_cpu], rtol=0, atol=0)
            return {"copied_records": len(host_ids_cpu), "copied_bytes": 1152 * len(host_ids_cpu)}

        boundary.update(
            replay="independent_cold_union_transfer_diagnostic",
            selected_historical_union_records=len(host_ids_cpu),
            destination_slots=len(host_ids_cpu),
            excludes="prefetch_hits_bounded_pool_batches_and_end_to_end_residual_transfer_cost",
            original_residual_replay=False,
        )
    return reset, launch, verify, boundary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="kernel_inputs_layer_<layer>.pt capture"
    )
    parser.add_argument("--kernel", choices=KERNELS, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--pool-slots", type=int, default=16384)
    parser.add_argument("--prefetch-limit", type=int, default=8192)
    parser.add_argument("--run-id")
    parser.add_argument("--capture-label", default="manual")
    parser.add_argument(
        "--metadata", type=Path, help="write accepted replay metadata after verification"
    )
    args = parser.parse_args()
    if args.warmups < 1 or args.pool_slots < 1 or not 0 <= args.prefetch_limit <= 8192:
        parser.error("warmups/pool-slots must be positive and prefetch-limit must be in [0,8192]")
    if args.metadata is not None and args.metadata.exists():
        parser.error("metadata path already exists; choose a fresh profile run")
    data = load_inputs(args.input)
    sources = source_manifest()
    torch.set_num_threads(8)
    torch.cuda.init()
    if torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("the captured DeepSeek ECHO kernels require SM90/Hopper")
    reset, launch, verify, boundary = prepare_replay(
        data, args.kernel, args.pool_slots, args.prefetch_limit
    )
    for _ in range(args.warmups):
        reset()
        output = launch()
        torch.cuda.synchronize()
    from experiments.deepseek_v32_echo_prefill.src.backend_provenance import (
        collect_backend_provenance,
        collect_flashinfer_runtime_artifacts,
    )

    backend_identity = collect_backend_provenance()
    reset()
    torch.cuda.synchronize()
    # NCU's kernel-name filter selects exactly one computational launch. The
    # indexer adapter's metadata initialization and causal-mask kernel are excluded.
    torch.cuda.profiler.start()
    try:
        output = launch()
        torch.cuda.synchronize()
    finally:
        torch.cuda.profiler.stop()
    verification = verify(output)
    if source_manifest() != sources:
        raise RuntimeError("kernel or harness source changed during replay")
    if collect_backend_provenance() != backend_identity:
        raise RuntimeError("Official backend libraries changed during replay")
    metadata = {
        "run_id": args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ"),
        "source_run_id": data["source_run_id"],
        "capture_label": args.capture_label,
        "input": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input),
        "layer": data.get("layer"),
        "kernel": args.kernel,
        "ncu_kernel_regex": KERNEL_REGEX[args.kernel],
        "warmups_before_profiler_start": args.warmups,
        "selected_launches_inside_profiler_api_range": 1,
        "query_start": data["query_start"],
        "tensor_metadata": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in data.items()
            if isinstance(value, torch.Tensor)
        },
        "measurement_boundary": boundary,
        "verification": verification,
        "source_sha256": sources,
        "backend_provenance": backend_identity,
        "flashinfer_runtime_artifacts": collect_flashinfer_runtime_artifacts(),
        "hardware": str(torch.cuda.get_device_properties(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dependencies": {
            name: importlib.metadata.version(name) for name in ("torch", "triton", "apache-tvm-ffi")
        },
        "accepted": True,
        "acceptance_scope": "captured_input_validation_and_replay_output_verification",
    }
    serialized = json.dumps(metadata, indent=2) + "\n"
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(serialized)
    print(serialized, end="", flush=True)


if __name__ == "__main__":
    main()
