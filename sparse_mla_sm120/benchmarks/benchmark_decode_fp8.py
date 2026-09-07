"""Standalone V32 decode benchmark for flash_mla_sm120.

Measures CUDA-event latency and estimates effective FLOP/s and bandwidth.
This does not use Nsight Compute counters; bandwidth is computed from the
logical bytes touched by q, packed KV, and output.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

import flash_mla_sm120


@dataclass(frozen=True)
class V32Config:
    d_qk: int = 576
    d_v: int = 512
    d_nope: int = 512
    d_rope: int = 64
    default_topk: int = 2048
    bytes_per_token: int = 656
    block_size: int = 64


CFG = V32Config()


def _cast_scale_inv_to_ue8m0(scales_inv: torch.Tensor) -> torch.Tensor:
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil())


def quantize_kv_v32(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack BF16 KV into the V32 FlashMLA FP8 cache layout.

    Input:  [num_blocks, block_size, 1, 576] bf16
    Output: [num_blocks, block_size, 1, 656] uint8
    """
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    nb, bs, hk, d = kv_bf16.shape
    assert d == 576 and hk == 1
    kv = kv_bf16.squeeze(2)

    bpt = d_nope + num_tiles * 4 + d_rope * 2
    result = torch.zeros(nb, bs, bpt, dtype=torch.uint8, device=kv.device)

    for ti in range(num_tiles):
        tile = kv[..., ti * tile_size : (ti + 1) * tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        result[..., ti * tile_size : (ti + 1) * tile_size] = fp8.view(torch.uint8)
        scale_bytes = scale.to(torch.float32).contiguous().view(torch.uint8).reshape(nb, bs, 4)
        result[..., d_nope + ti * 4 : d_nope + (ti + 1) * 4] = scale_bytes

    rope = kv[..., d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8)
    result[..., d_nope + num_tiles * 4 :] = rope.reshape(nb, bs, d_rope * 2)
    return result.view(nb, bs, 1, bpt).contiguous()


def flops(batch_size: int, num_heads: int, topk: int) -> float:
    return 2.0 * batch_size * num_heads * topk * (CFG.d_qk + CFG.d_v)


def logical_bytes(batch_size: int, num_heads: int, topk: int) -> int:
    kv = batch_size * topk * CFG.bytes_per_token
    q = batch_size * num_heads * CFG.d_qk * 2
    out = batch_size * num_heads * CFG.d_v * 2
    return kv + q + out


def repeated_kv_bytes(batch_size: int, num_heads: int, topk: int) -> int:
    """Approximate traffic after accounting for per-head-group KV rereads.

    The decode kernel processes at most 16 Q heads per CTA group. Different
    head groups gather the same selected KV entries into separate shared-memory
    buffers, so the packed KV traffic scales by ceil(num_heads / 16). This is
    still an estimate: it does not include scheduler metadata, partial buffers,
    combine traffic, or L2 hit effects.
    """
    head_groups = (num_heads + 15) // 16
    kv = batch_size * topk * CFG.bytes_per_token * head_groups
    q = batch_size * num_heads * CFG.d_qk * 2
    out = batch_size * num_heads * CFG.d_v * 2
    return kv + q + out


def percentile(sorted_values: list[float], pct: float) -> float:
    idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * pct))))
    return sorted_values[idx]


def bench(fn, warmup: int, reps: int, use_cuda_graph: bool) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if use_cuda_graph:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            fn()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fn()
        torch.cuda.synchronize()

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts[i].record()
            graph.replay()
            ends[i].record()
    else:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            starts[i].record()
            fn()
            ends[i].record()

    torch.cuda.synchronize()
    times_us = sorted(start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends))
    return {
        "p50_us": percentile(times_us, 0.50),
        "p95_us": percentile(times_us, 0.95),
        "min_us": times_us[0],
    }


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", default="16,64,128", help="Comma-separated head counts.")
    parser.add_argument("--batch-sizes", default="1,4,8", help="Comma-separated decode batch sizes.")
    parser.add_argument("--num-blocks", type=int, default=1024, help="KV cache blocks. 1024 blocks = 64K tokens.")
    parser.add_argument("--topk", type=int, default=CFG.default_topk, help="Sparse KV entries per decode token.")
    parser.add_argument("--topk-sweep", help="Comma-separated topk values. Overrides --topk.")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--no-cuda-graph", action="store_true", help="Measure normal launches instead of graph replay.")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    cache_tokens = args.num_blocks * CFG.block_size
    use_cuda_graph = not args.no_cuda_graph
    topk_values = parse_int_list(args.topk_sweep) if args.topk_sweep else [args.topk]

    print(f"Device: {props.name}, SM {props.major}{props.minor}, SMs={props.multi_processor_count}")
    print(
        f"V32 decode: d_qk={CFG.d_qk}, d_v={CFG.d_v}, topk={','.join(map(str, topk_values))}, "
        f"KV={CFG.bytes_per_token} B/token, cache_tokens={cache_tokens}"
    )
    print(f"Timing: {'CUDA graph replay' if use_cuda_graph else 'normal launches'}, warmup={args.warmup}, reps={args.reps}")
    print()

    kv_bf16 = (
        torch.randn(args.num_blocks, CFG.block_size, 1, CFG.d_qk, device=device, dtype=torch.bfloat16) / 10
    ).clamp(-1, 1)
    kv_cache = quantize_kv_v32(kv_bf16)
    sm_scale = CFG.d_qk ** -0.5

    header = (
        f"{'topk':>6} {'heads':>5} {'bs':>3} {'p50 us':>9} {'p95 us':>9} {'min us':>9} "
        f"{'GFLOP':>8} {'logMB':>8} {'repMB':>8} {'TFLOP/s':>9} {'logGB/s':>9} {'repGB/s':>9}"
    )
    print(header)
    print("-" * len(header))

    for topk in topk_values:
        if topk > cache_tokens:
            raise ValueError(f"topk={topk} exceeds cache_tokens={cache_tokens}")
        for num_heads in parse_int_list(args.heads):
            for batch_size in parse_int_list(args.batch_sizes):
                q = (
                    torch.randn(batch_size, num_heads, CFG.d_qk, device=device, dtype=torch.bfloat16) / 10
                ).contiguous()
                indices = torch.randint(
                    0, cache_tokens, (batch_size, topk), device=device, dtype=torch.int32
                ).contiguous()
                indices[:, -min(10, topk):] = -1

                def run():
                    return flash_mla_sm120.sparse_mla_decode_fwd(q, kv_cache, indices, sm_scale, CFG.d_v)

                timing = bench(run, args.warmup, args.reps, use_cuda_graph)
                f = flops(batch_size, num_heads, topk)
                b = logical_bytes(batch_size, num_heads, topk)
                rb = repeated_kv_bytes(batch_size, num_heads, topk)
                seconds = timing["p50_us"] * 1e-6
                tflops = f / seconds / 1e12
                gbps = b / seconds / 1e9
                repeated_gbps = rb / seconds / 1e9
                print(
                    f"{topk:6d} {num_heads:5d} {batch_size:3d} "
                    f"{timing['p50_us']:9.1f} {timing['p95_us']:9.1f} {timing['min_us']:9.1f} "
                    f"{f / 1e9:8.3f} {b / 1e6:8.3f} {rb / 1e6:8.3f} "
                    f"{tflops:9.2f} {gbps:9.1f} {repeated_gbps:9.1f}"
                )


if __name__ == "__main__":
    main()
