"""Benchmark DeepGEMM SM120 kernels on DeepSeek v3.2 attention shapes.

The script measures only DeepGEMM kernel time. Random tensor creation,
FP8/FP4 quantization, scale layout preparation, and metadata construction are
outside the timed region.

Example:
    DG_JIT_CACHE_DIR=/home/cage/dsa/.deep_gemm_cache \\
      .venv/bin/python docs/deepgemm_v32_benchmark.py --quick \\
      --output docs/deepgemm_v32_5080_results.md
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("DG_JIT_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".deep_gemm_cache"))

import torch

import deep_gemm
from deep_gemm.utils import (
    per_block_cast_to_fp8,
    per_custom_dims_cast_to_fp8,
    per_token_cast_to_fp4,
    per_token_cast_to_fp8,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "docs" / "config.json"


@dataclass(frozen=True)
class KernelResult:
    section: str
    name: str
    mode: str
    shape: str
    ms: float
    tflops: float
    gbps: float
    notes: str = ""


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def tensor_nbytes(x) -> int:
    if x is None:
        return 0
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    if isinstance(x, (tuple, list)):
        return sum(tensor_nbytes(v) for v in x)
    return 0


def bench_cuda(fn: Callable[[], object], warmups: int, iters: int) -> float:
    fn()
    torch.cuda.synchronize()
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def quantize_matrix(x: torch.Tensor, mode: str, operand: str):
    if mode == "fp8":
        if operand == "a":
            return per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=128), None, None, False
        return per_block_cast_to_fp8(x, use_ue8m0=True, gran_k=128), None, None, False

    if mode == "fp8_fp4w":
        if operand == "a":
            return per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=128), (1, 128), None, False
        return per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32), None, (1, 32), False

    if mode == "fp4":
        recipe = (1, 32)
        return per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32), recipe, recipe, False

    raise ValueError(f"unknown mode: {mode}")


def benchmark_gemm(name: str, m: int, n: int, k: int, mode: str, warmups: int, iters: int) -> KernelResult:
    a_raw = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_raw = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    a, recipe_a, _, disable = quantize_matrix(a_raw, mode, "a")
    b, _, recipe_b, _ = quantize_matrix(b_raw, mode, "b")
    del a_raw, b_raw
    d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    def run():
        deep_gemm.fp8_fp4_gemm_nt(
            a,
            b,
            d,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
            disable_ue8m0_cast=disable,
        )

    ms = bench_cuda(run, warmups, iters)
    flops = 2 * m * n * k
    bytes_ = tensor_nbytes(a) + tensor_nbytes(b) + tensor_nbytes(d)
    result = KernelResult("projection_gemm", name, mode, f"m={m}, n={n}, k={k}", ms, flops / (ms * 1e9), bytes_ / (ms * 1e6))
    del a, b, d
    cleanup()
    return result


def grouped_quantize(x: torch.Tensor, mode: str, operand: str):
    groups, mn, k = x.shape
    if mode == "fp8":
        data = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        if operand == "a":
            sf = torch.empty((groups, mn, ceil_div(k, 128)), device="cuda", dtype=torch.float32)
            for i in range(groups):
                data[i], sf[i] = per_token_cast_to_fp8(x[i], use_ue8m0=True, gran_k=128)
        else:
            sf = torch.empty((groups, ceil_div(mn, 128), ceil_div(k, 128)), device="cuda", dtype=torch.float32)
            for i in range(groups):
                data[i], sf[i] = per_block_cast_to_fp8(x[i], use_ue8m0=True, gran_k=128)
        return (data, sf), None, None, False

    if mode == "fp8_fp4w":
        if operand == "a":
            data = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            sf = torch.empty((groups, mn, ceil_div(k, 128)), device="cuda", dtype=torch.float32)
            for i in range(groups):
                data[i], sf[i] = per_token_cast_to_fp8(x[i], use_ue8m0=True, gran_k=128)
            return (data, sf), (1, 128), None, False

        data = torch.empty((groups, mn, k // 2), device="cuda", dtype=torch.int8)
        sf = torch.empty((groups, mn, ceil_div(k, 32)), device="cuda", dtype=torch.float32)
        for i in range(groups):
            data[i], sf[i] = per_token_cast_to_fp4(x[i], use_ue8m0=True, gran_k=32)
        return (data, sf), None, (1, 32), False

    raise ValueError(f"grouped mode not supported: {mode}")


def benchmark_grouped_gemm(name: str, batch: int, groups: int, n: int, k: int, mode: str, warmups: int, iters: int) -> KernelResult:
    max_m = batch
    a_raw = torch.randn((groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    b_raw = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)
    a, recipe_a, _, disable = grouped_quantize(a_raw, mode, "a")
    b, _, recipe_b, _ = grouped_quantize(b_raw, mode, "b")
    del a_raw, b_raw
    d = torch.empty((groups, max_m, n), device="cuda", dtype=torch.bfloat16)
    masked_m = torch.full((groups,), batch, device="cuda", dtype=torch.int32)

    def run():
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
            a,
            b,
            d,
            masked_m,
            batch,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
            disable_ue8m0_cast=disable,
        )

    ms = bench_cuda(run, warmups, iters)
    flops = 2 * groups * batch * n * k
    bytes_ = tensor_nbytes(a) + tensor_nbytes(b) + tensor_nbytes(d) + tensor_nbytes(masked_m)
    result = KernelResult("head_grouped_gemm", name, mode, f"groups={groups}, m/group={batch}, n={n}, k={k}", ms, flops / (ms * 1e9), bytes_ / (ms * 1e6))
    del a, b, d, masked_m
    cleanup()
    return result


def quantize_mqa_q(q: torch.Tensor, mode: str):
    seq_or_batch_heads = q.view(-1, q.shape[-1])
    if mode == "fp8":
        return q.to(torch.float8_e4m3fn), None
    if mode == "fp4":
        q_fp4, q_sf = per_token_cast_to_fp4(seq_or_batch_heads, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        return q_fp4.view(*q.shape[:-1], q.shape[-1] // 2), q_sf.view(*q.shape[:-1])
    raise ValueError(mode)


def quantize_mqa_kv(kv: torch.Tensor, mode: str):
    if mode == "fp8":
        return per_custom_dims_cast_to_fp8(kv, (0,), use_ue8m0=False)
    if mode == "fp4":
        data, sf = per_token_cast_to_fp4(kv, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        return data, sf.view(-1)
    raise ValueError(mode)


def benchmark_mqa_logits(seq_len: int, seq_len_kv: int, heads: int, dim: int, mode: str, warmups: int, iters: int) -> KernelResult:
    q_raw = torch.randn((seq_len, heads, dim), device="cuda", dtype=torch.bfloat16)
    kv_raw = torch.randn((seq_len_kv, dim), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((seq_len, heads), device="cuda", dtype=torch.float32)
    q = quantize_mqa_q(q_raw, mode)
    kv = quantize_mqa_kv(kv_raw, mode)
    del q_raw, kv_raw
    ks = torch.zeros((seq_len,), device="cuda", dtype=torch.int32)
    ke = torch.arange(seq_len, device="cuda", dtype=torch.int32) + (seq_len_kv - seq_len + 1)
    ke.clamp_(max=seq_len_kv)

    def run():
        deep_gemm.fp8_fp4_mqa_logits(
            q=q,
            kv=kv,
            weights=weights,
            cu_seq_len_k_start=ks,
            cu_seq_len_k_end=ke,
            clean_logits=False,
            logits_dtype=torch.bfloat16,
        )

    ms = bench_cuda(run, warmups, iters)
    cost = int((ke - ks).sum().item())
    flops = 2 * cost * heads * dim
    bytes_ = tensor_nbytes(q) + tensor_nbytes(kv) + tensor_nbytes(weights) + tensor_nbytes(ks) + tensor_nbytes(ke) + cost * 2
    result = KernelResult("indexer_mqa_logits", "prefill_index_score", mode, f"s={seq_len}, skv={seq_len_kv}, h={heads}, d={dim}", ms, flops / (ms * 1e9), bytes_ / (ms * 1e6))
    del q, kv, weights, ks, ke
    cleanup()
    return result


def make_fused_kv_cache(num_blocks: int, block_kv: int, dim: int, mode: str):
    raw = torch.randn((num_blocks, block_kv, 1, dim), device="cuda", dtype=torch.bfloat16)
    if mode == "fp8":
        amax = raw.abs().float().amax(dim=3, keepdim=True).clamp_min(1e-4)
        sf = amax / 448.0
        scaled = (raw * (1.0 / sf)).to(torch.float8_e4m3fn)
        fused = torch.empty((num_blocks, block_kv * (dim + 4)), device="cuda", dtype=torch.uint8)
        fused[:, : block_kv * dim] = scaled.view(num_blocks, block_kv * dim).view(torch.uint8)
        fused[:, block_kv * dim :] = sf.view(num_blocks, block_kv).view(torch.uint8)
        del raw, amax, sf, scaled
        return fused.view(num_blocks, block_kv, 1, dim + 4)

    if mode == "fp4":
        packed, sf = per_token_cast_to_fp4(raw.view(-1, dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        fused = torch.empty((num_blocks, block_kv * (dim // 2 + 4)), device="cuda", dtype=torch.uint8)
        fused[:, : block_kv * (dim // 2)] = packed.view(num_blocks, block_kv * (dim // 2)).view(torch.uint8)
        fused[:, block_kv * (dim // 2) :] = sf.view(num_blocks, block_kv).view(torch.uint8)
        del raw, packed, sf
        return fused.view(num_blocks, block_kv, 1, dim // 2 + 4)

    raise ValueError(mode)


def benchmark_paged_mqa(batch: int, avg_kv: int, heads: int, dim: int, mode: str, warmups: int, iters: int) -> KernelResult:
    next_n = 1
    block_kv = 64
    blocks_per_seq = ceil_div(avg_kv, block_kv)
    max_context_len = blocks_per_seq * block_kv
    num_blocks = batch * blocks_per_seq

    q_raw = torch.randn((batch, next_n, heads, dim), device="cuda", dtype=torch.bfloat16)
    q = quantize_mqa_q(q_raw, mode)
    del q_raw
    weights = torch.randn((batch * next_n, heads), device="cuda", dtype=torch.float32)
    kv_cache = make_fused_kv_cache(num_blocks, block_kv, dim, mode)
    context_lens = torch.full((batch, 1), avg_kv, device="cuda", dtype=torch.int32)
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(batch, blocks_per_seq)
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv, deep_gemm.get_num_sms())

    def run():
        deep_gemm.fp8_fp4_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            max_context_len,
            False,
            torch.bfloat16,
        )

    ms = bench_cuda(run, warmups, iters)
    cost = batch * avg_kv * next_n
    flops = 2 * cost * heads * dim
    bytes_ = tensor_nbytes(q) + tensor_nbytes(kv_cache) + tensor_nbytes(weights) + tensor_nbytes(context_lens) + tensor_nbytes(block_table) + cost * 2
    result = KernelResult("indexer_paged_mqa_logits", "decode_index_score", mode, f"batch={batch}, skv={avg_kv}, h={heads}, d={dim}, block={block_kv}", ms, flops / (ms * 1e9), bytes_ / (ms * 1e6))
    del q, weights, kv_cache, context_lens, block_table, schedule_meta
    cleanup()
    return result


def projection_cases(cfg: dict, quick: bool) -> Iterable[tuple[str, int, int, int]]:
    dim = cfg["dim"]
    heads = cfg["n_heads"]
    q_lora = cfg["q_lora_rank"]
    kv_lora = cfg["kv_lora_rank"]
    qk_nope = cfg["qk_nope_head_dim"]
    qk_rope = cfg["qk_rope_head_dim"]
    v_dim = cfg["v_head_dim"]
    index_heads = cfg["index_n_heads"]
    index_dim = cfg["index_head_dim"]

    decode_ms = [1, 16, 128] if not quick else [1, 128]
    prefill_ms = [512, 2048] if not quick else [512]

    base = [
        ("wq_a", q_lora, dim),
        ("wq_b", heads * (qk_nope + qk_rope), q_lora),
        ("wkv_a", kv_lora + qk_rope, dim),
        ("wkv_b_prefill", heads * (qk_nope + v_dim), kv_lora),
        ("wo", dim, heads * v_dim),
        ("index_wqi", index_heads * index_dim, q_lora),
        ("index_wki", index_dim, dim),
        ("index_weights", index_heads, dim),
    ]
    for m in decode_ms:
        for name, n, k in base:
            if name == "wkv_b_prefill":
                continue
            yield f"decode_m{m}_{name}", m, n, k
    for m in prefill_ms:
        for name, n, k in base:
            yield f"prefill_m{m}_{name}", m, n, k


def grouped_cases(cfg: dict, quick: bool) -> Iterable[tuple[str, int, int, int, int]]:
    groups = cfg["n_heads"]
    batches = [1, 16, 128] if not quick else [1, 128]
    for b in batches:
        yield f"decode_b{b}_wk_b", b, groups, cfg["kv_lora_rank"], cfg["qk_nope_head_dim"]
        yield f"decode_b{b}_wv_b", b, groups, cfg["v_head_dim"], cfg["kv_lora_rank"]


def format_markdown(results: list[KernelResult], args: argparse.Namespace) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# DeepGEMM v3.2 SM120 benchmark results",
        "",
        f"Date: {now}",
        "",
        "Environment:",
        "",
        "```text",
        f"GPU: {torch.cuda.get_device_name()}",
        f"PyTorch: {torch.__version__}",
        f"PyTorch CUDA runtime: {torch.version.cuda}",
        f"Compute capability: {torch.cuda.get_device_capability()}",
        f"DeepGEMM: {deep_gemm.__version__}",
        f"SMs used by DeepGEMM: {deep_gemm.get_num_sms()}",
        f"warmups={args.warmups}, iters={args.iters}, quick={args.quick}",
        "```",
        "",
        "| Section | Name | Mode | Shape | ms | TFLOP/s | GB/s |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(f"| {r.section} | `{r.name}` | `{r.mode}` | {r.shape} | {r.ms:.4f} | {r.tflops:.2f} | {r.gbps:.2f} |")
    lines += [
        "",
        "Notes:",
        "",
        "- Timings are CUDA-event averages over the kernel call only; tensor creation and FP8/FP4 quantization are excluded.",
        "- Projection GEMMs report `2*m*n*k`; grouped decode projections report `2*groups*m_per_group*n*k`.",
        "- MQA indexer logits report only QK/relu/weighted-sum math as `2*selected_tokens*heads*head_dim`; top-k selection after logits is not included.",
        "- GB/s is a logical tensor-byte rate for the measured kernel call, not a hardware counter.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run a smaller sweep")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sections", nargs="*", default=["projection", "grouped", "indexer"], choices=["projection", "grouped", "indexer"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(CONFIG_PATH.read_text())
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.set_float32_matmul_precision("highest")

    print(f"GPU: {torch.cuda.get_device_name()} | DeepGEMM {deep_gemm.__version__} | SMs {deep_gemm.get_num_sms()}")
    print(f"JIT cache: {os.environ['DG_JIT_CACHE_DIR']}")

    results: list[KernelResult] = []
    modes = ["fp8", "fp8_fp4w"]

    if "projection" in args.sections:
        for name, m, n, k in projection_cases(cfg, args.quick):
            for mode in modes:
                print(f"[projection] {name} {mode} m={m} n={n} k={k}", flush=True)
                results.append(benchmark_gemm(name, m, n, k, mode, args.warmups, args.iters))

    if "grouped" in args.sections:
        for name, batch, groups, n, k in grouped_cases(cfg, args.quick):
            for mode in modes:
                print(f"[grouped] {name} {mode} groups={groups} batch={batch} n={n} k={k}", flush=True)
                results.append(benchmark_grouped_gemm(name, batch, groups, n, k, mode, args.warmups, args.iters))

    if "indexer" in args.sections:
        index_heads = cfg["index_n_heads"]
        index_dim = cfg["index_head_dim"]
        prefill_shapes = [(512, 512)] if args.quick else [(512, 512), (2048, 2048)]
        decode_shapes = [(1, 8192), (64, 8192)] if args.quick else [(1, 8192), (64, 8192), (256, 8192), (64, 32768)]
        for s, skv in prefill_shapes:
            for mode in ("fp8", "fp4"):
                print(f"[indexer] prefill {mode} s={s} skv={skv}", flush=True)
                results.append(benchmark_mqa_logits(s, skv, index_heads, index_dim, mode, args.warmups, args.iters))
        for batch, skv in decode_shapes:
            for mode in ("fp8", "fp4"):
                print(f"[indexer] paged {mode} batch={batch} skv={skv}", flush=True)
                results.append(benchmark_paged_mqa(batch, skv, index_heads, index_dim, mode, args.warmups, args.iters))

    md = format_markdown(results, args)
    print()
    print(md)
    if args.output:
        args.output.write_text(md)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
