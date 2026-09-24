"""Synthetic single-sequence V3.2 extend benchmark with causal sparse attention."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

if __package__:
    from .deepseek_v32_decode import (
        CONFIG_PATH,
        V32DecodeRunner,
        attention_scale,
        bench_cuda,
        deep_gemm,
        load_config,
        make_kv_cache_v32,
        parse_ints,
        quantize_index,
    )
    from .deepseek_v32_extend_kernels import append_cache, quantize_activation
else:
    from deepseek_v32_decode import (
        CONFIG_PATH,
        V32DecodeRunner,
        attention_scale,
        bench_cuda,
        deep_gemm,
        load_config,
        make_kv_cache_v32,
        parse_ints,
        quantize_index,
    )
    from deepseek_v32_extend_kernels import append_cache, quantize_activation


@dataclass
class ExtendCase:
    history_len: int
    new_tokens: int
    x: torch.Tensor
    position_ids: torch.Tensor
    kv_cache: torch.Tensor
    index_keys: torch.Tensor
    index_scales: torch.Tensor

    @property
    def total_len(self):
        return self.history_len + self.new_tokens


class V32ExtendRunner(V32DecodeRunner):
    """Chunk all temporary activations and logits, retain history in latent form.

    Each chunk writes its KVs first. Per-query exclusive end offsets enforce
    causality within that chunk. Later chunks need no historical re-projection.
    """

    def __init__(self, cfg, mode="fp8", chunk_size=512):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        super().__init__(cfg, mode)
        try:
            import sparse_mla_sm120 as mla
        except ModuleNotFoundError as exc:
            if exc.name != "sparse_mla_sm120":
                raise
            import flash_mla_sm120 as mla
        if not hasattr(mla, "sparse_mla_prefill_fwd"):
            import flash_mla_sm120 as mla
        self.sparse_prefill = mla.sparse_mla_prefill_fwd
        self.chunk_size = chunk_size
        self.last_stages = {}
        for name in ("wq_a", "wq_b", "wkv_a", "index_wqi", "index_wki", "wk_b", "wv_b", "wo"):
            getattr(self, name).activation_quantizer = quantize_activation

    def make_case(self, history_len: int, new_tokens: int) -> ExtendCase:
        total = history_len + new_tokens
        if history_len < 0 or new_tokens < 1 or total > self.cfg.max_seq_len:
            raise ValueError("Require history >= 0, new_tokens > 0, total <= max_seq_len")
        kv = make_kv_cache_v32((total + 63) // 64, self.cfg.qk_head_dim, "cuda")
        keys, scales = quantize_index(
            torch.randn(total, self.cfg.index_head_dim, device="cuda", dtype=torch.bfloat16) / 10
        )
        return ExtendCase(
            history_len,
            new_tokens,
            torch.randn(new_tokens, self.cfg.dim, device="cuda", dtype=torch.bfloat16),
            torch.arange(history_len, total, device="cuda", dtype=torch.int64),
            kv,
            keys,
            scales.flatten(),
        )

    def project_chunk(self, case, start, stop):
        # Existing projection accepts a flat token axis named `batch`.
        count = stop - start
        padded = (count + 127) // 128 * 128
        x, positions = case.x[start:stop], case.position_ids[start:stop]
        if padded != count:
            x = torch.nn.functional.pad(x, (0, 0, 0, padded - count))
            positions = torch.nn.functional.pad(positions, (0, padded - count))
        projected = self.project(SimpleNamespace(batch=padded, x=x, position_ids=positions))
        return type(projected)(*(value[:count].contiguous() for value in vars(projected).values()))

    def post_wv_b(self, attn_out):
        # SM120 masked grouped GEMM needs full 128-row tiles (including tails).
        count = attn_out.shape[0]
        padded = (count + 127) // 128 * 128
        if padded != count:
            attn_out = torch.nn.functional.pad(attn_out, (0, 0, 0, 0, 0, padded - count))
        return super().post_wv_b(attn_out)[:count].contiguous()

    def write_chunk(self, case, projected, start):
        append_cache(
            projected.kv_current,
            projected.idx_k,
            case.kv_cache,
            case.index_keys,
            case.index_scales,
            case.history_len + start,
        )

    def index_logits(self, case, projected, start, stop):
        end = case.history_len + stop
        q, qs = quantize_index(projected.idx_q[:, 0])
        weights = (projected.idx_weights * qs[..., 0] * self.cfg.index_head_dim**-0.5).contiguous()
        ends = (case.position_ids[start:stop] + 1).to(torch.int32)
        logits = deep_gemm.fp8_fp4_mqa_logits(
            (q, None),
            (case.index_keys[:end], case.index_scales[:end]),
            weights,
            torch.zeros_like(ends),
            ends,
            False,
            0,
            torch.float32,
        )
        return logits, ends

    def select_indices(self, logits, ends, topk):
        from flashinfer.topk import top_k_ragged_transform

        # DeepGEMM allocates 1024-byte-aligned rows, then returns a narrower view.
        # Expose its allocated padding instead of copying the entire logits matrix.
        # Ragged lengths ensure neither padding nor future columns are selected.
        if not logits.is_contiguous():
            logits = logits.as_strided((logits.shape[0], logits.stride(0)), (logits.stride(0), 1))
        return top_k_ragged_transform(
            logits,
            torch.zeros_like(ends),
            ends,
            topk,
            deterministic=True,
        )

    @torch.inference_mode()
    def run_once(self, case: ExtendCase, profile=False) -> torch.Tensor:
        output = torch.empty_like(case.x)
        events = []

        def stage(name, fn, *args, **kwargs):
            if not profile:
                return fn(*args, **kwargs)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            result = fn(*args, **kwargs)
            end.record()
            events.append((name, begin, end))
            return result

        for start in range(0, case.new_tokens, self.chunk_size):
            stop = min(start + self.chunk_size, case.new_tokens)
            projected = stage("projection_norm_rope", self.project_chunk, case, start, stop)
            stage("cache_write", self.write_chunk, case, projected, start)
            logits, ends = stage("index_logits", self.index_logits, case, projected, start, stop)
            indices = stage("topk", self.select_indices, logits, ends, self.cfg.index_topk)
            del logits, ends
            attn = stage(
                "sparse_prefill",
                self.sparse_prefill,
                projected.q_attn,
                case.kv_cache,
                indices,
                attention_scale(self.cfg),
                self.cfg.kv_lora_rank,
                bf16_qk=False,
            )

            if isinstance(attn, tuple):
                attn = attn[0]

            def post(start=start, stop=stop, attn=attn):
                output[start:stop] = self.post_wo(self.post_wv_b(attn))

            stage("output_projection", post)
            del projected, indices, attn
        if profile:
            torch.cuda.synchronize()
            self.last_stages = {}
            for name, begin, end in events:
                self.last_stages[name] = self.last_stages.get(name, 0.0) + begin.elapsed_time(end)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--history-lens", default="4096,8192,16384,32768,65536")
    parser.add_argument("--new-tokens", default="1024,2048,4096")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--mode", choices=["fp8", "fp8_fp4w"], default="fp8")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument(
        "--cuda-graph", action="store_true", help="time captured replay after warmup"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("docs/model_extend_v32_results.json"))
    args = parser.parse_args()
    if args.iters < 1 or args.warmups < 0:
        parser.error("iters must be positive and warmups nonnegative")
    torch.manual_seed(args.seed)
    runner = V32ExtendRunner(load_config(args.config), args.mode, args.chunk_size)
    rows = []
    for history in parse_ints(args.history_lens):
        for count in parse_ints(args.new_tokens):
            case = runner.make_case(history, count)
            # Warm compilation/allocator before recording memory and timing.
            runner.run_once(case)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            graph = None
            if args.cuda_graph:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_out = runner.run_once(case)
                elapsed = bench_cuda(graph.replay, args.warmups, args.iters)
                if not torch.isfinite(graph_out).all().item():
                    raise RuntimeError("Non-finite captured output")
            else:
                elapsed = bench_cuda(
                    lambda case=case: runner.run_once(case), args.warmups, args.iters
                )
            out = runner.run_once(case, profile=True)
            if not torch.isfinite(out).all().item():
                raise RuntimeError(f"Non-finite output: history={history}, new_tokens={count}")
            row = {
                "history": history,
                "new_tokens": count,
                "chunk_size": args.chunk_size,
                "e2e_ms": elapsed,
                "tokens_per_second": count * 1000 / elapsed,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "stages_ms": runner.last_stages.copy(),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            if graph is not None:
                del graph, graph_out
            del case, out
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "mode": args.mode,
                "cuda_graph": args.cuda_graph,
                "synthetic": True,
                "batch_size": 1,
                "warmups": args.warmups,
                "iters": args.iters,
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
