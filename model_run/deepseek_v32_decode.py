"""Run a synthetic DeepSeek V3.2 decode step with DeepGEMM + sparse MLA.

This benchmark stitches together the decode-time pieces that are usually
measured separately:

1. DeepGEMM projection kernels for q, kv, indexer, per-head W_KB/W_VB, and W_O.
2. DeepGEMM paged MQA logits for the sparse indexer.
3. sparse_mla_sm120 / flash_mla_sm120 decode attention over packed V3.2 KV.

The tensors are randomly initialized and stay on GPU. No KV offload path is
used. Timing excludes random initialization and pre-quantization of persistent
weights/caches, but includes FlashInfer norm/RoPE, per-token activation quantization, cache update,
indexer top-k, sparse attention, and output projection in the end-to-end path.

Example:
    DG_JIT_CACHE_DIR=/home/cage/dsa/.deep_gemm_cache \\
      .venv/bin/python model_run/deepseek_v32_decode.py --quick \\
      --output docs/model_decode_v32_results.md
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DG_JIT_CACHE_DIR", str(ROOT / ".deep_gemm_cache"))

import deep_gemm
import torch
from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp4, per_token_cast_to_fp8

if __package__:
    from .deepseek_v32_ops import (
        FlashInferV32Ops,
        attention_scale,
        quantize_index,
    )
else:
    from deepseek_v32_ops import (
        FlashInferV32Ops,
        attention_scale,
        quantize_index,
    )

CONFIG_PATH = ROOT / "docs" / "config.json"
BLOCK_SIZE = 64
DECODE_TOKEN_LIMIT = 64


@dataclass(frozen=True)
class V32Config:
    dim: int
    n_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    original_seq_len: int = 4096
    max_seq_len: int = 163840
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0

    @property
    def qk_head_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def q_proj_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


@dataclass(frozen=True)
class BenchResult:
    batch: int
    history_len: int
    e2e_ms: float
    projection_ms: float
    indexer_ms: float
    attention_ms: float
    total_tokens: int
    topk: int
    output_shape: str


@dataclass(frozen=True)
class DetailRow:
    batch: int
    history_len: int
    section: str
    name: str
    shape: str
    ms: float
    tflops: float
    gbps: float
    bytes_gb: float


@dataclass(frozen=True)
class DetailCase:
    batch: int
    history_len: int
    output_shape: str
    rows: list[DetailRow]


@dataclass(frozen=True)
class SegmentProfile:
    name: str
    event_ms: float
    kernel_ms: float
    launch_gap_ms: float
    cpu_wall_ms: float
    kernel_count: int


@dataclass(frozen=True)
class CaseProfile:
    batch: int
    history_len: int
    segments: list[SegmentProfile]


@dataclass
class CaseTensors:
    batch: int
    history_len: int
    total_len: int
    blocks_per_seq: int
    x: torch.Tensor
    position_ids: torch.Tensor
    kv_cache: torch.Tensor
    kv_current_packed: torch.Tensor
    decode_block_ids: torch.Tensor
    decode_token_ids: torch.Tensor
    index_kv_cache: torch.Tensor
    index_current_record: torch.Tensor
    context_lens: torch.Tensor
    block_table: torch.Tensor
    sequence_offsets: torch.Tensor
    max_context_len: int


@dataclass
class Projected:
    qr: torch.Tensor
    q_nope: torch.Tensor
    q_pe: torch.Tensor
    q_attn: torch.Tensor
    kv_current: torch.Tensor
    idx_q: torch.Tensor
    idx_k: torch.Tensor
    idx_weights: torch.Tensor


def load_config(path: Path) -> V32Config:
    raw = json.loads(path.read_text())
    return V32Config(
        dim=raw["dim"],
        n_heads=raw["n_heads"],
        q_lora_rank=raw["q_lora_rank"],
        kv_lora_rank=raw["kv_lora_rank"],
        qk_nope_head_dim=raw["qk_nope_head_dim"],
        qk_rope_head_dim=raw["qk_rope_head_dim"],
        v_head_dim=raw["v_head_dim"],
        index_n_heads=raw["index_n_heads"],
        index_head_dim=raw["index_head_dim"],
        index_topk=raw["index_topk"],
        **{
            name: raw[name]
            for name in (
                "norm_eps",
                "rope_theta",
                "rope_factor",
                "original_seq_len",
                "max_seq_len",
                "beta_fast",
                "beta_slow",
                "mscale",
            )
            if name in raw
        },
    )


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


def bench_cuda(
    fn: Callable[[], object], warmups: int, iters: int, nvtx_label: str | None = None
) -> float:
    fn()
    torch.cuda.synchronize()
    for warmup_idx in range(warmups):
        if nvtx_label:
            torch.cuda.nvtx.range_push(f"{nvtx_label}/warmup_{warmup_idx}")
        fn()
        torch.cuda.synchronize()
        if nvtx_label:
            torch.cuda.nvtx.range_pop()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for iter_idx in range(iters):
        if nvtx_label:
            torch.cuda.nvtx.range_push(f"{nvtx_label}/iter_{iter_idx}")
        fn()
        torch.cuda.synchronize()
        if nvtx_label:
            torch.cuda.nvtx.range_pop()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _cast_scale_inv_to_ue8m0(scales_inv: torch.Tensor) -> torch.Tensor:
    return torch.pow(2, scales_inv.clamp_min(torch.finfo(torch.float32).tiny).log2().ceil())


def quantize_kv_v32(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 V3.2 KV records as 656 B/token.

    Input shape:  [num_blocks, block_size, 1, 576] bf16
    Output shape: [num_blocks, block_size, 1, 656] uint8
    """
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    nb, bs, hk, d = kv_bf16.shape
    assert d == d_nope + d_rope and hk == 1
    kv = kv_bf16.squeeze(2)
    bytes_per_token = d_nope + num_tiles * 4 + d_rope * 2
    result = torch.zeros(nb, bs, bytes_per_token, dtype=torch.uint8, device=kv.device)
    for tile_idx in range(num_tiles):
        start = tile_idx * tile_size
        end = start + tile_size
        tile = kv[..., start:end].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        result[..., start:end] = fp8.view(torch.uint8)
        scale_bytes = scale.to(torch.float32).contiguous().view(torch.uint8).reshape(nb, bs, 4)
        result[..., d_nope + tile_idx * 4 : d_nope + (tile_idx + 1) * 4] = scale_bytes
    rope = (
        kv[..., d_nope:]
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .reshape(nb, bs, d_rope * 2)
    )
    result[..., d_nope + num_tiles * 4 :] = rope
    return result.view(nb, bs, 1, bytes_per_token)


def make_kv_cache_v32(num_blocks: int, qk_head_dim: int, device: str) -> torch.Tensor:
    bytes_per_token = 512 + 4 * 4 + 64 * 2
    fused = torch.empty(
        (num_blocks, BLOCK_SIZE, 1, bytes_per_token), device=device, dtype=torch.uint8
    )
    chunk_blocks = 2048
    for start in range(0, num_blocks, chunk_blocks):
        end = min(start + chunk_blocks, num_blocks)
        raw = (
            torch.randn(
                (end - start, BLOCK_SIZE, 1, qk_head_dim), device=device, dtype=torch.bfloat16
            )
            / 10
        ).clamp(-1, 1)
        fused[start:end] = quantize_kv_v32(raw)
        del raw
    return fused


def make_index_kv_cache(tokens: int, head_dim: int, device: str) -> torch.Tensor:
    blocks = ceil_div(tokens, BLOCK_SIZE)
    padded_tokens = blocks * BLOCK_SIZE
    fused = torch.empty((blocks, BLOCK_SIZE, 1, head_dim + 4), device=device, dtype=torch.uint8)
    data_view, scale_view = index_cache_views(fused)
    chunk_tokens = 262144
    for start in range(0, padded_tokens, chunk_tokens):
        end = min(start + chunk_tokens, padded_tokens)
        raw = torch.randn((end - start, head_dim), device=device, dtype=torch.bfloat16) / 10
        packed, scales = pack_index_tokens(raw)
        data_view[start // BLOCK_SIZE : end // BLOCK_SIZE] = packed.view(-1, BLOCK_SIZE, head_dim)
        scale_view[start // BLOCK_SIZE : end // BLOCK_SIZE] = scales.view(-1, BLOCK_SIZE, 4)
        del raw, packed, scales
    return fused


def index_cache_views(cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepGEMM stores all keys THEN all scales within each page, not per token."""
    blocks, page_size, _, record_bytes = cache.shape
    head_dim = record_bytes - 4
    raw = cache.view(blocks, -1)
    return (
        raw[:, : page_size * head_dim].view(blocks, page_size, head_dim),
        raw[:, page_size * head_dim :].view(blocks, page_size, 4),
    )


def pack_index_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8, scale = quantize_index(tokens)
    scale_bytes = scale.contiguous().view(torch.uint8).reshape(tokens.shape[0], 4)
    return fp8.view(torch.uint8), scale_bytes


class QuantizedLinear:
    def __init__(self, out_features: int, in_features: int, mode: str = "fp8_fp4w"):
        self.out_features = out_features
        self.in_features = in_features
        self.mode = mode
        self.activation_quantizer = per_token_cast_to_fp8
        raw = torch.randn(
            (out_features, in_features), device="cuda", dtype=torch.bfloat16
        ) / math.sqrt(in_features)
        self.weight, self.recipe_b = self._quantize_weight(raw)
        del raw

    def _quantize_weight(self, raw: torch.Tensor):
        if self.mode == "fp8_fp4w":
            data, sf = per_token_cast_to_fp4(raw, use_ue8m0=True, gran_k=32)
            return (data, sf), (1, 32)
        if self.mode == "fp8":
            return per_block_cast_to_fp8(raw, use_ue8m0=True, gran_k=128), None
        raise ValueError(f"unsupported GEMM mode: {self.mode}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x2d = x.view(-1, self.in_features).contiguous()
        if self.mode == "fp8_fp4w":
            a = self.activation_quantizer(x2d, use_ue8m0=True, gran_k=128)
            recipe_a = (1, 128)
        else:
            a = self.activation_quantizer(x2d, use_ue8m0=True, gran_k=128)
            recipe_a = None
        out = torch.empty((x2d.shape[0], self.out_features), device=x.device, dtype=torch.bfloat16)
        if self.mode == "fp8":
            deep_gemm.fp8_gemm_nt(a, self.weight, out)
        else:
            deep_gemm.fp8_fp4_gemm_nt(
                a,
                self.weight,
                out,
                recipe_a=recipe_a,
                recipe_b=self.recipe_b,
            )
        return out.view(*x.shape[:-1], self.out_features)


class BF16Linear:
    """Indexer head-weight projection stays BF16, including in FP8 GEMM mode."""

    def __init__(self, out_features: int, in_features: int):
        self.out_features = out_features
        self.in_features = in_features
        self.weight = torch.randn(
            out_features, in_features, device="cuda", dtype=torch.bfloat16
        ) / math.sqrt(in_features)

    def __call__(self, x):
        return torch.nn.functional.linear(x, self.weight)


class GroupedLinear:
    def __init__(self, groups: int, out_features: int, in_features: int, mode: str = "fp8_fp4w"):
        self.groups = groups
        self.out_features = out_features
        self.in_features = in_features
        self.mode = mode
        self.activation_quantizer = per_token_cast_to_fp8
        raw = torch.randn(
            (groups, out_features, in_features), device="cuda", dtype=torch.bfloat16
        ) / math.sqrt(in_features)
        self.weight, self.recipe_b = self._quantize_weight(raw)
        del raw

    def _quantize_weight(self, raw: torch.Tensor):
        raw2d = raw.view(self.groups * self.out_features, self.in_features)
        if self.mode == "fp8_fp4w":
            data2d, sf2d = per_token_cast_to_fp4(raw2d, use_ue8m0=True, gran_k=32)
            data = data2d.view(self.groups, self.out_features, self.in_features // 2)
            sf = sf2d.view(self.groups, self.out_features, ceil_div(self.in_features, 32))
            return (data, sf), (1, 32)
        if self.mode == "fp8":
            if self.out_features % 128 != 0:
                raise ValueError("Grouped FP8 weights require out_features to be aligned to 128")
            data2d, sf2d = per_block_cast_to_fp8(raw2d, use_ue8m0=True, gran_k=128)
            data = data2d.view(self.groups, self.out_features, self.in_features)
            sf = sf2d.view(self.groups, self.out_features // 128, ceil_div(self.in_features, 128))
            return (data, sf), None
        raise ValueError(f"unsupported grouped GEMM mode: {self.mode}")

    def __call__(self, x_by_group: torch.Tensor) -> torch.Tensor:
        groups, m, k = x_by_group.shape
        assert groups == self.groups and k == self.in_features
        expected_stride = (m * k, k, 1)
        if x_by_group.stride() != expected_stride:
            canonical = torch.empty(
                (groups, m, k), device=x_by_group.device, dtype=x_by_group.dtype
            )
            canonical.copy_(x_by_group)
            x_by_group = canonical
        x2d = x_by_group.view(groups * m, k)
        data2d, sf2d = self.activation_quantizer(x2d, use_ue8m0=True, gran_k=128)
        data = data2d.view(groups, m, k)
        sf = sf2d.view(groups, m, ceil_div(k, 128))
        out = torch.empty(
            (groups, m, self.out_features), device=x_by_group.device, dtype=torch.bfloat16
        )
        masked_m = torch.full((groups,), m, device=x_by_group.device, dtype=torch.int32)
        if self.mode == "fp8":
            deep_gemm.m_grouped_fp8_gemm_nt_masked((data, sf), self.weight, out, masked_m, m)
        else:
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
                (data, sf),
                self.weight,
                out,
                masked_m,
                m,
                recipe_a=(1, 128),
                recipe_b=self.recipe_b,
            )
        return out


class V32DecodeRunner:
    def __init__(self, cfg: V32Config, mode: str, post_proj: str = "deepgemm"):
        self.cfg = cfg
        self.mode = mode
        self.post_proj = post_proj
        self.ops = FlashInferV32Ops(cfg)
        self.sparse_decode = self._load_sparse_decode()

        self.wq_a = QuantizedLinear(cfg.q_lora_rank, cfg.dim, mode)
        self.wq_b = QuantizedLinear(cfg.n_heads * cfg.q_proj_head_dim, cfg.q_lora_rank, mode)
        self.wkv_a = QuantizedLinear(cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.dim, mode)
        self.index_wqi = QuantizedLinear(
            cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank, mode
        )
        self.index_wki = QuantizedLinear(cfg.index_head_dim, cfg.dim, mode)
        self.index_weights = BF16Linear(cfg.index_n_heads, cfg.dim)
        self.wk_b = GroupedLinear(cfg.n_heads, cfg.kv_lora_rank, cfg.qk_nope_head_dim, mode)
        self.wv_b = GroupedLinear(cfg.n_heads, cfg.v_head_dim, cfg.kv_lora_rank, mode)
        self.wo = QuantizedLinear(cfg.dim, cfg.n_heads * cfg.v_head_dim, mode)
        self.wv_b_torch = torch.randn(
            (cfg.n_heads, cfg.kv_lora_rank, cfg.v_head_dim), device="cuda", dtype=torch.bfloat16
        ) / math.sqrt(cfg.kv_lora_rank)
        self.wo_torch = torch.randn(
            (cfg.dim, cfg.n_heads * cfg.v_head_dim), device="cuda", dtype=torch.bfloat16
        ) / math.sqrt(cfg.n_heads * cfg.v_head_dim)

    @staticmethod
    def _load_sparse_decode():
        try:
            import sparse_mla_sm120

            fn = getattr(sparse_mla_sm120, "sparse_mla_decode_fwd", None)
            if fn is not None:
                return fn
        except ModuleNotFoundError as exc:
            if exc.name != "sparse_mla_sm120":
                raise
        import flash_mla_sm120

        return flash_mla_sm120.sparse_mla_decode_fwd

    def make_case(self, batch: int, history_len: int) -> CaseTensors:
        if not 0 <= history_len < self.cfg.max_seq_len:
            raise ValueError("history_len must be within the configured RoPE context")
        if batch > DECODE_TOKEN_LIMIT:
            raise ValueError(
                f"sparse MLA decode path supports at most {DECODE_TOKEN_LIMIT} tokens; got batch={batch}"
            )
        total_len = history_len + 1
        blocks_per_seq = ceil_div(total_len, BLOCK_SIZE)
        capacity = blocks_per_seq * BLOCK_SIZE
        total_blocks = batch * blocks_per_seq
        x = torch.randn((batch, self.cfg.dim), device="cuda", dtype=torch.bfloat16)

        kv_cache = make_kv_cache_v32(total_blocks, self.cfg.qk_head_dim, "cuda")
        kv_current_packed = torch.empty((batch, 1, 656), device="cuda", dtype=torch.uint8)
        seq_ids_long = torch.arange(batch, device="cuda", dtype=torch.long)
        decode_block_ids = seq_ids_long * blocks_per_seq + (history_len // BLOCK_SIZE)
        decode_token_ids = torch.full(
            (batch,), history_len % BLOCK_SIZE, device="cuda", dtype=torch.long
        )
        index_kv_cache = make_index_kv_cache(
            total_blocks * BLOCK_SIZE, self.cfg.index_head_dim, "cuda"
        )
        index_current_record = torch.empty(
            (batch, self.cfg.index_head_dim + 4), device="cuda", dtype=torch.uint8
        )
        context_lens = torch.full((batch, 1), total_len, device="cuda", dtype=torch.int32)
        block_table = torch.arange(total_blocks, device="cuda", dtype=torch.int32).view(
            batch, blocks_per_seq
        )
        sequence_offsets = (torch.arange(batch, device="cuda", dtype=torch.int32) * capacity).view(
            batch, 1
        )
        return CaseTensors(
            batch=batch,
            history_len=history_len,
            total_len=total_len,
            blocks_per_seq=blocks_per_seq,
            x=x,
            position_ids=torch.full((batch,), history_len, device=x.device, dtype=torch.int64),
            kv_cache=kv_cache,
            kv_current_packed=kv_current_packed,
            decode_block_ids=decode_block_ids,
            decode_token_ids=decode_token_ids,
            index_kv_cache=index_kv_cache,
            index_current_record=index_current_record,
            context_lens=context_lens,
            block_table=block_table,
            sequence_offsets=sequence_offsets,
            max_context_len=total_len,
        )

    def project(self, case: CaseTensors) -> Projected:
        cfg = self.cfg
        qr = self.ops.q_norm(self.wq_a(case.x))
        q_proj = self.wq_b(qr).view(case.batch, cfg.n_heads, cfg.q_proj_head_dim)
        q_nope, q_pe = torch.split(q_proj, [cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)

        kv_proj = self.wkv_a(case.x)
        kv_latent, k_pe = torch.split(kv_proj, [cfg.kv_lora_rank, cfg.qk_rope_head_dim], dim=-1)
        kv_latent = self.ops.kv_norm(kv_latent)
        q_pe, k_pe = self.ops.apply_rope(q_pe, k_pe, case.position_ids, is_neox=False)
        kv_current = torch.cat([kv_latent, k_pe], dim=-1).contiguous()

        q_nope_by_head = q_nope.transpose(0, 1).contiguous()
        q_latent_by_head = self.wk_b(q_nope_by_head)
        q_latent = q_latent_by_head.transpose(0, 1).contiguous()
        q_attn = torch.cat([q_latent, q_pe], dim=-1).contiguous()

        idx_q = (
            self.index_wqi(qr)
            .view(case.batch, 1, cfg.index_n_heads, cfg.index_head_dim)
            .contiguous()
        )
        idx_k = self.ops.index_norm(self.index_wki(case.x))
        # Indexer rotates the FIRST 64 dims, using split-half (NeoX) pairing.
        rd = cfg.qk_rope_head_dim
        iq_pe, ik_pe = self.ops.apply_rope(
            idx_q[:, 0, :, :rd], idx_k[:, :rd], case.position_ids, is_neox=True
        )
        idx_q = torch.cat((iq_pe, idx_q[:, 0, :, rd:]), -1).unsqueeze(1)
        idx_k = torch.cat((ik_pe, idx_k[:, rd:]), -1)
        idx_weights = (
            self.index_weights(case.x).view(case.batch, cfg.index_n_heads).float()
            * (cfg.index_n_heads**-0.5)
        ).contiguous()
        return Projected(qr, q_nope, q_pe, q_attn, kv_current, idx_q, idx_k, idx_weights)

    def update_attention_cache(self, case: CaseTensors, projected: Projected) -> None:
        packed = quantize_kv_v32(projected.kv_current[:, None, None, :])[:, 0]
        case.kv_current_packed.copy_(packed)
        case.kv_cache[case.decode_block_ids, case.decode_token_ids, :, :] = case.kv_current_packed

    def update_index_cache(self, case: CaseTensors, projected: Projected) -> None:
        data, scales = pack_index_tokens(projected.idx_k)
        case.index_current_record[:, : self.cfg.index_head_dim] = data
        case.index_current_record[:, self.cfg.index_head_dim :] = scales
        keys, sf = index_cache_views(case.index_kv_cache)
        keys[case.decode_block_ids, case.decode_token_ids] = data
        sf[case.decode_block_ids, case.decode_token_ids] = scales

    def compute_index_logits(self, case: CaseTensors, projected: Projected) -> torch.Tensor:
        q_fp8, q_scale = quantize_index(projected.idx_q)
        weights = (
            projected.idx_weights * q_scale[:, 0, :, 0] * self.cfg.index_head_dim**-0.5
        ).contiguous()
        num_clusters = deep_gemm.get_num_sms()
        schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
            case.context_lens, BLOCK_SIZE, num_clusters
        )
        return deep_gemm.fp8_fp4_paged_mqa_logits(
            (q_fp8, None),
            case.index_kv_cache,
            weights,
            case.context_lens,
            case.block_table,
            schedule_meta,
            case.max_context_len,
            False,
            torch.float32,
        )

    def select_topk(self, case: CaseTensors, logits: torch.Tensor) -> torch.Tensor:
        k = min(self.cfg.index_topk, case.total_len)
        local = torch.topk(logits, k=k, dim=-1).indices.to(torch.int32)
        if k < self.cfg.index_topk:
            pad = torch.full(
                (case.batch, self.cfg.index_topk - k), -1, device="cuda", dtype=torch.int32
            )
            local = torch.cat([local, pad], dim=-1)
        valid = local >= 0
        global_indices = torch.where(valid, local + case.sequence_offsets, local)
        return global_indices.contiguous()

    def run_indexer(self, case: CaseTensors, projected: Projected) -> torch.Tensor:
        self.update_index_cache(case, projected)
        logits = self.compute_index_logits(case, projected)
        return self.select_topk(case, logits)

    def sparse_attention(
        self, case: CaseTensors, projected: Projected, indices: torch.Tensor
    ) -> torch.Tensor:
        sm_scale = attention_scale(self.cfg)
        out = self.sparse_decode(
            projected.q_attn, case.kv_cache, indices, sm_scale, self.cfg.kv_lora_rank
        )
        if isinstance(out, tuple):
            out = out[0]
        return out

    def post_wv_b(self, attn_out: torch.Tensor) -> torch.Tensor:
        if self.post_proj == "deepgemm":
            value_by_head = attn_out.transpose(0, 1).contiguous()
            return self.wv_b(value_by_head).transpose(0, 1).contiguous()
        return torch.einsum("bhc,hcd->bhd", attn_out, self.wv_b_torch).contiguous()

    def post_wo(self, head_out: torch.Tensor) -> torch.Tensor:
        if self.post_proj == "deepgemm":
            return self.wo(head_out.view(head_out.shape[0], self.cfg.n_heads * self.cfg.v_head_dim))
        return torch.matmul(
            head_out.view(head_out.shape[0], self.cfg.n_heads * self.cfg.v_head_dim),
            self.wo_torch.t(),
        )

    def run_attention(
        self, case: CaseTensors, projected: Projected, indices: torch.Tensor
    ) -> torch.Tensor:
        self.update_attention_cache(case, projected)
        attn_out = self.sparse_attention(case, projected, indices)
        head_out = self.post_wv_b(attn_out)
        return self.post_wo(head_out)

    def run_once(self, case: CaseTensors) -> torch.Tensor:
        projected = self.project(case)
        indices = self.run_indexer(case, projected)
        return self.run_attention(case, projected, indices)


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def iter_cases(args: argparse.Namespace) -> Iterable[tuple[int, int]]:
    batches = [1, 4, 8] if args.quick else parse_ints(args.batch_sizes)
    histories = [4096, 8192] if args.quick else parse_ints(args.history_lens)
    for batch in batches:
        for history_len in histories:
            yield batch, history_len


def run_benchmark(args: argparse.Namespace) -> list[BenchResult]:
    cfg = load_config(args.config)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("highest")
    runner = V32DecodeRunner(cfg, args.mode, args.post_proj)

    results: list[BenchResult] = []
    for batch, history_len in iter_cases(args):
        print(f"[case] batch={batch} history_len={history_len}", flush=True)
        case = runner.make_case(batch, history_len)
        projected = runner.project(case)
        indices = runner.run_indexer(case, projected)
        out = runner.run_attention(case, projected, indices)
        torch.cuda.synchronize()
        output_shape = "x".join(str(v) for v in out.shape)
        del out

        case_label = f"b{batch}_h{history_len}"
        projection_ms = bench_cuda(
            lambda case=case: runner.project(case),
            args.warmups,
            args.iters,
            f"benchmark/{case_label}/projection_all",
        )
        projected = runner.project(case)
        indexer_ms = bench_cuda(
            lambda case=case, projected=projected: runner.run_indexer(case, projected),
            args.warmups,
            args.iters,
            f"benchmark/{case_label}/indexer",
        )
        indices = runner.run_indexer(case, projected)
        attention_ms = bench_cuda(
            lambda case=case, projected=projected, indices=indices: runner.run_attention(
                case, projected, indices
            ),
            args.warmups,
            args.iters,
            f"benchmark/{case_label}/attention",
        )
        e2e_ms = bench_cuda(
            lambda case=case: runner.run_once(case),
            args.warmups,
            args.iters,
            f"benchmark/{case_label}/e2e",
        )

        results.append(
            BenchResult(
                batch=batch,
                history_len=history_len,
                e2e_ms=e2e_ms,
                projection_ms=projection_ms,
                indexer_ms=indexer_ms,
                attention_ms=attention_ms,
                total_tokens=history_len + 1,
                topk=cfg.index_topk,
                output_shape=output_shape,
            )
        )
        del case, projected, indices
        cleanup()
    return results


def _metric_tflops(flops: int, ms: float) -> float:
    return flops / (ms * 1e9) if flops and ms > 0 else 0.0


def _metric_gbps(bytes_: int, ms: float) -> float:
    return bytes_ / (ms * 1e6) if bytes_ and ms > 0 else 0.0


def _linear_flops(m: int, n: int, k: int) -> int:
    return 2 * m * n * k


def _bf16_nbytes(numel: int) -> int:
    return numel * torch.tensor([], dtype=torch.bfloat16).element_size()


def _linear_effective_bytes(x: torch.Tensor, layer: QuantizedLinear, out_features: int) -> int:
    m = x.numel() // layer.in_features
    act_quant = m * layer.in_features + m * ceil_div(layer.in_features, 128) * 4
    if isinstance(layer, BF16Linear):
        act_quant = 0
    out = _bf16_nbytes(m * out_features)
    return tensor_nbytes(x) + act_quant + tensor_nbytes(layer.weight) + out


def _grouped_linear_effective_bytes(x: torch.Tensor, layer: GroupedLinear) -> int:
    groups, m, k = x.shape
    act_quant = groups * m * k + groups * m * ceil_div(k, 128) * 4
    out = _bf16_nbytes(groups * m * layer.out_features)
    return tensor_nbytes(x) + act_quant + tensor_nbytes(layer.weight) + out + groups * 4


def _index_update_bytes(case: CaseTensors, cfg: V32Config) -> int:
    per_token = cfg.index_head_dim + 4
    return case.batch * (per_token + per_token + 2 * 8)


def _attn_update_bytes(case: CaseTensors, cfg: V32Config) -> int:
    packed = 512 + 4 * 4 + cfg.qk_rope_head_dim * 2
    return case.batch * (packed + packed + 2 * 8)


def _index_logits_bytes(
    case: CaseTensors, cfg: V32Config, logits: torch.Tensor | None = None
) -> int:
    tokens = case.batch * case.total_len
    bytes_ = case.batch * cfg.index_n_heads * cfg.index_head_dim
    bytes_ += tokens * (cfg.index_head_dim + 4)
    bytes_ += case.batch * cfg.index_n_heads * 4
    bytes_ += tensor_nbytes(case.context_lens) + tensor_nbytes(case.block_table)
    bytes_ += tensor_nbytes(logits) if logits is not None else case.batch * case.max_context_len * 2
    return bytes_


def _index_topk_bytes(case: CaseTensors, cfg: V32Config, logits: torch.Tensor) -> int:
    k = min(cfg.index_topk, case.total_len)
    return tensor_nbytes(logits) + case.batch * k * 4 + case.batch * cfg.index_topk * 4


def _sparse_mla_flops(case: CaseTensors, cfg: V32Config) -> int:
    k = min(cfg.index_topk, case.total_len)
    return 2 * case.batch * cfg.n_heads * k * (cfg.qk_head_dim + cfg.kv_lora_rank)


def _sparse_mla_bytes(case: CaseTensors, cfg: V32Config) -> int:
    k = min(cfg.index_topk, case.total_len)
    kv_record = 512 + 4 * 4 + cfg.qk_rope_head_dim * 2
    q = case.batch * cfg.n_heads * cfg.qk_head_dim * 2
    kv = case.batch * k * kv_record
    indices = case.batch * cfg.index_topk * 4
    out = case.batch * cfg.n_heads * cfg.kv_lora_rank * 2
    return q + kv + indices + out


def _decode_iter_flops(case: CaseTensors, cfg: V32Config) -> int:
    b = case.batch
    projection_flops = (
        _linear_flops(b, cfg.q_lora_rank, cfg.dim)
        + _linear_flops(b, cfg.n_heads * cfg.q_proj_head_dim, cfg.q_lora_rank)
        + 2 * cfg.n_heads * b * cfg.kv_lora_rank * cfg.qk_nope_head_dim
        + _linear_flops(b, cfg.qk_head_dim, cfg.dim)
        + _linear_flops(b, cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank)
        + _linear_flops(b, cfg.index_head_dim, cfg.dim)
        + _linear_flops(b, cfg.index_n_heads, cfg.dim)
    )
    indexer_flops = 2 * b * case.total_len * cfg.index_n_heads * cfg.index_head_dim
    attention_flops = _sparse_mla_flops(case, cfg)
    post_flops = 2 * cfg.n_heads * b * cfg.v_head_dim * cfg.kv_lora_rank + _linear_flops(
        b, cfg.dim, cfg.n_heads * cfg.v_head_dim
    )
    return projection_flops + indexer_flops + attention_flops + post_flops


def _decode_iter_bytes(
    case: CaseTensors,
    cfg: V32Config,
    runner: V32DecodeRunner,
    projected: Projected,
    logits: torch.Tensor,
    head_flat: torch.Tensor,
) -> int:
    b = case.batch
    q_proj = runner.wq_b(projected.qr).view(b, cfg.n_heads, cfg.q_proj_head_dim)
    q_nope, _ = torch.split(q_proj, [cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)
    q_nope_by_head = q_nope.transpose(0, 1).contiguous()
    attn_value_by_head = torch.empty(
        (cfg.n_heads, b, cfg.kv_lora_rank), device=case.x.device, dtype=torch.bfloat16
    )
    bytes_ = (
        _linear_effective_bytes(case.x, runner.wq_a, cfg.q_lora_rank)
        + _linear_effective_bytes(projected.qr, runner.wq_b, cfg.n_heads * cfg.q_proj_head_dim)
        + _grouped_linear_effective_bytes(q_nope_by_head, runner.wk_b)
        + _linear_effective_bytes(case.x, runner.wkv_a, cfg.qk_head_dim)
        + _linear_effective_bytes(
            projected.qr, runner.index_wqi, cfg.index_n_heads * cfg.index_head_dim
        )
        + _linear_effective_bytes(case.x, runner.index_wki, cfg.index_head_dim)
        + _linear_effective_bytes(case.x, runner.index_weights, cfg.index_n_heads)
        + _index_update_bytes(case, cfg)
        + _index_logits_bytes(case, cfg, logits)
        + _index_topk_bytes(case, cfg, logits)
        + _attn_update_bytes(case, cfg)
        + _sparse_mla_bytes(case, cfg)
        + _grouped_linear_effective_bytes(attn_value_by_head, runner.wv_b)
        + _linear_effective_bytes(head_flat, runner.wo, cfg.dim)
    )
    del q_proj, q_nope, q_nope_by_head, attn_value_by_head
    return bytes_


def _add_detail_row(
    rows: list[DetailRow],
    case: CaseTensors,
    section: str,
    name: str,
    shape: str,
    fn: Callable[[], object],
    flops: int,
    bytes_: int,
    warmups: int,
    iters: int,
) -> DetailRow:
    ms = bench_cuda(fn, warmups, iters, f"detail/{section}/{name}")
    row = DetailRow(
        batch=case.batch,
        history_len=case.history_len,
        section=section,
        name=name,
        shape=shape,
        ms=ms,
        tflops=_metric_tflops(flops, ms),
        gbps=_metric_gbps(bytes_, ms),
        bytes_gb=bytes_ / 1e9,
    )
    rows.append(row)
    return row


def run_detail_benchmark(args: argparse.Namespace) -> list[DetailCase]:
    cfg = load_config(args.config)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("highest")
    runner = V32DecodeRunner(cfg, args.mode, args.post_proj)

    cases: list[DetailCase] = []
    for batch, history_len in iter_cases(args):
        print(f"[detail] batch={batch} history_len={history_len}", flush=True)
        case = runner.make_case(batch, history_len)
        projected = runner.project(case)
        indices = runner.run_indexer(case, projected)
        out = runner.run_attention(case, projected, indices)
        torch.cuda.synchronize()
        output_shape = "x".join(str(v) for v in out.shape)
        del out

        rows: list[DetailRow] = []
        b = case.batch
        if args.detail_scope == "iter":
            projected = runner.project(case)
            runner.update_index_cache(case, projected)
            logits = runner.compute_index_logits(case, projected)
            indices = runner.select_topk(case, logits)
            attn_out = runner.sparse_attention(case, projected, indices)
            value_by_head = attn_out.transpose(0, 1).contiguous()
            head_by_group = runner.wv_b(value_by_head)
            head_out = head_by_group.transpose(0, 1).contiguous()
            head_flat = head_out.view(b, cfg.n_heads * cfg.v_head_dim)
            _add_detail_row(
                rows,
                case,
                "iter",
                "decode_iter",
                f"batch={b}, history={history_len}, full project+indexer+attention",
                lambda case=case: runner.run_once(case),
                _decode_iter_flops(case, cfg),
                _decode_iter_bytes(case, cfg, runner, projected, logits, head_flat),
                args.warmups,
                args.iters,
            )
            cases.append(
                DetailCase(
                    batch=batch, history_len=history_len, output_shape=output_shape, rows=rows
                )
            )
            del (
                case,
                projected,
                indices,
                logits,
                attn_out,
                value_by_head,
                head_by_group,
                head_out,
                head_flat,
            )
            cleanup()
            continue

        _add_detail_row(
            rows,
            case,
            "projection",
            "wq_a",
            f"m={b}, n={cfg.q_lora_rank}, k={cfg.dim}",
            lambda case=case: runner.wq_a(case.x),
            _linear_flops(b, cfg.q_lora_rank, cfg.dim),
            _linear_effective_bytes(case.x, runner.wq_a, cfg.q_lora_rank),
            args.warmups,
            args.iters,
        )
        qr = runner.ops.q_norm(runner.wq_a(case.x))
        _add_detail_row(
            rows,
            case,
            "projection",
            "wq_b",
            f"m={b}, n={cfg.n_heads * cfg.q_proj_head_dim}, k={cfg.q_lora_rank}",
            lambda qr=qr: runner.wq_b(qr),
            _linear_flops(b, cfg.n_heads * cfg.q_proj_head_dim, cfg.q_lora_rank),
            _linear_effective_bytes(qr, runner.wq_b, cfg.n_heads * cfg.q_proj_head_dim),
            args.warmups,
            args.iters,
        )
        q_proj = runner.wq_b(qr).view(b, cfg.n_heads, cfg.q_proj_head_dim)
        q_nope, _ = torch.split(q_proj, [cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)
        q_nope_by_head = q_nope.transpose(0, 1).contiguous()
        _add_detail_row(
            rows,
            case,
            "projection",
            "wk_b_grouped",
            f"groups={cfg.n_heads}, m/group={b}, n={cfg.kv_lora_rank}, k={cfg.qk_nope_head_dim}",
            lambda q_nope_by_head=q_nope_by_head: runner.wk_b(q_nope_by_head),
            2 * cfg.n_heads * b * cfg.kv_lora_rank * cfg.qk_nope_head_dim,
            _grouped_linear_effective_bytes(q_nope_by_head, runner.wk_b),
            args.warmups,
            args.iters,
        )
        _add_detail_row(
            rows,
            case,
            "projection",
            "wkv_a",
            f"m={b}, n={cfg.qk_head_dim}, k={cfg.dim}",
            lambda case=case: runner.wkv_a(case.x),
            _linear_flops(b, cfg.qk_head_dim, cfg.dim),
            _linear_effective_bytes(case.x, runner.wkv_a, cfg.qk_head_dim),
            args.warmups,
            args.iters,
        )
        _add_detail_row(
            rows,
            case,
            "projection",
            "index_wqi",
            f"m={b}, n={cfg.index_n_heads * cfg.index_head_dim}, k={cfg.q_lora_rank}",
            lambda qr=qr: runner.index_wqi(qr),
            _linear_flops(b, cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank),
            _linear_effective_bytes(qr, runner.index_wqi, cfg.index_n_heads * cfg.index_head_dim),
            args.warmups,
            args.iters,
        )
        _add_detail_row(
            rows,
            case,
            "projection",
            "index_wki",
            f"m={b}, n={cfg.index_head_dim}, k={cfg.dim}",
            lambda case=case: runner.index_wki(case.x),
            _linear_flops(b, cfg.index_head_dim, cfg.dim),
            _linear_effective_bytes(case.x, runner.index_wki, cfg.index_head_dim),
            args.warmups,
            args.iters,
        )
        _add_detail_row(
            rows,
            case,
            "projection",
            "index_weights",
            f"m={b}, n={cfg.index_n_heads}, k={cfg.dim}",
            lambda case=case: runner.index_weights(case.x),
            _linear_flops(b, cfg.index_n_heads, cfg.dim),
            _linear_effective_bytes(case.x, runner.index_weights, cfg.index_n_heads),
            args.warmups,
            args.iters,
        )

        projected = runner.project(case)
        index_update_bytes = _index_update_bytes(case, cfg)
        index_logits_flops = 2 * b * case.total_len * cfg.index_n_heads * cfg.index_head_dim
        _add_detail_row(
            rows,
            case,
            "indexer",
            "index_cache_update",
            f"batch={b}, record={cfg.index_head_dim + 4}B",
            lambda case=case, projected=projected: runner.update_index_cache(case, projected),
            0,
            index_update_bytes,
            args.warmups,
            args.iters,
        )
        _add_detail_row(
            rows,
            case,
            "indexer",
            "index_logits_deepgemm",
            f"batch={b}, tokens={case.total_len}, heads={cfg.index_n_heads}, dim={cfg.index_head_dim}",
            lambda case=case, projected=projected: runner.compute_index_logits(case, projected),
            index_logits_flops,
            _index_logits_bytes(case, cfg),
            args.warmups,
            args.iters,
        )
        logits = runner.compute_index_logits(case, projected)
        topk_bytes = _index_topk_bytes(case, cfg, logits)
        _add_detail_row(
            rows,
            case,
            "indexer",
            "index_topk",
            f"batch={b}, logits={tuple(logits.shape)}, topk={min(cfg.index_topk, case.total_len)}",
            lambda case=case, logits=logits: runner.select_topk(case, logits),
            0,
            topk_bytes,
            args.warmups,
            args.iters,
        )
        indices = runner.run_indexer(case, projected)

        attn_update_bytes = _attn_update_bytes(case, cfg)
        sparse_flops = _sparse_mla_flops(case, cfg)
        sparse_bytes = _sparse_mla_bytes(case, cfg)
        _add_detail_row(
            rows,
            case,
            "attention",
            "attn_cache_update",
            f"batch={b}, record=656B",
            lambda case=case, projected=projected: runner.update_attention_cache(case, projected),
            0,
            attn_update_bytes,
            args.warmups,
            args.iters,
        )
        _add_detail_row(
            rows,
            case,
            "attention",
            "sparse_mla_decode",
            f"batch={b}, heads={cfg.n_heads}, topk={min(cfg.index_topk, case.total_len)}, qk={cfg.qk_head_dim}, v={cfg.kv_lora_rank}",
            lambda case=case, projected=projected, indices=indices: runner.sparse_attention(
                case, projected, indices
            ),
            sparse_flops,
            sparse_bytes,
            args.warmups,
            args.iters,
        )
        attn_out = runner.sparse_attention(case, projected, indices)
        value_by_head = attn_out.transpose(0, 1).contiguous()
        wv_flops = 2 * cfg.n_heads * b * cfg.v_head_dim * cfg.kv_lora_rank
        wv_bytes = _grouped_linear_effective_bytes(value_by_head, runner.wv_b)
        _add_detail_row(
            rows,
            case,
            "projection",
            "post_wv_b_grouped",
            f"groups={cfg.n_heads}, m/group={b}, n={cfg.v_head_dim}, k={cfg.kv_lora_rank}",
            lambda value_by_head=value_by_head: runner.wv_b(value_by_head),
            wv_flops,
            wv_bytes,
            args.warmups,
            args.iters,
        )
        head_by_group = runner.wv_b(value_by_head)
        head_out = head_by_group.transpose(0, 1).contiguous()
        head_flat = head_out.view(b, cfg.n_heads * cfg.v_head_dim)
        wo_flops = _linear_flops(b, cfg.dim, cfg.n_heads * cfg.v_head_dim)
        wo_bytes = _linear_effective_bytes(head_flat, runner.wo, cfg.dim)
        _add_detail_row(
            rows,
            case,
            "projection",
            "post_wo",
            f"m={b}, n={cfg.dim}, k={cfg.n_heads * cfg.v_head_dim}",
            lambda head_flat=head_flat: runner.wo(head_flat),
            wo_flops,
            wo_bytes,
            args.warmups,
            args.iters,
        )

        cases.append(
            DetailCase(batch=batch, history_len=history_len, output_shape=output_shape, rows=rows)
        )
        del (
            case,
            projected,
            indices,
            logits,
            attn_out,
            value_by_head,
            head_by_group,
            head_out,
            head_flat,
        )
        cleanup()
    return cases


def format_detail_markdown(cases: list[DetailCase], args: argparse.Namespace) -> str:
    cfg = load_config(args.config)
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# DeepSeek V3.2 detailed decode benchmark",
        "",
        f"Date: {now}",
        "",
        "This benchmark reports concrete projection, indexer, and attention timings for the synthetic one-layer decode path. All tensors stay on GPU; no KV offload path is used.",
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
        f"mode={args.mode}, post_proj={args.post_proj}, detail_scope={args.detail_scope}, warmups={args.warmups}, iters={args.iters}",
        "```",
        "",
        "Model shape:",
        "",
        f"- dim={cfg.dim}, heads={cfg.n_heads}, q_lora={cfg.q_lora_rank}, kv_lora={cfg.kv_lora_rank}",
        f"- MLA q_dim={cfg.qk_head_dim}, v_dim={cfg.kv_lora_rank}, sparse topk={cfg.index_topk}",
        f"- Indexer heads={cfg.index_n_heads}, head_dim={cfg.index_head_dim}",
        "",
        "Method:",
        "",
        "- `detail_scope=modules` benchmarks isolated module calls only; it does not include aggregate indexer/attention/iteration rows.",
        "- `detail_scope=iter` benchmarks only the full decode iteration (`project -> indexer -> attention/post`).",
        "- Projection rows time the DeepGEMM wrapper call, including activation quantization, and exclude surrounding RMSNorm/split/cat/transposes unless the row name says total.",
        "- TFlop/s is throughput computed as math FLOPs divided by CUDA-event time. Rows with no meaningful FLOP model show 0.00.",
        "- Bandwidth is effective bytes divided by CUDA-event time. Bytes include the obvious tensor/cache input, quantized activation/scale buffers, persistent quantized weights/scales, and output; it is not a hardware counter.",
        "- Cache update rows measure paged slot writes of prepacked current-token records. Current-token KV/index packing is excluded from these rows.",
        "",
        "Command:",
        "",
        "```bash",
        " ".join(os.sys.argv),
        "```",
        "",
    ]
    for case in cases:
        lines += [
            f"## Batch {case.batch}, History KV {case.history_len}",
            "",
            f"Output shape: `{case.output_shape}`",
            "",
            "| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in case.rows:
            lines.append(
                f"| {row.section} | `{row.name}` | {row.shape} | {row.ms:.4f} | {row.tflops:.2f} | {row.gbps:.2f} | {row.bytes_gb:.4f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _event_device_us(event) -> float:
    for attr in (
        "self_device_time_total",
        "device_time_total",
        "self_cuda_time_total",
        "cuda_time_total",
    ):
        value = getattr(event, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def profile_cuda_kernels(fn: Callable[[], object]) -> tuple[float, int, float]:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    start = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], acc_events=True) as prof:
        fn()
        torch.cuda.synchronize()
    cpu_wall_ms = (time.perf_counter() - start) * 1e3
    kernel_us = 0.0
    kernel_count = 0
    for event in prof.events():
        if str(getattr(event, "device_type", "")).endswith("CUDA"):
            kernel_us += _event_device_us(event)
            kernel_count += 1
    return kernel_us / 1e3, kernel_count, cpu_wall_ms


def profile_step(name: str, fn: Callable[[], object], warmups: int, iters: int) -> SegmentProfile:
    event_ms = bench_cuda(fn, warmups, iters, f"profile/{name}")
    kernel_ms, kernel_count, cpu_wall_ms = profile_cuda_kernels(fn)
    return SegmentProfile(
        name=name,
        event_ms=event_ms,
        kernel_ms=kernel_ms,
        launch_gap_ms=max(event_ms - kernel_ms, 0.0),
        cpu_wall_ms=cpu_wall_ms,
        kernel_count=kernel_count,
    )


def run_profile(args: argparse.Namespace) -> list[CaseProfile]:
    cfg = load_config(args.config)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("highest")
    runner = V32DecodeRunner(cfg, args.mode, args.post_proj)

    profiles: list[CaseProfile] = []
    for batch, history_len in iter_cases(args):
        print(f"[profile] batch={batch} history_len={history_len}", flush=True)
        case = runner.make_case(batch, history_len)
        projected = runner.project(case)
        runner.update_index_cache(case, projected)
        logits = runner.compute_index_logits(case, projected)
        indices = runner.select_topk(case, logits)
        runner.update_attention_cache(case, projected)
        attn_out = runner.sparse_attention(case, projected, indices)
        head_out = runner.post_wv_b(attn_out)
        runner.post_wo(head_out)
        torch.cuda.synchronize()

        segments: list[SegmentProfile] = []
        segments.append(
            profile_step(
                "projection_all", lambda case=case: runner.project(case), args.warmups, args.iters
            )
        )
        projected = runner.project(case)
        segments.append(
            profile_step(
                "index_cache_update",
                lambda case=case, projected=projected: runner.update_index_cache(case, projected),
                args.warmups,
                args.iters,
            )
        )
        segments.append(
            profile_step(
                "index_logits_deepgemm",
                lambda case=case, projected=projected: runner.compute_index_logits(case, projected),
                args.warmups,
                args.iters,
            )
        )
        logits = runner.compute_index_logits(case, projected)
        segments.append(
            profile_step(
                "index_topk",
                lambda case=case, logits=logits: runner.select_topk(case, logits),
                args.warmups,
                args.iters,
            )
        )
        indices = runner.select_topk(case, logits)
        segments.append(
            profile_step(
                "attn_cache_update",
                lambda case=case, projected=projected: runner.update_attention_cache(
                    case, projected
                ),
                args.warmups,
                args.iters,
            )
        )
        segments.append(
            profile_step(
                "sparse_mla_decode",
                lambda case=case, projected=projected, indices=indices: runner.sparse_attention(
                    case, projected, indices
                ),
                args.warmups,
                args.iters,
            )
        )
        attn_out = runner.sparse_attention(case, projected, indices)
        segments.append(
            profile_step(
                "post_wv_b",
                lambda attn_out=attn_out: runner.post_wv_b(attn_out),
                args.warmups,
                args.iters,
            )
        )
        head_out = runner.post_wv_b(attn_out)
        segments.append(
            profile_step(
                "post_wo",
                lambda head_out=head_out: runner.post_wo(head_out),
                args.warmups,
                args.iters,
            )
        )
        segments.append(
            profile_step(
                "e2e_decode_layer",
                lambda case=case: runner.run_once(case),
                args.warmups,
                args.iters,
            )
        )

        profiles.append(CaseProfile(batch=batch, history_len=history_len, segments=segments))
        del case, projected, logits, indices, attn_out, head_out
        cleanup()
    return profiles


def format_profile_markdown(profiles: list[CaseProfile], args: argparse.Namespace) -> str:
    cfg = load_config(args.config)
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# DeepSeek V3.2 model decode profile",
        "",
        f"Date: {now}",
        "",
        "This profile uses DeepGEMM for projection, indexer logits, W_VB, and W_O, and flash_mla_sm120/sparse_mla_sm120 for sparse MLA decode. All tensors stay on GPU; no KV offload path is used.",
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
        f"mode={args.mode}, post_proj={args.post_proj}, warmups={args.warmups}, iters={args.iters}",
        "```",
        "",
        "Model shape:",
        "",
        f"- dim={cfg.dim}, heads={cfg.n_heads}, q_lora={cfg.q_lora_rank}, kv_lora={cfg.kv_lora_rank}",
        f"- MLA decode q_dim={cfg.qk_head_dim}, sparse topk={cfg.index_topk}",
        f"- Indexer heads={cfg.index_n_heads}, head_dim={cfg.index_head_dim}",
        "",
        "Method:",
        "",
        "- event_ms is CUDA event elapsed time for the segment and includes stream idle gaps between kernels.",
        "- kernel_ms is the sum of CUDA kernel self times reported by torch.profiler for one profiled segment run.",
        "- launch_gap_ms = max(event_ms - kernel_ms, 0). It approximates CPU launch overhead, stream idle time, and profiler/event mismatch; it is directional, not a hardware counter.",
        "- cpu_wall_ms is wall time for the profiled run and includes profiler overhead, so use it mainly as a sanity check.",
        "",
        "Summary:",
        "",
    ]
    gap_dominates = 0
    for case in profiles:
        e2e = next((seg for seg in case.segments if seg.name == "e2e_decode_layer"), None)
        top_gap = max(
            (seg for seg in case.segments if seg.name != "e2e_decode_layer"),
            key=lambda seg: seg.launch_gap_ms,
        )
        if e2e is not None:
            gap_pct = 100.0 * e2e.launch_gap_ms / max(e2e.event_ms, 1e-9)
            kernel_pct = 100.0 * e2e.kernel_ms / max(e2e.event_ms, 1e-9)
            if gap_pct > kernel_pct:
                gap_dominates += 1
            lines.append(
                f"- Batch {case.batch}, history {case.history_len}: e2e {e2e.event_ms:.3f} ms, "
                f"kernel self time {e2e.kernel_ms:.3f} ms ({kernel_pct:.1f}%), "
                f"launch/gap {e2e.launch_gap_ms:.3f} ms ({gap_pct:.1f}%). "
                f"Largest gap segment: {top_gap.name} ({top_gap.launch_gap_ms:.3f} ms, {top_gap.kernel_count} kernels)."
            )
    if gap_dominates:
        lines += [
            "",
            "Overall: the measured decode path is launch/scheduling dominated, not kernel-execution dominated. The main source is thousands of small kernels in projection/post-projection activation quantization and grouped GEMM setup, especially projection_all and post_wv_b. The sparse_mla_decode kernel itself is mostly kernel-execution time and has very small launch/gap overhead.",
            "",
        ]
    for case in profiles:
        lines += [
            f"## Batch {case.batch}, History KV {case.history_len}",
            "",
            "| Segment | event ms | kernel ms | launch/gap ms | kernels | CPU wall ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for seg in case.segments:
            lines.append(
                f"| {seg.name} | {seg.event_ms:.4f} | {seg.kernel_ms:.4f} | {seg.launch_gap_ms:.4f} | {seg.kernel_count} | {seg.cpu_wall_ms:.4f} |"
            )
        e2e = next((seg for seg in case.segments if seg.name == "e2e_decode_layer"), None)
        if e2e is not None:
            if e2e.kernel_ms >= 0.7 * e2e.event_ms:
                verdict = "kernel execution dominates the measured e2e time"
            else:
                verdict = "launch/scheduling gaps are a major part of the measured e2e time"
            lines += ["", f"Interpretation: {verdict} for this case.", ""]
    return "\n".join(lines) + "\n"


def format_markdown(results: list[BenchResult], args: argparse.Namespace) -> str:
    cfg = load_config(args.config)
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# DeepSeek V3.2 model decode benchmark",
        "",
        f"Date: {now}",
        "",
        "This benchmark runs one synthetic decode layer with full projection, sparse indexer, sparse MLA attention, per-head value projection, and final output projection. All tensors stay on GPU; no KV offload path is used.",
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
        f"mode={args.mode}, post_proj={args.post_proj}, warmups={args.warmups}, iters={args.iters}, quick={args.quick}",
        "```",
        "",
        "Model shape:",
        "",
        f"- `dim={cfg.dim}`, `heads={cfg.n_heads}`, `q_lora={cfg.q_lora_rank}`, `kv_lora={cfg.kv_lora_rank}`",
        f"- MLA decode `q_dim={cfg.qk_head_dim}` (`latent={cfg.kv_lora_rank}` + `rope={cfg.qk_rope_head_dim}`), `d_v={cfg.kv_lora_rank}`, sparse `topk={cfg.index_topk}`",
        f"- Indexer `heads={cfg.index_n_heads}`, `head_dim={cfg.index_head_dim}`",
        "",
        "Command:",
        "",
        "```bash",
        " ".join(os.sys.argv),
        "```",
        "",
        "| Batch | History KV len | Total len | Top-k | E2E ms | Projection ms | Indexer ms | Attention+post ms | Output |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.batch} | {r.history_len} | {r.total_tokens} | {r.topk} | "
            f"{r.e2e_ms:.4f} | {r.projection_ms:.4f} | {r.indexer_ms:.4f} | {r.attention_ms:.4f} | `{r.output_shape}` |"
        )
    lines += [
        "",
        "Notes:",
        "",
        "- E2E includes FlashInfer RMSNorm/LayerNorm and MLA/indexer YaRN RoPE, indexer Hadamard rotation, activation quantization, projected-token KV/index cache writes, paged-index logits, top-k, sparse MLA, W_VB and W_O. RoPE table construction is excluded.",
        "- Weights and historical cache remain synthetic. This is an attention sublayer benchmark, without embedding, residual addition, MLP or LM head. Norm affine parameters are unit/zero. Byte/FLOP estimates cover the major GEMMs and cache operations, not every auxiliary operation.",
        "- The previous sparse-then-DeepGEMM illegal-address was caused by non-canonical strides on size-1 grouped-GEMM dimensions after transpose/contiguous; GroupedLinear now canonicalizes strides before launching DeepGEMM.",
        "- Projection/indexer/attention columns are measured as isolated sections and therefore are not expected to add up exactly to E2E.",
        "- The sparse MLA decode wrapper is used for batch sizes up to 64 decode tokens, matching the current decode kernel limit.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--quick", action="store_true", help="run a small smoke/benchmark sweep")
    parser.add_argument("--batch-sizes", default="1,4,8,16,32,64")
    parser.add_argument("--history-lens", default="4096,8192,16384,32768")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--mode", choices=["fp8", "fp8_fp4w"], default="fp8")
    parser.add_argument("--post-proj", choices=["torch", "deepgemm"], default="deepgemm")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="write a segmented launch-vs-kernel profile instead of the sweep table",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="write per-projection/indexer/attention timing, TFlop/s, and bandwidth tables",
    )
    parser.add_argument(
        "--detail-scope",
        choices=["modules", "iter"],
        default="modules",
        help="with --detail, benchmark isolated modules or the full decode iteration",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.detail:
        details = run_detail_benchmark(args)
        md = format_detail_markdown(details, args)
    elif args.profile:
        profiles = run_profile(args)
        md = format_profile_markdown(profiles, args)
    else:
        results = run_benchmark(args)
        md = format_markdown(results, args)
    print()
    print(md)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
