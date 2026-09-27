"""NOSA block selection on resident K, with an unchanged QA-only default.

The scoring follows ``CompressK`` / ``compressed_attention`` and the stage-one
oracle in thunlp/NOSA at commit 1cbee77d607f9051b206a09c862bea28becb9e67:
https://github.com/thunlp/NOSA/blob/1cbee77d607f9051b206a09c862bea28becb9e67/modeling_llama_nosa.py
https://github.com/thunlp/NOSA/blob/1cbee77d607f9051b206a09c862bea28becb9e67/dependencies/infllmv2_cuda_impl/tests/test_stage1.py

This variant spends all dynamic blocks on query-aware selection: 47 with the
default 64-block budget, or 15 with a 32-block budget. It uses exactly 16 local
blocks, including the query's block, as specified in NOSA's
paper (https://arxiv.org/html/2510.13602v2#A2.SS2.SSS2). The upstream pooling
kernels' inclusive local boundary instead reserves 17 when configured with 16.
``mode="nosa"`` reproduces cxl-recsys's complete two-stage selection: its
inclusive local boundary reserves 17 blocks (current + 16 predecessors),
query-aware selection retains 33 blocks including sink/local, then learned
query-agnostic CIS fills a 64-block budget. Model-dtype compression and score
rounding match that implementation. All modes return logical block IDs.
"""

from dataclasses import dataclass

import torch

from cache.contracts import CacheAccess
from layers.attention import AttentionContext, BlockSelection
from models.nosa.scoring import NosaAttentionState, compress_sequence


@dataclass(frozen=True)
class NosaSelectionPolicy:
    """NOSA query-aware-only policy, defaulting to 64 blocks per query/KV head.

    Sink/local blocks are excluded from dynamic top-k, after score computation.
    Short contexts select only available causal blocks; invalid output slots
    are padded. The attention backend must still mask future tokens within the
    final partially visible block.
    """

    block_size: int = 64
    block_budget: int = 64
    sink_blocks: int = 1
    local_blocks: int = 16
    topk_blocks: int = 47
    compression_kernel_size: int = 32
    compression_stride: int = 16


class NosaIndexer:
    """Analyze post-RoPE Q/K without modifying the model or its cache state.

    Q has shape ``[query, query_head, head_dim]`` and the resident ``keys`` view
    has shape ``[token, kv_head, head_dim]``. Each contiguous group of query
    heads shares one KV head. Results have shape ``[query, kv_head, block_budget]``, with
    ascending valid IDs, ``-1`` padding, and an explicit validity mask.

    The default query-aware-only path uses FP32 compression and softmax.
    Full ``mode="nosa"`` retains checkpoint-dtype compression/score rounding,
    consumes resident CIS scores, and supports the Hopper Triton backend.
    Query chunking bounds temporary score storage. CUDA reference matmul
    precision is temporarily set to IEEE FP32 and restored afterward; this
    indexer is intended for the project's serial execution path.
    """

    def __init__(
        self,
        query_chunk_size: int = 64,
        *,
        block_budget: int = 64,
        mode: str = "query_aware",
        backend: str = "reference",
    ):
        if type(query_chunk_size) is not int or query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be a positive integer")
        if type(block_budget) is not int or block_budget not in (32, 64):
            raise ValueError("block_budget must be the integer 32 or 64")
        if mode not in ("query_aware", "nosa"):
            raise ValueError("NOSA indexer mode must be query_aware or nosa")
        if backend not in ("reference", "auto", "triton"):
            raise ValueError("NOSA indexer backend must be reference, auto or triton")
        if mode == "query_aware" and backend != "reference":
            raise ValueError("Query-aware-only analysis requires the reference backend")
        if mode == "nosa" and block_budget != 64:
            raise ValueError("Full NOSA selection requires block_budget=64")
        defaults = NosaSelectionPolicy()
        self.policy = NosaSelectionPolicy(
            block_budget=block_budget,
            topk_blocks=block_budget - defaults.sink_blocks - defaults.local_blocks,
        )
        self.query_chunk_size = query_chunk_size
        self.mode = mode
        self.backend = backend
        if mode == "nosa":
            self.policy = NosaSelectionPolicy(local_blocks=17, topk_blocks=15)

    @torch.no_grad()
    def __call__(self, q, cache_access: CacheAccess, context: AttentionContext) -> BlockSelection:
        layer_view = getattr(cache_access, "layer_view", None)
        if not callable(layer_view):
            raise NotImplementedError("NOSA reference indexer requires resident cache layer views")
        records = layer_view(context.layer_idx)
        if "keys" not in records:
            raise ValueError("NOSA resident cache view must contain keys")
        keys = records["keys"]
        self._validate_inputs(q, keys, context)
        # An offline view may also contain later queries. Trim it to this call's
        # last query; earlier queries still need the per-compressed-window mask.
        keys = keys[: context.query_start + context.query_length]
        if not torch.isfinite(q).all() or not torch.isfinite(keys).all():
            raise ValueError("NOSA indexer requires finite Q and K")
        cis = None
        if self.mode == "nosa":
            state = context.auxiliary_state
            cis = (
                state.cis_scores
                if isinstance(state, NosaAttentionState)
                else records.get("cis_scores")
            )
            if not isinstance(cis, torch.Tensor):
                raise ValueError("Full NOSA selection requires resident cis_scores")
            if (
                cis.ndim != 2
                or cis.shape[0] < len(keys)
                or cis.shape[1] != keys.shape[1]
                or cis.device != keys.device
                or cis.dtype != keys.dtype
                or q.dtype != keys.dtype
            ):
                raise ValueError(
                    "CIS must cover resident K with [token, KV head] shape and Q/K dtype/device"
                )
            cis = cis[: len(keys)]
            if not torch.isfinite(cis).all():
                raise ValueError("NOSA indexer requires finite CIS scores")

        old_precision = None
        if q.device.type == "cuda":
            old_precision = torch.backends.cuda.matmul.fp32_precision
            torch.backends.cuda.matmul.fp32_precision = "ieee"
        try:
            with torch.autocast(device_type=q.device.type, enabled=False):
                if self.mode == "nosa":
                    return self._select_nosa(q, keys, cis, context.query_start)
                return self._select(q, keys, context.query_start)
        finally:
            if old_precision is not None:
                torch.backends.cuda.matmul.fp32_precision = old_precision

    @staticmethod
    def _validate_inputs(q, keys, context):
        for name, tensor in (("Q", q), ("K", keys)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if tensor.ndim != 3 or any(size <= 0 for size in tensor.shape):
                raise ValueError(f"{name} must have nonempty [token, head, head_dim] shape")
            if not tensor.is_floating_point():
                raise ValueError(f"{name} must have a floating-point dtype")
        if q.device != keys.device:
            raise ValueError("Q and K must be on the same device")
        if q.shape[-1] != keys.shape[-1] or q.shape[1] % keys.shape[1]:
            raise ValueError(
                "Q/K head dimensions must agree and Q heads must divide into KV groups"
            )
        if type(context.query_start) is not int or context.query_start < 0:
            raise ValueError("query_start must be a nonnegative integer")
        if type(context.query_length) is not int or context.query_length != q.shape[0]:
            raise ValueError("query_length must match the number of Q tokens")
        if keys.shape[0] < context.query_start + context.query_length:
            raise ValueError("Resident K does not cover the query positions")

    def _select(self, q, keys, query_start):
        policy = self.policy
        query_count, query_heads, head_dim = q.shape
        kv_heads = keys.shape[1]
        group_size = query_heads // kv_heads
        block_count = (keys.shape[0] + policy.block_size - 1) // policy.block_size
        blocks = torch.arange(block_count, device=q.device)
        block_ids = torch.full(
            (query_count, kv_heads, policy.block_budget), -1, dtype=torch.long, device=q.device
        )

        kernel = policy.compression_kernel_size
        stride = policy.compression_stride
        if keys.shape[0] >= kernel:
            # Pool post-RoPE K, not raw projection output. Convert before the
            # reduction so BF16 inputs do not round the compressed vectors.
            compressed_k = keys.float().unfold(0, kernel, stride).mean(dim=-1)
            compressed_count = compressed_k.shape[0]
            compressed_ends = torch.arange(compressed_count, device=q.device) * stride + kernel - 1
            # Each 64-token block overlaps five 32-token windows, starting at
            # compressed indices 4*b-1 through 4*b+3. There are no overlap weights.
            pool_indices = blocks[:, None] * (policy.block_size // stride) + torch.arange(
                -1, policy.block_size // stride, device=q.device
            )
            pool_valid = (pool_indices >= 0) & (pool_indices < compressed_count)
            pool_indices = pool_indices.clamp(0, compressed_count - 1)
        else:
            compressed_k = None

        for start in range(0, query_count, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, query_count)
            positions = torch.arange(query_start + start, query_start + stop, device=q.device)
            query_blocks = positions // policy.block_size
            visible_blocks = blocks[None, :] <= query_blocks[:, None]
            reserved = visible_blocks & (
                (blocks[None, :] < policy.sink_blocks)
                | (blocks[None, :] >= query_blocks[:, None] - policy.local_blocks + 1)
            )
            eligible = visible_blocks & ~reserved

            if compressed_k is None:
                # Fewer than 32 tokens: the only logical block is the sink.
                block_scores = torch.zeros(
                    (stop - start, kv_heads, block_count), dtype=torch.float32, device=q.device
                )
            else:
                grouped_q = q[start:stop].float().reshape(-1, kv_heads, group_size, head_dim)
                logits = torch.einsum("qhgd,mhd->qhgm", grouped_q, compressed_k)
                logits *= head_dim**-0.5
                causal = compressed_ends[None, :] <= positions[:, None]
                logits.masked_fill_(~causal[:, None, None, :], -torch.inf)
                # The first 31 tokens have no complete causal compressed key.
                # Avoid all-masked softmax NaNs, then zero their probabilities.
                logits.masked_fill_(~causal.any(dim=-1)[:, None, None, None], 0)
                probabilities = logits.softmax(dim=-1)
                probabilities.masked_fill_(~causal[:, None, None, :], 0)
                # Normalization is per Q head, before summing within a KV group.
                # Sink/local remain in the softmax denominator at this stage.
                grouped_scores = probabilities.sum(dim=2)
                block_scores = (
                    grouped_scores[..., pool_indices]
                    .masked_fill(~pool_valid[None, None, :, :], -torch.inf)
                    .amax(dim=-1)
                )

            block_scores.masked_fill_(~eligible[:, None, :], -torch.inf)
            dynamic_slots = min(policy.topk_blocks, block_count)
            # Stable sorting of ascending IDs makes ties select the smaller ID.
            ranked = block_scores.argsort(dim=-1, descending=True, stable=True)[..., :dynamic_slots]
            dynamic_count = eligible.sum(dim=-1).clamp_max(policy.topk_blocks)
            dynamic_valid = (
                torch.arange(dynamic_slots, device=q.device)[None, :] < dynamic_count[:, None]
            )
            selected = torch.zeros_like(block_scores, dtype=torch.bool)
            selected.scatter_(-1, ranked, dynamic_valid[:, None, :].expand_as(ranked))
            selected |= reserved[:, None, :]
            ordered = torch.where(selected, blocks, block_count).sort(dim=-1).values
            output_slots = min(policy.block_budget, block_count)
            ordered = ordered[..., :output_slots]
            block_ids[start:stop, :, :output_slots] = ordered.masked_fill(
                ordered == block_count, -1
            )

        return BlockSelection(block_ids, policy.block_size, valid_mask=block_ids >= 0)

    def _select_nosa(self, q, keys, cis, query_start):
        backend = self.backend
        if backend == "auto":
            backend = "triton" if q.is_cuda else "reference"
        positions = torch.arange(query_start, query_start + len(q), device=q.device)
        kv_heads = keys.shape[1]
        query = q.reshape(len(q), kv_heads, q.shape[1] // kv_heads, q.shape[-1])
        compressed_k, compressed_cis = compress_sequence(keys), compress_sequence(cis)
        result = torch.full((len(q), kv_heads, 64), -1, device=q.device, dtype=torch.long)
        if backend == "triton":
            from operators.sm90.nosa_indexer import select_blocks
        else:
            select_blocks = select_nosa_blocks_reference
            # Prepare the shared layout once; each query head still retains
            # its own FP32 softmax normalizer.
            compressed_k = prepare_indexer_keys(compressed_k)
        for start in range(0, len(q), self.query_chunk_size):
            end = min(start + self.query_chunk_size, len(q))
            selected = select_blocks(
                query[start:end], compressed_k, compressed_cis, positions[start:end], len(keys)
            )
            result[start:end, :, : selected.shape[-1]] = selected
        return BlockSelection(result, 64, valid_mask=result >= 0)


def prepare_indexer_keys(keys):
    """Preserve cxl-recsys's per-query GEMM operand layout and FP32 rounding."""
    if keys.shape[1] == 1:
        return keys.float().contiguous()
    return keys.float().permute(1, 2, 0).contiguous().permute(2, 0, 1)


def compressed_scores_reference(query, compressed_keys, positions):
    """Grouped Q ``[Q,Hkv,G,D]`` to compressed scores ``[Q,Hkv,C]``.

    Compute per-Q-head FP32 softmax before summing GQA heads; match the
    source's model-dtype score rounding before max pooling and top-k.
    """
    count = len(compressed_keys)
    if not count:
        return torch.empty((*query.shape[:2], 0), dtype=query.dtype, device=query.device)
    if len(query) == 1:
        logits = torch.einsum(
            "qhgd,qchd->qhgc", query.float(), compressed_keys.float().contiguous()[None]
        )
    else:
        keys = prepare_indexer_keys(compressed_keys)
        queries = query.float()
        logits = torch.stack(
            [
                torch.bmm(queries[:, head], keys[:, head].T.expand(len(query), -1, -1))
                for head in range(query.shape[1])
            ],
            dim=1,
        )
    logits *= query.shape[-1] ** -0.5
    ends = torch.arange(count, device=query.device) * 16 + 31
    causal = ends[None, :] <= positions[:, None]
    logits.masked_fill_(~causal[:, None, None, :], -torch.inf)
    logits.masked_fill_(~causal.any(-1)[:, None, None, None], 0)
    probability = logits.softmax(-1).masked_fill(~causal[:, None, None, :], 0)
    return probability.sum(2).to(query.dtype)


def select_nosa_blocks_reference(query, compressed_keys, compressed_cis, positions, total_length):
    """cxl-recsys two-stage 33/64 selection, with stable ascending-ID ties.

    The query-aware stage includes sink and exactly 17 inclusive local blocks;
    CIS fills the remaining slots after those first 33 blocks are protected.
    For short contexts every visible block is selected. IDs are sorted and
    future blocks become -1; the caller pads the final width to 64.
    """
    block_count = (total_length + 63) // 64
    block_ids = torch.arange(block_count, device=query.device)
    q_block = positions // 64
    if block_count <= 64:
        blocks = block_ids.expand(len(query), query.shape[1], -1)
        return blocks.masked_fill(blocks > q_block[:, None, None], -1)
    scores = compressed_scores_reference(query, compressed_keys, positions)
    windows = block_ids[:, None] * 4 - 1 + torch.arange(5, device=query.device)
    valid = (windows >= 0) & (windows < len(compressed_keys))
    safe = windows.clamp(0, len(compressed_keys) - 1)
    block_scores = scores[:, :, safe].masked_fill(~valid, -torch.inf).amax(-1)
    cis = compressed_cis.T[:, safe].masked_fill(~valid, -torch.inf).amax(-1)
    cis = cis[None].expand(len(query), -1, -1).clone()
    causal = block_ids[None, :] <= q_block[:, None]
    mandatory = (block_ids[None, :] == 0) | (causal & (q_block[:, None] <= block_ids[None, :] + 16))
    block_scores.masked_fill_(mandatory[:, None, :], torch.inf)
    block_scores.masked_fill_(~causal[:, None, :], -torch.inf)
    cis.masked_fill_(mandatory[:, None, :], torch.inf)
    cis.masked_fill_(~causal[:, None, :], -torch.inf)
    selected_q = block_scores.argsort(dim=-1, descending=True, stable=True)[..., :33]
    cis.scatter_(-1, selected_q, torch.inf)
    cis.masked_fill_(~causal[:, None, :], -torch.inf)
    selected = cis.argsort(dim=-1, descending=True, stable=True)[..., :64].sort(-1).values
    return selected.masked_fill(selected > q_block[:, None, None], -1)
