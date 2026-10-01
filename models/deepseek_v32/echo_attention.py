"""Chunked DeepSeek indexer → ECHO prefetch → exact recall → sparse MLA."""

from contextlib import nullcontext

import torch

from cache.sparse_token_cache import SparseTokenCache, WorkingSetTooLarge
from operators.deepseek_v32.attention.device_only.mla import sparse_mla
from operators.deepseek_v32.attention.offload.mla import sparse_mla_from_pool


class EchoAttentionRunner:
    """One layer's attention execution; the model owns multi-layer commit.

    Indexer keys remain on the model GPU. Only the main MLA token records use
    offload. A too-large exact union splits attention consumption, preserving
    every query's original top-k and the full indexer search domain.
    """

    def __init__(self, attention, capacity, *, offload=False, slots=16384, chunk_size=1024):
        if chunk_size < 1 or offload and slots < chunk_size:
            raise ValueError("chunk_size must be positive and fit in offload slots")
        self.attention = attention
        self.cfg = attention.cfg
        self.chunk_size = chunk_size
        self.cache = SparseTokenCache(
            capacity,
            self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim,
            device=attention.device,
            slots=min(slots, capacity) if offload else None,
        )
        self.index_keys = torch.empty(
            (capacity, self.cfg.index_head_dim),
            dtype=torch.float8_e4m3fn,
            device=attention.device,
        )
        self.index_scales = torch.empty(capacity, dtype=torch.float32, device=attention.device)
        self.offset = torch.zeros(16, device=attention.device, dtype=torch.float32)
        self.last_indices = None
        self.capture_hook = None

    def _consume(self, q, indices, scope):
        if not self.cache.offload:
            with scope("sparse_mla"):
                return sparse_mla(q, self.cache.records, indices, self.cfg.attention_scale)
        try:
            with scope("offload_exact_recall"):
                physical = self.cache.ensure(indices)
        except WorkingSetTooLarge:
            if len(q) == 1:
                raise ValueError(
                    "offload slots must fit at least one query's exact selection"
                ) from None
            middle = len(q) // 2
            left = self._consume(q[:middle], indices[:middle], scope)
            right = self._consume(q[middle:], indices[middle:], scope)
            return torch.cat((left, right))
        with scope("sparse_mla"):
            return sparse_mla_from_pool(q, self.cache.records, physical, self.cfg.attention_scale)

    def forward(self, hidden, *, scope=None, capture_indices=False, normalized=False):
        from operators.deepseek_v32.indexer.echo import logits as index_logits

        scope = scope or (lambda _: nullcontext())
        outputs = []
        selections = []
        for start in range(0, len(hidden), self.chunk_size):
            part = hidden[start : start + self.chunk_size]
            position = self.cache.written
            with scope("attention_projection"):
                p = self.attention.project(part, position, normalized=normalized)
            with scope("cache_write"):
                self.cache.append(p.kv)
                end = position + len(part)
                self.index_keys[position:end] = p.index_k
                self.index_scales[position:end] = p.index_scale
            with scope("offload_prepare"):
                prefetch = self.cache.prepare_prefetch(position, len(part), self.offset)
            with scope("indexer_prefetch" if prefetch is not None else "indexer"):
                scores = index_logits(
                    p.index_q,
                    self.index_keys[:end],
                    p.index_weights,
                    self.index_scales[:end],
                    position,
                    prefetch=prefetch,
                )
            with scope("exact_topk"):
                # Mask explicitly before exact selection. The native kernel may
                # write padded/future logits as part of its 128-token tile.
                ends = torch.arange(position + 1, end + 1, device=part.device)
                valid = torch.arange(end, device=part.device)[None, :] < ends[:, None]
                scores = scores[:, :end].masked_fill(~valid, -torch.inf)
                values, indices = torch.topk(scores, min(self.cfg.index_topk, end), dim=-1)
                indices = torch.where(torch.isfinite(values), indices, -1).int()
                # ECHO's four-query offset is only a bucket-resolution hint.
                # Exclude causal padding to keep it finite for chunked prefill.
                tail = scores[-4:]
                finite = torch.isfinite(tail)
                self.offset[0] = tail.masked_fill(~finite, 0).sum() / finite.sum().clamp_min(1)
            del scores, valid, values
            if capture_indices:
                selections.append(indices)
            if self.capture_hook is not None:
                self.capture_hook(
                    p, self.index_keys[:end], self.index_scales[:end], indices, self.cache, position
                )
            attn = self._consume(p.q, indices, scope)
            with scope("attention_output"):
                outputs.append(self.attention.output(attn))
        self.last_indices = selections if capture_indices else None
        return torch.cat(outputs)

    def reset(self):
        self.cache.reset()
        self.offset.zero_()
