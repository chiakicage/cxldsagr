"""Indexer and main-attention boundaries independent of a model's KV layout.

An indexer returns logical block IDs, not gathered KV tensors. Main attention
receives the selection and cache access together so a future hardware backend
can overlap attention computation with fetching offloaded records internally.
The shared dense adapter and model-owned resident sparse adapters use this boundary.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from cache.contracts import CacheAccess


@dataclass(frozen=True)
class BlockSelection:
    """Logical block IDs with shape [query, head_or_group, selected_block].

    ``block_ids`` is integral; ``valid_mask`` has the same shape and excludes
    padded entries. Query/head axes may have size one to share a selection.
    Block numbering is relative to the request, independent of physical HBM or
    host placement. The model defines head grouping and legal causal blocks;
    the attention backend still enforces the token-level causal mask.
    """

    block_ids: torch.Tensor
    block_size: int
    valid_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class AttentionContext:
    layer_idx: int
    query_start: int
    query_length: int
    auxiliary_state: Any = None


class Indexer(Protocol):
    def __call__(
        self, q: torch.Tensor, cache_access: CacheAccess, context: AttentionContext
    ) -> BlockSelection: ...


class MainAttention(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        selection: BlockSelection | None,
        cache_access: CacheAccess,
        context: AttentionContext,
    ) -> torch.Tensor: ...


class DenseMainAttention:
    """Bridge resident NHD K/V to the existing ``attention(q, k, v)`` seam."""

    def __init__(self, attention: Callable):
        self.attention = attention

    def __call__(self, q, selection, cache_access, context):
        if selection is not None:
            raise NotImplementedError("Dense main attention does not implement block selection")
        layer_view = getattr(cache_access, "layer_view", None)
        if not callable(layer_view):
            raise NotImplementedError("Dense main attention requires resident cache layer views")
        records = layer_view(context.layer_idx)
        return self.attention(q, records["keys"], records["values"])


class ResidentLayerView:
    """Ephemeral cache access for a forward without a persistent KV session."""

    def __init__(self, layer_idx: int, **records: torch.Tensor):
        self.layer_idx = layer_idx
        self.records = records
        self.state = None
        self.length = 0
        self.max_seq_len = next(iter(records.values())).shape[0]

    def _check_layer(self, layer_idx):
        if layer_idx != self.layer_idx:
            raise IndexError("This resident view belongs to a different layer")

    def layer_view(self, layer_idx):
        self._check_layer(layer_idx)
        return self.records

    def get_layer_state(self, layer_idx):
        self._check_layer(layer_idx)
        return self.state

    def set_layer_state(self, layer_idx, state):
        self._check_layer(layer_idx)
        self.state = state
