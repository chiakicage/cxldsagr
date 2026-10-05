"""Lightweight attention selection and call contracts shared by model adapters and operators."""

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
class TokenSelection:
    """Logical token IDs with shape [query, selected_token].

    Token numbering is relative to the session. The consuming model defines
    padding and maps logical IDs to its physical cache, optionally fusing
    selection with prefetch. Constructing this value performs no tensor work.
    """

    token_ids: torch.Tensor


@dataclass(frozen=True)
class AttentionContext:
    layer_idx: int
    query_start: int
    query_length: int
    auxiliary_state: Any = None


class Indexer(Protocol):
    def __call__(
        self, q: torch.Tensor, cache_access: CacheAccess, context: AttentionContext
    ) -> BlockSelection | TokenSelection: ...


class MainAttention(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        selection: BlockSelection | TokenSelection | None,
        cache_access: CacheAccess,
        context: AttentionContext,
    ) -> torch.Tensor: ...
