"""Reserved integration point for the original NOSA block-level indexer."""

from dataclasses import dataclass

from cache.contracts import CacheAccess
from layers.attention import AttentionContext, BlockSelection


@dataclass(frozen=True)
class NosaSelectionPolicy:
    """Default causal 64-token block policy, recorded without implementing top-k.

    For each query, reserve the first visible block as the attention sink and
    the last 16 visible blocks as the local window. Use the original NOSA block
    scores for top-47 among remaining blocks. Deduplicate sink/local/top-k;
    short contexts use only available causal blocks. A partially visible block
    never permits attending to future tokens. Query-aware/agnostic variants are
    outside this initial integration contract.
    """

    block_size: int = 64
    block_budget: int = 64
    sink_blocks: int = 1
    local_blocks: int = 16
    topk_blocks: int = 47


class NosaIndexer:
    """Original NOSA scoring and block top-k will be connected here later."""

    def __init__(self):
        self.policy = NosaSelectionPolicy()

    def __call__(self, q, cache_access: CacheAccess, context: AttentionContext) -> BlockSelection:
        raise NotImplementedError("NOSA block scoring and top-k are not implemented")
