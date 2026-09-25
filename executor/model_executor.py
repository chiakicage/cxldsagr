"""Model-neutral, single-request chunked inference and cache ownership."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import torch

Output = Literal["hidden", "logits"]


def _positive_chunk_size(chunk_size: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")


def run_chunks(
    model: Any,
    ids: torch.Tensor,
    cache: Any,
    chunk_size: int,
    *,
    output: Output = "hidden",
    logits_to_keep: int = 0,
) -> torch.Tensor:
    """Consume all tokens and return only the final chunk's output.

    Hidden output includes every token in the final chunk and skips the LM head.
    Logits use the model's existing ``logits_to_keep`` option per chunk. This
    function also supports a populated cache; prefill/extend state constraints
    are enforced by :class:`ModelExecutor`.
    """
    import torch

    _positive_chunk_size(chunk_size)
    if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or not ids.numel():
        raise ValueError("ids must be a nonempty 1-D torch.long tensor")
    if ids.dtype != torch.long:
        raise ValueError("ids must be a nonempty 1-D torch.long tensor")
    if output not in ("hidden", "logits"):
        raise ValueError("output must be 'hidden' or 'logits'")
    if (
        isinstance(logits_to_keep, bool)
        or not isinstance(logits_to_keep, int)
        or logits_to_keep < 0
    ):
        raise ValueError("logits_to_keep must be a nonnegative integer")
    if output == "hidden" and logits_to_keep:
        raise ValueError(
            "hidden output returns all final-chunk features; do not set logits_to_keep"
        )
    end = cache.length + ids.numel()
    if end > cache.max_seq_len:
        raise ValueError("KV cache capacity exceeded by the complete input")
    if end > model.config.max_position_embeddings:
        raise ValueError("Input exceeds model.config.max_position_embeddings")
    with torch.inference_mode():
        for offset in range(0, ids.numel(), chunk_size):
            chunk = ids[offset : offset + chunk_size]
            if output == "hidden":
                result = model(chunk, cache, return_hidden=True)
            else:
                result = model(chunk, cache, logits_to_keep=logits_to_keep)
    return result


class ModelExecutor:
    """Execute token tensors against a model and a request-level cache manager.

    Models expose ``config.max_position_embeddings`` and their normal forward
    callable. Legacy callers providing only ``new_cache`` remain supported;
    real backends supply a manager with ``allocate`` and ``release`` methods.
    No request dictionary, tokenizer or scheduling policy enters this layer.
    """

    def __init__(self, model: Any, *, cache_manager: Any = None, chunk_size: int = 1024):
        _positive_chunk_size(chunk_size)
        self.model = model
        self.cache_manager = (
            cache_manager if cache_manager is not None else getattr(model, "cache_manager", None)
        )
        self.chunk_size = chunk_size

    @property
    def max_seq_len(self) -> int:
        return self.model.config.max_position_embeddings

    def allocate(self, max_seq_len: int) -> Any:
        if self.cache_manager is not None:
            return self.cache_manager.allocate(max_seq_len)
        return self.model.new_cache(max_seq_len)

    def release(self, cache: Any) -> None:
        if self.cache_manager is not None:
            self.cache_manager.release(cache)

    def prefill(
        self,
        ids: torch.Tensor,
        cache: Any,
        *,
        output: Output = "hidden",
        logits_to_keep: int = 0,
    ) -> torch.Tensor:
        if cache.length != 0:
            raise ValueError("prefill requires an empty cache")
        return run_chunks(
            self.model, ids, cache, self.chunk_size, output=output, logits_to_keep=logits_to_keep
        )

    def extend(
        self,
        ids: torch.Tensor,
        cache: Any,
        *,
        output: Output = "hidden",
        logits_to_keep: int = 0,
    ) -> torch.Tensor:
        """Append one or more tokens, including single-token decode."""
        if cache.length <= 0:
            raise ValueError("extend requires a nonempty cache")
        return run_chunks(
            self.model, ids, cache, self.chunk_size, output=output, logits_to_keep=logits_to_keep
        )
