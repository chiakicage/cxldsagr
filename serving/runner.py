"""Sequential local GR request execution, without network or arrival-time waits."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from executor.model_executor import ModelExecutor


@dataclass(frozen=True)
class RequestResult:
    metadata: dict[str, Any]
    last_hidden: torch.Tensor


def _integer(name: str, value: Any, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _span(name: str, value: Any, total: int) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must contain two token offsets")
    start, end = (_integer(name, offset) for offset in value)
    if not 0 <= start < end <= total:
        raise ValueError(f"{name} is outside the request")
    return start, end


def _validate_request(
    request: Mapping[str, Any], *, max_seq_len: int
) -> tuple[torch.Tensor, int, dict[str, Any]]:
    import torch

    if not isinstance(request, Mapping):
        raise TypeError("request must be a mapping")
    if request.get("model", "nosa") != "nosa":
        raise ValueError("the local GR runner currently supports model='nosa'")
    try:
        ids = torch.as_tensor(request.get("input_ids"))
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("input_ids must be a nonempty 1-D integer sequence") from exc
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if ids.ndim != 1 or not ids.numel() or ids.dtype not in integer_dtypes:
        raise ValueError("input_ids must be a nonempty 1-D integer sequence")
    if bool((ids < 0).any()):
        raise ValueError("input_ids must be nonnegative")
    total = ids.numel()
    if total > max_seq_len:
        raise ValueError(f"request length {total} exceeds model context {max_seq_len}")
    prefix = _integer("stable_prefix_tokens", request.get("stable_prefix_tokens"), 1)
    if prefix >= total:
        raise ValueError("stable_prefix_tokens must leave a nonempty candidate suffix")
    suffix = total - prefix
    lengths = {
        name: _integer(name, request[name])
        for name in (
            "total_input_tokens",
            "candidate_suffix_tokens",
            "instruction_tokens",
            "user_tokens",
            "item_tokens",
            "common_prefix_tokens",
        )
        if name in request
    }
    for name, expected in (("total_input_tokens", total), ("candidate_suffix_tokens", suffix)):
        if name in lengths and lengths[name] != expected:
            raise ValueError(f"{name} is inconsistent with input_ids and stable_prefix_tokens")
    instruction = lengths.get("instruction_tokens")
    user = lengths.get("user_tokens")
    item = lengths.get("item_tokens")
    if "history_token_span" in request:
        history = _span("history_token_span", request["history_token_span"], total)
        if history[1] != prefix:
            raise ValueError("history_token_span must end at stable_prefix_tokens")
        if instruction is not None and instruction != history[0]:
            raise ValueError("instruction_tokens is inconsistent with history_token_span")
        if user is not None and user != history[1] - history[0]:
            raise ValueError("user_tokens is inconsistent with history_token_span")
        instruction, user = history[0], history[1] - history[0]
    if instruction is not None:
        if instruction >= prefix or (user is not None and instruction + user != prefix):
            raise ValueError("instruction_tokens and user_tokens must match stable_prefix_tokens")
        user = prefix - instruction
    if user is not None:
        if not 0 < user <= prefix:
            raise ValueError("user_tokens must fit inside the stable prefix")
        if item is not None and user + item != total:
            raise ValueError("user_tokens + item_tokens must equal total_input_tokens")
    elif item is not None and not suffix <= item < total:
        raise ValueError("item_tokens must include the candidate suffix and leave user history")
    if "candidate_token_span" in request and _span(
        "candidate_token_span", request["candidate_token_span"], total
    ) != (prefix, total):
        raise ValueError("candidate_token_span must cover the candidate suffix")
    if lengths.get("common_prefix_tokens", 0) > total:
        raise ValueError("common_prefix_tokens exceeds request length")
    if "attention_mask" in request:
        try:
            mask = torch.as_tensor(request["attention_mask"])
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("attention_mask must be an all-ones 1-D sequence") from exc
        if mask.ndim != 1 or mask.numel() != total or not bool((mask == 1).all()):
            raise ValueError("attention_mask must be all ones and match input_ids")
    metadata = {
        key: value
        for key, value in request.items()
        if key not in ("input_ids", "attention_mask", "prompt")
    }
    metadata.update(
        model="nosa",
        total_input_tokens=total,
        stable_prefix_tokens=prefix,
        candidate_suffix_tokens=suffix,
    )
    return ids, prefix, metadata


class GRRunner:
    """Run one independent request at a time and yield its final backbone feature.

    The stable prefix contains instruction and user history; the remaining
    candidate suffix is extended in the same session. Metadata timestamps are
    preserved without sleeping. A session is released before yielding a result
    or propagating a model error, so pausing iteration retains no live session.
    """

    def __init__(self, executor: ModelExecutor, *, device: str | torch.device = "cuda:0"):
        self.executor = executor
        self.device = device

    def run(self, requests: Iterable[Mapping[str, Any]]) -> Iterator[RequestResult]:
        import torch

        for request in requests:
            ids, prefix, metadata = _validate_request(
                request, max_seq_len=self.executor.max_seq_len
            )
            ids = ids.to(device=self.device, dtype=torch.long)
            cache = self.executor.allocate(ids.numel())
            try:
                self.executor.prefill(ids[:prefix], cache)
                hidden = self.executor.extend(ids[prefix:], cache)
                # A view would retain every feature in the final chunk.
                with torch.inference_mode():
                    result = RequestResult(metadata=metadata, last_hidden=hidden[-1].clone())
                del hidden
            finally:
                self.executor.release(cache)
            del cache, ids
            yield result
