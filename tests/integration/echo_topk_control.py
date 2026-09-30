"""Correctness-only control for ECHO's nondeterministic top-k output order.

Atomic output-position allocation can permute the same selected tokens, changing
FlashMLA's floating-point reduction order. This control only permutes the emitted
multiset into logical-token order. It does not make boundary-tie membership
deterministic, replace the indexer, or establish native-serving determinism.
Never enable it in a performance experiment: its extra operations synchronize.
"""

from __future__ import annotations

import hashlib
import importlib
from contextlib import ExitStack, contextmanager
from pathlib import Path

TEST_CONTROL_ID = "test_only_logical_topk_order_v1"


def canonical_selected_order(
    selected, logical_to_physical, *, capacity: int, fused: bool, torch_module=None
):
    """Gather a permutation of selected, preserving every duplicate and -1 entry."""
    torch = torch_module or importlib.import_module("torch")
    if selected.ndim != 2 or logical_to_physical.ndim != 1:
        raise ValueError("expected [query, topk] selection and one complete request mapping")
    if selected.dtype not in (torch.int32, torch.int64):
        raise ValueError("selection must have an integer dtype")
    length = logical_to_physical.numel()
    if length <= 0 or type(capacity) is not int or capacity <= 0:
        raise ValueError("request length and pool capacity must be positive")
    if logical_to_physical.device != selected.device:
        raise ValueError("selection and request mapping must share a device")
    torch._assert_async(
        ((logical_to_physical >= 1) & (logical_to_physical < capacity)).all(),
        "request mapping contains padding or out-of-pool physical locations",
    )
    if logical_to_physical.unique().numel() != length:
        raise ValueError("request physical locations must be unique")
    limit = capacity if fused else length
    torch._assert_async(
        ((selected >= -1) & (selected < limit)).all(), "selection contains invalid indices"
    )
    valid = selected >= 0
    if fused:
        inverse = torch.full((capacity,), -1, dtype=torch.int64, device=selected.device)
        inverse[logical_to_physical] = torch.arange(
            length, dtype=torch.int64, device=selected.device
        )
        logical = inverse[selected]
    else:
        logical = selected
    torch._assert_async(
        ((~valid) | ((logical >= 0) & (logical < length))).all(),
        "selection refers to another request or an unallocated location",
    )
    order = torch.where(valid, logical, length).argsort(dim=-1, stable=True)
    reordered = selected.gather(-1, order)
    # Validate the control itself, not only the downstream model comparison.
    if not torch.equal(selected.sort(dim=-1).values, reordered.sort(dim=-1).values):
        raise AssertionError("correctness control changed the selected multiset")
    return reordered


@contextmanager
def scoped_logical_topk_order(model_runner):
    """Cover ALL cold-prefill chunks and extends of a serial correctness check.

    Prefer nesting inside open_echo_runner so hooks are removed before teardown.
    ECHO computes prefetch thresholds from logits before indexer.forward returns;
    this post-forward hook neither reads nor writes those prediction thresholds.
    """
    source = importlib.import_module("sglang.srt.layers.attention.nsa_backend")
    pool = model_runner.token_to_kv_pool
    layers = tuple(model_runner.model.model.layers)
    if len(layers) not in (1, 2, 3):
        raise ValueError("test control supports only the first 1-3 layers")
    marker = "_gr_correctness_topk_control"
    if hasattr(model_runner, marker):
        raise RuntimeError("top-k correctness controls cannot be nested")
    state = {
        "id": TEST_CONTROL_ID,
        "calls": 0,
        "preserves_selected_multiset": True,
        "resolves_boundary_ties": False,
        "performance_path": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }

    def hook(module, args, kwargs, selected):
        batch = kwargs.get("forward_batch", args[3] if len(args) > 3 else None)
        if (
            batch is None
            or batch.batch_size != 1
            or not batch.forward_mode.is_extend()
            or batch.token_to_kv_pool is not pool
            or selected is None
        ):
            raise ValueError("test control expects one ordinary EXTEND request")
        req_id = int(batch.req_pool_indices_cpu[0])
        length = int(batch.seq_lens_cpu[0])
        locations = batch.req_to_token_pool.req_to_token[req_id, :length]
        reordered = canonical_selected_order(
            selected,
            locations,
            capacity=pool.size if hasattr(pool, "device_pool") else pool.size + pool.page_size,
            fused=source.NSA_FUSE_TOPK,
        )
        state["calls"] += 1
        return reordered

    with ExitStack() as stack:
        setattr(model_runner, marker, TEST_CONTROL_ID)
        stack.callback(delattr, model_runner, marker)
        for layer in layers:
            handle = layer.self_attn.indexer.register_forward_hook(hook, with_kwargs=True)
            stack.callback(handle.remove)
        yield state
