"""Independent full-extend graph checks, outside benchmark samples."""

from __future__ import annotations

import hashlib

import torch

from cache.sparse_token_cache import MISSING


def tensor_identity(tensor):
    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def cache_state(model):
    """Audit maps and compare logical records without requiring stable tied slots."""
    model.synchronize()
    layers = []
    for block in model.blocks:
        cache, runner = block.cache, block.attention
        if cache.length != model.length or cache.written != model.length:
            raise AssertionError("full graph did not commit every layer")
        if cache._step_end is not None:
            raise AssertionError("full graph left an active cache transaction")
        row = {
            "length": cache.length,
            "written": cache.written,
            "indexer_visible_end": cache.indexer_visible_end,
            "records": tensor_identity(cache.host_records()),
            "index_keys": tensor_identity(runner.index_keys[: model.length]),
            "index_scales": tensor_identity(runner.index_scales[: model.length]),
            "hint": tensor_identity(runner.offset),
            "metrics": cache.metrics(),
        }
        if cache.offload:
            pool_layer = cache._pool.layers[cache.layer_id]
            ids = cache.logical_to_global(
                torch.arange(model.length, dtype=torch.int64, device=cache.device)
            ).cpu()
            forward = cache.host_to_device.detach().cpu()
            reverse = cache.device_to_host.detach().cpu()
            free = pool_layer.free.detach().cpu()
            priority = pool_layer.priority.detach().cpu()
            slots = forward[ids].long()
            resident = slots != MISSING
            used = (reverse != MISSING).nonzero().flatten()
            used = used[used != 0]
            if used.numel() and (
                not torch.equal(forward[reverse[used].long()].long(), used)
                or bool(free[used].any())
            ):
                raise AssertionError("full graph left inconsistent reverse/free maps")
            if resident.any():
                selected = slots[resident]
                if not torch.equal(reverse[selected].long(), ids[resident]):
                    raise AssertionError("full graph left inconsistent forward maps")
                records = cache.records.detach().cpu()[selected]
                host = cache.host.detach().cpu()[ids[resident]]
                if not torch.equal(records, host):
                    raise AssertionError("resident KV differs from logical host backing")
            if bool(free[0]) or int(free[1:].sum()) + used.numel() != cache.slots:
                raise AssertionError("free bitmap does not partition the slot pool")
            logical_priority = torch.zeros(model.length, dtype=priority.dtype)
            logical_priority[resident] = priority[slots[resident]]
            row.update(
                resident=tensor_identity(resident),
                logical_priority=tensor_identity(logical_priority),
                free_count=int(free.sum()),
                clock=tensor_identity(pool_layer.clock_tensor),
                map_invariants_passed=True,
            )
        layers.append(row)
    return {"length": model.length, "layers": layers}


def compare_cache_state(actual, expected):
    if actual != expected:
        changed = [
            f"layer_{layer}.{key}"
            for layer, (left, right) in enumerate(
                zip(actual["layers"], expected["layers"], strict=True)
            )
            for key in left.keys() | right.keys()
            if left.get(key) != right.get(key)
        ]
        raise AssertionError(f"full graph cache state differs: {changed}")
    return {"equal": True, "logical_slot_order": True, "layers": len(actual["layers"])}
