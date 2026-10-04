"""Native private recall retains checked maps, FIFO ordering and failure safety."""

import subprocess
import sys

import pytest
import torch

from cache.sparse_token_pool import MISSING, PRIORITY_LIMIT
from cache.tests.test_resident_metadata import _equal, _pools
from operators.common import kv_transfer
from operators.deepseek_v32.indexer import cache_ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _append(cache, data):
    cache.begin_step(len(data))
    for piece in data.split(cache.slots):
        cache.append(piece)
    cache.commit()


def _proofs_equal(pools, caches):
    _equal(pools, caches)
    for layer_id in range(2):
        states = [pool.layers[layer_id] for pool in pools]
        for field in (
            "map_generation",
            "resident_owner",
            "resident_end",
            "append_owner",
            "append_cursor",
            "lease_owner",
        ):
            assert getattr(states[0], field) == getattr(states[1], field), field
        if states[0].lease_owner is not None:
            torch.testing.assert_close(
                states[0].append_order, states[1].append_order, rtol=0, atol=0
            )


def _pair(pools, *, history=137, competitor=150):
    pairs = []
    for pool in pools:
        allocations = [pool.allocate_session(64) for _ in range(4)]
        allocations[0].release()
        allocations[2].release()
        session = pool.allocate_session(history)
        if history > 64:
            assert bool((session._pages[1:] < session._pages[:-1]).any())
        a = session.layer(0)
        b = pool.allocate_session(competitor).layer(0)
        xa = (torch.arange(history * pool.width).reshape(history, pool.width) % 197).to(
            device="cuda", dtype=pool.dtype
        )
        xb = (torch.arange(competitor * pool.width).reshape(competitor, pool.width) % 197 + 1).to(
            device="cuda", dtype=pool.dtype
        )
        _append(a, xa)
        _append(b, xb)
        pairs.append((a, b))
    return pairs


@pytest.mark.parametrize(
    "dtype,width", [(torch.uint8, 7), (torch.float32, 7), (torch.bfloat16, 576)]
)
@pytest.mark.parametrize("history", [137, 193])
def test_cuda_sparse_recall_partial_free_fragmented_pages_and_candidate_tail(dtype, width, history):
    pools = _pools(width=width, dtype=dtype)
    try:
        pairs = _pair(pools, history=history)
        for a, b in pairs:
            b.truncate(80)
            assert not a.all_history_resident
            a.reset_stats()
            a.begin_transient(35)
            a.append(torch.ones(35, width, device="cuda", dtype=dtype))
        caches = [pair[0] for pair in pairs]
        ids = torch.arange(history + 35, device="cuda", dtype=torch.int32).repeat(7, 1)
        ids[:, 0] = -5
        outputs = [cache._ensure_from_topk(ids) for cache in caches]
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        _proofs_equal(pools, caches)
        assert caches[0].metrics()["recalled_records"] > 0
        assert caches[0].metrics()["evicted_records"] > 0
        for cache, output in zip(caches, outputs, strict=True):
            torch.testing.assert_close(
                output[:, history:], ids[:, history:] - history + 194, rtol=0, atol=0
            )
            cache.discard_transient()
        _proofs_equal(pools, caches)
        # Alternate owners repeatedly, including a no-miss uncertified revisit.
        for owner in (1, 0, 0, 1, 0):
            caches = [pair[owner] for pair in pairs]
            ids = torch.randint(-3, caches[0].written, (23, 97), device="cuda", dtype=torch.int32)
            outputs = [cache._ensure_from_topk(ids) for cache in caches]
            torch.testing.assert_close(*outputs, rtol=0, atol=0)
            _proofs_equal(pools, caches)
    finally:
        for pool in pools:
            pool.close()


@pytest.mark.parametrize("clock", [None, PRIORITY_LIMIT - 2, PRIORITY_LIMIT - 1, PRIORITY_LIMIT])
def test_cuda_uncertified_all_hit_empty_padding_and_rollover_preserve_proofs(clock):
    pools = _pools(slots=8)
    try:
        caches = []
        for pool in pools:
            cache = pool.allocate_session(8).layer(0)
            _append(cache, torch.ones(8, 7, device="cuda"))
            pool.layers[0].resident_owner = None
            if clock is not None:
                pool.layers[0].clock = clock
            caches.append(cache)
        for ids in (
            torch.tensor([[0, 0, -1, 7]], device="cuda", dtype=torch.int32),
            torch.empty(0, 19, device="cuda", dtype=torch.int32),
            torch.full((7, 11), -3, device="cuda", dtype=torch.int32),
        ):
            before = [pool.layers[0].map_generation for pool in pools]
            outputs = [cache._ensure_from_topk(ids) for cache in caches]
            torch.testing.assert_close(*outputs, rtol=0, atol=0)
            _proofs_equal(pools, caches)
            assert before == [pool.layers[0].map_generation for pool in pools]
            assert all(
                pool.layers[0].append_owner == cache.session.owner
                for pool, cache in zip(pools, caches, strict=True)
            )
    finally:
        for pool in pools:
            pool.close()


def test_cuda_sparse_recall_pending_host_writes_and_stream_change():
    pools = _pools(width=576, dtype=torch.bfloat16, slots=193)
    try:
        pairs = _pair(pools)
        stream = torch.cuda.Stream()
        for owner in (0, 1, 0):
            caches = [pair[owner] for pair in pairs]
            ids = torch.randint(
                -1, caches[0].written, (128, 2048), device="cuda", dtype=torch.int32
            )
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                outputs = [cache._ensure_from_topk(ids) for cache in caches]
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(*outputs, rtol=0, atol=0)
            _proofs_equal(pools, caches)
    finally:
        for pool in pools:
            pool.close()


@pytest.mark.parametrize("stage", ["wait", "sort", "copy", "publish", "map"])
def test_cuda_sparse_recall_failure_matches_checked_state_and_other_user(monkeypatch, stage):
    pools = _pools(slots=8)
    try:
        pairs = []
        for pool in pools:
            a = pool.allocate_session(8).layer(0)
            b = pool.allocate_session(8).layer(0)
            _append(a, torch.ones(8, 7, device="cuda"))
            _append(b, torch.full((8, 7), 2.0, device="cuda"))
            pairs.append((a, b))
        original_clock = pools[0].layers[0].clock
        ids = torch.arange(4, device="cuda", dtype=torch.int32)[None]
        for index, (pool, (a, _)) in enumerate(zip(pools, pairs, strict=True)):
            if stage == "wait":
                target, name = pool, "wait_host"
            elif stage == "sort":
                target, name = torch, "argsort"
            elif stage in ("copy", "publish"):
                target, name = (
                    (cache_ops, "sparse_selection_publish")
                    if stage == "publish" and index == 0
                    else (kv_transfer, "gather_host_records")
                )
            else:
                target, name = (
                    (cache_ops, "sparse_selection_map") if index == 0 else (pool, "stamp")
                )
            original = getattr(target, name)

            def fail(*args, _name=name, _index=index, _original=original, **kwargs):
                if _name == "gather_host_records" or (stage == "map" and _index == 1):
                    # At these corresponding checked/native boundaries, record
                    # overwrite or the allocation event has already completed.
                    _original(*args, **kwargs)
                raise RuntimeError(f"injected {stage} failure")

            with monkeypatch.context() as patch:
                patch.setattr(target, name, fail)
                with pytest.raises(RuntimeError, match="injected"):
                    a._ensure_from_topk(ids)
            pool.drain()
            layer = pool.layers[0]
            assert layer.clock == original_clock + (2 if stage == "map" else 1)
            assert int(layer.clock_tensor) == layer.clock
            live = torch.where(layer.device_to_host != MISSING)[0]
            global_ids = layer.device_to_host[live]
            torch.testing.assert_close(layer.host_to_device[global_ids].long(), live)
            torch.testing.assert_close(layer.records[live], layer.host[global_ids.cpu()].cuda())
            assert bool((layer.priority[live] < layer.clock).all())
            assert a.metrics()["recalled_records"] == (4 if stage == "map" else 0)
        _proofs_equal(pools, [pair[0] for pair in pairs])
        for pool, (a, b) in zip(pools, pairs, strict=True):
            a.session.release()
            output = b._ensure_from_topk(torch.arange(8, device="cuda", dtype=torch.int32)[None])
            torch.testing.assert_close(
                b.records[output.long()], torch.full((1, 8, 7), 2.0, device="cuda")
            )
        _proofs_equal(pools, [pair[1] for pair in pairs])
    finally:
        for pool in pools:
            pool.close()


def test_cuda_invalid_private_selection_traps_in_isolated_process():
    code = r"""
import torch
from cache.sparse_token_pool import SharedSparseTokenPool
from operators.deepseek_v32.indexer import cache_ops
pool = SharedSparseTokenPool(128, 1, 1, 8, device="cuda", dtype=torch.float32, metadata_ops=cache_ops)
a, b = [pool.allocate_session(8).layer(0) for _ in range(2)]
for cache in (a,b):
    cache.begin_step(8); cache.append(torch.ones(8,1,device="cuda")); cache.commit()
a._ensure_from_topk(torch.tensor([[8]],device="cuda",dtype=torch.int32))
torch.cuda.synchronize()
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode != 0
    assert "CUDA" in result.stderr and ("launch" in result.stderr or "illegal" in result.stderr)
