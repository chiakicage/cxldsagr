"""Observe cache allocation peaks at allocation time, separate from activations."""

import pytest
import torch

from cache.indexer_cache import IndexerBufferSpec, IndexerCache
from models.nosa.tests.test_serving_resources import BUDGET, SCHEMES, make_backend, storages


@pytest.mark.parametrize("scheme", SCHEMES)
def test_workspace_growth_peak_keeps_old_and_new_storage_within_session_reservation(
    monkeypatch, scheme
):
    capacity, candidate, prefix = 66560, 1024, 65536
    backend = make_backend(scheme, chunk_size=512, context=capacity)
    plan = backend.plan_resources(
        BUDGET, {"max_session_capacity": capacity, "max_candidate_tokens": candidate}
    )
    backend.allocate_shared(plan)
    session = backend.create_session(capacity)
    reservation = backend.estimate_session_bytes(capacity, prefix)
    workspace = session.indexer_cache
    shared = backend.shared_bytes()
    shared_storages = storages(backend.resources, stop=(backend, backend.model))
    observations = []
    original_empty = torch.empty

    def observe_allocation(*args, **kwargs):
        result = original_empty(*args, **kwargs)
        if result.dtype == torch.uint8 and result.ndim == 1:
            # torch.empty has returned, but IndexerCache.workspace has not yet
            # assigned the replacement. The old owning tensor is still live.
            old = workspace._workspace
            assert old is not None
            current = backend.session_bytes(session)
            current_storages = storages(session, stop=(backend, backend.model, backend.resources))
            assert sum(current.values()) == sum(current_storages.values())
            assert set(current_storages).isdisjoint(shared_storages)
            new_storage = result.untyped_storage()
            old_storage = old.untyped_storage()
            assert new_storage.data_ptr() != old_storage.data_ptr()
            assert (str(old.device), old_storage.data_ptr()) in current_storages
            peak_hbm = current["hbm"] + new_storage.nbytes()
            assert peak_hbm <= reservation["hbm"]
            assert current["dram"] <= reservation["dram"]
            assert shared["hbm"] + peak_hbm <= plan.shared.hbm + reservation["hbm"]
            observations.append((old_storage.nbytes(), new_storage.nbytes(), peak_hbm))
        return result

    try:
        workspace.workspace(1, torch.uint8)
        # The checked native indexer requests scores followed by an aligned
        # 3328-byte validation region and head-wise ranking. Exercise both a
        # prefix-chunk request and the larger allowed candidate request. This
        # directly tests storage lifetimes; it does not run a CPU attention
        # approximation and call that native execution.
        heads = backend.config.num_key_value_heads
        blocks = (capacity + 63) // 64
        for queries in (backend.chunk_size, candidate):
            score_bytes = queries * heads * blocks * backend.dtype.itemsize
            requested = (score_bytes + 255) // 256 * 256 + 3328 + heads * 64 * 4
            with monkeypatch.context() as patch:
                patch.setattr(torch, "empty", observe_allocation)
                view = workspace.workspace(requested, torch.uint8)
            assert view.numel() == requested
            del view
        assert len(observations) == 2
        assert observations[1][0] == observations[0][1]
        assert observations[1][1] > observations[1][0] > 256
        # Post-allocation stats see only the new slab and therefore cannot
        # themselves prove the peak assertion above.
        assert observations[-1][2] > backend.session_bytes(session)["hbm"]
        assert backend.shared_bytes() == shared
        assert storages(backend.resources, stop=(backend, backend.model)) == shared_storages
    finally:
        backend.release_session(session)
        backend.close()


def test_cuda_isolated_indexer_allocator_peak_includes_both_live_slabs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; use scripts/run_tests.sh gpu to require CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.fail("NOSA allocation audit requires SM90/Hopper")
    cache = IndexerCache(
        1, 8192, {"unused": IndexerBufferSpec(0, (1,), torch.bfloat16)}, device="cuda"
    )
    try:
        cache.workspace(1 << 20, torch.uint8)
        torch.cuda.synchronize()
        old_bytes = cache._workspace.untyped_storage().nbytes()
        old_pointer = cache._workspace.data_ptr()
        before = torch.cuda.memory_allocated()
        before_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        cache.workspace(3 << 20, torch.uint8)
        torch.cuda.synchronize()
        new_bytes = cache._workspace.untyped_storage().nbytes()
        after = torch.cuda.memory_allocated()
        allocated_peak = torch.cuda.max_memory_allocated()
        reserved_peak = torch.cuda.max_memory_reserved()
        assert cache._workspace.data_ptr() != old_pointer
        assert allocated_peak >= before + new_bytes
        assert after == before - old_bytes + new_bytes
        assert allocated_peak > after
        assert reserved_peak >= max(before_reserved, allocated_peak)
        assert cache.stats()["resident_bytes"] == new_bytes
        # The fixture allocates only byte scratch: no model weights, outputs or
        # activations are included in these incremental allocator measurements.
    finally:
        cache.release()
