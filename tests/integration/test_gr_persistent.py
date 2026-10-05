"""Cross-module session eviction and reconstruction against an actual NOSA model."""

import pytest
import torch

from cache.prefix_pool import CacheFootprint
from models.nosa.execution.adapter import NosaServingBackend
from models.nosa.tests.test_model import tiny_config
from models.nosa.tests.test_sparse_model import initialized_sparse_model
from serving.persistent import PersistentGRRunner


@pytest.mark.parametrize("scheme", NosaServingBackend.schemes)
def test_multi_user_eviction_preserves_repeated_request_outputs(scheme):
    model = initialized_sparse_model(tiny_config(max_position_embeddings=128))
    backend = NosaServingBackend(model, scheme, chunk_size=32)
    limits = {"max_session_capacity": 96, "max_candidate_tokens": 16}
    plan = backend.plan_resources(CacheFootprint(1 << 30, 1 << 30), limits)
    backend.allocate_shared(plan)
    cost = backend.estimate_session_bytes(96, 80)
    # Exactly one session fits: sequence A, B, A must reconstruct A's prefix.
    requests = [
        {
            "user_id": uid,
            "input_ids": [(token + uid * 7) % 43 for token in range(96)],
            "stable_prefix_tokens": 80,
        }
        for uid in (0, 1, 0)
    ]
    try:
        with PersistentGRRunner(
            backend,
            hbm_budget_bytes=plan.shared.hbm + cost["hbm"],
            dram_budget_bytes=plan.shared.dram + cost["dram"],
            resource_limits=limits,
        ) as runner:
            results = list(runner.run(requests))
            assert [result.metrics["prefix_cache_hit"] for result in results] == [False] * 3
            assert results[2].metrics["is_revisit"]
            assert results[2].metrics["evicted_users"] == [1]
            assert len(runner.pool) == 1
            torch.testing.assert_close(results[0].hidden, results[2].hidden, atol=0, rtol=0)
        assert len(runner.pool) == 0
        assert backend.shared_bytes() == {"hbm": plan.shared.hbm, "dram": plan.shared.dram}
    finally:
        backend.close()
    assert backend.shared_bytes() == {"hbm": 0, "dram": 0}
