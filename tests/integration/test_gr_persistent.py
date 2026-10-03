"""Cross-module session eviction and reconstruction against an actual NOSA model."""

import pytest
import torch

from models.nosa.serving import NosaServingBackend
from models.nosa.tests.test_model import tiny_config
from models.nosa.tests.test_sparse_model import initialized_sparse_model
from serving.persistent import PersistentGRRunner


@pytest.mark.parametrize("scheme", NosaServingBackend.schemes)
def test_multi_user_eviction_preserves_repeated_request_outputs(scheme):
    model = initialized_sparse_model(tiny_config(max_position_embeddings=128))
    backend = NosaServingBackend(model, scheme, chunk_size=32)
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
    with PersistentGRRunner(
        backend, hbm_budget_bytes=cost["hbm"], dram_budget_bytes=cost["dram"]
    ) as runner:
        results = list(runner.run(requests))
        assert [result.metrics["prefix_cache_hit"] for result in results] == [False] * 3
        assert results[2].metrics["is_revisit"]
        assert results[2].metrics["evicted_users"] == [1]
        assert len(runner.pool) == 1
        torch.testing.assert_close(results[0].hidden, results[2].hidden, atol=0, rtol=0)
    assert len(runner.pool) == 0
