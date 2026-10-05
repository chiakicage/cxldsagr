"""Opt-in real checkpoint validation; no timing or persistent result artifacts."""

import os

import pytest
import torch

from cache.prefix_pool import CacheFootprint
from models.deepseek_v32.execution.adapter import SCHEMES, DeepSeekServingBackend
from models.deepseek_v32.execution.cache_resources import padded_tokens


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_SERVING_CHECKPOINT"),
    reason="set DEEPSEEK_SERVING_CHECKPOINT for the single-GPU checkpoint check",
)
@pytest.mark.parametrize("fixed_pools", [False, True], ids=["budget", "fixed_pools"])
def test_checkpoint_all_policies_copied_inputs_revisits_and_cache_reservations(fixed_pools):
    # Explicit checkpoint selection must fail if the GPU/dependencies are absent.
    model = DeepSeekServingBackend(
        os.environ["DEEPSEEK_SERVING_CHECKPOINT"],
        device="cuda:0",
        num_layers=10,
        chunk_size=256,
        sparse_pool_tokens=2304 if fixed_pools else 2048,
        host_arena_tokens=2 * padded_tokens(2304),
    )
    for layer in range(3, 10):
        assert (
            model.attentions[layer].wq_a.weight.data_ptr()
            != model.attentions[layer % 3].wq_a.weight.data_ptr()
        )
        assert (
            model.blocks[layer].mlp.up.weight.data_ptr()
            != model.blocks[layer % 3].mlp.up.weight.data_ptr()
        )
    prefix = [1 + ((token * 137) % 100000) for token in range(2304)]
    candidates = [[2 + token * 53 for token in range(16)], [3 + token * 79 for token in range(23)]]
    reference = []
    reference_logits = []
    for scheme in SCHEMES:
        # Policy belongs to the session; all old sessions are released first.
        model.configure_scheme(scheme)
        plan = model.plan_resources(
            None if fixed_pools else CacheFootprint(2**30, 2**30),
            {
                "max_session_capacity": len(prefix) + 23,
                "max_history_tokens": len(prefix),
                "max_candidate_tokens": 23,
            },
        )
        model.allocate_shared(plan)
        shared_staging = (
            model._dense_staging.storage_tensors()
            if scheme == "dense_prefetch" and not fixed_pools
            else ()
        )
        transient = fixed_pools or scheme in ("echo", "serial_sparse")
        session = model.create_session(
            len(prefix) if transient else len(prefix) + max(map(len, candidates))
        )
        budget = model.estimate_session_bytes(session.capacity, len(prefix))

        def check_allocation(
            session=session, budget=budget, plan=plan, shared_staging=shared_staging
        ):
            actual = model.session_bytes(session)
            assert all(actual[tier] <= budget[tier] for tier in actual)
            assert CacheFootprint.from_mapping(model.shared_bytes()).fits(plan.shared)
            if shared_staging:
                assert all(runner.cache.records is None for runner in session.runners)
                assert not session.stages
                assert all(
                    actual is original
                    for actual, original in zip(
                        model._dense_staging.storage_tensors(), shared_staging, strict=True
                    )
                )

        originals = {}

        def check_copy(layer, hidden, residual, originals=originals):
            if layer < 3:
                originals[layer] = (hidden.clone(), residual.clone())
            else:
                torch.testing.assert_close((hidden, residual), originals[layer % 3], rtol=0, atol=0)

        model.capture_hook = check_copy
        try:
            check_allocation()
            model.prefill(session, prefix)
            check_allocation()
            history_keys = [runner.index_keys.clone() for runner in session.runners]
            history_scales = [runner.index_scales.clone() for runner in session.runners]
            host_history = (
                [runner.cache.host_records() for runner in session.runners] if transient else []
            )
            # Force another user with identical local IDs through the same layer
            # pools, then revisit the original history after global eviction.
            other = model.create_session(session.capacity)
            try:
                if shared_staging:
                    assert all(
                        left.cache.host.data_ptr() != right.cache.host.data_ptr()
                        for left, right in zip(session.runners, other.runners, strict=True)
                    )
                model.prefill(other, [1 + (token + 31) % 100000 for token in prefix])
            finally:
                model.release_session(other)
            for visit, candidate in enumerate(candidates):
                originals.clear()
                hidden = model.extend_candidate(session, candidate).cpu()
                logits = model.last_logits.cpu()
                if scheme == "hbm":
                    reference.append(hidden)
                    reference_logits.append(logits)
                else:
                    torch.testing.assert_close(hidden, reference[visit], rtol=0, atol=0)
                    torch.testing.assert_close(logits, reference_logits[visit], rtol=0, atol=0)
                check_allocation()
                if transient:
                    assert session.length == len(prefix)
                    metrics = model.session_metrics(session)
                    assert metrics["candidate_persistence"] == "gpu_transient"
                    assert metrics["candidate_device_to_host_bytes"] == 0
                    for layer, runner in enumerate(session.runners):
                        assert torch.equal(
                            runner.index_keys.view(torch.uint8),
                            history_keys[layer].view(torch.uint8),
                        )
                        assert torch.equal(runner.index_scales, history_scales[layer])
                        assert torch.equal(runner.cache.host_records(), host_history[layer])
                    if model._shared_pool is not None:
                        assert not model._shared_pool._transient_owners
                model.truncate(session, len(prefix))
                assert session.length == len(prefix)
        finally:
            model.capture_hook = None
            model.release_session(session)

    model.close()
