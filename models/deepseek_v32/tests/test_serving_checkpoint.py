"""Opt-in real checkpoint validation; no timing or persistent result artifacts."""

import os

import pytest
import torch

from models.deepseek_v32.serving_backend import SCHEMES, DeepSeekServingBackend


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_SERVING_CHECKPOINT"),
    reason="set DEEPSEEK_SERVING_CHECKPOINT for the single-GPU checkpoint check",
)
def test_checkpoint_all_policies_copied_inputs_revisits_and_cache_reservations():
    # Explicit checkpoint selection must fail if the GPU/dependencies are absent.
    model = DeepSeekServingBackend(
        os.environ["DEEPSEEK_SERVING_CHECKPOINT"],
        device="cuda:0",
        num_layers=10,
        chunk_size=256,
        slots=2048,
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
    for scheme in SCHEMES:
        # Policy belongs to the session; all old sessions are released first.
        model.scheme = scheme
        session = model.create_session(len(prefix) + max(map(len, candidates)))
        budget = model.estimate_session_bytes(session.capacity, len(prefix))

        def check_allocation(session=session, budget=budget):
            actual = model.session_bytes(session)
            assert all(actual[tier] <= budget[tier] for tier in actual)

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
            for visit, candidate in enumerate(candidates):
                originals.clear()
                hidden = model.extend(session, candidate).cpu()
                if scheme == "hbm":
                    reference.append(hidden)
                else:
                    torch.testing.assert_close(hidden, reference[visit], rtol=0, atol=0)
                check_allocation()
                model.truncate(session, len(prefix))
                assert session.length == len(prefix)
        finally:
            model.capture_hook = None
            model.release_session(session)
