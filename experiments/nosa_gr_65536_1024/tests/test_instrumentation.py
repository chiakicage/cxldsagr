"""Check that module scopes follow the refactored runtime call sites."""

from contextlib import ExitStack

import torch

from experiments.nosa_gr_65536_1024.src.instrumentation import ModuleScopes


def test_module_scopes_follow_moved_rotary_and_attention_calls():
    from models.nosa.tests.test_model import initialized_model, tiny_config

    config = tiny_config()
    model = initialized_model(config)
    tokens = torch.tensor([1, 7, 6, 3])
    cache = model.new_cache(len(tokens))
    scopes = ModuleScopes()
    try:
        with ExitStack() as stack:
            scopes.install(model, stack)
            hidden = model(tokens, cache, return_hidden=True)
        assert hidden.shape == (len(tokens), config.hidden_size)
        assert scopes.calls["prefill/rope_prepare/shared"] == 1
        assert scopes.calls["prefill/rope_apply/shared"] == 2 * config.num_hidden_layers
        assert scopes.calls["prefill/attention_core/shared"] == config.num_hidden_layers
        for layer in range(config.num_hidden_layers):
            assert scopes.calls[f"prefill/q_proj/{layer}"] == 1
        calls = scopes.calls.copy()
        cache.reset()
        model(tokens, cache, return_hidden=True)
        assert scopes.calls == calls  # Instrumentation was fully removed by ExitStack.
    finally:
        model.cache_manager.release(cache)
