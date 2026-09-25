"""Check that module scopes follow the refactored runtime call sites."""

from contextlib import ExitStack
from dataclasses import asdict

import pytest
import torch

from experiments.nosa_gr_65536_1024.src.instrumentation import ModuleScopes
from experiments.nosa_gr_65536_1024.src.mfu import matrix_flops


def test_module_scopes_follow_moved_rotary_and_attention_calls():
    from models.nosa.tests.test_model import initialized_model, tiny_config

    config = tiny_config()
    model = initialized_model(config)
    tokens = torch.tensor([1, 7, 6, 3])
    expected = model(tokens, return_hidden=True)
    cache = model.new_cache(len(tokens))
    scopes = ModuleScopes()
    try:
        with ExitStack() as stack:
            scopes.install(model, stack)
            hidden = model(tokens, cache, return_hidden=True)
        assert hidden.shape == (len(tokens), config.hidden_size)
        torch.testing.assert_close(hidden, expected, rtol=0, atol=0)
        assert scopes.calls["prefill/rope_prepare/shared"] == 1
        assert scopes.calls["prefill/rope_apply/shared"] == config.num_hidden_layers
        assert scopes.calls["prefill/attention_core/shared"] == config.num_hidden_layers
        for layer in range(config.num_hidden_layers):
            assert scopes.calls[f"prefill/qkv_proj/{layer}"] == 1
            assert scopes.calls[f"prefill/gate_up_proj/{layer}"] == 1
            assert scopes.calls[f"prefill/decoder/{layer}"] == 1
            assert scopes.calls[f"prefill/swiglu_elementwise/{layer}"] == 1
            assert scopes.calls[f"prefill/post_attention_layernorm_add_residual/{layer}"] == 1
            input_category = "input_layernorm" + ("_add_residual" if layer else "")
            assert scopes.calls[f"prefill/{input_category}/{layer}"] == 1
        assert scopes.calls["prefill/final_norm_add_residual/shared"] == 1
        assert not any("/residual/" in key for key in scopes.calls)
        assert not any(
            key.split("/")[1] in {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
            for key in scopes.calls
        )
        # Compound scopes describe the execution boundary even on the unfused
        # CPU reference. Actual GPU fusion must be checked in kernel_names.
        assert model.model.norm.weight.device.type == "cpu"
        expected_flops = matrix_flops(asdict(config), prefix=0, query=len(tokens))
        assert sum(scopes.flops.values()) == sum(expected_flops.values())
        for category, total in expected_flops.items():
            assert (
                sum(count for key, count in scopes.flops.items() if key.split("/")[1] == category)
                == total
            )
        calls = scopes.calls.copy()
        cache.reset()
        model(tokens, cache, return_hidden=True)
        assert scopes.calls == calls  # Instrumentation was fully removed by ExitStack.
    finally:
        model.cache_manager.release(cache)


@pytest.mark.parametrize("keyword_residual", [False, True])
def test_norm_scope_accepts_positional_and_keyword_residual(keyword_residual):
    from models.nosa.tests.test_model import initialized_model

    model = initialized_model()
    norm = model.model.layers[0].input_layernorm
    x = torch.ones(3, model.config.hidden_size)
    residual = torch.zeros_like(x)
    scopes = ModuleScopes()
    with ExitStack() as stack:
        scopes.install(model, stack)
        norm(x)
        if keyword_residual:
            norm(x, residual=residual)
        else:
            norm(x, residual)
    assert scopes.calls["prefill/input_layernorm/0"] == 1
    assert scopes.calls["prefill/input_layernorm_add_residual/0"] == 1
    assert not scopes.flops  # Norm/residual work never contributes matrix FLOPs.


@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("category", ["qkv_proj", "gate_up_proj"])
def test_combined_projection_has_one_scope_and_combined_flops(use_bias, category):
    from models.nosa.tests.test_model import initialized_model, tiny_config

    config = tiny_config(attention_bias=use_bias, mlp_bias=use_bias)
    model = initialized_model(config)
    layer = model.model.layers[0]
    projection = layer.self_attn.qkv_proj if category == "qkv_proj" else layer.mlp.gate_up_proj
    output_width = (
        (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim
        if category == "qkv_proj"
        else 2 * config.intermediate_size
    )
    x = torch.ones(3, config.hidden_size)
    expected = projection(x)
    scopes = ModuleScopes()
    scopes.phase = "extend"
    with ExitStack() as stack:
        scopes.install(model, stack)
        actual = projection(x)
    torch.testing.assert_close(actual, expected)
    key = f"extend/{category}/0"
    assert actual.shape == (len(x), output_width)
    assert scopes.calls == {key: 1}
    assert scopes.flops == {key: 2 * len(x) * config.hidden_size * output_width}
