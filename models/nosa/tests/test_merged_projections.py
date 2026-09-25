"""Merged GEMM ownership and strict loading of NOSA split checkpoint tensors."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.nosa import layers as nosa_layers
from models.nosa.model import NosaForCausalLM
from models.nosa.tests.test_model import (
    dense_attention,
    initialized_model,
    split_projection_state,
    tiny_config,
    write_checkpoint,
)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("tokens", [1, 7])
@torch.inference_mode()
def test_qkv_projection_uses_one_gemm_and_observes_parameter_updates(monkeypatch, bias, tokens):
    config = tiny_config(attention_bias=bias)
    torch.manual_seed(32)
    layer = nosa_layers.NosaAttention(config, device="cpu", dtype=torch.float32)
    x = torch.randn(tokens, config.hidden_size)
    expected_parameters = {"qkv_proj.weight", "o_proj.weight"}
    if bias:
        expected_parameters |= {"qkv_proj.bias", "o_proj.bias"}
    parameters = dict(layer.named_parameters())
    assert set(parameters) == set(layer.state_dict()) == expected_parameters
    assert isinstance(layer.qkv_proj, nn.Linear)
    projected = []
    handle = layer.qkv_proj.register_forward_hook(
        lambda module, args, output: projected.append(output)
    )
    widths = (
        config.num_attention_heads * config.head_dim,
        config.num_key_value_heads * config.head_dim,
        config.num_key_value_heads * config.head_dim,
    )
    expected_parts = []

    def check_projection_view(actual, part):
        expected = expected_parts[part].view(tokens, -1, config.head_dim)
        torch.testing.assert_close(actual, expected)
        packed = projected[-1]
        assert actual.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
        assert actual.storage_offset() == packed.storage_offset() + sum(widths[:part])
        if tokens > 1:
            assert actual.stride(0) == packed.stride(0)

    def identity_rope(q, k, positions, cos_sin_cache):
        check_projection_view(q, 0)
        check_projection_view(k, 1)
        return q, k

    def attention(q, selection, cache_access, context):
        assert selection is None
        records = cache_access.layer_view(context.layer_idx)
        check_projection_view(records["values"], 2)
        return q

    monkeypatch.setattr(nosa_layers, "apply_rotary_qk", identity_rope)
    try:
        for invocation in range(2):
            if invocation:
                layer.qkv_proj.weight.add_(0.1)
                if bias:
                    layer.qkv_proj.bias.sub_(0.2)
            weights = layer.qkv_proj.weight.split(widths, dim=0)
            biases = (None,) * 3 if not bias else layer.qkv_proj.bias.split(widths, dim=0)
            expected_parts[:] = [
                F.linear(x, weight, offset) for weight, offset in zip(weights, biases, strict=True)
            ]
            expected = F.linear(expected_parts[0], layer.o_proj.weight, layer.o_proj.bias)
            actual = layer(x, None, None, attention, None, 0)
            torch.testing.assert_close(actual, expected)
            assert len(projected) == invocation + 1
            assert all(layer.get_parameter(name) is value for name, value in parameters.items())
    finally:
        handle.remove()


def test_checkpoint_allows_independent_packed_and_split_parameter_groups(tmp_path):
    config = tiny_config(attention_bias=True, mlp_bias=True)
    model = initialized_model(config)
    state = split_projection_state(config, model.state_dict())
    for block, packed, components, parameter in (
        ("self_attn", "qkv_proj", ("q_proj", "k_proj", "v_proj"), "weight"),
        ("mlp", "gate_up_proj", ("gate_proj", "up_proj"), "bias"),
    ):
        prefix = f"model.layers.0.{block}"
        for component in components:
            del state[f"{prefix}.{component}.{parameter}"]
        packed_key = f"{prefix}.{packed}.{parameter}"
        state[packed_key] = model.state_dict()[packed_key]
    write_checkpoint(tmp_path, config, state, sharded=True)
    loaded = NosaForCausalLM.from_pretrained(
        tmp_path, device="cpu", dtype=torch.float32, attention=dense_attention
    )
    torch.testing.assert_close(loaded.state_dict(), model.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize("projection", ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"])
@pytest.mark.parametrize("parameter", ["weight", "bias"])
def test_split_checkpoint_requires_every_projection_piece(tmp_path, projection, parameter):
    config = tiny_config(attention_bias=True, mlp_bias=True)
    state = split_projection_state(config, initialized_model(config).state_dict())
    block = "self_attn" if projection in ("q_proj", "k_proj", "v_proj") else "mlp"
    del state[f"model.layers.0.{block}.{projection}.{parameter}"]
    write_checkpoint(tmp_path, config, state, sharded=True)
    with pytest.raises((ValueError, RuntimeError)):
        NosaForCausalLM.from_pretrained(
            tmp_path, device="cpu", dtype=torch.float32, attention=dense_attention
        )


@pytest.mark.parametrize(
    "projection,packed", [("self_attn.k_proj", "qkv_proj"), ("mlp.up_proj", "gate_up_proj")]
)
@pytest.mark.parametrize("parameter", ["weight", "bias"])
@pytest.mark.parametrize("problem", ["shape", "integer", "mixed"])
def test_split_checkpoint_rejects_invalid_projection_data(
    tmp_path, projection, packed, parameter, problem
):
    config = tiny_config(attention_bias=True, mlp_bias=True)
    packed_state = initialized_model(config).state_dict()
    state = split_projection_state(config, packed_state)
    key = f"model.layers.0.{projection}.{parameter}"
    if problem == "shape":
        state[key] = state[key][:-1].clone()
    elif problem == "integer":
        state[key] = state[key].to(torch.int32)
    else:
        block = projection.split(".")[0]
        packed_key = f"model.layers.0.{block}.{packed}.{parameter}"
        state[packed_key] = packed_state[packed_key]
    write_checkpoint(tmp_path, config, state, sharded=True)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        NosaForCausalLM.from_pretrained(
            tmp_path, device="cpu", dtype=torch.float32, attention=dense_attention
        )


@pytest.mark.parametrize("projection", ["self_attn.qkv_proj", "mlp.gate_up_proj"])
@pytest.mark.parametrize("parameter", ["weight", "bias"])
def test_packed_checkpoint_rejects_nonfloating_projection(tmp_path, projection, parameter):
    config = tiny_config(attention_bias=True, mlp_bias=True)
    state = dict(initialized_model(config).state_dict())
    key = f"model.layers.0.{projection}.{parameter}"
    state[key] = state[key].to(torch.int32)
    write_checkpoint(tmp_path, config, state)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        NosaForCausalLM.from_pretrained(
            tmp_path, device="cpu", dtype=torch.float32, attention=dense_attention
        )
