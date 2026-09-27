"""NOSA model equations, CIS checkpoints, and transactional sparse decoding."""

import math

import pytest
import torch

from models.nosa.model import NosaForCausalLM
from models.nosa.tests.test_model import (
    dense_attention,
    split_projection_state,
    tiny_config,
    write_checkpoint,
)


def initialized_sparse_model(
    config=None, *, device="cpu", dtype=torch.float32, backend="reference"
):
    model = NosaForCausalLM(
        config or tiny_config(max_position_embeddings=256),
        device=device,
        dtype=dtype,
        attention_mode="sparse",
        sparse_backend=backend,
    )
    generator = torch.Generator().manual_seed(231)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            values = torch.randn(parameter.shape, generator=generator) * 0.08
            if "norm.weight" in name:
                values += 1
            elif name.endswith(".A"):
                values = -torch.arange(1, parameter.numel() + 1, dtype=torch.float32) * 4
            elif ".delta.weight" in name:
                values *= 6
            parameter.copy_(values)
    return model.eval()


def short_context_oracle(config, state, input_ids, *, include_cis=True):
    """Independent FP64 Llama + learned token bias, where all causal blocks fit.

    This does not call the model's RoPE, scoring, selection, or attention code.
    CIS uses log(1 + exp(delta(V))) * A and is added directly to QK logits.
    """
    assert len(input_ids) <= 64 * 64
    weights = {
        name: tensor.cpu().double()
        for name, tensor in split_projection_state(config, state).items()
    }

    def linear(value, name):
        result = value @ weights[name + ".weight"].T
        if name + ".bias" in weights:
            result += weights[name + ".bias"]
        return result

    def normalize(value, name):
        return (
            value
            / (value.square().mean(-1, keepdim=True) + config.rms_norm_eps).sqrt()
            * weights[name + ".weight"]
        )

    positions = torch.arange(len(input_ids), dtype=torch.float64)
    frequency = config.rope_theta ** (
        -torch.arange(0, config.head_dim, 2, dtype=torch.float64) / config.head_dim
    )
    attention_factor = 1.0
    if config.rope_scaling is not None:
        scaling = config.rope_scaling
        factor_key = (
            "long_factor"
            if config.max_position_embeddings > scaling["original_max_position_embeddings"]
            else "short_factor"
        )
        frequency /= torch.tensor(scaling[factor_key], dtype=torch.float64)
        attention_factor = scaling["attention_factor"]
    angles = positions[:, None] * frequency[None, :]
    phase = torch.polar(torch.full_like(angles, attention_factor), angles)

    def rotate(value):
        real, imaginary = value.chunk(2, dim=-1)
        rotated = torch.complex(real, imaginary) * phase[:, None, :]
        return torch.cat((rotated.real, rotated.imag), dim=-1)

    causal = positions[None, :] <= positions[:, None]
    groups = config.num_attention_heads // config.num_key_value_heads
    hidden = weights["model.embed_tokens.weight"][input_ids.cpu()]
    biases = []
    for index in range(config.num_hidden_layers):
        layer = f"model.layers.{index}"
        attention = layer + ".self_attn"
        normalized = normalize(hidden, layer + ".input_layernorm")
        query = linear(normalized, attention + ".q_proj").reshape(
            -1, config.num_attention_heads, config.head_dim
        )
        key = linear(normalized, attention + ".k_proj").reshape(
            -1, config.num_key_value_heads, config.head_dim
        )
        value = linear(normalized, attention + ".v_proj").reshape_as(key)
        projected = linear(value.flatten(1), attention + ".delta")
        cis = torch.logaddexp(torch.zeros_like(projected), projected) * weights[attention + ".A"]
        biases.append(cis)
        query, key = rotate(query), rotate(key)
        outputs = []
        for head in range(config.num_attention_heads):
            kv_head = head // groups
            logits = query[:, head] @ key[:, kv_head].T / math.sqrt(config.head_dim)
            if include_cis:
                logits += cis[:, kv_head][None, :]
            probability = logits.masked_fill(~causal, -torch.inf).softmax(-1)
            outputs.append(probability @ value[:, kv_head])
        attended = torch.stack(outputs, dim=1).flatten(1)
        hidden += linear(attended, attention + ".o_proj")
        normalized = normalize(hidden, layer + ".post_attention_layernorm")
        gate = linear(normalized, layer + ".mlp.gate_proj")
        up = linear(normalized, layer + ".mlp.up_proj")
        hidden += linear(gate.sigmoid() * gate * up, layer + ".mlp.down_proj")
    logits = linear(normalize(hidden, "model.norm"), "lm_head")
    return logits.float(), torch.stack(biases).float()


@pytest.mark.parametrize("bias", [False, True])
@torch.inference_mode()
def test_sparse_short_context_matches_independent_cis_math(bias):
    config = tiny_config(
        attention_bias=bias,
        mlp_bias=bias,
        rope_scaling={
            "rope_type": "longrope",
            "original_max_position_embeddings": 8,
            "short_factor": [1.1, 1.3, 1.5, 1.7],
            "long_factor": [1.1, 1.3, 1.5, 1.7],
            "attention_factor": 1.05,
        },
    )
    model = initialized_sparse_model(config)
    tokens = torch.tensor([1, 7, 3, 41, 5, 2, 4, 9, 12, 20, 3])
    expected, _ = short_context_oracle(config, model.state_dict(), tokens)
    without_cis, _ = short_context_oracle(config, model.state_dict(), tokens, include_cis=False)
    assert (expected - without_cis).abs().max() > 0.01
    torch.testing.assert_close(model(tokens), expected, atol=5e-6, rtol=5e-5)
    torch.testing.assert_close(model(tokens, logits_to_keep=2), expected[-2:], atol=5e-6, rtol=5e-5)
    hidden = model(tokens, return_hidden=True)
    torch.testing.assert_close(model.lm_head(hidden).float(), expected, atol=5e-6, rtol=5e-5)


@torch.inference_mode()
def test_sparse_cache_crosses_compression_and_block_boundaries_and_resets():
    model = initialized_sparse_model()
    tokens = (torch.arange(133) * 7 + 3) % model.config.vocab_size
    expected, expected_cis = short_context_oracle(model.config, model.state_dict(), tokens)
    cache = model.new_cache(len(tokens))
    assert cache.with_cis
    actual = []
    start = 0
    for stop in (31, 32, 47, 48, 63, 64, 97, 132, 133):
        actual.append(model(tokens[start:stop], cache))
        assert cache.length == stop
        start = stop
    torch.testing.assert_close(torch.cat(actual), expected, atol=5e-6, rtol=5e-5)
    torch.testing.assert_close(cache.cis_scores, expected_cis, atol=1e-5, rtol=5e-5)
    cache.reset()
    assert cache.length == 0
    assert all(cache.get_layer_state(i) is None for i in range(model.config.num_hidden_layers))
    replacement = torch.tensor([1, 9, 5, 7, 3])
    torch.testing.assert_close(model(replacement, cache), model(replacement), atol=5e-6, rtol=5e-5)
    assert cache.length == len(replacement)


@pytest.mark.parametrize("failure_stage", ["attention", "lm_head"])
@torch.inference_mode()
def test_sparse_failure_preserves_cis_and_committed_state_for_retry(monkeypatch, failure_stage):
    model = initialized_sparse_model()
    tokens = (torch.arange(67) * 5 + 1) % model.config.vocab_size
    expected, _ = short_context_oracle(model.config, model.state_dict(), tokens)
    cache = model.new_cache(len(tokens))
    model(tokens[:31], cache)
    previous = {name: buffer[:, :31].clone() for name, buffer in cache.buffers.items()}
    # Opaque request state must obey the same transaction as K/V and CIS.
    states = tuple({"committed_length": 31, "layer": i} for i in range(2))
    for index, state in enumerate(states):
        cache.set_layer_state(index, state)
    calls = 0
    attention = model.main_attention

    def failing_attention(q, selection, access, context):
        nonlocal calls
        calls += 1
        access.set_layer_state(context.layer_idx, {"uncommitted_length": 67})
        if calls == 2:
            raise RuntimeError("injected sparse attention failure")
        return attention(q, selection, access, context)

    def failing_head(hidden):
        raise RuntimeError("injected LM head failure")

    with monkeypatch.context() as patch:
        if failure_stage == "attention":
            patch.setattr(model, "main_attention", failing_attention)
        else:
            patch.setattr(model.lm_head, "forward", failing_head)
        with pytest.raises(RuntimeError, match="injected"):
            model(tokens[31:], cache)
    assert cache.length == 31
    for name, old in previous.items():
        torch.testing.assert_close(cache.buffers[name][:, :31], old, atol=0, rtol=0)
    assert all(cache.get_layer_state(i) is state for i, state in enumerate(states))
    torch.testing.assert_close(model(tokens[31:], cache), expected[31:], atol=5e-6, rtol=5e-5)
    assert cache.length == len(tokens)


@pytest.mark.parametrize("cache_mode", ["dense", "sparse"])
def test_sparse_and_dense_cache_layouts_cannot_be_mixed(cache_mode):
    sparse = initialized_sparse_model()
    dense = NosaForCausalLM(sparse.config, attention=dense_attention)
    source, target = (dense, sparse) if cache_mode == "dense" else (sparse, dense)
    cache = source.new_cache(3)
    with pytest.raises(ValueError, match="(?i)(cache|layout|CIS)"):
        target(torch.tensor([1, 2, 3]), cache)
    assert cache.length == 0


@pytest.mark.parametrize("layout", ["split", "packed"])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@torch.inference_mode()
def test_sparse_checkpoint_loads_every_cis_parameter(tmp_path, layout, sharded, bias):
    config = tiny_config(attention_bias=bias, mlp_bias=bias)
    model = initialized_sparse_model(config)
    state = dict(model.state_dict())
    if layout == "split":
        state = split_projection_state(config, state)
    write_checkpoint(tmp_path, config, state, sharded=sharded)
    loaded = NosaForCausalLM.from_pretrained(
        tmp_path,
        device="cpu",
        dtype=torch.float32,
        attention_mode="sparse",
        sparse_backend="reference",
    )
    assert loaded.ignored_checkpoint_keys == ()
    torch.testing.assert_close(loaded.state_dict(), model.state_dict(), atol=0, rtol=0)
    tokens = torch.tensor([1, 7, 2, 9])
    cache = loaded.new_cache(len(tokens))
    torch.testing.assert_close(loaded(tokens, cache), model(tokens), atol=0, rtol=0)
    assert cache.with_cis and cache.cis_scores.shape == (2, len(tokens), 2)
    # The established dense loader still accepts a full NOSA checkpoint.
    dense = NosaForCausalLM.from_pretrained(
        tmp_path, device="cpu", dtype=torch.float32, attention=dense_attention
    )
    expected_ignored = {
        f"model.layers.{index}.self_attn.{suffix}"
        for index in range(config.num_hidden_layers)
        for suffix in (("A", "delta.weight", "delta.bias") if bias else ("A", "delta.weight"))
    }
    assert set(dense.ignored_checkpoint_keys) == expected_ignored


@pytest.mark.parametrize("problem", ["missing_A", "missing_delta", "missing_bias", "shape"])
def test_sparse_checkpoint_rejects_missing_or_malformed_cis(tmp_path, problem):
    config = tiny_config(attention_bias=True)
    state = dict(initialized_sparse_model(config).state_dict())
    prefix = "model.layers.1.self_attn."
    if problem == "shape":
        state[prefix + "A"] = torch.ones(3)
    else:
        suffix = {"missing_A": "A", "missing_delta": "delta.weight", "missing_bias": "delta.bias"}[
            problem
        ]
        del state[prefix + suffix]
    write_checkpoint(tmp_path, config, state)
    with pytest.raises(ValueError, match="(?i)(checkpoint|shape|missing)"):
        NosaForCausalLM.from_pretrained(
            tmp_path,
            device="cpu",
            dtype=torch.float32,
            attention_mode="sparse",
            sparse_backend="reference",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton attention")
@torch.inference_mode()
def test_cuda_sparse_model_triton_matches_reference_prefill_and_decode():
    config = tiny_config(
        hidden_size=256,
        head_dim=64,
        intermediate_size=384,
        max_position_embeddings=256,
    )
    reference = initialized_sparse_model(config, device="cuda", dtype=torch.bfloat16)
    fused = NosaForCausalLM(
        config,
        device="cuda",
        dtype=torch.bfloat16,
        attention_mode="sparse",
        sparse_backend="triton",
    )
    fused.load_state_dict(reference.state_dict())
    tokens = (torch.arange(129, device="cuda") * 7 + 1) % config.vocab_size
    expected = reference(tokens)
    torch.testing.assert_close(fused(tokens), expected, atol=0.06, rtol=0.04)
    cache = fused.new_cache(len(tokens))
    actual = torch.cat(
        [fused(tokens[:31], cache), fused(tokens[31:128], cache), fused(tokens[128:], cache)]
    )
    torch.testing.assert_close(actual, expected, atol=0.06, rtol=0.04)
    assert cache.length == len(tokens)
    assert torch.isfinite(cache.cis_scores).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton attention")
@torch.inference_mode()
def test_cuda_sparse_decode_beyond_budget_matches_independent_selected_attention(monkeypatch):
    config = tiny_config(
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        intermediate_size=192,
        num_hidden_layers=1,
        max_position_embeddings=4097,
    )
    model = initialized_sparse_model(config, device="cuda", dtype=torch.bfloat16, backend="triton")
    tokens = (torch.arange(4097, device="cuda") * 7 + 1) % config.vocab_size
    cache = model.new_cache(len(tokens))
    model(tokens[:-1], cache, logits_to_keep=1)
    attention = model.main_attention
    seen = []

    def compare_selected_attention(query, selection, access, context):
        assert context.query_start == 4096 and context.query_length == 1
        blocks = selection.block_ids[0, 0][selection.valid_mask[0, 0]]
        assert blocks.numel() == blocks.unique().numel() == 64
        assert blocks.min() == 0 and blocks.max() == 64
        records = access.layer_view(context.layer_idx)
        positions = torch.arange(4097, device=query.device)
        allowed = torch.isin(positions // 64, blocks)
        # The sequence now has 65 causal blocks, so the indexer must omit one.
        assert (~allowed).sum() == 64
        key = records["keys"][:, 0].double()
        value = records["values"][:, 0].double()
        bias = records["cis_scores"][:, 0].double()
        logits = query[0].double() @ key.T / math.sqrt(config.head_dim) + bias[None, :]
        expected = logits.masked_fill(~allowed[None, :], -torch.inf).softmax(-1) @ value
        actual = attention(query, selection, access, context)
        torch.testing.assert_close(actual[0], expected.to(query.dtype), atol=0.02, rtol=0.02)
        seen.append(selection)
        return actual

    monkeypatch.setattr(model, "main_attention", compare_selected_attention)
    result = model(tokens[-1:], cache)
    assert len(seen) == 1 and cache.length == len(tokens)
    assert result.shape == (1, config.vocab_size) and torch.isfinite(result).all()
