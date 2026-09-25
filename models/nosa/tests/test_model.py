"""Dense NOSA math, cache semantics, checkpoint loading, and FlashInfer checks."""

import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from layers.attention import AttentionContext, BlockSelection, DenseMainAttention
from models.nosa.indexer import NosaIndexer, NosaSelectionPolicy
from models.nosa.model import NosaConfig, NosaForCausalLM
from operators.sm90.sparse_attention import SM90SparseAttention


def tiny_config(**kwargs):
    values = {
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 43,
        "max_position_embeddings": 32,
    }
    values.update(kwargs)
    return NosaConfig(**values)


def dense_attention(q, k, v):
    """Explicit per-head GQA with the causal diagonal at the bottom right."""
    outputs = []
    group_size = q.shape[1] // k.shape[1]
    rows = torch.arange(q.shape[0], device=q.device) + k.shape[0] - q.shape[0]
    columns = torch.arange(k.shape[0], device=q.device)
    allowed = columns[None, :] <= rows[:, None]
    for head in range(q.shape[1]):
        scores = q[:, head].double() @ k[:, head // group_size].double().T
        scores = scores / math.sqrt(q.shape[-1])
        probs = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
        outputs.append(probs @ v[:, head // group_size].double())
    return torch.stack(outputs, dim=1).to(q.dtype)


def initialized_model(config=None, *, device="cpu", dtype=torch.float32, attention=dense_attention):
    model = NosaForCausalLM(
        config or tiny_config(), device=device, dtype=dtype, attention=attention
    )
    generator = torch.Generator(device=device).manual_seed(1234)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            values = torch.randn(parameter.shape, generator=generator, device=device) * 0.12
            if "norm.weight" in name:
                values += 1
            parameter.copy_(values)
    return model.eval()


def reference_logits(config, weights, input_ids):
    """Independent FP64 Llama equations; RoPE uses complex multiplication."""
    weights = {name: tensor.double() for name, tensor in weights.items()}

    def linear(x, name):
        result = x @ weights[name + ".weight"].T
        if name + ".bias" in weights:
            result += weights[name + ".bias"]
        return result

    def norm(x, name):
        return (
            x
            / (x.square().mean(-1, keepdim=True) + config.rms_norm_eps).sqrt()
            * weights[name + ".weight"]
        )

    def rotate(x):
        frequency = config.rope_theta ** (
            -torch.arange(0, config.head_dim, 2, dtype=torch.float64) / config.head_dim
        )
        scale = 1.0
        if config.rope_scaling is not None:
            rope = config.rope_scaling
            original = rope["original_max_position_embeddings"]
            factor_name = (
                "long_factor" if config.max_position_embeddings > original else "short_factor"
            )
            frequency /= torch.tensor(rope[factor_name], dtype=torch.float64)
            scale = rope["attention_factor"]
        angles = torch.arange(len(input_ids), dtype=torch.float64)[:, None] * frequency[None, :]
        phases = torch.polar(torch.full_like(angles, scale), angles)
        real, imaginary = x.chunk(2, dim=-1)
        result = torch.complex(real, imaginary) * phases[:, None, :]
        return torch.cat((result.real, result.imag), dim=-1)

    hidden = weights["model.embed_tokens.weight"][input_ids]
    for index in range(config.num_hidden_layers):
        layer = f"model.layers.{index}"
        normalized = norm(hidden, layer + ".input_layernorm")
        q = linear(normalized, layer + ".self_attn.q_proj").reshape(
            -1, config.num_attention_heads, config.head_dim
        )
        k = linear(normalized, layer + ".self_attn.k_proj").reshape(
            -1, config.num_key_value_heads, config.head_dim
        )
        v = linear(normalized, layer + ".self_attn.v_proj").reshape(
            -1, config.num_key_value_heads, config.head_dim
        )
        attended = dense_attention(rotate(q), rotate(k), v).flatten(1)
        hidden = hidden + linear(attended, layer + ".self_attn.o_proj")
        normalized = norm(hidden, layer + ".post_attention_layernorm")
        gate = linear(normalized, layer + ".mlp.gate_proj")
        up = linear(normalized, layer + ".mlp.up_proj")
        hidden = hidden + linear(gate.sigmoid() * gate * up, layer + ".mlp.down_proj")
    return linear(norm(hidden, "model.norm"), "lm_head").float()


@pytest.mark.parametrize("use_longrope", [False, True])
@pytest.mark.parametrize("use_bias", [False, True])
@torch.inference_mode()
def test_model_matches_independent_dense_math(use_longrope, use_bias):
    rope = None
    if use_longrope:
        rope = {
            "rope_type": "longrope",
            "original_max_position_embeddings": 8,
            "short_factor": [1.5, 2.0, 2.5, 3.0],
            "long_factor": [1.5, 2.0, 2.5, 3.0],
            "attention_factor": 1.15,
        }
    config = tiny_config(rope_scaling=rope, attention_bias=use_bias, mlp_bias=use_bias)
    model = initialized_model(config)
    tokens = torch.tensor([1, 8, 3, 41, 9, 2, 7, 6, 11, 4, 1])
    actual = model(tokens)
    expected = reference_logits(config, model.state_dict(), tokens)
    assert actual.shape == (len(tokens), config.vocab_size)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(model(tokens, logits_to_keep=1), expected[-1:], atol=3e-6, rtol=3e-5)


@torch.inference_mode()
def test_longrope_within_original_context():
    config = tiny_config(
        max_position_embeddings=8,
        rope_scaling={
            "rope_type": "longrope",
            "original_max_position_embeddings": 8,
            "short_factor": [1.0, 1.4, 1.8, 2.2],
            "long_factor": [1.0, 1.4, 1.8, 2.2],
            "attention_factor": 1.1,
        },
    )
    model = initialized_model(config)
    tokens = torch.tensor([1, 3, 7, 11, 4, 2])
    torch.testing.assert_close(
        model(tokens), reference_logits(config, model.state_dict(), tokens), atol=3e-6, rtol=3e-5
    )


@torch.inference_mode()
def test_chunked_prefill_decode_and_cache_reset():
    config = tiny_config(
        rope_scaling={
            "rope_type": "longrope",
            "original_max_position_embeddings": 4,
            "short_factor": [1.2, 1.4, 1.8, 2.1],
            "long_factor": [1.2, 1.4, 1.8, 2.1],
            "attention_factor": 1.0,
        }
    )
    model = initialized_model(config)
    tokens = torch.tensor([1, 3, 5, 7, 8, 11, 2, 9, 8])
    expected = reference_logits(config, model.state_dict(), tokens)
    cache = model.new_cache(len(tokens))
    assert cache.length == 0
    chunks = []
    start = 0
    for size in (3, 3, 1, 1, 1):
        chunks.append(model(tokens[start : start + size], cache))
        start += size
        assert cache.length == start
    torch.testing.assert_close(torch.cat(chunks), expected, atol=3e-6, rtol=3e-5)

    cache.reset()
    assert cache.length == 0
    replacement = torch.tensor([9, 4, 3])
    torch.testing.assert_close(model(replacement, cache), model(replacement), atol=3e-6, rtol=3e-5)
    assert cache.length == len(replacement)


@torch.inference_mode()
def test_cache_overflow_does_not_advance_cache():
    model = initialized_model()
    cache = model.new_cache(4)
    tokens = torch.tensor([1, 3, 4, 7])
    model(tokens[:2], cache)
    with pytest.raises((ValueError, RuntimeError), match="(?i)(capacity|overflow|max|length)"):
        model(torch.tensor([1, 3, 4]), cache)
    assert cache.length == 2
    actual = model(tokens[2:], cache)
    torch.testing.assert_close(actual, model(tokens)[2:], atol=3e-6, rtol=3e-5)


def write_checkpoint(path, config, state, *, sharded=False):
    (path / "config.json").write_text(json.dumps(asdict(config)))
    if not sharded:
        save_file(state, path / "model.safetensors")
        return
    names = sorted(state)
    shards = (names[::2], names[1::2])
    mapping = {}
    for index, shard in enumerate(shards, start=1):
        filename = f"model-{index:05d}-of-00002.safetensors"
        save_file({name: state[name] for name in shard}, path / filename)
        mapping.update(dict.fromkeys(shard, filename))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))


@torch.inference_mode()
def test_backbone_features_skip_lm_head_and_support_candidate_extend():
    model = initialized_model()
    tokens = torch.tensor([1, 7, 5, 3, 9, 2])
    logits = model(tokens)
    hidden = model(tokens, return_hidden=True)
    torch.testing.assert_close(model.lm_head(hidden).float(), logits)
    cache = model.new_cache(len(tokens))

    # GR returns features and must never execute a vocabulary projection.
    def reject_lm_head(*args, **kwargs):
        raise AssertionError("GR must not execute lm_head")

    model.lm_head.forward = reject_lm_head
    model(tokens[:3], cache, return_hidden=True)
    extended = model(tokens[3:], cache, return_hidden=True)
    torch.testing.assert_close(extended, hidden[3:], atol=3e-6, rtol=3e-5)
    assert cache.length == len(tokens)
    with pytest.raises(ValueError, match="return_hidden"):
        model(tokens, return_hidden=True, logits_to_keep=1)


@pytest.mark.parametrize("sharded", [False, True])
@torch.inference_mode()
def test_checkpoint_loading_ignores_only_cis_weights(tmp_path, sharded):
    config = tiny_config()
    model = initialized_model(config)
    state = dict(model.state_dict())
    for index in range(config.num_hidden_layers):
        state[f"model.layers.{index}.self_attn.A"] = torch.ones(config.num_key_value_heads)
        state[f"model.layers.{index}.self_attn.delta.weight"] = torch.zeros(
            config.num_key_value_heads, config.num_key_value_heads * config.head_dim
        )
    write_checkpoint(tmp_path, config, state, sharded=sharded)
    tokens = torch.tensor([1, 7, 5, 3])
    loaded = NosaForCausalLM.from_pretrained(
        tmp_path, device="cpu", dtype=torch.float32, attention=dense_attention
    )
    assert set(loaded.state_dict()) == set(model.state_dict())
    torch.testing.assert_close(loaded(tokens), model(tokens), rtol=0, atol=0)
    loaded_cache = loaded.new_cache(len(tokens))
    assert loaded_cache.device == torch.device("cpu")
    assert loaded_cache.dtype == torch.float32
    torch.testing.assert_close(loaded(tokens, loaded_cache), model(tokens), rtol=0, atol=0)


@pytest.mark.parametrize("failure_stage", ["attention", "lm_head"])
def test_failed_forward_preserves_committed_prefix_and_allows_retry(monkeypatch, failure_stage):
    model = initialized_model()
    tokens = torch.tensor([1, 3, 7, 5, 2, 8])
    expected = model(tokens)
    cache = model.new_cache(len(tokens))
    model(tokens[:3], cache)
    prefix_keys = cache.keys[:, :3].clone()
    prefix_values = cache.values[:, :3].clone()
    calls = 0

    def failed_attention(q, k, v):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("attention failure")
        return dense_attention(q, k, v)

    def failed_head(hidden):
        raise RuntimeError("lm_head failure")

    with monkeypatch.context() as patch:
        if failure_stage == "attention":
            patch.setattr(model, "attention", failed_attention)
        else:
            patch.setattr(model.lm_head, "forward", failed_head)
        with pytest.raises(RuntimeError, match="failure"):
            model(tokens[3:], cache)
    assert cache.length == 3
    torch.testing.assert_close(cache.keys[:, :3], prefix_keys, rtol=0, atol=0)
    torch.testing.assert_close(cache.values[:, :3], prefix_values, rtol=0, atol=0)
    torch.testing.assert_close(model(tokens[3:], cache), expected[3:], atol=3e-6, rtol=3e-5)
    assert cache.length == len(tokens)


def test_indexer_and_main_attention_receive_logical_selection_and_request_state():
    model = initialized_model()
    tokens = torch.tensor([1, 3, 7, 5, 2, 8])
    expected = model(tokens)
    selections = []
    seen = []

    def indexer(q, cache_access, context):
        state = context.auxiliary_state
        assert state is None if context.query_start == 0 else state == context.query_start
        cache_access.set_layer_state(context.layer_idx, context.query_start + context.query_length)
        selection = BlockSelection(
            torch.zeros((q.shape[0], model.config.num_key_value_heads, 1), dtype=torch.long),
            block_size=64,
        )
        selections.append(selection)
        return selection

    def main_attention(q, selection, cache_access, context):
        assert selection is selections[-1]
        assert context.query_length == q.shape[0]
        records = cache_access.layer_view(context.layer_idx)
        assert records["keys"].shape[0] == context.query_start + context.query_length
        seen.append((context.layer_idx, context.query_start, context.query_length))
        return dense_attention(q, records["keys"], records["values"])

    model.indexer = indexer
    model.main_attention = main_attention
    cache = model.new_cache(len(tokens))
    first = model(tokens[:3], cache)
    second = model(tokens[3:], cache)
    torch.testing.assert_close(torch.cat((first, second)), expected, atol=3e-6, rtol=3e-5)
    assert seen == [(0, 0, 3), (1, 0, 3), (0, 3, 3), (1, 3, 3)]
    assert all(cache.get_layer_state(i) == len(tokens) for i in range(2))
    torch.testing.assert_close(model(tokens), expected, atol=3e-6, rtol=3e-5)


def test_reserved_backends_fail_explicitly_and_cache_step_is_aborted():
    model = initialized_model()
    cache = model.new_cache(4)
    tokens = torch.tensor([1, 3, 5])
    policy = NosaSelectionPolicy()
    assert (policy.block_size, policy.block_budget) == (64, 64)
    assert (policy.sink_blocks, policy.local_blocks, policy.topk_blocks) == (1, 16, 47)
    model.indexer = NosaIndexer()
    with pytest.raises(NotImplementedError, match="NOSA block scoring"):
        model(tokens, cache)
    assert cache.length == 0
    model.indexer = None
    model.main_attention = SM90SparseAttention()
    with pytest.raises(NotImplementedError, match="SM90 sparse attention"):
        model(tokens, cache)
    assert cache.length == 0
    model.main_attention = DenseMainAttention(dense_attention)
    assert model(tokens, cache).shape[0] == len(tokens)


def test_dense_adapter_rejects_nonresident_access_and_sparse_selection():
    dense = DenseMainAttention(dense_attention)
    context = AttentionContext(layer_idx=0, query_start=0, query_length=1)
    q = torch.zeros((1, 4, 8))
    with pytest.raises(NotImplementedError, match="resident"):
        dense(q, None, object(), context)
    selection = BlockSelection(torch.zeros((1, 1, 1), dtype=torch.long), block_size=64)
    with pytest.raises(NotImplementedError, match="block selection"):
        dense(q, selection, object(), context)


def test_legacy_cache_cursor_override_is_validated_before_forward():
    model = initialized_model()
    tokens = torch.tensor([1, 3, 5, 7])
    cache = model.new_cache(4)
    expected = model(tokens, cache)
    cache.length = 2
    torch.testing.assert_close(model(tokens[2:], cache), expected[2:], atol=3e-6, rtol=3e-5)
    cache.length = -1
    with pytest.raises(ValueError, match="cache length"):
        model(tokens, cache)
    cache.reset()
    cache.begin_step(1)
    with pytest.raises(RuntimeError, match="pending"):
        cache.length = 0
    cache.abort_step()
    model.cache_manager.release(cache)
    with pytest.raises(RuntimeError, match="released"):
        cache.length = 0


@pytest.mark.parametrize("problem", ["missing", "extra", "shape", "unknown_cis_layer"])
def test_checkpoint_loading_rejects_mismatches(tmp_path, problem):
    config = tiny_config()
    state = dict(initialized_model(config).state_dict())
    key = "model.layers.0.self_attn.q_proj.weight"
    if problem == "missing":
        del state[key]
    elif problem == "extra":
        state["unexpected.weight"] = torch.ones(3)
    elif problem == "shape":
        state[key] = state[key][:-1].clone()
    else:
        state[f"model.layers.{config.num_hidden_layers}.self_attn.A"] = torch.ones(2)
    write_checkpoint(tmp_path, config, state)
    with pytest.raises((ValueError, RuntimeError)):
        NosaForCausalLM.from_pretrained(
            tmp_path, device="cpu", dtype=torch.float32, attention=dense_attention
        )


def test_config_reads_nosa_metadata_and_eos(tmp_path):
    raw = asdict(tiny_config(vocab_size=73448))
    raw.update(
        {
            "architectures": ["SparseLlamaForCausalLM"],
            "model_type": "llama",
            "hidden_act": "silu",
            "eos_token_id": [2, 73440],
            "auto_map": {"AutoModelForCausalLM": "unused_remote_code.SparseLlamaForCausalLM"},
        }
    )
    (tmp_path / "config.json").write_text(json.dumps(raw))
    config = NosaConfig.from_pretrained(tmp_path)
    assert tuple(config.eos_token_id) == (2, 73440)
    assert config.num_attention_heads == 4 and config.num_key_value_heads == 2
    raw["eos_token_id"] = 2
    assert tuple(NosaConfig.from_dict(raw).eos_token_id) == (2,)


def test_unsupported_rope_frequency_switch_is_rejected():
    raw = asdict(tiny_config())
    raw["rope_scaling"] = {
        "rope_type": "longrope",
        "original_max_position_embeddings": 8,
        "short_factor": [1.0] * 4,
        "long_factor": [2.0] * 4,
        "attention_factor": 1.0,
    }
    with pytest.raises(ValueError):
        NosaConfig.from_dict(raw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FlashInfer")
@torch.inference_mode()
def test_flashinfer_prefill_and_decode_match_dense_attention():
    pytest.importorskip("flashinfer")
    config = tiny_config(hidden_size=256, head_dim=64, intermediate_size=384)
    model = initialized_model(config, device="cuda", dtype=torch.bfloat16, attention=None)
    reference = NosaForCausalLM(
        config, device="cuda", dtype=torch.bfloat16, attention=dense_attention
    ).eval()
    reference.load_state_dict(model.state_dict())
    tokens = torch.tensor([1, 7, 6, 3, 2, 8, 9, 11, 12], device="cuda")
    expected = reference(tokens)
    torch.testing.assert_close(model(tokens), expected, atol=0.06, rtol=0.03)
    cache = model.new_cache(len(tokens))
    chunks = [model(tokens[:4], cache), model(tokens[4:8], cache), model(tokens[8:], cache)]
    torch.testing.assert_close(torch.cat(chunks), expected, atol=0.06, rtol=0.03)
    assert cache.length == len(tokens)
    expected_hidden = reference(tokens, return_hidden=True)
    cache.reset()
    model(tokens[:4], cache, return_hidden=True)
    candidate_hidden = model(tokens[4:], cache, return_hidden=True)
    torch.testing.assert_close(candidate_hidden, expected_hidden[4:], atol=0.06, rtol=0.03)
    assert cache.length == len(tokens)


@pytest.mark.skipif(
    not os.environ.get("NOSA_MODEL_PATH"), reason="Set NOSA_MODEL_PATH for checkpoint"
)
def test_local_checkpoint_metadata_matches_model():
    """Opt-in check reads tensor metadata without loading the 8B checkpoint."""
    path = Path(os.environ["NOSA_MODEL_PATH"])
    config = NosaConfig.from_pretrained(path)
    with safe_open(path / "model.safetensors", framework="pt") as checkpoint:
        assert checkpoint.get_slice("model.embed_tokens.weight").get_shape() == [
            config.vocab_size,
            config.hidden_size,
        ]
        assert checkpoint.get_slice("lm_head.weight").get_shape() == [
            config.vocab_size,
            config.hidden_size,
        ]
        for index in range(config.num_hidden_layers):
            prefix = f"model.layers.{index}.self_attn"
            for projection, heads in (
                ("q", config.num_attention_heads),
                ("k", config.num_key_value_heads),
                ("v", config.num_key_value_heads),
            ):
                assert checkpoint.get_slice(f"{prefix}.{projection}_proj.weight").get_shape() == [
                    heads * config.head_dim,
                    config.hidden_size,
                ]
