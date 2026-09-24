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

from models.nosa.model import NosaConfig, NosaForCausalLM


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
