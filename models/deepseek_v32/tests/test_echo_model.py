import json
import math

import pytest
import torch
from safetensors.torch import save_file

from models.deepseek_v32.echo_model import (
    CheckpointAttention,
    CheckpointReader,
    Config,
    apply_rope,
    normalized_hadamard,
    quantize_index,
    rotary_frequencies,
)


@pytest.fixture
def checkpoint(tmp_path):
    config = {
        "hidden_size": 8,
        "num_attention_heads": 2,
        "q_lora_rank": 8,
        "kv_lora_rank": 8,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 4,
        "index_n_heads": 2,
        "index_head_dim": 8,
        "index_topk": 4,
        "num_hidden_layers": 2,
        "vocab_size": 16,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 128,
        "rope_theta": 10000,
        "rope_scaling": {
            "type": "yarn",
            "factor": 4,
            "original_max_position_embeddings": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1,
        },
        "quantization_config": {
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(31)
    raw = {}
    stem = "model.layers.0."
    attn = stem + "self_attn."
    for name, shape in (
        ("q_a_proj", (8, 8)),
        ("q_b_proj", (16, 8)),
        ("kv_a_proj_with_mqa", (12, 8)),
        ("kv_b_proj", (16, 8)),
        ("o_proj", (8, 8)),
        ("indexer.wq_b", (16, 8)),
        ("indexer.wk", (8, 8)),
    ):
        raw[attn + name + ".weight"] = (torch.randn(shape, generator=generator) * 8).to(
            torch.float8_e4m3fn
        )
        raw[attn + name + ".weight_scale_inv"] = torch.tensor([[0.03125]])
    for name in (
        stem + "input_layernorm.weight",
        attn + "q_a_layernorm.weight",
        attn + "kv_a_layernorm.weight",
        attn + "indexer.k_norm.weight",
    ):
        raw[name] = 0.8 + torch.rand(8, generator=generator) * 0.4
    raw[attn + "indexer.k_norm.bias"] = torch.randn(8, generator=generator) * 0.1
    raw[attn + "indexer.weights_proj.weight"] = torch.randn(2, 8, generator=generator).bfloat16()
    embedding = {"model.embed_tokens.weight": torch.randn(16, 8, generator=generator).bfloat16()}
    # Missing shard numbers and missing index are intentional: layer access is partial.
    save_file(raw, tmp_path / "model-00001-of-00163.safetensors")
    save_file(embedding, tmp_path / "model-00070-of-00163.safetensors")
    return tmp_path, config, raw | embedding


def reference_rope(x, angles, interleaved):
    a, b = (x.float()[..., ::2], x.float()[..., 1::2]) if interleaved else x.float().chunk(2, -1)
    complex_x = torch.complex(a, b)
    phases = torch.polar(torch.ones_like(angles), angles)
    phases = phases.reshape(len(x), *([1] * (x.ndim - 2)), -1)
    rotated = complex_x * phases
    if interleaved:
        return torch.view_as_real(rotated).flatten(-2).to(x.dtype)
    return torch.cat((rotated.real, rotated.imag), -1).to(x.dtype)


def reference_hadamard(x):
    width = x.shape[-1]
    matrix = torch.tensor(
        [[(-1.0) ** ((i & j).bit_count()) for j in range(width)] for i in range(width)]
    )
    return (x.float() @ matrix / math.sqrt(width)).to(x.dtype)


def reference_quantize(x):
    scale = x.float().abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
    mantissa, exponent = torch.frexp(scale)
    scale = torch.ldexp(torch.ones_like(scale), exponent - (mantissa == 0.5).int())
    return (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale


def test_checkpoint_header_lookup_does_not_require_all_shards(checkpoint):
    path, _, raw = checkpoint
    reader = CheckpointReader(path)
    assert len(reader.tensor_files) == len(raw)
    assert reader.tensor_metadata["model.embed_tokens.weight"]["shape"] == [16, 8]
    torch.testing.assert_close(
        reader.get_tensor("model.embed_tokens.weight"), raw["model.embed_tokens.weight"]
    )
    with pytest.raises(KeyError, match="Required tensor.*absent"):
        reader.get_tensor("model.layers.60.self_attn.q_a_proj.weight")


def test_fp8_dequantization_partial_blocks(tmp_path):
    data = torch.arange(129 * 131).reshape(129, 131).remainder(41).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.125, 0.25], [0.5, 1.0]])
    save_file(
        {"layer.weight": data, "layer.weight_scale_inv": scales}, tmp_path / "model.safetensors"
    )
    actual = CheckpointReader(tmp_path).linear_weight("layer", device="cpu")
    expected = torch.empty_like(data, dtype=torch.bfloat16)
    for row in range(129):
        for column in range(131):
            expected[row, column] = data[row, column].float() * scales[row // 128, column // 128]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_config_and_yarn_reference(checkpoint):
    path, _, _ = checkpoint
    cfg = Config.from_checkpoint(path)
    assert cfg.qk_head_dim == 12
    assert cfg.attention_scale == pytest.approx(8**-0.5 * (1 + 0.1 * math.log(4)) ** 2)
    # Use realistic rotary width to cover the YaRN ramp's interior and endpoints.
    cfg = Config()
    actual = rotary_frequencies(cfg, device="cpu")
    low = max(math.floor(64 * math.log(4096 / (32 * 2 * math.pi)) / (2 * math.log(10000))), 0)
    high = min(math.ceil(64 * math.log(4096 / (2 * math.pi)) / (2 * math.log(10000))), 63)
    expected = []
    for index in range(32):
        interpolation = max(0, min(1, (index - low) / (high - low)))
        base = 10000 ** (-2 * index / 64)
        expected.append(base * (1 - interpolation) + base / 40 * interpolation)
    torch.testing.assert_close(actual, torch.tensor(expected))


@pytest.mark.parametrize("interleaved", [True, False])
def test_rope_matches_official_complex_pairing(interleaved):
    x = torch.randn(5, 3, 64, generator=torch.Generator().manual_seed(8)).bfloat16()
    angles = torch.tensor([0, 1, 4096, 65536, 66559]).float()[:, None] * rotary_frequencies(
        Config(), device="cpu"
    )
    actual = apply_rope(x, angles, interleaved=interleaved)
    expected = reference_rope(x, angles, interleaved)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.015625)


def test_hadamard_and_quantization_match_dense_reference():
    x = torch.randn(7, 4, 128, generator=torch.Generator().manual_seed(7)).bfloat16()
    x[0].zero_()
    actual = normalized_hadamard(x)
    expected = reference_hadamard(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    quantized, scales = quantize_index(actual)
    ref_quantized, ref_scales = reference_quantize(expected)
    torch.testing.assert_close(scales, ref_scales, rtol=0, atol=0)
    torch.testing.assert_close(quantized.float(), ref_quantized.float(), rtol=0, atol=0)
    assert scales.isfinite().all() and scales.gt(0).all()


def test_checkpoint_project_against_independent_reference(checkpoint):
    path, _, raw = checkpoint
    model = CheckpointAttention(path, device="cpu")
    cfg = model.cfg
    hidden = model.embedding(torch.tensor([0, 7, 15]))
    torch.testing.assert_close(hidden, raw["model.embed_tokens.weight"][[0, 7, 15]], rtol=0, atol=0)
    actual = model.project(hidden, 65)
    stem = "model.layers.0."
    attn = stem + "self_attn."

    def weight(name):
        return (
            raw[attn + name + ".weight"].float() * raw[attn + name + ".weight_scale_inv"][0, 0]
        ).bfloat16()

    def linear(x, name):
        return (x.float() @ weight(name).float().T).bfloat16()

    def norm(x, name):
        return (
            x.float()
            / torch.sqrt(x.float().square().mean(-1, keepdim=True) + cfg.norm_eps)
            * raw[name]
        ).bfloat16()

    x = norm(hidden, stem + "input_layernorm.weight")
    qr = norm(linear(x, "q_a_proj"), attn + "q_a_layernorm.weight")
    q = linear(qr, "q_b_proj").reshape(3, 2, 8)
    # For the four-dimensional fixture the YaRN correction range collapses at zero.
    angles = torch.arange(65, 68).float()[:, None] * torch.tensor([1.0, 0.01 / 4])
    q_pe = reference_rope(q[..., 4:], angles, True)
    wkv_b = weight("kv_b_proj").reshape(2, 8, 8)
    q_latent = torch.stack(
        [(q[:, head, :4].float() @ wkv_b[head, :4].float()).bfloat16() for head in range(2)], 1
    )
    kv = linear(x, "kv_a_proj_with_mqa")
    expected_kv = torch.cat(
        (norm(kv[:, :8], attn + "kv_a_layernorm.weight"), reference_rope(kv[:, 8:], angles, True)),
        -1,
    )
    torch.testing.assert_close(actual.q, torch.cat((q_latent, q_pe), -1), rtol=0, atol=0)
    torch.testing.assert_close(actual.kv, expected_kv, rtol=0, atol=0)
    qi = linear(qr, "indexer.wq_b").reshape(3, 2, 8)
    ki = linear(x, "indexer.wk").float()
    ki = (
        (ki - ki.mean(-1, keepdim=True))
        / torch.sqrt(ki.var(-1, correction=0, keepdim=True) + cfg.norm_eps)
        * raw[attn + "indexer.k_norm.weight"]
        + raw[attn + "indexer.k_norm.bias"]
    ).bfloat16()
    qi = torch.cat((reference_rope(qi[..., :4], angles, False), qi[..., 4:]), -1)
    ki = torch.cat((reference_rope(ki[..., :4], angles, False), ki[..., 4:]), -1)
    qi, qs = reference_quantize(qi)
    ki, ks = reference_quantize(ki)
    weights = (x.float() @ raw[attn + "indexer.weights_proj.weight"].float().T) / math.sqrt(2)
    weights = weights * qs[..., 0] / math.sqrt(8)
    torch.testing.assert_close(actual.index_q.float(), qi.float(), rtol=0, atol=0)
    torch.testing.assert_close(actual.index_k.float(), ki.float(), rtol=0, atol=0)
    torch.testing.assert_close(actual.index_scale, ks[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(actual.index_weights, weights)
    assert all(t.is_contiguous() for t in vars(actual).values())
    attention = torch.randn(3, 2, 8, generator=torch.Generator().manual_seed(1)).bfloat16()
    heads = torch.stack(
        [(attention[:, h].float() @ wkv_b[h, 4:].float().T).bfloat16() for h in range(2)], 1
    )
    expected_output = linear(heads.flatten(1), "o_proj")
    torch.testing.assert_close(model.output(attention), expected_output, rtol=0, atol=0)


def test_project_chunk_positions_and_failures(checkpoint):
    path, _, _ = checkpoint
    model = CheckpointAttention(path, device="cpu")
    hidden = model.embedding(torch.tensor([1, 2, 3, 4]))
    whole = model.project(hidden, 10)
    left, right = model.project(hidden[:2], 10), model.project(hidden[2:], 12)
    for name, value in vars(whole).items():
        expected = torch.cat((getattr(left, name).float(), getattr(right, name).float()))
        torch.testing.assert_close(value.float(), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="context limit"):
        model.project(hidden, 126)
    with pytest.raises(ValueError, match="BF16"):
        model.project(hidden.float(), 0)
    with pytest.raises(ValueError, match="vocabulary"):
        model.embedding(torch.tensor([-1]))
    with pytest.raises(KeyError, match="Required tensor"):
        CheckpointAttention(path, layer_idx=1, device="cpu")


def test_explicit_projection_positions_match_checked_entry(checkpoint):
    path, _, _ = checkpoint
    model = CheckpointAttention(path, device="cpu")
    hidden = model.embedding(torch.tensor([0, 7, 15]))
    for start in (0, 64, 125):
        positions = torch.arange(start, start + len(hidden)).float()
        expected = model.project(hidden, start)
        actual = model.project_positions(hidden, positions)
        for name in vars(expected):
            torch.testing.assert_close(
                getattr(actual, name).float(), getattr(expected, name).float(), rtol=0, atol=0
            )
    with pytest.raises(ValueError, match="positions"):
        model.project_positions(hidden, torch.arange(len(hidden)))
    with pytest.raises(ValueError, match="positions"):
        model.project_positions(hidden, torch.zeros(len(hidden), 1))


def test_pre_normalized_attention_does_not_apply_input_norm_twice(checkpoint):
    path, _, raw = checkpoint
    model = CheckpointAttention(path, device="cpu")
    hidden = model.embedding(torch.tensor([1, 6, 14]))
    normalized = (
        hidden.float()
        / torch.sqrt(hidden.float().square().mean(-1, keepdim=True) + model.cfg.norm_eps)
        * raw["model.layers.0.input_layernorm.weight"]
    ).bfloat16()
    expected = model.project(hidden, 64)
    actual = model.project(normalized, 64, normalized=True)
    for name, tensor in vars(expected).items():
        torch.testing.assert_close(getattr(actual, name).float(), tensor.float(), rtol=0, atol=0)


def test_unsupported_checkpoint_quantization_fails(checkpoint):
    path, config, _ = checkpoint
    config["quantization_config"]["quant_method"] = "awq"
    (path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="AWQ"):
        Config.from_checkpoint(path)
