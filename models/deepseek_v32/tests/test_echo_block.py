import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import models.deepseek_v32.layers as block_module
from models.deepseek_v32.checkpoint import CheckpointReader
from models.deepseek_v32.config import Config
from models.deepseek_v32.layers import CheckpointBlock, CheckpointMoE, route_experts


@pytest.fixture
def block_checkpoint(tmp_path):
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
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "intermediate_size": 12,
        "moe_intermediate_size": 6,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 2,
        "topk_group": 1,
        "norm_topk_prob": True,
        "scoring_func": "sigmoid",
        "routed_scaling_factor": 2.5,
        "first_k_dense_replace": 1,
        "quantization_config": {
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(91)
    raw = {"model.embed_tokens.weight": torch.randn(16, 8, generator=generator).bfloat16()}

    def linear(stem, rows, columns):
        raw[stem + ".weight"] = (torch.randn(rows, columns, generator=generator) * 4).to(
            torch.float8_e4m3fn
        )
        raw[stem + ".weight_scale_inv"] = torch.tensor([[0.0625]])

    def mlp(stem, intermediate):
        linear(stem + ".gate_proj", intermediate, 8)
        linear(stem + ".up_proj", intermediate, 8)
        linear(stem + ".down_proj", 8, intermediate)

    for layer in range(2):
        stem = f"model.layers.{layer}."
        attn = stem + "self_attn."
        for name in (
            stem + "input_layernorm.weight",
            stem + "post_attention_layernorm.weight",
            attn + "q_a_layernorm.weight",
            attn + "kv_a_layernorm.weight",
            attn + "indexer.k_norm.weight",
        ):
            raw[name] = torch.rand(8, generator=generator) * 0.3 + 0.9
        raw[attn + "indexer.k_norm.bias"] = torch.randn(8, generator=generator) * 0.1
        raw[attn + "indexer.weights_proj.weight"] = torch.randn(
            2, 8, generator=generator
        ).bfloat16()
        for name, rows, columns in (
            ("q_a_proj", 8, 8),
            ("q_b_proj", 16, 8),
            ("kv_a_proj_with_mqa", 12, 8),
            ("kv_b_proj", 16, 8),
            ("o_proj", 8, 8),
            ("indexer.wq_b", 16, 8),
            ("indexer.wk", 8, 8),
        ):
            linear(attn + name, rows, columns)
        if layer == 0:
            mlp(stem + "mlp", 12)
        else:
            raw[stem + "mlp.gate.weight"] = torch.randn(4, 8, generator=generator).bfloat16()
            raw[stem + "mlp.gate.e_score_correction_bias"] = torch.tensor([0.1, -0.2, 0.3, 0.0])
            for expert in range(4):
                mlp(stem + f"mlp.experts.{expert}", 6)
            mlp(stem + "mlp.shared_experts", 6)
    save_file(raw, tmp_path / "model.safetensors")
    return tmp_path, raw


class CPUAttentionOracle:
    """Test-only causal attention using dense FP32 scores and selected records."""

    def __init__(self, attention, capacity, **kwargs):
        self.attention = attention
        self.cfg = attention.cfg
        self.cache = SimpleNamespace(written=0)
        self.capacity = capacity
        self.records = torch.empty(0, self.cfg.qk_head_dim, dtype=torch.bfloat16)
        self.keys = torch.empty(0, self.cfg.index_head_dim)
        self.scales = torch.empty(0)

    def forward(self, hidden, scope=None, normalized=False):
        p = self.attention.project(hidden, self.cache.written, normalized=normalized)
        self.records = torch.cat((self.records, p.kv))
        self.keys = torch.cat((self.keys, p.index_k.float()))
        self.scales = torch.cat((self.scales, p.index_scale))
        dots = torch.einsum("thd,sd->ths", p.index_q.float(), self.keys).relu()
        logits = (dots * p.index_weights[..., None]).sum(1) * self.scales
        result = []
        for query in range(len(hidden)):
            end = self.cache.written + query + 1
            ids = logits[query, :end].topk(min(end, self.cfg.index_topk)).indices
            records = self.records[ids].float()
            scores = p.q[query].float() @ records.T * self.cfg.attention_scale
            result.append((scores.softmax(-1) @ records[:, : self.cfg.kv_lora_rank]).bfloat16())
        self.cache.written += len(hidden)
        return self.attention.output(torch.stack(result))


def linear_reference(hidden, raw, stem):
    # Fixtures fit one block, so scalar powers-of-two provide an independent quantizer.
    rows = hidden.float().reshape(-1, hidden.shape[-1])
    quantized = []
    for row in rows:
        scale = 2.0 ** math.ceil(math.log2(max(float(row.abs().max()), 1e-4) / 448))
        quantized.append((row / scale).to(torch.float8_e4m3fn).float() * scale)
    activation = torch.stack(quantized)
    weight = raw[stem + ".weight"].float() * raw[stem + ".weight_scale_inv"][0, 0]
    return (activation @ weight.T).reshape(*hidden.shape[:-1], len(weight)).bfloat16()


def mlp_reference(hidden, raw, stem):
    gate = linear_reference(hidden, raw, stem + ".gate_proj")
    up = linear_reference(hidden, raw, stem + ".up_proj")
    intermediate = (gate.float() * gate.float().sigmoid() * up.float()).bfloat16()
    return linear_reference(intermediate, raw, stem + ".down_proj")


def routing_reference(hidden, gate, bias, cfg):
    raw_scores = (hidden.float() @ gate.float().T).sigmoid()
    ids, weights = [], []
    for row in raw_scores:
        corrected = row + bias
        per_group = cfg.n_routed_experts // cfg.n_group
        groups = [list(range(g * per_group, (g + 1) * per_group)) for g in range(cfg.n_group)]
        group_scores = [
            sum(sorted([float(corrected[i]) for i in group], reverse=True)[:2]) for group in groups
        ]
        retained_groups = sorted(range(cfg.n_group), key=lambda i: group_scores[i], reverse=True)[
            : cfg.topk_group
        ]
        candidates = [i for group in retained_groups for i in groups[group]]
        selected = sorted(candidates, key=lambda i: float(corrected[i]), reverse=True)[
            : cfg.num_experts_per_tok
        ]
        values = row[selected]
        if cfg.norm_topk_prob:
            values = values / values.sum()
        ids.append(selected)
        weights.append(values * cfg.routed_scaling_factor)
    return torch.tensor(ids), torch.stack(weights)


def moe_reference(hidden, raw, cfg, stem):
    ids, weights = routing_reference(
        hidden, raw[stem + ".gate.weight"], raw[stem + ".gate.e_score_correction_bias"], cfg
    )
    result = torch.zeros_like(hidden, dtype=torch.float32)
    for token in range(len(hidden)):
        for choice, expert in enumerate(ids[token].tolist()):
            expert_output = mlp_reference(
                hidden[token : token + 1], raw, stem + f".experts.{expert}"
            )
            result[token] += expert_output[0].float() * weights[token, choice]
    result += mlp_reference(hidden, raw, stem + ".shared_experts").float()
    return result.bfloat16()


def block_reference(block, hidden, residual, raw):
    """Official two-stream flow: norm sees each FP32 sum before BF16 rounding."""
    layer = block.layer_idx
    stem = f"model.layers.{layer}."

    def norm(value, name):
        return (
            value
            / torch.sqrt(value.square().mean(-1, keepdim=True) + block.cfg.norm_eps)
            * raw[stem + name + ".weight"]
        ).bfloat16()

    summed = hidden.float() if residual is None else hidden.float() + residual.float()
    normalized = norm(summed, "input_layernorm")
    saved = hidden if residual is None else summed.bfloat16()
    attention = CPUAttentionOracle(block.attention.attention, 128).forward(
        normalized, normalized=True
    )
    summed = attention.float() + saved.float()
    normalized = norm(summed, "post_attention_layernorm")
    output = (
        moe_reference(normalized, raw, block.cfg, stem + "mlp")
        if block.is_moe
        else mlp_reference(normalized, raw, stem + "mlp")
    )
    return output, summed.bfloat16()


def test_routing_uses_top_two_per_group_and_unbiased_weights():
    cfg = replace(
        Config(), dim=6, n_routed_experts=6, n_group=2, topk_group=1, num_experts_per_tok=2
    )
    probabilities = torch.tensor([0.9, 0.01, 0.01, 0.6, 0.55, 0.05])
    gate = torch.diag(torch.logit(probabilities))
    hidden = torch.ones(1, 6).bfloat16()
    ids, weights = route_experts(hidden, gate, torch.zeros(6), cfg)
    assert ids.tolist() == [[3, 4]]  # Group 1 wins the sum despite group's 0 larger maximum.
    torch.testing.assert_close(weights, torch.tensor([[0.6, 0.55]]) / 1.15 * 2.5)
    bias = torch.tensor([0, 1.0, 0, 0, 0, 0])
    ids, weights = route_experts(hidden, gate, bias, cfg)
    assert ids.tolist() == [[1, 0]]
    torch.testing.assert_close(weights, torch.tensor([[0.01, 0.9]]) / 0.91 * 2.5)


def test_checkpoint_grouped_moe_matches_per_expert_oracle(block_checkpoint):
    path, raw = block_checkpoint
    cfg = Config.from_checkpoint(path)
    moe = CheckpointMoE(CheckpointReader(path), "model.layers.1.mlp", cfg, device="cpu")
    assert not moe.shared.pack_gate_up
    hidden = raw["model.embed_tokens.weight"][[0, 5, 9, 13]]
    expected = moe_reference(hidden, raw, cfg, "model.layers.1.mlp")
    torch.testing.assert_close(moe(hidden), expected, rtol=0, atol=0)
    for name, weights in moe.weights.items():
        assert weights.dtype == torch.float8_e4m3fn
        for expert in range(4):
            torch.testing.assert_close(
                weights[expert].float(),
                raw[f"model.layers.1.mlp.experts.{expert}.{name}.weight"].float(),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("chunk_size", [1, 3, 5, 8])
@pytest.mark.parametrize("with_residual", [False, True])
def test_full_block_cpu_oracle_and_bounded_mlp_chunks(
    block_checkpoint, monkeypatch, layer, chunk_size, with_residual
):
    path, raw = block_checkpoint
    monkeypatch.setattr(block_module, "EchoAttentionRunner", CPUAttentionOracle)
    block = CheckpointBlock(path, layer, "cpu", capacity=128, chunk_size=chunk_size)
    if not block.is_moe:
        assert block.mlp.pack_gate_up
        assert not block.mlp._can_pack_gate_up(torch.empty(1, 8).bfloat16())
    hidden = raw["model.embed_tokens.weight"][[1, 3, 8, 12, 15]]
    residual = raw["model.embed_tokens.weight"][[2, 4, 9, 13, 14]] if with_residual else None
    original_hidden = hidden.clone()
    original_residual = None if residual is None else residual.clone()
    expected = block_reference(block, hidden, residual, raw)
    lengths = []
    original = block.mlp

    def recording_mlp(x, **kwargs):
        lengths.append(len(x))
        return original(x, **kwargs)

    block.mlp = recording_mlp
    torch.testing.assert_close(block.forward(hidden, residual), expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden, original_hidden, rtol=0, atol=0)
    if residual is not None:
        torch.testing.assert_close(residual, original_residual, rtol=0, atol=0)
    assert sum(lengths) == len(hidden) and max(lengths) <= chunk_size
    assert block.cache is block.attention.cache


def test_two_blocks_preserve_separate_residual_until_final_norm(block_checkpoint, monkeypatch):
    path, raw = block_checkpoint
    monkeypatch.setattr(block_module, "EchoAttentionRunner", CPUAttentionOracle)
    reader = CheckpointReader(path)
    blocks = [CheckpointBlock(path, i, "cpu", reader, capacity=128, chunk_size=2) for i in range(2)]
    hidden = raw["model.embed_tokens.weight"][[0, 2, 7]]
    reference_hidden = hidden
    residual = reference_residual = None
    for block in blocks:
        reference_hidden, reference_residual = block_reference(
            block, reference_hidden, reference_residual, raw
        )
        hidden, residual = block.forward(hidden, residual)
        torch.testing.assert_close(hidden, reference_hidden, rtol=0, atol=0)
        torch.testing.assert_close(residual, reference_residual, rtol=0, atol=0)
    combined = hidden.float() + residual.float()
    # A single-stream interface would round this sum before the final norm.
    assert not torch.equal(combined, combined.bfloat16().float())
    torch.testing.assert_close(
        combined, reference_hidden.float() + reference_residual.float(), rtol=0, atol=0
    )


def test_detached_block_uses_explicit_runner_without_binding_session_state(
    block_checkpoint, monkeypatch
):
    path, raw = block_checkpoint
    monkeypatch.setattr(block_module, "EchoAttentionRunner", CPUAttentionOracle)
    owned = CheckpointBlock(path, 0, "cpu", capacity=128, chunk_size=2)
    detached = CheckpointBlock(path, 0, "cpu", capacity=128, chunk_size=2, allocate_cache=False)
    runner = CPUAttentionOracle(detached.attention_layer, 128)
    hidden = raw["model.embed_tokens.weight"][[0, 2, 7]]
    expected = owned.forward(hidden)
    actual = detached.forward(hidden, attention=runner, chunk_size=len(hidden))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert detached.attention is None and detached.cache is None
    assert detached.chunk_size == 2
    with pytest.raises(RuntimeError, match="explicit attention runner"):
        detached.forward(hidden)


def test_incomplete_expert_checkpoint_fails_explicitly(block_checkpoint):
    path, raw = block_checkpoint
    del raw["model.layers.1.mlp.experts.3.down_proj.weight"]
    save_file(raw, path / "model.safetensors")
    with pytest.raises(KeyError, match="Required tensor.*experts.3.down_proj"):
        CheckpointMoE(
            CheckpointReader(path), "model.layers.1.mlp", Config.from_checkpoint(path), device="cpu"
        )
