"""NOSA decoder composition, projections, and model-specific attention wiring."""

import torch
from torch import nn

from models.attention_contracts import AttentionContext
from models.nosa.attention import ResidentLayerView
from models.nosa.feed_forward import SwiGLU
from models.nosa.normalization import RMSNorm
from models.nosa.rotary import NosaRotaryEmbedding, apply_rotary_qk

# Aliasing keeps both the old import and profiler isinstance checks meaningful.
NosaRMSNorm = RMSNorm


class NosaAttention(nn.Module):
    def __init__(self, config, *, device, dtype, with_cis=False):
        super().__init__()
        self.config = config
        self.with_cis = with_cis
        args = {"device": device, "dtype": dtype, "bias": config.attention_bias}
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim,
            **args,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, **args
        )
        if with_cis:
            self.A = nn.Parameter(
                torch.ones(config.num_key_value_heads, device=device, dtype=dtype)
            )
            self.delta = nn.Linear(
                config.num_key_value_heads * config.head_dim, config.num_key_value_heads, **args
            )

    def project(self, x, positions, cos_sin_cache):
        config = self.config
        q_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        # Split only creates views. RoPE and attention consume the row-strided
        # Q/K/V directly, without packing or materializing contiguous copies.
        q, k, v = self.qkv_proj(x).split((q_width, kv_width, kv_width), dim=-1)
        q = q.view(-1, config.num_attention_heads, config.head_dim)
        k = k.view(-1, config.num_key_value_heads, config.head_dim)
        v = v.view(-1, config.num_key_value_heads, config.head_dim)
        q, k = apply_rotary_qk(q, k, positions, cos_sin_cache)
        records = {"keys": k, "values": v}
        if self.with_cis:
            from models.nosa.scoring import cis_scores

            records["cis_scores"] = cis_scores(v, self.delta.weight, self.A, self.delta.bias)
        return q, records

    def attend(self, q, records, main_attention, cache, layer_idx, *, indexer=None):
        if cache is None:
            cache_access = ResidentLayerView(layer_idx, **records)
            start = 0
        else:
            cache.write_layer(layer_idx, **records)
            cache_access = cache
            start = cache.length
        context = AttentionContext(
            layer_idx=layer_idx,
            query_start=start,
            query_length=q.shape[0],
            auxiliary_state=cache_access.get_layer_state(layer_idx),
        )
        selection = None if indexer is None else indexer(q, cache_access, context)
        return main_attention(q, selection, cache_access, context)

    def forward(
        self, x, positions, cos_sin_cache, main_attention, cache, layer_idx, *, indexer=None
    ):
        q, records = self.project(x, positions, cos_sin_cache)
        attended = self.attend(q, records, main_attention, cache, layer_idx, indexer=indexer)
        return self.o_proj(attended.reshape(x.shape[0], -1))


class NosaMLP(SwiGLU):
    def __init__(self, config, *, device, dtype):
        super().__init__(
            config.hidden_size,
            config.intermediate_size,
            bias=config.mlp_bias,
            device=device,
            dtype=dtype,
        )


class NosaDecoderLayer(nn.Module):
    def __init__(self, config, *, device, dtype, with_cis=False):
        super().__init__()
        args = {"device": device, "dtype": dtype}
        self.self_attn = NosaAttention(config, with_cis=with_cis, **args)
        self.mlp = NosaMLP(config, **args)
        self.input_layernorm = NosaRMSNorm(config.hidden_size, config.rms_norm_eps, **args)
        self.post_attention_layernorm = NosaRMSNorm(config.hidden_size, config.rms_norm_eps, **args)

    def forward(
        self,
        x,
        positions,
        cos_sin_cache,
        main_attention,
        cache,
        layer_idx,
        *,
        indexer=None,
        residual=None,
    ):
        # Carry the pending MLP output separately across layers so its addition
        # can share the next RMSNorm kernel. The first layer starts at embedding.
        if residual is None:
            residual = x
            x = self.input_layernorm(x)
        else:
            x, residual = self.input_layernorm(x, residual)
        x = self.self_attn(
            x, positions, cos_sin_cache, main_attention, cache, layer_idx, indexer=indexer
        )
        x, residual = self.post_attention_layernorm(x, residual)
        return self.mlp(x), residual


class NosaModel(nn.Module):
    def __init__(self, config, *, device, dtype, with_cis=False):
        super().__init__()
        args = {"device": device, "dtype": dtype}
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, **args)
        self.layers = nn.ModuleList(
            NosaDecoderLayer(config, with_cis=with_cis, **args)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = NosaRMSNorm(config.hidden_size, config.rms_norm_eps, **args)
        self.rotary = NosaRotaryEmbedding()
