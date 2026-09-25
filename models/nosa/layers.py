"""NOSA decoder composition, projections, and model-specific attention wiring."""

from torch import nn

from layers.attention import AttentionContext, ResidentLayerView
from layers.feed_forward import SwiGLU
from layers.normalization import RMSNorm
from models.nosa.rotary import apply_rotary

# Aliasing keeps both the old import and profiler isinstance checks meaningful.
NosaRMSNorm = RMSNorm


class NosaAttention(nn.Module):
    def __init__(self, config, *, device, dtype):
        super().__init__()
        self.config = config
        args = {"device": device, "dtype": dtype, "bias": config.attention_bias}
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim, **args
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, **args
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, **args
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, **args
        )

    def forward(self, x, cos, sin, main_attention, cache, layer_idx, *, indexer=None):
        config = self.config
        q = self.q_proj(x).view(-1, config.num_attention_heads, config.head_dim)
        k = self.k_proj(x).view(-1, config.num_key_value_heads, config.head_dim)
        v = self.v_proj(x).view(-1, config.num_key_value_heads, config.head_dim)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        if cache is None:
            cache_access = ResidentLayerView(layer_idx, keys=k, values=v)
            start = 0
        else:
            cache.write_layer(layer_idx, keys=k, values=v)
            cache_access = cache
            start = cache.length
        context = AttentionContext(
            layer_idx=layer_idx,
            query_start=start,
            query_length=x.shape[0],
            auxiliary_state=cache_access.get_layer_state(layer_idx),
        )
        selection = None if indexer is None else indexer(q, cache_access, context)
        attended = main_attention(q, selection, cache_access, context)
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
    def __init__(self, config, *, device, dtype):
        super().__init__()
        args = {"device": device, "dtype": dtype}
        self.self_attn = NosaAttention(config, **args)
        self.mlp = NosaMLP(config, **args)
        self.input_layernorm = NosaRMSNorm(config.hidden_size, config.rms_norm_eps, **args)
        self.post_attention_layernorm = NosaRMSNorm(config.hidden_size, config.rms_norm_eps, **args)

    def forward(self, x, cos, sin, main_attention, cache, layer_idx, *, indexer=None):
        x = x + self.self_attn(
            self.input_layernorm(x), cos, sin, main_attention, cache, layer_idx, indexer=indexer
        )
        return x + self.mlp(self.post_attention_layernorm(x))


class NosaModel(nn.Module):
    def __init__(self, config, *, device, dtype):
        super().__init__()
        args = {"device": device, "dtype": dtype}
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, **args)
        self.layers = nn.ModuleList(
            NosaDecoderLayer(config, **args) for _ in range(config.num_hidden_layers)
        )
        self.norm = NosaRMSNorm(config.hidden_size, config.rms_norm_eps, **args)
