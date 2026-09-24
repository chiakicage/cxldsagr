"""NOSA's Llama backbone with ordinary FlashInfer causal full attention.

The NOSA sparse/CIS branch is intentionally disabled, matching the checkpoint's
upstream eager/flash_attention_2 branches. No checkpoint Python code is executed.
"""

import json
import math
from contextlib import ExitStack
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NosaConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None
    bos_token_id: int = 1
    eos_token_id: tuple[int, ...] = (2,)
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self):
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        if self.tie_word_embeddings:
            raise ValueError("This NOSA runner requires independent embedding and lm_head weights")
        for name in ("rms_norm_eps", "rope_theta"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        eos = self.eos_token_id
        object.__setattr__(self, "eos_token_id", (eos,) if isinstance(eos, int) else tuple(eos))
        if any(token < 0 or token >= self.vocab_size for token in self.eos_token_id):
            raise ValueError("eos_token_id is outside the vocabulary")
        scaling = self.rope_scaling
        if scaling is not None:
            if scaling.get("rope_type", scaling.get("type")) != "longrope":
                raise ValueError("Only default RoPE and NOSA's static LongRoPE are supported")
            for name in ("short_factor", "long_factor"):
                factors = scaling.get(name, [])
                if len(factors) != self.head_dim // 2 or any(
                    not math.isfinite(x) or x <= 0 for x in factors
                ):
                    raise ValueError(f"LongRoPE {name} needs head_dim / 2 positive finite values")
            # NOSA-8B uses identical arrays. Changing frequencies after caching K
            # would invalidate that cache; do not silently generalize this policy.
            if list(scaling["short_factor"]) != list(scaling["long_factor"]):
                raise ValueError("NOSA static LongRoPE requires equal short_factor and long_factor")
            attention_factor = scaling.get("attention_factor")
            if (
                attention_factor is None
                or not math.isfinite(attention_factor)
                or attention_factor <= 0
            ):
                raise ValueError("NOSA LongRoPE requires an explicit positive attention_factor")

    @classmethod
    def from_dict(cls, data: dict):
        if data.get("model_type", "llama") != "llama":
            raise ValueError("Expected a NOSA Llama checkpoint")
        if data.get("hidden_act", "silu") != "silu":
            raise ValueError("Only the NOSA SwiGLU (silu) MLP is supported")
        if data.get("pretraining_tp", 1) != 1:
            raise ValueError("pretraining_tp must be 1")
        values = {field.name: data[field.name] for field in fields(cls) if field.name in data}
        values.setdefault("head_dim", data["hidden_size"] // data["num_attention_heads"])
        values.setdefault("num_key_value_heads", data["num_attention_heads"])
        return cls(**values)

    @classmethod
    def from_pretrained(cls, model_path: str | Path):
        return cls.from_dict(json.loads((Path(model_path) / "config.json").read_text()))


class FlashInferFullAttention:
    """Single-request GQA over NHD K/V, already rotated by model LongRoPE."""

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if q.device.type != "cuda" or q.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("FlashInfer attention requires CUDA and bfloat16 or float16 tensors")
        import flashinfer

        with torch.cuda.device(q.device):
            if q.shape[0] == 1:
                return flashinfer.single_decode_with_kv_cache(
                    q[0],
                    k,
                    v,
                    kv_layout="NHD",
                    pos_encoding_mode="NONE",
                    use_tensor_cores=True,
                    sm_scale=q.shape[-1] ** -0.5,
                ).unsqueeze(0)
            # FlashInfer's causal mask aligns Q to the end of K. This also
            # handles a prefill chunk appended to an existing prefix cache.
            return flashinfer.single_prefill_with_kv_cache(
                q,
                k,
                v,
                causal=True,
                kv_layout="NHD",
                pos_encoding_mode="NONE",
                sm_scale=q.shape[-1] ** -0.5,
                backend="auto",
            )


class NosaKVCache:
    """Preallocated per-layer K/V on the model device, one unpadded request.

    K stores post-RoPE keys. Each tensor is [layer, capacity, kv_head, head_dim];
    attention sees a contiguous NHD prefix without repeating the GQA heads.
    """

    def __init__(self, config: NosaConfig, max_seq_len: int, *, device, dtype):
        if not 0 < max_seq_len <= config.max_position_embeddings:
            raise ValueError("Cache capacity must be within max_position_embeddings")
        self.config = config
        self.max_seq_len = max_seq_len
        self.length = 0
        shape = (config.num_hidden_layers, max_seq_len, config.num_key_value_heads, config.head_dim)
        self.keys = torch.empty(shape, device=device, dtype=dtype)
        self.values = torch.empty_like(self.keys)

    def reset(self):
        """Discard the prefix; subsequent calls overwrite it without reallocating."""
        self.length = 0


def rotary_cos_sin(config: NosaConfig, positions: torch.Tensor, dtype: torch.dtype):
    """FP32 LongRoPE phases, then cast before applying Llama split-half rotation."""
    exponent = (
        torch.arange(0, config.head_dim, 2, device=positions.device).float() / config.head_dim
    )
    factors = 1.0
    attention_factor = 1.0
    if config.rope_scaling is not None:
        factors = torch.tensor(config.rope_scaling["short_factor"], device=positions.device)
        attention_factor = config.rope_scaling["attention_factor"]
    inv_freq = 1.0 / (factors * config.rope_theta**exponent)
    angles = positions.float()[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return (angles.cos() * attention_factor).to(dtype), (angles.sin() * attention_factor).to(dtype)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    first, second = x.chunk(2, dim=-1)
    return x * cos[:, None, :] + torch.cat((-second, first), dim=-1) * sin[:, None, :]


class NosaRMSNorm(nn.Module):
    def __init__(self, size, eps, *, device, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(x.dtype)


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

    def forward(self, x, cos, sin, attention, cache, layer_idx):
        config = self.config
        q = self.q_proj(x).view(-1, config.num_attention_heads, config.head_dim)
        k = self.k_proj(x).view(-1, config.num_key_value_heads, config.head_dim)
        v = self.v_proj(x).view(-1, config.num_key_value_heads, config.head_dim)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        if cache is not None:
            start, end = cache.length, cache.length + x.shape[0]
            cache.keys[layer_idx, start:end].copy_(k)
            cache.values[layer_idx, start:end].copy_(v)
            k, v = cache.keys[layer_idx, :end], cache.values[layer_idx, :end]
        return self.o_proj(attention(q, k, v).reshape(x.shape[0], -1))


class NosaMLP(nn.Module):
    def __init__(self, config, *, device, dtype):
        super().__init__()
        args = {"device": device, "dtype": dtype, "bias": config.mlp_bias}
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, **args)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, **args)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, **args)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class NosaDecoderLayer(nn.Module):
    def __init__(self, config, *, device, dtype):
        super().__init__()
        args = {"device": device, "dtype": dtype}
        self.self_attn = NosaAttention(config, **args)
        self.mlp = NosaMLP(config, **args)
        self.input_layernorm = NosaRMSNorm(config.hidden_size, config.rms_norm_eps, **args)
        self.post_attention_layernorm = NosaRMSNorm(config.hidden_size, config.rms_norm_eps, **args)

    def forward(self, x, cos, sin, attention, cache, layer_idx):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attention, cache, layer_idx)
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


class NosaForCausalLM(nn.Module):
    """Inference-only Llama backbone; forward accepts a 1-D, unpadded token sequence.

    The optional attention callable is a test seam for independent dense numerical
    references. Production inference always defaults to FlashInferFullAttention.
    """

    def __init__(self, config: NosaConfig, *, device="cpu", dtype=torch.float32, attention=None):
        super().__init__()
        self.config = config
        self.attention = FlashInferFullAttention() if attention is None else attention
        self.model = NosaModel(config, device=device, dtype=dtype)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype
        )
        self.ignored_checkpoint_keys: tuple[str, ...] = ()
        self.eval()
        self.requires_grad_(False)

    def new_cache(self, max_seq_len: int) -> NosaKVCache:
        weight = self.model.embed_tokens.weight
        return NosaKVCache(self.config, max_seq_len, device=weight.device, dtype=weight.dtype)

    @torch.inference_mode()
    def forward(
        self, input_ids: torch.Tensor, cache: NosaKVCache | None = None, *, logits_to_keep=0
    ):
        if input_ids.ndim != 1 or input_ids.numel() == 0 or input_ids.dtype != torch.long:
            raise ValueError(
                "input_ids must be a nonempty 1-D torch.long tensor (one unpadded request)"
            )
        weight = self.model.embed_tokens.weight
        if input_ids.device != weight.device:
            raise ValueError("input_ids and model must be on the same device")
        if not isinstance(logits_to_keep, int) or logits_to_keep < 0:
            raise ValueError("logits_to_keep must be a nonnegative integer")
        start = 0
        if cache is not None:
            if (
                cache.config != self.config
                or cache.keys.device != weight.device
                or cache.keys.dtype != weight.dtype
            ):
                raise ValueError("KV cache must match the model config, device and dtype")
            if not 0 <= cache.length <= cache.max_seq_len:
                raise ValueError("Invalid KV cache length")
            start = cache.length
            if start + input_ids.numel() > cache.max_seq_len:
                raise ValueError("KV cache capacity exceeded")
        end = start + input_ids.numel()
        if end > self.config.max_position_embeddings:
            raise ValueError("Sequence exceeds config.max_position_embeddings")
        positions = torch.arange(start, end, device=input_ids.device)
        cos, sin = rotary_cos_sin(self.config, positions, weight.dtype)
        x = self.model.embed_tokens(input_ids)
        for idx, layer in enumerate(self.model.layers):
            x = layer(x, cos, sin, self.attention, cache, idx)
        if logits_to_keep:
            x = x[-logits_to_keep:]
        logits = self.lm_head(self.model.norm(x)).float()
        if cache is not None:
            cache.length = end
        return logits

    @classmethod
    def from_pretrained(
        cls, model_path: str | Path, *, device="cuda:0", dtype=torch.bfloat16, attention=None
    ):
        """Strictly load single-file or indexed safetensors, one tensor at a time.

        Only the known NOSA A/delta tensors may be omitted from this dense model.
        Shapes and all core keys are checked before allocating model weights.
        """
        from safetensors import safe_open

        model_path = Path(model_path)
        config = NosaConfig.from_pretrained(model_path)
        device = torch.device(device)
        if attention is None:
            if device.type != "cuda" or not torch.cuda.is_available():
                raise ValueError("NOSA FlashInfer inference requires an available CUDA device")
            if dtype not in (torch.bfloat16, torch.float16):
                raise ValueError("NOSA FlashInfer inference requires bfloat16 or float16")
        # Meta construction avoids allocating/random-initializing another 8B model.
        model = cls(config, device="meta", dtype=dtype, attention=attention)
        expected = dict(model.named_parameters())
        allowed_extra = {}
        for idx in range(config.num_hidden_layers):
            prefix = f"model.layers.{idx}.self_attn."
            allowed_extra[prefix + "A"] = (config.num_key_value_heads,)
            allowed_extra[prefix + "delta.weight"] = (
                config.num_key_value_heads,
                config.num_key_value_heads * config.head_dim,
            )
            if config.attention_bias:
                allowed_extra[prefix + "delta.bias"] = (config.num_key_value_heads,)

        index_path = model_path / "model.safetensors.index.json"
        weight_map = None
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text())["weight_map"]
            filenames = sorted(set(weight_map.values()))
        else:
            filenames = ["model.safetensors"]
        with ExitStack() as stack:
            handles = {}
            for filename in filenames:
                path = model_path / filename
                if not path.resolve().is_relative_to(model_path.resolve()):
                    raise ValueError(f"Checkpoint shard must be inside model directory: {filename}")
                handles[filename] = stack.enter_context(
                    safe_open(path, framework="pt", device="cpu")
                )
            actual = {}
            for filename, handle in handles.items():
                for name in handle.keys():  # noqa: SIM118 -- safe_open is not a mapping
                    if name in actual:
                        raise ValueError(f"Duplicate checkpoint tensor: {name}")
                    actual[name] = filename
            if weight_map is not None and actual != weight_map:
                raise ValueError("Safetensors index does not match the tensors in its shards")
            missing = expected.keys() - actual.keys()
            unexpected = actual.keys() - expected.keys() - allowed_extra.keys()
            if missing or unexpected:
                raise ValueError(
                    f"Checkpoint key mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
            for name, filename in actual.items():
                shape = tuple(handles[filename].get_slice(name).get_shape())
                wanted = tuple(expected[name].shape) if name in expected else allowed_extra[name]
                if shape != wanted:
                    raise ValueError(
                        f"Checkpoint shape mismatch for {name}: {shape}, expected {wanted}"
                    )
            for name in expected:
                tensor = handles[actual[name]].get_tensor(name)
                if not tensor.is_floating_point():
                    raise ValueError(f"Expected floating point checkpoint tensor: {name}")
                tensor = tensor.to(device=device, dtype=dtype)
                parent_name, parameter_name = name.rsplit(".", 1)
                model.get_submodule(parent_name).register_parameter(
                    parameter_name, nn.Parameter(tensor, requires_grad=False)
                )
            model.ignored_checkpoint_keys = tuple(sorted(actual.keys() - expected.keys()))
        return model
