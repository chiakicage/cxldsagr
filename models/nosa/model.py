"""NOSA model assembly, inference, and strict checkpoint loading.

The dense backend is runnable; indexer and SM90 offload backends are reserved.
"""

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from torch import nn

from cache.manager import CacheManager
from layers.attention import DenseMainAttention
from models.nosa.cache import NosaKVCache
from models.nosa.config import NosaConfig
from models.nosa.layers import (
    NosaAttention,
    NosaDecoderLayer,
    NosaMLP,
    NosaModel,
    NosaRMSNorm,
)
from models.nosa.rotary import apply_rotary, rotary_cos_sin
from operators.flashinfer import FlashInferFullAttention

# Preserve the original import surface while implementations live at their layer.
__all__ = [
    "FlashInferFullAttention",
    "NosaAttention",
    "NosaConfig",
    "NosaDecoderLayer",
    "NosaForCausalLM",
    "NosaKVCache",
    "NosaMLP",
    "NosaModel",
    "NosaRMSNorm",
    "apply_rotary",
    "rotary_cos_sin",
]


class NosaForCausalLM(nn.Module):
    """Inference-only Llama backbone; forward accepts a 1-D, unpadded token sequence.

    The legacy ``attention(q, k, v)`` callable remains a dense numerical test
    seam. ``indexer`` and ``main_attention`` expose the separate sparse-layer
    boundaries; the default path uses no indexer and FlashInfer full attention.
    """

    def __init__(
        self,
        config: NosaConfig,
        *,
        device="cpu",
        dtype=torch.float32,
        attention=None,
        main_attention=None,
        indexer=None,
    ):
        super().__init__()
        self.config = config
        self.attention = FlashInferFullAttention() if attention is None else attention
        self._dense_main_attention = DenseMainAttention(self.attention)
        self.main_attention = (
            self._dense_main_attention if main_attention is None else main_attention
        )
        self.indexer = indexer
        self.model = NosaModel(config, device=device, dtype=dtype)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype
        )
        self.ignored_checkpoint_keys: tuple[str, ...] = ()
        self.cache_manager = CacheManager(self._allocate_cache)
        self.eval()
        self.requires_grad_(False)

    def _allocate_cache(self, max_seq_len: int) -> NosaKVCache:
        # Read placement now: from_pretrained constructs on meta before loading.
        weight = self.model.embed_tokens.weight
        return NosaKVCache(self.config, max_seq_len, device=weight.device, dtype=weight.dtype)

    def new_cache(self, max_seq_len: int) -> NosaKVCache:
        return self.cache_manager.allocate(max_seq_len)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: NosaKVCache | None = None,
        *,
        logits_to_keep=0,
        return_hidden=False,
    ):
        """Return LM logits, or normalized backbone features for GR forward work.

        ``return_hidden=True`` skips the vocabulary projection and returns every
        input token's final hidden state, while updating the same KV cache.
        """
        if input_ids.ndim != 1 or input_ids.numel() == 0 or input_ids.dtype != torch.long:
            raise ValueError(
                "input_ids must be a nonempty 1-D torch.long tensor (one unpadded request)"
            )
        weight = self.model.embed_tokens.weight
        if input_ids.device != weight.device:
            raise ValueError("input_ids and model must be on the same device")
        if not isinstance(logits_to_keep, int) or logits_to_keep < 0:
            raise ValueError("logits_to_keep must be a nonnegative integer")
        if return_hidden and logits_to_keep:
            raise ValueError("return_hidden returns all input features; do not set logits_to_keep")
        start = 0
        if cache is not None:
            if (
                cache.config != self.config
                or cache.device != weight.device
                or cache.dtype != weight.dtype
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
        if cache is not None:
            cache.begin_step(input_ids.numel())
        try:
            positions = torch.arange(start, end, device=input_ids.device)
            cos, sin = rotary_cos_sin(self.config, positions, weight.dtype)
            x = self.model.embed_tokens(input_ids)
            # Preserve replacements of the legacy attention callable by tests or
            # profilers without allocating another adapter for every forward.
            self._dense_main_attention.attention = self.attention
            for idx, layer in enumerate(self.model.layers):
                x = layer(x, cos, sin, self.main_attention, cache, idx, indexer=self.indexer)
            if logits_to_keep:
                x = x[-logits_to_keep:]
            hidden = self.model.norm(x)
            output = hidden if return_hidden else self.lm_head(hidden).float()
            if cache is not None:
                cache.commit_step()
        except BaseException:
            if cache is not None:
                cache.abort_step()
            raise
        return output

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device="cuda:0",
        dtype=torch.bfloat16,
        attention=None,
        main_attention=None,
        indexer=None,
    ):
        """Strictly load single-file or indexed safetensors, one tensor at a time.

        Only the known NOSA A/delta tensors may be omitted from this dense model.
        Shapes and all core keys are checked before allocating model weights.
        """
        from safetensors import safe_open

        model_path = Path(model_path)
        config = NosaConfig.from_pretrained(model_path)
        device = torch.device(device)
        if attention is None and main_attention is None:
            if device.type != "cuda" or not torch.cuda.is_available():
                raise ValueError("NOSA FlashInfer inference requires an available CUDA device")
            if dtype not in (torch.bfloat16, torch.float16):
                raise ValueError("NOSA FlashInfer inference requires bfloat16 or float16")
        # Meta construction avoids allocating/random-initializing another 8B model.
        model = cls(
            config,
            device="meta",
            dtype=dtype,
            attention=attention,
            main_attention=main_attention,
            indexer=indexer,
        )
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
