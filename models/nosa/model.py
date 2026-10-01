"""NOSA model assembly, inference, and strict checkpoint loading.

Dense and resident NOSA block sparse inference share the same model and cache lifecycle.
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
    ``attention_mode="sparse"`` loads CIS parameters and selects the complete
    NOSA policy, with ``sparse_backend`` choosing reference or Triton arithmetic.
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
        attention_mode="dense",
        sparse_backend="auto",
        cache_backend="resident",
        offload_query_tile_size=128,
        offload_fetch_ctas=96,
        offload_overlap=True,
    ):
        super().__init__()
        self.config = config
        if attention_mode not in ("dense", "sparse"):
            raise ValueError("attention_mode must be dense or sparse")
        if sparse_backend not in ("auto", "reference", "triton"):
            raise ValueError("sparse_backend must be auto, reference or triton")
        if cache_backend not in ("resident", "offload"):
            raise ValueError("cache_backend must be resident or offload")
        if cache_backend == "offload" and attention_mode != "sparse":
            raise ValueError("Offloaded NOSA cache requires attention_mode='sparse'")
        if type(offload_query_tile_size) is not int or offload_query_tile_size <= 0:
            raise ValueError("offload_query_tile_size must be a positive integer")
        if type(offload_fetch_ctas) is not int or offload_fetch_ctas <= 0:
            raise ValueError("offload_fetch_ctas must be a positive integer")
        if type(offload_overlap) is not bool:
            raise ValueError("offload_overlap must be boolean")
        if attention_mode == "sparse" and any(
            value is not None for value in (attention, main_attention, indexer)
        ):
            raise ValueError("sparse mode owns attention and indexer; do not override them")
        self.attention_mode = attention_mode
        self.sparse_backend = sparse_backend
        self.cache_backend = cache_backend
        self.offload_query_tile_size = offload_query_tile_size
        self.offload_fetch_ctas = offload_fetch_ctas
        self.offload_overlap = offload_overlap
        self.attention = FlashInferFullAttention() if attention is None else attention
        self._dense_main_attention = DenseMainAttention(self.attention)
        self.main_attention = (
            self._dense_main_attention if main_attention is None else main_attention
        )
        self.indexer = indexer
        if attention_mode == "sparse":
            from models.nosa.attention import NosaSparseAttention
            from models.nosa.indexer import NosaIndexer

            self.main_attention = NosaSparseAttention(backend=sparse_backend)
            self.indexer = NosaIndexer(mode="nosa", backend=sparse_backend)
        self.model = NosaModel(
            config, device=device, dtype=dtype, with_cis=attention_mode == "sparse"
        )
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
        if self.cache_backend == "offload":
            from models.nosa.offload_cache import NosaOffloadCache

            return NosaOffloadCache(
                self.config,
                max_seq_len,
                device=weight.device,
                dtype=weight.dtype,
                with_cis=True,
                query_tile_size=self.offload_query_tile_size,
                fetch_ctas=self.offload_fetch_ctas,
                overlap=self.offload_overlap,
            )
        return NosaKVCache(
            self.config,
            max_seq_len,
            device=weight.device,
            dtype=weight.dtype,
            with_cis=self.attention_mode == "sparse",
        )

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
                or cache.with_cis != (self.attention_mode == "sparse")
            ):
                raise ValueError(
                    "KV cache must match the model config, device, dtype and CIS layout"
                )
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
            positions, cos_sin_cache = self.model.rotary(
                self.config, start, end, device=weight.device
            )
            x = self.model.embed_tokens(input_ids)
            # Preserve replacements of the legacy attention callable by tests or
            # profilers without allocating another adapter for every forward.
            self._dense_main_attention.attention = self.attention
            residual = None
            for idx, layer in enumerate(self.model.layers):
                x, residual = layer(
                    x,
                    positions,
                    cos_sin_cache,
                    self.main_attention,
                    cache,
                    idx,
                    indexer=self.indexer,
                    residual=residual,
                )
            if logits_to_keep:
                x = x[-logits_to_keep:]
                residual = residual[-logits_to_keep:]
            hidden, _ = self.model.norm(x, residual)
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
        attention_mode="dense",
        sparse_backend="auto",
        cache_backend="resident",
        offload_query_tile_size=128,
        offload_fetch_ctas=96,
        offload_overlap=True,
    ):
        """Strictly load single-file or indexed safetensors, one tensor at a time.

        The dense model ignores NOSA A/delta; sparse mode requires and loads them.
        Original Q/K/V and gate/up tensors are copied into merged parameter
        slices at load time. Runtime packed checkpoints are also accepted.
        Shapes and all core keys are checked before allocating model weights;
        no separate projection weights or packing cache remain after loading.
        """
        from safetensors import safe_open

        model_path = Path(model_path)
        config = NosaConfig.from_pretrained(model_path)
        device = torch.device(device)
        if attention_mode == "dense" and attention is None and main_attention is None:
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
            attention_mode=attention_mode,
            sparse_backend=sparse_backend,
            cache_backend=cache_backend,
            offload_query_tile_size=offload_query_tile_size,
            offload_fetch_ctas=offload_fetch_ctas,
            offload_overlap=offload_overlap,
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
            # Select either one packed tensor or all original component tensors
            # for each parameter. A mixture with duplicate components is rejected
            # by the same strict unexpected-key check as any other extra tensor.
            sources = {}
            source_shapes = {}
            for name, parameter in expected.items():
                module_name, kind = name.rsplit(".", 1)
                parent, _, projection = module_name.rpartition(".")
                if name not in actual and projection in ("qkv_proj", "gate_up_proj"):
                    if projection == "qkv_proj":
                        components = (
                            ("q_proj", config.num_attention_heads * config.head_dim),
                            ("k_proj", config.num_key_value_heads * config.head_dim),
                            ("v_proj", config.num_key_value_heads * config.head_dim),
                        )
                    else:
                        components = (
                            ("gate_proj", config.intermediate_size),
                            ("up_proj", config.intermediate_size),
                        )
                    parts = {
                        f"{parent}.{component}.{kind}": (width, *parameter.shape[1:])
                        for component, width in components
                    }
                else:
                    parts = {name: tuple(parameter.shape)}
                sources[name] = parts
                source_shapes.update(parts)
            missing = source_shapes.keys() - actual.keys()
            unexpected = actual.keys() - source_shapes.keys() - allowed_extra.keys()
            if missing or unexpected:
                raise ValueError(
                    f"Checkpoint key mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
            for name, filename in actual.items():
                shape = tuple(handles[filename].get_slice(name).get_shape())
                wanted = source_shapes[name] if name in source_shapes else allowed_extra[name]
                if shape != wanted:
                    raise ValueError(
                        f"Checkpoint shape mismatch for {name}: {shape}, expected {wanted}"
                    )

            def read_tensor(name):
                tensor = handles[actual[name]].get_tensor(name)
                if not tensor.is_floating_point():
                    raise ValueError(f"Expected floating point checkpoint tensor: {name}")
                return tensor

            for name, parts in sources.items():
                if len(parts) == 1:
                    tensor = read_tensor(name).to(device=device, dtype=dtype)
                else:
                    tensor = torch.empty(expected[name].shape, device=device, dtype=dtype)
                    start = 0
                    for source, shape in parts.items():
                        # Copy directly from the checkpoint into its final
                        # storage, including dtype conversion and device transfer.
                        tensor.narrow(0, start, shape[0]).copy_(read_tensor(source))
                        start += shape[0]
                parent_name, parameter_name = name.rsplit(".", 1)
                model.get_submodule(parent_name).register_parameter(
                    parameter_name, nn.Parameter(tensor, requires_grad=False)
                )
            model.ignored_checkpoint_keys = tuple(sorted(actual.keys() - source_shapes.keys()))
        return model
