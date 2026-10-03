"""Strict reduced-checkpoint loading and a hidden-only ECHO model adapter.

Metadata inspection uses only the standard library. ECHO, Torch and safetensors
are imported inside scoped_echo_adapter, from the pinned checkout. The adapter
reuses upstream decoder math; it does not implement attention or serving.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from uuid import uuid4

from models.deepseek_v32 import echo_cache

ECHO_REVISION = "bc1b75c1000010d0ac6f032ebaac283255c050b1"
ARCHITECTURE = "DeepseekV32ForCausalLM"
RUNTIME_ARCHITECTURES = (ARCHITECTURE, "DeepseekV3ForCausalLM")
PACKED_MODULES_MAPPING = {
    "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
}


class AdapterError(ValueError):
    """Unsupported metadata, runtime configuration, or source provenance."""


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtypes: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointSelection:
    model_path: Path
    num_layers: int
    config: dict[str, Any]
    selected_names: tuple[str, ...]
    shards: dict[str, tuple[str, ...]]
    tensor_specs: dict[str, TensorSpec]
    metadata_sha256: dict[str, str]
    shard_stats: dict[str, tuple[int, int]]
    metadata_fingerprint: str


@dataclass(frozen=True)
class EchoAdapterBindings:
    loader_class: type
    model_class: type
    selection: CheckpointSelection
    echo_revision: str
    echo_patch_sha256: str
    model_instance_id: str
    cache_adaptations: tuple[str, ...]
    cache_adapter_sha256: str


def _object_without_duplicates(pairs):
    value = {}
    for name, item in pairs:
        if name in value:
            raise AdapterError(f"duplicate metadata key: {name}")
        value[name] = item
    return value


def _read_metadata(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=_object_without_duplicates)
    if not isinstance(value, dict):
        raise AdapterError(f"expected metadata object: {path}")
    return value, hashlib.sha256(raw).hexdigest()


def _positive(config: Mapping, key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise AdapterError(f"config {key} must be a positive integer")
    return value


def validate_checkpoint_config(config: Mapping, num_layers: int) -> None:
    if type(num_layers) is not int or not 1 <= num_layers <= 5:
        raise AdapterError("num_layers must be between 1 and 5")
    if config.get("model_type") != "deepseek_v32" or config.get("architectures") != [ARCHITECTURE]:
        raise AdapterError("expected DeepSeek V3.2 CausalLM checkpoint architecture")
    if num_layers > _positive(config, "num_hidden_layers"):
        raise AdapterError("selected layers exceed the checkpoint")
    if num_layers > _positive(config, "first_k_dense_replace"):
        validate_moe_config(config)
    for name in (
        "hidden_size",
        "vocab_size",
        "q_lora_rank",
        "num_attention_heads",
        "qk_nope_head_dim",
        "v_head_dim",
        "intermediate_size",
        "index_n_heads",
        "index_topk",
        "max_position_embeddings",
    ):
        _positive(config, name)
    for name, expected in (
        ("kv_lora_rank", 512),
        ("qk_rope_head_dim", 64),
        ("index_head_dim", 128),
    ):
        if config.get(name) != expected:
            raise AdapterError(f"ECHO adapter requires {name}={expected}")
    if config.get("attention_bias", False) or config.get("hidden_act") != "silu":
        raise AdapterError("expected bias-free attention and SiLU dense MLP")
    quant = config.get("quantization_config")
    if (
        not isinstance(quant, dict)
        or quant.get("quant_method") != "fp8"
        or quant.get("weight_block_size") != [128, 128]
    ):
        raise AdapterError("expected an FP8 checkpoint with 128x128 weight scales")
    if config.get("torch_dtype") != "bfloat16":
        raise AdapterError("expected BF16 activation dtype in checkpoint config")


def validate_moe_config(config: Mapping) -> None:
    for name in (
        "moe_intermediate_size",
        "n_routed_experts",
        "n_shared_experts",
        "moe_layer_freq",
        "num_experts_per_tok",
        "n_group",
        "topk_group",
    ):
        _positive(config, name)
    if (
        config.get("topk_method") != "noaux_tc"
        or config.get("scoring_func") != "sigmoid"
        or config.get("norm_topk_prob") is not True
        or not isinstance(config.get("routed_scaling_factor"), (int, float))
        or config["routed_scaling_factor"] <= 0
        or config["n_routed_experts"] % config["n_group"]
        or config["topk_group"] > config["n_group"]
        or config["num_experts_per_tok"]
        > config["topk_group"] * config["n_routed_experts"] // config["n_group"]
    ):
        raise AdapterError("unsupported or inconsistent grouped MoE routing configuration")


def layer_mlp_shapes(config: Mapping, layer: int) -> tuple[dict, dict]:
    """Checkpoint projections and unquantized router tensors, shared with the audit."""
    h = config["hidden_size"]
    quantized, plain = {}, {}
    if layer < config["first_k_dense_replace"]:
        groups = {"mlp": config["intermediate_size"]}
    else:
        validate_moe_config(config)
        if layer % config["moe_layer_freq"]:
            groups = {"mlp": config["intermediate_size"]}
        else:
            experts = config["n_routed_experts"]
            groups = {f"mlp.experts.{i}": config["moe_intermediate_size"] for i in range(experts)}
            groups["mlp.shared_experts"] = (
                config["moe_intermediate_size"] * config["n_shared_experts"]
            )
            plain = {
                "mlp.gate.weight": (experts, h),
                "mlp.gate.e_score_correction_bias": (experts,),
            }
    for prefix, intermediate in groups.items():
        quantized.update(
            {
                f"{prefix}.gate_proj": (intermediate, h),
                f"{prefix}.up_proj": (intermediate, h),
                f"{prefix}.down_proj": (h, intermediate),
            }
        )
    return quantized, plain


def _tensor_specs(config: Mapping, num_layers: int) -> dict[str, TensorSpec]:
    h, q, kv = (config[name] for name in ("hidden_size", "q_lora_rank", "kv_lora_rank"))
    heads, rope, nope = (
        config[name] for name in ("num_attention_heads", "qk_rope_head_dim", "qk_nope_head_dim")
    )
    vd = config["v_head_dim"]
    ih, idim = config["index_n_heads"], config["index_head_dim"]
    plain_dtypes = ("BF16", "F16", "F32")
    specs = {"model.embed_tokens.weight": TensorSpec((config["vocab_size"], h), plain_dtypes)}
    projections = {
        "self_attn.q_a_proj": (q, h),
        "self_attn.q_b_proj": (heads * (nope + rope), q),
        "self_attn.kv_a_proj_with_mqa": (kv + rope, h),
        "self_attn.kv_b_proj": (heads * (nope + vd), kv),
        "self_attn.o_proj": (h, heads * vd),
        "self_attn.indexer.wq_b": (ih * idim, q),
        "self_attn.indexer.wk": (idim, h),
    }
    plain = {
        "input_layernorm.weight": (h,),
        "post_attention_layernorm.weight": (h,),
        "self_attn.q_a_layernorm.weight": (q,),
        "self_attn.kv_a_layernorm.weight": (kv,),
        "self_attn.indexer.k_norm.weight": (idim,),
        "self_attn.indexer.k_norm.bias": (idim,),
        "self_attn.indexer.weights_proj.weight": (ih, h),
    }
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}."
        mlp, router = layer_mlp_shapes(config, layer)
        for name, shape in (projections | mlp).items():
            specs[prefix + name + ".weight"] = TensorSpec(shape, ("F8_E4M3",))
            specs[prefix + name + ".weight_scale_inv"] = TensorSpec(
                tuple((dim + 127) // 128 for dim in shape), ("F32",)
            )
        for name, shape in (plain | router).items():
            dtypes = ("F32",) if name.endswith("e_score_correction_bias") else plain_dtypes
            specs[prefix + name] = TensorSpec(shape, dtypes)
    return specs


def _safe_shard_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise AdapterError(f"invalid shard path: {name!r}")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
        raise AdapterError(f"unsafe shard path: {name!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise AdapterError(f"shard escapes checkpoint root: {name!r}")
    return path


def inspect_checkpoint(model_path: str | Path, num_layers: int = 3) -> CheckpointSelection:
    """Read JSON metadata and selected shard stats, never weight payloads."""
    root = Path(model_path).resolve(strict=True)
    config, config_hash = _read_metadata(root / "config.json")
    validate_checkpoint_config(config, num_layers)
    index, index_hash = _read_metadata(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise AdapterError("weight_map must be an object")
    specs = _tensor_specs(config, num_layers)
    missing = sorted(specs.keys() - weight_map.keys())
    if missing:
        raise AdapterError("missing required checkpoint tensors: " + ", ".join(missing))
    prefixes = tuple(f"model.layers.{i}." for i in range(num_layers))
    unexpected = sorted(
        name for name in weight_map if name.startswith(prefixes) and name not in specs
    )
    if unexpected:
        raise AdapterError("unsupported selected-layer tensors: " + ", ".join(unexpected))
    groups = {}
    stats = {}
    for name in sorted(specs):
        shard = weight_map[name]
        path = _safe_shard_path(root, shard)
        groups.setdefault(shard, []).append(name)
        if shard not in stats:
            stat = path.stat()
            stats[shard] = (stat.st_size, stat.st_mtime_ns)
    hashes = {"config.json": config_hash, "model.safetensors.index.json": index_hash}
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "model_path": str(root),
                "num_layers": num_layers,
                "metadata": hashes,
                "selected_shard_stats": stats,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return CheckpointSelection(
        root,
        num_layers,
        copy.deepcopy(config),
        tuple(sorted(specs)),
        {name: tuple(tensors) for name, tensors in groups.items()},
        specs,
        hashes,
        stats,
        fingerprint,
    )


def iter_selected_weights(selection: CheckpointSelection, open_shard: Callable) -> Iterator:
    """Validate selected tensor headers before materializing each selected tensor."""
    for name, expected in selection.metadata_sha256.items():
        if _read_metadata(selection.model_path / name)[1] != expected:
            raise AdapterError(f"checkpoint metadata changed after inspection: {name}")
    for shard, names in selection.shards.items():
        path = _safe_shard_path(selection.model_path, shard)
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != selection.shard_stats[shard]:
            raise AdapterError(f"selected checkpoint shard changed after inspection: {shard}")
        with open_shard(str(path), framework="pt", device="cpu") as handle:
            present = set(handle.keys())
            for name in names:
                if name not in present:
                    raise AdapterError(f"indexed tensor missing from selected shard: {name}")
                spec = selection.tensor_specs[name]
                tensor_slice = handle.get_slice(name)
                if tuple(tensor_slice.get_shape()) != spec.shape:
                    raise AdapterError(f"checkpoint tensor shape mismatch: {name}")
                if tensor_slice.get_dtype() not in spec.dtypes:
                    raise AdapterError(f"checkpoint tensor dtype mismatch: {name}")
                yield name, handle.get_tensor(name)


def validate_runtime_environment(environ: Mapping[str, str] | None = None) -> None:
    environ = os.environ if environ is None else environ
    for name in ("FAKE_P_NODE", "DS_DEBUG_LAYERS"):
        if name in environ:
            raise AdapterError(f"{name} must be unset for real, explicitly selected prefill layers")


def _validate_runtime_config(config: Mapping, selection: CheckpointSelection) -> None:
    normalized = copy.deepcopy(dict(config))
    if normalized.get("model_type") not in ("deepseek_v32", "deepseek_v3"):
        raise AdapterError("unexpected ECHO runtime model type")
    if normalized.get("architectures") not in ([name] for name in RUNTIME_ARCHITECTURES):
        raise AdapterError("unexpected ECHO runtime architecture")
    # ECHO's HF fallback uses DeepseekV3Config; Transformers also renames torch_dtype.
    normalized["model_type"] = "deepseek_v32"
    normalized["architectures"] = [ARCHITECTURE]
    normalized["torch_dtype"] = normalized.get("torch_dtype", normalized.get("dtype"))
    validate_checkpoint_config(normalized, selection.num_layers)
    quant = normalized["quantization_config"]
    if "packed_modules_mapping" in quant:
        if quant["packed_modules_mapping"] != PACKED_MODULES_MAPPING:
            raise AdapterError(
                "runtime packed_modules_mapping differs from the adapter's Q/KV fusion"
            )
        # ECHO injects this loader-only field into hf_config before model construction.
        if "packed_modules_mapping" not in selection.config["quantization_config"]:
            del quant["packed_modules_mapping"]
    expected_rope = selection.config.get("rope_scaling")
    actual_rope = normalized.get("rope_scaling")
    if (
        isinstance(expected_rope, dict)
        and isinstance(actual_rope, dict)
        and "type" in expected_rope
        and "rope_type" not in expected_rope
        and "rope_type" in actual_rope
    ):
        # Transformers copies the legacy type into rope_type without removing type.
        if actual_rope["rope_type"] != expected_rope["type"]:
            raise AdapterError(
                "runtime rope_scaling.rope_type differs from checkpoint type: "
                f"expected {expected_rope['type']!r}, got {actual_rope['rope_type']!r}"
            )
        del actual_rope["rope_type"]
    if normalized["num_hidden_layers"] != selection.num_layers:
        raise AdapterError("runtime num_hidden_layers does not match the checkpoint selection")
    for key, value in selection.config.items():
        if (
            key not in ("num_hidden_layers", "transformers_version", "_name_or_path")
            and normalized.get(key) != value
        ):
            raise AdapterError(
                f"runtime checkpoint config changed: {key}: "
                f"expected {value!r}, got {normalized.get(key)!r}"
            )


def verify_echo_checkout(echo_path: str | Path) -> tuple[Path, str, str]:
    root = Path(echo_path).resolve(strict=True)
    if not (root / ".git").exists():
        raise AdapterError("ECHO path must be a Git checkout root")

    def git_output(*args):
        result = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, check=True, timeout=30
        )
        return result.stdout

    revision = git_output("rev-parse", "HEAD").decode().strip()
    if revision != ECHO_REVISION:
        raise AdapterError(f"ECHO revision mismatch: expected {ECHO_REVISION}, found {revision}")
    diff = git_output("diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--")
    return root, revision, hashlib.sha256(diff).hexdigest()


def _import_echo_runtime(root: Path) -> SimpleNamespace:
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise AdapterError("the ECHO model adapter requires a CUDA device")
    upstream = importlib.import_module("sglang.srt.models.deepseek_v2")
    registry = importlib.import_module("sglang.srt.models.registry")
    loader = importlib.import_module("sglang.srt.model_loader.loader")
    logits = importlib.import_module("sglang.srt.layers.logits_processor")
    host_pool = importlib.import_module("sglang.srt.mem_cache.memory_pool_host")
    for module in (upstream, registry, loader, logits, host_pool):
        if not Path(module.__file__).resolve().is_relative_to(root / "sglang" / "python"):
            raise AdapterError(
                f"imported SGLang module is outside pinned ECHO checkout: {module.__file__}"
            )
    return SimpleNamespace(
        torch=torch,
        upstream=upstream,
        registry=registry.ModelRegistry,
        loader_base=loader.DefaultModelLoader,
        logits_output=logits.LogitsProcessorOutput,
        safe_open=importlib.import_module("safetensors").safe_open,
        host_pool=host_pool,
    )


def _make_model_class(runtime: SimpleNamespace, selection: CheckpointSelection) -> type:
    torch, source = runtime.torch, runtime.upstream

    class ResidualMerge(torch.nn.Module):
        def forward(self, x, residual=None):
            return x if residual is None else (x + residual, None)

    class GRPrefixModel(source.DeepseekV32ForCausalLM):
        packed_modules_mapping: ClassVar[dict[str, list[str]]] = copy.deepcopy(
            PACKED_MODULES_MAPPING
        )

        def __init__(self, config, quant_config=None, prefix=""):
            torch.nn.Module.__init__(self)
            validate_runtime_environment()
            _validate_runtime_config(config.to_dict(), selection)
            configured_loader = source.get_global_server_args().load_format
            if (
                getattr(configured_loader, "checkpoint_metadata_fingerprint", None)
                != selection.metadata_fingerprint
            ):
                raise AdapterError(
                    "ModelRunner must use this scope's real prefix-only weight loader"
                )
            if quant_config is None or quant_config.get_name() != "fp8":
                raise AdapterError(
                    "runtime must preserve the selected checkpoint's FP8 quantization"
                )
            self.config, self.quant_config = config, quant_config
            self.pp_group = source.get_pp_group()
            self.tp_size = source.get_tensor_model_parallel_world_size()
            if self.tp_size != 1 or self.pp_group.world_size != 1:
                raise AdapterError("the reduced ECHO adapter currently requires TP=PP=1")
            self.fuse_qkv_a_proj = True
            self.num_fused_shared_experts = 0
            self.model = source.DeepseekV2Model(
                config, quant_config, prefix=source.add_prefix("model", prefix)
            )
            self._routed_experts_weights_of_layer = source.LazyValue(
                lambda: {
                    i: layer.mlp.get_moe_weights()
                    for i, layer in enumerate(self.model.layers)
                    if isinstance(layer.mlp, source.DeepseekV2MoE)
                }
            )
            self.model.norm = ResidualMerge()

        @torch.no_grad()
        def forward(self, input_ids, positions, forward_batch, input_embeds=None):
            validate_runtime_environment()
            if input_embeds is not None or getattr(forward_batch, "input_embeds", None) is not None:
                raise AdapterError("GR prefix execution must use real checkpoint token embeddings")
            if not forward_batch.forward_mode.is_extend():
                raise AdapterError("the reduced ECHO adapter supports prefill/extend only")
            hidden = self.model(input_ids, positions, forward_batch)
            # SGLang's DP postprocessing slices logits even when no sampling is requested.
            placeholder = hidden.new_empty((hidden.shape[0], 0))
            return runtime.logits_output(next_token_logits=placeholder, hidden_states=hidden)

    return GRPrefixModel


def _make_loader_class(runtime: SimpleNamespace, selection: CheckpointSelection) -> type:
    class PrefixOnlyLoader(runtime.loader_base):
        checkpoint_metadata_fingerprint = selection.metadata_fingerprint

        def _get_all_weights(self, model_config, model):
            if Path(model_config.model_path).resolve() != selection.model_path:
                raise AdapterError("ModelRunner checkpoint differs from the inspected checkpoint")
            yield from iter_selected_weights(selection, runtime.safe_open)

    return PrefixOnlyLoader


@contextmanager
def scoped_echo_adapter(
    echo_path: str | Path, model_path: str | Path, num_layers: int = 3
) -> Iterator[EchoAdapterBindings]:
    """Temporarily register a real, hidden-only model for a local ModelRunner.

    Set ECHO environment flags before entering, then construct ServerArgs with
    json_model_override_args='{"num_hidden_layers": N}'. Assign the yielded
    loader_class to server_args.load_format before constructing ModelRunner.
    Keep is_generation=True and bypass sampling. No server is started here.

    Keep this scope open throughout runner execution: it also corrects ECHO's
    extend-recall free-slot accounting and restores the original cache method
    on exit. Registry/path restoration does not unload imported modules or undo
    CUDA/distributed initialization. Use a fresh process for each backend/config.
    metadata_fingerprint is not a weight-content hash; model_instance_id prevents
    cache reuse across scopes and must be included in the serving cache identity.
    """
    validate_runtime_environment()
    selection = inspect_checkpoint(model_path, num_layers)
    root, revision, patch_hash = verify_echo_checkout(echo_path)
    old_path = sys.path[:]
    sys.path.insert(0, str(root / "sglang" / "python"))
    try:
        runtime = _import_echo_runtime(root)
        model_class = _make_model_class(runtime, selection)
        loader_class = _make_loader_class(runtime, selection)
        previous = {name: runtime.registry.models.get(name) for name in RUNTIME_ARCHITECTURES}
        for name in RUNTIME_ARCHITECTURES:
            runtime.registry.models[name] = model_class
        try:
            with echo_cache.scoped_extend_recall_fix(
                runtime.host_pool, runtime.torch
            ) as adaptation:
                yield EchoAdapterBindings(
                    loader_class,
                    model_class,
                    selection,
                    revision,
                    patch_hash,
                    str(uuid4()),
                    (adaptation,),
                    hashlib.sha256(Path(echo_cache.__file__).read_bytes()).hexdigest(),
                )
        finally:
            for name, old_class in previous.items():
                if old_class is None:
                    runtime.registry.models.pop(name, None)
                else:
                    runtime.registry.models[name] = old_class
    finally:
        sys.path[:] = old_path
