"""Inspect a reduced DeepSeek V3.2 experiment without reading weight payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import struct
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

MAX_HEADER_BYTES = 64 * 1024 * 1024
DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
}
FP8_DTYPES = {"F8_E4M3", "F8_E5M2"}


class AuditError(ValueError):
    """Invalid or unsupported checkpoint metadata."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object: {path}")
    return value


def read_safetensors_header(path: Path) -> dict:
    """Read the length and JSON header only; validate offsets against file size."""
    with path.open("rb") as handle:
        file_bytes = path.stat().st_size
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise AuditError(f"truncated safetensors length: {path}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes < 2 or header_bytes > MAX_HEADER_BYTES:
            raise AuditError(f"invalid safetensors header size {header_bytes}: {path}")
        if header_bytes > file_bytes - 8:
            raise AuditError(f"truncated safetensors header: {path}")
        raw_header = handle.read(header_bytes)
    header = json.loads(raw_header, object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise AuditError(f"safetensors header is not an object: {path}")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()
    ):
        raise AuditError(f"invalid safetensors metadata: {path}")
    payload_bytes = file_bytes - 8 - header_bytes
    spans = []
    tensors = {}
    for name, entry in header.items():
        if not isinstance(entry, dict):
            raise AuditError(f"invalid tensor metadata: {name}")
        dtype, shape, offsets = (entry.get(key) for key in ("dtype", "shape", "data_offsets"))
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise AuditError(f"unsupported dtype {dtype!r}: {name}")
        if not isinstance(shape, list) or not all(type(x) is int and x >= 0 for x in shape):
            raise AuditError(f"invalid tensor shape: {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(x) is int for x in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= payload_bytes
        ):
            raise AuditError(f"invalid tensor offsets: {name}")
        numel = math.prod(shape)
        if offsets[1] - offsets[0] != numel * DTYPE_BYTES[dtype]:
            raise AuditError(f"tensor byte count disagrees with shape/dtype: {name}")
        if offsets[1] > offsets[0]:
            spans.append((offsets[0], offsets[1], name))
        tensors[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": offsets,
            "numel": numel,
            "stored_bytes": offsets[1] - offsets[0],
        }
    end = 0
    for start, stop, name in sorted(spans):
        if start != end:
            raise AuditError(f"non-contiguous or overlapping tensor payload: {name}")
        end = stop
    if end != payload_bytes:
        raise AuditError(f"unindexed bytes in safetensors payload: {path}")
    return {
        "header_bytes": header_bytes,
        "file_bytes": file_bytes,
        "payload_bytes_read": 0,
        "metadata": metadata,
        "tensors": tensors,
    }


def _positive(config: dict, key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise AuditError(f"config {key} must be a positive integer")
    return value


def expected_tensors(config: dict, layers: int) -> dict:
    from models.deepseek_v32.echo_adapter import AdapterError, layer_mlp_shapes

    if type(layers) is not int or not 1 <= layers <= 5:
        raise AuditError("layers must be between 1 and 5")
    if config.get("model_type") != "deepseek_v32":
        raise AuditError("only deepseek_v32 checkpoint metadata is supported")
    if layers > _positive(config, "num_hidden_layers"):
        raise AuditError("selected layers extend beyond the checkpoint")
    _positive(config, "first_k_dense_replace")
    if config.get("attention_bias", False):
        raise AuditError("attention_bias=True is not supported")
    h, v = _positive(config, "hidden_size"), _positive(config, "vocab_size")
    q, kv = _positive(config, "q_lora_rank"), _positive(config, "kv_lora_rank")
    heads = _positive(config, "num_attention_heads")
    rope, nope = _positive(config, "qk_rope_head_dim"), _positive(config, "qk_nope_head_dim")
    vd = _positive(config, "v_head_dim")
    _positive(config, "intermediate_size")
    ih, idim = _positive(config, "index_n_heads"), _positive(config, "index_head_dim")
    quant = config.get("quantization_config") or {}
    if not isinstance(quant, dict):
        raise AuditError("quantization_config must be an object")
    method = quant.get("quant_method")
    if method not in (None, "fp8"):
        raise AuditError(f"unsupported checkpoint quantization: {method}")
    fp8 = method == "fp8"
    block = quant.get("weight_block_size", [128, 128])
    if (
        not isinstance(block, list)
        or len(block) != 2
        or not all(type(x) is int and x > 0 for x in block)
    ):
        raise AuditError("weight_block_size must contain two positive integers")
    matrix_dtypes = sorted(FP8_DTYPES) if fp8 else ["BF16", "F16", "F32"]
    specs = {"model.embed_tokens.weight": {"shape": [v, h], "dtypes": ["BF16", "F16", "F32"]}}
    projections = {
        "self_attn.q_a_proj": [q, h],
        "self_attn.q_b_proj": [heads * (nope + rope), q],
        "self_attn.kv_a_proj_with_mqa": [kv + rope, h],
        "self_attn.kv_b_proj": [heads * (nope + vd), kv],
        "self_attn.o_proj": [h, heads * vd],
        "self_attn.indexer.wq_b": [ih * idim, q],
        "self_attn.indexer.wk": [idim, h],
    }
    plain = {
        "input_layernorm.weight": [h],
        "post_attention_layernorm.weight": [h],
        "self_attn.q_a_layernorm.weight": [q],
        "self_attn.kv_a_layernorm.weight": [kv],
        "self_attn.indexer.k_norm.weight": [idim],
        "self_attn.indexer.k_norm.bias": [idim],
        "self_attn.indexer.weights_proj.weight": [ih, h],
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        try:
            mlp, router = layer_mlp_shapes(config, layer)
        except AdapterError as exc:
            raise AuditError(str(exc)) from exc
        for name, shape in (projections | mlp).items():
            shape = list(shape)
            specs[prefix + name + ".weight"] = {"shape": shape, "dtypes": matrix_dtypes}
            if fp8:
                scale_shape = [(dim + tile - 1) // tile for dim, tile in zip(shape, block)]
                specs[prefix + name + ".weight_scale_inv"] = {
                    "shape": scale_shape,
                    "dtypes": ["F32", "F8_E8M0"],
                }
        for name, shape in (plain | router).items():
            dtypes = ["F32"] if name.endswith("e_score_correction_bias") else ["BF16", "F16", "F32"]
            specs[prefix + name] = {"shape": list(shape), "dtypes": dtypes}
    return specs


def _shard_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise AuditError(f"invalid shard path: {name!r}")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
        raise AuditError(f"unsafe shard path: {name!r}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise AuditError(f"shard path escapes checkpoint root: {name!r}")
    return resolved


def audit_checkpoint(model_path: Path, layers: int) -> dict:
    root = model_path.resolve()
    config = read_json(root / "config.json")
    specs = expected_tensors(config, layers)
    index = read_json(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise AuditError("checkpoint index weight_map must be an object")
    missing = sorted(specs.keys() - weight_map.keys())
    if missing:
        raise AuditError("missing required tensors in checkpoint index: " + ", ".join(missing))
    prefixes = tuple(f"model.layers.{i}." for i in range(layers))
    selected = {
        name: shard
        for name, shard in weight_map.items()
        if name == "model.embed_tokens.weight" or name.startswith(prefixes)
    }
    unexpected = sorted(selected.keys() - specs.keys())
    if unexpected:
        raise AuditError("unsupported tensors in selected layers: " + ", ".join(unexpected))
    headers = {}
    for name in selected.values():
        path = _shard_path(root, name)
        if name not in headers:
            headers[name] = read_safetensors_header(path)
    tensors = {}
    for name, shard in sorted(selected.items()):
        tensor = headers[shard]["tensors"].get(name)
        if tensor is None:
            raise AuditError(f"index points to a shard missing tensor: {name} in {shard}")
        spec = specs[name]
        if tensor["shape"] != spec["shape"] or tensor["dtype"] not in spec["dtypes"]:
            raise AuditError(f"unexpected shape/dtype for {name}: {tensor}; expected {spec}")
        is_scale = name.endswith(".weight_scale_inv")
        bf16_bytes = 0 if is_scale else tensor["numel"] * (4 if tensor["dtype"] == "F32" else 2)
        tensors[name] = {**tensor, "shard": shard, "dequantized_bf16_bytes": bf16_bytes}
    return {
        "status": "passed_header_audit",
        "model_path": str(root),
        "layers": layers,
        "config": config,
        "index_metadata": index.get("metadata", {}),
        "metadata_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("config.json", "model.safetensors.index.json")
        },
        "fingerprint_scope": "JSON metadata files only; weight payloads are not hashed",
        "required_tensor_count": len(specs),
        "selected_tensor_count": len(tensors),
        "selected_stored_bytes": sum(t["stored_bytes"] for t in tensors.values()),
        "selected_dequantized_bf16_bytes": sum(
            t["dequantized_bf16_bytes"] for t in tensors.values()
        ),
        "weight_accounting": {
            "includes": "full embedding and first N attention/indexer/dense MLP/norm parameters",
            "excludes": ["final model norm", "lm_head", "later layers", "allocator overhead"],
            "stored": "checkpoint tensor payload bytes, including FP8 scales",
            "dequantized_bf16": "FP8/F16/BF16 parameters as BF16; F32 parameters retained; scales excluded",
        },
        "shards": {
            name: {k: v for k, v in header.items() if k != "tensors"}
            for name, header in headers.items()
        },
        "payload_bytes_read": 0,
        "tensors": tensors,
    }


def cache_capacity(
    config: dict, layers: int, prefix_tokens: int, candidate_tokens: int, num_users: int
) -> dict:
    for name, value in (
        ("layers", layers),
        ("prefix_tokens", prefix_tokens),
        ("candidate_tokens", candidate_tokens),
        ("num_users", num_users),
    ):
        if type(value) is not int or value <= 0:
            raise AuditError(f"{name} must be a positive integer")
    max_positions = _positive(config, "max_position_embeddings")
    if prefix_tokens + candidate_tokens > max_positions:
        raise AuditError("prefix_tokens + candidate_tokens exceeds max_position_embeddings")
    if _positive(config, "index_head_dim") != 128:
        raise AuditError("ECHO index capacity requires index_head_dim=128")
    mla = 2 * (_positive(config, "kv_lora_rank") + _positive(config, "qk_rope_head_dim"))
    index = _positive(config, "index_head_dim") + 4
    prefix_mla = layers * prefix_tokens * mla
    prefix_index = layers * prefix_tokens * index
    candidate_mla = layers * candidate_tokens * mla
    candidate_index = layers * candidate_tokens * index
    return {
        "kind": "capacity_estimate",
        "layers": layers,
        "prefix_tokens": prefix_tokens,
        "candidate_tokens": candidate_tokens,
        "num_users": num_users,
        "mla_layout": "BF16 latent KV plus BF16 RoPE key",
        "index_layout": "FP8 index key plus one F32 per-token scale",
        "mla_bytes_per_token_per_layer": mla,
        "index_bytes_per_token_per_layer": index,
        "per_user_prefix_mla_bytes": prefix_mla,
        "per_user_prefix_index_bytes": prefix_index,
        "all_users_prefix_mla_bytes": prefix_mla * num_users,
        "all_users_prefix_index_bytes": prefix_index * num_users,
        "one_active_candidate_mla_bytes": candidate_mla,
        "one_active_candidate_index_bytes": candidate_index,
        "all_resident_prefix_plus_one_candidate_bytes": (
            (prefix_mla + prefix_index) * num_users + candidate_mla + candidate_index
        ),
        "placement": "not chosen; these are logical payload sizes before HBM/host partitioning",
        "metadata_bytes": None,
        "workspace_bytes": None,
        "excluded": [
            "page alignment",
            "cache tables and replacement metadata",
            "staging buffers",
            "activations and attention workspace",
            "CUDA context",
            "weights",
        ],
        "assumptions": [
            "one candidate request active at a time",
            "candidate KV is not retained",
            "no 656-byte packed MLA record assumption",
        ],
    }


def probe_command(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return {
            "command": command,
            "status": "ok" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "status": "error", "error": str(exc)}


def probe_echo(echo_path: Path) -> dict:
    root = echo_path.resolve()
    probes = {
        "DeepGEMM/deep_gemm/__init__.py": [
            "fp8_mqa_logits_fuse_prefetch",
            "fp8_paged_mqa_logits_fused_v2",
        ],
        "sglang/python/sglang/srt/layers/attention/nsa/nsa_indexer.py": [
            "SGLANG_NSA_FUSE_LOGITS_RECALL_EXTEND",
            "SGLANG_NSA_FUSE_LOGITS_RECALL_DECODE",
        ],
        "sglang/python/sglang/srt/mem_cache/recall_ops.py": ["def recall_update_with_prefetch("],
        "sglang/python/sglang/srt/mem_cache/memory_pool_host.py": [
            "def recall_miss_tokens_extend_cuda_graph("
        ],
    }
    sources = {}
    for relative, symbols in probes.items():
        try:
            source = (root / relative).read_text(encoding="utf-8")
            sources[relative] = {"symbols": {s: s in source for s in symbols}}
        except (OSError, UnicodeError) as exc:
            sources[relative] = {"error": str(exc), "symbols": {s: False for s in symbols}}
    if (root / ".git").exists():
        revision = probe_command(["git", "-C", str(root), "rev-parse", "HEAD"])
        tracked_status = probe_command(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]
        )
    else:
        revision = {"status": "error", "error": "ECHO path is not a Git checkout root"}
        tracked_status = dict(revision)
    return {
        "path": str(root),
        "git_revision": revision,
        "git_tracked_status": tracked_status,
        "tracked_dirty": (
            bool(tracked_status["stdout"]) if tracked_status["status"] == "ok" else None
        ),
        "source_availability": sources,
        "probe_semantics": "literal source presence only; no module imports or compilation",
        "compiled_status": "unverified",
        "numerical_status": "unverified",
    }


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--echo-path", type=Path, default=Path("3rdparty/ECHO"))
    parser.add_argument("--layers", type=int, choices=(1, 2, 3, 4, 5), default=3)
    parser.add_argument("--prefix-tokens", type=positive_int, default=65536)
    parser.add_argument("--candidate-tokens", type=positive_int, default=1024)
    parser.add_argument("--num-users", type=positive_int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and (args.output.exists() or args.output.is_symlink()):
        parser.error(f"--output already exists: {args.output}")
    result = {
        "schema_version": 1,
        "kind": "feasibility",
        "is_benchmark": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "compiled_status": "unverified",
        "numerical_status": "unverified",
        "performance_status": "not_run",
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        checkpoint = audit_checkpoint(args.model_path, args.layers)
        result["checkpoint"] = checkpoint
        result["cache_capacity"] = cache_capacity(
            checkpoint["config"],
            args.layers,
            args.prefix_tokens,
            args.candidate_tokens,
            args.num_users,
        )
        exit_code = 0
    except (OSError, ValueError, UnicodeError) as exc:
        result["checkpoint"] = {"status": "failed", "error": str(exc)}
        exit_code = 1
    result["echo"] = probe_echo(args.echo_path)
    nvcc = shutil.which("nvcc")
    if not nvcc and Path("/usr/local/cuda/bin/nvcc").is_file():
        nvcc = "/usr/local/cuda/bin/nvcc"
    result["hardware_probes"] = {
        "nvidia_smi": probe_command(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
        ),
        "cuda_compiler": probe_command([nvcc or "nvcc", "--version"]),
        "interpretation": "optional environment probes; success does not establish executability",
    }
    result["exit_code_semantics"] = "zero means checkpoint header audit passed, not GPU readiness"
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
