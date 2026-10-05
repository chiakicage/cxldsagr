"""Capture actual resident operator inputs from an unchanged NOSA sparse forward.

Build an independent 64K sparse prefix from an empty cache, then observe the
1K suffix at selected layers. Each layer_XX.pt is a CPU tensor dictionary with
q, k, v, cis, compressed_k, ids and valid_mask. Load these local artifacts with
torch.load(weights_only=True).
The saved tensors preserve their original strides; metadata describes their
logical contents and provenance. This entry point does not measure performance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import torch

from executor.model_executor import run_chunks
from experiments.nosa_mfu.src.dense.capture import execution_split
from experiments.nosa_mfu.src.dense.sources import source_hashes
from experiments.nosa_mfu.src.sparse.capture import (
    make_request,
    rewind_cache,
    validate_request,
)
from models.nosa.infer import DEFAULT_MODEL_PATH
from models.nosa.model import NosaForCausalLM
from operators.nosa._native import build_info

ROOT = Path(__file__).resolve().parents[3]
PREFIX, NEW, CHUNK = 65536, 1024, 1024


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_metadata(tensor):
    if tensor.device.type != "cpu":
        raise ValueError("Tensor fingerprints require the saved CPU copy")
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(
            tensor.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }


def copy_preserving_strides(tensor):
    """Copy logical values without serializing unrelated QKV/cache storage."""
    result = torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device="cpu")
    # Initialize stride gaps too: torch.save serializes the entire owned storage.
    storage = result.untyped_storage()
    torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,)).zero_()
    result.copy_(tensor.detach())
    return result


def checkpoint_manifest(model_path, *, hash_contents):
    paths = set(model_path.glob("*.safetensors"))
    paths.update(
        model_path / name for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
    )
    paths.update(model_path.glob("*.safetensors.index.json"))
    manifest = {}
    for path in sorted(paths):
        stat = path.stat()
        item = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if hash_contents:
            item["sha256"] = sha256_file(path)
        manifest[path.name] = item
    if not any(name.endswith(".safetensors") for name in manifest):
        raise ValueError("Checkpoint contains no safetensors weights")
    return manifest


def snapshot_sources(destination):
    profile = ROOT / "experiments/nosa_mfu/src/sparse"
    sources = source_hashes(
        *sorted(Path(__file__).parent.glob("*.py")),
        ROOT / "experiments/nosa_mfu/scripts/capture.sh",
        profile / "capture.py",
        profile / "mfu.py",
        profile / "analyze.py",
        ROOT / "evaluation/validation.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    )
    for name, expected in sources.items():
        content = (ROOT / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError(f"Source changed while creating snapshot: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return sources


class InputRecorder:
    """Forward the exact model call, then copy its borrowed resident inputs."""

    def __init__(self, attention, config, layers, destination, *, prefix=PREFIX, queries=NEW):
        self.attention = attention
        self.config = config
        self.layers = frozenset(layers)
        self.destination = destination
        self.prefix = prefix
        self.queries = queries
        self.visited = []
        self.records = []

    def __call__(self, q, selection, cache_access, context):
        layer = context.layer_idx
        if (
            layer != len(self.visited)
            or context.query_start != self.prefix
            or context.query_length != self.queries
        ):
            raise ValueError("Capture must visit every suffix layer exactly once in model order")
        self.visited.append(layer)
        # No replacement inputs, selection, outputs, compression or model math.
        output = self.attention(q, selection, cache_access, context)
        if layer not in self.layers:
            return output
        if selection is None or selection.block_size != 64:
            raise ValueError("Capture requires the actual complete NOSA block selection")
        raw = cache_access.layer_view(layer)
        derived_cache = getattr(cache_access, "indexer_cache", None)
        if derived_cache is None:
            raise ValueError("Capture requires materialized model-owned compressed records")
        derived = derived_cache.layer_view(layer)
        cis = getattr(context.auxiliary_state, "cis_scores", None)
        if cis is None:
            cis = raw["cis_scores"]
        if (
            tuple(q.shape) != (self.queries, self.config.num_attention_heads, self.config.head_dim)
            or tuple(raw["keys"].shape)
            != (self.prefix + self.queries, self.config.num_key_value_heads, self.config.head_dim)
            or raw["values"].shape != raw["keys"].shape
            or cis.shape != raw["keys"].shape[:2]
            or selection.block_ids.shape != (self.queries, self.config.num_key_value_heads, 64)
            or not len(derived["compressed_keys"])
        ):
            raise ValueError("Actual operator geometry differs from the requested suffix")
        tensors = {
            "q": q,
            "k": raw["keys"],
            "v": raw["values"],
            "cis": cis,
            "compressed_k": derived["compressed_keys"],
            "ids": selection.block_ids,
        }
        if selection.valid_mask is not None:
            tensors["valid_mask"] = selection.valid_mask
        saved = {name: copy_preserving_strides(tensor) for name, tensor in tensors.items()}
        if "valid_mask" not in saved:
            saved["valid_mask"] = torch.ones_like(saved["ids"], dtype=torch.bool)
        for name, tensor in saved.items():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
                raise ValueError(f"Captured layer {layer} {name} contains nonfinite values")
        filename = f"layer_{layer:02d}.pt"
        target = self.destination / filename
        torch.save(saved, target)
        self.records.append(
            {
                "layer": layer,
                "query_start": context.query_start,
                "query_length": context.query_length,
                "queries": context.query_length,
                "block_size": selection.block_size,
                "selection_had_valid_mask": selection.valid_mask is not None,
                "file": filename,
                "file_sha256": sha256_file(target),
                "tensors": {name: tensor_metadata(tensor) for name, tensor in saved.items()},
                "original_storage_offsets": {
                    name: tensor.storage_offset() for name, tensor in tensors.items()
                },
            }
        )
        print(f"Captured actual attention inputs for layer {layer}: {filename}", flush=True)
        return output


@torch.inference_mode()
def capture(args, staging):
    sources = snapshot_sources(staging / "sources")
    print("Fingerprinting checkpoint contents", flush=True)
    checkpoint = checkpoint_manifest(args.model_path, hash_contents=True)
    request = make_request(args)
    if args.request_file is None:
        write_json(staging / "request.json", request)
    else:
        request_bytes = args.request_file.read_bytes()
        if json.loads(request_bytes) != request:
            raise RuntimeError("Request changed during input preparation")
        (staging / "request.json").write_bytes(request_bytes)
    validate_request(request, PREFIX, NEW)
    prefix_ids, new_ids = execution_split(request, PREFIX, NEW)
    print("Loading the full BF16 NOSA sparse model", flush=True)
    model = NosaForCausalLM.from_pretrained(
        args.model_path,
        device=args.device,
        dtype=torch.bfloat16,
        attention_mode="sparse",
        sparse_backend="triton",
    )
    config = model.config
    if (
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    ) != (32, 32, 2, 128):
        raise ValueError("Capture requires NOSA-8B: 32 layers, 32 Q heads, 2 KV heads, D128")
    if max(request["input_ids"]) >= config.vocab_size:
        raise ValueError("Request contains token IDs outside the checkpoint vocabulary")
    model.config = replace(
        config, max_position_embeddings=max(PREFIX + NEW, config.max_position_embeddings)
    )
    prefix_ids = torch.tensor(prefix_ids, dtype=torch.long, device=args.device)
    new_ids = torch.tensor(new_ids, dtype=torch.long, device=args.device)
    native_build = build_info()
    with ExitStack() as resources:
        cache = model.new_cache(PREFIX + NEW)
        resources.callback(model.cache_manager.release, cache)
        if cache.length != 0:
            raise ValueError("Capture must start with an independent empty resident cache")
        print(f"Building {PREFIX}-token sparse prefix in {CHUNK}-token chunks", flush=True)
        run_chunks(model, prefix_ids, cache, CHUNK, output="hidden")
        if cache.length != PREFIX:
            raise ValueError("Prefix did not commit exactly 65536 tokens")
        recorder = InputRecorder(model.main_attention, model.config, args.layers, staging)
        with patch.object(model, "main_attention", recorder):
            hidden = run_chunks(model, new_ids, cache, CHUNK, output="hidden")
        if cache.length != PREFIX + NEW or len(recorder.visited) != config.num_hidden_layers:
            raise ValueError("Captured suffix did not complete every layer and cache commit")
        if {record["layer"] for record in recorder.records} != set(args.layers):
            raise ValueError("Not every requested layer was captured")
        hidden_cpu = hidden.cpu()
        if not torch.isfinite(hidden_cpu).all().item():
            raise ValueError("Captured model output contains nonfinite values")
        rewind_cache(cache, PREFIX)
        replay = run_chunks(model, new_ids, cache, CHUNK, output="hidden").cpu()
        if cache.length != PREFIX + NEW or not torch.equal(hidden_cpu, replay):
            raise ValueError("Observation changed suffix hidden states or cache commit")
    torch.save({"hidden": hidden_cpu}, staging / "candidate_hidden.pt")
    if sources != source_hashes(*(ROOT / name for name in sources)):
        raise RuntimeError("Implementation changed during capture; use a stable source tree")
    final_checkpoint = checkpoint_manifest(args.model_path, hash_contents=False)
    if final_checkpoint != {
        name: {key: value for key, value in item.items() if key != "sha256"}
        for name, item in checkpoint.items()
    }:
        raise RuntimeError("Checkpoint files changed during capture")
    if native_build != build_info():
        raise RuntimeError("Native build inputs changed during capture")
    props = torch.cuda.get_device_properties(args.device)
    metadata = {
        "schema_version": 1,
        "run_id": args.run_id,
        "query_start": PREFIX,
        "queries": NEW,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "kind": "actual_sparse_model_operator_inputs",
        "args": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "workload": {
            "prefix_tokens": PREFIX,
            "new_tokens": NEW,
            "chunk_size": CHUNK,
            "attention_mode": "sparse",
            "sparse_backend_api": "triton",
            "requested_kernel_backend": args.kernel_backend,
            "indexer_execution": "cached_flashinfer_v1",
            "indexer_query_chunk_size": model.indexer.effective_query_chunk_size(
                torch.device(args.device)
            ),
            "dtype": "bfloat16",
        },
        "model_config": asdict(model.config),
        "checkpoint_context": config.max_position_embeddings,
        "checkpoint_path": str(args.model_path),
        "checkpoint_files": checkpoint,
        "request_sha256": sha256_file(staging / "request.json"),
        "source_sha256": sources,
        "native_build": native_build,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "gpu": {
            "name": props.name,
            "uuid": str(props.uuid),
            "sm_count": props.multi_processor_count,
            "capability": [props.major, props.minor],
            "total_memory": props.total_memory,
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("triton", "flashinfer-python", "apache-tvm-ffi")
        },
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("CXLDSAGR_")
            or key in ("CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING", "CUBLAS_WORKSPACE_CONFIG")
        },
        "layers": recorder.records,
        "observation_validation": {
            "all_layers_visited": recorder.visited,
            "candidate_hidden_file": "candidate_hidden.pt",
            "candidate_hidden_file_sha256": sha256_file(staging / "candidate_hidden.pt"),
            "candidate_hidden": tensor_metadata(hidden_cpu),
            "unobserved_suffix_replay_exact_equal": True,
        },
        "definitions": {
            "inputs": "Post-RoPE Q/K, unmodified V, actual CIS bias and attention-consumed selection; compressed records read directly from model-owned indexer cache",
            "trajectory": "One full sparse model and independent empty cache; 64K prefix then 1K suffix; no dense sidecar, no selection or compression recomputation",
            "copies": "CPU copies preserve original strides with zero-initialized storage gaps and independent storage; logical SHA256 excludes stride padding",
            "acceptance": "Unobserved replay of the same sparse suffix from the same prefix must be bitwise equal; does not establish cross-backend or model-quality equivalence",
            "measurement": "Input capture only; no timing or MFU result; CPU observation synchronization is outside all performance measurements",
        },
    }
    write_json(staging / "metadata.json", metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--request-file", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kernel-backend", choices=("native", "triton"), default="native")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 15, 31])
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("run-id must contain only letters, digits, underscores or hyphens")
    if len(set(args.layers)) != len(args.layers) or any(
        not 0 <= layer < 32 for layer in args.layers
    ):
        parser.error("layers must be distinct NOSA-8B layer indices in [0,31]")
    args.layers.sort()
    args.output_dir = args.output_dir.resolve()
    args.model_path = args.model_path.resolve()
    if args.request_file is not None:
        args.request_file = args.request_file.resolve()
    if args.output_dir.exists():
        parser.error("output-dir already exists; choose a new capture directory")
    # Existing GR request construction consumes these fixed experiment bounds.
    args.prefix_tokens, args.new_tokens = PREFIX, NEW
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability(args.device) != (9, 0):
        parser.error("Actual NOSA sparse input capture requires SM90/Hopper")
    staging = Path(tempfile.mkdtemp(prefix=f"nosa-kernel-inputs-{args.run_id}-", dir="/tmp"))
    claimed_output = False
    try:
        with patch.dict(os.environ, {"CXLDSAGR_SM90_BACKEND": args.kernel_backend}):
            capture(args, staging)
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        args.output_dir.mkdir()
        claimed_output = True
        shutil.copytree(staging, args.output_dir, dirs_exist_ok=True)
    except BaseException:
        if claimed_output:
            shutil.rmtree(args.output_dir)
        print(f"Capture failed; diagnostics remain outside experiments: {staging}", file=sys.stderr)
        raise
    else:
        shutil.rmtree(staging)
        print(f"Captured real model inputs: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
