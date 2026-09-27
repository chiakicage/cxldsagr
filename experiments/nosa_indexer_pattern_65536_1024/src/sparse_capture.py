"""Capture three NOSA patterns on paired dense and actual sparse trajectories.

The dense trajectory observes QA-only and complete NOSA on the same Q/K/CIS.
The sparse trajectory records the selection actually consumed by attention.
No inference math or timing benchmark is implemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import subprocess
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from executor.model_executor import run_chunks
from experiments.indexer_block_sparse_profile.src.capture import validate_request
from experiments.nosa_gr_65536_1024.src.capture import execution_split
from experiments.nosa_gr_65536_1024.src.sources import source_hashes
from experiments.nosa_indexer_pattern_65536_1024.src.analyze import summarize
from experiments.nosa_indexer_pattern_65536_1024.src.capture import (
    DEFAULT_REQUEST,
    ROOT,
    sha256_file,
    write_json,
)
from layers.attention import DenseMainAttention
from models.nosa.indexer import NosaIndexer
from models.nosa.infer import DEFAULT_MODEL_PATH
from models.nosa.model import NosaForCausalLM

PREFIX, NEW, CHUNK = 65536, 1024, 1024


class PatternRecorder:
    """Observe sidecar choices, or record and forward an existing selection unchanged."""

    def __init__(
        self,
        attention,
        config,
        prefix_tokens,
        query_length,
        *,
        indexers=None,
        selection_mode="observe",
        verbose=False,
    ):
        if selection_mode not in ("observe", "actual"):
            raise ValueError("selection_mode must be observe or actual")
        if selection_mode == "observe" and not indexers:
            raise ValueError("Observe mode requires sidecar indexers")
        if selection_mode == "actual" and indexers:
            raise ValueError("Actual selections must not be recomputed by sidecar indexers")
        self.attention = attention
        self.config = config
        self.prefix_tokens = prefix_tokens
        self.query_length = query_length
        self.indexers = dict(indexers or {})
        self.selection_mode = selection_mode
        self.verbose = verbose
        arms = self.indexers if selection_mode == "observe" else ("sparse_nosa64",)
        self.block_ids = {name: [] for name in arms}
        self.valid_masks = {name: [] for name in arms}
        self.attention_shapes = []

    def __call__(self, q, selection, cache_access, context):
        if (
            context.layer_idx != len(self.attention_shapes)
            or context.query_start != self.prefix_tokens
            or context.query_length != self.query_length
        ):
            raise ValueError("Capture must visit each layer once at the candidate boundary")
        records = cache_access.layer_view(context.layer_idx)
        k, v, cis = records["keys"], records["values"], records["cis_scores"]
        config = self.config
        if tuple(q.shape) != (self.query_length, config.num_attention_heads, config.head_dim):
            raise ValueError("Actual Q shape differs from the requested candidate boundary")
        if (
            tuple(k.shape)
            != (
                self.prefix_tokens + self.query_length,
                config.num_key_value_heads,
                config.head_dim,
            )
            or v.shape != k.shape
            or tuple(cis.shape) != tuple(k.shape[:2])
        ):
            raise ValueError("Actual K/V/CIS shapes differ from the candidate boundary")
        if self.selection_mode == "observe":
            if selection is not None:
                raise ValueError("Dense observation requires model.indexer=None")
            chosen = {
                name: indexer(q, cache_access, context) for name, indexer in self.indexers.items()
            }
        else:
            if selection is None:
                raise ValueError("Actual sparse capture requires the model's block selection")
            chosen = {"sparse_nosa64": selection}
        # In actual mode this is the identical selection object from model.indexer.
        # The dense adapter receives None and therefore applies no CIS bias/mask.
        output = self.attention(q, selection, cache_access, context)
        for name, selected in chosen.items():
            ids = selected.block_ids.detach().to(device="cpu", dtype=torch.int32).clone()
            valid = (
                ids >= 0
                if selected.valid_mask is None
                else selected.valid_mask.detach().to(device="cpu", dtype=torch.bool).clone()
            )
            if (
                selected.block_size != 64
                or ids.shape != (self.query_length, config.num_key_value_heads, 64)
                or valid.shape != ids.shape
            ):
                raise ValueError("Expected the NOSA 64-token / 64-block selection contract")
            self.block_ids[name].append(ids)
            self.valid_masks[name].append(valid)
        self.attention_shapes.append(
            {
                "layer": context.layer_idx,
                "query_start": context.query_start,
                "q": list(q.shape),
                "k": list(k.shape),
                "v": list(v.shape),
                "cis": list(cis.shape),
            }
        )
        if self.verbose:
            print(f"Captured {self.selection_mode} layer {context.layer_idx}", flush=True)
        return output


@torch.inference_mode()
def capture_extend(
    model, ids, cache, prefix_tokens, *, mode="dense", backend="reference", verbose=False
):
    """Record a suffix; the caller must construct its prefix using the same trajectory."""
    if mode not in ("dense", "sparse"):
        raise ValueError("mode must be dense or sparse")
    if cache.length != prefix_tokens or prefix_tokens <= 0:
        raise ValueError("Capture requires an independently completed prefix cache")
    if (model.indexer is None) != (mode == "dense"):
        raise ValueError("The model indexer must match the selected dense/sparse trajectory")
    observers = (
        {
            "dense_qa64": NosaIndexer(block_budget=64, query_chunk_size=64),
            "dense_nosa64": NosaIndexer(mode="nosa", backend=backend),
        }
        if mode == "dense"
        else None
    )
    recorder = PatternRecorder(
        model.main_attention,
        model.config,
        prefix_tokens,
        ids.numel(),
        indexers=observers,
        selection_mode="observe" if mode == "dense" else "actual",
        verbose=verbose,
    )
    with patch.object(model, "main_attention", recorder):
        hidden = run_chunks(model, ids, cache, ids.numel(), output="hidden")
    if (
        len(recorder.attention_shapes) != model.config.num_hidden_layers
        or cache.length != prefix_tokens + ids.numel()
    ):
        raise ValueError("Candidate capture did not complete all layers and commit the suffix")
    return recorder, hidden


@torch.inference_mode()
def run_trajectory(model, prefix_ids, new_ids, *, mode, backend="triton"):
    """Use a fresh cache and verify that observation does not change the suffix output."""
    with ExitStack() as stack:
        if mode == "dense":
            stack.enter_context(patch.object(model, "indexer", None))
            stack.enter_context(
                patch.object(model, "main_attention", DenseMainAttention(model.attention))
            )
        elif mode != "sparse":
            raise ValueError("mode must be dense or sparse")
        cache = model.new_cache(prefix_ids.numel() + new_ids.numel())
        stack.callback(model.cache_manager.release, cache)
        print(f"{mode} prefix: {prefix_ids.numel()} tokens from an empty cache", flush=True)
        run_chunks(model, prefix_ids, cache, CHUNK, output="hidden")
        recorder, hidden = capture_extend(
            model, new_ids, cache, prefix_ids.numel(), mode=mode, backend=backend, verbose=True
        )
        if not torch.isfinite(hidden).all().item():
            raise ValueError(f"{mode} candidate hidden contains nonfinite values")
        if any(cache.get_layer_state(i) is not None for i in range(model.config.num_hidden_layers)):
            raise ValueError("Replay cannot rewind opaque model state")
        cache.truncate(prefix_ids.numel())
        replay = run_chunks(model, new_ids, cache, NEW, output="hidden")
        if not torch.equal(hidden, replay):
            raise ValueError(f"Pattern observation changed {mode} candidate hidden states")
        return recorder, hidden.float().cpu().numpy()


def main(argv=None):
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--request-file", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    request_bytes = args.request_file.read_bytes()
    request = json.loads(request_bytes)
    validate_request(request, PREFIX, NEW)
    prefix, new = execution_split(request, PREFIX, NEW)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("The full pattern comparison requires CUDA/SM90 and FlashInfer")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (9, 0):
        parser.error("The sparse backend requires SM90")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print("Loading NOSA with strict A/delta weights and resident K/V/CIS", flush=True)
    model = NosaForCausalLM.from_pretrained(
        args.model_path,
        device=device,
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
        parser.error("This comparison requires NOSA-8B: 32 layers, 32 Q heads, 2 KV heads, D128")
    if max(request["input_ids"]) >= config.vocab_size:
        parser.error("Request exceeds checkpoint vocabulary")
    model.config = replace(
        config, max_position_embeddings=max(config.max_position_embeddings, PREFIX + NEW)
    )
    prefix_ids = torch.tensor(prefix, device=device)
    new_ids = torch.tensor(new, device=device)
    recorders, hidden = {}, {}
    for mode in ("dense", "sparse"):
        recorders[mode], hidden[mode] = run_trajectory(model, prefix_ids, new_ids, mode=mode)
    if not torch.equal(
        recorders["dense"].block_ids["dense_nosa64"][0],
        recorders["sparse"].block_ids["sparse_nosa64"][0],
    ):
        raise ValueError(
            "The unchanged first-layer Q/K/CIS must yield identical full NOSA selections"
        )
    arms = {}
    for mode, recorder in recorders.items():
        for name in recorder.block_ids:
            indexer = recorder.indexers[name] if mode == "dense" else model.indexer
            ids = torch.stack(recorder.block_ids[name]).numpy()
            valid = torch.stack(recorder.valid_masks[name]).numpy()
            if ids.shape != (32, NEW, 2, 64) or not valid.all():
                raise ValueError("Every candidate must select exactly 64 valid blocks")
            union, _ = summarize(
                ids,
                valid,
                prefix_tokens=PREFIX,
                total_tokens=PREFIX + NEW,
                head_dim=128,
                element_size=2,
            )
            if not ((union.sum(-1) >= 79) & (union.sum(-1) <= 1040)).all():
                raise ValueError("Candidate union exceeds its causal 79..1040 block bounds")
            arm = args.output_dir / "arms" / name
            arm.mkdir(parents=True)
            np.save(arm / "block_ids.npy", ids, allow_pickle=False)
            np.save(arm / "valid_mask.npy", valid, allow_pickle=False)
            arms[name] = {
                "label": {
                    "dense_qa64": "Dense + QA-only64",
                    "dense_nosa64": "Dense + full NOSA64",
                    "sparse_nosa64": "Sparse + full NOSA64",
                }[name],
                "activation_source": "dense_full_attention_post_rope"
                if mode == "dense"
                else "resident_block_sparse_attention_with_cis_post_rope",
                "prefix_propagation": mode,
                "candidate_propagation": mode,
                "indexer_mode": indexer.mode,
                "indexer_backend": indexer.backend,
                "indexer_policy": asdict(indexer.policy),
                "query_stage_blocks": 33 if indexer.mode == "nosa" else None,
                "selection_source": "sidecar_same_dense_activations"
                if mode == "dense"
                else "actual_attention_input_without_reselection",
            }
        np.save(args.output_dir / f"{mode}_candidate_hidden.npy", hidden[mode], allow_pickle=False)
    execution = {
        "prefix_tokens": PREFIX,
        "new_tokens": NEW,
        "total_tokens": PREFIX + NEW,
        "prefix_chunk_size": CHUNK,
        "candidate_forward_calls": 1,
        "instruction_tokens": request["instruction_tokens"],
        "history_tokens": request["user_tokens"],
        "prefix_span": [0, PREFIX],
        "extend_span": [PREFIX, PREFIX + NEW],
    }
    own = Path(__file__).resolve().parent
    sources = source_hashes(
        *own.glob("*.py"),
        *own.parent.joinpath("scripts").glob("*.sh"),
        ROOT / "experiments/indexer_block_sparse_profile/src/capture.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    )
    properties = torch.cuda.get_device_properties(device)
    files = sorted(args.model_path.glob("*.safetensors")) + [
        args.model_path / "config.json",
        args.model_path / "tokenizer.json",
    ]
    print("Fingerprinting checkpoint and captured inference sources", flush=True)
    metadata = {
        "schema_version": 1,
        "run_id": args.run_id,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source_run_id": args.request_file.resolve().parent.name,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "input_ids_sha256": hashlib.sha256(
            np.asarray(request["input_ids"], dtype="<i8").tobytes()
        ).hexdigest(),
        "model_config": asdict(model.config),
        "original_context": config.max_position_embeddings,
        "execution": execution,
        "dtype": "bfloat16",
        "element_size": 2,
        "block_size": 64,
        "weight_loading_mode": "sparse_with_strict_cis_weights",
        "arms": arms,
        "attention_shapes": {
            mode: recorder.attention_shapes for mode, recorder in recorders.items()
        },
        "indexer_query_chunk_sizes": {
            "dense_qa64": 64,
            "dense_nosa64": None,
            "sparse_nosa64": None,
        },
        "metric": "unique_selected_full_block_kv_payload_excluding_cis",
        "warmup": 0,
        "captures_per_trajectory": 1,
        "validation_replays_per_trajectory": 1,
        "performance_measured": False,
        "validation": {
            "hidden_finite": True,
            "dense_observer_hidden_max_abs": 0.0,
            "sparse_observer_hidden_max_abs": 0.0,
            "full_nosa_layer0_selection_equal": True,
        },
        "gpu": {
            "name": properties.name,
            "uuid": str(properties.uuid),
            "sm_count": properties.multi_processor_count,
            "capability": list(torch.cuda.get_device_capability(device)),
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in (
                "flashinfer-python",
                "triton",
                "numpy",
                "matplotlib",
                "safetensors",
                "tokenizers",
            )
        },
        "nvidia_smi": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"],
            text=True,
        ).strip(),
        "checkpoint_sha256": {file.name: sha256_file(file) for file in files},
        "source_sha256": sources,
    }
    (args.output_dir / "request.json").write_bytes(request_bytes)
    write_json(args.output_dir / "metadata.json", metadata)
    write_json(args.output_dir / "execution.json", execution)
    for relative in sources:
        target = args.output_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    print("Completed three-arm capture on independent dense/sparse prefix caches", flush=True)


if __name__ == "__main__":
    main()
