"""Record query-aware block choices on one GR request's dense NOSA trajectory.

This module does not run sparse attention or measure kernel/transfer performance.
Use scripts/run.sh to stage both capture and analysis outside the experiment
until they have completed successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from executor.model_executor import run_chunks
from experiments.nosa_gr_65536_1024.src.capture import execution_split
from experiments.nosa_gr_65536_1024.src.sources import source_hashes
from models.nosa.indexer import NosaIndexer
from models.nosa.infer import DEFAULT_MODEL_PATH
from models.nosa.model import NosaForCausalLM

ROOT = Path(__file__).resolve().parents[3]
SOURCE_RUN = "query_aware_fp32_65536_1024_20260925_01"
DEFAULT_REQUEST = (
    ROOT / "experiments/nosa_indexer_pattern_65536_1024/output/data" / SOURCE_RUN / "request.json"
)
UPSTREAM_COMMIT = "1cbee77d607f9051b206a09c862bea28becb9e67"


class CaptureAttention:
    """Observe selections without passing them to the dense attention backend."""

    def __init__(
        self,
        attention,
        config,
        prefix_tokens,
        query_length,
        *,
        indexer=None,
        verbose=False,
        block_budget=64,
    ):
        if type(block_budget) is not int or block_budget not in (32, 64):
            raise ValueError("block_budget must be 32 or 64")
        self.attention = attention
        self.config = config
        self.prefix_tokens = prefix_tokens
        self.query_length = query_length
        self.block_budget = block_budget
        self.indexer = (
            NosaIndexer(block_budget=block_budget, query_chunk_size=64)
            if indexer is None
            else indexer
        )
        self.verbose = verbose
        self.block_ids = []
        self.valid_masks = []
        self.attention_shapes = []

    def __call__(self, q, selection, cache_access, context):
        if selection is not None:
            raise ValueError("Pattern capture requires dense forward with model.indexer=None")
        config = self.config
        if (
            context.layer_idx != len(self.block_ids)
            or context.query_start != self.prefix_tokens
            or context.query_length != self.query_length
        ):
            raise ValueError("Capture must visit each layer once at the candidate boundary")
        records = cache_access.layer_view(context.layer_idx)
        k, v = records["keys"], records["values"]
        expected_q = (self.query_length, config.num_attention_heads, config.head_dim)
        expected_kv = (
            self.prefix_tokens + self.query_length,
            config.num_key_value_heads,
            config.head_dim,
        )
        if tuple(q.shape) != expected_q or tuple(k.shape) != expected_kv or v.shape != k.shape:
            raise ValueError("Actual Q/K/V shapes disagree with the candidate boundary")
        chosen = self.indexer(q, cache_access, context)
        ids = chosen.block_ids.detach().to(device="cpu", dtype=torch.int32).clone()
        valid = (
            torch.ones_like(ids, dtype=torch.bool)
            if chosen.valid_mask is None
            else chosen.valid_mask.detach().to(device="cpu").clone()
        )
        if chosen.block_size != 64 or ids.shape != (
            self.query_length,
            expected_kv[1],
            self.block_budget,
        ):
            raise ValueError(
                f"Expected the NOSA 64-token / {self.block_budget}-block selection contract"
            )
        if valid.shape != ids.shape or valid.dtype != torch.bool:
            raise ValueError("Selection validity mask must match block_ids")
        self.block_ids.append(ids)
        self.valid_masks.append(valid)
        self.attention_shapes.append(
            {
                "layer": context.layer_idx,
                "query_start": context.query_start,
                "q": list(q.shape),
                "k": list(k.shape),
                "v": list(v.shape),
            }
        )
        if self.verbose:
            print(
                f"Captured layer {context.layer_idx}: Q={list(q.shape)}, K={list(k.shape)}",
                flush=True,
            )
        return self.attention(q, None, cache_access, context)


@torch.inference_mode()
def capture_extend(
    model, ids, cache, prefix_tokens, *, indexer=None, verbose=False, block_budget=64
):
    """Run a single candidate extend and restore the original attention callable."""
    if model.indexer is not None:
        raise ValueError("Pattern capture requires model.indexer=None")
    if cache.length != prefix_tokens or prefix_tokens <= 0:
        raise ValueError("Capture requires a completed prefix cache")
    recorder = CaptureAttention(
        model.main_attention,
        model.config,
        prefix_tokens,
        ids.numel(),
        indexer=indexer,
        verbose=verbose,
        block_budget=block_budget,
    )
    with patch.object(model, "main_attention", recorder):
        hidden = run_chunks(model, ids, cache, ids.numel())
    if (
        len(recorder.block_ids) != model.config.num_hidden_layers
        or cache.length != prefix_tokens + ids.numel()
    ):
        raise ValueError("Capture did not complete every layer and commit the candidate suffix")
    return recorder, hidden


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--request-file", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--block-budget",
        type=int,
        choices=(32, 64),
        default=64,
        help="Total selected blocks: 1 sink + 16 local + (budget - 17) query-aware",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New staging data directory")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.request_file.is_file():
        parser.error(f"Exact source request is missing: {args.request_file}")
    request_bytes = args.request_file.read_bytes()
    request = json.loads(request_bytes)
    prefix_ids, new_ids = execution_split(request, 65536, 1024)
    if request.get("model") != "nosa":
        parser.error("Expected a NOSA GR request")
    if any(type(token) is not int or token < 0 for token in request["input_ids"]):
        parser.error("Request input_ids must be nonnegative integers")
    if request.get("attention_mask", [1] * 66560) != [1] * 66560:
        parser.error("Only one unpadded request is supported")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("The full NOSA capture requires CUDA and FlashInfer")
    torch.cuda.set_device(device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print("Loading NOSA weights for the exact 65536 + 1024 GR request", flush=True)
    model = NosaForCausalLM.from_pretrained(args.model_path, device=device, dtype=torch.bfloat16)
    config = model.config
    if (
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    ) != (32, 32, 2, 128):
        parser.error("This experiment requires NOSA-8B: 32 layers, 32 Q heads, 2 KV heads, d=128")
    if max(request["input_ids"]) >= config.vocab_size:
        parser.error("Request token ID exceeds the checkpoint vocabulary")
    original_context = config.max_position_embeddings
    model.config = replace(config, max_position_embeddings=max(original_context, 66560))
    cache = model.new_cache(66560)
    try:
        print("Dense prefix prefill: 64 chunks of 1024 tokens", flush=True)
        run_chunks(model, torch.tensor(prefix_ids, device=device), cache, 1024)
        recorder, hidden = capture_extend(
            model,
            torch.tensor(new_ids, device=device),
            cache,
            65536,
            verbose=True,
            block_budget=args.block_budget,
        )
        if not torch.isfinite(hidden).all().item():
            raise ValueError("Dense candidate hidden states contain nonfinite values")
        block_ids = torch.stack(recorder.block_ids).numpy()
        valid_mask = torch.stack(recorder.valid_masks).numpy()
        if block_ids.shape != (32, 1024, 2, args.block_budget) or not valid_mask.all():
            raise ValueError(
                f"Every candidate query / KV head must select {args.block_budget} valid blocks"
            )
        # Full validation is repeated by analyze; do not retain a malformed run.
        from experiments.nosa_indexer_pattern_65536_1024.src.analyze import summarize

        union, _ = summarize(
            block_ids,
            valid_mask,
            prefix_tokens=65536,
            total_tokens=66560,
            head_dim=128,
            element_size=2,
        )
        union_counts = union.sum(axis=-1)
        # The first query's budget plus the 15 later candidate blocks that its
        # causal selection cannot yet contain: 79 for budget64, 47 for budget32.
        minimum_union = args.block_budget + 15
        if not ((union_counts >= minimum_union) & (union_counts <= 1040)).all():
            raise ValueError(
                f"Union violates the {minimum_union}..1040 block bound for 1024 candidate queries"
            )
        np.save(args.output_dir / "block_ids.npy", block_ids, allow_pickle=False)
        np.save(args.output_dir / "valid_mask.npy", valid_mask, allow_pickle=False)
    finally:
        model.cache_manager.release(cache)
    execution = {
        "prefix_tokens": 65536,
        "new_tokens": 1024,
        "total_tokens": 66560,
        "prefix_chunk_size": 1024,
        "candidate_forward_calls": 1,
        "instruction_tokens": request["instruction_tokens"],
        "history_tokens": request["user_tokens"],
        "prefix_span": [0, 65536],
        "extend_span": [65536, 66560],
    }
    own_directory = Path(__file__).resolve().parent
    sources = source_hashes(
        *own_directory.glob("*.py"),
        own_directory.parent / "scripts/run.sh",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    )
    print("Fingerprinting checkpoint and recording reproducibility metadata", flush=True)
    checkpoint_files = sorted(args.model_path.glob("*.safetensors"))
    checkpoint_files += [args.model_path / "config.json", args.model_path / "tokenizer.json"]
    metadata = {
        "run_id": args.run_id,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "source_run_id": args.request_file.resolve().parent.name,
        "source_request_file": str(args.request_file.resolve()),
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "input_ids_sha256": hashlib.sha256(
            np.asarray(request["input_ids"], dtype="<i8").tobytes()
        ).hexdigest(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "execution": execution,
        "model_config": asdict(model.config),
        "original_context": original_context,
        "dtype": "bfloat16",
        "element_size": 2,
        "block_size": 64,
        "indexer_policy": asdict(recorder.indexer.policy),
        "indexer_compute_dtype": "float32",
        "indexer_query_chunk_size": 64,
        "tie_break": "smaller_block_id",
        "activation_source": "dense_full_attention_post_rope",
        "metric": "unique_selected_full_block_kv_payload",
        "warmup": 0,
        "captures": 1,
        "performance_measured": False,
        "upstream_commit": UPSTREAM_COMMIT,
        "paper": "https://arxiv.org/abs/2510.13602v2",
        "attention_shapes": recorder.attention_shapes,
        "validation": {
            "hidden_finite": True,
            "selected_blocks_per_query": args.block_budget,
            "all_queries_select_budget_unique_causal_blocks": True,
            "union_lower_bound": minimum_union,
            "union_min": int(union_counts.min()),
            "union_max": int(union_counts.max()),
        },
        "gpu": str(torch.cuda.get_device_properties(device)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("flashinfer-python", "numpy", "matplotlib", "safetensors", "tokenizers")
        },
        "nvidia_smi": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"],
            text=True,
        ).strip(),
        "checkpoint_sha256": {path.name: sha256_file(path) for path in checkpoint_files},
        "source_sha256": sources,
    }
    (args.output_dir / "request.json").write_bytes(request_bytes)
    write_json(args.output_dir / "execution.json", execution)
    write_json(args.output_dir / "attention_shapes.json", recorder.attention_shapes)
    write_json(args.output_dir / "metadata.json", metadata)
    for relative in sources:
        target = args.output_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    print(
        f"Capture complete: block_ids={block_ids.shape}, union range={union_counts.min()}..{union_counts.max()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
