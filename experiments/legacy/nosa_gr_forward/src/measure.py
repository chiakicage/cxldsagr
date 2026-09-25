"""Measure warmed single-request NOSA full attention on GR-generated inputs.

Run from the repository root with ``python -m experiments.legacy.nosa_gr_forward.src.measure``.
Measure backbone features only: full prefill, prefix prefill and candidate extend.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from executor.model_executor import run_chunks
from experiments.nosa_gr_65536_1024.src.sources import source_hashes
from GR.input_generator import TextConfig, create_input_generator
from GR.scheduling import ScheduleConfig
from models.nosa.infer import DEFAULT_MODEL_PATH
from models.nosa.model import NosaConfig, NosaForCausalLM


@torch.inference_mode()
def measure_stage(model, ids, cache, chunk_size):
    device = ids.device
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    hidden = run_chunks(model, ids, cache, chunk_size)
    torch.cuda.synchronize(device)
    ms = (time.perf_counter() - start) * 1000
    stats = {
        "ms": ms,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
    return stats, hidden[-1].float().clone()


@torch.inference_mode()
def measure_request(model, ids, *, prefix_tokens, chunk_size):
    """Full request and cached-prefix candidate forward, without a generation loop."""
    if not 0 < prefix_tokens < ids.numel():
        raise ValueError("prefix must leave a nonempty candidate suffix")
    cache = model.new_cache(ids.numel())
    try:
        full, full_hidden = measure_stage(model, ids, cache, chunk_size)
        cache.reset()
        prefix, _ = measure_stage(model, ids[:prefix_tokens], cache, chunk_size)
        extend, extend_hidden = measure_stage(model, ids[prefix_tokens:], cache, chunk_size)
        assert cache.length == ids.numel()
        if not torch.isfinite(full_hidden).all() or not torch.isfinite(extend_hidden).all():
            raise RuntimeError("nonfinite backbone output")
        result = {}
        for name, stats in (
            ("full_prefill", full),
            ("prefix_prefill", prefix),
            ("candidate_extend", extend),
        ):
            for metric, value in stats.items():
                result[f"{name}_{metric}"] = value
        result.update(
            kv_capacity_mib=cache.stats()["capacity_bytes"] / 2**20,
            full_vs_split_last_hidden_max_abs=float((full_hidden - extend_hidden).abs().max()),
            full_vs_split_last_hidden_cosine=float(
                torch.nn.functional.cosine_similarity(full_hidden, extend_hidden, dim=0)
            ),
        )
        return result
    finally:
        model.cache_manager.release(cache)


def summarize(rows):
    metrics = tuple(
        f"{phase}_{metric}"
        for phase in ("full_prefill", "prefix_prefill", "candidate_extend")
        for metric in ("ms", "peak_allocated_gib", "peak_reserved_gib")
    )
    return {
        key: {
            "median": statistics.median(values := [row[key] for row in rows]),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
        }
        for key in metrics
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--user-lengths", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument(
        "--item-lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096]
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.prefill_chunk_size, args.warmup, args.repeats) < 1:
        parser.error("chunk size, warmup and repeats must be positive")
    config = NosaConfig.from_pretrained(args.model_path)
    max_input = config.max_position_embeddings
    if min(args.user_lengths + args.item_lengths) < 1:
        parser.error("input lengths must be positive")
    if max(args.user_lengths) + max(args.item_lengths) > max_input:
        parser.error("input exceeds model context")
    # Refuse to overwrite a previous measurement.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    metadata = {
        "started_at": datetime.now(UTC).isoformat(),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": source_hashes(*Path(__file__).parent.glob("*.py")),
        "checkpoint_metadata_sha256": {
            name: hashlib.sha256((args.model_path / name).read_bytes()).hexdigest()
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
        },
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "flashinfer-python", "tokenizers", "safetensors")
        },
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "capability": torch.cuda.get_device_capability(device),
        "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
        "semantics": {
            "attention": "FlashInfer full causal GQA; NOSA A/delta disabled",
            "dtype": "bfloat16",
            "batch_size": 1,
            "prefix_reuse": "candidate_extend starts from completed stable prefix KV; full_prefill starts empty",
            "timing": "synchronized wall clock; includes Python and model backbone; excludes GR generation, input H2D, cache allocation, weight loading",
            "output": "normalized hidden states; no LM head, sampling, autoregressive decode, TTFT or TPOT",
            "schedule": "Beauty curve, 1000 synthetic users, weighted, seed as configured; timestamps saved but no arrival replay or queueing",
            "warmup": "one or more full requests per shape; excluded from summary",
        },
    }
    start = time.perf_counter()
    model = NosaForCausalLM.from_pretrained(args.model_path, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize(device)
    metadata["load_seconds"] = time.perf_counter() - start
    metadata["ignored_checkpoint_keys"] = list(model.ignored_checkpoint_keys)
    metadata["model_parameter_gib"] = (
        sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
    )
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Loaded model in {metadata['load_seconds']:.2f}s", flush=True)
    summaries = []
    with (
        (args.output_dir / "requests.jsonl").open("w") as requests_file,
        (args.output_dir / "measurements.jsonl").open("w") as measurements_file,
    ):
        for history in args.user_lengths:
            for item in args.item_lengths:
                start = time.perf_counter()
                generator = create_input_generator(
                    model="nosa",
                    tokenizer=args.model_path,
                    num_users=1000,
                    text_config=TextConfig(
                        user_lengths=(history,),
                        user_probabilities=(1.0,),
                        item_lengths=(item,),
                        item_probabilities=(1.0,),
                        max_input_tokens=max_input,
                    ),
                    schedule_config=ScheduleConfig(seed=args.seed, sampling="weighted"),
                )
                requests = list(generator.iter_generate(args.warmup + args.repeats))
                generation_seconds = time.perf_counter() - start
                rows = []
                for index, request in enumerate(requests):
                    request_key = f"h{history}_i{item}_r{index}"
                    warmup = index < args.warmup
                    requests_file.write(
                        json.dumps({"request_key": request_key, "warmup": warmup, **request}) + "\n"
                    )
                    requests_file.flush()
                    ids = torch.tensor(request["input_ids"], dtype=torch.long, device=device)
                    if ids.numel() != history + item:
                        raise ValueError("GR input length mismatch")
                    row = measure_request(
                        model,
                        ids,
                        prefix_tokens=request["stable_prefix_tokens"],
                        chunk_size=args.prefill_chunk_size,
                    )
                    row.update(
                        request_key=request_key,
                        warmup=warmup,
                        user_tokens=history,
                        item_tokens=item,
                        input_tokens=ids.numel(),
                        user_id=request["user_id"],
                        prefix_tokens=request["stable_prefix_tokens"],
                        candidate_tokens=request["candidate_suffix_tokens"],
                    )
                    measurements_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    measurements_file.flush()
                    if not warmup:
                        rows.append(row)
                    print(
                        f"{request_key} {'warmup' if warmup else 'measure'}: full={row['full_prefill_ms']:.1f}ms prefix={row['prefix_prefill_ms']:.1f}ms extend={row['candidate_extend_ms']:.1f}ms",
                        flush=True,
                    )
                summaries.append(
                    {
                        "user_tokens": history,
                        "item_tokens": item,
                        "input_tokens": history + item,
                        "repeats": len(rows),
                        "prefix_tokens": rows[0]["prefix_tokens"],
                        "candidate_tokens": rows[0]["candidate_tokens"],
                        "gr_generation_seconds": generation_seconds,
                        "kv_capacity_mib": rows[0]["kv_capacity_mib"],
                        "metrics": summarize(rows),
                    }
                )
                (args.output_dir / "summary.json").write_text(
                    json.dumps(summaries, indent=2) + "\n"
                )
    metadata["finished_at"] = datetime.now(UTC).isoformat()
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
