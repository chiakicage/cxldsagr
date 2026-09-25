"""Measure warmed single-request NOSA full attention on GR-generated inputs.

Run from the repository root with ``python -m experiments.legacy.nosa_generation.src.measure``.
Fixed-length greedy decoding deliberately continues past EOS for comparable timings.
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

from GR.input_generator import TextConfig, create_input_generator
from GR.scheduling import ScheduleConfig
from models.nosa.infer import DEFAULT_MODEL_PATH, load_tokenizer
from models.nosa.model import NosaConfig, NosaForCausalLM


@torch.inference_mode()
def measure_request(model, ids, *, output_tokens, chunk_size):
    """Synchronized wall time; exclude input transfer and cache allocation."""
    device = ids.device
    cache = model.new_cache(ids.numel() + output_tokens)
    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for offset in range(0, ids.numel(), chunk_size):
            logits = model(ids[offset : offset + chunk_size], cache, logits_to_keep=1)[-1]
        torch.cuda.synchronize(device)
        prefill_end = time.perf_counter()
        generated = [int(logits.argmax().item())]
        first_token = time.perf_counter()
        for _ in range(output_tokens - 1):
            token = torch.tensor([generated[-1]], dtype=torch.long, device=device)
            logits = model(token, cache, logits_to_keep=1)[-1]
            generated.append(int(logits.argmax().item()))
        torch.cuda.synchronize(device)
        end = time.perf_counter()
        eos = set(model.config.eos_token_id)
        first_eos = next((i for i, token in enumerate(generated) if token in eos), None)
        return {
            "prefill_ms": (prefill_end - start) * 1000,
            "ttft_ms": (first_token - start) * 1000,
            "decode_ms": (end - first_token) * 1000,
            "tpot_ms": (end - first_token) * 1000 / (output_tokens - 1),
            "decode_tokens_per_second": (output_tokens - 1) / (end - first_token),
            "prefill_tokens_per_second": ids.numel() / (prefill_end - start),
            "request_ms": (end - start) * 1000,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "kv_capacity_mib": (cache.keys.nbytes + cache.values.nbytes) / 2**20,
            "generated_ids": generated,
            "first_eos_index": first_eos,
            "decode_steps": output_tokens - 1,
        }
    finally:
        model.cache_manager.release(cache)


def summarize(rows):
    metrics = (
        "prefill_ms",
        "ttft_ms",
        "decode_ms",
        "tpot_ms",
        "decode_tokens_per_second",
        "prefill_tokens_per_second",
        "request_ms",
        "peak_allocated_gib",
        "peak_reserved_gib",
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
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/legacy/nosa_generation/output/data/run"),
    )
    args = parser.parse_args()
    if args.output_tokens < 2 or min(args.prefill_chunk_size, args.warmup, args.repeats) < 1:
        parser.error("output-tokens must be >= 2; chunk size, warmup and repeats must be positive")
    config = NosaConfig.from_pretrained(args.model_path)
    max_input = config.max_position_embeddings - args.output_tokens
    if min(args.user_lengths + args.item_lengths) < 1:
        parser.error("input lengths must be positive")
    if max(args.user_lengths) + max(args.item_lengths) > max_input:
        parser.error("input + output exceeds model context")
    # Refuse to overwrite a previous measurement.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    tokenizer, _ = load_tokenizer(args.model_path)
    metadata = {
        "started_at": datetime.now(UTC).isoformat(),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "experiments/legacy/nosa_generation/src/measure.py",
                "models/nosa/model.py",
                "GR/input_generator.py",
                "GR/analysis/heat_curves.csv",
            )
        },
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
            "prefix_reuse": False,
            "timing": "synchronized wall clock; includes Python and greedy sampling; excludes GR generation, input H2D, cache allocation, weight loading",
            "output": "fixed length, ignore EOS; decode steps = output tokens - 1",
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
                        output_tokens=args.output_tokens,
                        chunk_size=args.prefill_chunk_size,
                    )
                    row.update(
                        request_key=request_key,
                        warmup=warmup,
                        user_tokens=history,
                        item_tokens=item,
                        input_tokens=ids.numel(),
                        user_id=request["user_id"],
                    )
                    stop = row["first_eos_index"]
                    row["text_before_eos"] = tokenizer.decode(
                        row["generated_ids"][:stop], skip_special_tokens=True
                    )
                    measurements_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    measurements_file.flush()
                    if not warmup:
                        rows.append(row)
                    print(
                        f"{request_key} {'warmup' if warmup else 'measure'}: prefill={row['prefill_ms']:.1f}ms decode={row['decode_tokens_per_second']:.2f}tok/s",
                        flush=True,
                    )
                summaries.append(
                    {
                        "user_tokens": history,
                        "item_tokens": item,
                        "input_tokens": history + item,
                        "repeats": len(rows),
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
