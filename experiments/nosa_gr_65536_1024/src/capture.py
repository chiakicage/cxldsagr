"""Warm NOSA GR forward work, then capture light and module-annotated nsys ranges.

The context override is confined to this experiment and its GR generator.
Use nsys --capture-range=cudaProfilerApi --capture-range-end=stop.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import subprocess
import time
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import torch

from executor.model_executor import run_chunks
from experiments.nosa_gr_65536_1024.src.instrumentation import ModuleScopes
from experiments.nosa_gr_65536_1024.src.sources import source_hashes
from GR.input_generator import INSTRUCTION, TextConfig, create_input_generator
from GR.scheduling import ScheduleConfig
from models.nosa import request_format
from models.nosa.infer import DEFAULT_MODEL_PATH, load_tokenizer
from models.nosa.model import NosaForCausalLM


class NVTXScopes(ModuleScopes):
    def scope(self, category, layer="shared"):
        return torch.cuda.nvtx.range(f"nosa::{self.phase}/{category}/{layer}")


def execution_split(request, prefix_tokens, new_tokens):
    """Require the execution boundary to coincide with the GR candidate boundary."""
    if prefix_tokens <= 0 or new_tokens <= 0:
        raise ValueError("execution lengths must be positive")
    ids = request["input_ids"]
    if len(ids) != prefix_tokens + new_tokens:
        raise ValueError("input length must equal execution prefix + new tokens")
    if (
        request["stable_prefix_tokens"] != prefix_tokens
        or request["candidate_suffix_tokens"] != new_tokens
    ):
        raise ValueError("GR semantic boundary must equal the execution boundary")
    return ids[:prefix_tokens], ids[prefix_tokens:]


@torch.inference_mode()
def audit_extend(model, ids, cache, prefix_tokens, chunk_size):
    """Record actual per-layer attention shapes outside measured regions."""
    if ids.numel() > chunk_size:
        raise ValueError("extend must execute as one chunk")
    calls = []
    attention = model.attention

    def checked_attention(q, k, v):
        expected_kv = prefix_tokens + ids.numel()
        if q.shape[0] != ids.numel() or k.shape[0] != expected_kv or v.shape[0] != expected_kv:
            raise ValueError("actual attention shapes disagree with execution split")
        calls.append(
            {"layer": len(calls), "q": list(q.shape), "k": list(k.shape), "v": list(v.shape)}
        )
        return attention(q, k, v)

    cache.length = prefix_tokens
    with patch.object(model, "attention", checked_attention):
        hidden = run_chunks(model, ids, cache, chunk_size)
    if len(calls) != model.config.num_hidden_layers or cache.length != prefix_tokens + ids.numel():
        raise ValueError("layer count or final KV length mismatch")
    return calls, hidden


@torch.inference_mode()
def timed_forward(model, ids, cache, prefix_length, chunk_size):
    cache.length = prefix_length
    torch.cuda.synchronize(ids.device)
    start = time.perf_counter()
    hidden = run_chunks(model, ids, cache, chunk_size)
    submitted = time.perf_counter()
    torch.cuda.synchronize(ids.device)
    end = time.perf_counter()
    if cache.length != prefix_length + ids.numel():
        raise ValueError("forward did not consume exactly the requested tokens")
    return hidden, {"wall_ms": (end - start) * 1000, "host_submit_ms": (submitted - start) * 1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--prefix-tokens", type=int, default=65536, help="Instruction + history cached tokens"
    )
    parser.add_argument(
        "--new-tokens",
        type=int,
        default=1024,
        help="Candidate suffix tokens, excluding instruction",
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/nosa_gr_65536_1024/output/data/run")
    )
    args = parser.parse_args()
    if min(args.prefix_tokens, args.new_tokens, args.chunk_size, args.warmup, args.repeats) < 1:
        parser.error("lengths and repetition counts must be positive")
    if args.chunk_size < args.new_tokens:
        parser.error("chunk-size must cover all extend tokens in one forward")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    tokenizer, _ = load_tokenizer(args.model_path)
    instruction_tokens = len(
        tokenizer.encode(request_format.prefix(INSTRUCTION), add_special_tokens=False).ids
    )
    if args.prefix_tokens <= instruction_tokens:
        parser.error("prefix must have room for instruction and history")
    history_tokens = args.prefix_tokens - instruction_tokens
    gr_item_budget = args.new_tokens + instruction_tokens
    total = args.prefix_tokens + args.new_tokens
    with patch.object(request_format, "MAX_INPUT_TOKENS", total):
        generator = create_input_generator(
            model="nosa",
            tokenizer=tokenizer,
            num_users=1000,
            text_config=TextConfig(
                user_lengths=(history_tokens,),
                user_probabilities=(1.0,),
                item_lengths=(gr_item_budget,),
                item_probabilities=(1.0,),
                max_input_tokens=total,
            ),
            schedule_config=ScheduleConfig(seed=42, sampling="weighted"),
        )
        request = next(generator.iter_generate(1))
    prefix_tokens = args.prefix_tokens
    assert request["user_tokens"] == history_tokens
    assert request["item_tokens"] == gr_item_budget
    assert request["total_input_tokens"] == total
    assert request["stable_prefix_tokens"] == args.prefix_tokens
    assert request["candidate_suffix_tokens"] == args.new_tokens
    prefix_ids, new_ids = execution_split(request, prefix_tokens, args.new_tokens)
    execution = {
        "prefix_tokens": len(prefix_ids),
        "new_tokens": len(new_ids),
        "total_tokens": total,
        "prefix_span": [0, prefix_tokens],
        "extend_span": [prefix_tokens, total],
        "gr_stable_prefix_tokens": request["stable_prefix_tokens"],
        "gr_candidate_suffix_tokens": request["candidate_suffix_tokens"],
        "instruction_tokens": instruction_tokens,
        "history_tokens": history_tokens,
        "gr_item_budget_including_instruction": gr_item_budget,
        "history_tail_tokens_in_extend": 0,
    }
    (args.output_dir / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    print("Execution boundary:", json.dumps(execution), flush=True)
    (args.output_dir / "request.json").write_text(json.dumps(request) + "\n")
    model = NosaForCausalLM.from_pretrained(args.model_path, device=device, dtype=torch.bfloat16)
    original_context = model.config.max_position_embeddings
    model.config = replace(model.config, max_position_embeddings=max(total, original_context))
    ids = torch.tensor(request["input_ids"], device=device, dtype=torch.long)
    extend_ids = ids[prefix_tokens:]
    assert extend_ids.numel() == args.new_tokens
    with ExitStack() as resources:
        full_cache = model.new_cache(total)
        resources.callback(model.cache_manager.release, full_cache)
        prefix_cache = model.new_cache(total)
        resources.callback(model.cache_manager.release, prefix_cache)
        with torch.inference_mode():
            run_chunks(model, ids[:prefix_tokens], prefix_cache, args.chunk_size)
        assert prefix_cache.length == prefix_tokens
        shapes, _ = audit_extend(model, extend_ids, prefix_cache, prefix_tokens, args.chunk_size)
        (args.output_dir / "attention_shapes.json").write_text(json.dumps(shapes, indent=2) + "\n")
        timings = {"full_prefill": [], "extend": []}
        work = (
            ("full_prefill", ids, full_cache, 0),
            ("extend", extend_ids, prefix_cache, prefix_tokens),
        )
        outputs = {}
        for phase, tokens, cache, prefix in work:
            for index in range(args.warmup + args.repeats):
                hidden, stats = timed_forward(model, tokens, cache, prefix, args.chunk_size)
                if index >= args.warmup:
                    timings[phase].append(stats)
            outputs[phase] = hidden[-1].float().clone()
            print(phase, "unprofiled", timings[phase], flush=True)
        validation = {
            "finite": all(bool(torch.isfinite(out).all()) for out in outputs.values()),
            "last_hidden_max_abs": float((outputs["full_prefill"] - outputs["extend"]).abs().max()),
            "last_hidden_cosine": float(
                torch.nn.functional.cosine_similarity(
                    outputs["full_prefill"], outputs["extend"], dim=0
                )
            ),
        }
        assert validation["finite"]
        assert validation["last_hidden_max_abs"] == 0, "Full and split execution disagree"
        # Source is kept alongside the requested report for reproducibility.
        meta = {
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "model_config": asdict(model.config),
            "original_context": original_context,
            "experiment_context": model.config.max_position_embeddings,
            "instruction_tokens": instruction_tokens,
            "execution": execution,
            "attention_shapes": shapes,
            "gpu": str(torch.cuda.get_device_properties(device)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "flashinfer": importlib.metadata.version("flashinfer-python"),
            "nsys": subprocess.check_output(["nsys", "--version"], text=True).strip(),
            "nvidia_smi": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
                    "--format=csv",
                ],
                text=True,
            ).strip(),
            "timings": timings,
            "validation": validation,
            "medians": {
                phase: {
                    metric: statistics.median(row[metric] for row in rows) for metric in rows[0]
                }
                for phase, rows in timings.items()
            },
            "source_sha256": source_hashes(),
        }
        (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        for name in meta["source_sha256"]:
            source = Path(name)
            (args.output_dir / ("source_" + "_".join(source.parts))).write_bytes(
                source.read_bytes()
            )
        captured = []
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStart()
        # Minimal annotations for a less perturbed launch/gap audit.
        for phase, tokens, cache, prefix in work:
            for index in range(1 if phase == "full_prefill" else 3):
                with torch.cuda.nvtx.range(f"GR/light/{phase}/{index}"):
                    _, stats = timed_forward(model, tokens, cache, prefix, args.chunk_size)
                captured.append({"kind": "light", "phase": phase, "iteration": index, **stats})
        # A separate copy of the same work carries module names and layer IDs.
        scopes = NVTXScopes()
        with ExitStack() as stack:
            scopes.install(model, stack)
            for phase, tokens, cache, prefix in work:
                scopes.phase = phase
                with (
                    torch.cuda.nvtx.range(f"GR/detailed/{phase}/0"),
                    scopes.scope("model_misc"),
                ):
                    _, stats = timed_forward(model, tokens, cache, prefix, args.chunk_size)
                captured.append({"kind": "detailed", "phase": phase, "iteration": 0, **stats})
        torch.cuda.synchronize(device)
        torch.cuda.cudart().cudaProfilerStop()
        meta["captured"] = captured
        (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(captured, indent=2), flush=True)


if __name__ == "__main__":
    main()
