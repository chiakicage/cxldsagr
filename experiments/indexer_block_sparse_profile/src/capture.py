"""Measure full NOSA sparse prefill/extend, then profile in a separate process.

The benchmark process is not launched under nsys. Both phases include all
decoder layers and normalized hidden output, with resident K/V/CIS. Loading,
tokenization, cache allocation/reset and prefix setup are outside the timers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import torch

from executor.model_executor import run_chunks
from experiments.nosa_gr_65536_1024.src.capture import execution_split
from experiments.nosa_gr_65536_1024.src.sources import source_hashes
from GR.input_generator import INSTRUCTION, TextConfig, create_input_generator
from GR.scheduling import ScheduleConfig
from models.nosa import request_format
from models.nosa.infer import DEFAULT_MODEL_PATH, load_tokenizer
from models.nosa.model import NosaForCausalLM

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/indexer_block_sparse_profile"


def validate_workload(prefix_tokens, new_tokens, chunk_size, warmup, repeats, profile_repeats):
    for name, value in (
        ("prefix_tokens", prefix_tokens),
        ("new_tokens", new_tokens),
        ("chunk_size", chunk_size),
        ("warmup", warmup),
        ("repeats", repeats),
        ("profile_repeats", profile_repeats),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if prefix_tokens % chunk_size:
        raise ValueError("prefix_tokens must align to chunk_size at the candidate boundary")
    if new_tokens > chunk_size:
        raise ValueError("chunk_size must cover the entire extend in one chunk")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def fingerprint_sources():
    return source_hashes(
        *sorted((EXPERIMENT / "src").glob("*.py")),
        *sorted((EXPERIMENT / "scripts").glob("*.sh")),
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    )


def validate_profile_metadata(current, benchmark):
    """Compare runtime metadata with its JSON round-trip representation."""
    for key in (
        "model_config",
        "checkpoint_path",
        "checkpoint_files",
        "checkpoint_config_sha256",
        "request_sha256",
        "source_sha256",
        "torch",
        "triton",
        "flashinfer",
        "gpu",
    ):
        # Frozen config tuples (e.g. EOS IDs) become arrays in metadata.json.
        value = json.loads(json.dumps(current[key], allow_nan=False))
        if value != benchmark[key]:
            raise ValueError(f"Profile and benchmark differ in {key}")


def make_request(args):
    if args.request_file is not None:
        request = json.loads(args.request_file.read_text())
    else:
        tokenizer, _ = load_tokenizer(args.model_path)
        instruction = len(
            tokenizer.encode(request_format.prefix(INSTRUCTION), add_special_tokens=False).ids
        )
        if args.prefix_tokens <= instruction:
            raise ValueError("Prefix must contain instruction and history")
        total = args.prefix_tokens + args.new_tokens
        with patch.object(request_format, "MAX_INPUT_TOKENS", total):
            generator = create_input_generator(
                model="nosa",
                tokenizer=tokenizer,
                num_users=1000,
                text_config=TextConfig(
                    user_lengths=(args.prefix_tokens - instruction,),
                    user_probabilities=(1.0,),
                    item_lengths=(args.new_tokens + instruction,),
                    item_probabilities=(1.0,),
                    max_input_tokens=total,
                ),
                schedule_config=ScheduleConfig(seed=42, sampling="weighted"),
            )
            request = next(generator.iter_generate(1))
    validate_request(request, args.prefix_tokens, args.new_tokens)
    return request


def validate_request(request, prefix_tokens, new_tokens):
    execution_split(request, prefix_tokens, new_tokens)
    if request.get("model") != "nosa":
        raise ValueError("Expected a NOSA GR request")
    if any(type(token) is not int or token < 0 for token in request["input_ids"]):
        raise ValueError("Request input_ids must be nonnegative integers")
    total = prefix_tokens + new_tokens
    if request.get("attention_mask", [1] * total) != [1] * total:
        raise ValueError("Only one unpadded GR request is supported")


def rewind_cache(cache, prefix_length):
    """Reuse an immutable resident prefix; reject future opaque indexer state.

    Current NOSA stores K/V/CIS as per-token records and does not maintain an
    incremental compressed state. Cursor rewind is therefore confined to this
    repeated-request experiment; it never mutates the committed prefix bytes.
    """
    if prefix_length == 0:
        cache.reset()
        return
    if cache.length < prefix_length:
        raise ValueError("The requested stable prefix has not been built")
    if any(cache.get_layer_state(i) is not None for i in range(cache.config.num_hidden_layers)):
        raise ValueError("Replaying extend cannot rewind opaque layer state")
    cache.length = prefix_length


@torch.inference_mode()
def timed_forward(model, ids, cache, prefix_length, chunk_size):
    rewind_cache(cache, prefix_length)
    torch.cuda.synchronize(ids.device)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Allocate/init events outside the measured region, including lazy CUDA setup.
    begin.record()
    end.record()
    end.synchronize()
    started = time.perf_counter()
    begin.record()
    hidden = run_chunks(model, ids, cache, chunk_size, output="hidden")
    submitted = time.perf_counter()
    end.record()
    torch.cuda.synchronize(ids.device)
    finished = time.perf_counter()
    if cache.length != prefix_length + len(ids):
        raise ValueError("Forward did not commit exactly the requested tokens")
    return hidden, {
        "wall_ms": (finished - started) * 1000,
        "host_submit_ms": (submitted - started) * 1000,
        "cuda_span_ms": begin.elapsed_time(end),
    }


def runtime_metadata(args, model, original_context):
    props = torch.cuda.get_device_properties(model.model.embed_tokens.weight.device)
    checkpoint = args.model_path.resolve()
    return {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "run_id": args.run_id,
        "args": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "model_config": asdict(model.config),
        "checkpoint_context": original_context,
        "checkpoint_path": str(checkpoint),
        "checkpoint_files": {
            path.name: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(checkpoint.glob("*.safetensors"))
        },
        "checkpoint_config_sha256": hashlib.sha256(
            (checkpoint / "config.json").read_bytes()
        ).hexdigest(),
        "gpu": {
            "name": props.name,
            "capability": [props.major, props.minor],
            "uuid": str(props.uuid),
            "total_memory": props.total_memory,
            "sm_count": props.multi_processor_count,
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": importlib.metadata.version("triton"),
        "flashinfer": importlib.metadata.version("flashinfer-python"),
        "nvidia_smi": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
                "--format=csv",
            ],
            text=True,
        ).strip(),
        "source_sha256": fingerprint_sources(),
        "measurement_boundary": {
            "output": "normalized hidden states of the final chunk; no LM head",
            "included": "embedding, all decoder layers, CIS, KV writes, indexer, block sparse attention, final norm, Python submission and CUDA completion",
            "excluded": "checkpoint loading, tokenizer/GR generation, H2D input copy, cache allocation/reset, prefix setup for extend, profiler/module scopes in primary timings",
            "cuda_span_ms": "CUDA event elapsed span including launch gaps; not kernel active time",
            "host_submit_ms": "host submission including model-internal CUDA waits; overlaps GPU work",
        },
    }


def audit_extend(model, ids, cache, prefix_tokens):
    from experiments.indexer_block_sparse_profile.src.instrumentation import SparseScopes

    rewind_cache(cache, prefix_tokens)
    with SparseScopes(model, phase="extend", timing=False, audit=True) as scopes:
        hidden = run_chunks(model, ids, cache, len(ids))
    records = scopes.collect()
    if not torch.isfinite(hidden).all().item():
        raise ValueError("Sparse extend produced nonfinite hidden states")
    attention = [row for row in records if row["stage"] == "block_sparse_attention"]
    if len(attention) != model.config.num_hidden_layers:
        raise ValueError("Audit must visit every decoder layer exactly once")
    for row in attention:
        if row["query_start"] != prefix_tokens or row["query_length"] != len(ids):
            raise ValueError("Audit disagrees with the 64K/1K candidate boundary")
        details = row["details"]
        expected = {
            "q_shape": [len(ids), model.config.num_attention_heads, model.config.head_dim],
            "k_shape": [
                prefix_tokens + len(ids),
                model.config.num_key_value_heads,
                model.config.head_dim,
            ],
            "v_shape": [
                prefix_tokens + len(ids),
                model.config.num_key_value_heads,
                model.config.head_dim,
            ],
            "selection_shape": [len(ids), model.config.num_key_value_heads, 64],
            "block_size": 64,
            "block_budget": 64,
        }
        if any(details[key] != value for key, value in expected.items()):
            raise ValueError("Actual Q/K/V or block selection geometry disagrees with the workload")
        if prefix_tokens >= 64 * 64 and details["valid_blocks_min"] != 64:
            raise ValueError("Every long-context query/head must select 64 valid blocks")
    return records


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "profile"), default="benchmark")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--request-file", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefix-tokens", type=int, default=65536)
    parser.add_argument("--new-tokens", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-repeats", type=int, default=1)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Staging data directory")
    return parser


@torch.inference_mode()
def run(args):
    validate_workload(
        args.prefix_tokens,
        args.new_tokens,
        args.chunk_size,
        args.warmup,
        args.repeats,
        args.profile_repeats,
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("This experiment requires SM90 CUDA, Triton and FlashInfer")
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise ValueError("This experiment requires an SM90/Hopper GPU")
    torch.cuda.set_device(device)
    total = args.prefix_tokens + args.new_tokens
    workload = {
        "prefix_tokens": args.prefix_tokens,
        "new_tokens": args.new_tokens,
        "total_tokens": total,
        "chunk_size": args.chunk_size,
        "attention_mode": "sparse",
        "backend": "triton",
        "dtype": "bfloat16",
        "block_size": 64,
        "block_budget": 64,
        "local_blocks": 17,
        "query_stage_blocks": 33,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "profile_repeats": args.profile_repeats,
    }
    if args.mode == "benchmark":
        args.output_dir.mkdir(parents=True, exist_ok=False)
        request = make_request(args)
        write_json(args.output_dir / "request.json", request)
        measurements = {
            "schema_version": 1,
            "run_id": args.run_id,
            "workload": workload,
            "timings": {},
            "profiles": {},
        }
    else:
        measurements = json.loads((args.output_dir / "measurements.json").read_text())
        if measurements["run_id"] != args.run_id or measurements["workload"] != workload:
            raise ValueError("Profile must use the benchmark run ID and workload")
        if measurements["profiles"]:
            raise ValueError("Profile results already exist")
        request = json.loads((args.output_dir / "request.json").read_text())
        validate_request(request, args.prefix_tokens, args.new_tokens)
    print(
        f"Loading all NOSA layers; mode={args.mode}, P={args.prefix_tokens}, Q={args.new_tokens}",
        flush=True,
    )
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
        raise ValueError("The experiment requires NOSA-8B: 32 layers, 32 Q heads, 2 KV heads, D128")
    if max(request["input_ids"]) >= config.vocab_size:
        raise ValueError("Request token ID exceeds the checkpoint vocabulary")
    model.config = replace(
        config, max_position_embeddings=max(total, config.max_position_embeddings)
    )
    meta = runtime_metadata(args, model, config.max_position_embeddings)
    meta["request_sha256"] = hashlib.sha256(
        (args.output_dir / "request.json").read_bytes()
    ).hexdigest()
    if args.mode == "profile":
        benchmark_meta = json.loads((args.output_dir / "metadata.json").read_text())
        validate_profile_metadata(meta, benchmark_meta)
    ids = torch.tensor(request["input_ids"], device=device, dtype=torch.long)
    with ExitStack() as resources:
        full_cache = model.new_cache(total)
        prefix_cache = model.new_cache(total)
        resources.callback(model.cache_manager.release, full_cache)
        resources.callback(model.cache_manager.release, prefix_cache)
        print("Building the stable sparse prefix outside measurement", flush=True)
        run_chunks(model, ids[: args.prefix_tokens], prefix_cache, args.chunk_size)
        work = (
            ("full_prefill", ids, full_cache, 0),
            ("extend", ids[args.prefix_tokens :], prefix_cache, args.prefix_tokens),
        )
        if args.mode == "benchmark":
            audit = audit_extend(model, ids[args.prefix_tokens :], prefix_cache, args.prefix_tokens)
            write_json(args.output_dir / "attention_audit.json", audit)
        outputs = {}
        for phase, tokens, cache, prefix in work:
            stats = []
            count = args.warmup + (args.repeats if args.mode == "benchmark" else 0)
            for iteration in range(count):
                hidden, timing = timed_forward(model, tokens, cache, prefix, args.chunk_size)
                outputs[phase] = hidden.float().clone()
                if args.mode == "benchmark" and iteration >= args.warmup:
                    stats.append(timing)
                print(
                    f"{phase} {'warmup' if iteration < args.warmup else 'sample'} {iteration}: {timing['wall_ms']:.3f} ms",
                    flush=True,
                )
            if args.mode == "benchmark":
                measurements["timings"][phase] = stats
        validation = {
            "finite": all(torch.isfinite(out).all().item() for out in outputs.values()),
            "full_vs_extend_hidden_max_abs": float(
                (outputs["full_prefill"] - outputs["extend"]).abs().max()
            ),
        }
        if not validation["finite"] or validation["full_vs_extend_hidden_max_abs"] != 0:
            raise ValueError(
                f"Identically chunked full and prefix-ready execution disagree: {validation}"
            )
        meta["validation"] = validation
        meta["cache_capacity_bytes_per_session"] = full_cache.stats()["capacity_bytes"]
        if args.mode == "profile":
            from experiments.indexer_block_sparse_profile.src.instrumentation import SparseScopes

            meta["instrumented_timings"] = {}
            torch.cuda.synchronize(device)
            torch.cuda.cudart().cudaProfilerStart()
            try:
                for phase, tokens, cache, prefix in work:
                    measurements["profiles"][phase] = []
                    meta["instrumented_timings"][phase] = []
                    for iteration in range(args.profile_repeats):
                        with torch.cuda.nvtx.range(f"NOSA/profile/{phase}/{iteration}"):
                            with SparseScopes(model, phase=phase, timing=True) as scopes:
                                hidden, timing = timed_forward(
                                    model, tokens, cache, prefix, args.chunk_size
                                )
                            records = scopes.collect()
                        if not torch.isfinite(hidden).all().item():
                            raise ValueError("Profile produced nonfinite hidden output")
                        if not torch.equal(hidden.float(), outputs[phase]):
                            raise ValueError(
                                "Profiling changed the model's candidate hidden states"
                            )
                        measurements["profiles"][phase].append(records)
                        meta["instrumented_timings"][phase].append(timing)
                        print(
                            f"Profiled {phase}/{iteration}: {len(records)} module intervals",
                            flush=True,
                        )
            finally:
                torch.cuda.synchronize(device)
                torch.cuda.cudart().cudaProfilerStop()
        else:
            execution = {
                **workload,
                "prefix_span": [0, args.prefix_tokens],
                "extend_span": [args.prefix_tokens, total],
                "gr_stable_prefix_tokens": request["stable_prefix_tokens"],
                "gr_candidate_suffix_tokens": request["candidate_suffix_tokens"],
                "instruction_tokens": request["instruction_tokens"],
                "history_tokens": request["user_tokens"],
                "gr_item_budget_including_instruction": request["item_tokens"],
            }
            write_json(args.output_dir / "execution.json", execution)
            for name in meta["source_sha256"]:
                destination = args.output_dir / "sources" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((ROOT / name).read_bytes())
        write_json(
            args.output_dir
            / ("metadata.json" if args.mode == "benchmark" else "profile_metadata.json"),
            meta,
        )
        write_json(args.output_dir / "measurements.json", measurements)


def main():
    parser = _build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
