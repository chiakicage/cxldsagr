"""Serve repeated GR users with persistent, budgeted caches on one Hopper GPU.

This CLI performs no separate warmup. Request timings include first-use JIT
compilation; formal paired experiments provide their own warmup and validation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

SCHEMES = {
    "deepseek_v32": ("hbm", "echo", "serial_sparse", "dense_prefetch"),
    "nosa": ("hbm", "serial_sparse", "dense_prefetch", "overlap"),
}
DEFAULT_PATHS = {
    "deepseek_v32": Path("/preset-models"),
    "nosa": Path("/mnt/ssd-wlcb/chenkaiqi/NOSA-8B"),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=tuple(SCHEMES), default="deepseek_v32")
    parser.add_argument(
        "--scheme",
        choices=("hbm", "echo", "serial_sparse", "dense_prefetch", "overlap"),
        default="hbm",
        help="DeepSeek: hbm/echo/serial_sparse/dense_prefetch; NOSA: hbm/serial_sparse/dense_prefetch/overlap",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Checkpoint and tokenizer directory; default depends on model",
    )
    parser.add_argument("--num-users", type=int, default=8)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument(
        "--max-revisits",
        type=int,
        help="Optional legacy per-user revisit cap; default is uncapped",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--history-tokens",
        type=int,
        default=16384,
        help="Complete fixed prefix including instruction",
    )
    parser.add_argument(
        "--candidate-tokens", type=int, default=1024, help="Complete changing candidate suffix"
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        help="Explicit generation context limit; NOSA supports a configured limit up to 262144",
    )
    parser.add_argument("--hbm-budget-gib", type=float, default=1.0)
    parser.add_argument("--dram-budget-gib", type=float, default=16.0)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--deepseek-slots", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _validate_args(args, parser) -> None:
    if args.scheme not in SCHEMES[args.model]:
        parser.error(f"{args.model} supports schemes: {', '.join(SCHEMES[args.model])}")
    for name in ("num_users", "count", "history_tokens", "candidate_tokens", "chunk_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.max_revisits is not None and args.max_revisits < 0:
        parser.error("--max-revisits must be nonnegative")
    if args.context_limit is not None and args.context_limit < 1:
        parser.error("--context-limit must be positive")
    for name in ("hbm_budget_gib", "dram_budget_gib"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    args.hbm_budget_bytes = int(args.hbm_budget_gib * 2**30)
    args.dram_budget_bytes = int(args.dram_budget_gib * 2**30)
    if args.hbm_budget_bytes < 1:
        parser.error("--hbm-budget-gib must allow at least one byte")
    if args.model == "deepseek_v32" and args.deepseek_slots < args.chunk_size:
        parser.error("--deepseek-slots must be at least --chunk-size")
    if args.model_path is None:
        args.model_path = DEFAULT_PATHS[args.model]


def _build_generator(args):
    from tokenizers import Tokenizer

    from GR.input_generator import INSTRUCTION, MODEL_FORMATS, TextConfig, create_input_generator
    from GR.scheduling import ScheduleConfig

    tokenizer = Tokenizer.from_file(str(args.model_path / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    model_format = MODEL_FORMATS[args.model]
    instruction_tokens = len(
        tokenizer.encode(model_format.prefix(INSTRUCTION), add_special_tokens=False).ids
    )
    if args.history_tokens <= instruction_tokens:
        raise ValueError("history-tokens must leave user history after the fixed instruction")
    context_limit = args.context_limit or model_format.MAX_INPUT_TOKENS
    supported_limit = 262144 if args.model == "nosa" else model_format.MAX_INPUT_TOKENS
    if context_limit > supported_limit:
        raise ValueError("context-limit exceeds the supported backend context")
    if args.history_tokens + args.candidate_tokens > context_limit:
        raise ValueError("history-tokens + candidate-tokens exceeds the model's GR context limit")
    return create_input_generator(
        model=args.model,
        tokenizer=tokenizer,
        heat_source="curve",
        curve_dataset="beauty",
        curve_field="interaction_count",
        num_users=args.num_users,
        context_limit=args.context_limit,
        text_material="synthetic",
        text_config=TextConfig(
            user_lengths=(args.history_tokens - instruction_tokens,),
            user_probabilities=(1.0,),
            item_lengths=(args.candidate_tokens + instruction_tokens,),
            item_probabilities=(1.0,),
            max_input_tokens=args.history_tokens + args.candidate_tokens,
        ),
        schedule_config=ScheduleConfig(
            seed=args.seed,
            sampling="weighted",
            arrival="constant",
            qps=1,
            max_revisits=args.max_revisits,
        ),
    )


def _build_backend(args):
    import torch

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("persistent GR serving requires an available SM90/Hopper CUDA device")
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise ValueError("persistent GR serving requires an SM90/Hopper GPU")
    torch.cuda.set_device(device)
    if args.model == "deepseek_v32":
        from models.deepseek_v32.serving_backend import DeepSeekServingBackend

        return DeepSeekServingBackend(
            args.model_path,
            scheme=args.scheme,
            device=device,
            chunk_size=args.chunk_size,
            slots=args.deepseek_slots,
        )
    from models.nosa.serving import NosaServingBackend

    return NosaServingBackend.from_pretrained(
        args.model_path,
        scheme=args.scheme,
        device=device,
        chunk_size=args.chunk_size,
        max_seq_len=args.history_tokens + args.candidate_tokens,
    )


def _emit(value) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    try:
        from serving.persistent import PersistentGRRunner

        generator = _build_generator(args)
        backend = _build_backend(args)
        _emit(
            {
                "status": "started",
                "model": args.model,
                "scheme": args.scheme,
                "model_path": str(args.model_path),
                "num_users": args.num_users,
                "count": args.count,
                "max_revisits": args.max_revisits,
                "context_limit": args.context_limit,
                "seed": args.seed,
                "history_tokens": args.history_tokens,
                "candidate_tokens": args.candidate_tokens,
                "hbm_budget_bytes": args.hbm_budget_bytes,
                "dram_budget_bytes": args.dram_budget_bytes,
                "timing": "no separate warmup; request latency includes first-use JIT compilation, excludes weight loading and GR generation",
                "heat": generator.population.metadata,
                "backend": backend.describe(),
            }
        )
        completed = revisits = hits = evictions = 0
        visits_per_user = Counter()
        total_ms = revisit_ms = 0.0
        with PersistentGRRunner(
            backend,
            hbm_budget_bytes=args.hbm_budget_bytes,
            dram_budget_bytes=args.dram_budget_bytes,
        ) as runner:
            for result in runner.run(generator.iter_generate(args.count)):
                metrics = result.metrics
                _emit(
                    {
                        "status": "completed",
                        "metrics": metrics,
                        "hidden_shape": list(result.hidden.shape),
                        "hidden_dtype": str(result.hidden.dtype),
                        "hidden_device": str(result.hidden.device),
                    }
                )
                completed += 1
                visits_per_user[metrics["user_id"]] += 1
                revisits += metrics["is_revisit"]
                hits += metrics["prefix_cache_hit"]
                evictions += len(metrics["evicted_users"])
                total_ms += metrics["latency_ms"]
                if metrics["is_revisit"]:
                    revisit_ms += metrics["latency_ms"]
                del result
        _emit(
            {
                "status": "finished",
                "requests": completed,
                "first_visits": completed - revisits,
                "revisits": revisits,
                "returning_users": sum(count > 1 for count in visits_per_user.values()),
                "prefix_cache_hits": hits,
                "evicted_users": evictions,
                "mean_latency_ms": total_ms / completed if completed else None,
                "revisit_mean_latency_ms": revisit_ms / revisits if revisits else None,
                "separate_warmup": False,
            }
        )
    except ImportError as exc:
        parser.exit(1, f"error: missing or incompatible dependency: {exc}. Run uv sync.\n")
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
