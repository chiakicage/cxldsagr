"""Run generated GR requests locally, printing one JSON summary per completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.nosa.infer import DEFAULT_MODEL_PATH


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--attention-mode", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--sparse-backend", choices=("auto", "reference", "triton"), default="auto")
    parser.add_argument("--cache-backend", choices=("resident", "offload"), default="resident")
    parser.add_argument(
        "--offload-query-tile-size",
        type=int,
        default=128,
        help="Query group size for first-use fetch byte counters",
    )
    parser.add_argument(
        "--offload-fetch-ctas",
        type=int,
        default=96,
        help="Maximum CTAs contributing spare producer warps to fetch",
    )
    parser.add_argument(
        "--no-fetch-overlap",
        action="store_true",
        help="Fetch the full sparse union before whole-query attention",
    )
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--num-users", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--user-lengths", type=int, nargs="+", help="Defaults to GR TextConfig")
    parser.add_argument("--item-lengths", type=int, nargs="+", help="Defaults to GR TextConfig")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.count < 1 or args.num_users < 1 or args.prefill_chunk_size < 1:
        parser.error("count, num-users and prefill-chunk-size must be positive")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    import torch

    from executor.model_executor import ModelExecutor
    from GR.input_generator import TextConfig, create_input_generator
    from GR.scheduling import ScheduleConfig
    from models.nosa.model import NosaConfig, NosaForCausalLM
    from serving.runner import GRRunner

    config = NosaConfig.from_pretrained(args.model_path)
    defaults = TextConfig()
    user_lengths = tuple(args.user_lengths or defaults.user_lengths)
    item_lengths = tuple(args.item_lengths or defaults.item_lengths)
    text_config = TextConfig(
        user_lengths=user_lengths,
        user_probabilities=(1 / len(user_lengths),) * len(user_lengths),
        item_lengths=item_lengths,
        item_probabilities=(1 / len(item_lengths),) * len(item_lengths),
        max_input_tokens=min(config.max_position_embeddings, defaults.max_input_tokens),
    )
    generator = create_input_generator(
        model="nosa",
        tokenizer=args.model_path / "tokenizer.json",
        num_users=args.num_users,
        text_config=text_config,
        schedule_config=ScheduleConfig(seed=args.seed),
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("NOSA CLI inference requires an available CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("this CUDA device does not support bfloat16; use --dtype float16")
    model = NosaForCausalLM.from_pretrained(
        args.model_path,
        device=device,
        dtype=getattr(torch, args.dtype),
        attention_mode=args.attention_mode,
        sparse_backend=args.sparse_backend,
        cache_backend=args.cache_backend,
        offload_query_tile_size=args.offload_query_tile_size,
        offload_fetch_ctas=args.offload_fetch_ctas,
        offload_overlap=not args.no_fetch_overlap,
    )
    runner = GRRunner(ModelExecutor(model, chunk_size=args.prefill_chunk_size), device=device)
    for result in runner.run(generator.iter_generate(args.count)):
        torch.cuda.synchronize(device)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "attention_mode": args.attention_mode,
                    "metadata": result.metadata,
                    "feature_shape": list(result.last_hidden.shape),
                    "feature_dtype": str(result.last_hidden.dtype),
                    "feature_device": str(result.last_hidden.device),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
