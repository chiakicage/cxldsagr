"""Generate text with NOSA weights and FlashInfer full causal attention."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

DEFAULT_MODEL_PATH = Path("/mnt/ssd-wlcb/chenkaiqi/NOSA-8B")


def load_tokenizer(model_path: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load local tokenizer assets without importing checkpoint Python code."""
    from tokenizers import Tokenizer

    model_path = Path(model_path)
    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    with (model_path / "tokenizer_config.json").open(encoding="utf-8") as file:
        tokenizer_config = json.load(file)
    # Generation manages sequence lengths explicitly; never silently truncate.
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer, tokenizer_config


def encode_prompt(
    tokenizer: Any,
    tokenizer_config: dict[str, Any],
    prompt: str,
    *,
    raw_prompt: bool = False,
    system_prompt: str | None = None,
    disable_thinking: bool = False,
) -> list[int]:
    """Apply the checkpoint chat template, or encode an unformatted raw prompt."""
    if not prompt:
        raise ValueError("prompt must not be empty")
    if raw_prompt:
        if system_prompt is not None or disable_thinking:
            raise ValueError("--system-prompt and --disable-thinking require chat mode")
        return tokenizer.encode(prompt, add_special_tokens=True).ids

    from jinja2.sandbox import ImmutableSandboxedEnvironment

    template = tokenizer_config.get("chat_template")
    if not isinstance(template, str) or not template:
        raise ValueError("tokenizer_config.json has no chat_template; use --raw-prompt")
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    def raise_exception(message: str) -> None:
        raise ValueError(message)

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.globals["raise_exception"] = raise_exception
    template_context = {
        key: value
        for key, value in tokenizer_config.items()
        if key.endswith("_token") and isinstance(value, str)
    }
    template_context.update(messages=messages, add_generation_prompt=True)
    if disable_thinking:
        template_context["enable_thinking"] = False
    rendered = environment.from_string(template).render(**template_context)
    # Chat templates provide their own delimiters. In particular, do not add BOS.
    return tokenizer.encode(rendered, add_special_tokens=False).ids


def _validate_generation_options(
    max_new_tokens: int, prefill_chunk_size: int, temperature: float, top_p: float
) -> None:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if prefill_chunk_size < 1:
        raise ValueError("prefill_chunk_size must be positive")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative (0 selects greedy decoding)")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")


def _sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
) -> int:
    import torch

    if temperature == 0:
        return int(logits.argmax().item())
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1:
        probabilities, indices = probabilities.sort(descending=True)
        # Keep the token that crosses top_p, including the first token.
        probabilities.masked_fill_(probabilities.cumsum(-1) - probabilities >= top_p, 0)
        sample = torch.multinomial(probabilities, 1, generator=generator)
        return int(indices[sample].item())
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def generate(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 128,
    prefill_chunk_size: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 0,
) -> tuple[list[int], dict[str, Any]]:
    """Run one request, with chunked prefill and one-token cached decoding."""
    import torch

    from executor.model_executor import ModelExecutor

    _validate_generation_options(max_new_tokens, prefill_chunk_size, temperature, top_p)
    if input_ids.ndim != 1 or input_ids.numel() == 0 or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be a nonempty 1-D torch.long tensor")
    prompt_tokens = input_ids.numel()
    max_seq_len = prompt_tokens + max_new_tokens
    if max_seq_len > model.config.max_position_embeddings:
        raise ValueError(
            f"prompt ({prompt_tokens}) + max_new_tokens ({max_new_tokens}) exceeds "
            f"max_position_embeddings ({model.config.max_position_embeddings})"
        )
    eos_ids = model.config.eos_token_id
    if eos_ids is None:
        eos_ids = []
    elif isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    eos_ids = set(eos_ids)
    device = input_ids.device
    generator = torch.Generator(device=device).manual_seed(seed)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    model.eval()
    executor = ModelExecutor(model, chunk_size=prefill_chunk_size)
    cache = executor.allocate(max_seq_len)
    generated_ids = []
    decode_steps = 0
    stopped_on_eos = False
    try:
        with torch.inference_mode():
            synchronize()
            prefill_start = time.perf_counter()
            logits = executor.prefill(input_ids, cache, output="logits", logits_to_keep=1)[-1]
            synchronize()
            prefill_seconds = time.perf_counter() - prefill_start
            decode_start = time.perf_counter()
            for step in range(max_new_tokens):
                token = _sample_token(
                    logits, temperature=temperature, top_p=top_p, generator=generator
                )
                generated_ids.append(token)
                if token in eos_ids:
                    stopped_on_eos = True
                    break
                if step + 1 < max_new_tokens:
                    token_ids = torch.tensor([token], device=device, dtype=torch.long)
                    logits = executor.extend(token_ids, cache, output="logits", logits_to_keep=1)[
                        -1
                    ]
                    decode_steps += 1
            synchronize()
            decode_seconds = time.perf_counter() - decode_start
    finally:
        executor.release(cache)
    return generated_ids, {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(generated_ids),
        "decode_steps": decode_steps,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "stopped_on_eos": stopped_on_eos,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", help="User message, or raw text with --raw-prompt")
    prompts.add_argument("--prompt-file", type=Path, help="UTF-8 prompt file")
    parser.add_argument("--raw-prompt", action="store_true", help="Skip the chat template")
    parser.add_argument("--system-prompt", help="Optional system message for the chat template")
    parser.add_argument(
        "--disable-thinking", action="store_true", help="Pass enable_thinking=False to the template"
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.0, help="0: greedy; >0: sampling")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not __package__:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parents[1]
        # A local cache.py/layers.py beats a namespace directory even when the
        # root appears first. Remove the script directory before runtime imports.
        sys.path[:] = [str(repo_root)] + [
            entry
            for entry in sys.path
            if Path(entry or ".").resolve() not in (repo_root, script_dir)
        ]
    try:
        _validate_generation_options(
            args.max_new_tokens, args.prefill_chunk_size, args.temperature, args.top_p
        )
        prompt = (
            args.prompt_file.read_text(encoding="utf-8")
            if args.prompt_file is not None
            else args.prompt
        )
        import torch

        device = torch.device(args.device)
        if device.type != "cuda":
            raise ValueError(
                "FlashInfer inference requires a CUDA device (for example --device cuda:0)"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; check the NVIDIA driver and CUDA PyTorch installation"
            )
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("this CUDA device does not support bfloat16; use --dtype float16")

        # Fail before allocating model weights if a runtime dependency is absent.
        import flashinfer  # noqa: F401
        import safetensors  # noqa: F401

        from models.nosa.model import NosaConfig, NosaForCausalLM

        config = NosaConfig.from_pretrained(args.model_path)
        tokenizer, tokenizer_config = load_tokenizer(args.model_path)
        token_ids = encode_prompt(
            tokenizer,
            tokenizer_config,
            prompt,
            raw_prompt=args.raw_prompt,
            system_prompt=args.system_prompt,
            disable_thinking=args.disable_thinking,
        )
        if not token_ids:
            raise ValueError("the encoded prompt is empty")
        if len(token_ids) + args.max_new_tokens > config.max_position_embeddings:
            raise ValueError(
                f"prompt ({len(token_ids)}) + max_new_tokens ({args.max_new_tokens}) exceeds "
                f"max_position_embeddings ({config.max_position_embeddings})"
            )
        dtype = getattr(torch, args.dtype)
        load_start = time.perf_counter()
        model = NosaForCausalLM.from_pretrained(args.model_path, device=device, dtype=dtype)
        torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - load_start
        generated_ids, stats = generate(
            model,
            torch.tensor(token_ids, dtype=torch.long, device=device),
            max_new_tokens=args.max_new_tokens,
            prefill_chunk_size=args.prefill_chunk_size,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        print(tokenizer.decode(generated_ids, skip_special_tokens=True))
        stats["load_seconds"] = load_seconds
        print(json.dumps(stats, ensure_ascii=False), file=sys.stderr)
    except ImportError as exc:
        parser.exit(1, f"error: missing or incompatible dependency: {exc}. Run uv sync.\n")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
