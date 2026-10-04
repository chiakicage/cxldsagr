"""Deterministic GR multi-user requests and identities shared by every cache scheme."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from GR.heat import HeatPopulation
from GR.input_generator import (
    INSTRUCTION,
    MODEL_FORMATS,
    InputGenerator,
    TextConfig,
    create_input_generator,
)
from GR.scheduling import ScheduleConfig


def _positive_integer(name: str, value: int, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def token_sha256(ids: list[int] | tuple[int, ...]) -> str:
    """Sign exact token IDs, independent of tensor dtype, device, or Python hash seed."""
    return hashlib.sha256(_json(list(ids)).encode()).hexdigest()


@dataclass(frozen=True)
class WorkloadConfig:
    model: str
    num_users: int
    requests: int
    history_tokens: int = 8192
    candidate_tokens: int = 1024
    seed: int = 42
    heat_dataset: str | None = "beauty"
    heat_field: str | None = None
    max_revisits: int | None = None
    context_limit: int | None = None
    sampling: str = "weighted"

    def __post_init__(self) -> None:
        if self.model not in MODEL_FORMATS:
            raise ValueError("model must be deepseek_v32 or nosa")
        for name in ("num_users", "requests", "history_tokens", "candidate_tokens"):
            _positive_integer(name, getattr(self, name))
        _positive_integer("seed", self.seed, 0)
        if self.sampling not in ("weighted", "sequential"):
            raise ValueError("sampling must be weighted or sequential")
        if self.max_revisits is not None:
            _positive_integer("max_revisits", self.max_revisits, 0)
        if self.sampling == "sequential":
            if self.max_revisits is not None:
                raise ValueError("sequential sampling does not support max_revisits")
            object.__setattr__(self, "heat_dataset", None)
            object.__setattr__(self, "heat_field", None)
        elif not isinstance(self.heat_dataset, str) or not self.heat_dataset:
            raise ValueError("heat_dataset must be a nonempty curve dataset name")
        elif self.heat_field is None:
            object.__setattr__(
                self,
                "heat_field",
                "pv_share" if self.heat_dataset.startswith("industrial_") else "interaction_count",
            )
        if self.sampling == "weighted" and (
            not isinstance(self.heat_field, str) or not self.heat_field
        ):
            raise ValueError("heat_field must be a nonempty curve field name")
        default_limit = MODEL_FORMATS[self.model].MAX_INPUT_TOKENS
        if self.context_limit is not None:
            _positive_integer("context_limit", self.context_limit)
            supported_limit = 262144 if self.model == "nosa" else default_limit
            if self.context_limit > supported_limit:
                raise ValueError(f"context_limit exceeds the {supported_limit}-token backend limit")
        effective_limit = default_limit if self.context_limit is None else self.context_limit
        if self.history_tokens + self.candidate_tokens > effective_limit:
            raise ValueError("history + candidate tokens exceeds model context")


@dataclass(frozen=True)
class Workload:
    requests: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]

    def write(self, output_dir: str | Path) -> None:
        """Write reproducibility data; an existing workload is never silently replaced."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        paths = (output / "requests.jsonl", output / "workload.json")
        if any(path.exists() for path in paths):
            raise FileExistsError("requests.jsonl or workload.json already exists")
        with paths[0].open("x", encoding="utf-8") as handle:
            for row in self.requests:
                handle.write(_json(row) + "\n")
        paths[1].write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False) + "\n")


def build_workload(
    config: WorkloadConfig, *, tokenizer: str | Path | Tokenizer | None = None
) -> Workload:
    """Use the GR text generator with heat-weighted or explicit cyclic accesses.

    ``history_tokens`` includes the complete fixed instruction and user history;
    ``candidate_tokens`` is the complete changing suffix. Both are exact KV
    boundaries, unlike GR's user/item text budgets. Weighted mode draws users
    independently with replacement from the selected normalized heat. Sequential
    mode cycles through explicit synthetic IDs without reading a heat curve.
    """
    tokenizer_path = None
    if tokenizer is None:
        tokenizer = MODEL_FORMATS[config.model].TOKENIZER_PATH
    if not isinstance(tokenizer, Tokenizer):
        tokenizer_path = Path(tokenizer)
        if tokenizer_path.is_dir():
            tokenizer_path /= "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    instruction = len(
        tokenizer.encode(
            MODEL_FORMATS[config.model].prefix(INSTRUCTION), add_special_tokens=False
        ).ids
    )
    if config.history_tokens <= instruction:
        raise ValueError("history_tokens must leave room for history after the fixed instruction")
    generator_options = {
        "model": config.model,
        "tokenizer": tokenizer,
        "context_limit": config.context_limit,
        "text_config": TextConfig(
            user_lengths=(config.history_tokens - instruction,),
            user_probabilities=(1.0,),
            item_lengths=(config.candidate_tokens + instruction,),
            item_probabilities=(1.0,),
            max_input_tokens=config.history_tokens + config.candidate_tokens,
            history_cache_users=min(config.num_users, config.requests),
        ),
        "schedule_config": ScheduleConfig(
            seed=config.seed,
            sampling=config.sampling,
            arrival="constant",
            qps=1,
            max_revisits=config.max_revisits,
        ),
    }
    if config.sampling == "sequential":
        population = HeatPopulation(
            {user: 1.0 for user in range(config.num_users)},
            {
                "source": "explicit_synthetic_ids",
                "synthetic": True,
                "selected_users": config.num_users,
                "selected_weight_sum": float(config.num_users),
                "user_identity": "synthetic IDs 0..N-1",
                "weight_semantics": "equal placeholders required by GR; no heat curve or probabilistic sampling",
            },
        )
        generator = InputGenerator(population=population, **generator_options)
    else:
        generator = create_input_generator(
            **generator_options,
            heat_source="curve",
            curve_dataset=config.heat_dataset,
            curve_field=config.heat_field,
            num_users=config.num_users,
            text_material="synthetic",
        )
    prefixes: dict[int, tuple[int, ...]] = {}
    previous_candidates: dict[int, tuple[int, ...]] = {}
    counts: Counter[int] = Counter()
    requests = []
    identities = []
    access_trace = []
    for request_id, request in enumerate(generator.iter_generate(config.requests)):
        user = request["user_id"]
        ids = request["input_ids"]
        prefix = tuple(ids[: config.history_tokens])
        candidate = tuple(ids[config.history_tokens :])
        if (
            request["stable_prefix_tokens"] != config.history_tokens
            or request["candidate_suffix_tokens"] != config.candidate_tokens
        ):
            raise ValueError("GR generator did not preserve the requested KV boundaries")
        if prefixes.setdefault(user, prefix) != prefix:
            raise ValueError(f"user {user}: stable prefix changed across visits")
        if user in previous_candidates and previous_candidates[user] == candidate:
            raise ValueError(f"user {user}: candidate did not change across visits")
        if request["visit_index"] != counts[user]:
            raise ValueError(f"user {user}: noncontiguous visit indices")
        request.update(
            request_id=request_id,
            visit_number=counts[user] + 1,
            is_revisit=counts[user] > 0,
            prefix_sha256=token_sha256(prefix),
            candidate_sha256=token_sha256(candidate),
            input_sha256=token_sha256(ids),
        )
        counts[user] += 1
        previous_candidates[user] = candidate
        requests.append(request)
        access_trace.append(
            {
                key: request[key]
                for key in (
                    "request_id",
                    "user_id",
                    "visit_index",
                    "previous_request_id",
                    "timestamp",
                )
            }
        )
        identities.append(
            {
                key: request[key]
                for key in (
                    "request_id",
                    "user_id",
                    "visit_index",
                    "visit_number",
                    "is_revisit",
                    "timestamp",
                    "stable_prefix_tokens",
                    "candidate_suffix_tokens",
                    "prefix_sha256",
                    "candidate_sha256",
                    "input_sha256",
                )
            }
        )
    tokenizer_hash = hashlib.sha256(tokenizer.to_str().encode()).hexdigest()
    identity = {
        "config": asdict(config),
        "heat_sha256": generator.population.metadata.get("sha256"),
        "tokenizer_sha256": tokenizer_hash,
        "requests": identities,
    }
    manifest = {
        "schema_version": 1,
        **identity,
        "workload_sha256": hashlib.sha256(_json(identity).encode()).hexdigest(),
        "access_trace_sha256": hashlib.sha256(_json(access_trace).encode()).hexdigest(),
        "tokenizer_path": str(tokenizer_path.resolve()) if tokenizer_path else None,
        "heat": generator.population.metadata,
        "schedule": asdict(generator.schedule_config),
        "context": {
            "format_default_tokens": MODEL_FORMATS[config.model].MAX_INPUT_TOKENS,
            "effective_limit_tokens": generator.context_limit,
            "override_explicit": config.context_limit is not None,
            "interpretation": "generation boundary only; does not validate model quality",
        },
        "text_config": json.loads(_json(asdict(generator.text_config))),
        "measurement_policy": {
            "request_execution": "serial; synthetic timestamps do not cause wall-clock sleeps",
            "cache_start": "empty after separate warmups for each scheme",
            "revisit": "visit_index > 0, independent of cache residency or hit status",
            "history_tokens": "complete stable prefix including fixed instruction",
            "candidate_tokens": "complete changing suffix",
            "sampling": (
                "deterministic cycles through synthetic IDs 0..num_users-1; no heat or random draws"
                if config.sampling == "sequential"
                else "independent draws with replacement from normalized heat; explicit legacy "
                "caps renormalize weights over remaining eligible users"
            ),
            "request_limit": (
                "exactly config.requests sequential accesses"
                if config.sampling == "sequential"
                else "exactly config.requests draws unless an explicit legacy cap exhausts all users"
            ),
            "user_probability": (
                "not applicable to sequential accesses; weights are GR API placeholders"
                if config.sampling == "sequential"
                else "normalized heat; actual user counts and revisits are observed outputs"
            ),
        },
        "observed": {
            "requests": len(requests),
            "unique_users": len(counts),
            "returning_users": sum(count > 1 for count in counts.values()),
            "first_visits": len(counts),
            "revisits": sum(count - 1 for count in counts.values()),
            "max_visits": max(counts.values(), default=0),
            "max_revisits": max((count - 1 for count in counts.values()), default=0),
        },
        "users": [
            {
                "user_id": user,
                "weight": weight,
                "probability": (
                    None
                    if config.sampling == "sequential"
                    else weight / generator.population.metadata["selected_weight_sum"]
                ),
                "visits": counts[user],
                "revisits": max(counts[user] - 1, 0),
                "prefix_sha256": token_sha256(prefixes[user]) if user in prefixes else None,
            }
            for user, weight in generator.population.weights.items()
        ],
    }
    return Workload(tuple(requests), manifest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_FORMATS), required=True)
    parser.add_argument("--num-users", type=int, required=True)
    parser.add_argument(
        "--requests",
        type=int,
        default=128,
        help="total accesses; a weighted-mode legacy revisit cap may end the trace early",
    )
    parser.add_argument(
        "--max-revisits", type=int, help="maximum revisits per user, excluding first access"
    )
    parser.add_argument("--history-tokens", type=int, default=8192)
    parser.add_argument("--candidate-tokens", type=int, default=1024)
    parser.add_argument(
        "--context-limit",
        type=int,
        help="explicit generation boundary; does not validate model quality",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", choices=("weighted", "sequential"), default="weighted")
    parser.add_argument("--heat-dataset", default="beauty")
    parser.add_argument(
        "--heat-field",
        help="curve field; defaults to pv_share for industrial, interaction_count otherwise",
    )
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        workload = build_workload(
            WorkloadConfig(
                model=args.model,
                num_users=args.num_users,
                requests=args.requests,
                history_tokens=args.history_tokens,
                candidate_tokens=args.candidate_tokens,
                seed=args.seed,
                heat_dataset=args.heat_dataset,
                heat_field=args.heat_field,
                sampling=args.sampling,
                max_revisits=args.max_revisits,
                context_limit=args.context_limit,
            ),
            tokenizer=args.tokenizer,
        )
        workload.write(args.output_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "workload_sha256": workload.manifest["workload_sha256"],
                **workload.manifest["observed"],
            }
        )
    )


if __name__ == "__main__":
    main()
