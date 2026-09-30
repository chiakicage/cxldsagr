"""Prepare a reproducible GR request trace, without running a serving benchmark.

Atomic publication requires Linux renameat2(RENAME_NOREPLACE) support.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_directory(source: Path, destination: Path) -> None:
    """Publish a complete trace atomically without replacing any existing target."""
    if sys.platform != "linux":
        raise OSError(errno.ENOSYS, "atomic publication requires Linux renameat2")
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise OSError(errno.ENOSYS, "libc does not expose renameat2") from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    # AT_FDCWD=-100; RENAME_NOREPLACE=1 also protects existing empty directories.
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        message = os.strerror(code)
        if code in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
            message = "atomic publication requires renameat2(RENAME_NOREPLACE) support"
        raise OSError(code, message, str(destination))


def _positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def text_config_kwargs(prefix_tokens: int, candidate_tokens: int, instruction_tokens: int) -> dict:
    """Map physical prefix/suffix lengths to GR's user/item budget convention."""
    for name, value in (
        ("prefix_tokens", prefix_tokens),
        ("candidate_tokens", candidate_tokens),
        ("instruction_tokens", instruction_tokens),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if prefix_tokens <= instruction_tokens:
        raise ValueError("prefix_tokens must exceed the encoded instruction length")
    return {
        "user_lengths": (prefix_tokens - instruction_tokens,),
        "user_probabilities": (1.0,),
        "item_lengths": (candidate_tokens + instruction_tokens,),
        "item_probabilities": (1.0,),
        "max_input_tokens": prefix_tokens + candidate_tokens,
    }


def create_generator(args):
    # The preparation CLI and its validation tests need no model or tokenizer imports.
    from tokenizers import Tokenizer

    from GR import dataset, heat, input_generator, scheduling
    from models.deepseek_v32 import request_format

    token_path = args.tokenizer.resolve()
    if token_path.is_dir():
        token_path /= "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(token_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    instruction_tokens = len(
        tokenizer.encode(
            request_format.prefix(input_generator.INSTRUCTION), add_special_tokens=False
        ).ids
    )
    text_config = input_generator.TextConfig(
        **text_config_kwargs(args.prefix_tokens, args.candidate_tokens, instruction_tokens),
        history_cache_users=args.history_cache_users,
    )
    schedule_config = scheduling.ScheduleConfig(
        seed=args.seed, qps=args.qps, arrival=args.arrival, sampling=args.sampling
    )
    generator = input_generator.create_input_generator(
        model="deepseek_v32",
        heat_source="curve",
        curve_dataset=args.curve_dataset,
        num_users=args.num_users,
        text_material="synthetic",
        tokenizer=tokenizer,
        text_config=text_config,
        schedule_config=schedule_config,
    )
    root = Path(__file__).resolve().parents[3]
    source_paths = [
        Path(module.__file__).resolve()
        for module in (input_generator, scheduling, heat, dataset, request_format)
    ]
    source_paths.append(Path(__file__).resolve())
    metadata = {
        "model": "deepseek_v32",
        "heat_source": "curve",
        "curve_dataset": args.curve_dataset,
        "heat": generator.population.metadata,
        "text_material": "synthetic",
        "text_config": asdict(text_config),
        "schedule_config": asdict(schedule_config),
        "tokenizer_path": str(token_path),
        "tokenizer_sha256": _sha256_file(token_path),
        "tokenizers_version": version("tokenizers"),
        "source_sha256": {str(path.relative_to(root)): _sha256_file(path) for path in source_paths},
    }
    return generator, metadata


class _TraceValidation:
    def __init__(self, generator, prefix_tokens: int, candidate_tokens: int):
        self.weights = generator.population.weights
        self.instruction_ids = generator.prefix_ids
        self.instruction_tokens = len(generator.prefix_ids)
        self.prefix_tokens = prefix_tokens
        self.candidate_tokens = candidate_tokens
        text_config_kwargs(prefix_tokens, candidate_tokens, self.instruction_tokens)
        self.counts = Counter()
        self.prefix_digests = {}
        self.previous = {}
        self.timestamp = -1.0
        self.total = 0

    def validate(self, row: dict) -> dict:
        uid = row["user_id"]
        if type(uid) is not int or uid not in self.weights:
            raise ValueError("request user_id is absent from the configured population")
        visit = row["visit_index"]
        if type(visit) is not int or visit != self.counts[uid]:
            raise ValueError(f"user {uid}: visit_index must start at zero and increase by one")
        if type(row["task_id"]) is not int or row["task_id"] != self.total:
            raise ValueError("task_id must start at zero and increase by one")
        ids = row["input_ids"]
        if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("input_ids must be a list of nonnegative integers")
        total_tokens = self.prefix_tokens + self.candidate_tokens
        expected = {
            "model": "deepseek_v32",
            "instruction_tokens": self.instruction_tokens,
            "stable_prefix_tokens": self.prefix_tokens,
            "candidate_suffix_tokens": self.candidate_tokens,
            "total_input_tokens": total_tokens,
            "user_tokens": self.prefix_tokens - self.instruction_tokens,
            "item_tokens": self.candidate_tokens + self.instruction_tokens,
            "history_token_span": [self.instruction_tokens, self.prefix_tokens],
            "candidate_token_span": [self.prefix_tokens, total_tokens],
        }
        if len(ids) != total_tokens:
            raise ValueError("input_ids length does not match the requested exact token budget")
        if ids[: self.instruction_tokens] != self.instruction_ids:
            raise ValueError("request instruction token IDs differ from the generator prefix")
        for name, value in expected.items():
            if row[name] != value:
                raise ValueError(f"{name} does not match the requested exact token boundaries")
        digest = hashlib.sha256(_json_bytes(ids[: self.prefix_tokens])).hexdigest()
        if uid in self.prefix_digests and self.prefix_digests[uid] != digest:
            raise ValueError(f"user {uid}: stable prefix token IDs changed on a revisit")
        timestamp = row["timestamp"]
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
            or timestamp < 0
            or timestamp < self.timestamp
        ):
            raise ValueError("timestamps must be finite, nonnegative and nondecreasing")
        previous = self.previous.get(uid)
        if row["previous_request_id"] != (previous[0] if previous else None):
            raise ValueError("previous_request_id does not match the previous visit")
        interval = row["revisit_interval_s"]
        if previous:
            if not isinstance(interval, (int, float)) or not math.isclose(
                interval, timestamp - previous[1], rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError("revisit_interval_s does not match the request timestamps")
        elif interval is not None:
            raise ValueError("the first visit must not have a revisit interval")
        self.prefix_digests[uid] = digest
        self.previous[uid] = (row["task_id"], timestamp)
        self.timestamp = timestamp
        self.counts[uid] += 1
        self.total += 1
        return {**row, "stable_prefix_sha256": digest}


def write_workload(
    generator,
    *,
    output_dir: Path,
    count: int,
    prefix_tokens: int,
    candidate_tokens: int,
    generator_metadata: dict,
) -> dict:
    """Validate and stage every artifact before publishing a new trace directory."""
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    validator = _TraceValidation(generator, prefix_tokens, candidate_tokens)
    metadata_fingerprint = hashlib.sha256(_json_bytes(generator_metadata)).hexdigest()
    ancestor = output_dir.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    # Staging in an existing ancestor keeps the final rename on the same filesystem.
    with tempfile.TemporaryDirectory(prefix=".gr-workload-", dir=ancestor) as temporary:
        stage = Path(temporary) / "trace"
        stage.mkdir()
        requests_digest = hashlib.sha256()
        with (stage / "requests.jsonl").open("wb") as destination:
            for row in generator.iter_generate(count):
                if validator.total >= count:
                    raise ValueError("generator yielded more requests than requested")
                encoded = _json_bytes(validator.validate(row)) + b"\n"
                requests_digest.update(encoded)
                destination.write(encoded)
        if validator.total != count:
            raise ValueError("generator yielded fewer requests than requested")
        weights = generator.population.weights.values()
        if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError("population weights must be finite and positive")
        weight_sum = math.fsum(weights)
        if not math.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError("population weights must have a finite, positive sum")
        with (stage / "users.jsonl").open("wb") as destination:
            for uid, weight in sorted(generator.population.weights.items()):
                user = {
                    "user_id": uid,
                    "weight": weight,
                    "normalized_heat_weight": weight / weight_sum,
                    "request_count": validator.counts[uid],
                    "stable_prefix_tokens": prefix_tokens,
                    "candidate_suffix_tokens": candidate_tokens,
                    "history_token_span": [validator.instruction_tokens, prefix_tokens],
                    "candidate_token_span": [prefix_tokens, prefix_tokens + candidate_tokens],
                    "stable_prefix_sha256": validator.prefix_digests.get(uid),
                    "generator_metadata_sha256": metadata_fingerprint,
                }
                destination.write(_json_bytes(user) + b"\n")
        stats = {
            "requests": count,
            "population_users": len(generator.population.weights),
            "observed_users": len(validator.prefix_digests),
            "first_visit_requests": len(validator.prefix_digests),
            "revisit_requests": count - len(validator.prefix_digests),
            "total_input_tokens": count * (prefix_tokens + candidate_tokens),
        }
        metadata = {
            "schema_version": 1,
            "artifact_type": "prepared_gr_workload",
            "serving_executed": False,
            "content_is_synthetic": True,
            "timestamp_unit": "seconds",
            "prefix_digest_encoding": "UTF-8 compact JSON integer array",
            "unvisited_user_prefix_digest": None,
            "generator": generator_metadata,
            "generator_metadata_sha256": metadata_fingerprint,
            "instruction_tokens": validator.instruction_tokens,
            "stable_prefix_tokens": prefix_tokens,
            "candidate_suffix_tokens": candidate_tokens,
            "history_token_span": [validator.instruction_tokens, prefix_tokens],
            "candidate_token_span": [prefix_tokens, prefix_tokens + candidate_tokens],
            "requests_sha256": requests_digest.hexdigest(),
            "users_sha256": _sha256_file(stage / "users.jsonl"),
            "stats": stats,
        }
        (stage / "requests.sha256").write_text(
            f"{requests_digest.hexdigest()}  requests.jsonl\n", encoding="utf-8"
        )
        (stage / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # Parents may be shared with concurrent runs; only private staging is cleaned up.
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        _publish_directory(stage, output_dir)
    return metadata


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New trace directory")
    parser.add_argument("--prefix-tokens", type=_positive_int, default=65536)
    parser.add_argument("--candidate-tokens", type=_positive_int, default=1024)
    parser.add_argument("--num-users", type=_positive_int, default=128)
    parser.add_argument("--count", type=_positive_int, default=512)
    parser.add_argument(
        "--history-cache-users",
        type=_positive_int,
        default=8,
        help="CPU text-generation cache only; does not enable serving KV reuse",
    )
    parser.add_argument("--curve-dataset", default="beauty")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qps", type=_positive_int, default=1)
    parser.add_argument("--arrival", choices=("poisson", "constant"), default="poisson")
    parser.add_argument(
        "--sampling", choices=("weighted", "uniform", "sequential"), default="weighted"
    )
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise FileExistsError(f"output directory already exists: {args.output_dir}")
        generator, metadata = create_generator(args)
        result = write_workload(
            generator,
            output_dir=args.output_dir,
            count=args.count,
            prefix_tokens=args.prefix_tokens,
            candidate_tokens=args.candidate_tokens,
            generator_metadata=metadata,
        )
    except (ValueError, TypeError, KeyError, OSError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output_dir": str(args.output_dir), **result["stats"]}))


if __name__ == "__main__":
    main()
