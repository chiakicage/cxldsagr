"""Fixed-capacity workload contract for the four NOSA serving methods."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/nosa_motivation"
SCHEMA = "nosa-motivation-v2"
CHECK_SCHEMA = "nosa-motivation-check-v1"
BENCH_SCHEMA = "nosa-motivation-bench-v1"
METHODS = ("hbm", "dense_prefetch", "sync_sparse", "async_sparse")
BACKEND_SCHEMES = {
    "hbm": "hbm",
    "dense_prefetch": "dense_prefetch",
    "sync_sparse": "serial_sparse",
    "async_sparse": "overlap",
}
WARMUP_POLICY = "two_users_then_first_user_revisit_v1"
NUMERICAL_ATOL = 0.016
NUMERICAL_RTOL = 0.016


def parser():
    from experiments.nosa_motivation.src.validation import add_mode_arguments

    command = argparse.ArgumentParser(description=__doc__)
    add_mode_arguments(command)
    command.add_argument("--run-id", required=True)
    command.add_argument("--output-dir", type=Path)
    command.add_argument("--model-path", type=Path, default=Path("/mnt/ssd-wlcb/chenkaiqi/NOSA-8B"))
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--num-users", type=int, default=16)
    command.add_argument("--rounds", type=int, default=2)
    command.add_argument("--history-tokens", type=int, default=65536)
    command.add_argument("--candidate-tokens", type=int, default=128)
    command.add_argument("--chunk-size", type=int, default=1024)
    command.add_argument("--sparse-pool-tokens", type=int, default=65536)
    command.add_argument("--host-arena-tokens", type=int, default=16777216)
    command.add_argument(
        "--compute-graphs",
        action="store_true",
        help="prepare validated pure-compute graphs before measured requests",
    )
    command.add_argument("--seed", type=int, default=42)
    command.add_argument(
        "--peak-bf16-tflops",
        type=float,
        default=989.5,
        help="declared dense BF16 peak; 989.5 TFLOP/s is the H200 reference, not sparse peak",
    )
    return command


def configuration(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run ID must contain letters, digits, underscores or hyphens")
    names = (
        "num_users",
        "rounds",
        "history_tokens",
        "candidate_tokens",
        "chunk_size",
        "sparse_pool_tokens",
        "host_arena_tokens",
    )
    if any(type(getattr(args, name)) is not int or getattr(args, name) <= 0 for name in names):
        raise ValueError("workload sizes and capacities must be positive integers")
    if args.rounds < 2 or args.num_users < 2:
        raise ValueError("at least two users and two rounds are required")
    if type(args.seed) is not int or args.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    peak = getattr(args, "peak_bf16_tflops", 989.5)
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError("declared dense BF16 peak must be finite and positive")
    if args.history_tokens % 64 or args.chunk_size % 64:
        raise ValueError(
            "fixed NOSA currently requires history and chunk sizes to be multiples of 64"
        )
    padded_history = (args.history_tokens + 63) // 64 * 64
    if args.sparse_pool_tokens % 64 or args.host_arena_tokens % 64:
        raise ValueError("P and NH must be multiples of 64")
    if args.sparse_pool_tokens < padded_history:
        raise ValueError("dense prefetch requires P to hold a complete padded history")
    if args.host_arena_tokens < padded_history * args.num_users:
        raise ValueError("NH must retain every requested history; this experiment does not fill NH")
    if args.history_tokens + args.candidate_tokens > 262144:
        raise ValueError("history plus candidate exceeds NOSA's supported context limit")
    return {
        **{name: getattr(args, name) for name in names},
        "seed": args.seed,
        "methods": list(METHODS),
        "backend_schemes": dict(BACKEND_SCHEMES),
        "requests_per_method": args.num_users * args.rounds,
        "layers": 32,
        "sampling": "sequential",
        "padded_history_tokens": padded_history,
        "workspace_query_tokens": max(args.chunk_size, args.candidate_tokens),
        "warmup_policy": WARMUP_POLICY,
        "warmup_request_indices": [0, 1, args.num_users],
        "resource_mode": "fixed_pools",
        "byte_subbudgets": None,
        "noncache_headroom_bytes": None,
        "candidate_persistence": "gpu_transient",
        "enable_compute_graphs": getattr(args, "compute_graphs", False),
        "peak_bf16_tflops": peak,
        "numerical_atol": NUMERICAL_ATOL,
        "numerical_rtol": NUMERICAL_RTOL,
        "output": "all candidate normalized hidden states; no LM head or decode",
    }


def resource_limits(config):
    return {
        "max_session_capacity": config["history_tokens"] + config["candidate_tokens"],
        "max_history_tokens": config["history_tokens"],
        "max_candidate_tokens": config["candidate_tokens"],
    }


def build_workload(config, checkpoint):
    from GR.workload import WorkloadConfig
    from GR.workload import build_workload as build

    return build(
        WorkloadConfig(
            model="nosa",
            num_users=config["num_users"],
            requests=config["requests_per_method"],
            history_tokens=config["history_tokens"],
            candidate_tokens=config["candidate_tokens"],
            seed=config["seed"],
            sampling="sequential",
            context_limit=config["history_tokens"] + config["candidate_tokens"],
        ),
        tokenizer=checkpoint,
    )
