"""Plan ECHO capacities from total HBM times a fraction minus model memory.

The result reserves host pages for histories, per-layer transient candidate
KV tails and one merged indexer scratch for a whole candidate batch.
Planning is not GPU execution or capacity validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-id", required=True)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--model-path", type=Path, default=Path("/preset-models"))
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--dram-budget-gib", type=float, required=True)
    result.add_argument("--hbm-fraction", type=float, default=0.9)
    result.add_argument(
        "--total-hbm-gib", type=float, help="supply with --model-hbm-gib for CPU planning"
    )
    result.add_argument(
        "--model-hbm-gib",
        type=float,
        help="model loading footprint; omit both HBM values to measure",
    )
    result.add_argument("--noncache-headroom-gib", type=float, default=0.0)
    result.add_argument(
        "--noncache-headroom-note", help="source and validation boundary of the planning headroom"
    )
    dimension = result.add_mutually_exclusive_group(required=True)
    dimension.add_argument("--fixed-p", type=int, help="keep P fixed and maximize whole-user NH")
    dimension.add_argument("--fixed-nh", type=int, help="keep NH fixed and maximize P")
    result.add_argument("--history-tokens", type=int, default=65536)
    result.add_argument("--candidate-tokens", type=int, default=128)
    result.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="history prefill chunk; candidates execute as one whole batch",
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run ID must contain letters, digits, underscores or hyphens")
    for name in ("dram_budget_gib", "noncache_headroom_gib", "total_hbm_gib", "model_hbm_gib"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(args.hbm_fraction) or not 0 < args.hbm_fraction <= 1:
        raise ValueError("hbm_fraction must be in (0, 1]")
    if (args.total_hbm_gib is None) != (args.model_hbm_gib is None):
        raise ValueError("supply both total and model HBM, or omit both to measure")
    if min(args.history_tokens, args.candidate_tokens, args.chunk_size) < 1:
        raise ValueError("history, candidate and chunk size must be positive")
    root = Path(__file__).resolve().parents[3]
    output = args.output_dir or root / "experiments/cache_management/output/data" / args.run_id
    if output.exists():
        raise FileExistsError(output)
    from models.deepseek_v32.config import Config
    from models.deepseek_v32.execution.capacity import EchoCapacityPlanner

    cfg = Config.from_checkpoint(args.model_path)
    hardware = {"memory_source": "supplied_total_and_model_memory"}
    if args.total_hbm_gib is None:
        import torch

        from models.deepseek_v32.execution.adapter import DeepSeekServingBackend

        os.environ.setdefault("CXLDSAGR_SM90_BACKEND", "native")
        torch.cuda.set_device(args.device)
        torch.cuda.synchronize(args.device)
        torch.cuda.empty_cache()
        free_before, total = torch.cuda.mem_get_info(args.device)
        allocated_before = torch.cuda.memory_allocated(args.device)
        backend = DeepSeekServingBackend(
            args.model_path,
            scheme="echo",
            device=args.device,
            num_layers=10,
            chunk_size=args.chunk_size,
            sparse_pool_tokens=max(32768, args.chunk_size, args.candidate_tokens),
            workspace_query_tokens=max(args.chunk_size, args.candidate_tokens),
        )
        try:
            torch.cuda.synchronize(args.device)
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(args.device)
            model_bytes = free_before - free
            if model_bytes < 0:
                raise RuntimeError("device free memory grew during loading; repeat on an idle GPU")
            hardware = {
                "memory_source": "device_free_memory_delta_across_ten_block_model_loading_after_empty_cache",
                "gpu_name": torch.cuda.get_device_name(args.device),
                "total_hbm_bytes": total,
                "free_hbm_before_model_bytes": free_before,
                "free_hbm_after_model_bytes": free,
                "model_hbm_bytes": model_bytes,
                "torch_model_allocation_delta_bytes": torch.cuda.memory_allocated(args.device)
                - allocated_before,
                "torch_allocated_after_model_bytes": torch.cuda.memory_allocated(args.device),
                "torch_reserved_after_model_bytes": torch.cuda.memory_reserved(args.device),
            }
        except BaseException as measurement_error:
            try:
                backend.close()
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve both original failures
                raise BaseExceptionGroup(
                    "Model memory measurement and resource cleanup failed",
                    [measurement_error, cleanup_error],
                ) from None
            raise
        else:
            backend.close()
    else:
        total = int(args.total_hbm_gib * 2**30)
        model_bytes = int(args.model_hbm_gib * 2**30)
        hardware.update(total_hbm_bytes=total, model_hbm_bytes=model_bytes)
    planner = EchoCapacityPlanner(
        cfg,
        num_layers=10,
        history_tokens=args.history_tokens,
        candidate_tokens=args.candidate_tokens,
        chunk_size=args.chunk_size,
        total_hbm_bytes=total,
        model_hbm_bytes=model_bytes,
        dram_budget_bytes=int(args.dram_budget_gib * 2**30),
        hbm_fraction=args.hbm_fraction,
        noncache_headroom_bytes=int(args.noncache_headroom_gib * 2**30),
    )
    plan = (
        planner.max_host_capacity(sparse_pool_tokens=args.fixed_p)
        if args.fixed_p is not None
        else planner.max_device_capacity(host_arena_tokens=args.fixed_nh)
    )
    sources, source_contents = {}, {}
    for relative in (
        "models/deepseek_v32/config.py",
        "models/deepseek_v32/execution/capacity.py",
        "models/deepseek_v32/execution/adapter.py",
        "models/deepseek_v32/execution/cache_resources.py",
        "models/deepseek_v32/execution/planning.py",
        "models/deepseek_v32/cache/session.py",
        "cache/capacity.py",
        "cache/sparse_token_pool.py",
        "cache/host_allocation.py",
        "pyproject.toml",
        "uv.lock",
        "experiments/cache_management/src/capacity_plan.py",
    ):
        source_contents[relative] = (root / relative).read_bytes()
        sources[relative] = hashlib.sha256(source_contents[relative]).hexdigest()
    result = {
        "schema": "echo-capacity-plan-v1",
        "run_id": args.run_id,
        "status": "static_plan_not_execution",
        "candidate_persistence": planner.candidate_persistence,
        "checkpoint_config_sha256": hashlib.sha256(
            (args.model_path / "config.json").read_bytes()
        ).hexdigest(),
        "hardware": hardware,
        "parameters": {
            "candidate_persistence": planner.candidate_persistence,
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        "sources": sources,
        "plan": plan,
    }
    output.mkdir(parents=True)
    (output / "checkpoint_config.json").write_bytes((args.model_path / "config.json").read_bytes())
    for relative, contents in source_contents.items():
        snapshot = output / "source" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(contents)
    (output / "plan.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
