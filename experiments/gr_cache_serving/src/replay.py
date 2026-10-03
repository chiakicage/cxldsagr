"""Measure the synchronous reduced-layer GR prototype, not online serving.

No model math, cache policy, top-k reordering, or artificial arrivals are added.
Model initialization and kernel warmup are excluded; all measured requests start
from an empty logical prefix cache and execute in trace order without sleeping.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path

from experiments.gr_cache_serving.src.workload import _publish_directory

ROOT = Path(__file__).resolve().parents[3]
MODES = ("resident", "sparse_sync", "echo_gr_adapted", "dense_prefetch")


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def distribution(values):
    return {
        "count": len(values),
        "mean_ms": math.fsum(values) / len(values) if values else None,
        "p50_ms": percentile(values, 0.5),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values) if values else None,
        "sum_ms": math.fsum(values),
    }


def summarize(rows):
    service_ms = math.fsum(row["service_ms"] for row in rows)
    return {
        "requests": len(rows),
        "users": len({row["user_id"] for row in rows}),
        "all": distribution([row["service_ms"] for row in rows]),
        "first_visit": distribution([row["service_ms"] for row in rows if row["first_visit"]]),
        "prefix_hit": distribution([row["service_ms"] for row in rows if row["prefix_reused"]]),
        "revisit": distribution([row["service_ms"] for row in rows if not row["first_visit"]]),
        "revisit_reprefill": distribution(
            [
                row["service_ms"]
                for row in rows
                if not row["first_visit"] and not row["prefix_reused"]
            ]
        ),
        "prefill": distribution([row["prefill_ms"] for row in rows if not row["prefix_reused"]]),
        "candidate": distribution([row["candidate_ms"] for row in rows]),
        "prefix_hit_candidate": distribution(
            [row["candidate_ms"] for row in rows if row["prefix_reused"]]
        ),
        "host_evictions": sum(len(row["evicted_user_ids"]) for row in rows),
        "prefill_count": sum(not row["prefix_reused"] for row in rows),
        "prefill_total_ms": math.fsum(row["prefill_ms"] for row in rows),
        "inverse_mean_service_requests_per_s": len(rows) * 1000 / service_ms
        if service_ms
        else None,
        "prefix_hit_fraction": sum(row["prefix_reused"] for row in rows) / len(rows)
        if rows
        else None,
        "revisit_prefix_hit_fraction": (
            sum(row["prefix_reused"] for row in rows if not row["first_visit"])
            / sum(not row["first_visit"] for row in rows)
        )
        if any(not row["first_visit"] for row in rows)
        else None,
    }


def load_trace(path):
    metadata = json.loads((path.parent / "metadata.json").read_text())
    rows, prefixes = [], {}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for raw in handle:
            digest.update(raw)
            row = json.loads(raw)
            ids = row["input_ids"]
            length = row["stable_prefix_tokens"]
            if (
                length != metadata["stable_prefix_tokens"]
                or len(ids) - length != metadata["candidate_suffix_tokens"]
            ):
                raise ValueError("trace token boundaries disagree with metadata")
            prefix_digest = hashlib.sha256(
                json.dumps(ids[:length], separators=(",", ":")).encode()
            ).hexdigest()
            uid = row["user_id"]
            if prefixes.setdefault(uid, prefix_digest) != prefix_digest:
                raise ValueError("same-user prefix changes within this fixed-history trace")
            rows.append(row)
    if (
        digest.hexdigest() != metadata["requests_sha256"]
        or len(rows) != metadata["stats"]["requests"]
    ):
        raise ValueError("trace digest or request count disagrees with metadata")
    if not rows:
        raise ValueError("trace must not be empty")
    return rows, metadata


@contextmanager
def timed_phases(runner):
    originals = {name: getattr(runner, name) for name in ("prefill", "extend")}
    state = {"prefill_ms": 0.0, "candidate_ms": 0.0}

    def wrap(method, key):
        def call(*args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return method(*args, **kwargs)
            finally:
                state[key] += (time.perf_counter_ns() - start) / 1e6

        return call

    runner.prefill = wrap(originals["prefill"], "prefill_ms")
    runner.extend = wrap(originals["extend"], "candidate_ms")
    try:
        yield state
    finally:
        for name, method in originals.items():
            setattr(runner, name, method)


def source_fingerprints():
    paths = [
        "serving/echo_runner.py",
        "serving/echo_cache_manager.py",
        "models/deepseek_v32/echo_adapter.py",
        "models/deepseek_v32/echo_cache.py",
        "models/deepseek_v32/echo_dense.py",
        "operators/sm90/pinned_gather.py",
        "models/deepseek_v32/echo_kernel.py",
        "models/deepseek_v32/echo_recall.py",
        "models/deepseek_v32/echo_index.py",
        "serving/echo_budget.py",
        "experiments/gr_cache_serving/src/memory_guard.py",
        "experiments/gr_cache_serving/src/host_memory.py",
        "experiments/gr_cache_serving/src/fixed_budget.py",
        "experiments/gr_cache_serving/src/transfer.py",
        "experiments/gr_cache_serving/src/replay.py",
        "experiments/gr_cache_serving/src/preflight.py",
        "experiments/gr_cache_serving/src/workload.py",
        "experiments/gr_cache_serving/scripts/run_replay.sh",
        "experiments/gr_cache_serving/scripts/run_fixed_budget_sweep.sh",
        "experiments/gr_cache_serving/scripts/run_dense_window.sh",
        "experiments/gr_cache_serving/scripts/run_dense_direct.sh",
    ]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def pool_accounting(runner):
    pool = runner.runner.token_to_kv_pool
    device_pool = getattr(pool, "device_pool", pool)
    return {
        "logical_pool_tokens": int(runner.runner.token_to_kv_pool_allocator.size),
        "device_mla_pool_tokens_per_layer": int(device_pool.size),
        "device_mla_buffers_bytes": sum(x.nbytes for x in device_pool.kv_buffer),
        "host_mla_buffers_bytes": sum(x.nbytes for x in pool.kv_buffer)
        if hasattr(pool, "device_pool")
        else 0,
        "resident_index_buffers_bytes": sum(x.nbytes for x in pool.index_k_with_scale_buffer),
        "dense_extra": runner.provenance.get("dense_memory_accounting"),
        "scope": "explicit major buffers only; maps, metadata, model and workspace included separately in torch peak",
        "equal_total_hbm_budget_enforced": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--dense-prefetch-schedule", choices=("layer_end", "attention_window"))
    parser.add_argument("--dense-prefetch-transport", choices=("gpu_direct", "cpu_staging"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path", type=Path, default=Path("/mnt/nfs/share/models/DeepSeek-V3.2")
    )
    parser.add_argument("--host-users", type=int, default=128)
    parser.add_argument("--profile", choices=("fixed_budget_512",))
    parser.add_argument("--host-cache-budget-gib", type=float)
    parser.add_argument("--layers", type=int, choices=(1, 2, 3, 4, 5), default=3)
    parser.add_argument("--hbm-budget-gib", type=float)
    parser.add_argument("--workspace-reserve-gib", type=float, default=8)
    parser.add_argument("--non-torch-reserve-gib", type=float, default=2)
    parser.add_argument("--collect-transfers", action="store_true")
    parser.add_argument("--device-cache-tokens", type=int, default=66624)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="per-request logging; not valid performance output",
    )
    args = parser.parse_args(argv)
    if args.dense_prefetch_schedule is not None and args.mode != "dense_prefetch":
        parser.error("--dense-prefetch-schedule requires --mode dense_prefetch")
    if args.dense_prefetch_transport is not None and args.mode != "dense_prefetch":
        parser.error("--dense-prefetch-transport requires --mode dense_prefetch")
    if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1" and not args.diagnostic:
        parser.error("CUDA_LAUNCH_BLOCKING=1 requires --diagnostic; cannot publish performance")
    if args.diagnostic and args.output_dir.resolve().is_relative_to(ROOT / "experiments"):
        parser.error("diagnostic output must stay outside experiments")
    if args.output_dir.exists():
        parser.error("output directory already exists")
    if args.host_users < 1 or args.device_cache_tokens <= 0 or args.device_cache_tokens % 64:
        parser.error("positive host users and page-aligned device cache required")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        raise RuntimeError(f"GPU already has compute processes; refusing to interfere: {active}")
    source_snapshot = source_fingerprints()
    trace, trace_metadata = load_trace(args.trace)
    prefix = trace_metadata["stable_prefix_tokens"]
    suffix = trace_metadata["candidate_suffix_tokens"]
    host_tokens = args.host_users * prefix + ((suffix + 63) // 64 + 1) * 64
    if prefix % 64 or args.device_cache_tokens < prefix + suffix:
        parser.error("current runner needs aligned prefix and device capacity >= active context")

    host_memory_preflight = None
    host_plan = host_availability = host_guard = None
    if args.host_cache_budget_gib is not None:
        from experiments.gr_cache_serving.src.host_memory import (
            HostCacheGuard,
            inspect_host_availability,
            plan_host_cache,
        )

        host_plan = plan_host_cache(
            users=args.host_users,
            layers=args.layers,
            prefix=prefix,
            suffix=suffix,
            budget_bytes=int(args.host_cache_budget_gib * 2**30),
        )
        host_availability = inspect_host_availability(host_plan)
    if args.mode != "resident":
        from experiments.gr_cache_serving.src.host_memory import inspect_pinned_capacity

        host_memory_preflight = inspect_pinned_capacity(host_tokens * args.layers * 1152)

    import torch

    from experiments.gr_cache_serving.src.transfer import TransferMeter
    from models.deepseek_v32.echo_kernel import PREFETCH_PHASE_PATCH_ID
    from serving.echo_cache_manager import EchoCacheManager
    from serving.echo_runner import open_echo_runner

    budget = guard = None
    if args.hbm_budget_gib is not None:
        from experiments.gr_cache_serving.src.memory_guard import MemoryGuard
        from experiments.gr_cache_serving.src.preflight import audit_checkpoint
        from serving.echo_budget import GIB, plan_echo_budget

        audit = audit_checkpoint(args.model_path, args.layers)
        budget = plan_echo_budget(
            mode=args.mode,
            num_layers=args.layers,
            users=args.host_users,
            prefix_tokens=prefix,
            suffix_tokens=suffix,
            checkpoint_stored_bytes=audit["selected_stored_bytes"],
            total_hbm_bytes=int(args.hbm_budget_gib * GIB),
            workspace_reserve_bytes=int(args.workspace_reserve_gib * GIB),
            non_torch_reserve_bytes=int(args.non_torch_reserve_gib * GIB),
            device_cache_tokens=args.device_cache_tokens,
        )
        if args.profile:
            from experiments.gr_cache_serving.src.fixed_budget import validate_configuration

            validate_configuration(args, trace_metadata, budget)
        host_tokens = budget.logical_pool_tokens
        args.device_cache_tokens = budget.device_cache_tokens
        guard = MemoryGuard(
            torch,
            total_bytes=budget.total_hbm_bytes,
            non_torch_reserve_bytes=budget.non_torch_reserve_bytes,
        )
    elif args.profile:
        parser.error("fixed-budget profile requires the common GPU budget")
    if host_plan:
        host_guard = HostCacheGuard(torch, host_plan)

    started = datetime.now(UTC).isoformat()
    init_start = time.perf_counter()
    rows = []
    with (
        guard if guard else nullcontext(),
        open_echo_runner(
            model_path=args.model_path,
            echo_path=ROOT / "3rdparty/ECHO",
            num_layers=args.layers,
            mode=args.mode,
            max_total_tokens=host_tokens,
            device_cache_tokens=args.device_cache_tokens,
            prefill_chunk=1024,
            kernel_patch=PREFETCH_PHASE_PATCH_ID if args.mode == "echo_gr_adapted" else None,
            mem_fraction_static=budget.torch_limit_bytes
            / torch.cuda.get_device_properties(0).total_memory
            if budget
            else 0.6,
            dense_context_tokens=budget.dense_context_tokens if budget else None,
            dense_prefetch_schedule=args.dense_prefetch_schedule,
            dense_prefetch_transport=args.dense_prefetch_transport,
        ) as runner,
    ):
        init_seconds = time.perf_counter() - init_start
        accounting = pool_accounting(runner)
        accounting["common_budget_plan"] = budget.metadata() if budget else None
        accounting["equal_total_hbm_budget_enforced"] = budget is not None
        if host_guard:
            expected_host_bytes = (
                0 if args.mode == "resident" else host_plan["main_kv_tensor_bytes"]
            )
            if accounting["host_mla_buffers_bytes"] != expected_host_bytes:
                raise MemoryError("actual Host KV layout differs from the fixed budget plan")
            host_guard.sample(runner)
        if guard:
            guard.check()
        warm_start = time.perf_counter()
        warm = EchoCacheManager(runner, host_capacity_tokens=host_tokens, hbm_retained_users=None)
        try:
            for _ in range(2):
                result = warm.execute(trace[0])
                if not bool(torch.isfinite(result.hidden_states).all()):
                    raise RuntimeError("nonfinite warmup output")
                del result
        finally:
            warm.close()
        torch.cuda.synchronize()
        if (
            runner._prefixes
            or runner.runner.token_to_kv_pool_allocator.available_size() != host_tokens
        ):
            raise RuntimeError("warmup did not release all logical prefix slots")
        for allocator in getattr(runner.runner.token_to_kv_pool, "device_pool_allocator", ()):
            if int(allocator.available_size()) != args.device_cache_tokens:
                raise RuntimeError("warmup did not release device KV slots")
        warm_seconds = time.perf_counter() - warm_start
        torch.cuda.reset_peak_memory_stats()
        manager = EchoCacheManager(
            runner, host_capacity_tokens=host_tokens, hbm_retained_users=None
        )
        visits = Counter()
        loop_start = time.perf_counter()
        try:
            with (
                timed_phases(runner) as phase,
                TransferMeter(runner) if args.collect_transfers else nullcontext() as meter,
            ):
                for ordinal, request in enumerate(trace):
                    phase.update(prefill_ms=0.0, candidate_ms=0.0)
                    if meter:
                        meter.begin_request()
                    uid = request["user_id"]
                    if args.diagnostic:
                        print(
                            json.dumps(
                                {
                                    "begin_ordinal": ordinal,
                                    "user_id": uid,
                                    "seen_users": len(visits),
                                    "visit_index": visits[uid],
                                }
                            ),
                            flush=True,
                        )
                    begin = time.perf_counter_ns()
                    try:
                        result = manager.execute(request)
                    except BaseException:
                        print(
                            json.dumps(
                                {
                                    "failed_ordinal": ordinal,
                                    "user_id": uid,
                                    "completed_requests": len(rows),
                                }
                            ),
                            flush=True,
                        )
                        raise
                    elapsed = (time.perf_counter_ns() - begin) / 1e6
                    transfer = meter.finish_request() if meter else {}
                    if guard:
                        guard.check()
                    if host_guard:
                        host_guard.sample(runner, manager)
                    # Validation is outside the timed service interval, never called a quality score.
                    if not bool(torch.isfinite(result.hidden_states).all()):
                        raise RuntimeError(f"nonfinite output at request {ordinal}")
                    torch.cuda.synchronize()
                    rows.append(
                        {
                            "ordinal": ordinal,
                            "user_id": uid,
                            "visit_index": visits[uid],
                            "first_visit": visits[uid] == 0,
                            "prefix_reused": result.prefix_reused,
                            "service_ms": elapsed,
                            **phase,
                            "other_manager_ms": elapsed
                            - phase["prefill_ms"]
                            - phase["candidate_ms"],
                            "evicted_user_ids": list(result.evicted_user_ids),
                            "retained_prefix_tokens": result.retained_tokens,
                            "d2h_mla_payload_bytes": (
                                ((0 if result.prefix_reused else prefix) + suffix)
                                * 1152
                                * args.layers
                                if args.mode != "resident"
                                else 0
                            ),
                            "d2h_counter_scope": "derived from all successful real KV writes, not PCIe bus traffic",
                            **transfer,
                        }
                    )
                    visits[uid] += 1
                    del result
                    if (ordinal + 1) % 32 == 0:
                        print(
                            json.dumps(
                                {
                                    "completed": ordinal + 1,
                                    "mode": args.mode,
                                    "last_service_ms": elapsed,
                                    "elapsed_s": time.perf_counter() - loop_start,
                                }
                            ),
                            flush=True,
                        )
            loop_seconds = time.perf_counter() - loop_start
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            final_prefix_tokens = manager.retained_tokens
        finally:
            manager.close()
        provenance = json.loads(json.dumps(runner.provenance))
        gpu_name = torch.cuda.get_device_name()
        gpu_total = torch.cuda.get_device_properties(0).total_memory

    memory_guard = guard.metadata if guard else None
    if args.profile:
        from experiments.gr_cache_serving.src.fixed_budget import validate_outcomes

        validate_outcomes(rows, budget.retained_users_capacity)
    if source_fingerprints() != source_snapshot:
        raise RuntimeError("measurement sources changed during replay; refusing to publish")
    summary = summarize(rows)
    summary.update(
        {
            "schema": 1,
            "artifact_type": "diagnostic_only"
            if args.diagnostic
            else "synchronous_gr_prototype_replay",
            "mode": args.mode,
            "experiment_profile": args.profile,
            "repetition": args.repetition,
            "started_utc": started,
            "trace_path": str(args.trace.resolve()),
            "trace_metadata": trace_metadata,
            "model_path": str(args.model_path.resolve()),
            "runner_provenance": provenance,
            "source_sha256": source_snapshot,
            "pool_accounting": accounting,
            "host_users_capacity": args.host_users,
            "workload_population_users": trace_metadata["stats"]["population_users"],
            "host_cache_capacity_users": 0 if args.mode == "resident" else args.host_users,
            "configured_device_cache_tokens": args.device_cache_tokens,
            "budget_plan": budget.metadata() if budget else None,
            "memory_guard": memory_guard,
            "host_memory_preflight": host_memory_preflight,
            "host_budget_plan": host_plan,
            "host_availability_preflight": host_availability,
            "host_memory_observation": host_guard.metadata() if host_guard else None,
            "transfer_instrumentation": {
                "enabled": args.collect_transfers,
                "scope": "MLA payload only; GPU counter snapshots included in service time, reduction/readback excluded; no bus overhead or index/model traffic",
            },
            "retained_users_capacity": budget.retained_users_capacity
            if budget
            else args.host_users,
            "hbm_retained_users": None,
            "online_heat_policy": "none; native ECHO FIFO/priority admission",
            "host_policy": "whole-prefix LRU",
            "eviction_tier": "HBM" if args.mode == "resident" else "host",
            "legacy_host_evictions_field": "counts prefix evictions from eviction_tier, including HBM-only",
            "final_retained_prefix_tokens": final_prefix_tokens,
            "model_init_seconds_excluded": init_seconds,
            "warmup_seconds_excluded": warm_seconds,
            "warmup": "first trace request twice, then release all prefix/device slots; not timed",
            "cache_start": "empty logical prefix and MLA device pools after kernel warmup",
            "topk_order": "native; no correctness-test sorting hooks",
            "measurement": {
                "timer": "perf_counter_ns around EchoCacheManager.execute",
                "included": "validation/hash, host LRU, cold prefill when needed, candidate, synchronous forward completion and suffix release",
                "excluded": "trace loading, model startup, kernel warmup, output finite check, logging, teardown"
                + (", Host allocator/metadata/RSS checks" if host_guard else ""),
                "synchronization": "existing runner synchronizes every forward including each 1024-token prefill chunk",
                "arrival_policy": "as-fast-as-possible serial trace order; timestamps not replayed; no queue/network/batching",
                "throughput": "inverse mean service time, NOT measured online requests/s or SLO capacity",
                "output": f"candidate hidden states on GPU, complete first {args.layers} layers; no final norm/head/recommendation",
                "numerical_validation": "every output finite; prior controlled-order integration tests, not native bitwise parity",
            },
            "loop_wall_seconds_including_checks_and_logging": loop_seconds,
            "peak_torch_allocated_bytes": peak_allocated,
            "peak_torch_reserved_bytes": peak_reserved,
            "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "gpu": {"name": gpu_name, "total_bytes": gpu_total},
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "sglang",
                        "sgl-kernel",
                        "deep-gemm",
                        "flash-mla",
                        "fast-hadamard-transform",
                    )
                },
                "cuda_home": os.environ.get("CUDA_HOME"),
            },
        }
    )
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".replay-", dir=args.output_dir.parent) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (stage / "requests.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        snapshot = {
            name: {"sha256": digest, "text": (ROOT / name).read_text()}
            for name, digest in source_snapshot.items()
        }
        (stage / "source_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        _publish_directory(stage, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "summary": summarize(rows)}), flush=True)


if __name__ == "__main__":
    main()
