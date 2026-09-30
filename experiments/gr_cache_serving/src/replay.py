"""Measure the existing synchronous three-layer GR prototype, not online serving.

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
from contextlib import contextmanager
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
        "inverse_mean_service_requests_per_s": len(rows) * 1000 / service_ms
        if service_ms
        else None,
        "prefix_hit_fraction": sum(row["prefix_reused"] for row in rows) / len(rows)
        if rows
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
        "models/deepseek_v32/echo_kernel.py",
        "models/deepseek_v32/echo_recall.py",
        "experiments/gr_cache_serving/src/replay.py",
        "experiments/gr_cache_serving/scripts/run_replay.sh",
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path", type=Path, default=Path("/mnt/nfs/share/models/DeepSeek-V3.2")
    )
    parser.add_argument("--host-users", type=int, default=128)
    parser.add_argument("--device-cache-tokens", type=int, default=66624)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="per-request logging; not valid performance output",
    )
    args = parser.parse_args(argv)
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

    import torch

    from models.deepseek_v32.echo_kernel import PREFETCH_PHASE_PATCH_ID
    from serving.echo_cache_manager import EchoCacheManager
    from serving.echo_runner import open_echo_runner

    started = datetime.now(UTC).isoformat()
    init_start = time.perf_counter()
    rows = []
    with open_echo_runner(
        model_path=args.model_path,
        echo_path=ROOT / "3rdparty/ECHO",
        num_layers=3,
        mode=args.mode,
        max_total_tokens=host_tokens,
        device_cache_tokens=args.device_cache_tokens,
        prefill_chunk=1024,
        kernel_patch=PREFETCH_PHASE_PATCH_ID if args.mode == "echo_gr_adapted" else None,
    ) as runner:
        init_seconds = time.perf_counter() - init_start
        accounting = pool_accounting(runner)
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
            with timed_phases(runner) as phase:
                for ordinal, request in enumerate(trace):
                    phase.update(prefill_ms=0.0, candidate_ms=0.0)
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
            "repetition": args.repetition,
            "started_utc": started,
            "trace_path": str(args.trace.resolve()),
            "trace_metadata": trace_metadata,
            "model_path": str(args.model_path.resolve()),
            "runner_provenance": provenance,
            "source_sha256": source_snapshot,
            "pool_accounting": accounting,
            "host_users_capacity": args.host_users,
            "configured_device_cache_tokens": args.device_cache_tokens,
            "hbm_retained_users": None,
            "online_heat_policy": "none; native ECHO FIFO/priority admission",
            "host_policy": "whole-prefix LRU",
            "final_retained_prefix_tokens": final_prefix_tokens,
            "model_init_seconds_excluded": init_seconds,
            "warmup_seconds_excluded": warm_seconds,
            "warmup": "first trace request twice, then release all prefix/device slots; not timed",
            "cache_start": "empty logical prefix and MLA device pools after kernel warmup",
            "topk_order": "native; no correctness-test sorting hooks",
            "measurement": {
                "timer": "perf_counter_ns around EchoCacheManager.execute",
                "included": "validation/hash, host LRU, cold prefill when needed, candidate, synchronous forward completion and suffix release",
                "excluded": "trace loading, model startup, kernel warmup, output finite check, logging, teardown",
                "synchronization": "existing runner synchronizes every forward including each 1024-token prefill chunk",
                "arrival_policy": "as-fast-as-possible serial trace order; timestamps not replayed; no queue/network/batching",
                "throughput": "inverse mean service time, NOT measured online requests/s or SLO capacity",
                "output": "candidate hidden states on GPU, complete first 3 layers; no final norm/head/recommendation",
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
        _publish_directory(stage, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "summary": summarize(rows)}), flush=True)


if __name__ == "__main__":
    main()
