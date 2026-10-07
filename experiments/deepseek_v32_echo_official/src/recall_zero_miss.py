"""Isolate the unchanged official recall kernel's zero-miss scan cost.

This diagnostic uses an all-false device bitmap and unused one-record buffers.
It measures no host KV transfers and does not validate full ECHO numerics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
from functools import partial
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "bench", "profile"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--host-sizes", type=int, nargs="+", default=[65536, 1048576, 16777216])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(size <= 0 for size in args.host_sizes):
        raise ValueError("host sizes must be positive")

    # Reuse the Engine launcher's experiment-owned writable paths before imports.
    from experiments.deepseek_v32_echo_official.scripts.run_engine import (
        ECHO_ROOT,
        ROOT,
        scoped_runtime_environment,
    )

    if not args.output.resolve().is_relative_to((ROOT / "output/data").resolve()):
        raise ValueError("output must be inside this experiment's output/data")
    os.environ.update(scoped_runtime_environment(dict(os.environ)))

    import torch
    import triton

    source = ECHO_ROOT / "sglang/python/sglang/srt/mem_cache/recall_ops.py"
    identity = {
        "source": str(source),
        "source_sha256": digest(source),
        "harness_sha256": digest(Path(__file__)),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "flag_value": False,
        "block": 256,
        "num_warps": 4,
    }
    if args.mode != "check":
        if args.check is None:
            raise ValueError("bench/profile requires an independent --check")
        checked = json.loads(args.check.read_text())
        if checked["mode"] != "check" or checked["identity"] != identity:
            raise ValueError("check identity mismatch")
        if not set(args.host_sizes) <= {row["host_size"] for row in checked["rows"]}:
            raise ValueError("host size was not checked")
        if not all(row["unchanged"] for row in checked["rows"]):
            raise ValueError("zero-miss check failed")

    spec = importlib.util.spec_from_file_location("official_recall_diagnostic", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module._recall_update_extend_kernel
    rows = []
    for size in args.host_sizes:
        flags = torch.zeros(size, dtype=torch.bool, device="cuda")
        counter = torch.zeros(1, dtype=torch.uint32, device="cuda")
        device_ids = torch.ones(1, dtype=torch.int32, device="cuda")
        host_map = torch.full((1,), 123, dtype=torch.int32, device="cuda")
        device_map = torch.full((1,), 456, dtype=torch.int64, device="cuda")
        device_kv = torch.full((1, 1, 576), 3, dtype=torch.bfloat16, device="cuda")
        unused_host_kv = torch.full_like(device_kv, 7)

        launch = partial(
            kernel[(triton.cdiv(size, 256),)],
            flags,
            counter,
            device_ids,
            host_map,
            device_map,
            device_kv,
            unused_host_kv,
            size,
            512,
            64,
            256,
        )

        compiled = launch()
        torch.cuda.synchronize()
        row = {"host_size": size, "grid_x": triton.cdiv(size, 256)}
        if args.mode == "bench":
            for _ in range(10):
                launch()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(20):
                    launch()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            samples = []
            for _ in range(20):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / 20)
            row.update(samples_ms=samples, median_ms=statistics.median(samples))
            del graph
        unchanged = (
            counter.item() == 0
            and host_map.item() == 123
            and device_map.item() == 456
            and bool((device_kv == 3).all())
            and bool((unused_host_kv == 7).all())
            and not bool(flags.any())
            and device_ids.item() == 1
        )
        if not unchanged:
            raise AssertionError("zero-miss kernel modified its inputs or outputs")
        row["unchanged"] = unchanged
        row["registers"] = compiled.n_regs
        rows.append(row)

    result = {
        "schema": "official-echo-zero-miss-diagnostic-v1",
        "mode": args.mode,
        "identity": identity,
        "rows": rows,
        "boundary": "All-false bitmap; no KV transfer. Bench is CUDA-event time for 20 unchanged kernel nodes per graph replay, 20 samples after warmup; excludes bitmap clear, sum, allocator, and Engine. Profile runs are separate.",
        "check": str(args.check) if args.check else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
