#!/usr/bin/env python3
"""Measure RTX 5080 GEMM throughput and memory bandwidth with PyTorch.

Run from the repository root with:

    .venv/bin/python draft/benchmark_5080.py

The script uses CUDA events, so timings are device-side rather than wall-clock
Python timings.
"""

from __future__ import annotations

import gc
import statistics
import subprocess

import torch


def gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


def try_print_nvidia_smi() -> None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic helper only.
        print(f"nvidia-smi unavailable: {exc}")
        return

    print("NVIDIA-SMI")
    print(output.rstrip())
    print("")


def bench_stats(fn, *, warmup: int, inner: int, outer: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_ms = []
    for _ in range(outer):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end) / inner)

    return min(samples_ms), statistics.median(samples_ms)


def gemm_case(
    label: str,
    dtype: torch.dtype,
    n: int,
    *,
    tf32: bool | None,
) -> tuple[str, int, float, float, float, float]:
    if tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.set_float32_matmul_precision("high" if tf32 else "highest")

    a = torch.randn((n, n), device="cuda", dtype=dtype)
    b = torch.randn((n, n), device="cuda", dtype=dtype)
    c = torch.empty((n, n), device="cuda", dtype=dtype)
    torch.cuda.synchronize()

    inner = 8 if n >= 12288 else 12
    best_ms, median_ms = bench_stats(
        lambda: torch.mm(a, b, out=c),
        warmup=6,
        inner=inner,
        outer=6,
    )
    flops = 2.0 * n * n * n
    best_tflops = flops / (best_ms * 1e-3) / 1e12
    median_tflops = flops / (median_ms * 1e-3) / 1e12

    del a, b, c
    torch.cuda.empty_cache()
    gc.collect()
    return label, n, best_ms, best_tflops, median_ms, median_tflops


def run_gemm_sweep() -> None:
    print("GEMM sweep: best_ms best_TFLOP/s median_ms median_TFLOP/s")
    cases = [
        ("FP16", torch.float16, [8192, 12288, 16384], None),
        ("BF16", torch.bfloat16, [8192, 12288, 16384], None),
        ("FP32_TF32_on", torch.float32, [8192, 12288, 16384], True),
        ("FP32_TF32_off", torch.float32, [4096, 8192, 12288], False),
    ]

    for label, dtype, sizes, tf32 in cases:
        for n in sizes:
            try:
                row = gemm_case(label, dtype, n, tf32=tf32)
            except RuntimeError as exc:
                print(f"{label:14s} n={n:5d} FAILED {type(exc).__name__}: {exc}")
                torch.cuda.empty_cache()
                gc.collect()
                continue

            _, _, best_ms, best_tflops, median_ms, median_tflops = row
            print(
                f"{label:14s} n={n:5d} "
                f"best_ms={best_ms:8.3f} best={best_tflops:7.2f} "
                f"median_ms={median_ms:8.3f} median={median_tflops:7.2f}",
                flush=True,
            )


def run_bandwidth_sweep() -> None:
    print("")
    print("Bandwidth sweep")
    for gib_per_array in [0.5, 1.0, 2.0]:
        n = int(gib_per_array * (1024**3) / 4)
        n = n // 1024 * 1024

        a = torch.empty(n, device="cuda", dtype=torch.float32)
        b = torch.empty(n, device="cuda", dtype=torch.float32)
        c = torch.empty(n, device="cuda", dtype=torch.float32)
        a.normal_()
        b.normal_()
        c.zero_()
        torch.cuda.synchronize()

        inner = 40 if gib_per_array <= 1.0 else 20
        best_ms, median_ms = bench_stats(
            lambda: torch.add(a, b, alpha=1.2345, out=c),
            warmup=10,
            inner=inner,
            outer=7,
        )
        bytes_moved = 3 * n * 4
        print(
            f"add size={gib_per_array:3.1f}GiB/array "
            f"best_ms={best_ms:7.3f} "
            f"best_GB/s={bytes_moved / (best_ms * 1e-3) / 1e9:7.1f} "
            f"median_GB/s={bytes_moved / (median_ms * 1e-3) / 1e9:7.1f}",
            flush=True,
        )

        del c
        torch.cuda.empty_cache()
        gc.collect()

        dst = torch.empty_like(a)
        torch.cuda.synchronize()
        best_ms, median_ms = bench_stats(
            lambda: dst.copy_(a),
            warmup=10,
            inner=inner,
            outer=7,
        )
        bytes_copied = n * 4
        print(
            f"copy size={gib_per_array:3.1f}GiB      "
            f"best_ms={best_ms:7.3f} "
            f"logical_GB/s={bytes_copied / (best_ms * 1e-3) / 1e9:7.1f} "
            f"hbm2x_GB/s={2 * bytes_copied / (best_ms * 1e-3) / 1e9:7.1f}",
            flush=True,
        )

        del a, b, dst
        torch.cuda.empty_cache()
        gc.collect()


def main() -> None:
    torch.cuda.init()
    try_print_nvidia_smi()

    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print("DEVICE")
    print(f"name: {props.name}")
    print(f"compute capability: {props.major}.{props.minor}")
    print(f"SMs: {props.multi_processor_count}")
    print(f"total memory: {gib(total)}")
    print(f"free memory before bench: {gib(free)}")
    print(f"torch: {torch.__version__}, torch cuda: {torch.version.cuda}")
    print("")

    run_gemm_sweep()
    run_bandwidth_sweep()


if __name__ == "__main__":
    main()
