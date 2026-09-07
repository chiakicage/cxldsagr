# RTX 5080 benchmark results

Date: 2026-07-01

Command:

```bash
.venv/bin/python docs/benchmark_5080.py
```

Environment:

```text
GPU: NVIDIA GeForce RTX 5080
Driver: 595.71.05
System CUDA shown by nvidia-smi: 13.2
PyTorch: 2.12.1+cu130
PyTorch CUDA runtime: 13.0
Compute capability: 12.0
SMs: 84
Total memory: 15.47 GiB
Persistence mode: enabled
Clock state: user locked / boosted; sustained BF16 GEMM sampled around 2.84-2.86 GHz SM clock
Power limit: 360 W
```

Summary:

| Test | Size | Best result |
| --- | ---: | ---: |
| FP16 tensor-core GEMM | 8192 | 120.44 TFLOP/s |
| BF16 tensor-core GEMM | 16384 | 120.08 TFLOP/s |
| FP32 GEMM with TF32 allowed | 12288 | 60.90 TFLOP/s |
| FP32 GEMM with TF32 disabled | 4096 | 38.66 TFLOP/s |
| STREAM-like add, c = a + alpha * b | 1.0 GiB per array | 843.8 GB/s |
| Device-to-device copy, logical bytes | 2.0 GiB | 411.3 GB/s |
| Device-to-device copy, HBM read + write | 2.0 GiB | 822.5 GB/s |

Raw GEMM sweep:

```text
GEMM sweep: best_ms best_TFLOP/s median_ms median_TFLOP/s
FP16           n= 8192 best_ms=   9.129 best= 120.44 median_ms=   9.235 median= 119.06
FP16           n=12288 best_ms=  31.115 best= 119.26 median_ms=  31.303 median= 118.54
FP16           n=16384 best_ms=  73.747 best= 119.27 median_ms=  73.776 median= 119.23
BF16           n= 8192 best_ms=   9.155 best= 120.10 median_ms=   9.155 median= 120.10
BF16           n=12288 best_ms=  31.136 best= 119.18 median_ms=  31.187 median= 118.99
BF16           n=16384 best_ms=  73.251 best= 120.08 median_ms=  73.306 median= 119.99
FP32_TF32_on   n= 8192 best_ms=  18.446 best=  59.61 median_ms=  18.495 median=  59.45
FP32_TF32_on   n=12288 best_ms=  60.929 best=  60.90 median_ms=  60.933 median=  60.90
FP32_TF32_on   n=16384 best_ms= 144.525 best=  60.86 median_ms= 145.518 median=  60.45
FP32_TF32_off  n= 4096 best_ms=   3.555 best=  38.66 median_ms=   3.720 median=  36.94
FP32_TF32_off  n= 8192 best_ms=  28.692 best=  38.32 median_ms=  28.833 median=  38.13
FP32_TF32_off  n=12288 best_ms= 100.090 best=  37.08 median_ms= 100.146 median=  37.05
```

Raw bandwidth sweep:

```text
Bandwidth sweep
add size=0.5GiB/array best_ms=  1.910 best_GB/s=  843.1 median_GB/s=  843.0
copy size=0.5GiB      best_ms=  1.310 logical_GB/s=  409.7 hbm2x_GB/s=  819.5
add size=1.0GiB/array best_ms=  3.818 best_GB/s=  843.8 median_GB/s=  843.6
copy size=1.0GiB      best_ms=  2.629 logical_GB/s=  408.4 hbm2x_GB/s=  816.8
add size=2.0GiB/array best_ms=  7.648 best_GB/s=  842.4 median_GB/s=  842.2
copy size=2.0GiB      best_ms=  5.222 logical_GB/s=  411.3 hbm2x_GB/s=  822.5
```

Notes:

- GEMM throughput is computed as `2 * n^3 / time`.
- The copy benchmark reports both logical copied bytes and HBM traffic. A device-to-device copy reads from source and writes to destination, so the HBM read-plus-write number is about 2x the logical copy bandwidth.
- The benchmark uses CUDA event timing, not Python wall-clock timing.
- These results were taken after enabling persistence mode and locking/boosting graphics clocks. The earlier 2026-06-30 run saw FP16/BF16 around 76 TFLOP/s because sustained GEMM ran around 1.79 GHz SM clock; this run sampled sustained BF16 GEMM around 2.84-2.86 GHz.
