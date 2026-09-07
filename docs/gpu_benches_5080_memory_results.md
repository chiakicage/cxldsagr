# RTX 5080 memory microbench results

Date: 2026-07-05

Device:

```text
NVIDIA GeForce RTX 5080
SM 120
SM count: 84
Measured clock used by benchmarks: 1792 MHz
```

## Summary

| Target | Benchmark | Working set / pattern | Result |
| --- | --- | --- | ---: |
| L2 latency | line-stride pointer chase | 16 MB device memory, 128B stride | 348.0 cycles |
| Global memory latency | line-stride pointer chase | 256 MB device memory, 128B stride | 711.7 cycles |
| CPU pinned host latency | mapped pinned line-stride pointer chase | 1 MB mapped host memory, 128B stride | 1243.0 cycles |
| L2 bandwidth, peak | `gpu-l2-stream` fast sweep | 1 MB working set, triad median | 3609 GB/s |
| L2 bandwidth, 4-16 MB | `gpu-l2-stream` fast sweep | 4-16 MB working set, triad median | ~1820 GB/s |
| Global memory bandwidth, peak | `gpu-stream` | STREAM-like scale kernel | 833 GB/s |
| Global memory read bandwidth | `gpu-stream` | read kernel | 788 GB/s |
| CPU pinned host bandwidth | mapped pinned stream read | 256 MB sequential read | 19.64 GB/s |

Latency values are cycles per dependent 128B-strided load. The pinned-host latency uses mapped pinned memory via `cudaHostAlloc(..., cudaHostAllocMapped)` and `cudaHostGetDevicePointer(...)`.

## Raw Outputs

| File | Contents |
| --- | --- |
| `gpu-benches/gpu-stream/rtx5080.txt` | global/HBM stream bandwidth sweep |
| `gpu-benches/gpu-l2-stream/rtx5080_fast.txt` | selected working-set L2 stream sweep |
| `gpu-benches/gpu-host-pinned/rtx5080_line_stride.txt` | mapped pinned host bandwidth and latency plus device latency controls |

## Commands

```bash
make -C gpu-benches/gpu-stream
./gpu-benches/gpu-stream/cuda-stream > gpu-benches/gpu-stream/rtx5080.txt

make -C gpu-benches/gpu-l2-stream
./gpu-benches/gpu-l2-stream/cuda-l2-stream > gpu-benches/gpu-l2-stream/rtx5080_fast.txt

make -C gpu-benches/gpu-host-pinned
./gpu-benches/gpu-host-pinned/cuda-host-pinned --lat-bytes 1048576 --reps 1 \
  > gpu-benches/gpu-host-pinned/rtx5080_line_stride.txt
```

Notes:

- `gpu-stream` was reduced from 256M doubles to 96M doubles per array to avoid OOM on this 16 GB RTX 5080 while still keeping the working set far beyond L2.
- `gpu-l2-stream` was changed to a selected-size sweep rather than the full 210-point sweep, because the full sweep is much longer than needed for this measurement.
- `gpu-host-pinned` is a new benchmark for mapped pinned host-memory stream reads and line-stride pointer-chase latency.
