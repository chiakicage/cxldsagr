# RTX 5080 `cp.async.bulk.prefetch` host-memory notes

Date: 2026-07-05

Device:

```text
NVIDIA GeForce RTX 5080
SM 120
SM count: 84
```

## What the instruction does

The instruction studied here is the ordinary-address form:

```ptx
cp.async.bulk.prefetch.L2.global [srcMem], size;
```

According to NVIDIA PTX ISA 9.3, it is a non-blocking instruction that may
initiate an asynchronous prefetch from global memory to L2. The source address
must be 16-byte aligned, and `size` must be a multiple of 16 bytes. It is a weak
memory operation and requires `sm_90` or higher.

This is different from `prefetch.tensormap` / CuTe
`prefetch_tma_descriptor`, which prefetches a TMA descriptor, not payload data.

References:

- PTX ISA 9.3, `cp.async.bulk.prefetch`: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- CUDA Programming Guide, asynchronous bulk copies: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html

## Local benchmark

Benchmark code:

```text
gpu-benches/gpu-tma-host/
```

The benchmark compares normal vectorized loads with two prefetch modes:

- `stream no-prefetch`: each CTA reads a contiguous tile using normal global
  loads.
- `stream prefetch-ahead`: thread 0 in the CTA prefetches a future tile with
  `cp.async.bulk.prefetch.L2.global`, commits the bulk async group, and keeps at
  most four read groups pending.
- `stream prefetch+wait`: thread 0 prefetches the current tile, commits, waits
  with `cp.async.bulk.wait_group.read 0`, then all CTA threads read the tile
  normally.

PTX generation was checked and contains:

```text
cp.async.bulk.prefetch.L2.global
cp.async.bulk.commit_group
cp.async.bulk.wait_group.read
```

Command:

```bash
make -C gpu-benches/gpu-tma-host

./gpu-benches/gpu-tma-host/cuda-tma-host-prefetch \
  --bytes 536870912 --tile-bytes 16384 \
  --prefetch-distance 2 --reps 9 --trials 512
```

## Results

Sequential runs, 512 MiB buffer, grid = 336 CTAs, block = 256 threads.

Mapped pinned host memory:

| Tile | no prefetch | prefetch ahead | prefetch + wait | timed tile cycles, no prefetch | timed tile cycles, after prefetch+wait |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 KiB | 21.99 GB/s | 17.33 GB/s | 24.64 GB/s | 1850 | 1843 |
| 16 KiB | 15.73 GB/s | 15.83 GB/s | 19.40 GB/s | 6832 | 6731 |
| 64 KiB | 11.16 GB/s | 10.75 GB/s | 13.38 GB/s | 27395 | 26989 |

Device-memory control:

| Tile | no prefetch | prefetch ahead | prefetch + wait | timed tile cycles, no prefetch | timed tile cycles, after prefetch+wait |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 KiB | 799.52 GB/s | 718.51 GB/s | 711.96 GB/s | 934 | 570 |
| 16 KiB | 800.94 GB/s | 690.14 GB/s | 739.90 GB/s | 3458 | 1882 |
| 64 KiB | 799.60 GB/s | 498.54 GB/s | 738.86 GB/s | 13557 | 7051 |

## Interpretation

For device memory, the timed-tile test shows the expected behavior: after
`cp.async.bulk.prefetch + wait_group.read 0`, the subsequent normal loads are
much faster. That means the instruction is functioning as an L2 data prefetch
for normal GPU global memory.

For mapped pinned host memory, the same timed-tile test barely improves. This is
the important result: on this 5080/platform, the instruction does not appear to
turn CPU pinned host memory into a reliable L2-hit path for later normal loads.
The `prefetch+wait` streaming mode can show higher end-to-end bandwidth, but the
timed load section does not become meaningfully cheaper, so the improvement is
more likely from scheduling/transaction shaping than from stable L2 residency.

The prefetch-ahead mode did not help host memory. It was neutral at 16 KiB and
slower at 4 KiB/64 KiB. It also slows the device-memory control because the
extra prefetch instructions compete with an already well-coalesced streaming
load.

## Practical takeaways

- `cp.async.bulk.prefetch.L2.global` is useful for predictable future accesses
  to device memory when the kernel can issue the prefetch early enough and later
  reuse the data from L2.
- For GPU reads from CPU pinned host memory, it is not a magic PCIe latency or
  bandwidth fix. The measured host path is still governed by the same roughly
  `22 GB/s` platform limit seen in the PCIe host-memory benchmarks.
- There may be a narrow use for `prefetch+wait` as a transaction-shaping tool
  before a CTA consumes a host-memory tile, but this is not free overlap: waiting
  for the prefetch makes it part of the critical path.
- The more promising use in an offloaded KV-cache kernel is still data staging:
  use bulk async copy/TMA-style movement to shared memory when the tile is reused
  by enough math to hide the host fetch. For pure streaming reads from host
  memory, prefetch alone is not enough.
