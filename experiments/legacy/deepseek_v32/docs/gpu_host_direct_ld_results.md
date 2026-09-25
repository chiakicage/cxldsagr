# GPU direct load from CPU pinned host memory

Date: 2026-07-05

Device:

```text
GPU: NVIDIA GeForce RTX 5080
PCIe link during benchmark: Gen5 x16
CPU host memory: DDR4-3600 dual channel
THP mode: madvise
```

## Benchmark

Added a focused direct-load benchmark:

```text
gpu-benches/gpu-host-direct-ld/
```

Build and run:

```bash
make -C gpu-benches/gpu-host-direct-ld

./gpu-benches/gpu-host-direct-ld/cuda-host-direct-ld \
  --bytes 1073741824 --reps 7
```

The benchmark compares:

- `cudaHostAlloc(..., cudaHostAllocMapped)`
- THP-backed anonymous memory registered with `cudaHostRegisterMapped`
- C++ `uint4` loads
- explicit inline PTX `ld.global.v4.u32` loads

It also prints `/proc/self/smaps` for the THP-backed registered buffer. In the
successful run the 1 GiB buffer showed:

```text
AnonHugePages:   1048576 kB
VmFlags: ... hg
```

## Main Result

Initial 1 GiB buffer result, grid = 8192, block = 256:

| Host buffer | C++ `uint4` load | PTX `ld.global.v4.u32` |
| --- | ---: | ---: |
| `cudaHostAllocMapped`, 4 KiB pages | 21.91 GB/s | 21.66 GB/s |
| THP-backed `cudaHostRegisterMapped` | 36.10 GB/s | 36.11 GB/s |

The C++ load and explicit PTX load are effectively the same, so the normal C++
mapped read kernel is already measuring direct GPU `ld.global` from CPU host
memory.

## Optimization Attempts

The best direct-read configuration found so far is:

```bash
./gpu-benches/gpu-host-direct-ld/cuda-host-direct-ld \
  --bytes 1073741824 --reps 5 --block 64 --grid 16384
```

Use THP-backed `cudaHostRegisterMapped` and a `uint2` load shape. This makes
each warp issue a 256B contiguous request rather than the 512B request shape of
`uint4`.

Best 1 GiB result:

| Kernel/read shape | THP-backed host read bandwidth |
| --- | ---: |
| C++ `uint4` load | 37.35 GB/s |
| C++ `uint2` load | 39.43 GB/s |
| C++ `uint2` load, ILP 2 | 35.02 GB/s |
| C++ `uint2` load, ILP 4 | 15.02 GB/s |
| C++ `uint2` load, ILP 8 | 13.23 GB/s |
| C++ `uint32_t` load | 25.41 GB/s |
| C++ `ulonglong4` load | 17.50 GB/s |
| `cp.async.bulk.prefetch` + wait | 13.59 GB/s |
| `cp.async.bulk.prefetch` ahead | 13.35 GB/s |
| `cp.async.bulk` global-to-shared, 4 KiB tile | 20.65 GB/s |
| `cp.async.bulk` global-to-shared, 16 KiB tile | 34.00 GB/s |
| `cp.async.bulk` global-to-shared, 64 KiB tile | 33.93 GB/s |
| `cp.async` 4B global-to-shared, 16 KiB tile | 31.97 GB/s |
| `cp.async` 8B global-to-shared, 16 KiB tile | 33.02 GB/s |
| `cp.async` 16B global-to-shared, 16 KiB tile | 27.91 GB/s |
| `cp.async` 4B global-to-shared, 64 KiB tile | 33.75 GB/s |
| `cp.async` 8B global-to-shared, 64 KiB tile | 33.85 GB/s |
| `cp.async` 16B global-to-shared, 64 KiB tile | 33.86 GB/s |

Changing the GPU endpoint PCIe Max Read Request Size from 256B to 512B and
1024B did not improve the best `uint2` direct-read result. The setting was
restored to 256B after the experiment.

MRRS sweep for the best direct-read shape, 1 GiB buffer, block = 64, grid =
16384:

| GPU endpoint MRRS | THP `uint2` direct-read bandwidth |
| ---: | ---: |
| 256B | 39.37 GB/s |
| 512B | 39.41 GB/s |
| 1024B | 39.47 GB/s |

The differences are within run-to-run noise. The original DevCtl value
`0x193f` was restored after the sweep.

## Grid Sweep

Best `uint4` load bandwidth over grid size:

| Grid CTAs | 4 KiB `cudaHostAllocMapped` | THP-backed registered |
| ---: | ---: | ---: |
| 512 | 22.68 GB/s | 35.58 GB/s |
| 1024 | 22.12 GB/s | 35.57 GB/s |
| 2048 | 21.83 GB/s | 35.94 GB/s |
| 4096 | 22.00 GB/s | 36.19 GB/s |
| 8192 | 21.58 GB/s | 36.13 GB/s |
| 16384 | 21.52 GB/s | 36.16 GB/s |

The result is not very sensitive to CTA count once enough concurrency is
present. The 4 KiB pinned path saturates around `21-23 GB/s`; the THP-backed
registered path saturates around `36 GB/s`.

## Notes

Per-thread ILP did not help. Even with independent accumulators and
warp-contiguous access, `uint2` ILP 2/4/8 was slower than the simple one-load
loop. For this host-mapped path, the best tested direct-load shape is one
`uint2` per thread per loop iteration.

## Interpretation

THP-backed registered memory helps GPU direct loads from CPU pinned host memory,
but it does not reach the `cudaMemcpy` copy-engine numbers:

- Direct GPU `ld.global` from THP-backed host memory, optimized: `~39.4 GB/s`
- `cudaMemcpy` H2D from THP-backed host memory: `~44-46 GB/s`
- `cudaMemcpy` D2H to THP-backed host memory: `~53 GB/s`

So the copy engine path is still better for bulk movement. The best direct-read
path reached about `86-90%` of THP-backed H2D `cudaMemcpy`, depending on which
copy result is used as the baseline. `cp.async.bulk.prefetch`, `cp.async.bulk`,
and ordinary `cp.async` did not close the gap on this platform.

The likely reason is that SM-issued host reads and copy-engine DMA are not the
same hardware path. The copy engine can generate larger/more efficient DMA
traffic, while SM `ld.global` is constrained by warp load coalescing, outstanding
request handling, and PCIe read completion behavior. For dense streaming, use
`cudaMemcpy`/staging if possible; for sparse/on-demand host access, the best
tested direct-read path is THP-backed registered memory plus one 8B-per-thread
coalesced load, with enough CTAs to keep the SM-issued read path full.


## Cross-check against `gfd_gpu_pcie_read_benchmark`

A later retest in `gfd/examples/10_gpu_pcie_read_benchmark.cu` reported higher
THP-backed `cp.async` numbers, about `38-40 GB/s` for the 128 MiB 64 KiB-token
configuration. That is not a contradiction; the two benchmarks are not measuring
the same kernel shape.

Key differences:

- `gfd_gpu_pcie_read_benchmark` splits all transfers into 4 KiB CTAs. For a
  128 MiB / 64 KiB-token run, it launches 32768 CTAs, each CTA issuing one 16B
  `cp.async` per thread.
- The original `gpu-host-direct-ld` ordinary `cp.async` rows used 16 KiB or
  64 KiB tiles. Each CTA loops over many `cp.async` operations per thread before
  commit/wait, which is a different stress pattern.
- `gpu-host-direct-ld` writes one 64-bit `partial` value per thread to global
  memory. At `grid=32768, block=256`, that is 64 MiB of extra device writes that
  are not counted in the reported host-read bandwidth. The GFD shared-only path
  writes only one 32-bit sink value per 4 KiB CTA.
- A temporary `cp.async.ca` vs `cp.async.cg` control in the GFD benchmark showed
  no material difference; cache modifier is not the explanation.

Targeted control, 128 MiB THP-backed registered host memory, `grid=8192`,
`block=256`:

| Kernel | Median bandwidth |
| --- | ---: |
| `bulk 4KiB` | 36.09 GB/s |
| `cp.async16 4K` | 36.17 GB/s |
| `cp.async16 16K` | 34.14 GB/s |
| `cp.async16 64K` | 33.51 GB/s |

With `grid=32768`, `block=256` so the 4 KiB control also has one CTA per 4 KiB
tile:

| Kernel | Median bandwidth |
| --- | ---: |
| `bulk 4KiB` | 35.19 GB/s |
| `cp.async16 4K` | 35.94 GB/s |
| `cp.async16 16K` | 34.12 GB/s |
| `cp.async16 64K` | 33.41 GB/s |

The remaining gap to the GFD shared-only number is mainly benchmark work: the
host-direct-load benchmark reduces into per-thread global `partial` output,
while the GFD shared-only benchmark reduces per CTA and writes one small sink.
For comparing host-read capability, prefer matching tile size, output work, and
counted bytes before drawing conclusions across the two benchmark families.
