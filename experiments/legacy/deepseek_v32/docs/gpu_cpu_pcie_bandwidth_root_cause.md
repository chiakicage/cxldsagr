# GPU/CPU PCIe bandwidth root cause notes

Date: 2026-07-05

Device:

```text
GPU: NVIDIA GeForce RTX 5080
PCIe link during benchmark: Gen5 x16
CPU: Intel Core i5-12600K
Memory: DDR4-3600 dual channel, 4 x 32 GB
IOMMU: enabled, Intel DMAR present
THP mode: madvise
```

## Correction to the earlier hypothesis

The earlier explanation that GPU-host bandwidth was mainly limited by CPU DRAM
bandwidth was too coarse.

CPU STREAM-like testing shows this host can sustain about `45-46 GB/s` DRAM
traffic. Therefore `cudaMemcpy` H2D/D2H around `20-22 GB/s` with
`cudaHostAlloc` cannot be explained by CPU DRAM bandwidth alone.

## Key experiment

The benchmark in `gpu-benches/gpu-pcie-host/` was extended to compare:

- `cudaHostAlloc(..., cudaHostAllocMapped)`
- ordinary 4 KiB-backed memory + `cudaHostRegisterMapped`
- 2 MiB-aligned anonymous memory + `MADV_HUGEPAGE` + `cudaHostRegisterMapped`

Command:

```bash
make -C gpu-benches/gpu-pcie-host cuda-pcie-host

./gpu-benches/gpu-pcie-host/cuda-pcie-host \
  --bytes 1073741824 --reps 5
```

The benchmark also prints `/proc/self/smaps` information for the registered
host buffers. With ordinary registered memory:

```text
KernelPageSize:        4 kB
MMUPageSize:           4 kB
AnonHugePages:         0 kB
```

With `MADV_HUGEPAGE`:

```text
KernelPageSize:        4 kB
MMUPageSize:           4 kB
AnonHugePages:   1048576 kB
VmFlags: ... hg
```

## Results

1 GiB buffer, RTX 5080, link reported as `Gen5 x16` during the benchmark.

| Host buffer | mapped read | mapped write | H2D | D2H | H2D x4 | D2H x4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cudaHostAllocMapped` | 21.71 GB/s | 15.27 GB/s | 21.30 GB/s | 20.49 GB/s | 20.68 GB/s | 20.52 GB/s |
| 4 KiB memory + `cudaHostRegisterMapped` | 20.88 GB/s | 15.50 GB/s | 20.62 GB/s | 20.48 GB/s | 19.77 GB/s | 20.28 GB/s |
| THP-backed memory + `cudaHostRegisterMapped` | 36.14 GB/s | 17.28 GB/s | 43.99 GB/s | 53.20 GB/s | 45.88 GB/s | 53.30 GB/s |

Multi-stream copies do not help the 4 KiB-backed path. They do slightly improve
H2D on the THP-backed path, from about `44 GB/s` to about `46 GB/s`.

## Interpretation

The slow `~20-22 GB/s` result is not the PCIe Gen5 x16 link limit, and it is not
the CPU DRAM bandwidth limit. It is specific to the default 4 KiB-page pinned
host-memory path.

The most likely bottleneck is page-granularity overhead in the DMA/IOMMU path:

- The system has Intel DMAR/IOMMU enabled.
- The GPU is in its own IOMMU group with the audio function.
- The NVIDIA endpoint capability dump does not show ATS support.
- A 1 GiB 4 KiB-page buffer has 262,144 pages/IOMMU mappings.
- A 1 GiB THP-backed buffer has only 512 2 MiB huge pages.
- Switching only the host buffer backing from 4 KiB pages to THP-backed memory
  changes H2D from `~21 GB/s` to `~44-46 GB/s`, and D2H from `~20 GB/s` to
  `~53 GB/s`.

This is strong evidence that the original path was limited by DMA mapping /
IOTLB / scatter-gather overhead for 4 KiB pinned pages. Hugepage-backed
registered memory greatly reduces that overhead.

## Practical conclusion

For high-bandwidth GPU/CPU transfers on this machine, do not use plain
`cudaHostAlloc` as the final measurement path. Use hugepage-backed host memory
and register it:

```cpp
void* ptr = nullptr;
posix_memalign(&ptr, 2 * 1024 * 1024, bytes);
madvise(ptr, bytes, MADV_HUGEPAGE);
touch_all_pages(ptr, bytes);
cudaHostRegister(ptr, bytes, cudaHostRegisterMapped);
```

Then use either `cudaMemcpyAsync` or `cudaHostGetDevicePointer` for mapped
access.

Expected bandwidth on this system:

- `cudaMemcpy` H2D from THP-registered host memory: `~44-46 GB/s`
- `cudaMemcpy` D2H to THP-registered host memory: `~53 GB/s`
- GPU zero-copy mapped read from THP-registered host memory: `~36 GB/s`
- GPU zero-copy mapped write to THP-registered host memory: still low,
  `~17 GB/s`

The direct-load result was validated and optimized with a dedicated benchmark in
`gpu-benches/gpu-host-direct-ld/`. C++ `uint4` loads and explicit inline PTX
`ld.global.v4.u32` loads perform the same: about `21-23 GB/s` from default
4 KiB pinned memory and about `36-37 GB/s` from THP-backed registered memory.
Changing the read shape to one `uint2` per thread improves the THP-backed path
to about `39.4 GB/s`. `cp.async.bulk.prefetch`, `cp.async.bulk` global-to-shared,
ordinary `cp.async` global-to-shared, per-thread ILP 2/4/8, and MRRS 512B/1024B
were tested and did not beat the `uint2` direct load. See
`docs/gpu_host_direct_ld_results.md`.

So the revised answer is: the original `~20 GB/s` was a default pinned-memory
page-size/IOMMU-path artifact. With THP-backed registered memory, GPU/CPU
`cudaMemcpy` bandwidth is much closer to the practical limits of this DDR4-3600
dual-channel + PCIe5 x16 platform.
