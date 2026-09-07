# RTX 5080 PCIe host-memory bandwidth

Date: 2026-07-05

Device:

```text
NVIDIA GeForce RTX 5080
SM 120
SM count: 84
Async copy engines: 2
PCI bus id: 00000000:01:00.0
PCIe link during benchmark: current Gen5 x16, max Gen5 x16
```

Idle `nvidia-smi` may report `Gen1 x16`; the benchmark process uses NVML and observed the link upclocking to `Gen5 x16` while running.

System observations:

```text
CPU: Intel Core i5-12600K
NUMA nodes: 1
GPU topology affinity: node 0 / CPUs 0-15
CPU governor: powersave
Resizable BAR / BAR1: 16 GiB
GPU IOMMU group: GPU function plus audio function only
Kernel cmdline: no explicit intel_iommu=on/off option
```

## Results

Best observed bandwidths:

| Path | Best observed |
| --- | ---: |
| GPU direct read from mapped pinned host memory | 21.87 GB/s |
| GPU direct write to mapped pinned host memory | 16.48 GB/s |
| `cudaMemcpyAsync` host-to-device | 22.31 GB/s |
| `cudaMemcpyAsync` device-to-host | 21.57 GB/s |
| `cudaMemcpyAsync` bidirectional H2D+D2H aggregate | 22.74 GB/s |

Detailed runs:

| Run | Host allocation | mapped read | mapped write | memcpy H2D | memcpy D2H | bidir H2D+D2H aggregate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 512 MiB | default pinned | 21.60 GB/s | 15.88 GB/s | 22.40 GB/s | 20.55 GB/s | 22.67 GB/s |
| 512 MiB | write-combined pinned | 21.29 GB/s | 16.21 GB/s | 22.04 GB/s | 21.13 GB/s | 22.73 GB/s |
| 2 GiB | default pinned | 21.46 GB/s | 14.98 GB/s | 20.99 GB/s | 20.46 GB/s | 22.45 GB/s |
| 2 GiB | write-combined pinned | 21.29 GB/s | 14.85 GB/s | 21.62 GB/s | 20.36 GB/s | 22.50 GB/s |

CPU memory bandwidth cross-check:

| Threads | CPU read | CPU write | CPU copy |
| ---: | ---: | ---: | ---: |
| 1 | 33.49 GB/s | 19.30 GB/s | 23.09 GB/s |
| 6 | 34.19 GB/s | 27.61 GB/s | 19.94 GB/s |
| 16 | 23.81 GB/s | 23.12 GB/s | 14.79 GB/s |

Follow-up STREAM-like CPU memory testing with persistent pinned threads measured
about `45-46 GB/s` of sustainable DRAM traffic on this DDR4-3600 dual-channel
system. CPU copy application bandwidth is about `22.5 GB/s`, which corresponds
to about `45 GB/s` of DRAM traffic because each byte is both read and written.
See `docs/cpu_memory_bandwidth_results.md`.

Further follow-up showed that the original `~20-22 GB/s` GPU-host numbers were
not primarily a CPU DRAM bandwidth limit. They were specific to the default
4 KiB-page pinned host-memory path. THP-backed anonymous memory registered with
`cudaHostRegisterMapped` reached `~44-46 GB/s` H2D and `~53 GB/s` D2H. See
`docs/gpu_cpu_pcie_bandwidth_root_cause.md`.

## Commands

```bash
make -C gpu-benches/gpu-pcie-host

./gpu-benches/gpu-pcie-host/cuda-pcie-host \
  --bytes 536870912 --reps 15 \
  > gpu-benches/gpu-pcie-host/rtx5080.txt

./gpu-benches/gpu-pcie-host/cuda-pcie-host \
  --bytes 1073741824 --reps 10 \
  > gpu-benches/gpu-pcie-host/rtx5080_1g.txt

./gpu-benches/gpu-pcie-host/cuda-pcie-host \
  --bytes 1073741824 --reps 10 --grid 16384 \
  > gpu-benches/gpu-pcie-host/rtx5080_1g_grid16384.txt

./gpu-benches/gpu-pcie-host/cuda-pcie-host \
  --bytes 536870912 --reps 10 \
  > gpu-benches/gpu-pcie-host/rtx5080_wc_bidir.txt

./gpu-benches/gpu-pcie-host/cuda-pcie-host \
  --bytes 2147483648 --reps 6 \
  > gpu-benches/gpu-pcie-host/rtx5080_2g_wc_bidir.txt

make -C gpu-benches/gpu-pcie-host cpu-mem-bw

./gpu-benches/gpu-pcie-host/cpu-mem-bw \
  --bytes 1073741824 --reps 5 --threads 1 \
  > gpu-benches/gpu-pcie-host/cpu_mem_1t.txt
```

The benchmark code is in `gpu-benches/gpu-pcie-host/`.

## Revised Diagnosis

- The measured one-direction path using `cudaHostAlloc` is about `20-22 GB/s`,
  despite the link capability/current state being `Gen5 x16`.
- This low number is not explained by CPU DRAM bandwidth alone: CPU STREAM-like
  traffic reaches `45-46 GB/s`.
- It is also not fixed by write-combined `cudaHostAlloc` or by multiple CUDA
  streams.
- The key difference is page backing. Plain `cudaHostAlloc` and ordinary
  4 KiB-backed `cudaHostRegisterMapped` both stay around `20-22 GB/s`.
  THP-backed memory plus `cudaHostRegisterMapped` reaches `~44-46 GB/s` H2D and
  `~53 GB/s` D2H.
- The likely bottleneck is DMA/IOMMU/scatter-gather overhead for 4 KiB pinned
  pages. The system has Intel DMAR/IOMMU enabled, and hugepage-backed registered
  memory greatly reduces the number of mappings for the same transfer size.

Practical recommendation:

1. For high-bandwidth H2D/D2H benchmarking, allocate 2 MiB-aligned anonymous
   memory, call `madvise(..., MADV_HUGEPAGE)`, touch the pages, then use
   `cudaHostRegisterMapped`.
2. Treat plain `cudaHostAlloc` numbers as measuring the default 4 KiB pinned
   page path, not the PCIe5 x16 link ceiling.
3. BIOS/firmware checks are still useful, but the current evidence points first
   to host pinned-memory page granularity / IOMMU behavior.

Recommended root/BIOS checks:

```bash
sudo cpupower frequency-set -g performance
sudo dmidecode -t memory
```

In BIOS, verify:

- Memory is dual-channel and running at expected XMP/EXPO speed.
- PCIe slot is set to Gen5, not Auto/compatibility mode.
- Above 4G decoding and Resizable BAR are enabled.
- ASPM/power-saving PCIe options are disabled for performance testing.
- Try toggling VT-d/IOMMU for comparison if the system is dedicated to benchmarking.
