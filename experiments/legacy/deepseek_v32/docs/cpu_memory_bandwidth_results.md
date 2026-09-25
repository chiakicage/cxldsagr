# CPU memory bandwidth notes

Date: 2026-07-05

System:

```text
CPU: Intel Core i5-12600K
NUMA nodes: 1
Memory: 4 x 32 GB KingBank DDR4 DIMM
Configured memory speed: 3600 MT/s
Ranks: 2 per DIMM
Topology: dual-channel desktop platform, 2 DIMMs per channel
```

Theoretical raw bandwidth for DDR4-3600 dual channel:

```text
3600 MT/s * 8 bytes/channel * 2 channels = 57.6 GB/s
57.6 GB/s = 53.6 GiB/s
```

## Benchmark

Added a STREAM-like CPU benchmark:

```text
gpu-benches/gpu-pcie-host/cpu_mem_stream.cpp
gpu-benches/gpu-pcie-host/cpu-mem-stream
```

It uses persistent worker threads, explicit CPU affinity, and AVX2 streaming
stores for non-temporal write/copy cases. `copy` reports both application
bandwidth and estimated DRAM traffic bandwidth.

The CPU governor was temporarily changed from `powersave` to `performance` for
the measurements and restored to `powersave` after the run.

Build:

```bash
make -C gpu-benches/gpu-pcie-host cpu-mem-stream
```

Best stable command:

```bash
./gpu-benches/gpu-pcie-host/cpu-mem-stream \
  --bytes 4294967296 --reps 5 --cpus 0-15 \
  --ops read,copy,nt-write,nt-copy
```

Result:

| Path | App bandwidth | DRAM traffic bandwidth |
| --- | ---: | ---: |
| read | 45.88 GB/s | 45.88 GB/s |
| copy | 22.48 GB/s | 44.95 GB/s |
| non-temporal write | 46.30 GB/s | 46.30 GB/s |
| non-temporal copy | 22.70 GB/s | 45.40 GB/s |

With a 1 GiB buffer and all CPUs:

| Path | App bandwidth | DRAM traffic bandwidth |
| --- | ---: | ---: |
| read | 45.76 GB/s | 45.76 GB/s |
| copy | 22.36 GB/s | 44.72 GB/s |
| non-temporal write | 46.11 GB/s | 46.11 GB/s |

Using only the first hardware thread of the six P-cores
(`--cpus 0,2,4,6,8,10`) was very similar:

| Path | App bandwidth | DRAM traffic bandwidth |
| --- | ---: | ---: |
| read | 44.65 GB/s | 44.65 GB/s |
| copy | 22.04 GB/s | 44.08 GB/s |
| non-temporal write | 47.70 GB/s | 47.70 GB/s |

Adding P-core hyperthreads and/or E-cores did not materially improve sustained
bandwidth.

## IMC counter cross-check

The uncore IMC free-running counters agree with the benchmark traffic shape.
For example, 20 timed reps plus 3 warmups of a 1 GiB read produced mostly IMC
read traffic:

```text
25248.58 MiB  uncore_imc_free_running/data_read/
 2214.83 MiB  uncore_imc_free_running/data_write/
27463.68 MiB  uncore_imc_free_running/data_total/
```

For non-temporal write, traffic flips to mostly IMC write:

```text
 1574.34 MiB  uncore_imc_free_running/data_read/
25749.01 MiB  uncore_imc_free_running/data_write/
27323.23 MiB  uncore_imc_free_running/data_total/
```

For copy, IMC read and write traffic are both present:

```text
25271.55 MiB  uncore_imc_free_running/data_read/
25755.80 MiB  uncore_imc_free_running/data_write/
51027.60 MiB  uncore_imc_free_running/data_total/
```

## Conclusion

The sustainable CPU DRAM bandwidth on this machine is about `45-46 GB/s`, or
roughly `78-80%` of the DDR4-3600 dual-channel raw theoretical peak of
`57.6 GB/s`.

This appears to be the real platform limit, not a benchmark artifact:

- The result is stable from 1 GiB to 4 GiB buffers.
- Pinning to P-cores, adding hyperthreads, adding E-cores, and switching the CPU
  governor to `performance` do not move the peak much.
- IMC counters show the expected read/write traffic and confirm that the memory
  controller is seeing traffic in the same range.

The gap to `57.6 GB/s` is expected for a real DDR4 desktop setup. The headline
number is a raw transfer-rate ceiling; sustained STREAM-like traffic loses
efficiency to DRAM commands, refresh, row policy, read/write turnarounds,
controller scheduling, and the fact that this system has four dual-rank DIMMs
installed as two DIMMs per channel.

For PCIe/GPU-host-memory experiments, the relevant number is therefore not the
raw `57.6 GB/s`, but the measured sustainable host DRAM traffic of about
`45-46 GB/s`. A CPU `memcpy` application rate of `~22.5 GB/s` corresponds to
about `45 GB/s` of DRAM traffic because every copied byte is read and written.
