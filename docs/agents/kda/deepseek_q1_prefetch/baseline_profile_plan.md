# Current FIFO preparation baseline profile

Objective: explain the current H64K/A1 ECHO preparation cost before writing
another candidate. Root assigned GPU0 and CPUs 0-7. Production/native sources
are frozen; this task adds only a baseline experiment harness and analysis.

## Draft

Read the exact initial priority and clock from the published CUB check's
`echo_default_graph_prefetch_evidence.pt`. Verify that all three layer inputs
are byte-identical, H=65,536, P=65,600, single-session cold, and clock=256.
Call the current `cache_ops.prepare_prefetch` with those tensors, query_count=1
and the actual host capacity. Allocate the original scratch shapes. This does
not allocate or execute a model and does not invoke any rejected candidate.

Correctness checks the full stable FIFO list, complete journal/counter/stats
reset, one-use request metadata and unchanged priorities. Graph replay must
repeat the same operation safely after output buffers are dirtied. Save the
input and output tensors, source snapshot, actual loaded DSO hashes and runtime
identity. A separate profile process must require this exact check receipt.

NCU uses direct complete preparation calls, following twenty warmups outside
ProfilerAPI collection. Collect every kernel in one complete call, first full
plus PM sampling, then an independent source/SourceCounters run. Kernel replay,
cache flush and base-clock control are explicit. These kernel durations exclude
the memsets, CPU dispatch and launch gaps visible in the full NSYS stage.
Do not report their sum as complete API latency.

## Executable plan

1. Add `experiments/deepseek_v32_mfu/src/q1_prepare_baseline.py` with independent
   `check` and `profile` modes and fresh output directories. No computation
   implementation is copied into the harness.
2. Run its check on GPU0/CPUs0-7 with the saved current check input. Hash original
   sources before/after, archive them, and identify the actually loaded native
   image from the process mappings. Require immutable hashes through the run.
3. Run NCU twice with `--profile-from-start off --replay-mode kernel
   --cache-control all --clock-control base`, `--set full --section PmSampling
   --section PmSampling_WarpStates` or explicit `--section LaunchStats
   --section SourceCounters`. Installed NCU 2026.1.1 has no named `source` set;
   its section catalog was checked before collection.
   The harness emits one ProfilerAPI-scoped call with an NVTX name and no reset
   kernels inside it. Use fresh data/log/profile run IDs.
4. Parse every action with `ncu_report`; archive all metrics, PM samples, rule
   results and source-correlated stalls. Verify the expected six kernels:
   preparation, histogram, exclusive sum, two radix OneSweep passes and final
   slot/journal preparation. Report any discrepancy rather than guessing.
5. Send root report paths, source/native/input hashes, measured bottlenecks and
   the bounded-free proof. No candidate implementation until diagnosis and
   root review are complete.

Project experiment layout overrides the NCU skill's generic root `profile/`
layout. Checks go under `/tmp/cxldsagr-checks/`; archived run data, raw reports
and logs go to this experiment's `output/{data,profile,log}/` directories.

## Completed baseline evidence

Check `q1_prepare_baseline_check_20261008_01` passed one eager call and four
graph replays after dirtying output buffers. All three saved layer inputs are
identical; the entire stable FIFO list, journal reset, counter/statistics,
request metadata and unchanged priority bytes passed.

Independent NCU full and source collections completed as
`q1_prepare_baseline_ncu_20261008_01`. Raw reports are
`experiments/deepseek_v32_mfu/output/profile/q1_prepare_baseline_ncu_20261008_01/{full,source}.ncu-rep`.
All six expected kernels are present in both reports. Analysis through
`ncu_report` is in the matching `output/data/.../analysis/` directory, including
all metrics, PM samples, rules and per-source stalls. `full_details.txt`
retains the readable rule report. `acceptance.json` verifies matching
check/full/source identities and binds the archived original input and DSOs.

The 2,730 source/header identities match across all three processes. Original
input SHA256 is `23b73bbce9cb3288b8f5eb5dff9f3051cce745a82d82bb87d02a7b34a155e66b`.
The actually mapped ECHO DSO is `cxldsagr_echo_indexer_9407651aba592b4a.so`,
SHA256 `3a207197a196d5b3db02f602b156c7bbc577ab3fc47c6912b299735f02f86693`.
Copies of the DSO and source tree are archived with each process's record.

| Kernel | Full NCU duration, us | Grid | Block | SM throughput, % | DRAM read throughput, % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Key preparation | 3.616 | 128 | 256 | 2.143 | 3.065 |
| Radix histogram | 4.128 | 1,188 | 128 | 28.719 | 2.713 |
| Histogram exclusive sum | 2.432 | 2 | 256 | 0.053 | 0.053 |
| Radix OneSweep pass 1 | 8.352 | 12 | 384 | 0.821 | 1.386 |
| Radix OneSweep pass 2 | 8.224 | 12 | 384 | 0.811 | 1.413 |
| Slot/journal preparation | 3.296 | 128 | 256 | 3.021 | 3.347 |

The OneSweep passes use 0.04545 waves per SM and only 0.240/0.221 eligible
warps per scheduler cycle. They dominate the measured kernels while DRAM
read throughput remains below 1.5% of peak. This supports testing whether the
proved free-slot case can avoid sorting; it does not support a bandwidth
saturation diagnosis. Do not add these profiler durations to the old NSYS
memsets or gaps, or use their sum as complete API latency.

Source sampling attributes 74 long-scoreboard samples in key preparation to
the loaded priority check at `echo_sparse_recall.cuh:183`, and 76 in slot
preparation to consuming the sorted key at line 212. OneSweep's leading
barrier/radix-rank locations have only about 5-6 samples each, insufficient
for a precise per-line time attribution. The `*_not_issued` metrics overlap
their base metrics and must not be added to them.

Collector warnings are retained: six `ctc__*` metrics are unavailable on this
device, and the default PM interval is coarse relative to these short kernels.
Neither warning invalidates the captured launch geometry or the cited
available counters. No candidate implementation or performance claim follows
from this baseline-only run; root review is the next gate.
