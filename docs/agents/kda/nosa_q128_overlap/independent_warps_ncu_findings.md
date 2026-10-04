# Independent-warps Q128: current-source NCU findings

Run `nosa_q128_independent_warps_ncu_gpu0_20261004_01` completed on 2026-10-04.
This is an isolated diagnostic of the rejected 64-CTA candidate, not its
promotion, API latency, a new overlap acceptance or an integrated model result.
The frozen collection plan remains [independent_warps_ncu_plan.md](independent_warps_ncu_plan.md).

## Identity and measurement scope

The full report used 46 strict application-replay invocations; SourceCounters
used 13. Together with standalone validation, all 60 GPU application invocations
performed four checked calls each: original halves1, two halves2 warmups and one
untraced halves2 target. All 240 calls passed exact/FP32, payload, queue,
library identity and cleanup checks. The separate CPU audit also passed.
Collection exited zero; both collection stderr files are empty.

GPU0/NUMA0, the original request16/layer16 Q128/H65536 operands and ownership
state, canonical register8 allocator, eight intra-op/OMP/MKL/OpenBLAS threads,
clock-control none and cache-control all were preserved. Both reports contain
one target-NVTX `fused_main` action. The parser uses direct source/SASS lookup
on absolute PCs; independent offset attribution uses the reports' function-base
metadata. Neither guesses relocation from a minimum sampled PC.
All 4,968 static instructions match each report after explicit disassembler
format normalization. Production runtime remains `02b5d0…`; no candidate was
integrated by this diagnostic.

Raw collection directory:
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_independent_warps_ncu_20261004/output/nosa_q128_independent_warps_ncu_gpu0_20261004_01/`.
Analysis directory:
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_ncu_analysis_20261004/`.

| Artifact | SHA256 |
|---|---|
| Full NCU report | `724835751d9bbbb4882cbf375ac0370f154d990aa77a80fbfe64bed35dd04c36` |
| SourceCounters report | `004fc3ffc6915798d1ba6ea9f5c3c66f17a2f6a875bbbfc734a5b3ae37311cb4` |
| Parser | `5fe94f950f09fdaba8d84b184a644c639c7779c3bf8c9f8a3c5b8396543d62da` |
| Independent combined report review | `64064e38126b992f605c6af79d57cac2d709ab585f6cc69ebc26760d9a9b34a8` |
| Final independent application ledger | `4a6de7c3282b4eb572ea7f17e6c16339af5e3e57c6a92da514f13dbc76de3985` |

## Resource, issue and utilization observations

The action is 64 blocks × 256 threads on 132 SMs, or 0.48485 waves. Compiled
register count is 240/thread; requested dynamic shared memory is 181,344 B,
with 1,024 B of separately reported driver shared memory. The theoretical
occupancy is 12.5%, achieved occupancy 11.71%. Register and shared-memory limits
each allow one CTA per SM. Existing per-CTA dynamic-register safety checks still
apply; NCU's configured stack limit is not the compiled frame size.

Only 0.077 eligible warps per scheduler per active cycle were observed; issue
activity was 7.38%. Tensor-pipe activity was 3.45% of elapsed cycles and 7.63%
of active cycles. These signals support insufficient runnable work, not a
compute-throughput ceiling. Generic NCU speedup estimates are not measured
improvements or permission to change numerical math.

SM active-cycle min/mean/max are 0 / 307,906 / 670,480. The report exposes no
per-SM instance vector for these metrics. The small grid already leaves SMs
unused, so the all-SM spread cannot by itself quantify useful-CTA imbalance.

## Exact source attribution

The full report contains 15,924 base long-scoreboard samples. Within that same
counter, the independent region attribution is:

| Region | Long-scoreboard samples | Share of this counter |
|---|---:|---:|
| Stores dependent on host K/V loads | 6,943 | 43.60% |
| Consumer K/V transaction waits | 6,725 | 42.23% |
| Historical-ready polling | 1,024 | 6.43% |
| Producer pipeline acquire | 22 | 0.138% |

The main host-dependent store PCs are function offsets `0xf7c0` and `0xf800`.
Consumer loop-K waits are larger than loop-V waits; these are transaction
barrier dependencies, not additional host loads. The second report independently
reproduces the same ordering. Writer fences, ready publication, producer
empty-buffer waits and WGMMA waits do not dominate these sampled locations.
The source report's base warp-unit counts are 6,366/644 for the two host stores,
5,482/1,148 for consumer loop-K/V waits, and 7/3 for producer K/V acquire.
These counts include their stated sampled stall families; they must not be
added to, or confused with, the long-scoreboard-only table above.

This establishes dependency locations, not additive API-time fractions.
K(next) still precedes V(current), but aggregate samples cannot prove how much
that order blocks V. The original V-first rejection is not reversed by these
observations. Another queue-fanout change is not supported as the first response
to these host-data and consumer-availability dependencies.

## Host-sector amplification

Both reports identify the same exact access inefficiency at the two host-load
PCs, `0xf750` and `0xf770`. Each executes 14,368 full-warp instructions:

| Per host K or V load PC | Observed source counter |
|---|---:|
| True thread instructions | 459,776 |
| Ideal 32-byte sectors | 229,888 |
| Theoretical sectors | 258,624 |
| Sectors per full warp | 18 versus 16 ideal |

The two loads sum to 517,248 theoretical sectors. This exactly matches the
full report's separate TEX system-memory read-miss counter, or 16,551,936 B.
SourceCounters reproduces the per-PC sector values but has no TEX counter.
Logical K+V payload remains 14,712,832 B, corresponding to
459,776 sectors. The ratio is exactly 1.125. Both dependent HBM stores are
16 sectors per warp, with zero theoretical excess. Thus the amplification is
localized to host loads, rather than an unexplained aggregate HBM counter.

The access pattern is consistent with a host base aligned to 16 B but not 32 B:
each half warp reads a contiguous 256-byte token, and a 16-byte offset would
span nine sectors instead of eight. The profiled target's host addresses were
not retained, so this remains a specific explanation to verify. An allocation
probe and a controlled aligned-buffer diagnostic can test it without changing
the frozen CUDA binary or selection.

HBM DRAM throughput was only 0.11% of peak in this report. That is not a host
link utilization metric. The observed system-memory sectors do not establish
a calibrated PCIe ceiling, and the whole intrusive NCU duration must not become
an uninstrumented API latency. Local-memory accesses also exist: 7,196 executed
LDL and 11,044 STL instructions, with local loads largely served by L1. NCU's
local-spill recommendation does not establish spills as the dominant cause.

## Timeline and extraction limits

The full report has 158,879 timed-warp samples from one sampling pass. Long
scoreboard accounts for 128,077 of those samples and remains dominant through
most of that pass; the sample count drops near its end. The plot and raw bins
are `current_independent_full_01/timed_sample_profile.{png,svg}` and
`timed_sample_summary.json` in the analysis directory. Percentages are sample
shares, not elapsed-time or copy/compute-overlap fractions.

The matching workload window is 382.144 us. The first 192 samples precede its
start marker by 128 ns; all were retained. Timed PCs use another application-
replay code base and direct source lookups fail for them. No relocation is
inferred, so the timeline is not assigned fetch/consumer roles. PM series and
screen `%globaltimer` intervals retain their separate clocks.

The installed NCU Python binding has a misleading `has_value(arg)` docstring:
the argument is a value-kind enum, not an instance ordinal. The reviewed parser
uses `value(i)`, preserves missing values, separates base from conditioned
`*_not_issued` counters, and records raw unit snapshots. Two sampling unit labels
vary with lookup order; counts are preserved and no mixed-unit sum is treated
as physical time. Details and independent API checks are in `PARSER_NOTES.md`
and the sibling NCU review directory.

## Next bounded decision

The follow-up GPU4 allocation-only probe reproduced the proposed address
residue: three fresh K/V tensor pairs all had data/storage pointers 16 modulo32,
with exact shape, stride and pinning. The pairs reused two32 MiB blocks and all
tensor weakrefs died after release. This supports the allocation mechanism but
does not recover the old profiled target's pointers. Raw host active counters
grew across this reuse while allocator-owned bytes stayed64 MiB, so they cannot
be treated as physical live capacity. Probe identity and the revised lifetime
checks are in [aligned_host_diagnostic_plan.md](aligned_host_diagnostic_plan.md).

The [controlled buffer comparison](aligned_host_diagnostic_findings.md) now
confirms18→16 host sectors per warp with unchanged exact output, logical payload,
unique-read and cleanup checks. The next screen tests internal overlap.
Padding changes pinned allocator bucket
capacity; a diagnostic allocation is not automatically a budget-safe runtime
design. Do not promote an aligned harness as a production cache change.

The independent-warp loop still serializes successive host-load/store pairs.
If alignment alone does not resolve the screen, a separately planned change
can increase per-warp load-level parallelism within the verified CTA register
pool. That remains a subsequent hypothesis, not an authorized implementation
or measured gain. Every candidate still requires the original numerical,
unique-read, every-sample 90%, full-model and complete-API promotion gates.
