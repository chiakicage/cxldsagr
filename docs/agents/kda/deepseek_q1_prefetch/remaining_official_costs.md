# Remaining ECHO costs with the official kernel

Read-only diagnosis of `deepseek_h64k_a1_free_20261008_01`; no new GPU work or
production changes. The raw source is the current decode-gap report and its
four retained SQLite captures. The independent CPU review is
`/tmp/cxldsagr-checks/q1_echo_remaining_gap_independent.json` (SHA256
`11f731fa497c039273039d6ee01ec8ea79fc02b69972142d28437fcede2ad7eb`).
Archived scripts, results, this note and consulted source snapshots are indexed
in `experiments/deepseek_v32_echo_official/output/data/q1_echo_remaining_costs_20261008_01/index.json`.

The fused kernel itself takes 8.448/8.800/8.416 us in official SGLang and
10.752/23.680/48.704 us locally: sums 25.664/83.136 us, a 57.472 us difference.
Both traces launch the same named template with 132 CTAs and 768 threads per
CTA. These facts do not establish equal runtime binaries, clock behavior or
workload. Local main KV is cold; official residency is natural and only its
post-request snapshot was saved. Inputs, physical GPUs and frameworks differ.

## Completed transfers and reservation work differ

Six saved local check evidences contain 18 positive-zero FP32 `initial_hint[1]`
tensors. Their file and tensor hashes were independently checked on CPU in
`/tmp/cxldsagr-checks/q1_echo_hint_independent.json` (SHA256
`5bd63818e79b8c338a94054534984ecfaae2fa7c04a4f5b5f6c50a0a7dc2f142`). All 15 clean
formal layer samples complete 64 prefetches, but their reservation counters
are 536–621 for L0, 3,097 for L1 and 33,471 for L2. These are record counters,
not counts of atomic instructions. The official launch threshold was not
saved; its source-level EMA update does not establish that launch's value.

The independent component run `q1_official_prefetch_bench_20261008_02` changes
the map while retaining each layer's input and zero threshold. Its prepared
API contains the official fused kernel and causal clean; its full API also
packs K/scales and builds block-table/schedule metadata. Staging reset is
outside timing. Neither boundary includes promotion, exact top-k or recall.

| Layer | Cold prepared us | Resident prepared us | Cold full us | Resident full us | Cold graph reservation counters |
| --- | ---: | ---: | ---: | ---: | --- |
| L0 | 16.656 | 12.640 | 21.712 | 18.544 | 501–623 |
| L1 | 35.712 | 12.784 | 42.288 | 18.672 | 3,097 |
| L2 | 50.480 | 12.688 | 56.528 | 18.704 | 33,464 |

All cold component calls copy 64 records / 73,728 B; all resident-map calls
reserve and copy zero. The resident case does not consume a validated warm
device KV pool. Its nearly constant latency and zero reservation count show
why 64 completed transfers alone cannot describe the cold work. The component
L2 counter 33,464 is distinct from the formal run's 33,471. Its medians cannot
be subtracted from the formal kernel-only profile to estimate pure IO time.

The official prefetch warps read a per-query budget flag before each scheduled
task, scan scores above the threshold, test host-to-device misses, and reserve
miss counts by warp. Several tasks can observe available budget before their
reservations commit. `atomicAdd` counts those misses; only `valid_count` inside
the 64-record bound is copied. The source therefore permits counters above
64 without copying more than 64 records. This explains counter semantics,
not the percentage of kernel time spent in atomics or host reads. Keep the
official kernel, strict threshold and prediction semantics unchanged.

## Local adaptation remains measurable

Current formal L0–L2 activity sums are below. They are intrusive node durations,
not independent API latency or additive full-model speedup estimates.

| Local work | Activity sum us |
| --- | ---: |
| Bounded free-slot preparation, 9 kernels | 18.080 |
| Packing + context/block tables + schedule + staging-table preparation | 33.824 |
| Official causal clean | 6.944 |
| Promotion validation + record copy/publication | 21.760 |
| Prefetch finalization | 19.424 |
| Finite-mean hint + decode EMA, 12 kernels | 36.288 |

Packing contributes 16.160 us of the 33.824 us preparation sum. The full
indexer-fused stage is 145.664 us: 83.136 us official fused work plus 62.528 us
of preparation, clean and promotion. These surrounding costs are separate
from the quoted fused-kernel difference. Bounded preparation and 256-thread
promotion validation already have accepted clean gains; their old costs must
not be counted again as remaining headroom.

At most two next optimization targets are justified by current evidence:

1. **The exact local Q1 hint update.** Its 36.288 us includes 25.280 us in the
   existing one-CTA PyTorch sum, plus 4.640 us masking/counting, 3.616 us
   publication and 2.752 us decode EMA. Investigate launch/scratch fusion while
   preserving the exact reduction order, nonfinite handling and separate EMA
   rounding. `offset[0]` remains observable by future Q>1 calls and cannot be
   omitted just because official decode consumes `offset[1]`. There is no
   current hint NCU evidence or measured candidate benefit; the fixed sum tree
   may limit parallelization.
2. **Repeated indexer-layout and metadata preparation.** Clean prepared/full
   component medians differ by 5.056–6.576 us per layer, and current formal
   preparation sums to 33.824 us over three layers. Investigate reuse of truly
   graph-invariant tables/metadata and incremental packing of unchanged
   history, with explicit cache identity, append/rollback, storage-budget and
   graph-lifetime checks. Keep the official input ABI and score arithmetic.
   This is measured work to target, not a validated savings estimate. Current
   promotion checks must precede copy; combining them without a global ordering
   proof or discarding false-positive predictions would change the contract.

## Longer gaps also appear in official ECHO

| Published window | Activities | Positive gaps | Median gap ns | Idle us |
| --- | ---: | ---: | ---: | ---: |
| Official HBM | 136 | 98 | 128 | 14.274 |
| Local HBM | 193 | 148 | 416 | 66.848 |
| Official ECHO | 291 | 275 | 448 | 127.648 |
| Local ECHO | 261 | 210 | 416 | 112.065 |

Independent raw SQLite reads match every selected interval and kernel launch
field and find no other process activity inside these windows. CSV endpoints
retain raw times; crossing HBM final norms are clipped before integer unions.
The medians describe gaps between merged recorded activity intervals, not
dependency-edge latency. Official ECHO also has longer gaps than its HBM
reference, so this observation is not unique to the local implementation.
Different workloads, collection histories and frameworks prevent attributing
it to a shared mechanism or deriving a cross-framework speedup. The local
ABBA preparation diagnosis remains a separate controlled observation.
