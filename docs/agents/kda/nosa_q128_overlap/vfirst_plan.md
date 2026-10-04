# Original Group8 current-V-first candidate

Status: rejected after the bounded layer-16 replay on 2026-10-04. Correctness
passed; complete-API latency and all three overlap samples failed promotion.

## Decision and baseline

Group4 is rejected. Although its representative API improved, all eight full 32-layer
sparse hidden comparisons failed the existing 0.016 tolerance; HBM/dense controls
remained exact. Its three overlap samples also failed. Its stage remains frozen
for diagnosis. Do not combine this candidate with Group4 arithmetic changes.

For original Group8, GPU0/NUMA0 actual layer 16 / request 16 replay gave median
complete-API CUDA/wall times of 0.754912/0.779994 ms, versus original serial
0.611424/0.636028 ms, over 31 samples. Three original Group8 page/stripe ratios
were 0.709719, 0.695349, 0.708886. Evidence is outside experiments at
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_group4_candidate_20261004/gpu0_matrix/`.
The existing harness records selected build identities, mapped binaries, source
hashes and CPU/GPU identity. It lacks a separate fresh complete dependency-header
inventory; this engineering comparison is not a final experiment publication.

## Draft and source argument

Test one scheduling change before changing host-load register demand. The producer
currently issues K(next) before V(current). Its next-K readiness wait can delay
already host-ready current V. Reverse only these two complete issue calls for
original Group8. Preserve initial K, query/output handshakes, final V, all memory
ordering and every consumer operation. Require exact accepted Group8 output.

Independent source review found no new cycle in the current two-stage separate
K/V pipelines. Before producer iteration i, K through i and V through i-1 have
been issued. Reusing V(i-2) requires consumer progress through QK(i-1)/PV(i-2),
whose inputs are already issued. Reusing K(i-1) after issuing V(i) has the same
property. Each stream retains its own index/phase sequence. Both consumer WGs can
drain the oldest stages using already-issued inputs. Keep `barrier_O.wait`: V
aliases epilogue storage. This argument is not GPU validation. Consumers still
wait for next K before consuming current V; earlier V may reduce K lookahead,
so benefit remains uncertain. No host-load ILP change is combined with this test.

## Executable plan

1. Create an isolated stage from the validated replay sources. Keep its Group8
   path selected for every invocation, including workspace planning. Record the
   parent manifest, updated source/harness hashes and explicit TMA order metadata.
   Do not edit repository runtime or the frozen Group4 parent.
2. Swap only Group8's adjacent producer issue calls. Compile through the normal
   loader outside the first-launch timeout; retain cooperative residency and
   register guards. Then bound the first actual correctness replay.
3. Replay actual layer 16 on GPU0/NUMA0. Every output must reproduce the accepted
   Group8 hash exactly and pass the independent FP32 tolerance 0.016. Check unique
   pages, stripes and bytes. Reject numerical failure before timing.
4. Measure 31 complete uninstrumented API samples with state restoration/poisoning
   excluded from timing, and three separate internal samples. Compare original
   serial Group8 and the parent async Group8; repeat a serial control on the new
   build if necessary. Existing page/stripe auditors and 90% gates remain unchanged.
5. Retain or reject on complete API latency and overlap. All-layer, full-model,
   formal serving and dependent-experiment gates remain mandatory before promotion.
   Keep valid reports until accepted replacements exist.

## Completed candidate and rejection

Run/stage ID: `nosa_q128_vfirst_candidate_20261004_01`, stored outside experiments
at `/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_vfirst_candidate_20261004_01/`.
Only the fused Group8 producer's adjacent issue calls changed; Group4 was disabled
in the loader and replay, and all workspace/trace planning used Group8. Repository
runtime sources and the parent stage remained unchanged. The normal loader compiled
build `3c9d1cb6af3823f8` successfully before the 180-second replay bounds.

The actual request-16 / layer-16 input used H=65,536, A=128, two KV heads, D=128
and BF16 on GPU0/NUMA0. Every initial, warmup, timed and profiled result reproduced
the accepted Group8 output hash
`3b98c15b81abbabb77a4d3368a0dd67f6dbc3599da37e072636e7cdb8a74886f`
and passed the independent FP32 check at `atol=rtol=0.016`. The maximum tolerance
ratio was 0.388199. Each call reported exactly 14,712,832 transferred bytes; each
profile independently checked 449 unique pages and 3,592 nonempty stripes.

| Complete API, 31 samples each | Median CUDA ms | Median wall ms |
|---|---:|---:|
| Parent original Group8 async | 0.754912 | 0.779994 |
| V-first Group8 async | 0.758784 | 0.783232 |
| Original serial Group8 control on the new build | 0.608864 | 0.632722 |

Timing includes the complete workspace API and GPU completion; state restoration,
poisoning and numerical checks are outside the interval. These are separate-process
operator replays, not an interleaved comparison or serving measurement. CPU
environments matched before/after all five runs and matched the parent async run.
The V-first CUDA median was 24.6% above the serial control, with no observed
improvement over the parent async median.

The three page-envelope ratios were 0.711660, 0.686685 and 0.686629; the respective
nonempty-stripe ratios were identical. All six values failed the unchanged 0.90
gate. Math coverage remained `softmax_only`; no interval definitions were expanded.
The candidate is rejected for latency and overlap. It did not proceed to all-layer,
full-model or serving performance work, and no existing report was replaced.

Evidence is in the stage's `compile/summary.json`, `data/run_sequence.json`,
`data/async_benchmark_01.json`, `data/serial_benchmark_01.json` and
`data/async_profile_01.json` through `data/async_profile_03.json`, with separate
stdout/stderr files under `log/`. All five replay processes exited zero and their
stderr files were empty. Final checks verified the 15-file stage and parent source
identities, selected build and frozen replay helper. Source manifest SHA256:
`5bd82b735ddd74764286b785fd4a55ee1095dfe116682e9ab1b274e4caf5955c`;
replay helper SHA256:
`11e5f1ecd13af46520655b681e36d5eec709d5b958d886bb1d189e62ff8db9f7`.
