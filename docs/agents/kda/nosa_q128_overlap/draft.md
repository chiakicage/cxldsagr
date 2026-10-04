# Draft: establish the Q128 bottleneck before Group4

## Observed baseline

Source and the accepted candidate-only CUDA trace agree: Group8 produces `ceil(128/8)*2=32` attention CTAs, each with 384 threads. Default `fetch_ctas=96` is clamped to 32. Warp 0 retains TMA, producer warps 1–3 supply 96 fetch threads, and two consumer warpgroups supply attention. The actual compiled CTA has 168 registers per thread and a 64,512-register pool. The trace's static estimated occupancy field reports zero because of its shared-memory model; it is not evidence that the running kernel had zero occupancy. The native launch checks opt-in shared memory and runtime occupancy separately.

Serial fetch launches 1,024 CTAs of 128 threads for this history geometry. Q128 does not activate the two-head/256-work-item head-phased specialization. Both heads start work concurrently; fetch order is block 0 first, then descending blocks with interleaved heads. Prepared attention unions are ascending and consumed in reverse paired-page order.

Diagnostic request 16/sample 0 moves 289,079,296 payload bytes over 32 layers. The separately instrumented internal replays report async stripe-copy union 12.107776 ms and serial copy union 5.766528 ms. These are diagnostic work intervals, not uninstrumented latency. Async copy intervals are almost continuous. Of 3.449952 ms without a simultaneous recorded softmax update, 0.246080 ms precedes the first update, zero follows the last update, and 3.203872 ms lies between updates. Recorded math covers softmax only, so these gaps cannot be called proof that all attention math was idle.

The candidate-only CUDA timeline reports async main kernels totaling 12.092423 ms; serial fetch totals 5.844576 ms and original FA3 totals 3.121523 ms. Async compaction adds 0.219552 ms. Thus compaction removal alone cannot close the observed main-kernel difference. The current page and stripe ratios are identical per layer and span 0.6320567–0.7805486; all 32 applicable layer samples fail 0.90. One sample establishes this failure, not repeatability.

## Hypotheses and unknowns

The strongest source-supported hypothesis is insufficient useful host-load concurrency with only 32 spare producer warpgroups. NCU must distinguish long-scoreboard pressure, memory-system throughput, register spills, barrier stalls, queue atomics, and consumer starvation before this becomes a causal diagnosis. Direct pinned-host loads are not HBM traffic; low `dram` utilization alone cannot identify their bottleneck.

The secondary hypothesis is synchronization between page publication and similarly ordered compute batches. Blindly applying head phasing to Q128 does not add workers and can reduce simultaneously active attention groups. The current evidence shows no copy tail beyond the final softmax update. A scheduling change requires its own measured justification.

Group4 could double real attention CTAs and fetch warpgroups while preserving the global unique-page queue. It changes the native CTA from M128/12 warps/two consumer warpgroups to M64/8 warps/one consumer warpgroup. The pinned FlashInfer scheduler has two-/three-consumer assertions; a one-consumer scheduler requires explicit adaptation. Q/KV layouts, `Rows`, `Capacity`, fallback coverage, trace offsets, scratch planning, register allocation, and output stores must be audited together. Group2 is not a direct alternative: M32 violates the upstream `CTA_Q % 64 == 0` condition.

Group4 also changes per-group sparse unions and KV tile pairing. Its accumulation may therefore differ from Group8 despite identical selection semantics. Original Group8 remains bound to captured output hashes; the candidate is assessed against independent references at the repository's existing operator/full-model tolerances (`atol=0.016`, `rtol=0.016`), with exactness reported separately. This numerical contract is set before implementation, not adjusted in response to a candidate failure.

## First steps

1. Finish the executable baseline plan and an actual-input replay using the existing native API. Its build already includes `-lineinfo`; no CUDA code extraction or duplicate launch implementation is needed.
2. Restore captured ownership tags and valid hit records before every call. Poison nonowned history pages, stage actual suffix values through the real API, and verify both output hash and independently recomputed copied bytes. Reset after each warmup.
3. Collect full/PM and source-counter NCU profiles for representative actual layers, using application replay. Read available SM90 metrics through `ncu_report`; do not import B200 metric names or performance claims.
4. Revise the Group4 implementation plan using actual counters. Only then implement one isolated candidate.

## Knowledge-base use

KernelWiki `technique-tile-scheduling` (`wiki/techniques/tile-scheduling.md`, source-reported) supplies the static-stride scheduling model, matching this kernel's `ordinal += compute_ctas`. Its listed sources (`doc-nvidia-tuning-guide`, `doc-cutlass-blackwell`, `pr-cutlass-2161`) are Blackwell focused; they do not prove Q128 speedup. `technique-warp-specialization` (`wiki/techniques/warp-specialization.md`, source-reported) distinguishes Hopper's 128-thread WGMMA warpgroups and register accumulators from Blackwell's TMEM model. Its source excerpts were followed; no SM100 CLC/TMEM design or performance number is transferred to Hopper here. All Q128 numerical findings above come from this repository's source and accepted diagnostic evidence.

Promotion, rejection, and retention rules are in [task.md](task.md). Concrete commands are in [implementation_plan.md](implementation_plan.md).
