# NOSA candidate efficiency: remaining evidence and next experiment

> Current checkpoint: the pool-scan successor is integrated and its affected
> fixed/GR measurements are published. See the [publication record](nosa_pool_scan_publication.md)
> for current run IDs and retained input mappings. The earlier API, repeat,
> probe and intervention results below keep their historical source boundaries;
> references to an external-only candidate describe that earlier stage.
> Further measurements are limited to H64K. The subsequent bounded
> [prequeue diagnostic](nosa_hbm_prequeue_diagnostic.md) completed both requests:
> it supports sensitivity to prior submission, while ordinary serving gains,
> the remaining post-check tails and compute/IO dominance remain unresolved.
> The later [captured-copy pilot](nosa_captured_copy_pilot.md) preserved exact
> H64K outputs but did not consistently improve ordinary candidate medians;
> that external version is not promoted. Its mean/tail differences do not
> establish scan elimination or a full-serving gain.
> The later [descriptor-reuse diagnostic](nosa_copy_descriptor_result.md) found
> approximately 0.1 ms lower candidate medians against its matched external
> control. It is not promoted; the next bounded target is
> [indexer allocation/dispatch](nosa_indexer_dispatch_plan.md).

The accepted fixed family is `nosa_motivation_sm90_20261004_03`, with matching
profile `nosa_motivation_profile_sm90_20261004_03` and API reference
`nosa_attention_reference_sm90_20261004_02`. Runtime source is
`a73ad32b0a3322cfb06c66121f50345d2ef6004b465733115fee6006a0f2f18a`.
The [published experiment](../../../experiments/nosa_motivation/README.md)
separates full-request proximity from candidate efficiency and preserves the
failed async gates. This follow-up concerns research items 2.1 and 4.1.

## What the current evidence establishes

HBM candidate extend takes 15.988631 and 16.935379 ms for requests 0 and 16.
Independent GEMM/BMM plus FA3 compositions total 12.140224 and 12.277728 ms.
Their MFU ratios are 75.930354% and 72.497510%. These are sums of independent
API medians, not complete alternative model executions. Their complementary
fractions are not measurements of CPU launch cost or recoverable waste.

A read-only review identified all 32 candidate layer scopes in each of six
accepted HBM traces, following the 64 history chunks. GPU activity was joined
through CUDA runtime/driver correlation and excluded parent annotations.
The following are medians of three instrumented samples for each request:

| Quantity | Request 0 ms | Request 16 ms |
| --- | ---: | ---: |
| Candidate compute-API device activity union | 12.728753 | 12.820709 |
| Candidate per-invocation API span union | 13.256183 | 13.365640 |
| Separately identifiable omitted nonmatrix kernels | 1.796494 | 1.798919 |
| Mixed indexer QK / normalization / GQA / pooling kernels | 2.567275 | 2.563864 |
| Unscoped staging/cache copies inside the API device hull | 0.347328 | 0.349218 |
| Profiled complete candidate extend | 24.175597 | 23.975729 |

The omitted kernels include graph RMSNorm/residual, RoPE, CIS transforms and
SiLU (about 0.676–0.677 ms), plus indexer finite validation, compression/CIS
ranking and stable selection (about 1.120–1.122 ms). FA3 prepare/repair and GEMM
split-K reduction already belong to their reference APIs. The two mixed indexer
kernels both issue QK matrix products: the first obtains normalization and the
second recomputes scores for normalized probabilities. Their matrix/nonmatrix
shares cannot be separated with these traces.

The profiler inflates candidate extend materially. No profiled interval is
subtracted from the uninstrumented 3.848407/4.657651 ms composition differences.
The API device hull contains gaps, but host work, dependencies, submission and
instrumentation remain confounded. Neither gaps nor this table prove that CPU
launch, compute or IO dominates the unprofiled request.

## Diagnostic sequence

1. Establish a minimally instrumented, synchronized complete-candidate baseline
   using the accepted full32 checkpoint, exact saved requests and preceding cache
   state. Keep H65536/A128/C1024, P65536/NH16777216, compute graphs, precision,
   GPU5/NUMA1 and register8. Repeat ordinary runs before adding instrumentation;
   keep every sample and separate request 0/16 from the observed tail requests.
2. Measure the complete A128 indexer API, including both score passes, checked
   preparation and stable selection, on captured real operands. Compare it with
   the matched score-matrix subset while naming the different work boundaries.
   This locates an optimization opportunity; it does not by itself prove a
   complete-candidate improvement.
3. Use aligned CPU/runtime/CUDA evidence only where needed to distinguish
   dispatch, staging/dependencies and device work. The existing trace contains
   64 graph launches, 320 kernel launches and 157 async copies per candidate API
   CPU hull; counts alone do not measure recoverable cost. Verify instrumentation
   overhead against the ordinary baseline before interpreting gaps.
4. For tail localization, replay dense request ID 20 and sparse request ID 22
   with their immediate neighbors and the exact prior user/cache trajectory.
   These are one-based figure positions 21 and 23. All are hits; useful work and
   graph counts are unchanged. The completed pool-scan probe below identifies
   an actual host cost in matched diagnostic tails; the formal events themselves
   were not instrumented.

### Ordinary repetitions completed

Two fresh sequential worker processes repeated the complete accepted trace with
the original method order and warmups, using the existing `run_case` without
per-layer instrumentation. Both passed all 128 full-output comparisons against
the accepted run, graph/lifecycle/transfer checks and final source/native,
checkpoint and recorded CPU-environment guards. GPU5 was released afterward.
This completes the first diagnostic step; no API timing was repeated.

| Tail request, zero-based | Repeat 1 extend / neighbor excess ms | Repeat 2 extend / neighbor excess ms |
| --- | ---: | ---: |
| Dense 20 | 104.305 / 37.412 | 104.338 / 37.338 |
| Sync 22 | 60.219 / 35.744 | 60.422 / 35.752 |
| Async 22 | 63.031 / 35.624 | 63.280 / 35.718 |

The same request is the largest candidate latency among 16 revisits for its
method in the formal run and both repetitions. All 256 same-request structural,
useful-FLOP, transfer and graph comparisons agree with the formal records. These
observations establish recurrence under the measured conditions, not a cause.
The two repetitions are engineering diagnostics, not a new formal publication
or statistical confidence estimate. Comparing their HBM candidate times to the
retained API sums does not create a contemporaneously remeasured API reference.

Evidence is in
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_candidate_repeat_20261004_01/` and the
independent read-only comparison in sibling `nosa_candidate_repeat_review_20261004_01/`.
The initial config-serialization preflight failure (tuple versus JSON list) had
zero timed requests; it is preserved separately. Driver correction normalized
serialization only and changed no runtime/configuration value.

| Receipt | SHA256 |
| --- | --- |
| Validated repetition analysis | `38fd0d133d08999818ae2a24dcedf9e3e18699b9107b6a497abfff5b2101414d` |
| Worker launch/completion | `083e165b680bd859d84cdcf43c8e61c3ee383c660ebacbfa5bfcd6fd5510c51e` |
| Independent timing comparison | `89739b378cf63332f3758401e688cfd4b19a62b64787a241802fd74fe4d75769` |
| Independent comparison provenance | `ef5ab13b269874a0cb788689ff7c974a7d6fece94405c4ee34ddd055a17cad2d` |

### GC and allocator probe completed

Run `nosa_gc_pool_probe_20261004_01` repeated the complete four-method trace
with the original warmups and method order. It recorded GC callbacks and the
existing pool-guard/referrer calls separately, with preallocated records and
end-only IO. No profiler, per-layer ranges, extra CUDA synchronization or tensor
allocation was added. All 128 outputs were exact; the existing graph, lifetime,
transfer, source/native and CPU-environment checks passed. Capture reported no
overflow or errors. An independent reconstruction matched all 99 candidate
interval count/sum/union fields to the raw events.

| Tail request, zero-based | Extend ms | GC union ms | Inclusive pool guard ms | Nested referrer scan ms |
| --- | ---: | ---: | ---: | ---: |
| Dense 20 | 104.866715 | 1.162762 | 36.699800 | 36.471005 |
| Sync 22 | 60.249477 | 0 | 36.105803 | 36.012608 |
| Async 22 | 63.422280 | 0 | 36.624069 | 36.528546 |

These matched tails contain approximately 36 ms of host referrer scanning.
This is scan work, not GC collection time. The inclusive guard contains its
referrer scan; overlapping and nested durations must not be added. The dense
scan starts 67.731294 ms after candidate entry, while the sync/async scans start
at 0.111608/0.112557 ms. Annotated neighbors and HBM samples have no full scan.
A gen1/gen2 collection-stamp change triggers this ordinary-generation scan in
`_reject_python_pools`; gen0-only changes use the narrower path.

Probe-minus-repeat candidate differences for these three tails are
+0.562/+0.529 ms, +0.031/−0.173 ms and +0.391/+0.142 ms versus repeats 1/2.
Across methods, revisit median paired differences are +0.116…+0.299 ms versus
repeat 1 and +0.010…+0.097 ms versus repeat 2; first-visit offload differences
are larger. These describe different runs and do not isolate observer overhead.
Only 11 requests have absolute candidate annotations; unannotated events cannot
be attributed from this capture. This probe does not measure removable savings
or retroactively instrument the formal run; the later intervention is separate.

Raw probe material is in
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_nsys_followup_20261004_01/gc_probe_01/`;
independent reconstruction is in sibling `nosa_gc_probe_review_20261004_01/`.
The final seal in sibling `nosa_gc_probe_seal_20261004_01/` inventories 251 raw
files and retains all 512 paired request/extend comparisons.

| Receipt | SHA256 |
| --- | --- |
| Independent interval reconstruction | `9ee62f9a0321b9b9a9fd455a7fa83161c96fedfded902dc0f6bcabca794df462` |
| Final probe seal | `682ae6868a778918492be9bbbfbd6e89394837ef372e881929253ba0ba13324b` |
| Raw-file inventory | `757229906e37892066d1d8bb6f83ba866e9236f7ca694d80100ad2ccd35b352b` |

### Pool-scan optimization remains external

The immediate measured target is the full pool scan. Preserve GC-visible empty
MemPool rejection, subclass inventory/retry, audit errors and events, frozen
generation exclusion, and conservative behavior during active collection or
reentrancy. Do not disable GC, change its thresholds or freeze the heap to hide
the cost.

The first CPU-only native visitor prototype is not ready for integration.
`PyUnstable_GC_VisitObjects` also visits frozen objects, and the installed
CPython 3.12.13 callback convention differs from its versioned documentation.
The one-pass prototype reduced ordinary-heap helper time, but regressed on a
large frozen heap and did not establish equivalent audit-event behavior.
Its findings and evidence remain in
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_pool_visitor_investigation_20261004_01/`
at SHA256 `c34460c81a14294b7d1920544f3be43417d7317bead13aa7b43126f4fae6eba1`
and `43ca0f79d197ee6894a357c62c68879860de113f341306a54d2066fee62f6e87`.
Later ordinary-generation prototypes exposed trace/signal and dynamic-lookup
gaps. The latest external successor has passed bounded semantic review and nine
actual CUDA empty-MemPool cases. Its first full-trace baseline/native pair is
complete, with all 128 outputs in each arm exact to the accepted outputs.
Dense20/sync22/async22 extend times change from 104.600/60.356/63.451 ms to
90.800/46.532/49.468 ms. These requests remain maxima, about 22–24 ms above
their native revisit medians. Offload revisit means improve, but their medians
and first-visit extend means increase; 47 of 48 offload first-visit candidates
are slower in native. HBM also has a new maximum at request 24. One sequential
pair cannot isolate run order or system drift. A second pair with reversed arm
order and unchanged driver/prototype also completed all 256 exact-output checks.
It reduced the same three tails by 13.351/13.841/13.351 ms. Offload revisit
extend means improve in both pairs; first-visit and typical effects remain mixed.
The pooled means are descriptive repeats of the same IDs, not independent
samples or an isolated net-effect estimate.
The [pool-scan contract](nosa_pool_scan_optimization.md) records the rejected
versions, exact identities, complete first-pair evidence and remaining
acceptance. Production is unchanged; the observed partial tail reduction does
not establish a uniform or net serving speedup.

The complete A128 indexer API replay and bounded Nsight diagnostic below are
now available. Nsight's observer overhead and event-loss warnings leave complete
dispatch/staging attribution unresolved; further captures are not automatically
required before acting on the directly measured host scan.
Graph helper fusion remains a smaller hypothesis. Any kernel
candidate should follow the project's KDA workflow. Preserve finite rejection,
BF16 rounding, stable selections, causal masks, transaction semantics, unique
host reads and asynchronous buffer lifetimes. Do not remove required work to
improve the reported ratio.

An external integration memo is prepared in sibling
`nosa_pool_scan_integration_feasibility_20261004_01/memo.md`, with receipt SHA256
`86f43d819a2a7c67faa362e5c688c228debeba3048f6103e9b5c6d778cc8436a`.
It proposes a literal getter-preserving resolver expression in the original
helper, a once-only authenticated optional loader, retained original fallback
and retries, and explicit fixed/GR provenance. Initialization reentry must use
original inspection without deadlocking or publishing partial setup. This is
a plan for a new candidate, not an integrated or verified runtime.

### Bounded Nsight diagnostic completed

Runs `nosa_nsys_hbm16_graph_20261004_01` and
`nosa_nsys_hbm16_node_20261004_01` each execute HBM requests 0–16 after warmup,
capturing only request 16. All 34 saved outputs independently match the accepted
outputs exactly. This bounded history differs from the complete four-method
process trajectory. Minimal native NVTX ranges do not wrap individual layer
APIs, and no extra synchronization or tensor copies were added.

| Candidate diagnostic | Graph mode | Node mode |
| --- | ---: | ---: |
| Original runner extend ms | 17.979575 | 21.060208 |
| Relative to two ordinary observations' median | +7.917885% | +26.408610% |
| Observed kernel/memcpy/memset union ms | 7.832886 | 13.498574 |
| Separately reported graph-envelope union ms | 5.493834 | 0 |
| CUDA API CPU interval union ms | 3.636874 | 5.342762 |

Both candidate observations exceed the predeclared 5% diagnostic tolerance;
whole-request changes are smaller because history prefill dominates. Graph
envelopes are not node activity. Neither mode establishes a complete activity
inventory or GPU idle time. The analyzer leaves those completeness/idle fields
unset, clips phase-crossing activities and refuses ambiguous correlations.

Both SQLite `DIAGNOSTIC_EVENT` tables warn that not all NVTX and CUDA events
might have been collected. Stderr is clean, but it does not override the database
warnings. Nsight reports software instrumentation because hardware tracing is
unsupported. Zero unmatched observed launch correlations cannot rule out
missing events. The observed graph-mode records also contain 323
`cuKernelGetName` and 256 `cudaGetDriverEntryPointByVersion` calls; their origin
is unresolved and they are not counted as ordinary production overhead.

The node capture retains 803 kernel and 161 memcpy records in the candidate
window, which also contains 64 graph-launch API records. It locates observed FA3, GEMM and indexer
work, but no profiled interval is subtracted from ordinary request timing to
estimate CPU cost. Evidence is in
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_nsys_followup_20261004_01/`, with
separate graph/node outputs and `capture_pair_analysis_01/`. No automatic rerun
is planned merely to seek a complete-idle claim. The final receipt SHA256 is
`092a879359199f89fd45d3ace55eed8da966a2ec77e5180021d1047d966e5512`.

### Complete A128 indexer API replay completed

The accepted attention captures also contain actual raw Q/K/CIS for all 32
candidate layers at requests 0/16. Their original Q stride was restored, and
native history compression was reconstructed from the saved H-token prefix.
Full-prefix and 64-chunk reconstruction agree byte for byte; the two captures'
history K/CIS prefixes also agree for every layer. No new model forward or input
capture was needed.

Runs `nosa_indexer_api_q128_r0_20261004_02` and
`nosa_indexer_api_q128_r16_20261004_01` passed exact full IDs/masks and downstream
attention-output checks for all 32 layers. Three separate layer-0 nonfinite
Q/new-K/new-CIS trials were rejected and retained prefix state/bytes remained
unchanged. The timing keeps both score passes, finite preparation, stable
selection, output allocation and the original 32-layer deferred finite decision.
All 42 samples are retained across the complete indexer and two arithmetic
subsets, with three warmups and seven repeats for each input/API group.

| Independent 32-layer API median | Request 0 wall / event ms | Request 16 wall / event ms |
| --- | ---: | ---: |
| Complete current indexer | 4.792001 / 4.769504 | 5.058357 / 5.034816 |
| One actual-operand raw QK product per layer | 2.025725 / 2.013600 | 2.025631 / 2.013152 |
| Two actual-operand raw QK products per layer | 4.022915 / 4.010912 | 4.022597 / 4.010240 |

These are consecutive standalone API calls, without model-layer interleaving.
History reconstruction, candidate staging, transaction begin/discard and output
comparisons are outside timing. The event span includes submission gaps and the
finite decision; it is not a kernel-activity union. Allocation lifetimes and
cache temperature differ from the complete serving request.

The new QK subset uses actual BF16 Q and native-compressed K, grouped as
`[2,2048,128] × [2,128,4103]`, with shared GQA K and FP32 output. Packing is
outside timing, and every product element is checked against IEEE FP32 BMM.
Each product pass logically writes 2.003418 GiB of explicit output across
32 layers; two passes write 4.006836 GiB. The native fused score path does not
materialize those outputs. These byte counts are not measured HBM traffic, and
the raw products do not give a native-score lower bound or a removable-overhead
estimate.

The published composition's earlier indexer QK term remains distinct:
2.070080 ms, median of five CUDA-event samples, using random BF16 operands with
physically expanded GQA K, BF16 output and CUDA Graph timing. Neither reference
is silently replaced by the new grouped result. No API replay interval is
subtracted from the formal candidate wall time.

Evidence is in
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_indexer_api_followup_20261004_01/`.
The independent read-only analysis is in sibling
`nosa_indexer_api_analysis_20261004_01/results/`, with receipt SHA256
`24b11330345327ebc1054ff1333d8ea3e6a1f158a6871c15b2ffb38266318915`.
The initial request-0 attempt stopped at native-build preflight because its
compiler PATH lacked the recorded `.venv/bin` prefix; it has zero timed samples.
Its frozen helper copies are retained. Restoring the accepted PATH produced
an exact native-build match before the successful new run.

A candidate must improve paired complete-candidate timing on the same inputs,
then pass the relevant correctness gates and affected formal replacements before
publication. Keep the accepted reports until those replacements are ready.
The earlier Q128 fetch successors remain rejected external diagnostics; no new
fetch implementation has been selected. Ordinary owned A1024, resident/module,
pattern and cold-union overlap reports are outside this wrapper's affected path.

## Evidence

The review ran without GPU execution and changed no runtime or experiment data:
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_candidate_api_review_20261004_03/`.

| File | SHA256 |
| --- | --- |
| `candidate_api_review.json` | `64423be2c4f65f2f392715585c8c73a0bba009251896f596167093b9593c5bbd` |
| `candidate_findings.json` | `443b04b556db76ce1427e37d1d232ac1272fcba75547fefd437eae4b7494ab72` |
| `candidate_findings.md` | `6b1b9f5107242283dab0819cecd8eca9a8761d4317c0845cead683c2f3100f0e` |

The detailed receipt preserves all candidate API scopes, CPU/device intervals,
kernel names, union arithmetic and trace/source hashes. This document is an
execution plan, not an additional performance result or evidence that the user's
complete overhead objective has been achieved.
