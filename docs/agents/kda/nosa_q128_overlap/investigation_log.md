# Investigation log

## 2026-10-04: baseline and Group4 rejection

Full/source NCU run `nosa_q128_ncu_baseline_20261004_02` established the original
32-CTA launch, low tensor utilization and substantial data/pipeline waiting.
Its source PCs, units and profiling boundaries remain in [checkpoint.md](checkpoint.md).
That valid baseline evidence is retained.

The first staged candidate rebuilt unions for four-query groups and launched
64 real M64 attention CTAs. Representative API latency improved, but all eight
sparse full-hidden comparisons in the independent full 32-layer fixture failed
the fixed 0.016 tolerance, while HBM/dense controls remained exact. All three
representative overlap samples failed. Group4 is rejected; same-input attention
diagnostics are a separate validation boundary and cannot substitute for the
failed full-hidden contract.

## 2026-10-04: original Group8 V-first rejection

Run `nosa_q128_vfirst_candidate_20261004_01` reversed only the adjacent current-V /
next-K producer calls in original Group8. The source dependency review found no
new pipeline cycle, and actual layer-16 correctness preserved the saved hash and
unique payload. The 31-sample async CUDA/wall medians were 0.758784/0.783232 ms,
versus 0.608864/0.632722 ms for the original serial Group8 control. Three
page/stripe ratios were 0.711660, 0.686685 and 0.686629. The candidate is rejected
for latency and overlap; details and source identity are in
[vfirst_plan.md](vfirst_plan.md). Repository runtime and valid reports were not changed.

## 2026-10-04: active shared-union row-half candidate

Root authorized one replacement candidate after closing V-first. Preserve each
original eight-query union and split its original M128 row positions into two
contiguous M64 halves. This keeps physical GQA16 head order and the original page
pairing; it does not use four-query unions or the earlier proposed GQA8 remap.
The concrete contract is [union_halves_plan.md](union_halves_plan.md).

The stage is
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_union_halves_candidate_20261004/`.
Native and Python/resource work have separate owners and are undergoing review
and compilation. Exact same-input saved-hash equality across all 32 layers and
both serial/async routes is mandatory before full-model or performance work.
Independent FP32 tolerance is additional evidence, not an exception to that gate.
This is the only active candidate; no promotion is claimed.

The staged exact operator run `same_input_all 32_gpu4_02` subsequently passed
all 64 comparisons, with original saved Group8 hashes checked first, independent
FP32 comparisons, unique host bytes, async physical page/stripe envelopes and
strict cleanup. Root independently reopened all records and recomputed the raw
interval audits. Summary SHA256:
`192ee52c5d0ebece87760583138aad06a24098673ef4c9069fc817e6138f6002`.
The run's ratios are diagnostic only and remain below 0.90; correctness does not
establish overlap or latency acceptance. The independent full32 fixture is now
allocated on GPU4. It subsequently passed with all 16 graph outputs bitwise exact,
four eager references, actual Q128/halves2 and Q1024/halves1 sparse dispatches,
and five backend cleanup checks. Evidence SHA256:
`8f30cc3b94423cf029fc9973da2cd8237639e29a3e20bbc243a839048fd5dd8f`.
The coordinator independently reopened both gates and allocated GPU0/NUMA0 for
the prepared complete API and per-sample overlap matrix. Runtime integration,
formal serving and dependent experiment remeasurements remain outstanding.

The performance matrix completed as
`nosa_q128_union_halves_perf_gpu0_20261004_01`. All four 31-sample latency
configurations and six separate internal samples passed correctness, source and
cleanup checks. Async-half CUDA/wall medians were 0.632576/0.669274 ms versus
original serial Group8's 0.644640/0.669534 ms. Its page/stripe ratios were 0.710582,
0.728228 and 0.695460, failing every 0.90 check. Row halves are rejected for
promotion; exact operator and full-model checks are not performance acceptance.
Root independently reopened and recomputed decision
`3b5935595cab6677bc94d065216871fc87fe53030e5a1d098843bffc3ba5e0c5`.

A read-only reconstruction of the saved exact-gate traces identifies substantial
per-union and per-head copy dependencies, while also finding globally uncovered
windows with already copied next pairs. It distinguishes copy completion from
ready publication/TMA and does not attribute the residual gaps to one cause.
Supplementary WGMMA issue-to-wait spans would include delayed observation and
intervening waits; they cannot replace the fixed softmax-only overlap metric.
The next narrow feasibility review considers batching two adjacent stripe loads
before stores, while retaining physical uniqueness, original queue order and
all math. No implementation or improvement is claimed from that review.

The independent feasibility review found the two-stripe hypothesis sufficiently
concrete for one isolated successor. Root froze
[stripe2_ilp_plan.md](stripe2_ilp_plan.md) before edits and assigned native,
early-screen and independent-review ownership. It retains the original queue,
halves1 implementation and all math, increases only halves2 producer registers
to 64 subject to actual resource checks, and uses per-warp first-load timestamps.
A separate early-rejection screen can stop a failed representative overlap
candidate before expensive all-layer/full-model gates. It cannot promote a
candidate or relax any subsequent gate. No GPU launch or integration follows
from source plausibility alone.

The two-stripe implementation passed independent selected-SASS/resource review,
then the GPU4 partial-page gate at H=1,7,8,9,63 with all ten calls exact against
original Group8. The GPU0 representative screen subsequently passed original,
successor and profiled output/payload checks, but stopped at its first internal
sample: page and stripe unions were 387.328 us, softmax intersection 240.416 us,
and both ratios 0.6207038995373431. Root independently recomputed these quantities
from all 449 page and 3,592 stripe rows. The candidate is rejected; no further
samples, all-layer/full-model gates or latency runs followed. Detailed identities,
setup-only failures and evidence boundaries are in [checkpoint.md](checkpoint.md).

Root subsequently authorized one queue-priority candidate from the frozen
row-half parent, retaining one-stripe copies and24/240 registers. Independent
CPU reconstruction agrees on all 32 original prepared unions and finds that
minimum real pair-consumption rank advances several shared head1 dependencies.
It also delays some head0 dependencies; these queue-position proxies predict no
latency or overlap result. The bounded compactor, fallback and fresh validation
contract are frozen in [pair_priority_plan.md](pair_priority_plan.md).

Earliest-pair priority passed all 20 expanded partial checks, including all-native
complete-block selections, and the real-input screen's exact output/payload/queue
checks. Its first internal sample had 367.968 us copy union, 325.568 us softmax
union and 308.928 us intersection. Both overlap ratios were 0.8395512653274197,
so the helper stopped and rejected the candidate before any subsequent sample
or latency/full-model run. Root and HBM-audit independently recomputed the saved
records and final bindings. The 59.040 us uncovered copy splits into 17.344 us
startup and 41.696 us across 103 internal gaps, with no final copy tail. This does
not justify a tail-only fix or changing the metric. Residual diagnosis now informs
the next decision; the candidate and previous stages remain frozen outside
experiments, and repository runtime is unchanged.

Residual diagnosis is complete and independently checked. Root selected one
fanout tie refinement under [frontier_fanout_plan.md](frontier_fanout_plan.md):
keep minimum pair rank, prioritize more distinct original unions at that rank,
then preserve original ordinal. It adds only a bounded shared-key reference pass;
no additional HBM storage, launch or compute partition is selected. Singleton
dependencies may move later, and the evidence predicts no acceptance result.
Fresh partial/representative gates precede any full-model or performance work.

Frontier fanout then passed 93 native CPU tests, 147 helper tests, independent
actual-image/resource review and all 20 fresh partial calls. Both fused-main
bodies remain byte-identical to the actual parent; the changed compactor adds no
local spills or HBM workspace. Root and HBM independently audited exact output
pairs, fixed FP32 checks, physical queue/bytes/readiness, all 882 softmax rows,
40 pages,104 stripes and cleanup. Summary SHA256 is
`635099d7c93dce6c5b1d27e2e45a74a28c141c82e90032b08dadb18b161366ea`.
The fresh representative GPU0 screen passed original and successor correctness,
then rejected `internal_1` at 0.8307389668149162 for both metrics. Copy union was
374.144 us, softmax union 328.320 us and intersection 310.816 us. No later samples,
all-layer/full-model checks or API latency work followed. Result SHA256:
`729f27dacb3f2991736e332835028019a14ea568c18c5ba2763d19f88cf02477`.
Frontier fanout is rejected and frozen outside experiments; queue tie priority
alone has not met the fixed overlap gate.


Independent producer-warps subsequently passed 150 native and 158 screen CPU
checks plus independent source/interleaving and actual selected-image review.
Halves1 SASS/ELF identity holds; halves2 only preserves math source, with reviewed
fetch synchronization and fresh numerical acceptance required. Root froze the
screen at `d77b40a6043bfa18231aa1e640bb2bb2fee27f173dd014dc7041febf1d5c06c1`
with prelaunch review `685c20a3f73174d9877ab284ede496188e439d65654c3c27f969751a45591639`.
Partial20 and representative overlap gates are next. These static results do
not replace the rejected parent samples or any formal experiment.


Independent-warp partial20 passed on GPU4 and three independent audits accepted
all numerical, queue, byte, physical stripe and lifecycle checks. GPU0 screen
`nosa_q128_independent_warps_screen_gpu0_20261004_01` then passed both representative
correctness calls but rejected internal sample1 at 0.8005504587 for both metrics.
Its copy/softmax/intersection unions are 348.8/298.144/279.232 us; uncovered copy
is 20.928 us startup plus 48.64 us internal, with no tail. No later samples or API
latency runs were executed. This candidate is rejected and remains isolated.
The next diagnosis must identify the current64-CTA stalls; the old32-CTA NCU
report cannot establish the new independent-warp bottleneck.

## Current independent-warps NCU diagnosis, 2026-10-04

Run `nosa_q128_independent_warps_ncu_gpu0_20261004_01` completed with full and
SourceCounters reports. All 60 GPU application invocations passed their four
checked calls; no candidate was integrated. Independent source attribution
matched all 4,968 static instructions through each report's function base.
The [findings](independent_warps_ncu_findings.md) retain report identities,
resource/issue metrics, source counters and timed-sample extraction limits.

Host-dependent stores and consumer K/V transaction waits account for 43.60%
and 42.23% of the full report's base long-scoreboard samples; producer acquire
accounts for 0.138%. These are sample shares, not additive latency fractions.
Both host loads require 18 theoretical sectors per warp versus 16 ideal;
the total matches 517,248 TEX sysmem read-miss sectors in the full report,
1.125 times logical payload sectors. HBM stores show no such excess.
The next bounded check tests whether actual host-pointer alignment explains
this access amplification, using unchanged native code and explicit backing
capacity accounting. The existing 0.800550 overlap rejection still stands.

## Same-binary aligned-host sector comparison, 2026-10-04

Run `nosa_q128_aligned_host_ncu_gpu0_20261004_01` completed all 24 checked
GPU calls and both CPU audits. The [findings](aligned_host_diagnostic_findings.md)
record report identities and independently checked PC rows. With actual mapped
addresses changing from residue16 modulo32 to residue0, both host LDGs change
from18 to16 sectors per warp, and TEX sectors from517,248 to459,776. This
11.11% reduction removes the original12.5% amplification above payload. Output,
instruction count, logical payload and HBM stores remain unchanged.

Aligned owners alone occupy128 MiB of known rounded allocator backing versus
64 MiB for original owners; the diagnostic wrapper retains both, totaling192 MiB.
No runtime allocation is changed. The next isolated screen checks each aligned
internal page-envelope and stripe-copy ratio against0.90, stopping immediately
on failure. No API latency follows a failed screen. The four-pair ILP proposal
remains unselected.

## Aligned internal screen rejected, 2026-10-04

Run `nosa_q128_aligned_internal_gpu0_20261004_01` passed its two exact controls
then rejected internal sample1 at0.8408366534 for both metrics. Raw unions are
321.280 us copy,290.528 us softmax and270.144 us intersection, independently
recomputed by root. All numerical, payload, queue, host/native identity and
cleanup checks passed before the ratio decision. No later sample or API latency
was run. The [alignment findings](aligned_host_diagnostic_findings.md) retain
the receipt hashes and capacity limits. This establishes that eliminating the
sector amplification is insufficient for this candidate's overlap acceptance.
Root selected one isolated four-pair host-load ILP preparation with original
host buffers; production sources remain unchanged.

## Four-pair ILP screen rejected, 2026-10-04

The isolated full-stripe schedule and halves2 producer80 budget passed independent
compiled review and all twenty fresh GPU4 partial calls. GPU0 run
`nosa_q128_four_pair_ilp_screen_gpu0_20261004_01` preserved both representative
exact controls, numerical/payload checks and original host residue16, then rejected
`internal_1` at 0.7027860967419911 for both overlap metrics. Cleanup and final
source/native/helper integrity passed. No later internal sample, complete-API,
all32 or full32 candidate run followed. The eight-load schedule and stack 32→0
change were tested together; this single trace establishes neither ILP-only
causality nor a statistical latency difference. No native Q128 candidate is
selected this cycle; the [candidate record](four_pair_ilp_plan.md) retains the
bounded evidence and immutable helper identities.
