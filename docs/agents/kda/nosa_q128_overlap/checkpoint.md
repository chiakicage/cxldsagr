# NOSA Q128 optimization checkpoint

Status on 2026-10-04: Group4, original Group8 V-first, shared-union row halves,
two-stripe fetch ILP, earliest-pair queue priority, frontier fanout and independent
producer-warp stripes are rejected for promotion. The latest independent-warp
candidate passed all 20 fresh partial checks and representative exact/FP32,
payload, queue and lifecycle checks. Its first internal sample reached 0.800550
for both overlap metrics, below 0.90; the screen stopped immediately. No further
samples, all-layer/full-model acceptance or complete-API latency run followed.
Fresh current-source profiling completed under
[independent_warps_ncu_plan.md](independent_warps_ncu_plan.md). The
[findings](independent_warps_ncu_findings.md) locate the largest sampled
dependencies at host-dependent stores and consumer K/V transaction waits, and
identify 12.5% host-sector amplification. The subsequent
[same-binary alignment comparison](aligned_host_diagnostic_findings.md) records
actual mapped addresses and eliminates that amplification: host/TEX sectors fall
from 517,248 to 459,776 while output and payload remain exact. This diagnostic
does not reverse the failed overlap gate. The aligned internal screen subsequently
stopped at its first sample with both ratios0.8408366534; no API timing followed.
Its extra pinned backing also precludes direct allocator promotion. Root selected
isolated four-pair host-load ILP preparation with original allocations next.
The frozen [independent_warps_plan.md](independent_warps_plan.md) records that
candidate's original authorization. Repository runtime remains unchanged; the
unaffected resident reports and the separate A1024 offload report have been
replaced after fresh audits. They do not establish Q128 serving performance.

## Evidence identity and boundary

Run `nosa_q128_ncu_baseline_20261004_02` is staged at `/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_ncu_runs/nosa_q128_ncu_baseline_20261004_02`. Its layer-16/request-16 replay uses the finalized `nosa_attention_reference_sm90_20261004_01/capture_000016` and accepted diagnostic `nosa_motivation_current_diagnostic_20261004_02`. Full and source reports contain one `fused_main` action each. Both collection stderr files are empty, and every application replay completed its output and payload checks.

Validation reproduces attention hash `3b98c15b81abbabb77a4d3368a0dd67f6dbc3599da37e072636e7cdb8a74886f` and 14,712,832 bytes over 449 unique miss pages. The validation replay's page and stripe overlap ratios are both 0.6846563815; this single intrusive replay is not an acceptance result. CPU affinity and calling-thread memory policy record NUMA node 0 binding and PyTorch intra-op threads 8. Old serving metadata did not record this environment, so the baseline replay is not claimed to prove a match to the historical CPU/NUMA placement.

The observed device is NVIDIA M403, capability 9.0, 132 SMs. NCU full/source durations are 587.968/585.600 microseconds; application replay, cache control and instrumentation make these diagnostic kernel durations, not complete-API or serving latency.

## Six-dimension diagnosis

| Dimension | Direct evidence | Interpretation and limit |
| --- | --- | --- |
| Launch and occupancy | Grid 32, block 384, 168 registers/thread; dynamic shared memory 197,728 B, reported total 198,752 B; 0.242424 waves/SM. Register and shared-memory occupancy limits each permit one CTA/SM. | Only 32 real attention CTAs supply 32 sets of spare fetch warps. The kernel is far below a full device wave. Active-cycle warp occupancy is 18.673%, consistent with one 12-warp CTA; the trace's earlier static zero estimate is not actual occupancy. |
| Work balance | SM active cycles: minimum 0, average 255,958.068, maximum 1,075,097. Per-SM instances are unavailable. | The aggregate mean/max ratio corresponds to about 31.4 fully active SMs and is consistent with the 32-CTA grid. It does not identify individual SMs or quantify load balance among the occupied CTAs. |
| Stalls | Full report: 11,219 long-scoreboard samples of 16,049; source report: 11,237 of 16,035. Average eligible warps/active cycle is 0.118122; issue-active fraction is 0.091244. | Most sampled long-scoreboard sites wait for TMA transactions, followed by stores dependent on direct host loads. This supports a producer/data-arrival bottleneck; it does not prove all attention work is idle during every wait. |
| Tensor work | Tensor pipeline 2.172284% of elapsed peak and 9.183322% of active-cycle peak. | Low utilization accompanies the small grid and data waits. It is not evidence of a saturated tensor compute pipeline. |
| Time variation | Fifteen `pmsampling:` warp-stall series and timestamp-correlated `warpsampling:` series are available; SM/tensor throughput timelines were not collected. Series have different replay-pass timestamp origins and zeros around the target window. | Examine each series on its own relative time axis. Warp counts are not utilization percentages. Absolute timestamps from different passes cannot establish temporal overlap. The accepted internal copy/softmax instrumentation remains the overlap evidence. |
| Memory and spills | TEX system-memory read misses: 459,776 sectors × 32 B = 14,712,832 B, exactly the unique payload. Full HBM read/write counters are 1,467,904/503,040 B. Source local spill read/write counters are 11,064/23,474 dynamic instructions; full local-load L1 hit rate is 100%. | Direct host traffic is accounted through the TEX aperture metric, not HBM bandwidth. Aggregate system-memory metrics include other source units and are not substituted for payload. Spills exist, but these counters do not establish them as the primary bottleneck. |

## Source-counter corroboration

| Source region | Full long-scoreboard samples | Source-report long-scoreboard samples |
| --- | ---: | ---: |
| CUTLASS `barrier.h:424`, transaction barrier wait | 4,244 | 4,260 |
| CUTLASS `sm90_pipeline.hpp:620`, consumer wait | 2,885 | 2,896 |
| CUTLASS `sm90_pipeline.hpp:621`, consumer wait | 1,374 | 1,339 |
| Owned fused source line 290, store after host K/V loads | 1,568 | 1,583 |
| Owned fused source line 377, ready-flag acquire loop | 706 | 684 |

The source SASS at owned lines 288/289 uses `LDG.E.128.STRONG.SYS`; line 377 uses `LDG.E.STRONG.GPU`. Barrier samples additionally concentrate around fetch task broadcast, writer completion, and loop synchronization (source lines 263, 297 and 324). This corroborates actual fetch/pipeline synchronization overhead rather than an inferred launch-only explanation.

Only `smsp__pcsamp_*` correlations are source PCs. `warpsampling:*` correlations are timestamps and must never be passed to source lookup. The source report identifies local spill reads/writes in instruction units; the full report's derived spill field has an inconsistent displayed byte unit, so it is not used as transferred bytes. Raw reports and the coordinator's metric exports retain the original values and units.

The long-scoreboard PM series has 400 nonzero samples of 553. Its first-to-last nonzero window spans 598.496 microseconds; decile raw means within that series are 1093, 1237, 1355, 1564, 1578, 1578, 1609, 1493, 1466 and 1154. These warp-counter deltas show the stall burden persists through the middle of the sampled window, rather than appearing only at the tail. They are not occupancy or utilization percentages. The coordinator saved the bin definitions and timing boundary in the run's `data/metrics/full_async_sparse_l16_pm_bins.json`.

## Completed candidate decisions

Group4 changed the eight-query union into four-query unions. Its representative
complete API improved, but all eight sparse full-hidden comparisons in the
full 32-layer fixture failed the existing 0.016 tolerance; HBM/dense controls
remained exact. All three representative overlap samples also failed. Group4 is
rejected. Separate same-input attention diagnostics do not overturn the independent
full-hidden failure. Its engineering evidence remains outside experiments under
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_group4_candidate_20261004/`.

Original Group8 V-first changed only adjacent producer issue order. It preserved
the accepted layer-16 output hash, independent FP32 tolerance and exact unique
payload, but failed performance acceptance:

| Complete API, 31 samples | Median CUDA ms | Median wall ms |
|---|---:|---:|
| Original Group8 async parent | 0.754912 | 0.779994 |
| V-first Group8 async | 0.758784 | 0.783232 |
| Original serial Group8 control on the V-first build | 0.608864 | 0.632722 |

V-first's three page/stripe ratios were 0.711660, 0.686685 and 0.686629. Every
sample failed 0.90. It is rejected; no all-layer or full-model work followed.
These are separate-process, actual-input operator replays, not serving results.
The source identity, run files and validation boundary are recorded in
[vfirst_plan.md](vfirst_plan.md).

The NCU evidence above still supports both data-arrival and pipeline-wait concerns.
It does not assign the gaps between softmax intervals to a single cause. V-first's
result supplies no measured benefit from that producer-order change. The existing
softmax/page/stripe metric definitions and 90% gate remain unchanged.

## Shared-union row halves: numerical pass, performance rejection

The candidate starts from repository Group8 sources and includes neither rejected
change. It retains 32 original unions, page pairing and masks, while splitting each
original M128 row range into two real M64 work items. It has an explicit
Q128/two-KV-head opt-in, 64 work identities, the original union-indexed storage and
one unique physical host-fetch queue. Duplicated HBM-to-shared K/V and CIS work
must be distinguished from unique host bytes.

The fixed stopping order is static/compile safety, exact same-input saved hashes
for all 32 layers in both serial-half and async-half modes, independent full-model
acceptance, then API/overlap performance. Passing 0.016 cannot excuse a failed
same-input hash. The implementation and ownership contract is
[union_halves_plan.md](union_halves_plan.md); no promotion is implied by source
plausibility or CPU checks. Existing valid reports remain retained.

## Shared-union exact operator acceptance

Run `same_input_all 32_gpu4_02` completed on GPU4/NUMA1 with register8 and eight
CPU threads. All 32 layers in both serial-half and async-half mode reproduced the
saved original Group8 output hash exactly. All 64 independent FP32 comparisons,
unique-byte checks and cleanup checks passed. All 32 async records passed both
original and candidate page/stripe identity and envelope audits. The coordinator
independently reopened every record, recomputed all 64 page/stripe audit pairs
and verified the 19 distinct mapped native/library files. Every call returned to
the stable 33,554,432-byte allocated reference-workspace baseline.

The summary is at the candidate stage's
`validation/same_input_all 32_gpu4_02/summary.json`, SHA256
`192ee52c5d0ebece87760583138aad06a24098673ef4c9069fc817e6138f6002`.
The 12-file source manifest is
`0c1b8a95db867d9e5a34747c0066719f19e83383c8dfed72b58dc1fb9b31892a`;
the native-source manifest is
`fd647aafb3a4d4256f911276090fe9ebd360b91a04b64f5a3437022e69fb4377`.
This is correctness evidence on captured operator inputs, not full-model or
performance acceptance. Its diagnostic overlap ratios remain below 0.90.

Attempt01 stopped before candidate execution because the Python CUDA binding
could not inspect a host stub registered with the selected shared CUDART. The
corrected inspector resolves APIs through the already loaded selected native
library and verifies their shared-CUDART mapping. No kernel source changed for
that helper fix. Full32 wrapper
`55340fab37d99290d48abb3c804bcb64ca3037b472600b6d13d96b29ecf1a13d`
also serializes execution and postcheck failures separately. The independent
fixture and numerical contract remain unchanged.

`full32_gpu4_01` then passed on GPU4/NUMA1. Its four eager references and 16 graph
requests cover two users/two visits, independent empty-prefix backends, real host
misses, retained-output lifetime, candidate discard and all five backend cleanup
checks. All 16 graph outputs are bitwise exact, including both sparse methods;
the fixed 0.016 tolerance was not needed. Actual dispatch observations include 128
Q128/halves2 calls and 4096 Q1024/halves1 calls for each sparse method. The
unchanged fixture uses one PyTorch intra-op thread while the environment and host
registration thread count are eight; this is correctness evidence, not timing.
Evidence SHA256:
`8f30cc3b94423cf029fc9973da2cd8237639e29a3e20bbc243a839048fd5dd8f`.

Root independently reopened this evidence and the frozen performance
prerequisites. GPU0/NUMA0 is allocated for
`nosa_q128_union_halves_perf_gpu0_20261004_01`: four 31-sample complete API
configurations and three separate internal samples per async variant. Prepared
harness manifest:
`644071d73bba17f47edef2aa519e9077ca51f5ba46f56933791b66cff88c76a3`.
Run `nosa_q128_union_halves_perf_gpu0_20261004_01` completed all ten processes
with valid numerical, provenance and cleanup checks. The coordinator reopened
all result files, checked their hashes and independently recomputed the decision.

| Complete API, 31 samples | Median CUDA ms | Median wall ms |
|---|---:|---:|
| Original serial Group8 | 0.644640 | 0.669534 |
| Original async Group8 | 0.798464 | 0.822284 |
| Matching serial halves | 0.706208 | 0.729741 |
| Async halves | 0.632576 | 0.669274 |

Async halves narrowly beat the original serial medians (1.01907× CUDA and
1.00039× wall). This observed wall difference is only about 0.260 microseconds;
it does not establish a robust latency advantage. Every candidate page-envelope
and stripe-copy overlap sample failed 0.90: 0.710582, 0.728228 and 0.695460.
The candidate is rejected for promotion. Decision SHA256:
`3b5935595cab6677bc94d065216871fc87fe53030e5a1d098843bffc3ba5e0c5`.
These are actual-input operator replays on GPU0, not serving measurements. The
staged numerical evidence remains useful for diagnosis; repository runtime and
valid experiment reports remain unchanged.

## Two-stripe ILP: partial correctness pass, early overlap rejection

The isolated implementation follows the frozen [stripe2_ilp_plan.md](stripe2_ilp_plan.md).
Both selected sm_90a builds passed source, actual-library/SASS and resource review.
Original halves1 SASS is byte-identical to the parent. Halves2 issues all eight
possible host vector loads before its first dependent payload store, has no
local-memory load/store sites, and uses 64/240 producer/consumer registers.
Its 38,912-register demand fits the actual 61,440-register CTA pool; configured
dynamic shared memory is 181,392 bytes. First-load and end-marker placement
preserves separate stripe windows and excludes the new minimum reduction.

`partial_gpu4_02` passed H=1,7,8,9,63 on GPU4/NUMA1, CPUs72-79. All ten original
and successor calls passed fixed FP32 checks, exact original-output comparison,
poisoned payload, unique bytes, page/stripe envelopes, ready counts and cleanup.
Each call returned to the stable 33,554,432-byte allocated reference baseline.
PyTorch intra-op, OMP/MKL/OpenBLAS and registration used eight threads; recorded
PyTorch inter-op remained its default 96. Summary SHA256:
`9f98ea3a5177a54e723c0bb8ea0f37c5800d50e708e449af5e871d87a56d6c21`.
Subsequent source review identifies a coverage limit: these synthetic selections
include the incomplete final logical block. The causal selection filter keeps
earlier unions active, while unions reaching that tail have zero prepared counts
and use repair. The old calls therefore mix fused computation and repair; their
checks did not require every union to remain active with all fallback flags zero.
The queue-priority successor adds separate complete-block selections with those
explicit requirements. The old ten calls are not described as repair-only.
Attempt01 failed in CPU provenance validation before any GPU case: its frozen
allocator identity omitted the venv PATH prefix used by execution. The corrected
header changes only that environment field and derived hashes; no native code,
build or binary changed. A separate pre-exec bootstrap lookup error also ran no
GPU work. Both are setup failures, not numerical failures.

The corrected screen freeze is `prelaunch_manifest_02.json`, SHA256
`30cc709513b570da610c5728c49cd0bf40573b86e5ee2a6fc68c17401368840b`.
Run `nosa_q128_stripe2_screen_gpu0_20261004_01` used GPU0/NUMA0 with full NUMA0
CPU affinity and register8. Original Group8, successor correctness and the first
internal sample all reproduced the saved layer16/request16 output hash exactly,
passed the fixed FP32 check and copied exactly 14,712,832 bytes. All three calls
passed payload, runtime-identity and cleanup checks. The 449 page envelopes and
3,592 nonempty stripe intervals have the same 387.328 us union; 240.416 us
intersects the unchanged softmax windows. Both ratios are 0.6207038995373431.

The screen rejected the candidate at `internal_1`, as required by the 0.90 gate.
It collected no later sample and establishes neither full-model acceptance nor
complete-API latency. Root independently reopened all records, recomputed the
physical envelopes and interval intersections, and verified 20 mapped libraries.
Result SHA256: `d2aaba5cd82f319b0f377081d0e1a9d10a649f0c14ff2548dedae7b4ee84266b`.
Evidence remains outside experiments under
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_stripe2_ilp_screen_20261004/output/nosa_q128_stripe2_screen_gpu0_20261004_01/`.
Read-only diagnosis and independent reconstruction of all 32 original unions
support testing earliest real pair-consumption priority on the row-half parent.
This rejected implementation is not integrated; no correctness or performance
acceptance transfers to the queue-priority successor.

## Earliest-pair queue priority: correctness pass, early rejection

The candidate follows [pair_priority_plan.md](pair_priority_plan.md), starting
from the one-stripe row-half parent. Only the bounded compactor, its ordering
after preparation and build metadata change. Both selected sm_90a fused-main
bodies match the actual parent libraries, preserving instruction offsets,
operands/control and machine words after whitespace normalization. The compactor
uses 26 registers, no stack/local spills, 16,384 bytes of declared shared keys and
17,408 bytes reported by cuobjdump/ELF. Main dynamic shared storage remains
197,728/181,344 bytes for halves1/2; halves2 retains 24/240 registers and a
33,792-register demand within the 61,440-register CTA pool.

Freeze SHA256:
`49dc548b275b852b9743b4e2f6026a29e3515a73c69d4d129252a27048caded5`.
Review SHA256:
`96905eb4020726fb05410aff37fde247f9f510179bd274502d11ccd6b45e7497`.
The final screen helper passed 141 CPU tests. These are preparation checks, not
GPU acceptance.

GPU4/NUMA1 run `partial_gpu4_01` passed 20 calls: five histories, two selection
modes and original/successor kernels. Complete-block selections execute all 32
unions without fallback; mixed selections have 30, 30, 30, 28, 16 active unions at
H=1,7,8,9,63. All output pairs are bitwise equal, all fixed FP32/payload/queue/
byte checks pass, and every call returns to the 33,554,432-byte allocated reference
baseline. Root and HBM-audit independently checked 882 actual softmax intervals,
40 page and 104 stripe records, expected physical work IDs, ready counts, runtime
resources and 20 mapped libraries. Summary SHA256:
`db6f4c804269a5df2e17f2996fdb3cfc0be00060bdcf183b6207dcda34c74369`.
Evidence is under the native stage's `output/partial_gpu4_01/`; it is synthetic
partial correctness, not full-model acceptance.

GPU0/NUMA0 run `nosa_q128_pair_priority_screen_gpu0_20261004_01` passed original
Group8, successor correctness and the first internal sample's numerical/payload/
queue checks. Actual priority order matches the independently reconstructed 449
miss pages (canonical slot-list SHA256
`681dbdc30cf512bad15e6bd8cb95da3acf1a274923129084bff47661cb3d1bac`).
Every call copies 14,712,832 bytes and passes cleanup/identity checks. The internal
sample contains 449 pages, 3,592 stripes and 3,030 actual softmax intervals. Page
and stripe unions both equal 367.968 us; their intersection with softmax is 308.928
us, giving 0.8395512653274197. The helper correctly rejects at `internal_1` and
collects no later sample. Result SHA256:
`38f294f8d916890bd84ae0fde1abc81be3a2afd16c5c4bdc0f472c8068d28013`.
Root independently recomputed queue order, physical intervals and ratios; the
separate HBM audit agrees. The result and `root_audit.json` are under
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_pair_priority_screen_20261004/output/nosa_q128_pair_priority_screen_gpu0_20261004_01/data/`.

The 59.040 us uncovered copy consists of 17.344 us before the first softmax,
41.696 us in 103 internal gaps, and no copy tail after the final softmax. The
largest internal gap is 3.680 us. Softmax union is 325.568 us; with these observed
union lengths held fixed, even perfect placement could cover only 88.4773% of the
copy union. This is a bound on this trace, not a proof that 0.90 is impossible for
another implementation. Progress requires changing actual copy or consumer
timing/duty. No window reinterpretation is permitted. Copy end is still distinct
from readiness, including the producer's next-K-before-current-V dependency.
No complete API latency was measured, so this screen is not an end-to-end
speedup result. The implementation is rejected and remains isolated.

The completed residual audit identifies shared dependencies within equal-rank
buckets as a bounded follow-up hypothesis. Root authorized a rank/fanout/ordinal
tie rule with no weights, extra HBM workspace or additional launches. It retains
the parent mainloop and16KiB shared key array. A second reference pass counts
original unions at the minimum rank, without duplicating compute halves. This
may delay singleton dependencies and does not predict a0.90 pass. Fresh source,
packed-field concurrency, correctness and performance gates remain mandatory.

## Frontier fanout: static and partial acceptance

The isolated successor passed 93 native CPU tests and 147 helper tests. Independent
review checked the concurrent packed fields and both actual selected images.
Both fused-main raw ELF text sections and normalized SASS match the immediate
parent. The compactor remains REG26, STACK0, LOCAL0, with 16,384 declared key bytes
and 17,408 reported shared bytes. Fresh dynamic shared sizes are 197,728/181,344
bytes for halves1/2; their register demands 64,512/33,792 fit the actual CTA pools
64,512/61,440. These are compile/resource facts, not timing results.

Freeze SHA256 is
`056162ab670e2ef9728e13b659da7b2c0850951e5abfbab5754adacd6d36ccbe`;
prelaunch review SHA256 is
`e8c32bd743a9fdfdf20ea2aa973592fa6e8b8033173c00c1ebbc9f67d7422e8e`.
Compilation, freezing and execution all use the same canonical one-prefix venv
PATH and production private allocator fingerprint. Native source manifest SHA256:
`4fd7aeedb32b722e0824f8069285596d639f4682886a42029e54ace8a3d4de36`.

Fresh `partial_gpu4_01` passed all 20 calls on GPU4/NUMA1, CPUs72–79, with eight
intra-op and pinned-registration threads. Inter-op remains the recorded default
96. H=1,7,8,9,63 each covers original/successor and all_blocks/complete_blocks.
All output pairs are exact; fixed FP32, poisoned payload, unique bytes, queue,
readiness and cleanup pass. Root independently reopened all 882 softmax records,
40 pages,104 nonempty stripes and 20 mapped library identities. Mixed selections
have 30/30/30/28/16 active unions; complete selections always have 32 and no
fallback. Every call returns to 33,554,432 allocated and reserved bytes.

Partial summary SHA256:
`635099d7c93dce6c5b1d27e2e45a74a28c141c82e90032b08dadb18b161366ea`.
Independent partial review SHA256:
`8a13bd5b5ac81d482e6d4edba2945208c8eec0019cbefb67894de7ea0f13583d`.
Raw prepared-page arrays were checked by the frozen partial helper but are not
retained; actual counts and physical work records are retained. These partial
checks have no overlap threshold and establish no serving performance.

Root allocated GPU0/NUMA0 for
`nosa_q128_frontier_fanout_screen_gpu0_20261004_01`, with full NUMA0 affinity,
eight intra-op/OMP/MKL/OpenBLAS/registration threads and the same allocator.
The screen completed original Group8 and successor correctness, then stopped at
`internal_1`: page and stripe unions were both 374.144 us, softmax union 328.320 us,
and intersection 310.816 us. Both ratios were 0.8307389668149162, below 0.90.
The actual 449-page queue matched
`dbdefd7fd8c5fcda4f856559675e3820aac3770f7230af4229e4a6a9859477a3`.
Exact outputs, fixed FP32, poisoned payload, unique bytes and cleanup passed.
Frontier fanout is rejected; no further samples, all-layer/full-model checks or
API latency runs followed, and the staged runtime remains unintegrated.

Screen result SHA256:
`729f27dacb3f2991736e332835028019a14ea568c18c5ba2763d19f88cf02477`.
Root independently reopened all three calls, 449 pages,3,592 stripes,3,030 softmax
rows and 20 mapped libraries and recomputed both ratios. This single diagnostic
does not establish a statistically meaningful speed difference from the prior
pair-priority sample.


## Independent producer-warps: static freeze

The candidate passed 150 native CPU checks and 158 screen checks. Independent
source review covered 69,188 assertions and 320 interleavings. Actual halves1
SASS and ELF text match the parent; halves2 changes are reviewed at instruction
level. Compiler scheduling changed some unaffected math instructions, so only
math source identity is claimed. Main stack is 32 bytes; the halves2 fetch region
contains two LDL and two STL instructions. This is not a spill-free main kernel.
Actual CTA register pools/demands are 64512/64512 and 61440/33792; dynamic shared
storage is 197728/181344 bytes. The independent writer-fence, full-warp collective,
marker and acq_rel publication chain passed compiled review.

Fresh screen freeze SHA256:
`d77b40a6043bfa18231aa1e640bb2bb2fee27f173dd014dc7041febf1d5c06c1`.
It binds native source manifest `69aff0326d295a7c22902759a2ccf8996b0fba56841ff280c71ef2d809b67ed1`,
prelaunch review `685c20a3f73174d9877ab284ede496188e439d65654c3c27f969751a45591639`,
and both explicit independent source/instruction artifacts. Freeze ran with CUDA
hidden and one canonical venv PATH prefix. No GPU numerical or overlap acceptance
is implied by this static milestone.


## Independent producer-warps: partial acceptance and screen rejection

Fresh partial20 completed on GPU4, CPUs72–79/NUMA1, register8. All ten original/
successor output and reference digest pairs match, fixed FP32 tolerances remain
0.016 with zero failed elements, and every call returns allocated and reserved
storage to its 33,554,432-byte reference-vendor baseline. Root, experiment and
HBM audit independently checked 40 pages, 104 nonempty stripes, 882 softmax rows,
unique bytes, queues/readiness, selected libraries and cleanup. Summary SHA256:
`643653035c5bf0e2265ac5fadc538d0011a8ac67eaf34d751f2561a345b89aef`.
HBM audit SHA256: `f197e56a3b27f2dffe3e54c2aaa70f36804a8db6b1a2c214b6690d330b8b31b7`.

Run `nosa_q128_independent_warps_screen_gpu0_20261004_01` then used GPU0 with full
NUMA0 affinity and unchanged allocator/thread/GC policy. Original and successor
representative correctness passed. First internal sample copied 14,712,832 bytes
over 449 pages / 3592 nonempty stripes and recorded 3030 actual softmax rows.
Both ratios are 0.8005504587155964. Copy union is 348.8 us, softmax union 298.144 us,
and intersection 279.232 us. Root's interval recomputation gives 69.568 us uncovered:
20.928 us startup, 48.64 us internal and zero tail. A shorter copy union than an
older single sample is not a statistical latency comparison or API benefit.

The screen exited at its first failed gate and final integrity passed. Result
SHA256: `d1cd5615218a0ce1e5d27767ca18378341bdd331920d22580da32fab92f9aaac`.
Raw data and root audit remain in the isolated screen output directory. The
candidate is rejected; it is not integrated or an experiment deliverable.

## Four-pair host ILP: partial acceptance and screen rejection

The [isolated candidate](four_pair_ilp_plan.md) passed the 3,120-case CPU physical
address audit, twelve focused helper tests, exact-loader compilation and
independent selected-source/SASS/resource review. Halves1 and the priority
compactor are unchanged. Halves2 issues eight host vector loads before dependent
stores with producer80/consumer240; its selected-main stack changes 32→0, so
producer-budget/stack effects remain confounded with ILP.

Root's `nosa_q128_four_pair_ilp_partial_gpu4_20261004_01` passed all twenty calls,
with independent raw and lifetime/native acceptance. The representative
`nosa_q128_four_pair_ilp_screen_gpu0_20261004_01` then passed both exact controls
and rejected `internal_1` at 0.7027860967419911 for both overlap metrics. All three
calls passed numerical, payload, original-residue16 host, native and cleanup
checks; final integrity passed. The screen exited 2 as intended. No later sample,
API latency, all32 or full32 candidate run followed. No Q128 native successor is
selected for integration. Detailed receipt identities remain in the candidate
plan; frozen stages and production native sources are unchanged.
