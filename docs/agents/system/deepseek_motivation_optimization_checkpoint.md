# Motivation optimization checkpoint

## Goal status

On 2026-10-04 the user selected C10 as the final optimization revision and
requested closure. No further candidate promotion is planned for this round.
C10's first-visit gap is 1.88%–4.89%, with first-visit E2E MFU of
42.37%–44.44%. Its formal trajectory and independent output/runtime/arithmetic
audits have passed. Matching profile `motivation_c10_profile_20261004_01` and
fresh official `echo_official_c10_20261004_u16_r2_01` formal/repeat have now
passed independent numerical, source/runtime and arithmetic checks. The profile
also passed raw-SQL graph attribution. Coordinated report replacement and
scoped cleanup are complete: the two public reports now use C10, with all
final raw data, logs, profiles and source snapshots retained. The original aspiration of E2E MFU
matching matrix API MFU is not claimed achieved. Pending C11 production changes
were archived and restored to C10; R1 remains temporary research.
This completes the user-requested closure at C10. See the
[C10 closure record](deepseek_motivation_c10_closure.md).

## Current candidate: C10 token validation and rotary output

C10's opt-in CPU token predicate passed 244 semantic cases, 62 selected-mode
existing tests and 187 default/NOSA/helper CPU checks. Its two production
component timing windows saved about 1.62 ms per H64K+A128 validation. Default
and fallback validation stay exactly C9; motivation and official explicitly
opt in and record the loaded extension identity. These are not E2E gains.

The direct main-MLA rotary destination passed 26 production tests and all six
projection tensors in nine real-layer/Q cases against frozen C9. Paired graph
GPU savings are 7.648–7.744 us at Q128 and 39.920–40.512 us at Q1024; one eager
group is effectively flat. The same vendor kernel and indexer rotation remain.
The combined four-scheme H64K gate passed 221 tests with no skips.
Formal run `motivation_c10_20261004_u16_r2_01` completed with exit 0;
session 21984 and monitor 26372 have ended. All 1,594 data files and two logs
were relocated with identical hashes. Independent audit verified 128 output
payloads, 96 offload/HBM comparisons and 32 C10/C9 HBM comparisons, all byte-exact;
104,960 graph replays, all 256 memory snapshots, native CPU identity and runtime
Triton artifacts also passed. First means are HBM 2176.886, ECHO 2283.433,
serial 2217.737 and dense 2233.838 ms; revisit means are 2170.244, 24.947,
18.456 and 31.889 ms. The complete change does not isolate component gains.

Frozen C10 root: `/tmp/deepseek-motivation-c10_host_rope_v2-frozen-6mdyfgj0`,
503 files, map SHA-256
`6d9718b6177800112683e721650e6ae99081c71948a75555580006d0e457ffbe`.
The earlier freeze was replaced to fix capture-time wrapping of the selected
validator instance and add its regression. Executed-source SHA is
`11fc11b18b2e70baf450a82ab4fad66f2f8d4e22cda2e9f36bf74453a400df8f`.
See [C10 plan](deepseek_motivation_c10_execution_plan.md). The first Triton
top-k finalizer is rejected: all six complete APIs slowed down despite exact
outputs. T2 finite-tie repair passed GPU correctness but regressed three of
six complete API timing cases, including large causal and revisit shapes;
it is also rejected. T3 and cleanup audit V2 passed their component gates but
were deferred when the user selected C10 for closure. Their pending production
edits were archived outside experiments and restored to C10; neither is part
of the final implementation.
C10 is now the latest accepted formal/profile pair; its first-visit matrix API
MFU is 53.38%–56.18%, distinct from formal E2E MFU. See the
[matching profile note](deepseek_motivation_c10_profile.md).

## Previous accepted formal/profile pair: C9 handwritten Triton

`motivation_c9_triton_20261004_u16_r2_01` completed with all 128 outputs exact,
including 96 offload/HBM comparisons and 32 C9/C7 HBM comparisons. Independent
source/runtime, lifecycle, allocation, replay and arithmetic audits passed.
First means are HBM 2203.859, ECHO 2306.008, serial 2245.959 and dense 2250.105
ms. Revisit means are 2194.362, 26.873, 20.555 and 34.022 ms respectively.
See [C9 integration](deepseek_motivation_c9_integration.md) for complete bounds.

The full four-scheme profile `motivation_c9_triton_profile_20261004_01` passed
numerical, source/runtime, direct-SQL attribution and aggregate arithmetic
audits: 80 exact outputs, 12 captures, 45,933 matrix calls, 274,122 physical
GPU activities and 6,560 measured graph replays. All 1,595 files were relocated
with matching hashes. Source SHA-256 is
`13e87328311d3b7ce2df2cde230f1e612f61c5c2502ebe39dad811d3b19f2996`.
The 200 attached discrete process observations span the profile and contain
only its Python GPU PID. One saved py-spy stack observes the unprofiled prepare
branch; it is not a warmup or measured-capture sample.

Exec session 97660 and root monitor 76308 exited. The subsequent official
GPU gate passed all three cases, including eager and graph H64K mirrors; its
CLI/report enablement passed 107 CPU checks. Those GPU results precede C10;
changed production paths receive new validation and measurements.

## Previous accepted formal run: C7a plus exact ECHO hint

`motivation_c7_hint_20261004_u16_r2_01` completed; exec 25953 / PID1810918
exited 0. Frozen root is `/tmp/deepseek-motivation-c7_hint-frozen-a44l_2uk`.
Executed-source SHA-256:
`cc141544c664dfe0e99119d7889b1fa6d1f2728062995ab564b279dca7285652`.
Freeze-map SHA-256:
`d08a95eced7c8923945699384f84785ff29d48d3ba63ab8eb5ee9d152736fba3`.
All 1577 data/log files were relocated with matching hashes. Full source,
output, lifecycle, capacity and timing audits passed, including 128 payloads,
96 byte-exact offload/HBM comparisons and all 32 C6/C7 HBM payloads. The separate
saved-data arithmetic audit reproduces all eight report groups and 24 MFU
stages. See [C7 integration](deepseek_motivation_c7_hint_integration.md).

| Scheme | First mean ms | Revisit mean ms | First E2E MFU % | Revisit E2E MFU % |
|---|---:|---:|---:|---:|
| hbm | 2320.410 | 2328.411 | 41.6912 | 41.5479 |
| echo | 2423.584 | 26.867 | 39.9164 | 8.3612 |
| serial_sparse | 2358.644 | 20.684 | 41.0154 | 10.8607 |
| dense_prefetch | 2375.511 | 34.537 | 40.7242 | 6.5043 |

Compared with C6, ECHO first-visit mean improves 19.415 ms and dense revisit
mean improves 5.725 ms. HBM and serial change little. The combined trace does
not isolate each component's contribution. Formal hold has ended. All 100
attached GPU monitor samples show no unexpected PID; sampling spans the run
but does not continuously observe gaps between samples.

C7a and hint passed all 176 combined GPU checks without failures/skips and
all 109 source identities remained unchanged. Global CPU regression passed
2764 tests, 947 explicit optional/hardware skips and 58 subtests. Details are in
[combined validation](deepseek_motivation_c7_hint_validation.md).

## C9 integration provenance

The user switched to handwritten Triton on 2026-10-04 and retained the request
to avoid production `torch.compile`. Native V4 remains offline-only; it has no
GPU correctness or timing result and will not be integrated. Continue with the
[direct Triton plan](../kda/deepseek_linear_quantization/triton_implementation_plan.md)
under the same exactness and complete-API promotion gates. Existing native
V1/V2/V3 results remain component engineering evidence.

Handwritten Triton T1 is selected for integration. Its explicit-toolchain
`screen_03` passed all 213 fixtures, eight lifecycle cases, 16 graphs and
24 paired replays; both independent audits also checked the 300 observed
specializations. Screen SHA-256:
`ccbd8b282bdcdc415e3ee90ee8fefac2627ffde8ad65291d97abfeb743638ea3`.
Two quiet 16-shape timing windows each passed 4,800 pairs, using different
order seeds. Ordinary allocating API speedup was 1.47–1.97x in the confirmation;
complete owned-output Graph timing was preserved (0.986–1.030x), with about
3% improvement at M1024/K16384 and K18432. Small-shape Graph speedups are not
established. `bench_02` and `bench_03` SHA-256 values are respectively
`3a6c4ea29417817af5f83a39e246a27957023ab4573b7f03503e7676198165d6`
and `e572a7d1060790b4b1ff276a46a0248b0179e70d95b8556c85cfe7e8048fd46c`.
Evidence lives under `/tmp/deepseek_linear_triton_quantization_v1_20261004/`.

Both timed APIs explicitly used Triton's bundled PTXAS, SHA-256
`c960a4f238b17d5c5d3c01ad2bbc1ebd2c5aecc459cb4d223bff10b45f9b8fca`.
The original API's default TorchInductor compiler is a different binary;
its environment mutation interrupted the first timing attempt. The accepted
comparison uses a shared compiler and is distinct from historical C8 default
toolchain timing. No numerical mismatch caused that interruption.

Production integration passed the 34 public quantizer tests, exact small C8
cross-version checkpoint gate, 37 dense/grouped/packed MLP tests, nine actual
checkpoint MLP cases and 176 four-scheme H64K checks. Global CPU regression
passed 2,850 tests with 1,022 explicit hardware/optional skips and 58 subtests.
The [C9 validation record](deepseek_motivation_c9_validation.md) binds source,
compiler, live runtime artifacts and graph memory evidence. The C9 formal
trajectory is accepted as recorded above. Its frozen root is
`/tmp/deepseek-motivation-c9_triton-frozen-0496ye32`; matching four-scheme
matrix profiling is accepted. SFA and projection sharing remain unintegrated.

C8 norm and packed/shared MLP production correctness passed. Its formal run
is deferred under the user's new preference to avoid `torch.compile`; KDA will
replace the remaining linear activation quantizer before the next promoted
full measurement. See the [C8 plan](deepseek_motivation_c8_integration_plan.md).

The accepted C8 integration is frozen as a baseline at
`/tmp/deepseek-motivation-c8_torch_reference-frozen-rb0_2eqx`, 483 files,
freeze-map SHA-256
`cae2da8b9a58907ae35f99b5eeda017348b1a0a1ce1e1421fd6ca7703c956bb6`.
It has no formal latency or MFU result. Its evidence includes:

- Norm: 71 production GPU tests, four source-bound live IR specializations,
  and independent saved-data audits. Component timing remains the earlier
  32-group/3840-sample/64-trace result. The temporary production wrapper's
  final report-key typo and exit1 are preserved; corrected CPU audits establish
  acceptance of the already completed tests/IR records.
- MLP: 37 production GPU tests without skips and nine actual-checkpoint
  layer/Q cases with exact full-down results, streams, retained outputs and
  changed-input graph replays. The first heterogeneous test run exhausted
  Dynamo's code-cache limit. Per-case compiler reset fixed test isolation;
  production code/limits and within-case replay semantics were unchanged.
- Combined: 176 four-scheme H64K tests passed in 92.39s; all 119 source paths
  and native/norm identities matched. Each scheme's graph-private reserved
  storage was 5,771,362,304 bytes, within the existing plan. This is a dual
  eager/graph correctness process, not a full-NH or process-peak experiment.
- Global CPU: 2778 passed, 1005 explicit optional/hardware skips and 58
  subtests. Another 100 motivation/runtime-report tests passed. Subsequent
  test-isolation-only edits passed their focused CPU/GPU checks.

C7b capped gather remains a separate unselected prototype. Two independent
1500-sample sweeps confirm about 2% component benefit for cap264 gather-first;
serial and isolated gather controls regress. Its 42-case profile and root
raw-SQLite audit passed, preserving all 3612 physical activities. Whole-kernel
intersections do not establish internal useful-work overlap or uncached PCIe
bandwidth. Both sweeps/profile use frozen C7_hint model source and a C6 request
fixture; no production transport change is selected.

All GPU sessions above have exited and released their devices. No other CUDA
job may overlap new performance measurements. Existing published reports and
backing data remain. After optimization, the updated user goal also requires the
[official ECHO comparison](deepseek_echo_official_optimization_followup.md).

### Native activation quantization candidate

The temporary CUDA C++ `warp1_cta4` implementation passed `screen_01`: all 115
fixture FP8/scale byte comparisons, backing/output guards and six stream/lifetime
cases (12 graphs, 18 paired changed-input replays). Root independently rehashed
862 source/build/oracle paths. The actual imported DeepGEMM helper equals the
pinned source. Generated oracle PTX and native PTX preserve the reciprocal-448
constant, full FP32 reciprocal and SATFINITE E4M3 conversion without FTZ.

The first quiet timing window `bench_01` completed on physical GPU2, with
16 BF16 shapes, 10 warmups and 50 repetitions: 4800 paired observations across
eager allocating, graph copy/replay/owned-output and borrowed replay diagnostic
boundaries, using both wall and event timers. Pre/post numerical and identity
checks passed. The scalar candidate is rejected for promotion: at
Q1024/K18432 its graph-owned event median was 84.016 us versus 54.592 us for
the frozen C8 public API; borrowed replay was 47.552 us versus 19.008 us.
Eager host overhead improvements do not override this graph regression.

Evidence is under `/tmp/deepseek_linear_native_quantization_20261004/`.
`screen_01.json` SHA-256 is
`553afa3484e7a0f27dfff8ac28dbfffd12a9fa47a20cc7aac7390b3c15f26442`.
Source and dependency identity remained unchanged through both GPU runs.
V2 added aligned BF16 vector loads/packed stores and four groups per warp,
removing device-side row/group division. V3 replaced only its aligned maximum
with unsigned absolute-float-bit reduction and native warp REDUX. Both passed
213 fixtures (63 fast cases) plus eight lifecycle cases, with 16 graphs and
24 paired changed-input replays; the generic fallback stayed unchanged.
Both also completed separate quiet 16-shape/4800-pair benchmarks. V2's
Q1024/K18432 owned-graph event median was 65.776 versus 56.000 us for its
contemporaneous original helper; V3 reached 60.640 versus 56.144 us. V3's
borrowed replay was 22.736 versus 20.592 us. Neither meets promotion criteria.
Their frozen engineering roots are the corresponding `_v2_20261004` and
`_v3_20261004` variants of the temporary quantizer directory.

The deferred V4 proposal would replace the aligned kernel's power-of-two
reciprocal with exponent-bit construction, after a direct GPU witness against
`div.full` for every scale exponent from 105 through 254. It never reached GPU
validation or timing. The user then switched to Triton; V4 and the native loader
below are inactive engineering prototypes. Production now dispatches the T1
Triton quantizer, with integration acceptance tracked above.

The isolated production-loader prototype passed 11 CPU checks, including real
C++/TVM FFI load, unchanged parent environment, source/dependency identity and
concurrent build/load behavior. It builds under a fixed SM90 child environment
and observes loaded artifacts separately from stable build identity. This does
not constitute CUDA validation or native-quantizer promotion.

The official runner now accepts optional common compute callbacks; CPU dispatch
checks pass, but its experiment still rejects graphs pending actual checkpoint
validation. These later official-source edits are outside the frozen C8
acceptance; no new official performance result is claimed.

## C6 evidence supporting current candidates

C6 full quiet run `motivation_c6_20261004_u16_r2_01` passed all independent
checks. Its source SHA-256 is
`bd3e91265fed36f182d291b1bea3a659a2214c686e8236ed42da663c07775bff`;
see [C6 integration](deepseek_motivation_c6_integration.md).

The dense diagnostic `motivation_c6_dense_profile_20261004_01` completed and
passed independent output, attribution and stream-event audits. Its 63688
physical GPU activities include revisit gathers totaling 14.708 ms with zero
matrix overlap. All nine preceding projection graph host calls finish before
the associated gather ends, yet their matrix work starts afterward. This is
scheduling evidence, not a proven hardware cause. See the
[profile checkpoint](deepseek_motivation_graph_profile_checkpoint.md).

C7a full-miss helper median improved 2.775675 to 1.929521 ms, half-miss
2.052901 to 1.203246 ms and uncertified hit 0.454114 to 0.184507 ms.
Certified hit is effectively unchanged. Private M-sized ticket buffers survive
next-layer scratch reuse; delayed-copy and four Event-failure cases passed.
The exact hint helper passed 186 prototype cases, 43 production tests and
18 existing top-k checks. Q4/N65536 complete production API improved
0.093168 to 0.067551 ms, preserving the original FP32 torch.sum order and
all hint bits. Both components are included in the accepted C7 trace.

## Previous accepted trace: C4

Run `motivation_c4_20261004_u16_r2_02` completed successfully (exec 21202).
H65536/A128/C1024/P65536/NH16777216, 16 users and two rounds, ten independent
checkpoint dense-block copies, full candidate hidden and final logits.
The frozen source is `/tmp/deepseek-motivation-c4-frozen-48yhnrcs`;
executed-source manifest SHA256 is
`88796ef27ddbbd5b0b6e969552b9d88d6ed9feb581cc8520bd9e28c9aa2dfa75`.
Data/logs are in the original experiment output tree. Independent report audit
passed all 128 requests; all 32 HBM outputs and input identities also match C3
bitwise. FLOPs/E2E analysis is `output/data/motivation_c4_flops_20261004_02`.

| Scheme | First mean ms | Revisit mean ms | First E2E MFU % | Revisit E2E MFU % |
|---|---:|---:|---:|---:|
| hbm | 2415.863 | 2422.401 | 40.0439 | 39.9359 |
| echo | 2532.248 | 39.909 | 38.2035 | 5.6288 |
| serial_sparse | 2453.739 | 33.398 | 39.4258 | 6.7262 |
| dense_prefetch | 2470.470 | 40.330 | 39.1588 | 5.5700 |

MFU uses total ideal compute time divided by total E2E time for each group;
use only the `e2e` rows, not sums across the overlapping prefix/extend/e2e rows.
C4 has not received new full graph-node API profiling yet. C3 API values below
are diagnostic history, not C4 operator MFU.

C4 combines aligned suffix-only causal masking, owned packed token input,
one-query-per-storage accounting and guarded completed-transient truncate.
The four-scheme real-checkpoint gate passed including H=P65536/C1024/A128,
A121 eager fallback, source ownership and completion-failure handling; see
[combined validation](deepseek_motivation_combined_validation.md).
No other delegated CUDA jobs ran during the full trace. Attached monitoring
covers 21 discrete samples beginning 235.238 seconds after metadata run start,
with no unexpected PID; it is not continuous monitoring of the initial phase.

The initial suffix01 launch (exec 77811) failed before CUDA because the isolated
root omitted .gitmodules/root environment files. These were copied and hashed;
full source-snapshot CPU precheck passed before suffix02. Failed logs remain
outside experiments at
`/tmp/deepseek-motivation-motivation_c4_20261004_u16_r2_01-logs.je3FZ0`.

## Work in progress after C4

- C5 top-k wrapper: retains official exact sorted SMALL selection, directly
  consumes its int32 output and uses one in-place nonfinite mask. Component
  accepted: 18 GPU tests, six exact benchmark patterns, launches 11 to 4.
  Q1024/N65536 complete API median 0.466106 to 0.434206 ms; Q128/N65664
  0.103387 to 0.090159 ms. Current selection source SHA256 is
  `0e68ecb4e16e40e9dda1541693ce3e00e41ccd4a05ea2e708d04dd2545a4dcd8`.
  [Top-k checkpoint](../kda/deepseek_topk_wrapper/checkpoint.md).
- C5 direct output BMM layout: exact real-weight and full callback probes
  passed; it is included in the current C6 source and combined validation.
  [Output layout plan](deepseek_output_layout_plan.md).
- C6 miss-path native metadata passed 272 GPU suite checks and five exact
  full-state benchmark patterns. Q128/H=P65536 all-miss complete API median
  is 3.464231 to 1.834823 ms; kernels decrease from 133 to 21. Root's
  20,000-case pure CPU ordering check and independent lifecycle review also
  passed. See [sparse recall checkpoint](../kda/deepseek_sparse_recall/checkpoint.md).
  Preserve victim tombstones before overwrite and publish new maps only
  after generic gather. One miss-count synchronization retains exact Python
  map-generation/append-proof behavior; no speculative invalidation shortcut.
- C6 graph boundary: accepted component probe is integrated as graph policy
  v2, with eager expansion BMM into smaller static input and borrowed saved
  residual. Full combined GPU/lifetime checks and the C6 formal trace passed.
  [Graph boundary plan](deepseek_motivation_graph_boundary_plan.md).
- Vendor Triton mixed-dtype norm candidate rejected by strict FP32 identity
  on its first [9,7168] case, despite matching final BF16 there. No timing,
  full sweep or production change. Evidence is under
  `/tmp/deepseek_norm_probe_20261004`. Both direct vendor CuTe mixed-tensor
  attempts also failed compilation: unchanged constructor rejected incompatible
  arithmetic layouts, and an instance-only copy-width change rejected the copy
  predicate layout. FP32 controls were exact, but no mixed-tensor output or
  performance result was produced. A local adapter design is under CPU review.

All newer candidates are distinct from the frozen C4 measurement. Do not
publish C4 numbers as their results. Schedule GPU timings sequentially and
poll existing handles before restarting jobs.

## C3 diagnostic evidence

C3 full graph profile `motivation_c3_profile_20261004_01` completed and passed
independent attribution: 45,933 matrix API calls, 357,548 total GPU activities,
6,560 graph replays. Its first sampled matrix API MFU is
50.85/52.94/51.35/51.28% for HBM/ECHO/serial/dense, versus its formal E2E
38.29/36.77/37.63/37.57%. The timing denominators are distinct. Full evidence:
[graph profile checkpoint](deepseek_motivation_graph_profile_checkpoint.md).

C3 source/trace details remain in [C3 integration](deepseek_motivation_c3_integration.md).
C3's formal run contained concurrent CUDA validation and retained outliers;
C4 is the subsequent quiet trace. Do not assert an unproven cause for C3 outliers.
Leading measured C3 costs were about 206–220 ms exact top-k, 55 ms explicit
compute-graph input copies, 270–274 ms graph nonmatrix nodes and about 137 ms
of resident indexer causal mask work (the latter is inside its API scope).
C4 addresses the causal mask. Do not double-count API helpers or subtract
instrumented GPU/CPU sums directly from formal wall latency.

## C1/C2 implementation

- C1: conservative per-layer full-residency proof; one stable FIFO append order
  consumed during uninterrupted cold construction; exact old eviction ordering;
  all-hit ECHO resident-indexer dispatch; dense miss-scan/copy/event elision;
  host waits only before reads; same-stream dependency elision; cached host-page
  runs and internally validated logical ranges.
- C2: direct single-chunk block outputs and BMM into final query layout; remove
  duplicate GPU causal mask; compute the ECHO offset hint only for ECHO offload,
  including all-hit calls so the hint survives into a later miss revisit.
- Shared storage added by C1 is 8*P*layers (5 MiB here), represented in both pool
  and offline component planners. P and NH themselves remain unchanged.
- Formal runner now records disabled TF32 and verifies it stays unchanged.
  Static FLOPs distinguish dynamic indexer dispatch from known fused padding.
  `src.aggregate_mfu` supplies precision-weighted aggregate API comparisons.

## Accepted full-shape screening

Run: `motivation_opt_screen_20261004_01`.
Source manifest SHA256:
`65608fb77f6cc4b01b43b122effc0670d902fceb7471e625bd4f29beb8e507b7`.
Data: `experiments/deepseek_v32_motivation/output/data/motivation_opt_screen_20261004_01/`.
FLOPs analysis: `output/data/motivation_opt_flops_screen_20261004_01/` in that experiment.

H200/SM90 GPU0, H=65536, A=128, C=1024, P=65536, NH=16777216,
ten independent dense blocks, seed42. Two users and two rounds; three warmup
requests per scheme, then four measured requests per scheme. This is a screening
run, not the required sixteen-user final publication. Sample means are:

| Scheme | First E2E ms | First history ms | First candidate ms | Revisit E2E ms | First E2E MFU % |
|---|---:|---:|---:|---:|---:|
| hbm | 2520.277 | 2480.932 | 29.353 | 2523.444 | 38.385 |
| echo | 3485.231 | 3429.816 | 45.330 | 64.853 | 27.757 |
| serial_sparse | 3758.389 | 3703.888 | 44.158 | 57.007 | 25.740 |
| dense_prefetch | 4352.031 | 4065.934 | 276.100 | 67.019 | 22.229 |

The two samples expose substantial variability (serial first requests 4116.055
and 3400.723 ms; dense 4119.997 and 4584.066 ms). The unusually large dense
first-candidate mean requires investigation, not deletion. Two-user revisits
have different retained token residency from the original sixteen-user trace;
do not present them as a direct final comparison against old revisit means.

Run audit passed: 16 request checks, all 12 offload hidden/logit comparisons
bit-exact, 12 warmup checks including host recall, no candidate D2H, source and
installed backend identities verified. The first two HBM inputs have identical
saved input hashes to the original `_02` run, and their full hidden/logits are
also bit-exact across versions. No new per-operator MFU is claimed yet.

## Correctness and review

- Actual Hopper indexer/selection tests: 24 passed.
- C1 cache/pool/transient/dense tests passed, including CUDA stream ordering.
  Original-allocator parity, other-user interleaving, snapshot/truncate, reused
  pages and fragmented writeback are covered.
- Independent review ran 20 randomized traces, five users/three appends each,
  including forced clock rollover; maps, priorities, counters and clocks matched.
- Real checkpoint fixed-pool test passed: four schemes, ten independent blocks,
  2304-token histories, alternating users, A=16/23, exact outputs and retained
  history, allocation/accounting checks.
- C2 model/block tests: 30 passed; 54 checkpoint projection tensors exact. More
  geometry and graph-only diagnostic evidence is in `deepseek_motivation_c2_checkpoint.md`.
- Global CPU run: 2692 passed, 846 explicit GPU/optional skips, two failures.
  Those revealed a stale test assertion and the separate capacity planner's
  missing append-order bytes. Both are fixed; their focused suites then passed
  all 77 tests. Motivation tests passed all 36. Ruff passed on changed files.

## Completed C1/C2 profiling / next actions

Dense C1/C2 Nsight profiling completed successfully: two captures, six exact
output checks, exported SQLite, and completed pipeline analysis. The source
SHA256 is `40ab02f75af037f68c335230e5b10eff8211bf8323f7a465aadbe4412b9ab069`.
Sessions `81317` and `58035` have completed; do not restart them.

- Exec session: `81317`.
- Script PID at launch: `1612265`; profiler `1612276`; Python `1612376`.
- Run ID: `motivation_opt_profile_screen_20261004_01`.
- Reference: accepted screen above; scheme `dense_prefetch` (two captures).
- Staging: `/tmp/deepseek-motivation-profile-motivation_opt_profile_screen_20261004_01.JzOCCi`.
- Command: `CUDA_VISIBLE_DEVICES=0 bash experiments/deepseek_v32_motivation/scripts/profile.sh --run-id motivation_opt_profile_screen_20261004_01 --reference-run experiments/deepseek_v32_motivation/output/data/motivation_opt_screen_20261004_01 --scheme dense_prefetch`.

Analysis in the run's `analysis/` directory conserves activity counts and times,
with zero unattributed GPU activities. Instrumented cold history has 2604.014 ms
GPU activity and 2928.171 ms gaps. The cold request has 12,356 stream
synchronizations, compared with 24,063 in the original dense capture. Cold-history
CPU-exclusive costs include pool operations 657 ms, eviction 547 ms, logical
mapping 297 ms, stamping 230 ms, and unique 165 ms. Unique GPU work remains
113.899 ms. Revisit gather is 14.720 ms with zero matrix overlap. These instrumented
times cannot be subtracted from formal wall latency; the capture does not explain
the formal dense first-candidate mean of 276 ms.

The source freeze ended after this capture. C3 trusted all-resident metadata and
bounded compute islands are being implemented. Cache transactions remain eager.
Compute replay needs explicit KV source lifetimes, graph reserved-memory accounting,
and actual replay-node MFU evidence before acceptance.

Current accepted sixteen-user reports and their output remain published with
pending-rerun notes. Do not replace them with this two-user screen. Other affected
experiments: echo_prefill, official ECHO shared model, and static echo_cache capacity
plans. Retained NOSA gr_serving reports are unaffected by these DeepSeek changes.
