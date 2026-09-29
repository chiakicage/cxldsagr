# NOSA SM90 KDA execution plan

## Acceptance and measurement

Continue until complete resident NOSA indexer and complete block sparse
attention each reach 40% useful MFU on all three captured layers. Only L31
attention reaches the target in the latest checkpoint; the goal is unfinished.
Workload:65536 prefix +1024 queries, BF16,32 Q heads,2 KV heads,D128;
full NOSA33/64 selection, CIS, exact masking and stable score ties.
The actual GPU reports NVIDIA H20Z/SM90,132 SMs. The denominator remains
989 TFLOPS. Useful FLOPs exclude masked/padded work and QK recomputation:
indexer34,616,115,200; attention68,190,994,432. Forty percent requires
87.503 us and172.374 us respectively.

Count validation, incremental derived-cache preparation, selection and all
auxiliary kernels. Report complete kernel sums separately from event API and
wall completion intervals. Prefix construction, ordinary K/V/CIS append writes
and transaction setup are outside this append-module measurement. These are
resident measurements, not offloading results.

## Integrated implementation

- Model-owned incremental compression and stable CIS pool; finite validation
  and ranked preparation share request scratch. The checked combined C++ entry
  reduces host submissions, validates writable aliasing before launch, and
  uses a request-owned pinned status flag. Model code retains reserve/finish/
  abort ownership. Invalid Q/new K/new CIS preserves all derived bytes. The
  checked entry rejects graph capture; async preparation remains capturable.
- Large score/selection uses Q16/N128, four consumer warpgroups, a producer
  warpgroup and three K buffers. Two consumer cohorts alternate first-pass
  QK and normalization. The second pass and exact selector share one kernel;
  ranked preparation removes a separate CIS ranking launch. Current fused
  implementation has96 initial registers and zero stack/LDL/STL in the
  accepted descriptor+tail combination; main remeasurement has passed.
- Score and attention preserve explicit FP32 RN scaling/addition, natural-unit
  maxima and subtraction, then convert centered differences for exp2. This
  fixes confirmed large-common-offset and QK/CIS-cancellation failures in HEAD
  and earlier candidates. Existing numerical tolerances remain unchanged.
- BF16 attention with matching K/V strides uses installed FlashInfer 0.6.18
  FA3 headers from q>=1, eight-query unions, KV128 and two stages. Q moves
  directly from its original strides through TMA; the epilogue writes final
  output and marks nonfinite converted values. For ceil(queries/8) * KV_heads ==256, actual union-tile counts now
  order work by descending size with ascending logical-batch ties. The complete
  path has four kernels: selection preparation, work sort, FA3 and native repair.
  Other positive FA3 geometries retain three kernels. Empty queries launch
  nothing; FP16 and independent K/V strides retain native dispatch.
- Native BF16 PV divides V by a power of two based on actual compacted token
  count, then restores scale after normalization. Integer conversion preserves
  subnormal and nonfinite values. A finite-output conversion guard also applies
  to FP16. This fixes confirmed finite-V unnormalized accumulation overflow.
- Exact second-pass pruning is integrated only for rows=1024, KV heads=2,
  compressed count=4159, blocks=1040 and query_start=65536. Both QK passes use
  aligned N128 operands. Upward FP16 tile-max/last-column summaries bound the
  exact normalized/GQA/BF16-pooled scores; only bounds strictly below the exact
  partial rank33 cutoff may skip tiles. Unknown bounds compute their tiles.
  Other geometries and failed resource guards retain the original fused path;
  public score-only calls are unpruned. The current kernel has96 initial registers,
  zero stack and216704 dynamic shared bytes, plus1024 driver-reserved bytes.
  The shared Q16 first pass and four seeds/cutoffs are retained; the tail now
  uses two independent Q8 lists and K buffers. This avoids computing the
  other cohort's retained tiles. The final clean frozen patch and all five
  delivered source hashes match main.
  The final selector now reuses the exact seed rank33 cutoff as a search lower
  bound when no NaN keys exist, and sorts only the64 retained IDs in its final
  ascending pass. Main integration matches all frozen delivery hashes; main
  numerical regression passes211 tests with one expected empty-graph warning.
  The preceding selector-only complete-module remeasurement completed at
  `/tmp/nosa-main-selector-combo-fa3-v2-root-20260929`. Q8-tail and FA3-v3
  main acceptance now passes349 GPU tests (490 deselected, one expected empty
  graph warning),990 CPU tests (589 resource/CUDA skips,34 subtests), and91
  focused attention tests. These earlier acceptance stages are superseded by the
  latest361-test GPU check and completed descriptor+tail measurement below.
- Build provenance records FA3 flags, installed-header hash, version and source
  hashes. Revision5 identifies the narrow pruning path; native_fa3_v3 requires
  the exact four-kernel sequence for256 work items, three for other positive
  FA3 geometries, and none for empty queries. It retains the reviewed
  numerical/transfer configuration. v2 rejects the sorter; matching Triton
  controls may retain complete v2 or v3 native build provenance. All453
  main experiment attribution tests pass. All preparation, cutoff and repair work remains attributed.
  Publication now registers the exact48-source BF16-pair graph and its
  AST-equivalent formatted-adapter variant; unknown source changes still fail.

## Latest verified BF16 pair conversion checkpoint

The exact pair-conversion patch is integrated from
`/tmp/nosa-bf16-pair-pack-root-20260929` after all109 frozen files and all five
source files were checked. Current header SHA256 `99e0e6ce9f5685e639016e24732cce249a3b2c043ff0629b021866ad0f051409`.
Private fused savings are0.648/0.703/0.547 us on L0/L15/L31, with60/60/59 wins
out of60 pairs. Original arithmetic, rounding mode, selection and CIS remain.
Three real cases,30 synthetic,5 edges,36 native tests,1,441,792 independent
conversion pairs and8 direct-pruning cases pass. SASS removes98 F2FP and98 PRMT
sites, retains96 registers and zero stack/local traffic;13 other functions are
identical. Full/source NCU and independent review are frozen with the candidate.

All7 main components rebuilt;361 GPU tests pass (490 deselected, expected
empty-graph warning). Complete run `kda_main_bf16_pair_v3_development` passes
all18 strict source/snapshot/trace and3/4-kernel attribution checks, independent
compression/selection equality, and zero changed captured IDs. Records:
`/tmp/nosa-main-bf16-pair-v3-root-20260929`, measurement in `data/measure/`.

| Layer | Complete indexer µs | MFU | Complete attention µs | MFU |
|---|---:|---:|---:|---:|
|0|141.471|24.741%|175.391|39.312%|
|15|137.854|25.390%|173.247|39.798%|
|31|137.504|25.455%|170.879|40.350%|

GPU0, same real captures/989 TFLOPS denominator,10 warmups,30 eager samples,
3 profiler repeats. Eager event API: indexer330.304/345.344/335.040 us,
attention288.080/289.632/281.824 us. Wall: indexer338.863/351.797/341.541 us,
attention293.984/295.696/287.760 us. These intervals are distinct from kernel sums.
Only L31 attention reaches40%; indexer needs about50–54 us less complete time.
Attention code is unchanged. This unpaired checkpoint does not replace private
paired A/B evidence or paper experiments. The following checkpoint predates
BF16 pair conversion.

## Current private investigations (runtime unchanged)

The fixed tail-only cluster2 prototype preserves source-level owner semantics,
uses separate Q8 TMA/GMMA layouts and unique outgoing halos. A host CuTe check
shows two packed Q8 slots are not byte-layout-equivalent to the original Q16:
16384/32768 offsets differ. The candidate pairs its new Q8 producer/consumer
layouts. With217856 dynamic shared bytes, actual cluster capacity is66 for64
required. Its SM90a CPU gate fails:96 registers,80-byte stack,35 LDL/34 STL,
and ptxas C7507 removes the requested24/112 register reconfiguration to maintain
minimum register requirements. All13 ordinary functions' .text bytes match.
No GPU kernel was launched. This is a compiler/resource failure of that source,
not a measured rejection of every tail-only cluster schedule.
Frozen evidence: `/tmp/nosa-tail-only-cluster2-candidate-20260929/CPU_GATE.md`.
A separate fixed phase96 prototype restores a common register state around
both cluster joins, then repartitions by warpgroup, preserving the selector's
112-register budget. It also fails the CPU gate:96 registers,96-byte stack,
51 LDL/43 STL and no emitted register adjustments, with C7507 again. No GPU
was launched. Its reviewed protocol and failed build are preserved in
`/tmp/nosa-cluster2-phase96-review-20260929/REVIEW.md` and
`/tmp/nosa-tail-only-cluster2-phase96-candidate-20260929/CPU_GATE.md`.
The subsequent c70-based producer-lifetime candidate moves count reads inside
each ticket iteration but still has80-byte stack,35 LDL/34 STL and no emitted
adjustments. Native producer peak liveness rises27→31; source scopes alone
did not reduce register pressure. This point also stops before GPU execution.
Its frozen evidence is
`/tmp/nosa-tail-only-cluster2-lifetime-candidate-20260929/CPU_GATE.md`.
A distinct original-c70 producer32 review proves the exact61440-register CTA
pool and source-level progress conditions. Its subsequent one-line CPU point
also fails:96 registers,80-byte stack,35 LDL/34 STL, C7507 and no native register
adjustments. The entire pruned raw.text is byte-identical to c70; increasing
that budget did not affect the native program. All13 ordinary functions and
resources match; actual cluster capacity remains66>=64. No GPU was launched.
Frozen evidence:
`/tmp/nosa-cluster2-producer32-review-20260929/REVIEW.md` and
`/tmp/nosa-tail-only-cluster2-producer32-candidate-20260929/CPU_GATE.md`.
The split-CFG successor keeps complete producer and consumer lifetimes separate
through both cluster joins. It passes the CPU gate:96 registers, zero stack/
LDL/STL, native32/112 reconfiguration,13 identical ordinary functions/resources,
and actual cluster capacity66>=64. Independent PTX/native CFG and participant
reviews preserve two dynamic join epochs per thread. This supports the role-CFG
hypothesis without establishing a universal cause of C7507.
Its first GPU gate passes L15 exact IDs/masks, FP32-reference comparison,
five poisoned graph replays per binary and output guards. The trace confirms
the actual cluster specialization. Performance nevertheless fails: fused
122.628160→124.866881us, paired+2.191358us,0/60 wins. This scope excludes
finite validation/compression/ranking and is not complete-indexer MFU. No
integration, broader GPU testing or NCU follows. Frozen CPU evidence:
`/tmp/nosa-tail-only-cluster2-split-cfg-candidate-20260929/CPU_GATE.md`;
GPU gate: `/tmp/nosa-cluster2-complete-gate-root-20260929/REPORT.md`.

The first rolled-group4 native repair stopped at its CPU gate:64 registers,
8-byte stack and4 LDL/3 STL, attributable to compaction loop/remainder state.
Its targeted restore64 successor removes spills and passes focused correctness,
with9 ordinary functions' SASS/control/resources unchanged. Natural L0 complete
graph API improves172.647→171.283us (paired1.413us,60/60 wins), but legal
union512 all-fallback regresses830.577→895.925us (+7.868%,0/60 wins), exceeding
the predeclared2% limit. It is rejected. No other fallback timing, L15/L31,
full7055 poison sweep or NCU follows that failure. Frozen report:
`/tmp/nosa-attention-fa3-group4-restore64-20260929/REPORT.md`.
A distinct four-concurrent-engine CTA repair passes its padded layout proof:
39936-byte stride,159744 total,114688 host CuTe addresses and36864 transpose
checks. The initial39552-byte-stride review did not establish that proof; it
is not evidence that the old stride is incorrect. The subsequent fixed CPU
prototype fails:64 registers,16-byte stack,11 LDL/3 STL, with24/104 retained.
One HGMMA24/EX2-12 body remains. Original9 functions' SASS/control is identical,
but their shared alignment changes16->1024 and prepare static shared grows
9488->10240. Actual and diagnostic10-function.text/resource correspondence,
base0x400, padded offsets, TMA and descriptor checks pass. No GPU or occupancy
query followed the failure. Frozen evidence:
`/tmp/nosa-attention-parallel4-swizzle-proof-20260929/ERRATUM.md` and
`/tmp/nosa-attention-fa3-parallel4-padded-20260929/CPU_GATE.md`.
A separate host proof covers runtime1024 alignment with1023 guard bytes and
160767 total dynamic reservation,7340032 address translations over64 possible
base phases. That proof alone does not establish ordinary-resource restoration
or fix spills. The completed spill audit identifies tid,2*engine and warp;
the bounded unsigned-coordinate audit preserves signed views for CuTe/model
arithmetic. A single combined CPU prototype used runtime-aligned
private storage, mask/shift thread extraction with bounded int views, and
barrier-ID recomputation inside each original volatile barrier asm. It keeps
the39936 engine stride,1024 threads,four concurrent256-thread engines and
24/104 allocation. The first build stops at a PTX syntax error:zero-operand
inline asm emits the escaped `%%tid.x` literally. No candidate binary/resource
result exists from that build. A separate syntax-only successor changes those
two strings to `%tid.x`,preserving the exact configuration and all other
source; no resource sweep or GPU acceptance/performance claim. Evidence:
`/tmp/nosa-attention-parallel4-alignment-audit-root-20260929/REPORT.md`.
Supporting frozen reviews:
`/tmp/nosa-attention-parallel4-spill-audit-20260929/REPORT.md` and
`/tmp/nosa-attention-parallel4-coordinate-domain-20260929/REPORT.md`.
Current private build:
`/tmp/nosa-attention-fa3-parallel4-resource-syntax-20260929`;
the failed build remains separately frozen at
`/tmp/nosa-attention-fa3-parallel4-resource-fix-20260929`.
The syntax successor passes the complete CPU gate:REG64,zero spills,24/104,
all9 ordinary raw.text/resources/shared alignments unchanged,and actual full64
address/TMA/GMMA/barrier/exit review. GPU1 focused checks pass real L0,three
legal all-fallback inputs,tails,20 isolated flags and235 poison sweeps. Actual
loaded repair confirmsREG64/local0/maxThreads1024,excluding the spilled
compatibility image. Natural L0 complete GPU graph API nevertheless fails:
175.129595→175.356483us,paired saving−0.290241us,10/60wins. No fallback timing,
other layers,broad suite or NCU follows;the candidate is rejected. All94 CPU
files remain unchanged before/after the gate. Result:
`/tmp/nosa-attention-parallel4-gate-root-20260929/REPORT.md`.

The Q4/N128 double-accumulator review rejects the proposed160-thread/static192
two-CTA resource estimate. SM90 allocates registers within four subpartitions:
192 registers rounds to6144/warp,allowing8 warps/SM,but two160-thread CTAs need
10 warps. That point can admit only1 CTA/SM,despite a61440-register total.
The existing ordinary Q4/128-thread kernel does fit2 CTAs atREG186/192 with
83072B shared. A separate fixed CPU candidate retains that128-thread layout,
two K buffers and two accumulators,prepares RN scale/mask/max before next MMA
issue,and finishes the current exponential/denominator work read-only.
The fixed compile uses190 registers with zero stack/local/LDL/STL and keeps all
14 existing functions' raw.text/resources identical. It nevertheless fails the
asynchronous gate:C7518 and32 HGMMA each followed by wait0 serialize the intended
overlap. No GPU launch or retune follows;the prepared GPU harness is not run.
The warning alone does not identify its source cause. Private candidate:
`/tmp/nosa-q4-readonly-candidate-20260929`;
prepared focused gate:`/tmp/nosa-q4-readonly-gate-root-20260929`.
The fixed common-wait successor moves wait0 before tile-parity dispatch and
both initial K loads before the first issue. Actual PTX has one common wait0,
but SM90a still emits32 single-HGMMA ARRIVE/wait0 windows and zero pending
MUFU; both finishes move before the next issue. REG192,zero stack/local and
all14 ordinary kernels remain unchanged. It fails the CPU gate without any
Driver/GPU call:`/tmp/nosa-q4-common-wait-candidate-20260929/CPU_GATE.md`.
A bounded source review admits one fixed rolled16-pair candidate with explicit
rows1024/count4159/start65536 guards. It replaces accumulator-parity and
next-issue-exists branches with two fixed issue sites, retaining ready phases,
tile31/32 masks and a separate tile32 drain. The pair-loop backedge still
carries a pending batch; native batching/overlap is unproved. The one CPU build
retains three8-HGMMA batches,REG155,zero stack/local and14 unchanged ordinary
kernels. Both halves interleave66 MUFU instructions after the first next HGMMA
and before its wait0; the rolled16 loop and pending-accumulator write safety
pass. No MUFU follows the final committing HGMMA, so the frozen admission's
stricter post-commit-overlap gate fails. This is distinct from the predecessor's
C7518 serialization. No Driver/GPU call was made and it is not integrated.
Admission:`/tmp/nosa-fixed-pair-readonly-history-review-20260929/REPORT.md`;
candidate:`/tmp/nosa-q4-fixed-pair-candidate-20260929`.

The fixed standalone256-item single-warp attention sorter has passed its
native resource comparison:REG32,zero stack/local/shared traffic
and no CTA barriers; all8 other kernels remain unchanged. It preserves the
exact descending ceil(count/2),ascending-ID key and uses8 keys per lane.
The previous256-thread sort has30 static SHFL instructions across8 warps;
the new single warp has240,so aggregate shuffle work is unchanged. The old
audit's36 VIMNMX omitted36 predicated instructions; the candidate's separate
erratum corrects that count to72. CPU evidence is frozen. Its prepared GPU gate
was not run after the user requested publication of the current accepted
implementation; no performance result exists and the sorter is not integrated.
Candidate:`/tmp/nosa-attention-warp256-sort-candidate-20260929`.

A new audit corrected the interpretation of the older Q16/N64x2 first-pass
failure:89.496-versus75.655us belonged to a serialized binary, with24 HGMMA/
24 waits. An unchanged-source CPU diagnostic reproduces it and reports ptxas
C7515/C7517. A subsequent fixed repair forms8-MMA batches and actual partial
next-A/B-exp overlap (8 EX2 before the remaining wait,26 afterward), retaining
96 initial registers and zero stack/local traffic. Nevertheless its fresh L15
paired gate also fails:75.335→84.258us, paired+8.900us,0/60 wins. All32768
maximum/inverse values remain bit-identical, independent reference and five
poison replays per variant pass. No full-fused expansion, other layers or NCU
follows. It is rejected; cross-run differences from89.496 are not a causal
speedup measurement. Frozen reports:
`/tmp/nosa-firstpass-n64-dedup-audit-20260929/REPORT.md` and
`/tmp/nosa-q16-n64-rawacc-cpu-gate-20260929/REPORT.md`.
The fixed asymmetric N32/N96 successor also passes layout/resource and focused
correctness checks but regresses L15 firstpass-only74.933→81.713us, paired
+6.765us,0/60 wins. All32768 maximum/inverse values match; independent reference
and five poison replays per binary pass. Actual96 registers, zero stack/local,
three8-MMA batches and partial overlap remain. No other widths, full-fused
expansion or NCU follow this failure. Frozen report:
`/tmp/nosa-firstpass-n32-n96-audit-root-20260929/REPORT.md`.

## Latest verified descriptor+tail checkpoint

Q descriptor lifetime plus per-warp tail keep was integrated. That checkpoint header
SHA256 `28eef5b9cb1a816d8e4c7abb4c2e2baa227a2cb4ea2be653ae86b47e4a2d518f`.
The exact private combination is frozen at
`/tmp/nosa-qdesc-tail-combined-root-20260929` (104 verified files), source4549e5cb,
binarya46695ee. All3 real/reference/poison cases,30 synthetic,5 edges,36 native
and8 direct-pruning oracle cases pass; actual selected top33 keys and final
64 IDs/CIS remain exact. It has96 registers and zero stack/LDL/STL/local sectors.

All7 main components rebuilt;361 GPU regressions pass (490 deselected, expected
empty-graph warning). Complete run `kda_main_qdesc_tail_v3_development` passes
all18 strict source/snapshot/trace and3/4-kernel attribution checks, independent
compression/selection equality, and zero changed captured IDs. Run records:
`/tmp/nosa-main-qdesc-tail-v3-root-20260929`.

| Layer | Complete indexer µs | MFU | Complete attention µs | MFU |
|---|---:|---:|---:|---:|
|0|142.849|24.502%|175.296|39.333%|
|15|138.114|25.342%|173.087|39.835%|
|31|138.431|25.284%|170.653|40.403%|

Same GPU0 and real inputs,989 TFLOPS denominator,10 warmups,30 eager samples
and3 profiler repeats. Event API: indexer348.400/366.960/352.272 us,
attention286.688/295.792/282.032 us. Wall: indexer355.125/373.731/359.329 us,
attention293.067/303.168/291.217 us. These host-inclusive intervals are distinct from
complete kernel sums. Only L31 attention reaches40%; indexer still needs about
51–55 us less complete time. Attention code is unchanged. This unpaired main
checkpoint does not substitute for the matched private A/B measurements.
Paper reruns/publication and the40% objective remain incomplete.

The following offset checkpoint predates the combined change.

## Latest verified offset-lifetime checkpoint

The validated offset-lifetime patch is integrated after all58 delivery hashes
and the exact constexpr parent were checked. Current pruned header SHA256 is
`5e09c7e3cc0cdee88416eafd9d94c32f33e91da7a718a5fa1f3d208e41a8502e`.
All7 native components rebuilt;361 GPU regressions pass (490 deselected,
expected empty-graph warning). Complete run
`kda_main_constexpr_offset_v3_development` passes all18 strict source/snapshot/
trace and3/4-kernel attribution checks, with zero changed captured IDs and
independent compression/selection equality. Records and source snapshots:
`/tmp/nosa-main-constexpr-offset-v3-root-20260929`.

| Layer | Complete indexer µs | MFU | Complete attention µs | MFU |
|---|---:|---:|---:|---:|
|0|152.928|22.887%|175.232|39.348%|
|15|145.119|24.119%|173.250|39.798%|
|31|145.149|24.114%|171.070|40.305%|

Same GPU0, real captured inputs and989 TFLOPS denominator;10 warmups,30 eager
samples and3 profiler repeats. Event API: indexer333.088/362.080/345.232 us,
attention287.056/285.376/279.072 us; wall339.192/367.957/351.083 and
292.885/291.697/285.548 us. Only L31 attention reaches40% in kernel sums.
This is an unpaired development checkpoint; published experiments remain
pending rerun. The following constexpr checkpoint predates this change.

## Latest verified constexpr checkpoint

The guarded-geometry constexpr candidate is now integrated from
`/tmp/nosa-pruned-constexpr-root-20260929/candidate.patch`, SHA256
`891254ab9a3e608fb640532206e2778b218a5782b547333a641528f6a9f43687`.
It changes only already-guarded shape scalars in the pruned header; dynamic
strides, arithmetic and selection order remain unchanged. Private complete
fused A/B saves5.412/6.243/6.121 us on L0/L15/L31, with60/60 wins each.
All3 real same-score/FP32/poison checks,30 synthetic cases,5 nonfinite/FTZ
cases and36 native tests pass. Actual128-CTA,640-thread launch is verified.
Stack grows16→104 B and local traffic is nonzero;13/14 kernel SASS functions
are unchanged. The full/source NCU evidence is retained privately. All7 main
components recompiled,361 GPU tests passed (490 deselected, expected empty
graph warning), and complete run `kda_main_constexpr_v3_development` passed
all18 strict source/snapshot/trace and3/4 kernel attribution checks at
`/tmp/nosa-main-constexpr-v3-root-20260929`. No captured selection IDs changed.

| Layer | Complete indexer us | MFU | Complete attention us | MFU |
|---|---:|---:|---:|---:|
|0|153.569|22.792%|174.944|39.412%|
|15|144.960|24.145%|173.726|39.689%|
|31|145.888|23.992%|169.889|40.585%|

Same GPU0/input capture/989-TFLOPS denominator,10 warmups,30 eager samples,
3 profiler repeats. Event API medians are364.368/351.408/349.456 us for
indexer and295.376/290.928/283.760 us for attention; wall medians are370.525/
358.999/354.996 and300.570/296.609/289.363 us respectively. Only L31 attention
passes40% in these complete kernel sums. No paper report has been replaced.
The checkpoint below predates the constexpr specialization.

## Previous seed-support/common-page checkpoint

The exact seed-support cutoff patch and attention common-page fast path are
now integrated after private acceptance. The124-test focused attention suite,990-test CPU regression (601 skips), and
361-test full GPU regression (490 deselected, expected empty-graph warning)
pass in main. Complete-module measurement and all18 strict attribution checks
have passed. The current run is `kda_main_seed_support_common_v3_development`
under `/tmp/nosa-main-seed-support-common-v3-root-20260929`, using the same
captured inputs, GPU0,10 warmups,30 eager repeats and3 profiler repeats.

| Layer | Indexer kernels us | Useful MFU | Attention kernels us | Useful MFU |
|---|---:|---:|---:|---:|
| 0 | 158.882 | 22.030% | 174.689 | 39.470% |
| 15 | 151.362 | 23.124% | 173.283 | 39.790% |
| 31 | 150.557 | 23.248% | 170.689 | 40.395% |

Only L31 attention meets40% in complete kernel sums. Complete indexer remains
well above its87.503 us budget. Eager event API medians are338.304/366.832/
351.344 us for indexer and289.120/288.304/278.704 us for attention; wall medians
are344.491/373.145/357.878 and295.826/295.102/285.450 us respectively.
These distinct intervals include host submission gaps and are not replaced by
kernel sums. All repeated outputs, incremental/full compression, standalone
selection, source/snapshot and trace hashes pass. Captured selection differs
in zero elements. The private `validate_complete.py` reproduces all18 checks,
including fixed useful FLOPs and exact3/4 kernel sequences on one stream/device.
Latest-source real native/Triton all-row FP32 remeasurement has passed at
`/tmp/nosa-main-seed-support-common-v3-real-20260929`. Both backends pass all3072
query rows and exact graph/eager equality; source and snapshot hashes match.
Native attention graph API is168.947/167.582/164.509 us, or40.811/41.144/41.912%
MFU. These graph API values remain distinct from the complete-module profiler
sums above. The unchanged attention acceptance is rtol=atol=.016; the separate
stricter atol=.001 diagnostic still has0/7/976 outside native elements and is
not declared passed. `validate_real.py` preserves this acceptance and identity
check. All measurements in this section are private development checkpoints,
not replacement paper results.
Seed-support changes only the first rank33 search, enumerating at most150
unique seed/halo/mandatory keys and preserving enough -infinities; invalid
support assumptions retain the full helper. Private fused A/B saves3.44/3.54/
3.78 us on all3 layers,60/60 wins each, with225 oracle cases and real/synthetic
acceptance. Patch SHA256:
`4e12bf4e561aa2e8d62cc143882825a825a9a66da50319f3638ce42d2fe6289d`.
Common-page attention skips element masks only when all8 queries select both
fully causal pages; RN QK/CIS arithmetic and all kernels remain unchanged.
Private complete graph API is170.118/169.152/165.402 us;124 GPU tests,7055
poisoned graph cases,17 mapping scenarios and all3072 real-row equality pass.
Patch SHA256:
`5ccbd693601ee4291ed6d7ad0886cbe14ce8a613a35ed17a4bd4fce4cf5d0718`.
Attribution stays native_fa3_v3 with the exact source hash; there is no new
launch-sequence or numerical contract. All main components have recompiled and passed the full GPU regression.

## Previous verified checkpoint

Input capture: `kda_inputs_baseline_20260928_1345`, actual sparse-model layers
0/15/31 with original strides. Complete main development run:
`kda_main_split_tail_fa3_v3_development`, saved under
`/tmp/nosa-main-split-tail-fa3-v3-root-20260929`.
GPU0;10 warmups,30 eager repetitions,3 separate profiler repetitions.

| Layer | Indexer kernels us | Useful MFU | Attention kernels us | Useful MFU |
|---|---:|---:|---:|---:|
| 0 | 160.895 | 21.754% | 181.697 | 37.947% |
| 15 | 155.104 | 22.566% | 180.478 | 38.204% |
| 31 | 155.618 | 22.492% | 176.962 | 38.963% |

Event API intervals are381.904/364.800/395.520 us for indexer and
297.488/292.832/298.288 us for attention. These include host submission gaps.
Every repeated output, incremental/full compression, standalone selection and
source-stability check passes. Captured IDs differ in zero elements. All18
actual sequences pass strict revision5 / FA3-v3 attribution, with3 indexer
kernels and4 attention kernels. The private `validate_complete.py` preserves
exact trace/source hashes and the3/4 sequence checks. Source snapshots,
input/dependency hashes and every sample remain in the run directory.
These are development measurements, not replacement paper results.

Main combined implementation validation:349 GPU tests pass,490 deselected,
one expected empty-graph warning;990 CPU tests pass,589 CUDA/optional-resource
skips,34 subtests. Focused attention/numerical/dispatch/work-order suite passes
91 tests. All453 experiment attribution tests pass. Logs use prefixes
`/tmp/nosa-main-split-tail-fa3-v3-{cpu,gpu}` and
`/tmp/nosa-main-fa3-v3-attention` with separate stdout/stderr.
The previous selector-only FA3-v2 run remains a private development checkpoint
at `/tmp/nosa-main-selector-combo-fa3-v2-root-20260929`; cross-run differences
are not substituted for the independent paired promotion measurements.

Independent all-row real native/Triton FP32 remeasurement is complete at
`/tmp/nosa-main-split-tail-fa3-v3-real-20260929/data`. Both backends pass
all3072 query rows, exact graph/eager equality and source/snapshot checks.
Native attention graph API medians are175.539/173.834/170.118 us, or
39.279/39.664/40.530% MFU. Only L31 reaches40% in this separate graph API
metric; these are not substituted for complete-module kernel sums above.
The unchanged acceptance tolerance is rtol=atol=.016; the separately recorded
stricter atol=.001 diagnostic has0/7/976 native outside elements, out of
4,194,304 elements per layer, and is not declared passed.

## Active independent work

Root owns integration, complete-module measurement and publication. Independent
candidates stay in /tmp and use separate GPUs; frozen deliveries are immutable.

Current work after BF16 pair main acceptance:

- Root finished the pair checkpoint (126 frozen run files) and rejected the
  pool-init barrier and prefix-rank zero-sentinel points after their measured
  gates. Main runtime sources still match the accepted measurement snapshot.
- Attention warp0 kth mapping passes static/correctness gates but fails its
  one L0 complete-API timing gate by4.104 us paired,0/60 wins. Its69 frozen
  files are verified at `/tmp/nosa-attention-fa3-warp-kth-20260929/REPORT.md`;
  no broader measurement follows. That report corrects prior EP/audit naming:
  QueryEmpty enum0 maps to hardware barrier8; initialization uses hardware0.
  The old EP serial-row/shared-residency rejection remains valid, but a pending
  QueryEmpty arrive is not a hardware-barrier0 hazard.
- A fixed rolled group4 native repair wrapper is under read-only review:
  512 CTAs may avoid repair8's parallelism shortage, but loop/register lifetime,
  code size and all-fallback throughput still need evidence. No prototype yet.
- Tail-only cluster4 indexer collaboration stops at its capacity gate: Driver
  query of the accepted cubin finds30 active clusters versus32 required for
  the128-CTA grid, losing the current single-wave first pass. No prototype.
  A targeted read-only cluster2 query finds66 active versus64 required; a
  separate cluster2 work-distribution/communication audit is now pending.
  Keep firstpass/seeds/cutoff, owner DSM core/halo and the original selector;
  capacity on the old binary does not validate a new protocol or speedup.
  `/tmp/nosa-tail-only-cluster4-audit-20260929/occupancy/` contains queries;
  root numerical ownership review is
  `/tmp/nosa-tail-only-cluster4-protocol-root-20260929/PROTOCOL_REVIEW.md`.
  These are distinct from prior one-pass cluster logit storage.

Current private follow-on work after the offset checkpoint:

- Q descriptor lifetime is privately accepted and frozen at
  `/tmp/nosa-q-descriptor-lifetime-root-20260929/REPORT.md`. Three-layer fused
  paired savings3.983/2.360/2.595 us,60/60 each; all3 real/reference/poison,
  30 synthetic,5 edges and36 native tests pass. Stack48→0B and LDL/STL15/12→0/0;
  actual full NCU local load/store sectors are zero. Source3b698215, binaryf1ec0f6c.
- Tail per-warp keep is independently accepted at
  `/tmp/nosa-tail-warp-keep-20260929/REPORT.md`: paired savings4.139/3.665/3.995 us,
  all60/60. Every WG keeps its original collective/wait/release; the guard
  skips only excluded rows' tail max/inverse reads and normalized writeout.
  Eight direct-pruning oracle cases preserve actual top33 IDs/keys and full
  final CIS selection; discarded below-cut pool entries may differ.
- The private combination source4549e5cb passes L15 against descriptor-only
  source127.773→123.886 us (paired4.086 us saving,60/60). It retains zero spill
  and13 unchanged other SASS functions. Combined acceptance, main rebuild and complete-module measurement have
  passed. Latest complete MFU is recorded above; private timings remain partial.
- Per-WG leader arrivals are semantically reviewed but rejected by the single
  L15 GPU gate: paired+2.303 us,0/60 wins, stack48→72B. No other leader variant
  was tested. `/tmp/nosa-wg-release-candidate-20260929/REPORT.md`.

Completed follow-on work from the constexpr checkpoint:

- The offset-lifetime patch is privately accepted and now integrated:
  `/tmp/nosa-constexpr-offset-liveness-audit-20260929/probe.patch`, SHA256
  `96fbeeb49fdb449d2b10f9c1f2fb40222ac9d1d308b3485432cb7b25e8c4ba82`.
  Same constexpr parent; three-layer fused paired savings0.309/0.246/0.383 us,
  all existing real/synthetic/edge acceptance passes. Stack104→48 B, with
  unchanged96 initial registers and13 other kernel functions. Do not infer a
  large speedup from reduced stack; the eliminated masked LDLs are rare.
- The one-WG representative barrier audit passed, but its GPU candidate
  regressed as recorded above and is rejected.
- Two initial seed tiles were rejected offline: total seed+tail WG work grows
  4.20/16.57/20.05%, and all worst work chains lengthen. No implementation.
  `/tmp/nosa-seed2-feasibility-20260929/REPORT.md`.
- Three-stage/global-CIS stopped after historical/resource review. The old
  packed-Q/O measurement cannot be relabeled as current performance; no new
  prototype was run. `/tmp/nosa-attention-stage3-global-cis-audit-20260929/AUDIT.md`.

The entries below retain the accepted stages leading to the current source;
their original private measurements are distinct from the latest main run.

- Q8 tail delivery is integrated from
  `/tmp/nosa-q16-split-tail-delivery-20260929/integration.patch`, SHA256
  `9f699d1dc4e6276b685adf29cab024986919df7394236dc523d4e244e4d81f25`.
  On GPU6, selector baseline versus selector+split-tail fused graph medians:
  L0 160.075→147.029 us, L15 139.902→139.545 us, L31 139.998→139.439 us.
  Each comparison uses60 paired samples of50 calls; wins60/58/60.
  These exclude complete validation and preparation.36 native tests,30 clean
  synthetic cases,24 audit synthetic cases, three full scaled-dot audits,
  exceptional/FTZ certificates and poisoned graph replays pass. Actual resources:
  640 threads,96 initial registers,16-byte stack,9 LDL/7 STL sites,1 CTA/SM.
  GPU6 has profiled the frozen combined binary: first-pass PM Tensor49.731%
  and XU54.313% at1.81GHz. A four-WG ring suggestion was withdrawn after
  finding the identical rejected root experiment. A fixed logical N128 made
  from two N64 WGMMA accumulators has since regressed75.655→89.496 us on
  L15 (60/60 losses), despite bitwise max/inverse and zero spills; it is stopped.
  Separate WGMMA/normalizer warpgroups with FP32 shared handoff subsequently
  stopped at the resource/bandwidth feasibility gate, as recorded below.
- Sorted attention is integrated from
  `/tmp/nosa-attention-fa3-sorted-delivery-20260929/delta_from_safe_pv.patch`,
  together with `/tmp/nosa-fa3-v3-attribution-20260929/attribution.patch`.
  Private complete graph API medians on GPU7 are175.888/175.194/171.074 us,
  versus189.627/185.611/178.170 us paired baselines. These are the sorting-only
  private measurements; latest main complete kernel sums appear above.
  All112 attention tests and17 extended mapping/poison cases pass privately;
  all3 real-layer outputs match the preceding implementation bitwise.
  A separate cooperative prepare+sort fusion was tested. Actual launch
  support is present and the256-CTA grid fits its792-CTA residency capacity;
  graph and exact-output acceptance pass, but complete timings show no
  consistent gain. Cooperative fusion is rejected. The common-page mask fast
  path has passed extended acceptance and is integrated as described above.
  The subsequent eight-query repair CTA regressed all-fallback calls and was
  rejected, as recorded below.
- GPU1: final seed-cut early acceptance passes numerical/oracle/graph checks
  but regresses L15/L31 by about0.4 us and is rejected. The exact seed-support
  first cutoff is integrated. A fixed264-worker Q4 tail service plus latest
  standalone selector takes26.990/28.190/34.423 us; with a seed-restore proxy
  it takes33.018/34.214/40.434 us. These omit actual stage1 export, and do not
  establish net improvement. L31 is especially tight; no worker/layout sweep
  is authorized. Resource and boundary acceptance is being finalized privately.

The old halfhalo phase diagnosis measured first-pass/normalizer about64.5 us,
bounds/init7.3 us, seeds11.5 us, cutoff/organization5.9–6.8 us and final
selection10.8–11.9 us, with tails44 us in L0 and25.2 us in L15/L31. These are
old clock-instrumented thread0 phase paths, not timings of the new combination
or complete modules. Even removing that entire old tail alone would not meet
the87.503 us complete-indexer budget. Full/source NCU showed first-pass Tensor
52.9% and XU57.9% at1.81GHz, so this is not a claimed hardware lower bound.

## Rejected directions and evidence

Private diagnostic files are not paper deliverables:

- Packed BF16 integer pooling stops at the native-benefit gate. A robust
  NaN/sentinel encoding can preserve the pool, but its filtering/decoding
  roughly consumes the scalar conversion/max savings. Entire old pool/mask/halo
  regions are only1.245% of dynamic warp instructions; no compile/GPU candidate.
  `/tmp/nosa-packed-pool-audit-20260929/REPORT.md`.

- Prefix-rank zero-sentinel predicate removal is exact and passes125 GPU tests
  plus9 invalid transactions, but L0 complete preparation regresses0.022 us
  paired (12/60 wins), despite L15/L31 savings0.051/0.042 us. Not integrated;
  no repeated timing or NCU.72 registers/zero spill,12 unchanged other functions.
  `/tmp/nosa-prepare-rank-zero-sentinel-root-20260929/REPORT.md`.

- FP16 RU summary packing stops at ISA feasibility: PTX f16x2 conversion
  supports RN/RZ only, not RP. Replacing RU breaks the bound even for the
  first FP32 value above1 or a positive subnormal. No compile or GPU task.
  `/tmp/nosa-fp16-summary-pair-audit-20260929/REPORT.md`.

- Removing the pool-init/sort barrier passes the L15 exact/reference/poison
  gate but saves only0.055 us paired,39/60 wins; with ranking0.058 us,39/60.
  No stable practical gain justifies integration or broader testing. Actual
  BAR sites7→6,96 initial registers and zero spill;13 other functions match.
  `/tmp/nosa-pool-init-barrier-root-20260929/REPORT.md`.

- Inline epilogue repair retaining the native schedule stops offline: up to8
  rows serialize while FA3 retains197712 B shared and1 CTA/SM, versus native
  repair39552 B and4 CTAs/SM. Prior separate repair8 regressed valid fallback
  calls47–68%; pointer/flag/barrier lifetime requirements are documented.
  `/tmp/nosa-attention-inline-ep-repair-audit-20260929/REPORT.md`.

- First-pass31+2 peeling preserves tested L15 exact/reference/poison behavior,
  but regresses fused122.772→123.135 us (paired+0.207 us,13/60 wins); with
  ranking paired+0.174 us,9/60 wins. It retains96 initial registers, zero
  stack/local and13 identical other functions. No broader testing or NCU.
  `/tmp/nosa-firstpass-peel-pair-root-20260929/REPORT.md`.

- CTA-local histogram inversion preserves attention mapping/numerics but
  regresses L0 complete API by3.452/3.481/3.514 us in three paired trials.
  It removes the sort launch but adds startup work to all256 CTAs. All124
  tests,91 mappings,17 edges,7055 poison cases and3 real bitwise checks pass;
  performance rejection stops further layer/fallback/NCU measurements.
  `/tmp/nosa-attention-fa3-inline-histogram-20260929/REPORT.md`.
- Single-warp prefix ranking passes tested exact outputs and invalid-input
  write protection, but L15 complete preparation regresses12.196→19.780 us,
  0/60 wins. Registers rise72→80 without spills; the removal of rank CTA
  barriers does not compensate for the serial work. No broad GPU expansion.
  `/tmp/nosa-prepare-warp-rank-candidate-20260929/REPORT.md`.

- Early first-pass issue credit preserves the all-thread phase and K lifetime
  proofs, passes exact/reference/poison checks and the zero-spill SASS gate,
  but regresses L15 fused123.251→125.304 us. Paired median regression2.079 us,
  0/60 wins; including ranking also regresses. No broader tests or NCU follow
  this failed single point. Original completion-based cohort ordering remains.
  `/tmp/nosa-firstpass-issue-credit-candidate-20260929/REPORT.md`.

- Overlapping immutable-prefix ranking with finite validation through staged
  scratch preserves tested derived/ranking bytes but regresses L15 complete
  preparation12.156→12.509 us,0/60 paired wins. The new first kernel has92
  registers versus38 for the old scan; neither spills. The second kernel is
  shorter, but the first grows more. No broad tests or NCU expansion follows
  this failed gate. Main preparation remains unchanged.
  `/tmp/nosa-prepare-rank-overlap-root-20260929/REPORT.md`.

- A cheaper no-kept-tail certificate for reusing the final seed cutoff stops
  offline. It applies to1583/695/458 of2048 rows in L0/L15/L31, strictly fewer
  than the previously rejected final-cut certificate. In L15/L31 all CTAs with
  the largest Q8 tail counts have zero eligible rows. Tile counts are not
  measured CTA runtime; this is a distribution gate, not a speed measurement.
  No source candidate or compilation. See
  `/tmp/nosa-no-tail-cutoff-audit-root-20260929/REPORT.md`.

- Full FP32 logits materialization and one-pass cluster layouts lost to storage,
  synchronization or spilling; see `/tmp/nosa-score-next-20260929/REPORT.md`.
- Mixed polynomial/SFU exponentials were slower despite sampled accuracy;
  hardware exp remains. `/tmp/nosa-score-poly-root-20260929/REPORT.md`.
- N256/Q8/two-consumer first pass remains88.138 vs76.747 us after removing
  spilling; `/tmp/nosa-score-n256-root-20260929/REPORT.md`.
- Four-cohort N128 first pass is77.918 vs76.418 us, with exact numerics and
  zero stack; `/tmp/nosa-score-cohort4-root-20260929/REPORT.md`.
- Raw-QK max before RN scaling passes100,564,992 raw-value checks,36 native
  tests and24 synthetic selector cases. Its first-pass floor improves~2.1%,
  but full fused gain is only0.4 us and score-only regresses~0.9 us. It remains
  private; `/tmp/nosa-normalizer-rawmax-v1-20260929/REPORT.md`.
- Persistent ranking snapshots save about1 us on the target append but add
  maintenance and boundary regressions. Shared compression layout eliminates
  bank conflicts without consistent append benefit. Neither is integrated;
  `/tmp/nosa-prefix-snapshot-v2-20260929/REPORT.md` and
  `/tmp/nosa-compression-layout-v1-20260929/REPORT.md`.

- Packed half2 exponentials offer no demonstrated SFU throughput route: native
  PTX f16x2/bf16x2 each expands into two scalar MUFU instructions plus PRMT.
  FP16 input rounding alone can exceed the existing inverse tolerance. This
  stops at offline ISA/SASS and numerical review; no GPU speed claim.
  `/tmp/nosa-half2-exp2-audit-20260929/REPORT.md`.
- Q16 static112 with32-thread producer fails the actual512-thread kernel limit.
  A512-thread static128 inline-producer alternative removes spills but regresses
  all3 layers by0.2–0.5 us. `/tmp/nosa-prune-producer32-root-20260929/REPORT.md`.
- Attention group12/KV64, owned group4 with actual two-CTA residency, and
  group8/KV192 all regress. Sorted+masked-CIS rewrite also regresses; accepted
  sorting keeps arithmetic unchanged. Producer16 is illegal for setmaxnreg.

- Runtime selection-based query grouping was examined only offline. Greedy
  64-query windows reduce total KV128 tile counts by6.19/3.50/3.42% across
  L0/L15/L31, before Q gathering, grouping and output/causal mapping costs.
  This does not justify a GPU prototype or speedup claim.
  `/tmp/nosa-attention-query-grouping-root-20260929/REPORT.md`.
- Final seed-cut certificate is rejected: it saves0.67 us on L0 but regresses
  L15/L31 by about0.4 us despite exact oracle/real/poison acceptance.
  `/tmp/nosa-final-cut-certificate-20260929/REPORT.md`.
- Shared FP32 handoff to separate first-pass normalizer WGs stops offline:
  the extra shared traffic and full-stage63488>61440 register demand do not
  support the required50–55 us first-pass window.
  `/tmp/nosa-q16-shared-handoff-feasibility-20260929/REPORT.md`.
- Fixed Q4 dynamic tail service stops after current-phase diagnosis: L31's
  measured conditional tail window is smaller than the new service/selector
  chain even before stage1 export. Instrumentation speeds the full kernel by
  about3.3 us, so this is not a hard impossibility bound.
  `/tmp/nosa-current-fused-phase-20260929/REPORT.md`.
- Repair8 saves normal helper time but regresses valid all-fallback complete
  calls47–68%; the separate required-flag2048-CTA candidate has no stable
  complete-call gain. Neither is integrated.
  `/tmp/nosa-attention-fa3-repair8-20260929/REPORT.md` and
  `/tmp/nosa-attention-fa3-required-flag-20260929/REPORT.md`.
- Fixed128-CTA two-work pairing stops offline: real tile-count scheduling is
  worse than132-SM dynamic assignment and cross-work membership reuse needs
  extra synchronization. `/tmp/nosa-attention-paired-cta-audit-20260929/AUDIT.md`.
- Per-CTA shared CIS sorting passes the exact CPU/L15 GPU gate but saves only
  0.072 us in fused L15,37/60 wins; no stable full-fused gain. Both baseline
  and candidate have104-byte stack. No broad follow-up is run.
  `/tmp/nosa-cis-shared-sort-20260929/REPORT.md`.
- Interleaving the two independent normalizer head sums regresses fused L15
  129.951→132.454 us,0/60 wins; it stops after the single-instance gate.
  `/tmp/nosa-normalizer-head-interleave-root-20260929/REPORT.md`.

- Q8-local four-seed ranking stops offline: second-pass WG work changes
  0/−1.97/−1.04%, while seed+tail TMA requests rise40–50%; L0/L31 worst
  chains remain unchanged. Finite captured top33 IDs/keys stay exact.
  `/tmp/nosa-q8-local-seed-audit-root-20260929/REPORT.md`.
- Attention rescale attribution is corrected in the new frozen audit:
  line64's3229 samples belong to elementwise exp, not rescale_o. Main-loop
  rescale skips entire warps93.5–93.9% of the time; replacing that branch
  with unconditional predicated FMUL issuance has no supported advantage.
  `/tmp/nosa-attention-rescale-exp-audit-20260929/REPORT.md`.

- Exact per-row all-exp-first rewriting stops offline: current firstpass SASS
  already overlaps62/66 exponentials with at least two other MUFU results
  before first use; pair/serial additions fill existing scheduling slots.
  No independent dependency removal was identified, so no compile was run.
  `/tmp/nosa-inplace-exact-exp-audit-20260929/REPORT.md`.
- Attention initial-scale-one removes first-PV dead rescaling semantically and
  passes numerical tests, but L31 complete timings are unstable; the candidate
  is rejected without further fallback/NCU runs. Its report preserves the full
  distributions: `/tmp/nosa-attention-fa3-initial-scale-one-20260929/REPORT.md`.
  Attention already generates in-place exp before the original sum;60/64 sum
  operations are interleaved before the final exp, so this adds no new source
  candidate. `/tmp/nosa-attention-exp-order-audit-20260929/REPORT.md`.

- Exact FTZ skipping stops at the distribution gate: all three real layers
  have zero finite score or denominator-rescale exp2 inputs<=−127. Mask/padding
  −Inf is counted separately. Score logits are an independent IEEE FP32 bmm
  reference, not a bitwise WGMMA trace; minima are far from the threshold.
  The PTX review preserves the signed-zero limitation of an error-bound-only
  proof. No skip implementation or compile.
  `/tmp/nosa-exact-ftz-distribution-20260929/REPORT.md`.

## Final acceptance and publication still required

1. Finish main acceptance of the integrated finite-V repair and exact pruning,
   preserving unchanged numerical tolerances, full policy and real/synthetic data.
2. Freeze sources, recompile in main, validate graph/fallback/alias/cache
   behavior and measure both complete modules on all3 layers. Continue targeted
   profiling and optimization; individual-kernel wins do not establish40% MFU.
3. Run final global CPU/GPU checks and register exact final SUPPORTED_GRAPHS.
4. Rerun affected experiments: `nosa_kernel_mfu` and sparse/full-NOSA parts of
   `nosa_indexer_pattern_65536_1024` remain pending. The matched full-model
   `indexer_block_sparse_profile` rerun completed on 2026-09-29 (see below).
5. Publish accepted new run IDs and reports, then replace and clean superseded
   affected report/raw artifacts in the same update.

The user subsequently requested a summary and publication of the current
implementation before further optimization. The BF16-pair/FA3-v3 checkpoint
is therefore being committed with its existing accepted complete-module
measurement, keeping the original run ID and source hashes. Only equivalent
Python formatting and reviewed attribution-graph registration are added at
publication. See `docs/nosa_sm90_checkpoint.md`. This checkpoint publication
does not mark the40% goal complete or replace the distinct pending experiments.

The full-model comparison has since been published at `94bf521` as
`sparse_native_h200_gpu1_20260929_01` and `sparse_triton_h200_gpu1_20260929_01`.
Both runs passed source/request/device identity, complete kernel attribution
and candidate-output checks; 553 experiment CPU tests and 361 global GPU tests
passed. The corresponding older sparse report and raw artifacts were replaced
after acceptance. See the [updated report](../experiments/indexer_block_sparse_profile/README.md).

Preserve `kernel_mfu_h200_gpu1_20260928_071840` and affected pattern reports with
their original implementation and measurement meaning until their own reruns
pass acceptance. Their replacement and the 40% objective remain unfinished.
