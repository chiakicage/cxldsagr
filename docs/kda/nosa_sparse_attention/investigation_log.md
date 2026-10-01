# NOSA resident attention: investigation log

These are records from the resident KDA work on 2026-09-28/29, reorganized
from the former combined execution plan. Wording such as “current”, “active”,
“latest” and “pending” refers to that historical point. Later entries can close
an earlier pending gate; use the component checkpoint and current plan for
present status. The document move did not execute any candidate.

Private `/tmp` paths preserve original evidence locations and are not guaranteed
to remain available. They are not experiment deliverables. Unpaired development
runs, partial-kernel timings, CPU-only gates and unexecuted harnesses retain
their original limitations. Rejected or unmeasured candidates are not integrated.

## Accepted implementation detail at publication

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

- Build provenance records FA3 flags, installed-header hash, version and source
  hashes. Revision5 identifies the narrow pruning path; native_fa3_v3 requires
  the exact four-kernel sequence for256 work items, three for other positive
  FA3 geometries, and none for empty queries. It retains the reviewed
  numerical/transfer configuration. v2 rejects the sorter; matching Triton
  controls may retain complete v2 or v3 native build provenance. All453
  main experiment attribution tests pass. All preparation, cutoff and repair work remains attributed.
  Publication now registers the exact48-source BF16-pair graph and its
  AST-equivalent formatted-adapter variant; unknown source changes still fail.

## Joint development measurements

The accepted BF16-pair run and earlier shared tables, event/wall intervals,
input captures, source hashes and acceptance counts are preserved in the
[indexer investigation log](../nosa_indexer/investigation_log.md). Its
“Earlier joint development measurements” section includes the accepted
seed-support/common-page attention patch and the original native/Triton
all-row FP32 checks. Old stricter diagnostics that failed are not relabeled
as passed; development graph API measurements are distinct from kernel sums.

## Final recorded private gates

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

## Earlier candidate handoffs and investigations

The early rolled-group4 review below preceded the failed measured gate above.
It is not a current prototype request. The single-warp sorter passed its CPU
gate but was not GPU-timed before publication.

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

- Three-stage/global-CIS stopped after historical/resource review. The old
  packed-Q/O measurement cannot be relabeled as current performance; no new
  prototype was run. `/tmp/nosa-attention-stage3-global-cis-audit-20260929/AUDIT.md`.

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

## Rejected directions and evidence

- Inline epilogue repair retaining the native schedule stops offline: up to8
  rows serialize while FA3 retains197712 B shared and1 CTA/SM, versus native
  repair39552 B and4 CTAs/SM. Prior separate repair8 regressed valid fallback
  calls47–68%; pointer/flag/barrier lifetime requirements are documented.
  `/tmp/nosa-attention-inline-ep-repair-audit-20260929/REPORT.md`.

- CTA-local histogram inversion preserves attention mapping/numerics but
  regresses L0 complete API by3.452/3.481/3.514 us in three paired trials.
  It removes the sort launch but adds startup work to all256 CTAs. All124
  tests,91 mappings,17 edges,7055 poison cases and3 real bitwise checks pass;
  performance rejection stops further layer/fallback/NCU measurements.
  `/tmp/nosa-attention-fa3-inline-histogram-20260929/REPORT.md`.

- Attention group12/KV64, owned group4 with actual two-CTA residency, and
  group8/KV192 all regress. Sorted+masked-CIS rewrite also regresses; accepted
  sorting keeps arithmetic unchanged. Producer16 is illegal for setmaxnreg.

- Runtime selection-based query grouping was examined only offline. Greedy
  64-query windows reduce total KV128 tile counts by6.19/3.50/3.42% across
  L0/L15/L31, before Q gathering, grouping and output/causal mapping costs.
  This does not justify a GPU prototype or speedup claim.
  `/tmp/nosa-attention-query-grouping-root-20260929/REPORT.md`.

- Repair8 saves normal helper time but regresses valid all-fallback complete
  calls47–68%; the separate required-flag2048-CTA candidate has no stable
  complete-call gain. Neither is integrated.
  `/tmp/nosa-attention-fa3-repair8-20260929/REPORT.md` and
  `/tmp/nosa-attention-fa3-required-flag-20260929/REPORT.md`.

- Fixed128-CTA two-work pairing stops offline: real tile-count scheduling is
  worse than132-SM dynamic assignment and cross-work membership reuse needs
  extra synchronization. `/tmp/nosa-attention-paired-cta-audit-20260929/AUDIT.md`.

- Attention rescale attribution is corrected in the new frozen audit:
  line64's3229 samples belong to elementwise exp, not rescale_o. Main-loop
  rescale skips entire warps93.5–93.9% of the time; replacing that branch
  with unconditional predicated FMUL issuance has no supported advantage.
  `/tmp/nosa-attention-rescale-exp-audit-20260929/REPORT.md`.

- Attention initial-scale-one removes first-PV dead rescaling semantically and
  passes numerical tests, but L31 complete timings are unstable; the candidate
  is rejected without further fallback/NCU runs. Its report preserves the full
  distributions: `/tmp/nosa-attention-fa3-initial-scale-one-20260929/REPORT.md`.
  Attention already generates in-place exp before the original sum;60/64 sum
  operations are interleaved before the final exp, so this adds no new source
  candidate. `/tmp/nosa-attention-exp-order-audit-20260929/REPORT.md`.
