# NOSA selected-source publication checkpoint

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

> Superseded publication checkpoint. The historical identities, commands and
> findings below describe the earlier 114-file implementation. Current formal,
> profile and API results are in the [pool-scan publication](nosa_pool_scan_publication.md).
> Replaced report/raw families have been removed after accepted replacements;
> required follow-up inputs are preserved through the mapping recorded there.

The fixed P/NH and ordinary GR budget replacement families are published. Their
runtime/harness sources remained frozen throughout final review and publication.
The requested overall efficiency outcome is still incomplete: candidate-only
proximity and compute/IO dominance remain unresolved, and the fixed
async path is slower than sync. Current results are in the
[fixed report](../../../experiments/nosa_motivation/README.md) and
[旧 GR 实验（已结束）](experiment_organization.md#retired-gr-serving).

## Fixed P/NH family

| Scope | Accepted identity |
| --- | --- |
| Formal measurement | `nosa_motivation_sm90_20261004_03` |
| Matching profile | `nosa_motivation_profile_sm90_20261004_03` |
| Independent APIs | `nosa_attention_reference_sm90_20261004_02` |
| Runtime, 114 files | `a73ad32b0a3322cfb06c66121f50345d2ef6004b465733115fee6006a0f2f18a` |

The promoted FA3 wrapper observes the current stream once, uses the raw-stream
FFI scope and reuses that stream for existing lifetime records. Production
targeted GPU checks passed (11 tests), followed by independent full32 H64K/A128
prefix construction: four eager-reference requests and 16 graph requests across
four methods, all exact, with real host misses and output/discard/cleanup checks.
The formal fixed run used the private allocator provider without fallback.

All 128 formal saved outputs were reopened, including 96 exact non-HBM/HBM
comparisons. The matching profile contains 24 timelines and 12 internal-work
records, 36 exact outputs and 384 layer samples. All 96 applicable async layer
samples fail both 90% overlap targets; 96 no-fetch async samples have no ratio.
The API audit checked the complete useful matrix numerator and 2,112 exact FA3
receipts. Raw materialized QK/PV validation samples three batch rows per call.

HBM full-request MFU proximity is 97.985442% / 99.286714% for request IDs 0/16
against independent GEMM/BMM+FA3 API compositions. Candidate ratios are
75.930354% / 72.497510%. These compositions sum independent medians; they are
neither complete observed requests nor performance ceilings. Fixed sync/async
latency ratios are 0.973566 overall and 0.912206 for revisits. No failed gate or
valid latency sample was removed.

The report publishes 19 selected assets, including its manifest. The README and
two API log files bring the installation to 22 files. All installed hashes and
17 links were verified before deleting exactly these superseded directories:

| Directory | Files | Bytes |
| --- | ---: | ---: |
| `experiments/nosa_motivation/output/data/nosa_motivation_sm90_20261004_02` | 247 | 159,779,818 |
| `experiments/nosa_motivation/output/data/nosa_attention_reference_sm90_20261004_01` | 2,292 | 24,018,152,618 |
| `experiments/nosa_motivation/output/log/nosa_motivation_sm90_20261004_02` | 2 | 10,135 |

Before deletion, all old path sets and file hashes were checked against the
reviewed inventory. The accepted three new data families remain in place.
Required frozen Q128 operands/provenance had already been preserved externally:
467 files / 2,301,547,815 bytes, with byte-for-byte verification and 64 CPU reopen
cases. No frozen diagnostic helper or manifest was rewritten. Owned A1024,
resident/module, pattern and cold-union overlap report families were unaffected
by this reserved-workspace wrapper and were not removed.

The publication helper, exact old inventory, install plan, reviews and receipt
are retained in
`experiments/nosa_motivation/output/data/nosa_motivation_sm90_20261004_03/publication/`.
External preparation is
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_motivation_publication_20261004_03/`.

| Receipt | SHA256 |
| --- | --- |
| Formal independent review | `74364c1ad795d8d99e9719825faa79b88c3c34965d2769ff3614928db081212f` |
| Profile independent review | `b3c5c69452ccfa2d3c49613d0ffc7faf60850bdcb1247b7d9383c04e90392b86` |
| API independent review | `4befd399cb8a1cfe23b2fdfd1873e595a2cf54abc6d4e026822765fcba9919d0` |
| Publication plan | `a780255750c216e65d2bc2ea339a1c888cb8119a14d4deef92580fc1bbcc9527` |
| Independent staging review | `684325dd6c56a4bbf9c6394656b282b5abddd08d8f24ada9fa26046999b76a4e` |
| Installed publication and cleanup | `cd2f2860b17ff69172ece36cbc426e2f8d09ed161be97d309e0327975d7f36cd` |
| Independent post-install review | `a549f684e18fa17f31af7eae400e3cf41c9c5c14f2cb35ef8221ac1499005f81` |
| Q128 preservation independent review | `46963b93074f9d4390f2ecda8e7365facd372115bdb0853754e00a6e10fef210` |

The combined report acceptance supersedes pending fields in immutable earlier
review receipts, which preserve the evidence available when they were created.
Figure generation and CPU report regeneration are not additional measurements.

## Ordinary GR budget family

H4K/H16K/H64K formal and matching profile `_20261004_02` runs passed independent
review, retaining the original 444 workload records and their configuration.
There are 1,776 formal requests and 1,332 exact non-HBM/HBM hidden comparisons.
Formal source covers 280 files at
`8dc00e23469a67ebee67eb603859c9de9987e9151520a607d1bb20a498e547f9`;
profile source covers 293 files at
`1dee55b4238a0b19d944082805f93dfdad18bc314a34c6315c2627d56add8bbf`.

HBM has the lowest all-request mean in 21/21 combinations. Overlap's all-request
mean is lower than serial in 19/21 combinations, but its revisit mean is lower
in only 2/17 nonempty combinations. All nine internal profile samples fail the
90% overlap target. Admission supports the active-allocation ledger, not a proof
that total physical HBM stays inside the byte budget. These shorter Beauty traces
do not replace fixed-loop user coverage or establish workload representativeness.

The HBM lazy host-flag reporting fix passed 167 CPU tests; ordinary shared-path
GPU checks passed four tests, and the full32 checkpoint gate passed 16 complete
outputs. No runtime allocation was added to satisfy the validator. GR did not
record a target allocator-provider receipt, so the fixed run's private-provider
claim is not reused here.

Publication installed 63 package files and two localized navigation edits.
It retained 57 new report assets, replaced 45 old asset paths and removed 12
stale report files. The exact six old run families were then removed from 18
output directories (1,638 files / 623,125,320 bytes). All six new formal/profile
families remain. Independent post-install review passed. Details and links are
in the [GR execution record](nosa_gr_selected_source_execution.md).

| Receipt | SHA256 |
| --- | --- |
| GR installed publication and cleanup | `6e7c84cba390ec78cf250fa03b8bd74fbe589e18c7ba0e85959d4c3b07537aed` |
| GR independent post-install review | `0631deb7c04e3c4d00cdf34e81e717a196518efced7054302b6ad71e076a9441` |

## Remaining efficiency work

The six accepted HBM traces contain about 1.80 ms of separately identifiable
nonmatrix activity omitted by the matrix/FA3 composition, plus mixed indexer
score/normalization work that cannot be decomposed from these traces. Candidate
profile windows are materially inflated. No subtraction from formal wall time
is used to label CPU cost. A subsequent matched diagnostic directly measured
approximately 36 ms of host referrer scanning in dense request ID 20 and sparse
request ID 22 tails. The formal events themselves were not instrumented.

The [candidate follow-up](nosa_candidate_efficiency_followup.md) records the
source-backed findings and bounded diagnostics. Two fresh uninstrumented trace
repetitions have since passed all 256 full-output and structural checks; the
same three candidate tails recur. The completed GC/allocator probe passed 128
exact outputs and 99 independently reconstructed interval checks. Referrer scans
take 36.471005/36.012608/36.528546 ms in its dense/sync/async tails; direct GC
callback unions are 1.162762/0/0 ms. Nested scan and inclusive guard intervals
cannot be added. The final probe seal is
`682ae6868a778918492be9bbbfbd6e89394837ef372e881929253ba0ba13324b`.
These probe timing differences do not isolate observer overhead or removable
scan cost.

Complete H64K/A128 indexer API replays have now passed for request captures 0/16
in `nosa_indexer_api_q128_r0_20261004_02` and
`nosa_indexer_api_q128_r16_20261004_01`. Each retained seven samples after three
warmups. The 32-layer wall/event medians are 4.792001/4.769504 ms and
5.058357/5.034816 ms. Saved checks cover exact selection and downstream attention
for all 32 layers, history reconstruction/preservation and original finite-input
rejection. The replay includes the original indexer and deferred finite decision;
it runs consecutive API calls, not a complete model request. Separate grouped
raw-QK replays take about 2.026 ms for one product and 4.023 ms for two, but write
2.003418 GiB of explicit FP32 output per product pass. Native fused score does not
materialize that output, so these times provide neither a native-score lower
bound nor removable-overhead estimates. The accepted API composition is unchanged.
Analysis and all 42 samples are in
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_indexer_api_analysis_20261004_01/results/`;
the receipt is
`24b11330345327ebc1054ff1333d8ea3e6a1f158a6871c15b2ffb38266318915`.

The first external ordinary-generation pool-scan intervention pair also
completed: `nosa_pool_scan_intervention_20261004_01_baseline_01` followed by
`nosa_pool_scan_intervention_20261004_01_native_01`. Each arm passed 128 live exact
output checks; all 256 observations and 128 matched request pairs were retained.
The three known offload candidate tails shortened by 13.8–14.0 ms but remained
their groups' maxima. Offload revisit extend means fell by 0.772–0.866 ms, while
their medians rose by 0.056–0.222 ms. First-visit extend means rose by
0.550–0.595 ms, with 47/48 paired intervals higher in the native arm. HBM native
request 24 became a new maximum at 18.651 ms, despite remaining below the declared
tail screen. This is a mixed result from one sequential pair; run order and
system noise remain in the differences, and an isolated net benefit is not
established. The analysis is in
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_pool_scan_intervention_analysis_20261004_01/`;
its receipt is
`2a6a7a86fc5198d4296eb59519474b72ba00e5247f6551984d60843b97880fd7`.

A reversed-order pair has also completed all 256 exact-output checks. The three
known tails shortened by 13.351/13.841/13.351 ms, and offload revisit extend
means improved in both pairs. First-visit and typical effects remain mixed;
descriptively pooled first-visit extend means are +0.177/+0.253/+0.304 ms.
Both pairs and every sample are retained; reversing order does not remove drift
or make repeated request IDs independent samples.

The bounded Nsight graph/node HBM16 diagnostic completed 34 exact-output checks.
Candidate wall changed to 17.979575/21.060208 ms, +7.92%/+26.41% versus the
ordinary median, exceeding the chosen 5% diagnostic tolerance. Both exported
databases warn of possible missing CUDA/NVTX events. Node mode reports an
observed 13.498574 ms activity union; graph envelopes are separate, and neither
mode proves a complete activity inventory or GPU idle time. The
[follow-up record](nosa_candidate_efficiency_followup.md) preserves these limits.

The reviewed successor is now integrated, with production CPU/CUDA and both
full-checkpoint gates passed; its fresh performance families are running. Complete
dispatch/staging attribution remains unresolved. The repeated partial tail
benefit does not establish a uniform or isolated net speedup. These diagnostics do
not replace accepted report/source results. New diagnostic results stay outside
experiment deliverables unless they meet an explicit
experiment purpose and measurement contract. A performance change must pass
correctness and replace all affected results with fresh identities before its
benefit is published. No new Q128 fetch candidate is selected.
