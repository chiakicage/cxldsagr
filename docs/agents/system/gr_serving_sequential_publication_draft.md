# Sequential 64K engineering documentation draft

2026-10-02. Prepared while the sequential measurement is running. This file proposes replacements
for `gr_serving_task.md` and `gr_serving_review.md`; neither original file has been changed.
It contains no accepted new performance result. Root must complete the evidence ledger below,
review the resulting text, and publish the new 64K experiment before promoting these drafts.
Measurement/model/operator source remains frozen. No test or GPU run was started for this draft.

## Proposed replacement for gr_serving_task.md

### Persistent GR serving: current execution contract

The current requested delivery is one controlled **65536-token history +128-token candidate**
case with **16 users and 32 requests**. User IDs are exactly `0,1,...,15,0,1,...,15`.
The allowed cache budgets are **4 GiB HBM and 64 GiB local CPU DRAM** for every scheme in both
models. No heat curve or stochastic user sampling participates in this sequence. The content
seed is 42; it affects synthetic history/candidate text, not access order.

The launch identity is `gr_serving_h200_20261002_sequential_u16_t32_h64k_01`.
Final measurement, audit, native-profile and publication statuses must be filled from the
accepted artifacts before this document can state completion. The original industrial
1024-user / 4096-request attempt was stopped during input generation without measured results.
The smaller sequential case is a separately requested experiment; it does not constitute a
completed industrial-heat experiment or a new 4K/16K measurement.

DeepSeek runs its checkpoint workload surrogate: ten independent dense-block copies of source
layers 0/1/2, each supplied with its corresponding source hidden/residual input, plus embedding,
final norm and LM head. It contains 7,827,793,408 parameters and no MoE execution. Its four
schemes are HBM, ECHO, serial sparse and dense layerwise prefetch. NOSA runs all 32 checkpoint
layers with its complete sparse selection/CIS semantics; its four schemes are HBM, serial sparse,
dense layerwise prefetch and native sparse-fetch/attention overlap. Prefix chunk size is 1024;
DeepSeek sparse offload uses 32768 main-KV slots. DeepSeek executes all candidate hidden and the
last-token LM head; NOSA returns all candidate hidden only. Comparisons remain within each model.

Each scheme starts its measured sequence from an independently empty cache after separate
warmup. A request includes cache admission/eviction, any cold or evicted-history reconstruction,
all candidate work, synchronization, truncation and cleanup. Weight loading, compilation,
input generation and between-request correctness checking/serialization are outside the
per-request timer. Verify the warmup count and timer parameters in final metadata before
reporting them as measured configuration. Full process wall duration is separate from summed
request latency and must not be replaced by the earlier runtime estimate.

Global whole-user LRU releases both tiers on eviction. Admission first reserves conservative
model-declared cache capacity, including main KV, indexer/derived records, staging, mapping,
scratch and pending append. Actual cache storage is checked against that reservation. All layers
and asynchronous work must finish before commit or stage reuse; successful candidates truncate
back to the fixed history. A second-pass miss is still a revisit and includes reconstruction in
its latency. Weights, ordinary activations, sampled cache storage, conservative reservations and
CUDA allocator peaks remain distinct quantities.

The expected complete sequence has 16 first visits and 16 revisits per case, with visit indices
zero and one. The expected matrix is eight cases and 256 measured requests/correctness records;
192 comparisons are non-HBM against saved HBM candidate hidden. These are predeclared counts,
not acceptance findings until the completed run audit confirms every row. Workload metadata has
`sampling="sequential"`, null heat dataset/field/hash and explicit synthetic IDs. Equal unit
weights are inert GR interface placeholders; they are not measured popularity or random
probabilities. The independently checked input signature is
`c3511ba7f3fb306def87447c0b7d7c545e253f5f113fb98288987612befedda9` and must match the formal trace.

### Retained results and replacement boundary

| Geometry | Retained latency run | Retained profile run | Boundary |
|---|---|---|---|
| 4096 +128 | `gr_serving_h200_20261002_h4k_01` | `gr_serving_h200_20261002_h4k_profile_02` | Original Beauty/capped short trace and 4/16 GiB budgets; no new execution |
| 16384 +128 | `gr_serving_h200_20261002_h16k_01` | `gr_serving_h200_20261002_h16k_profile_01` | Original Beauty/capped short trace and 4/16 GiB budgets; no new execution |
| 65536 +128 | `gr_serving_h200_20261002_sequential_u16_t32_h64k_01` | Must fill the actual accepted new profile run ID | Current sequential 16/32 trace and 4/64 GiB budgets; acceptance still pending in this draft |

The old `gr_serving_h200_20261002_h64k_01` and
`gr_serving_h200_20261002_h64k_profile_01` outputs and `report/h64k/` assets remain intact until
the replacement's numerical, measurement, independent audit, native-profile and publication
gates pass. Then replace the old 64K report and remove those two old runs' data/log/profile
directories in the same publication update. Keep the 4K/16K reports, raw outputs and provenance
unchanged. The exact inventory is in [the publication inventory](gr_serving_industrial_publication.md).

Do not report old/new geometries as a common controlled history sweep. Their request orders,
budgets and earlier population caps differ. Remove the old all-three-geometries completion/count
claims from this current contract. In particular, `168 cases`, `3552 measurements`,
`2664 non-HBM comparisons`, `20/21 configurations`, and the old nine-profile aggregate are not
current combined evidence after the 64K replacement. Any retained historical 4K/16K observations
must name their original run and boundary individually.

### Validation, native profile and research handoff

CPU workload/order preparation has passed; those checks do not substitute for the new GPU run.
The final evidence must independently authenticate all input identities, every numerical record,
saved HBM references, exact LRU states, cache budgets, source/dependency identity and derived
report values. Record the new formal source digest, independent audit and accepted profile
identity from their artifacts, rather than carrying over the old 64K source digest.

The new native diagnostic must use the first three saved NOSA revisits, request IDs 16/17/18,
and the same physical GPU as the formal run. Each serial/overlap sample builds its own sparse
prefix from an empty diagnostic session. It executes the full model and compares complete
candidate hidden to both an unprofiled control and the formal HBM reference; only layer 31 is
instrumented. This diagnostic does not recreate formal LRU occupancy or measure serving latency.
Preserve all valid samples, including any below 90% page/stripe overlap. Source freeze extends
through the audit and diagnostic.

After final publication root applies the current project Research Supervisor and records the
research implications in the existing state/roadmap. This engineering contract must not declare
the tentative GR scene, task quality, general user scaling, or cross-model design superiority
settled by the sequential run. NOSA whole-user LRU still does not implement finite token/page
slots. No network, queueing, concurrency, CXL/RDMA or real arrival replay is measured.

## Proposed replacement for gr_serving_review.md

### GR serving acceptance review: sequential 64K case

This review covers `gr_serving_h200_20261002_sequential_u16_t32_h64k_01` only. It must distinguish
the formal run, independent artifact audit, native-profile verification, report publication and
Supervisor handoff; each status and evidence path remains pending until verified. Acceptance of
the retained 4K/16K reports belongs to their original run identities and does not prove anything
about the new 64K results. The earlier industrial attempt has no measured result.

The explicit contract is 65536 +128 tokens, 16 users traversed twice in increasing ID order,
32 requests per scheme, and identical 4 GiB HBM / 64 GiB DRAM cache limits. The sequence has
16 first visits followed by 16 revisits; the first-visit/revisit label comes from visit index,
never cache residency. Heat metadata is null, placeholder user weights are not sampling
probabilities, and seed 42 affects only content. This controlled access order is not an
industrial request log or a representative workload claim.

### Evidence and numerical acceptance

The completed audit must confirm the predeclared eight model/scheme cases, all request IDs 0–31
once per case in order, 256 correctness records, 192 non-HBM comparisons, and 64 saved HBM
reference tensors. Full candidate hidden shapes are `[128,7168]` for DeepSeek and `[128,4096]`
for NOSA. HBM self-comparisons are not counted as independent non-HBM comparisons. Report
the actual exact/error/tolerance findings only after reading the audit and numerical records.

Every input/prefix/candidate hash and immutable per-user history must match the saved token rows.
Second-pass candidates must differ, previous-request IDs must be 0–15 for requests 16–31,
and independent arithmetic replay must give user `request_id % 16` and visit
`request_id // 16`. The access signature must match both model workloads. Audit every reservation,
post-truncate sample and pre-cleanup allocation against the two byte caps, and recompute exact
LRU hits/victims from the actual reservation-derived capacities. Planning hit/miss predictions
are not observed outcomes.

CPU inspection of every saved HBM tensor confirms shape, dtype, finite elements and file
identity. The non-HBM results are checked through their persisted numerical records; this is not
an independent model rerun. Preserve that limitation and the distinction between complete-hidden
comparison and recommendation correctness.

### Findings and measurement limits

Fill a model-local table for each scheme using accepted all/first/revisit request counts,
mean/median/p95/p99, cache hits/misses, eviction counts, reservation bytes and observed allocation
peaks. Include measured cold-prefix/candidate/admission/cleanup contributions where recorded.
Use the audited whole 32-request trace for the primary comparison, and report second-pass
behavior separately. Do not substitute internal interval ratios for request latency or infer
that avoided reconstruction necessarily wins on the complete trace.

No latency ranking, speedup, measured capacity benefit or overlap percentage is prewritten in
this draft. The old six-request 64K values and the 20/21 three-geometry statement must not appear
as findings of this run. Sixteen revisits are descriptive samples from one deterministic trace;
their p95/p99 are not stable production-tail estimates or confidence intervals. Changing the
64K schedule and DRAM budget prevents a controlled comparison with the old 4K/16K histories.

Retain the model endpoint distinction and workload-surrogate limitation. Record model weights,
sampled cache use, conservative admission reservations and allocator peaks separately. Do not
claim isolated activation peaks or physical process-memory peaks from CUDA allocator statistics.
Check whether model-transition retention affects reported peaks before attributing them to an
individual model. User-session hits are not DeepSeek record-slot hit rates, and NOSA still
uses one full logical-layer staging range for sparse offload.

### Native-profile acceptance

Fill the actual new profile run ID/source digest and authenticate its formal-run linkage.
Verify request IDs 16/17/18, user IDs 0/1/2, original token identities and saved HBM references.
The expected six serial/overlap records have 18 full-hidden comparisons; read their actual
verification status before marking acceptance. Record unique historical bytes, page/nonempty
stripe counts, extra tracing storage, raw interval identities and consumed selections.

For each of the three overlap samples, report both page-envelope and nonempty stripe-copy
ratios independently. A page envelope must equal its nonempty stripes' minimum start and
maximum end; gaps are not real copy. The intersection uses actual consumer softmax intervals,
not the kernel lifetime or all attention work. Both ratios must be >=0.9 in every sample before
claiming 90%; retain valid below-threshold results. Serial unmeasured math ratios remain null.
Diagnostic empty-prefix construction and tracing memory do not inherit formal LRU/budget claims.

### Source identity, publication and handoff acceptance

Read the final formal and profile source manifests, current/saved source bytes, dependency
gitlinks and any explicit diagnostic-only differences. The launch-time aggregate source digest
recorded in the execution handoff is
`75909d12fb212c7a80277bae5e52acc340136fd88a9848da7bc51cb8e5750e58`; this is an expected identity,
not an accepted source audit in this draft. Source-file counts and final digests must come from
the new audit, not the old 153/166-file checks.

Match every copied report file to accepted originals and every derived table/figure to its
recorded generation inputs and SHA256. Independently recalculate README tables and identify the
actual renderer version. Inspect all figure panels for units, labels, sample counts and readable
request indices; one population is not a scaling plot. Verify report inventory and relative
links, and keep ignored output paths as plain code paths. Confirm old 64K retirement only after
new publication, and confirm untouched 4K/16K provenance remains valid.

Report the outcome of the new Supervisor handoff only after root performs it and its affected
research statements are checked against this run's accepted evidence. Do not reuse the earlier
"all three matrices complete / no gate remains" conclusion. The controlled small group does
not satisfy the deferred industrial-heat or unrun 4K/16K requests by implication.

## Evidence ledger to complete before promotion

| Required field | Source to inspect | Current draft status |
|---|---|---|
| Formal run ID, terminal exit, accepted metadata, exact parameters and elapsed wall duration | Launcher completion plus new `metadata.json` | Run ID known; terminal/result acceptance not filled |
| Actual case/row/correctness/reference counts | New independent audit and complete JSONL/reference set | Expected 8 /256 /256 /64; actual acceptance not filled |
| Exact non-HBM comparisons, max error and tolerances | New correctness rows and independent audit | Expected 192 comparisons; findings not filled |
| All input/access/workload hashes and no-heat sequential provenance | Both formal `workloads/*/16/` directories and independent audit | CPU access signature known; formal audit not filled |
| Model descriptions, endpoints, checkpoint/build/GPU identities | New model metadata, source manifest and audit | Contract known; final provenance not filled |
| Formal source aggregate, membership count, dependency commits and audited source status | New formal source manifest plus audit | Launch digest recorded above; accepted identity not filled |
| Each model/scheme's all/first/revisit counts and mean/median/p95/p99 | Accepted analysis plus independent recomputation from measurements | No measured values filled |
| Exact hits, misses, evictions, LRU capacity and reservations per scheme | Independent case audit and request rows | No observed values filled |
| Cache sampled maxima, reservations, weights, allocator allocated/reserved peaks and attribution | Accepted audit and metadata | No measured values filled |
| New native profile run ID, source digest, formal-source comparison and GPU identity | New profile metadata/source snapshot | Not yet filled; do not reuse old profile ID |
| Six sample identities, selections, 18 hidden checks, unique bytes/pages/stripes and extra storage | New profile analysis, saved samples and verifier result | Expected request IDs 16/17/18; findings not filled |
| Both ratios per overlap sample, page-envelope equality, serial null semantics and 90% gate | Reanalysis of actual raw native traces | No values or gate verdict filled |
| Accepted report inventory, copied/derived file hashes, renderer identity, README value/link checks | New `report/h64k/provenance.json` and publication reviewer record | Not yet filled |
| Visual inspection attribution and reviewed panels | Actual image review and reviewer record | Not yet performed in this draft |
| Old 64K report/output replacement and unchanged 4K/16K provenance | Current files after accepted publication | Not performed; preserve all old results meanwhile |
| Supervisor completion and affected status/roadmap/source claims | Root's later Supervisor run and resulting documents | Pending; this draft does not invoke Supervisor |

Root should replace pending phrasing only with verified results and remove this ledger from the
reader-facing final review once its evidence is carried by concrete links and concise findings.
Keep internal execution history separate from the research question or proposed paper narrative.
