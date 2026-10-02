# GR serving acceptance review

**Latest researcher correction (2026-10-02):** ECHO's implementation has unresolved
MFU and cache-policy problems, and the current DeepSeek implementation's MFU is
also not yet correct. The old DeepSeek HBM/ECHO/serial-sparse/dense-prefetch
comparison is therefore not an accepted baseline comparison. Its ordering must
not support claims about baseline quality, offload overhead or ECHO's design
value. Numerical equality, declared budget checks and artifact consistency do not
validate implementation efficiency, the MFU accounting or cache-policy fairness.
The old budget auditor checks reservations and boundary tensor samples; coverage
of temporary DeepSeek cache scratch and its full execution peak still needs review.
The precise causes and fixes need a separate audit; this update makes no
operator change, supplies no corrected MFU and performs no GPU remeasurement.
Keep the recorded run IDs, source identities, measurements and selected report
artifacts until corrected affected runs pass acceptance and are published.
Retain reasonable slow controls in the corrected comparison.

The earlier correction also remains: the user rejected the nominal population
sweep and absence of revisit misses as an answer to cache-capacity scaling.
The old trace does not establish active-user scaling. IID heat sampling without
the default revisit cap and a heat-free sequential schedule have CPU preparation
checks, but no accepted replacement GPU result. The sequential 16-user,
32-request 64K attempt `gr_serving_h200_20261002_sequential_u16_t32_h64k_01`
failed the final source-identity guard; its rows are not experiment results.
[The rerun record](gr_serving_sequential_rerun.md) describes a proposed isolated
run 02, not an accepted run. Replacing the trace alone would not resolve the
newly identified ECHO/DeepSeek baseline problems.

Updated 2026-10-02 from accepted artifacts and the independent CPU auditor.
This replaces the obsolete run02/05 review and its former 24-case, 1 GiB contract.
Formal latency data, native L31 profiles and report publication for 4K, 16K and
64K passed the original numerical, measurement and publication checks described
below. Those checks remain evidence about the frozen runs; they do not override
the subsequent baseline and workload corrections above. The 64K records use
their original six-request trace. GR representativeness, task quality, baseline
validity and general serving value remain open.

The original review agent executed the CPU artifact audit and read-only checks.
GPU profiling belonged to the separate profile agent. The latest status edit
only reconciles the documents and checks their links; it changes no model,
operator source or measurement artifact.

## Retained trace protocol and evidence state

Both models use seven configured populations: 1/8/32/64/128/256/512. Requests are
serial, with a fixed per-user history and changing 128-token candidates. The
Beauty interaction-count weights drive seeded sampling; after a user's first
visit and at most eight revisits, it leaves the eligible population. There is no
forced initial tour and no equal per-user quota. Every scheme receives the same
saved token trace within a model/population.

The common allowed cache budget is 4 GiB HBM plus 16 GiB local CPU DRAM. It is an
artificial quota below the H200's physical capacity. It includes retained KV,
derived/indexer records, mappings, staging, scratch and pending append capacity.
Weights and ordinary activations are outside this cache quota. Admission uses
conservative per-session reservations; both tiers are released on whole-user LRU
eviction. The candidate suffix is truncated after successful execution.

| History + candidate | Formal run ID | Requests per population | DeepSeek sparse slots | Original checks; baseline/workload corrections above apply |
|---|---|---|---:|---|
| 4096 + 128 | `gr_serving_h200_20261002_h4k_01` | 9 for N=1; 32 otherwise | 4096 | Accepted 56 cases / 1608 rows; independent audit and L31 profile accepted; report published |
| 16384 + 128 | `gr_serving_h200_20261002_h16k_01` | 9 for N=1; 32 otherwise | 8192 | Accepted 56 cases / 1608 rows; independent audit, L31 profile and report publication accepted |
| 65536 + 128 | `gr_serving_h200_20261002_h64k_01` | 6 for every population | 32768 | Accepted 56 cases / 336 rows; independent data audit, L31 profile and report publication accepted |

The 64K run explicitly sets `allow_empty_revisits=true`. Its short trace has
no revisits at N=32/128/256/512: all 32 corresponding model/scheme scopes retain
count zero, null latency statistics and null hit rate, with blank CSV values.
Neither first-visit latency nor zero substitutes for a missing revisit estimate.
NOSA uses an explicit context-generation override beyond its default 32768-token
format boundary; this is not evidence of recommendation quality at 64K.

Recorded formal elapsed time is approximately 8.258 minutes for 4K and 31.983
minutes for 16K, and 32.797 minutes for 64K. These are run wall durations, not
summed request latency or a claim that a strict combined 30-minute deadline was
achieved. Different request caps and DeepSeek slot capacities must remain visible
in cross-history reports;
the history sweep is not a single-variable latency experiment.

## Numerical, input and budget acceptance

The 16K and 64K auditors exited successfully and wrote, respectively,
`experiments/gr_serving/output/data/gr_serving_h200_20261002_h16k_01/audit.json`
and `experiments/gr_serving/output/data/gr_serving_h200_20261002_h64k_01/audit.json`.
Both commands included the expected run ID and independent expected source digest.
The 4K audit is published in
[audit.json](../../../experiments/gr_serving/report/h4k/audit.json).
The independently checked numerical coverage is:

| History | Cases | Requests / correctness records | Exact non-HBM comparisons | Saved HBM tensors | Finite HBM elements checked |
|---:|---:|---:|---:|---:|---:|
| 4096 | 56 | 1608 | 1206 | 402 | 289,800,192 |
| 16384 | 56 | 1608 | 1206 | 402 | 289,800,192 |
| 65536 | 56 | 336 | 252 | 84 | 60,555,264 |

Every accepted history passed these gates:

- 56 complete model/scheme/population cases, with exactly one correctness record
  per measured request. All non-HBM comparisons have exact BF16
  element equality, zero absolute/relative error and `atol=rtol=0`.
- Full candidate hidden shapes `[128,7168]` for DeepSeek replay and `[128,4096]`
  for NOSA. Every saved HBM tensor was inspected on CPU. HBM's own records are
  self-comparisons; they are not counted as independent non-HBM comparisons.
- Recomputed input/prefix/candidate token hashes, fixed per-user prefixes,
  changing candidates, contiguous visits, exact scheme pairing, and independent
  seeded heat/cap replay. Population size is not confused with observed users.
- Every reservation, post-truncate allocation and pre-cleanup allocation within
  both byte caps. Independent reuse-distance/last-visit ordering agrees with every
  hit, exact eviction victim, retained-user count and total reservation.
- Independently checked all/first/revisit summaries and every per-request CSV
  value, including cache-miss revisits. Phase sums agree with request latency.

These checks audit persisted numerical records and saved HBM references; they
are not a second execution of the non-HBM models. Full non-HBM output tensors
are not persisted. Correctness is established for the measured candidates and
frozen implementation, not for task quality or arbitrary serving workloads.

The accepted 4K and 16K histories have identical visit coverage per scheme:

| Configured users | Requests | Observed users / first visits | Revisits | Observed maximum revisits |
|---:|---:|---:|---:|---:|
| 1 | 9 | 1 | 8 | 8 |
| 8 | 32 | 8 | 24 | 8 |
| 32 | 32 | 21 | 11 | 2 |
| 64 | 32 | 21 | 11 | 3 |
| 128 | 32 | 28 | 4 | 2 |
| 256 | 32 | 30 | 2 | 1 |
| 512 | 32 | 29 | 3 | 1 |

At N=128/256/512, revisit p95/p99 comes from only 4/2/3 observations. It is a
single-trace descriptive interpolation, not a stable service-tail estimate or
an independent-run confidence interval.

The 64K coverage is identical across both models and all schemes, with six
requests in every case:

| Configured users | Requests | Observed users / first visits | Revisits | Observed maximum revisits |
|---:|---:|---:|---:|---:|
| 1 | 6 | 1 | 5 | 5 |
| 8 | 6 | 4 | 2 | 2 |
| 32 | 6 | 6 | 0 | 0 |
| 64 | 6 | 5 | 1 | 1 |
| 128 | 6 | 6 | 0 | 0 |
| 256 | 6 | 6 | 0 | 0 |
| 512 | 6 | 6 | 0 | 0 |

Only N=1/8/64 have revisit observations, with 5/2/1 samples respectively.
The population sweep does not imply that all configured users were visited.
The independently replayed heat schedule, all 336 CSV rows and all 168 summary
groups passed; an additional small-artifact check agreed with metadata coverage,
all counts/means, and all 1176 JSON/CSV latency/hit-rate cells.

## Recorded observations and current interpretation limits

DeepSeek latency and reservation values below remain observations of the old
implementation. Their original numerical/artifact acceptance does not establish
a reasonable baseline after the MFU and cache-policy correction. NOSA's separate
numerical and internal-profile records are not invalidated by that correction;
their limited trace coverage and softmax-only overlap boundaries still apply.

Across histories, NOSA overlap has a lower all-request mean than serial sparse
in 20 of 21 configurations: 6/7 at 4K, 7/7 at 16K and 7/7 at 64K. Its revisit
mean is lower in only 2/7, 1/7 and 0/3 nonempty scopes, respectively. These counts
were independently checked against the accepted summaries. Full-request latency
includes cold-prefix construction, whose workload differs from retained-prefix
candidate execution. The measured full-trace improvements remain valid even
though the revisit profiles fall below 90%; those profiles cannot isolate the
cause of the improvements or establish a stable revisit benefit. NOSA HBM has
the lowest all-request mean in all 21 configurations under this cache quota.

**4K:** all measured revisits hit their user prefix sessions. NOSA HBM admits
29 sessions and the N=256 trace evicts one user, but that user does not revisit.
This run does not establish an offload revisit-capacity benefit. All-request
means rise mainly with the first-visit fraction. NOSA overlap does not show a
stable revisit-latency improvement over serial sparse.

**16K, DeepSeek:** HBM admits 19 sessions; ECHO and serial sparse admit 36 each,
and dense prefetch admits 68. HBM has 34 evictions across the seven separate
cases, but zero revisit misses. All offload methods likewise have zero revisit
misses. Therefore larger offload capacity has not produced an observed DeepSeek
revisit-capacity advantage in this trace. Revisit means are 36.51–37.65 ms for
HBM, 36.19–39.80 ms for dense prefetch, 44.84–46.91 ms for serial sparse and
58.62–61.84 ms for ECHO. These include the old implementation's full cache,
gather, launch and synchronization costs. With unresolved MFU and ECHO
cache-policy problems, they cannot support a baseline or design ranking.

**16K, NOSA:** HBM admits seven sessions while each offload scheme admits 31.
HBM has 108 evictions and 13 revisit misses across 63 revisits in seven separate
cases; each offload method has zero evictions and zero revisit misses. The
observed revisit benefit is therefore tied to avoiding prefix reconstruction
under this quota and access trace.

| Users | Revisits | HBM revisit misses | HBM revisit mean (ms) | Serial-sparse revisit mean (ms) |
|---:|---:|---:|---:|---:|
| 1 | 8 | 0 | 29.96 | 40.00 |
| 8 | 24 | 1 | 53.53 | 40.05 |
| 32 | 11 | 3 | 187.27 | 40.49 |
| 64 | 11 | 3 | 187.88 | 41.15 |
| 128 | 4 | 3 | 465.31 | 41.41 |
| 256 | 2 | 2 | 611.78 | 42.61 |
| 512 | 3 | 1 | 223.27 | 41.72 |

This does not imply lower overall service latency for the sampled trace. NOSA
HBM has the lowest all-request mean in every 16K population, because first
visits dominate the larger populations and offload prefix construction costs
remain included. The accepted trace supports a revisit-capacity observation,
while its all-request averages favor HBM. NOSA overlap again has no consistent
revisit improvement over serial sparse.

**64K:** DeepSeek's admitted session capacities are 5 for HBM, 9 for ECHO and
serial sparse, and 17 for dense prefetch. NOSA admits one HBM session or seven
sessions with any offload scheme, with offload limited by DRAM. DeepSeek HBM has
four evictions and NOSA HBM has 27 across their seven independent cases; offload
has none. Nevertheless, every scheme hits all eight observed revisits across
the seven cases. No evicted user returns within this six-request trace, so the
64K run does not demonstrate an offload revisit-capacity benefit.

NOSA HBM has the lowest all-request mean at every 64K population and the lowest
revisit mean in the three nonempty scopes. NOSA overlap has a higher revisit mean
than serial sparse in all three scopes, despite lower all-request means that
also include prefix construction. DeepSeek HBM has the lowest revisit mean in
the three nonempty scopes; dense prefetch has slightly lower all-request means
at N=8/32/64/128/256/512, while HBM is slightly lower at N=1. Those DeepSeek
orders are retained old-run observations and cannot support a baseline ranking
while MFU and ECHO cache-policy problems remain unresolved. No cross-history
scaling result follows from these short traces.

## Source and publication provenance

All accepted latency audits verify 153 current/saved source files, the manifest
aggregate and three clean tracked submodules. All three runs record base Git revision
`16dc058014ee0474a0fa0893ad89271b890aeb92`; the source manifest, rather than the
base revision alone, identifies the measured workspace changes.

| Artifact | Source-manifest SHA-256 |
|---|---|
| 4K formal latency | `67f91c34ead43d4056500661caa1c67e8ef8a50bc25327e081a0561bbab43060` |
| 16K formal latency | `45511acf6b7ec0e0a9bf3e21a4f9b1c9b74f88a46f03303f3f93cc9460dfd44c` |
| 64K formal latency | `45511acf6b7ec0e0a9bf3e21a4f9b1c9b74f88a46f03303f3f93cc9460dfd44c` |
| 4K, 16K and 64K independent profiles | `cd72caf5edbac19a54d162e26d3339b05dd0671048162d970729669af5d0c48b` |

The 4K and 16K latency manifests differ only in `src/profile.py` and
`tests/test_profile.py`, which are not executed by formal latency measurement.
Performance/runtime source bytes are unchanged. The profile snapshot additionally
captures imported analysis sources, `pyproject.toml` and `uv.lock`, so its larger
166-file manifest legitimately has a different aggregate digest. The 4K profile
explicitly records the two diagnostic changes; the 16K and 64K profiles have
empty allowed-drift lists. No old number is represented as a result from another
source ID.

Verified tracked dependency commits:

| Dependency | Recorded gitlink = actual HEAD |
|---|---|
| DeepGEMM | `b64107f2b9599ca76445b7f62eedf66bae1d095b` |
| DeepJIT | `e5bdee2bc4ca519eba00cfc5f0c6e950e6a96a16` |
| CUTLASS | `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8` |

The 4K [provenance record](../../../experiments/gr_serving/report/h4k/provenance.json)
was independently checked: eight copied artifacts match their original outputs
and declared hashes, six derived-artifact hashes match, two profile copies match,
and the separately recorded renderer hash matches. Raw analysis files remain
unchanged. The readable summary changes categorical spacing and panel scales;
it is derived from verified summary values. Figure rendering and copying are
publication work, not additional performance samples.

The 16K [provenance record](../../../experiments/gr_serving/report/h16k/provenance.json)
and publication passed a separate read-only check. All 17 declared report files
are present with no undeclared files: eight copied artifacts are byte-identical
to accepted outputs and match declared hashes; six derived-artifact hashes and
two profile copies match; the supplemental renderer hash matches and its source
is byte-identical to the 4K renderer. Original artifacts also agree with the
independent auditor's evidence hashes where recorded.

The review independently recomputed all seven population-coverage rows and eight
cache/memory rows from raw request rows, metadata and audit values, including
evictions, revisit misses, reservations and allocation maxima. All six 16K README
tables agree with their sources, including 112 latency cells with mean/p95 rounded
to two decimals, memory unit conversions and profile values. All 36 local README
links/anchors resolve; none links ignored output artifacts. The conclusions
correctly retain NOSA HBM's lowest all-request mean in all seven populations,
distinguish the revisit-capacity observation from full-trace performance, and
state the small revisit samples, null serial-profile ratios, artificial cache
quota and activation/allocator limits. At that publication check, 64K evidence
was still pending; its completed publication is reviewed below.

Visual-inspection attribution is separate from these data checks: the root agent
viewed the readable summary; the publishing agent's provenance records inspection
of all 14 request panels and four readable-summary panels. This review checked
those records and artifact hashes, not the images themselves. The 16K publication
is accepted; no tensor audit, test suite or GPU execution was repeated for this
publication check.

The 64K [provenance record](../../../experiments/gr_serving/report/h64k/provenance.json)
and publication passed the final independent read-only check. Exactly 19 declared
files are present. Eight copied artifacts match their accepted originals and
declared hashes, eight derived-artifact hashes match, and both profile copies
match their accepted originals. The standalone renderer and its recorded 4K
ancestor match their source hashes; input summary/per-request identities and
the NOSA context override agree with the original run. The accepted 4K/16K
publication hashes remain valid.

All seven 64K coverage rows and eight cache/memory rows were recomputed from raw
measurement rows, metadata reservations/allocator peaks and audit maxima. All
five 64K README tables match: 112 latency cells include 32 explicit `— (n=0)`
cells, and memory conversions, capacities, profile bytes/counts and ratios agree
with their sources. All 54 local README links/anchors resolve, with no links to
ignored output artifacts. The report preserves the six-request cap, actual
coverage, all-hit revisit boundary, 64K context override, one-user profile,
artificial quota and memory-measurement limitations.

This reviewer inspected all four readable-summary panels and all fourteen
readable per-request panels. Gray columns correctly mark absent revisits, curves
have gaps instead of substituted zero values, and request ticks show the exact
IDs 0–5. Renderer inspection confirms direct use of accepted means/p95 and all
per-request rows. The original SVG limitations are disclosed, with raw SVGs
preserved and separate readable views. The root agent also viewed the readable
summary; the publisher's provenance additionally records the original fourteen
per-request panels. No measurement, tensor audit or test suite was repeated for
publication acceptance.

## Native NOSA layer-31 profiles

Accepted profile runs are `gr_serving_h200_20261002_h4k_profile_02`,
`gr_serving_h200_20261002_h16k_profile_01` and
`gr_serving_h200_20261002_h64k_profile_01`. The 4K and 16K profiles use the saved
eight-user workload's first three revisits, request IDs 2/3/7. The 64K profile
uses the saved one-user workload's first three revisits, request IDs 1/2/3;
its six-request eight-user trace has only two revisits. Every profile has one
serial-sparse and one overlap sample per request. Each sample starts from an
independent empty sparse-prefix cache; it is not a replay of the formal LRU
residency. Numerical controls compare
all candidate hidden elements against uninstrumented execution and formal HBM.
For both 16K and 64K, the profile agent verified all 18 full `[128,4096]` hidden
comparisons, actual selection, raw trace identities, unique-copy/stripe accounting
and all 166 saved/current profile source identities. Its 64K CPU `--verify-only`
check passed with CUDA hidden. This review separately inspected metadata and
analysis, including all 18 exact numerical records, formal-source linkage,
sample identities, null serial ratios, stripe counts and ratio arithmetic; it
did not repeat the profile agent's tensor audit or GPU execution.

| History | Population | Request ID | Unique historical K+V bytes | Page-envelope ratio | Nonempty stripe-copy ratio |
|---:|---:|---:|---:|---:|---:|
| 4096 | 8 | 2 | 4,161,536 | 72.8733% | 72.8733% |
| 4096 | 8 | 3 | 4,161,536 | 74.3360% | 74.3360% |
| 4096 | 8 | 7 | 4,161,536 | 75.2381% | 75.2381% |
| 16384 | 8 | 2 | 6,881,280 | 78.3693% | 78.3693% |
| 16384 | 8 | 3 | 7,176,192 | 79.8333% | 79.8333% |
| 16384 | 8 | 7 | 7,274,496 | 81.2149% | 81.2149% |
| 65536 | 1 | 1 | 7,536,640 | 76.8733% | 76.8733% |
| 65536 | 1 | 2 | 7,274,496 | 73.1752% | 73.1752% |
| 65536 | 1 | 3 | 7,536,640 | 79.2700% | 79.2700% |

Every overlap sample is below the 90% requirement; these are valid below-threshold
results and remain retained. For each sample, the page envelope equals the min/max
of its nonempty stripes. The two global interval unions happen to coincide,
with zero envelope-only union, explaining the equal ratios. For 16K, the three
samples contain 210/219/222 historical pages and 1680/1752/1776 nonempty stripes.
For 64K, these counts are 230/222/230 pages and 1840/1776/1840 nonempty stripes.
The changed profile population and requests preclude treating these samples as
a controlled history-only comparison.

The numerator intersects copy intervals with measured **consumer softmax**
intervals, not complete attention work. Ratios are not PCIe wire occupancy,
full-attention overlap, or overall serving-latency hiding. Serial samples lack
math-window instrumentation and retain null ratios. Request-level performance
comes from the separate uninstrumented formal runs. Profile storage adds
212,992 bytes of trace allocation per 16K sample and 655,360 bytes per 64K sample,
and retains selections; this one-session diagnostic has no LRU admission and is
not a formal-budget test.

## Limitations that must remain in the report

- The researcher has identified ECHO MFU/cache-policy problems and DeepSeek MFU
  problems. Audit useful versus executed FLOPs, precision-specific peak, timing
  boundaries and compute efficiency, together with slot retention/clearing,
  exact recall, duplicate transfers and query splitting. These are audit topics,
  not diagnosed causes. Correct and remeasure the affected model-local controls
  before accepting baseline rankings. Existing output equality does not close
  this performance-baseline gate.
- DeepSeek is a checkpoint workload surrogate: ten independent copies of source
  dense layers 0/1/2 with corresponding copied hidden/residual inputs, no MoE,
  7,827,793,408 parameters including endpoints. It is neither a trained 8B model
  nor the full 61-layer DeepSeek validation. Its last-token LM head differs from
  NOSA's hidden-only endpoint; latency rankings are within each model.
- NOSA is the full 32-layer sparse model with checkpoint CIS semantics. Its
  offload path has one full logical-layer staging range; whole-session LRU does
  not imply a NOSA token/page-slot eviction system. Dense prefetch means dense
  history transfer with unchanged sparse attention selection.
- DeepSeek exact unions can exceed sparse slots and require query-consumption
  splitting. ECHO may clear historical slots before fused prefetch. A user-prefix
  hit is not a token/page hit rate, and “DRAM tier” does not establish per-row
  DRAM transfer volume. Extra Python/gather/launch costs cannot all be attributed
  to DRAM bandwidth or fusion itself.
- Cache samples, conservative reservations and process CUDA allocator peaks are
  distinct quantities. Allocator peaks may include a previous model still live
  at a model transition and retained allocator blocks. They are not isolated
  model peaks or physical process HBM peaks. Ordinary activation has no separate
  peak measurement; output tensor payloads are not the full activation footprint.
- This is synthetic fixed-history serving with a borrowed heat curve, without
  real recommendation labels, concurrent requests, network work, arrival sleeps
  or queueing measurements. Neither representativeness nor task quality follows
  from exact model execution. Tail estimates from very few revisits remain weak.

## Current completion boundary and follow-up

The three old matrices and their native profiles/publications passed their
original checks, including 2664 exact non-HBM candidate-hidden comparisons.
These are retained evidence for the recorded execution and inputs. The latest
researcher correction reopens acceptance of the DeepSeek/ECHO baseline, and the
earlier workload correction leaves scale/capacity evaluation incomplete.
The failed sequential run provides no accepted replacement evidence.

Next work is to audit and fix ECHO MFU/cache policy and DeepSeek MFU, identify
the affected comparisons, and then remeasure with a frozen source snapshot and
fresh run IDs. The comparison must preserve exact selection and numerical
correctness, equivalent cache budgets and declared output/timing boundaries.
Independently audit measurement completeness before replacing the affected
reports and cleaning their old artifacts. Workload redesign must expose actual
users, revisit misses and reuse distances; replacing one trace does not by
itself validate active-user scaling or the GR scene.

Research state and tasks are maintained in [status](../../status.md),
[roadmap](../../roadmap.md), [sources](../research-supervisor/sources.md) and
[updates](../research-supervisor/updates.md). Their current interpretation must
include these corrections. No Supervisor file was edited by this review.
GR remains a candidate scene; HBM/CPU DRAM remains the scope. DeepSeek remains
an untrained checkpoint workload surrogate, and NOSA remains a full sparse model
with logical-layer staging rather than finite page slots.

Withdrawn runs 01–05, failed/calibration diagnostics and the failed sequential
run are not comparison groups or valid experiment deliveries. Valid NOSA
below-threshold profiles and reasonable slow controls stay visible within their
measured scope. No GPU test, profile or experiment was rerun for this status edit.
