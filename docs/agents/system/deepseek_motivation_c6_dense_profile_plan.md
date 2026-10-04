# C6 dense-prefetch diagnostic plan

Status: the C6 dense-only diagnostic is complete and independently audited.
The authorized GPU 0 profile exited successfully; GPU 0 was released before
the CPU analysis. Formal C6, twenty profile outputs, full graph attribution,
useful work, launch exposure and physical gather overlap all passed their
checks. This completes the diagnostic, not the overall optimization goal.
Publication reports remain unchanged. See the completed-result section below
and the [profile checkpoint](deepseek_motivation_graph_profile_checkpoint.md).

## Objective and source boundary

Measure the remaining fixed-pool dense-prefetch metadata work, actual
mapped-host gather overlap with computation, and matrix API MFU for the
candidate segment. Relate the sampled API result to matching formal C6
candidate and request latency without interpreting their difference as
isolated CPU overhead or guaranteed removable work.

Use the frozen tree `/tmp/deepseek-motivation-c6-frozen-5vbx7859`, formal run
`motivation_c6_20261004_u16_r2_01`, source-manifest SHA-256
`bd3e91265fed36f182d291b1bea3a659a2214c686e8236ed42da663c07775bff`.
The formal job was started by root as session 52572 and exited successfully.
Root relocated 1,573 verified files to the shared repository. Its accepted
data path is
`/mnt/ssd-wlcb/chenkaiqi/cxldsagr/experiments/deepseek_v32_motivation/output/data/motivation_c6_20261004_u16_r2_01`.
Working-tree C7 changes must not
enter this capture; a later C7 result requires separate validation and a new
run ID. The accepted C3 profile remains historical evidence for C3 only.

The frozen source already supports a single `--scheme dense_prefetch` capture.
`load_reference` checks all executed model/cache/operator/serving dependencies
against the formal source manifest. Profile provenance adds instrumentation
sources, so its aggregate source hash differs from the formal hash without
implying a different production implementation.

## Capture and acceptance

Executed profile ID: `motivation_c6_dense_profile_20261004_01`.
The command below ran from the frozen root under root's exclusive GPU 0 grant:

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash experiments/deepseek_v32_motivation/scripts/profile.sh \
  --run-id motivation_c6_dense_profile_20261004_01 \
  --reference-run /mnt/ssd-wlcb/chenkaiqi/cxldsagr/experiments/deepseek_v32_motivation/output/data/motivation_c6_20261004_u16_r2_01 \
  --scheme dense_prefetch --device cuda:0
```

The command uses the accepted original-repository reference while executing
the frozen source. The wrapper owns staging, profiler flags, exports,
classified output directories and failure handling. It must not be bypassed
to retain a failed run under `experiments/`.

The existing script records CUDA, NVTX and OS runtime activity with graph node
tracing and three CUDA profiler ranges:

| Trace | Scope | Request |
|---|---|---|
| 1 | Graph allocation and clone lineage | Setup only |
| 2 | Cold dense-prefetch request | Request 0 |
| 3 | First revisit after all sixteen users | Request 16 |

Three warmup requests and requests 0–16 produce twenty exact hidden/logit
comparisons against the accepted formal HBM outputs. Requests 1–15 execute
the real preparation trajectory without profiling. Require `accepted=true`,
twenty exact outputs, two request captures, one setup capture, verified source
and backend identities, and unchanged precision settings. Check cold miss and
revisit history-hit metrics. Setup graph replay must not enter request counts.

Graph-v2 attribution must discover all graph nodes through the recorded clone
lineage and count eager `v_expand` once per executed layer/block. Each finish
template must omit `v_expand`; every replay must have complete node membership,
matching executable/process/launch identity, and zero unattributed work.
For the intended 64 history chunks, ten layers, and one candidate batch per
request, the expected replay counts are 1,300 cold and twenty revisit. Actual
work verification remains authoritative if accepted configuration differs.
There should be 7,260 graph matrix API invocations and 660 eager expansions
across the two request captures under that configuration. MLA split counts
must follow actual execution while preserving useful work.

Record GPU process observations through the complete profile window. These
are discrete samples and do not establish continuous isolation between
samples. Hold other authorized GPU jobs while the diagnostic runs; its
instrumented wall time still does not replace the quiet formal measurement.

## Existing CPU analysis chain

Use the accepted C6 run's matching fresh FLOP analysis,
`motivation_c6_flops_20261004_01`. Preserve its actual input
identities and precision-specific reference peaks. Do not reuse C3 API times.
After capture acceptance, run from the frozen source with no visible GPUs:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  -m experiments.deepseek_v32_motivation.src.verify_profile_flops \
  --profile-run PROFILE_DATA --flops C6_FLOPS_DATA/flops.json
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  -m experiments.deepseek_v32_motivation.src.analyze_pipeline \
  --sqlite PROFILE_DATA/capture_2.sqlite \
  --sqlite PROFILE_DATA/capture_3.sqlite \
  --output-dir PROFILE_DATA/analysis
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  -m experiments.deepseek_v32_motivation.src.operator_mfu \
  --profile-run PROFILE_DATA --flops C6_FLOPS_DATA/flops.json \
  --output-dir PROFILE_DATA/analysis/operator_mfu
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  -m experiments.deepseek_v32_motivation.src.aggregate_mfu \
  --operator-dir PROFILE_DATA/analysis/operator_mfu \
  --flops-dir C6_FLOPS_DATA \
  --output-dir PROFILE_DATA/analysis/aggregate_mfu
```

Replace `PROFILE_DATA` and `C6_FLOPS_DATA` with accepted directories when
executing. The FLOP verifier and aggregate analysis derive schemes and capture
groups from data; they do not require all four schemes. A targeted frozen-tree
CPU check passed seven existing verifier/aggregation tests, and the profile
CLI exposed the required single-scheme option. This validates preparation,
not an unrun profile.

## Dense-specific attribution and overlap

Add a run-local CPU audit under the accepted profile's `analysis/`. It should
read raw SQLite independently of production attribution, preserve all GPU
activities, and save its source and input hashes beside its JSON/CSV output.
No production instrumentation change is required: existing scopes record
`PoolHistoryPrefetch.prefetch/wait/drain`, target-layer `host_gather`, cache
operations and native kernels, actual matrix calls and graph replays.

Report history and candidate separately, and distinguish cold candidate from
revisit candidate. Cold history can have no host misses while still executing
metadata calls. Sum per-call `host_gather.records/requested_bytes` and reconcile
them with the captured segment counters for dense requested/resident/fetched
records and actual transfer bytes. Gather runs as a mapped-host CUDA kernel,
so a missing HtoD memcpy row does not mean zero host traffic.

Use non-overlapping accounting categories to retain the complete request:

- Matrix APIs and their owned helper activities, with primary matrix kernels
  also reported separately.
- Host gather kernels and ordinary D2H/H2D copies, separated by actual kind,
  stream, target layer and segment.
- Metadata GPU work, including the native resident-history helper, logical
  range/miss construction, slot allocation, map updates, protection/stamping,
  selection and truncate/cleanup. Preserve full kernel names and owning scopes;
  do not infer that every pool scope is a transfer or every launch is useful
  matrix work.
- Other model kernels, graph nonmatrix work and unowned copies. No activity may
  disappear because it lacks a matrix owner.

Compute overlap from actual GPU intervals, never from CPU NVTX duration or a
whole-kernel envelope used for both sides. Let G be the union of host-gather
kernel intervals, C the union of primary matrix kernels, and A the union of
all matrix-owned activities. Report `|G intersect C| / |G|` and
`|G intersect A| / |G|`, plus the corresponding overlap time and uncovered
gather time. If G is empty, the ratio is undefined, not zero or 100%.

For each gather targeting layer j, also intersect it with the actual compute
of layer j−1 in the same segment/chunk. Layer-0 gather is startup work and must
be reported separately. Report any overlap with other layers separately; a
request-wide overlap union cannot establish the intended one-layer lookahead.
Preserve each physical kernel's start/end and stream and check the target
layer's first consumer follows its gather completion. This establishes observed
scheduling, not proof of internal copy/compute overlap inside a fused kernel.

For CPU metadata/control, report per-thread exclusive NVTX time, launch/API
counts, and runtime/driver interval unions. Intersect those API intervals with
the complete GPU busy union before interpreting exposure in GPU gaps. Long
blocking API calls overlapping device execution are not wholly removable CPU
overhead. Keep metadata planning before copy launch distinct from the gather
kernel itself; both can constrain lookahead.

## MFU and delivery

Candidate API MFU is `100 * sum(ideal_ms) / sum(per-call GPU activity union ms)`,
including attributed matrix helpers. Do not average operator percentages.
Report core-kernel time separately and preserve exact useful FLOPs by chunk,
layer and precision. Report formal C6 extend/request MFU from the accepted
formal stage mean as a separate quantity. The operator diagnostic contains
one cold and one revisit sample; it is not a new latency distribution.

Deliver the independent audit, per-layer gather/compute overlap table, metadata
kernel inventory, launch-exposure analysis, candidate operator/aggregate MFU,
and a concise checkpoint interpreting the remaining work. Keep them in
`output/data/<profile_run>/analysis/` and link only selected tracked report
materials if a later publication step is explicitly performed. Relocation to
the shared repository must preserve source snapshots, raw profiles and all
file hashes. Do not replace or delete valid old reports during this diagnostic.

## Formal acceptance available before profiling

The standalone audit in the formal run's
`analysis/independent_formal_audit/` imports no production check/report code.
It independently verified all 128 measured payloads, including 96 offload/HBM
comparisons, and all 32 C4/C6 complete request records and HBM payload fields.
All comparisons are byte-exact. It also verified source snapshots, LRU and
page/token admission, transient candidate state, twelve recorded warmup
trajectory rows, timing-stage sums and 104,960 graph replay deltas. Warmup
output tensors are not saved in the formal run; the independent warmup check
therefore covers recorded trajectory/counters only.

`report_arithmetic_crosscheck.json` independently agrees with all eight report
groups and 24 formal MFU stages. C6 dense revisit is 40.262 ms E2E and
34.335 ms candidate, with E2E MFU 5.579%. C4's corresponding E2E mean is
40.330 ms; medians are 40.091 ms in C6 and 39.893 ms in C4. These observations
show no meaningful dense revisit improvement and motivate the planned
remaining-metadata/overlap diagnosis. All samples remain in the aggregates.

## Completed diagnostic, 2026-10-04

The wrapper session 68476 exited 0. Profile source SHA-256 is
`a21b2b423605d0c807c7e7ed1409adbae3d0da0566bbd0d7b3fb361edf62942c`;
its production dependencies match the accepted C6 formal source above.
All 1,480 original data/log/profile files (159,991,484 bytes) were hashed before
and after relocation to the shared repository. The accepted run's
`relocation_manifest.json` preserves their identities. The resulting data is
`experiments/deepseek_v32_motivation/output/data/motivation_c6_dense_profile_20261004_01`.

The independent CPU numerical audit checked all twenty saved hidden/logit
payloads against the original C6 formal HBM files: three warmup, one cold,
fifteen preparation and one revisit. Every tensor is finite and byte-exact;
CUDA remained uninitialized. This saved-profile check is distinct from the
formal run's warmup trajectory-only audit.

All four planned production analysis entrypoints completed from frozen C6
source without a visible GPU. The independent raw-SQL audit agrees on all
9,243 matrix invocations/primary kernels, 26,366 matrix-owned activities,
63,688 total GPU activities, 1,320 graph replays, 45,652 graph node activities
and all 45 operator groups. There are 7,260 graph matrix APIs and 660 eager
value expansions; finish templates contain no value expansion.

Revisit candidate has ten mapped-host gather kernels, transferring 655,360
records / 754,974,720 bytes in 14.708310 ms of gather GPU activity. Its actual
overlap with primary matrix kernels and with all matrix-owned activities is
zero, both request-wide and for each target layer j paired with actual layer
j-1. Layer 0 is startup. Cold history/candidate have no host misses and no
gathers; their overlap ratios are undefined. Actual main-stream metadata/copy
work overlaps gather for 0.457412 ms.

Raw CUDA event analysis resolves all twenty dependency edges: metadata-ready
before each gather and gather-ready before target consumption. For every
target 1-9, the previous projection graph is submitted before gather completion
but starts matrix work afterward. No captured main/context synchronization
API intervenes between those gather and graph submissions. The explicit wait
for that gather comes later. This establishes observed queueing, not its
cause; event GPU timestamps are unavailable, no driver table is present, and
one unmatched context-sync row lies after these launch windows.

Candidate API MFU is 28.846001% cold and 30.743763% revisit. Matching formal
candidate MFU is 18.471866% and 6.542486%, with mean candidate latency
12.161162 and 34.335472 ms. Formal request MFU is 40.699495% cold and
5.579492% revisit. These use distinct formal/profile observations; their
difference is not isolated CPU time or a promised optimization benefit.

Run-local CPU sources and evidence are in `analysis/`:
`independent_attribution_audit.py`, `operator_mfu/crosscheck.json`,
`independent_numerical_audit.py` / `.json`, `dense_overlap.py`,
`dense_overlap_audit.json`, `dense_overlap_by_layer.csv`,
`dense_segment_overlap.csv`, `dense_kernel_inventory.csv`,
`independent_launch_overlap.py`, `launch_overlap.csv`,
`stream_dependency_audit.py` / `.json`, `stream_dependency_by_layer.csv`,
`stream_event_edges.csv`, `cuda_events_with_api.csv`, `cuda_sync_with_api.csv`
and `queued_main_copies.csv`. A second audit,
`independent_overlap_review.py` / `.json`, verifies all 63,688 physical
activities and twenty event edges directly against SQLite. Empty matrix intersections produce no CSV;
zero overlap is recorded explicitly in the audit and summary tables.

The run includes 83 process-list observations, beginning 6.589 s before its
metadata start and ending 61.696 s after completion. No unexpected GPU process
was observed. Sample intervals range from 5.097 to 29.611 s, so they do not
establish continuous isolation. Monitoring source, observations and exact
boundaries are preserved with hashes in the accepted run. No later GPU work
was performed under this grant.
