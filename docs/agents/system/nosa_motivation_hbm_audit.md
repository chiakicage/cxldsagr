# NOSA HBM-only performance audit

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

Date: 2026-10-04. Diagnostic only; no formal four-method result is published here.

The complete NOSA-8B checkpoint at `/mnt/ssd-wlcb/chenkaiqi/NOSA-8B` was loaded
through `NosaServingBackend.from_pretrained`, with all 32 layers. The workload
uses H=65,536, A=128, prefix chunk=1,024, BF16 and full NOSA selection. It returns
all normalized candidate hidden states, without the LM head. Tokens are fixed
random vocabulary IDs (seed 42), not the formal saved GR workload. The diagnostic
uses the existing generic shared HBM backend, before the new fixed-pool backend.

GPU 1 is identified by PyTorch as NVIDIA H200, SM90, 132 SMs, 143,167 MiB.
`nvidia-smi` exposes the platform alias NVIDIA M403. The Python process uses
`CUDA_VISIBLE_DEVICES=1`, one CPU intra-op thread and the repository `.venv`.

## Executed evidence

The temporary harness is `/tmp/nosa_hbm_audit.py`. A separate CPU profile is
`/tmp/nosa_hbm_cpuaudit.py`. Results are under
`/tmp/nosa_hbm_audit_20261004/`; stdout/stderr are separate files under `/tmp`.
These are engineering diagnostics, not experimental deliverables. The source
was not snapshotted before execution; these numbers must not identify a later
optimized source revision or be used as formal pre/post publication evidence.

After one full empty-cache prefix and one candidate warmup, five unprofiled
candidate wall times were 99.962, 98.773, 98.573, 98.288 and 99.292 ms. Median
host submit time was 98.759 ms and the median CUDA event span was 98.744 ms.
CUDA events include launch gaps and are not active GPU time. A subsequent
independent empty-cache H=65,536 prefix took 2,562.603 ms including completion.

The separate torch-profiler candidate capture contains 802 kernels and 64
copies. The union of device activity is 12.012 ms inside a 41.382 ms device
hull, leaving 29.370 ms without recorded device work. At least 21.706 ms occurs
before the next correlated CUDA runtime submission begins. Driver submissions
were not included in that lower-bound count. This is an instrumented timeline;
its timing must not be subtracted from the independent 98.773 ms wall result.
The profiler's `gpu_user_annotation` is an overlapping parent and is excluded
from the active union. Summing profiler table parent rows would double count.

Main kernel aggregates from that capture:

| Work | GPU duration sum | Count |
| --- | ---: | ---: |
| All `aten::mm` children, including CIS delta | 4.710 ms | 160 API calls |
| FA3 main attention | 2.662 ms | 32 kernels |
| Indexer normalizer | 1.620 ms | 32 kernels |
| Indexer scores | 0.670 ms | 32 kernels |
| Selection prefix | 0.491 ms | 32 kernels |
| Finite check partials | 0.359 ms | 32 kernels |
| Indexer guarded compression/ranking | 0.270 ms | 32 kernels |
| Attention numerical repair | 0.266 ms | 32 kernels |

## Priority diagnosis

The CPU profile identifies allocator validation as the largest avoidable cost.
An instrumented candidate took 333 ms; `validate_allocator` occupied 293 ms.
Each execution lease calls `_reject_python_pools` twice, scanning all objects
returned by `gc.get_objects`. This produced 907,733 generator iterations and
907,731 `issubclass` checks in one request. The underlying model forward used
40 ms in the same CPU profile. This explains why the device hull is much shorter
than the whole request wall time. CPU profiling magnifies absolute timings, so
the 293 ms is attribution evidence, not an unprofiled overhead estimate.

1. Replace the repeated global object scan with allocator checks that preserve
   the cache admission contract without traversing the entire object graph on
   every lease. Re-run allocator mutation and custom-pool rejection tests.
2. Reduce model launch overhead around projection, normalization, RoPE, CIS and
   MLP. The profiled model itself leaves 29.370 ms of device gaps, so fixing the
   allocator alone will not meet the matrix API MFU objective.
3. Evaluate the 32 checked-indexer D2H flags and associated synchronization.
   Preserve nonfinite rejection and transactional rollback; silently deleting
   validation is not an acceptable optimization.

Full useful matrix work includes QKV/output/MLP projections, CIS delta,
compressed-key indexer QK and actual selected causal attention QK/PV. The
projection-only GPU sum above is not a complete matrix API denominator.
`experiments/nosa_motivation/src/flops.py` provides the full analytical count
and an intrusive selection validator for a separate untimed replay.

Reproduction environment for the temporary diagnostic:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  .venv/bin/python /tmp/nosa_hbm_audit.py
```

The `.venv/bin` PATH entry is needed because FlashInfer JIT invokes `ninja` by
name. A first attempt without it failed before any benchmark sample; it is not
included in the timings above.

## Fixed-pool and compute-graph diagnostic follow-up

The subsequent fixed HBM backend uses an incremental allocator guard. On the
same H200 and fixed random-token workload, five candidate times were 27.490,
27.066, 26.885, 26.697 and 26.555 ms. The allocator no longer dominates:
the CPU profile contains about 1 ms of lease/allocator overhead within a
42 ms instrumented candidate. This compares different cache implementations
and evolving sources; it is an optimization diagnostic, not a controlled
publication speedup. This diagnostic's source-stability gate detected changes
to `models/nosa/compute_graphs.py` during the run.

With 128 pure compute graphs (project and finish, two query shapes, 32 layers),
five candidate times were 20.632, 20.373, 20.400, 20.496 and 41.907 ms. Median
is 20.496 ms; the high sample is retained. An independent empty-cache prefix
took 2,464.035 ms. Source snapshots and complete results are in
`/tmp/nosa_hbm_fixed_graph_diag_20261004/`; its source-stability check detects
an experiment-config edit, so these remain diagnostic results.

The graph profile contains 802 kernels and 192 copies. Device active union is
12.244 ms within a 28.487 ms device hull; gap is 16.243 ms. At least 14.274 ms
precedes the next correlated runtime or driver submission. Complete compute API
work is 11.902 ms: project 1.002 ms, finish 4.366 ms, indexer 3.496 ms and
attention 3.037 ms. Project/finish include normalization, RoPE, CIS and activation
helpers. Graph input staging copies remain outside these API intervals. Runtime
correlation attributes every device activity; graph scopes and indexer/attention
scopes each cover all 32 layers. Graph-node kernel times are not assigned to a
nonexistent eager `aten::mm` call.

Candidate useful matrix work is 2,351,618,326,528 FLOPs, including all projection,
CIS, compressed-score and causal attention work. Using the 989.5 TFLOP/s dense
BF16 H200 SXM reference, independent candidate wall MFU is 11.595%; the separate
complete API profile MFU is 19.968%. Their ratio is about 58%, so the user's MFU
proximity objective is still unmet. This is a reference-normalized utilization
metric, not a tensor-core counter or an achieved-clock normalization.

The separate intrusive replay validates the actual full 64-block selection of
all 32 candidate layers: unique causal IDs, required current block, validity
counts and exactly 16,648,192 causal token pairs across 32 query heads per layer.
Both instrumented replays reproduce every candidate hidden value exactly.
This checks same-backend replay consistency; it does not replace independent
cache construction for resident/offload correctness acceptance.

Remaining measured overhead includes 64 calls to attention workspace validation
(about 5 ms under CPU profiling), per-layer checked-indexer synchronization,
allocator snapshots over 128 graph pools (about 5 ms in that CPU profile), and
graph weight identity checks (about 2 ms). The absolute CPU-profile times are
instrumented attribution figures, not isolated costs to subtract from wall time.

The committed diagnostic CLI is:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python -m experiments.nosa_motivation.src.profile_hbm \
  --run-id hbm_graph_diagnostic --compute-graphs \
  --output-dir /tmp/hbm_graph_diagnostic/data \
  --profile-dir /tmp/hbm_graph_diagnostic/profile
```

`--allow-source-drift` may be used during concurrent implementation work; it
records the mismatch and keeps the result diagnostic. It does not approve a
formal performance comparison. Eight CPU tests cover causal work accounting,
actual-selection invariants, graph-node attribution, missing correlations,
failure staging and preservation of whole-candidate invocation geometry.

## Attention workspace dispatch improvement

The public reserved-workspace path now performs common tensor validation and
one complete workspace validation, then passes the returned actual-shape views
to a private FA3 launch helper. Direct low-level FA3 workspace calls still run
the same guard. Previously the public path validated the workspace twice and
created its views three times. The native and Triton paths without an explicit
workspace retain their previous dispatch and numerical behavior.

Changed files are `operators/nosa/attention/device_only/api.py`,
`operators/nosa/attention/device_only/_fa3.py` and
`operators/nosa/attention/workspace.py`. Validation still covers native/Hopper
selection, BF16/D128/GQA16 geometry, matching K/V strides, device/dtype,
query/head capacity, scratch aliasing and unsupported CUDA Graph capture.

Executed same-process A/B: `/tmp/nosa_wrapper_ab.py`, results in
`/tmp/nosa_wrapper_ab_20261004/summary.json`. It loads one complete checkpoint,
constructs the same 64K history and uses pure compute graphs for both arms.
Only the previous API and reserved-launch Python callables are loaded from the
saved diagnostic source snapshot and temporarily patched in memory. No source
file is swapped or restored. Ten samples per arm alternate order. Median
candidate wall time is 20.426879 ms before and 18.740029 ms after (8.258% lower).
All 20 complete candidate outputs equal the common reference exactly.
CPU profiles confirm 64 to 32 workspace validations and 96 to 32 view-building
calls per candidate. The trace contains the full sample distribution; 59.8–73.9
ms outliers remain and are not discarded.

The remaining long outliers correspond to the allocator guard's occasional
whole-object scan after GC generation changes. One instrumented after-case
spent 42 ms in `_reject_python_pools` within a 70 ms request. Incremental checks
removed the regular cost but have not removed this tail. This observation was
returned to the cache implementation agent.

Validation after the dispatch change: 109 operator/workspace tests passed,
including native/FA3 numerical edge and PV-range cases. A fresh process passed
all 27 shared-serving and workspace tests, covering all four serving methods,
failure rollback, exact owned/shared outputs, and early rejection of capacity,
unsupported dispatch, scratch aliasing and capture. A combined earlier process
had 22 serving failures because earlier operator graph-capture tests retained
private CUDA segments; the allocator correctly rejected them before serving.
Those failures are not reported as passes. The fresh-process run establishes
the serving result without that test-order state.

Affected performance results remain pending: the new fixed NOSA motivation
trace has not yet been published; generic `gr_serving` HBM/dense-prefetch results
remain tied to their original run IDs. Existing results are retained until new
complete measurements are accepted. The full-checkpoint A/B above is an
engineering diagnostic, not a replacement for either experiment's request trace.

A separate post-change complete profile at
`/tmp/nosa_hbm_fixed_wrapper_diag_20261004/` records candidate median 18.897150 ms,
one empty-cache prefix 2,379.895 ms, wall MFU 12.576% and complete API MFU 19.916%.
The device active union is 12.274 ms; device hull is 24.916 ms, with 12.642 ms of
gaps and 10.779 ms before the next correlated submission. Complete API union is
11.933 ms. All 32 layers retain complete API attribution and exact candidate
replay/selection checks. The wall/API MFU ratio is now about 63%, still below a
reasonable interpretation of close. Formal multi-user measurements and the
remaining CPU overhead work remain open.

## Complete history-plus-candidate workload and independent matrix APIs

The first request generated with the actual formal GR configuration was replayed
through the complete backend: 16 users, two rounds, seed 42, H=65,536,
A=128, history chunk=1,024. The three empty-history backend wall measurements
were 2,380.238, 2,387.084 and 2,387.682 ms. This timer includes prefill, candidate
execution and final synchronization. Session creation and token upload precede
the timer; this is not the full `PersistentGRRunner.execute` admission/eviction
timer. The complete candidate hidden from the profiled replay is exactly equal.
Artifacts are `/tmp/nosa_full_request_profile_20261004/`, and the executed
temporary harness is `/tmp/nosa_full_request_profile.py`.

The full workload contains 1,166,661,887,459,328 useful matrix FLOPs. Its median
wall MFU is 49.393%. The separate profile attributes 2,059.315 ms to complete
compute APIs, corresponding to 57.254% MFU; their ratio is 86.27%. Device active
union is 2,102.255 ms inside a 2,568.443 ms device hull. At least 357.077 ms of
the 466.187 ms gap occurs before the next correlated submission. Every API
scope covers all 2,080 layer invocations (32 layers times 65 model calls), and
no device activities lack launch correlation. Candidate-only efficiency is
therefore not representative of the complete cold-history request.

The 2.721-second CPU profile identifies the remaining work in descending order:

| Scope | Instrumented cumulative CPU time | Calls |
| --- | ---: | ---: |
| Indexer | 1.787 s | 2,080 |
| Checked native indexer API | 0.632 s | 1,088 |
| Workspace validation | 0.165 s | 2,080 |
| Graph weight identity | 0.106 s | 65 |
| Allocator validation | 0.066 s | 4 |

These are nested CPU durations, not additive host overhead. Checked native
indexer calls wait for GPU completion, so their entire duration must not be
labelled Python work. The evidence supports deferring finite-input host
decisions to a transaction boundary with device guards and rollback preserved.
It also supports checking graph weight identity once within an exclusive
prefill execution lease, instead of walking parameters at every chunk, provided
the immutable-model contract remains enforced. Runtime implementation of these
changes is owned by the parent agent; the values above predate them.

The independent reference executes every checkpoint layer's actual weight
matrices for each required query size, and BF16 `torch.bmm` at every compressed
key count. Its CUDA Graph submission removes Python gaps while retaining actual
library APIs. BMM physically expands each KV head to its GQA query-head group
and materializes rectangular score matrices. It is not the fused native
indexer, which avoids those materialized scores. Causal useful FLOPs and dense
executed FLOPs are recorded separately. No attention QK/PV, normalization,
softmax, selection, cache, RoPE or activation is silently included.

The first independent reference, `/tmp/nosa_raw_matrix_api_20261004/summary.json`,
used seven samples per linear shape and five per BMM shape, following three
warmups. Shape-weighted linears took 1,325.715 ms at 75.919% MFU; indexer BMM
took 439.464 ms at 8.086% useful MFU. Together they took 1,765.180 ms at 59.031%.
Adding the separately profiled complete FA3 attention API (368.790 ms) yields a
composition estimate of 2,133.969 ms and 55.251% MFU. That sum is not a measured
request; its wall/API ratio is 89.40% for this particular pair of diagnostics.

The versioned helper `experiments/nosa_motivation/src/matrix_baseline.py` was
subsequently executed and checked exact matrix-work conservation:
1,031,063,293,919,232 GEMM/BMM FLOPs. It produced 1,872.070 ms and 55.661% combined
MFU. BMM remained close (439.886 ms), while large MLP GEMMs were slower, giving
70.275% linear MFU. Both measurements are retained in `/tmp`; the slower
reference must not be selected merely to make the proximity condition easier.
The final comparison needs a matched, explicitly identified reference run.

The maintained diagnostic now supports the full workload, native build/header
identity, loaded-library hashes and independent matrix APIs:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python -m experiments.nosa_motivation.src.profile_hbm \
  --run-id hbm_full_reference --compute-graphs --workload gr \
  --full-request --raw-matrix-reference \
  --output-dir /tmp/hbm_full_reference/data \
  --profile-dir /tmp/hbm_full_reference/profile
```

## Maintained full CLI after deferred validation

**Invalid for performance conclusions pending a numerical fix.** The subsequent
full-checkpoint multi-user graph-HBM correctness test passed its first user but
failed the second user's hidden-output comparison. That failure uses the same
frozen runtime as this diagnostic. Repeating the same history in this harness
established determinism, not equivalence to an independently computed resident
reference or correct behavior when moving between users. The source and native
identity checks still establish which implementation ran; they do not establish
its numerical correctness. No formal experiment was published from this run.
The observations below are retained only in this internal engineering audit and
`/tmp` for diagnosis, and must be rerun after the fix and multi-user acceptance.

Independent static inspection found that the expanded joint async path changes
short-prefill dispatch. `api._launch_scores` retains Triton when rows are at
least 1,024 and compressed-key count is below 2,047. Allowing the joint path
whenever a deferred flag exists bypasses that guard and directly submits the
native fused selector for those same shapes. The short-shape async tests use
`separate_reference`, which calls native `indexer_ranked_out`, so both arms
exercise native scoring; they do not verify equivalence to the old Triton
dispatch. Restoring the original count guard is therefore a numerical-path
correction as well as a dispatch change. Confirmation that it resolves the
multi-user failure was initially pending; after restoring that guard, the
parent agent reported a passing full-checkpoint multi-user graph/eager
comparison across all four methods. The measurements below still belong to
the rejected expanded-dispatch implementation and cannot be reused. Finite flags occupy distinct graph-owned
per-layer bytes, native writable-buffer alias checks include the finite flag,
and producer/consumer kernels share one stream; static inspection found no
clear finite-flag scratch reuse defect.

Run `hbm_full_reference_deferred_20261004` executed the complete maintained CLI
on physical GPU 1 after the parent agent froze the deferred-validation runtime.
The complete artifact set is under
`/tmp/nosa_hbm_full_reference_deferred_20261004/`; this remains an engineering
diagnostic rather than a formal four-method request trace. Source manifest
SHA-256 is `d3f4e98b00570fdbf99740c909c23f82c87180001ea803653a0d3295ff951140`.
All source files and native build dependencies remained unchanged during the
run. The 21 mapped project, FlashInfer, PyTorch and CUDA library identities also
matched after warmup and at completion, including after the independent matrix
reference. Exact hidden replays passed, and the separate selection replay
verified every one of the 32 layers.

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python -m experiments.nosa_motivation.src.profile_hbm \
  --run-id hbm_full_reference_deferred_20261004 \
  --compute-graphs --workload gr --full-request --raw-matrix-reference \
  --output-dir /tmp/nosa_hbm_full_reference_deferred_20261004/data \
  --profile-dir /tmp/nosa_hbm_full_reference_deferred_20261004/profile
```

The full backend request wall samples were 2,299.787, 2,306.815 and 2,343.087 ms.
The timer includes fresh-history prefill, the candidate and synchronization;
session allocation and input upload remain outside it. All 2,080 invocations of
each API scope were present, with no uncorrelated device activities. The useful
work numerator remains 1,166,661,887,459,328 FLOPs.

| Workload | Wall median | Useful wall MFU | Complete API union | API MFU | Wall/API MFU ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full H+A backend request | 2,306.815 ms | 51.111% | 2,105.046 ms | 56.010% | 91.253% |
| Candidate only | 17.884 ms | 13.289% | 11.887 ms | 19.994% | 66.467% |

API durations come from a separate instrumented replay, so their ratio to
uninstrumented wall time is a diagnostic comparison, not a decomposition of the
same sample. The full trace has 2,148.833 ms of device activity inside a
2,343.307 ms hull. Gaps total 194.474 ms, including 124.993 ms before the next
correlated host submission. The earlier pre-deferred trace had 466.187 ms of
gaps and 357.077 ms before submission; these are separate runs, not an
alternating A/B experiment. The candidate samples retain their first outlier:
55.255, 18.075, 17.884, 17.821 and 17.841 ms. Its cause was not isolated, so the
outlier is not attributed to GC as an established result.

The matched independent matrix reference conserved all
1,031,063,293,919,232 GEMM/BMM useful FLOPs. Shape-weighted checkpoint linears
took 1,427.895 ms at 70.486% MFU; the rectangular GQA-expanded indexer BMMs took
438.993 ms at 8.095% useful MFU. The combined reference is 1,866.888 ms at
55.815% useful MFU. Adding this run's separately profiled full attention API
(387.766 ms) gives a composition estimate of 2,254.655 ms at 52.294% useful MFU;
the full-request wall MFU is 97.739% of that estimate. This sum is not an
observed request and does not make the materialized BMM equivalent to the
native fused indexer. Earlier raw-reference observations remain visible above;
the matched run was selected by execution identity, not by a favorable ratio.

The separate CPU profile now records 0.647 s inside the indexer, compared with
1.787 s in the earlier diagnostic. There are 65 deferred finite decisions with
0.733 s inside `Tensor.item`; this includes waiting for queued GPU work and
cannot be labelled pure CPU overhead. Weight-identity checks still take
0.105 s across 65 invocations, and attention workspace validation takes
0.170 s across 2,080 invocations. These nested instrumented durations are not
additive wall components. Remaining candidate launch overhead is visible in
its 66.467% wall/API ratio. This diagnostic does not establish steady multi-user
serving performance or replace the pending formal experiment.

The maintained trace/FLOP tests passed in a fresh CPU test run: 8 passed in
1.26 s. No runtime source changes were made as part of this diagnostic.

## Accepted-request matrix-reference integration

The formal profile path replays accepted `PersistentGRRunner.execute` requests
under the same runtime, native build, checkpoint, precision and GPU UUID. It
should supply `matrix_comparison.compare_request_apis` with the accepted
measurement row, the complete API timeline from that replay, and the independent
raw matrix artifact. Wall MFU must use that row's `latency_ms`, which includes
token validation/upload, session admission and eviction, required history
construction, candidate execution and cleanup. Neither the backend-only timer
nor the profiler's outer root is a substitute: the outer root additionally
contains post-latency diagnostics.

`matrix_comparison.audit_matrix_reference` reopens every matrix sample and
checks query geometry, checkpoint layer count and weight shapes, compressed-key
counts, causal-useful and dense-executed FLOPs, median arithmetic, and request
multiplicity. It reconstructs both full-history request and candidate-only
references from one raw artifact. A history tail with the same query size as
the candidate still contributes only one linear invocation to candidate-only
work. Missing/duplicate shapes and incomplete compute API attribution fail.

The resulting comparison keeps three different quantities visible:

- Full-model useful FLOPs over accepted runner wall time and over the separately
  profiled complete-compute-API GPU execution-span union. Each invocation spans
  its first correlated device activity through its final completion, preserving
  gaps inside that API call; nested spans are unioned. This excludes host work
  before the first device activity and is not API CPU/wall latency. The activity
  union and its MFU remain separately named because they omit internal gaps.
- Matching GEMM/BMM useful FLOPs over independent raw API medians and over the
  complete runner wall time. This shared-numerator ratio retains admission,
  attention and helper costs in the wall denominator.
- Raw matrix medians plus separately profiled attention API time, explicitly a
  composition estimate. It is not an observed request or an independent
  full-model matrix reference and cannot establish a passed proximity gate.

The new comparison tests plus existing FLOP/trace tests passed: 18 tests in
1.33 s. The helper also recomputed the saved raw artifact's geometry and totals
on CPU; that integrity check does not make its rejected runtime or performance
conclusions valid. No new GPU profiling is scheduled before the complete formal
measurement is accepted.

Before the final source freeze, the formal timeline analyzer replaced its
per-device-event scan over every scope with a per-thread interval index and
cached launch ownership. Inclusive containment and original stable ordering
for equal-duration scopes are preserved. The real trace also exposed a
duplicate-root issue: PyTorch emits CPU and GPU annotations with the same
request name. Root and scope lookup now use CPU `user_annotation` events;
GPU annotation envelopes remain excluded from device work.

CPU reanalysis of the saved full trace took 0.599 s with indexing versus
104.301 s with the prior implementation. Every returned metric was identical.
For this comparison, both inputs adapted the HBM diagnostic's `matrix_api/`
scope prefix to the formal analyzer's `nosa::` prefix. Only the old analyzer's
input omitted redundant GPU annotation events to bypass its duplicate-root
failure; actual GPU activities were identical. Validation details are in
`/tmp/nosa_timeline_index_validation_20261004.json`. The complete motivation
CPU suite passed 84 tests in 2.48 s, including native-library provenance,
scope-index containment and duplicate-root regressions. The CPU comparison job
finished before formal GPU measurement was allowed to start.

## Independent attention references integrated on 2026-10-04

The accepted measurement is `nosa_motivation_sm90_20261004_02`. Its matching
one-sample timeline/internal-work profile is
`nosa_motivation_current_diagnostic_20261004_02`, currently held under
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_motivation_current_profile/`.
Those identities supersede the rejected diagnostic above for current work;
no old diagnostic performance value is reused as an attention reference.

Three new experiment modules implement the missing independent attention
coverage: `attention_reference.py`, `attention_reference_audit.py`, and
`profile_attention_reference.py`. The intrusive HBM replay records actual
Q, selections and validity per layer/invocation, plus one final K/V/CIS snapshot
per layer. Every visible prefix is checked against that final snapshot.
The independently submitted public resident FA3 API restores original operand
strides and must reproduce every captured attention output hash. Separate
materialized selected-QK/PV BMMs cover the same causal useful work while reporting
padded execution and duplicated KV bytes. QK uses BF16 inputs/FP32 output; PV
uses BF16 inputs/output. Deterministic first/middle/last product rows are checked
against IEEE FP32 products. The sampled-product check is not full raw-attention
or model equivalence.

Capture begins only after HBM warmup. Mapped native identities must be an
unchanged subset of the accepted measurement's final inventory; independent
benchmark additions are recorded separately. `--audit-data` reopens source,
checkpoint/precision/GPU identities, accepted request output hashes, native
observations, copied matrix evidence, operand files, repeated-history proofs,
all fixed 3-warmup/7-repeat statistics, useful-work coverage and the combined
arithmetic without executing GPU work. HBM hits use candidate-only work.
HBM misses may reuse first-request prefix timings only after exact repeated
Q/selection/output and final prefix K/V/CIS verification.

Both the materialized-BMM sum and independent-FA3 composition remain visible.
They are sums of independently measured medians, not observed requests.
Beating the slower materialized reference alone cannot pass the efficiency
goal. No new numeric closeness threshold was introduced.

Independent code review passed before integration. Eleven CPU tests passed
on the integrated files; Ruff passed. These checks do not establish Hopper
execution. The first engineering smoke configuration was rejected by workload
generation because A=8 could not hold its mandatory 39-token candidate text.
The corrected smoke uses all 32 actual checkpoint layers at H=1024, A=128,
chunk=1024; CPU generation of its four requests passed. Its GPU result and the
full accepted-workload reference are pending as of this entry. No runtime or
existing profile module was changed by this integration.

## Validated independent attention result

The corrected engineering smoke passed all 64 attention calls from all 32
checkpoint layers at H=1024/A=128/chunk=1024: independent empty-cache request
hidden states matched, public resident FA3 outputs matched every captured hash,
and the sampled raw products met their declared tolerances. This is an API and
numerical smoke, not a formal serving measurement.

The full reference `nosa_attention_reference_sm90_20261004_01` then completed
against accepted measurement `nosa_motivation_sm90_20261004_02` and profile
`nosa_motivation_current_diagnostic_20261004_02`. Its published data is under
`experiments/nosa_motivation/output/data/nosa_attention_reference_sm90_20261004_01/`.
An independent CPU-only reopen audit, bound to NUMA1 with CUDA hidden, passed
with no stderr. It checked all 2,112 attention calls, all operand files and
native/source identities, the fixed sampling statistics, complete useful-matrix
FLOP coverage, and request 16's reuse proof for 2,048 history calls and 32 final
prefix layers. The reopen result is
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_attention_reference_sm90_20261004_01.reopen_audit.json`.

Both sampled HBM requests rebuild history. Their numerator is the same
1,166,661,887,459,328 useful matrix FLOPs, and the full-request denominator is
the accepted runner `latency_ms`, including admission and cleanup.

| HBM request | Accepted runner ms | Runner MFU | Independent GEMM/BMM + FA3 ms | Composition MFU | Runner/composition MFU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0, first visit | 2,430.182 | 48.5166% | 2,386.912 | 49.3961% | 98.2195% |
| 16, revisit with HBM miss | 2,495.851 | 47.2401% | 2,387.013 | 49.3940% | 95.6393% |

The shared full-request GEMM/BMM reference contributes 1,882.030 ms. Independent
FA3 attention contributes 504.882 ms for request 0; request 16 combines the
exactly verified repeated history with its own candidate timings. These full
samples are close to this measured API composition. The reference is a sum of
independent API medians, not an observed standalone request or a theoretical
performance ceiling. It still uses the independently materialized indexer BMM
reference and excludes non-matrix operations from that component. Two sampled
requests do not establish proximity for every user or request.

The separate materialized-attention reference is much slower: its QK/PV sum is
2,368.032 ms for request 0, giving a complete raw-matrix sum of 4,250.062 ms.
The raw sums for requests 0 and 16 are 1.7489 and 1.7027 times their accepted
runner latencies. Across request 0's invocations, materialization expands
8,607,114,461,184 BF16 K/V bytes, versus 26,572,947,456 unique selected BF16 K/V
bytes summed over the same invocations. These are traffic/operand-volume sums,
not simultaneous allocation sizes or a unique-record count for the whole
request. This duplication explains why beating the materialized reference
alone cannot establish an efficient model implementation.

Candidate-only work needs a separate reading. Its numerator is
2,351,618,326,528 useful matrix FLOPs and its measured denominator below is
accepted `extend_ms`, which excludes history construction and admission. This
phase diagnostic must not replace the full runner denominator above.

| HBM request | Accepted extend ms | Extend MFU | Independent GEMM/BMM + FA3 ms | Composition MFU | Extend/composition MFU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 19.6079 | 12.1205% | 12.4281 | 19.1226% | 63.3831% |
| 16 | 70.3229 | 3.3795% | 12.5291 | 18.9684% | 17.8166% |

The candidate GEMM/BMM sum is 6.9864 ms. Candidate FA3 sums are 5.4416 and
5.5427 ms; raw materialized QK/PV sums are 5.6592 and 5.1921 ms. The candidate
materialized path is therefore not uniformly slower than FA3, and both remain
reported. The 70.3229 ms accepted sample is retained without attributing it to
GC, allocation or another cause that has not been isolated. Long-history work
dominates the full-request numerator and time, so the close full-request
comparison conceals unresolved candidate overhead. No blanket efficiency pass
or new numeric proximity threshold follows from this reference.

## Current-source candidate overhead diagnostic, 2026-10-04

A bounded diagnostic is selected at integrated source `02b5d0…` while the
independent Q128 host-alignment helper is prepared. It uses the existing
`profile_hbm.py` in an external copy with one explicit change: intra-op threads
1 → 8 to match the intended environment. Production and experiment source files
remain unchanged. Helper identity and outputs are under
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_hbm_current_candidate_20261004_01/`.

Use GPU5/NUMA1, CPUs48–55, canonical register8 allocator and eight CPU threads;
preserve GC and the default inter-op count. The full checkpoint builds H65536
once and repeatedly extends the first generated GR request's A128 candidate,
with chunk1024 and pure-compute graphs. Nine unprofiled samples follow one
warmup; separate cProfile and torch.profiler replays diagnose CPU costs and
correlated GPU gaps. Existing exact replay/selection and source/library checks
remain enabled. A second prefix timing remains the helper's separate observation.

This can identify remaining candidate overhead on current code. It is neither
a multi-user trace nor a new independent matrix/FA3 reference, and cannot replace
the formal four-method rerun. No runtime optimization is selected by this plan.

The diagnostic completed with exit0. All nine wall samples are retained:
15.085510–16.320977 ms, median15.112404 ms. Source digest remained `02b5d0…`;
the separate profiler and selection replays matched the repeated candidate
output exactly across all32 layers. This retained-history backend replay does
not reproduce the original multi-user runner's admission/rebuild conditions,
so it does not resolve the old70.3229 ms outlier or establish a code-only gain.

The separate GPU trace accounts for32 scopes of each of projection, indexer,
attention and finish, with803 kernels,161 copy/memset activities and no missing
launch correlations. API execution spans total12.252267 ms; their activity union
is11.870885 ms. The trace's full GPU hull is23.347691 ms with11.132940 ms of gaps,
including9.658511 ms before subsequent submission. These are intrusive trace
observations, not an additive breakdown of the15.112404 ms wall median and not
an independent matrix API reference.

The CPU profile points to repeated indexer preparation, attention validation,
stream wrappers and lease-level weight/allocator checks. Some cProfile call
counts are inconsistent with the32-layer execution (for example one recorded
`forward_layer` and34/5 graph replay calls). Its cumulative costs therefore only
suggest candidate locations; they cannot quantify savings. Any CPU wrapper
change needs a controlled unprofiled comparison and preserved guard semantics.
