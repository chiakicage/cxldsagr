# CUDA Graph operator attribution checkpoint

Scope: diagnostic attribution for `deepseek_v32_motivation` compute islands.
This work does not establish the full workload's performance or replace a
published experiment. Old accepted reports remain in place until the new
formal/profile pair is checked and published.

## Mechanism

1. Trace graph allocation in a separate CUDA profiler range. During the actual
   capture, query the live CUDA graph's node set around each matrix API. Record
   actual shapes, precision, useful FLOPs, GPU node types, independent weight
   pointers, residual branch, layer and query-count key.
2. Resolve graph/node IDs with the **already loaded Nsight CUPTI instance**.
   Loading the ordinary CUDA CUPTI library alongside Nsight returned status 39
   in the first exploratory probe. CUDA 13 exports the seven-argument
   `cudaStreamGetCaptureInfo`; the implemented inspector handles that ABI.
3. Instantiation clones the capture graph. The executable graph ID is queried
   with `cuptiGetGraphExecId`; replay node lineage comes from Nsight's
   `CUDA_GRAPH_NODE_EVENTS.originalGraphNodeId`. Node handles and numeric ID
   bit patterns are never treated as interchangeable IDs.
4. Runtime graph scopes record actual replay identities and instantiate the
   capture-time matrix ledger. These synthetic API rows claim zero CPU scope
   duration. Eager MLA receives Q origin metadata before the graph block runs;
   graph replay bypasses the normal projection wrapper that records it.
5. Correlate GPU activities to the actual replay launch and resolve each node
   through the captured clone lineage. Require complete node-set equality per
   replay, unique matrix ownership, matching process and executable identity,
   and matching NVTX replay scopes. Memcpy/memset tables that lack graphId use
   exact node lineage plus process, replay scope and complete graph membership.
6. Calculate matrix API GPU-active time from its real replay-node intervals,
   including helpers. Preserve nonmatrix graph activities in pipeline totals.
   Separately reconcile useful work per chunk, independent layer, operator and
   precision against the workload expansion. Sparse MLA may split without
   changing useful FLOPs.

## Evidence obtained

Hardware: GPU 3, Hopper SM90; PyTorch 2.12.1+cu130; Nsight Systems
2025.6.3.541-256337736014v0. Engineering artifacts are outside `experiments/`
at `/tmp/motivation_graph_probe/`; they are not paper experiment results.

- `inspector_probe.py` / `inspector_result.json`: a 32x32 FP32 matmul plus add,
  with two replays and separate setup/request profiler ranges. Captured graph
  ID 1 became executable ID 2. Three explicit clone edges resolved all six
  physical replay node activities. Each matmul API included its primary kernel
  and helper. `actual_attribution.json` stores actual measured intervals.
- `block_control_probe.py` / `block_control_result.json`: a small synthetic
  `CheckpointBlock` fixture, one independent layer and Q keys {2, 5}. Four
  graph keys captured. Two blocks at positions 0 and 5 replayed both islands.
  Annotated and unannotated output tuples matched bitwise.
- `block_control_attribution.json`: four observed replays, 244 graph GPU node
  activities, 24 matrix API invocations and 24 primary kernels. All 256 GPU
  activities are retained, including 12 eager activities. Query origins were
  [0, 5]. This validates instrumentation mechanics, not real-checkpoint quality
  or full serving performance.
- The new FLOP verifier also reconciled all 45,933 matrix calls from the
  existing eager profile with 45,933 independent layer/operator groups,
  without modifying old artifacts. This is a formula/shape cross-check rather
  than an independent hardware instruction count.

## Entrypoints

Capture after accepting a matching-source graph-enabled formal run:

```bash
bash experiments/deepseek_v32_motivation/scripts/profile.sh \
  --run-id NEW_PROFILE_ID --reference-run FORMAL_RUN --device cuda:0
```

The script uses `--cuda-graph-trace=node`. Four schemes produce 12 traces:
setup traces 1, 4, 7, 10; request traces 2, 3, 5, 6, 8, 9, 11, 12. Optional
`--scheme` captures one scheme, with three traces. Setup is excluded from
request timing. No `--compute-graphs` switch is needed: the accepted reference
configuration determines execution mode.

```bash
python -m experiments.deepseek_v32_motivation.src.verify_profile_flops \
  --profile-run PROFILE --flops FLOPS/flops.json
python -m experiments.deepseek_v32_motivation.src.analyze_pipeline \
  --sqlite PROFILE/capture_2.sqlite --sqlite PROFILE/capture_3.sqlite \
  --sqlite PROFILE/capture_5.sqlite --sqlite PROFILE/capture_6.sqlite \
  --sqlite PROFILE/capture_8.sqlite --sqlite PROFILE/capture_9.sqlite \
  --sqlite PROFILE/capture_11.sqlite --sqlite PROFILE/capture_12.sqlite \
  --output-dir PROFILE/analysis
python -m experiments.deepseek_v32_motivation.src.operator_mfu \
  --profile-run PROFILE --flops FLOPS/flops.json \
  --output-dir PROFILE/analysis/operator_mfu
python -m experiments.deepseek_v32_motivation.src.aggregate_mfu \
  --operator-dir PROFILE/analysis/operator_mfu --flops-dir FLOPS \
  --output-dir PROFILE/analysis/aggregate_mfu
```

The publication helper now renders current run IDs, actual counts and measured
groups, including weighted mixed resident/fused indexer calls. It still
requires an independent `crosscheck.json` for publication. Do not copy the old
crosscheck into a new analysis.

## Full C3 profile accepted and independently audited

The matching runs are formal `motivation_c3_20261004_u16_r2_01`, profile
`motivation_c3_profile_20261004_01` and FLOPs
`motivation_c3_flops_20261004_01`. The profile completed all four schemes:
eight request captures and four setup captures, with 80 exact-output checks.
The profile source SHA-256 is
`42cacd62967a282fefc8d40f0fec1e6b02832701c53bd3a9e9bf0fcfd6696caf`;
its formal reference source SHA-256 is
`d7eda579163ab9d51a211b2a5d7434f22d597d726ce2c665fa826144cec27d02`.

All three profile output categories were moved from the isolated source tree
to the original repository without overwriting a destination. The 1,572 files
were SHA-256 verified before and after the cross-device move. Source snapshots
remain in the data run. The run-local `relocation_manifest.json` records every
file identity and both paths. Source was not changed to run the analysis.

All four analysis entrypoints above ran with `CUDA_VISIBLE_DEVICES=''` from
`/tmp/deepseek-motivation-c3-profile-frozen-4d3qdyh4`, writing to
`experiments/deepseek_v32_motivation/output/data/motivation_c3_profile_20261004_01/analysis/`.
The resulting `flops_verification.json`, `pipeline.json`, `operator_mfu/` and
`aggregate_mfu/` are CPU analysis of this accepted profile. No replacement
report was published and no prior valid report or run was removed.

Independent raw-SQL audit source is `analysis/independent_attribution_audit.py`;
its result is `analysis/operator_mfu/crosscheck.json`. It imports none of the
production attribution/MFU implementation and independently joins process and
launch correlation, traverses explicit clone edges, binds eager matrix scopes,
and uses an endpoint sweep for GPU interval unions. It checks every invocation
and all 195 aggregate groups, not just selected examples:

- 45,933 matrix calls and primary kernels, including 39,360 graph matrix APIs;
  all per-call active/span/sum durations and useful-work denominators agree.
- 131,096 matrix-owned GPU activities; all 357,548 captured GPU activities are
  retained, with exact request busy unions and spans.
- 6,560 graph replays and 236,656 graph GPU activities; every expected node
  occurs exactly once and belongs to its actual executable and replay launch.
- 5,696 explicit clone edges. The 6,400 memcpy/memset activities without a
  graph ID are resolved by node lineage, process and actual launch; no IDs are
  inferred arithmetically.
- All 160 graph templates agree with capture-time shapes, precision, work and
  matrix node sets. Each scheme retains distinct pointers for all ten layers
  across the twelve recorded weight names; query keys reuse the same layer's
  weights.
- All eight traces have zero unattributed GPU activities, preserve stage
  GPU counts/time and preserve exclusive per-thread CPU NVTX unions.

## C3 MFU and cost interpretation

The first two numeric columns below use the formal mean request wall time;
the matrix API columns use a single instrumented cold/revisit request per
scheme. Matrix API time sums per-invocation unions, including each API's GPU
helpers. It excludes CPU overhead, external top-k/cache work and launch gaps.
The cross-measurement difference is diagnostic, not a directly removable cost.

| Scheme / visit | Formal E2E ms | E2E MFU % | Matrix API active ms | Matrix API MFU % |
|---|---:|---:|---:|---:|
| HBM / first | 2526.549 | 38.290 | 1902.625 | 50.846 |
| HBM / revisit | 2526.522 | 38.290 | 1921.626 | 50.343 |
| ECHO / first | 2631.164 | 36.767 | 1827.228 | 52.944 |
| ECHO / revisit | 43.412 | 5.175 | 12.827 | 17.513 |
| Serial sparse / first | 2570.870 | 37.630 | 1883.877 | 51.352 |
| Serial sparse / revisit | 37.905 | 5.926 | 7.905 | 28.416 |
| Dense prefetch / first | 2575.018 | 37.569 | 1886.578 | 51.278 |
| Dense prefetch / revisit | 84.533 | 2.657 | 7.880 | 28.507 |

HBM revisit rebuilds history under the fixed quota. Its work is not comparable
to an offload history hit's candidate-only request. C3 records TF32 disabled;
the useful-work denominator is precision-specific and is not a hardware
Tensor Core utilization counter.

C3 dense-prefetch revisit includes request 16 at 674.951 ms, versus roughly
44.6–45.7 ms for the other fifteen requests. Its cause is not established;
retain the observation and do not attribute the full mean reduction in a
later run to code changes. The [C4 integration audit](deepseek_motivation_c4_integration.md)
compares all samples and medians as well as means.

The full profiled cold request GPU busy/span/gap times (ms) are respectively
HBM 2459.687/2512.760/53.073, ECHO 2460.399/2843.674/383.275, serial sparse
2502.818/2580.449/77.630 and dense prefetch 2507.294/2593.229/85.935.
Instrumented request walls differ from formal means; use the formal run for
latency conclusions.

The leading actionable costs are:

1. **History exact top-k: 206.456–219.833 ms** outside matrix APIs. Filtered
   selection takes about 101–105 ms, stable ordering 43–44 ms and final index
   construction 34–35 ms. Preserve exact causal selection and tie behavior;
   removing sorting without proving the downstream contract is not valid.
2. **History graph auxiliary work: about 270–274 ms.** Projection contributes
   136–139 ms and finish 133–135 ms. Serial sparse has 73.900 ms of float
   cast/copy kernels, 56.587 ms of BF16 copy kernels, 29.175 ms of BF16
   conversion and 25.216 ms of concatenation within these groups. Graph replay
   removes host launches but leaves these physical operations. Direct final
   layouts and fewer temporary casts/copies are promising; preserve FP32 norm
   weights and pre-rounding residual arithmetic.
3. **Causal masking inside the resident indexer API: about 134–139 ms.** In
   serial sparse's full cold request, `CompareFunctor<long>` takes 90.398 ms
   and `masked_fill` 46.420 ms. History indexer API time is 495.022 ms versus
   357.045 ms for the core QK kernel. A specialized mask path that avoids the
   large intermediate boolean tensor is a concrete candidate. This cost is
   already included in matrix API time and must not be added to it again.
4. **History offload resident-selection metadata: 60.381–62.489 ms** for 640
   native calls. Additional `pool_operation` work brings its stage to
   64.242–67.855 ms. History D2H copies cost about 15.9–16.0 ms. These are
   measured implementation costs, not a general unavoidable offload penalty.
5. **External graph input copies: about 55 ms**, mainly 46.7–46.8 ms for
   finish inputs plus about 8 ms for projection inputs. Independent source
   replay copies add about 5.6 ms. Alias/layout changes require exact-output
   and buffer-lifetime checks.
6. **Candidate cache control remains launch-heavy.** ECHO/serial revisit
   has 1,856/1,520 `cudaLaunchKernel` calls despite only 20 graph replays.
   Exact argsort/unique GPU work costs 2.099/1.412 ms combined; their CPU
   submission and synchronization boundaries add exposure. Dense full-history
   gather itself takes 14.710 ms; ECHO's inseparable fused indexer/prefetch
   takes 8.002 ms plus 0.353 ms residual gather. No internal fusion overlap
   claim follows from the single fused interval.

`analysis/independent_nonmatrix.csv` excludes matrix-owned helpers and retains
full kernel names. `analysis/launch_overlap.csv` and its standalone source
independently intersect real CPU API intervals with the complete GPU busy
union. They prevent long synchronous API calls from being mislabeled as pure
CPU overhead. For example:

- HBM cold kernel-launch API union is 985.203 ms, of which 972.245 ms overlaps
  GPU work; only 12.958 ms lies in GPU gaps. Its graph-launch union is
  455.897 ms, with 449.633 ms overlapping GPU work.
- ECHO cold synchronization union is 1426.147 ms, including 1378.584 ms with
  active GPU work and 47.516 ms in GPU gaps. Serial and dense synchronization
  unions similarly overlap active GPU work for 1304.607 and 1120.093 ms.
- Revisit ordinary-launch/graph-launch time exposed in GPU gaps is
  ECHO 5.654/1.537 ms, serial 6.957/3.688 ms and dense 3.155/0.111 ms.

These intersections describe exposure, not causality or a guaranteed speedup.
API source/family rows overlap and must not be summed. The next implementation
must be remeasured under a new run ID; C3 results do not validate later CPU
admission, cleanup or layout changes.

Concurrent changes to profiling files from another workspace writer were
observed. Pre-review source copies are at
`/tmp/motivation_graph_review_snapshot/`; root runs formal/profile jobs from
isolated source trees and must use each run's own source manifest.

## Graph-v2 attribution smoke, 2026-10-04

The `deepseek-compute-islands-v2` policy expands values eagerly into the static
finish input and retains the projection capture's saved tensor. Attribution
review found no required production change: capture-time templates discover
their matrix APIs dynamically, and the eager BMM wrapper forwards `out=` and
identifies `wv_b` as `v_expand`. Useful-work verification remains independent
of whether a matrix API is eager or captured.

Engineering evidence is at `/tmp/motivation_graph_probe_v2/`, separately from
the earlier graph-v1 probes and accepted C3 profile. Hardware was GPU 2,
NVIDIA H200, logical `cuda:0` under `CUDA_VISIBLE_DEVICES=2`; PyTorch was
2.12.1+cu130. This was an authorized correctness/counting smoke alongside
independent numerical work on another GPU. It provides no performance result.
Nsight recorded separate setup and request profiler ranges with
`--trace=cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi
--capture-range-end=repeat:2`. Process session 73505 exited successfully and
released GPU 2 before the CPU audits.

`block_control_probe.py` constructs one small synthetic checkpoint block,
captures query keys {2, 5}, and runs two Q=5 blocks at positions 0 and 5.
The capture ledger contains four graphs: each projection has 106 GPU nodes
and seven matrix APIs; each finish has fourteen GPU nodes and four matrix
APIs. Annotated and unannotated hidden/residual tuples match bitwise, and
replay query origins remain [0, 5]. The probe records source files and their
hashes before allocation and checks that they remain unchanged at completion.

Both `production_attribution.json` and `independent_attribution.json` pass:

- Four observed request replays contain exactly 240 graph GPU activities,
  resolved through 240 explicit clone edges from the setup trace. Every
  replay has complete, unique node membership and one actual graph launch.
- All 250 request GPU activities are retained: 244 kernels and six copies.
  Each of the 24 matrix APIs owns exactly one primary kernel. Twenty-two APIs
  are instantiated from graph templates; the other two are eager `v_expand`,
  each inside its actual `compute_graph_value_expansion` CPU scope.
- No projection or finish template contains `v_expand`; the finish graph
  therefore cannot double-count the eager BMM. Capture-template API rows
  have no request NVTX range and claim no CPU duration.
- The independent audit uses raw read-only SQLite queries and imports no
  production attribution code. It checks process/launch correlation, explicit
  clone ancestry, graph membership, API ownership, matrix shapes/work and
  complete counts against the production result.
- Negative checks with empty lineage or one deleted observed clone edge
  both fail with `missing, duplicate or unknown GPU graph nodes for replay`.
  Neither incomplete lineage nor a numeric-ID inference is accepted.

The complete source manifest is `source_sha256.json`; the audited production
graph source has SHA-256
`95432e0d32b1f7e52648b5abc17348a8b09638abb5b4b9cfe46137fabf2bfe3b`,
and `echo_model.py` has SHA-256
`7ccb7ffb0445483569d75074749d1dd4099134288827f3e1f77ed7973ed42e71`.
The independent result also hashes its raw inputs and audit scripts.

This validates graph-v2 instrumentation mechanics only. The
[output-layout checkpoint](deepseek_output_layout_checkpoint.md) records
separate C5/C6 component evidence. Full production numerical acceptance and
the next quiet full-workload measurement remain separate requirements; old
C3 profile numbers do not validate graph-v2 performance.

## C6 dense-only profile accepted and independently audited

This section covers formal `motivation_c6_20261004_u16_r2_01`, profile
`motivation_c6_dense_profile_20261004_01` and FLOPs
`motivation_c6_flops_20261004_01`. Earlier C3 measurements above remain C3
evidence. The C6 diagnostic does not validate later C7 source changes or
complete the overall first-visit/E2E optimization goal. The
[C6 profile plan](deepseek_motivation_c6_dense_profile_plan.md) records scope
and commands; root owns integration and publication decisions.

The profile ran from `/tmp/deepseek-motivation-c6-frozen-5vbx7859` against the
absolute accepted formal reference in the shared repository. It captured only
`dense_prefetch`: setup trace 1, cold request trace 2 and first revisit trace 3.
Profile source SHA-256 is
`a21b2b423605d0c807c7e7ed1409adbae3d0da0566bbd0d7b3fb361edf62942c`;
formal source SHA-256 is
`bd3e91265fed36f182d291b1bea3a659a2214c686e8236ed42da663c07775bff`.
Instrumentation expands the profile source manifest; production dependency
identity was verified against formal C6 before and after execution.

The configuration is sixteen users, two rounds, H=65,536, A=128, chunk=1,024,
ten independent checkpoint workload blocks, P=65,536 and NH=16,777,216.
Three warmup requests precede measurement. Precision remains mixed FP8/BF16
with TF32 disabled; FP32 norm weights and residual arithmetic are retained.
PyTorch records NVIDIA H200, 132 SMs, 150,121,545,728 device bytes,
PyTorch 2.12.1+cu130 / CUDA 13.0. The monitor's `nvidia-smi` model label is
M403; both records are preserved without rewriting hardware provenance.

Root granted GPU 0 exclusively for this capture. Wrapper session 68476 exited
0, and GPU 0 was released before all subsequent CPU analysis. The 1,480
original data/log/profile files, totaling 159,991,484 bytes, were hashed before
and after relocation. Their identities are in `relocation_manifest.json`.
The accepted data directory is
`experiments/deepseek_v32_motivation/output/data/motivation_c6_dense_profile_20261004_01`.

The run contains 83 process-list samples, 77 with the profile Python process
1778201 and six empty. No unexpected process was observed. Monitoring starts
at 2026-10-03 20:00:22.059805 UTC, 6.589247 s before profile metadata start,
and ends at 20:07:45.808096 UTC, 61.695702 s after completion. Actual intervals
are 5.096962-29.611132 s. This is bounded process observation, not proof of
continuous isolation. `gpu_observations_boundary.json` records exact times,
monitor source and attachment hashes.

The standalone numerical audit checks all twenty saved profile outputs against
the accepted formal HBM files by request ID, input identity and exact uint8
tensor bytes. Three warmup, one cold, fifteen preparation and one revisit
payload all pass for hidden and logits; all values are finite. It imports no
project acceptance/report code, and CUDA remains uninitialized. Source, tensor
and file hashes are in `analysis/independent_numerical_audit.py` / `.json`.
The formal run does not save warmup tensors; this separate profile numerical
check must not be described as a formal warmup tensor recheck.

Frozen C6 production analysis completed for FLOP verification, pipeline,
operator MFU and aggregate MFU. The independent raw-SQL audit then checked
every invocation, graph node, pipeline union and all 45 operator groups:

- 9,243 matrix calls / primary kernels: 9,102 cold and 141 revisit. The useful
  work matches the corresponding shape, precision, chunk and actual layer.
- 7,260 graph matrix APIs and 660 eager `v_expand` calls. Finish templates omit
  expansion, so it is not double-counted.
- 26,366 matrix-owned GPU activities and all 63,688 physical GPU activities.
  All nonmatrix work and copies are retained; none is unattributed.
- 1,320 request graph replays, 45,652 graph GPU activities and 1,374 explicit
  clone edges, with complete executable/process/launch membership.

Audit source/result are `analysis/independent_attribution_audit.py` and
`analysis/operator_mfu/crosscheck.json`. The normalized physical activities in
`analysis/independent_gpu_activities.jsonl` preserve timestamps, launch and
process identity, stream, actual layer, segment/chunk, copy kind and bytes.

### C6 candidate MFU and remaining metadata

API active time below is the sum of per-call GPU unions including owned
helpers. API MFU is sum(ideal time) / sum(API active time), not an average of
operator percentages. Formal stage means use all sixteen samples per group;
the profile has one cold/revisit sample. Cross-measurement differences do not
isolate CPU cost or guarantee an available speedup.

| Phase / segment | API active ms | API MFU % | Formal mean ms | Formal MFU % |
|---|---:|---:|---:|---:|
| Cold candidate | 7.787539 | 28.846001 | 12.161162 | 18.471866 |
| Cold history | 1779.178790 | 54.247513 | 2357.703334 | 40.936458 |
| Cold request | 1786.966329 | 54.136814 | 2376.949976 | 40.699495 |
| Revisit candidate | 7.306827 | 30.743763 | 34.335472 | 6.542486 |
| Revisit request | 7.306827 | 30.743763 | 40.261614 | 5.579492 |

Revisit candidate GPU work includes 14.708310 ms mapped-host gather,
2.701870 ms indexer API (2.627085 ms primary core), 1.208521 ms MLA,
0.838438 ms exact top-k, 0.630083 ms cache argsort, 0.471970 ms eviction,
0.462915 ms slot allocation and 0.326656 ms native resident selection.
Cache argsort/eviction/slot allocation each have ten instrumented API calls
but respectively 220/220/230 GPU activities. This remains substantial metadata
work despite compute graph replay. The `replay_input_copy` stage costs
0.317060 ms; projection graph input preparation separately costs 0.038465 ms.
`pool_operation` copies cost 0.257760 ms.

`pool_operation` has 1.042626 ms total owned GPU activity and 14.387386 ms
exclusive per-thread CPU NVTX duration. Its 53.940697 ms summed inclusive
duration contains nested scopes and must not be added as disjoint CPU work.
The stage contains the resident-selection and copy costs above; do not add
those to its total again. Gather CPU scope duration is 0.635501 ms and does
not describe its 14.708310 ms physical GPU copy interval.

Independent launch exposure for revisit records 5.645 ms synchronization API
union, of which 4.485 ms intersects GPU activity and 1.124 ms lies in GPU gaps
within the request GPU span. Kernel-launch API union is 4.318 ms, with
1.672 ms GPU overlap and 2.646 ms in gaps; graph-launch union is 1.839 ms,
with 1.749 ms overlap and 0.090 ms in gaps. Memcpy API union is 1.840 ms,
with 0.778 ms overlap and 0.993 ms in gaps. Outside-span intervals account for
the remainder. API source/family rows overlap and must not be summed. These
are exposure measurements, not causal or wholly removable CPU overhead.

### C6 physical gather overlap and observed queueing

Cold history requests/resolves 20,643,840 resident records and cold candidate
655,360 resident records without host gathers. Their gather overlap ratios are
undefined, not zero or 100%. Revisit candidate has ten gather kernels, one per
actual target layer, transferring 655,360 records / 754,974,720 bytes. All are
history misses, with a 14.708310 ms sum/union of gather GPU intervals on stream
29. Matrix computation runs on stream 7.

The gather union intersects neither primary matrix kernels nor any
matrix-owned helper activities: both overlap times and ratios are zero.
Every target j=1..9 also has zero overlap with actual layer j-1 in the same
segment/chunk. Target 0 is startup. No other layer accounts for hidden matrix
overlap. Target-layer first matrix and first MLA consumption both follow its
gather completion. Only 0.457412 ms intersects any main-stream GPU work;
those intersections are metadata/copies. The independent per-layer audit is
`analysis/dense_overlap_by_layer.csv`, with segment counters/ratios in
`dense_segment_overlap.csv` and full inventory in `dense_kernel_inventory.csv`.

The separate `analysis/independent_overlap_review.py` / `.json` rechecks all
63,688 physical activities against raw SQL and all twenty event edges without
importing the preceding audits or production analysis. It confirms actual
layer ownership, byte counters, null cold ratios and all overlap results.

Raw event and synchronization analysis resolves twenty edges: main-stream
metadata-ready events consumed by gather stream 29, then gather-ready events
consumed by main stream 7. For every lookahead target 1-9, the previous layer's
projection graph is submitted while that gather remains active, but its first
matrix helper and core start afterward. There is no captured main/context
synchronization API between each gather submission and the preceding layer's
projection graph submission. The explicit wait on this gather-ready event is
submitted later than the projection graph. Stream 29 is recorded as
non-blocking and stream 7 as null; both have priority 0.
The independent review additionally verifies that each graph API finishes
0.555525-1.036319 ms before gather ends. Its first matrix helper starts
24.992-44.992 us after gather completion, and the first core starts
29.408-50.144 us afterward. No synchronization API interval intersects the
investigated launch window, including APIs that begin before the window.

For target layer 1, relative to the first candidate GPU activity:

| Recorded interval / event | Start ms | End ms |
|---|---:|---:|
| Layer-1 gather GPU | 7.137032 | 8.608081 |
| Layer-0 projection graph CPU launch | 7.893033 | 8.052556 |
| First layer-0 workspace D2D copy CPU launch | 7.276123 | 7.289815 |
| Same D2D copy GPU | 8.557361 | 8.608369 |
| Layer-0 first matrix helper GPU start | 8.633073 | N/A |
| Layer-0 first primary GEMM GPU start | 8.637489 | N/A |

Each gather launch records 65,536 CTAs, 128 threads per CTA, twenty registers
per thread and zero static/dynamic shared memory. These launch resources are
not a measurement of concurrent CTA residency or scheduling. The observed
delay does not establish CPU submission, SM resource contention, copy-engine
or queue interaction, or an unrecorded dependency as the cause.

`analysis/stream_dependency_audit.py` imports no production attribution code.
Its JSON, per-layer CSV, event edges, raw events/sync rows joined to APIs and
queued main-stream copies preserve the evidence and hashes. All 81 event GPU
timestamps are zero, so actual completion time cannot be inferred from them.
Raw syncType 2 occurs for both `cudaStreamSynchronize` and
`cudaStreamWaitEvent`; the audit classifies by joined API names and keeps the
raw enum. No DRIVER table is present. One context-sync row lacks a recorded
API correlation and occurs after all investigated launch windows. Absence of
an observed runtime wait therefore does not prove absence of every device
dependency. Further causal tests need a separate authorized GPU run and must
not be inferred from this single revisit trace.

No publication README, report material or old accepted run was changed or
removed during this diagnostic. All subsequent implementation changes require
their own exact-output acceptance and matching-source performance evidence.
