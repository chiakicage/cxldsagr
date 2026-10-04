# ECHO shared cache implementation checkpoint

> **2026-10-03 撤回说明：** 本文涉及的旧 DeepSeek 4 GiB / W / chunk 对照已按用户要求撤回，
> 相关实验源码与运行产物已清理；下文仅保留当时的工程过程，不再证明当前容量或性能。
> 当前入口为[固定 P/NH 容量实验](../../../experiments/deepseek_v32_echo_cache/README.md)。

> Experiment location: [deepseek_v32_echo_cache](../../../experiments/deepseek_v32_echo_cache/README.md).
> Historical commands, source paths and hashes below remain unchanged; use the
> [migration record](echo_cache_experiment_migration.md) to locate moved artifacts.
> This directory migration is not a new measurement.

Updated: 2026-10-03. Goal: implement the complete
[implementation plan](echo_cache_implementation_plan.md), including P6.
Status: **P0–P6 complete.** The runtime gates, standalone sweep and seven formal
GR runs / 896 requests are accepted. C1024/W1024 passed the serving-default
review; the final reports are published and affected-old-result cleanup is
applied. The [publication ledger](echo_cache_publication_ledger.json) records
the accepted identities, selected configuration and cleanup receipt.

Follow-up: the shared/dense host ledger now includes each pinned allocator bin,
and all DeepSeek schemes reserve 40 B of observed CPU execution scratch. The
previous logical-host accounting was insufficient. Updated GPU2 core/dense
checks passed 44 cases; details and source boundaries are in the
[pinned allocation checkpoint](echo_pinned_allocation_checkpoint.md). Earlier
checkpoint passes below predate this correction and do not validate its source.

## Current acceptance summary

- The frozen real layers 0–2, independent-empty-cache 64K+1K gate passed,
  followed by 279 affected GPU tests and a 2,338-test CPU regression. Runtime
  source identities and the CPU result's main-tree boundary are recorded below.
- Four memory gates accepted 11 cases, 352 requests and 172 observed phases.
  The [memory audit](echo_cache_memory_audit.md) and
  [acceptance ledger](memory_acceptance_summary.json) distinguish directly
  sampled configurations from analytic shape bounds and process-memory limits.
- Both sampled C1024/C2048 diagnostics are accepted. Each completed all 128
  four-scheme requests and audited detailed traffic on 16 predeclared requests.
- `20261003_echo_shared_chunks_02` is accepted and
  published（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/chunk_sweep/chunk_report.md`）.
  C256/C1024/C2048 passed the unchanged numerical gates and were timed;
  C512 was excluded by one cross-C extend-hidden failure and C4096 by budget.
- The five-case formal GR matrix and C1024/C2048 deployment repetitions are
  accepted, with complete source/artifact and candidate-hidden/logits audits for
  all 7 runs / 896 requests. Run identities and review boundaries are in the
  [formal GR monitor record](echo_cache_freeze_gate.md); the first C256 case uses
  the separately documented [auditor recovery](echo_gr_auditor_recovery.md).
  C1024/W1024 passed final review, and the
  GR report（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/echo_chunks/results.md`） is
  published. The publication ledger records completed replacement and cleanup.

## Current implementation

- Shared global host pages and per-layer device pools, session views, sentinel slot,
  native actual-claim prefetch and FIFO event priorities are implemented. See
  [native evidence](echo_cache_native_checkpoint.md) and
  [core evidence](echo_cache_core_checkpoint.md) for source identities and independent
  official-helper differential checks.
- The model executes index-K write → fused historical prefetch → exact top-k → direct
  HBM main-KV append plus asynchronous host backing → exact recall → MLA. A source
  reservation precedes projection so pending copies plus current projection stay
  inside the two-source window. Main KV, indexer-visible length, and committed
  length remain distinct.
- Serving uses one backend-owned pool per layer. Shared resources, per-session
  indexer/page/hint state, active workspace and host page admission have separate
  reservations. CPU free-page arrays and private CPU page tables are included.
  NH planning leaves DRAM for all assigned-page ownership tables.
- Prefill scheduling is at model level, with one transaction and one last-token
  head per request. Candidate execution defaults to a complete batch, independent
  of prefill C. Oversized exact unions split consumption only and count fallback.
- Failure drains precede rollback. Failed drain poisons the backend. Partial pool
  construction cleans up, and failed cache replacement poisons the model.
- Full model cache reservations are checked per device before cache allocation,
  with an aggregate DRAM limit. Diagnostic snapshots are separate CPU storage and
  validate the unchanged session/host ownership domain.
- Detailed full-union/fused-before-hit/cross-consumer reread diagnostics are opt-in;
  formal latency does not enable their map snapshots or synchronous inspection.

## Historical verification before the accepted replacement freeze

The entries in this section describe earlier source versions and milestones.
They are code checks, not paper experiments; current accepted gates are recorded
under the replacement-freeze section below. Later runtime changes require
affected checks to be repeated.

- Native GPU 1: 24 checks, including changed 64K+1K indexer bitwise parity with
  official resident DeepGEMM and differential official allocator/priority helpers.
- Shared generic cache GPU 0: 17 checks, including CPU reference cases, cross-owner
  reuse, partial-free recall, nondefault streams, native metadata integration,
  snapshot invalidation and retained-source lifetime.
- Real checkpoint GPU 2: `DEEPSEEK_SERVING_CHECKPOINT=/mnt/user-ssd/chenkaiqi/DeepSeek-V3.2`
  with `models/deepseek_v32/tests/test_serving_checkpoint.py` passed four schemes on
  the independent 10-block dense input-replay surrogate. Prefix 2304; candidate
  lengths 16 and 23; all candidate hidden and last-token head agree. This is not
  the 64K serving experiment or the three-layer 64K+1K gate. Latest observed repeat: 21.05 s.
- A later GPU 2 regression passed 115 checks (the then-optional 61-layer test was
  skipped), including 10-block four-scheme A→B→A cross-session eviction and revisit
  checks. This remains a numerical regression, not a performance result.
- Targeted model/cache resource CPU checks passed, including impossible budget
  rejection before allocation, source reservation before projection, explicit
  candidate geometry and poison handling.
- The final live-tree checkpoint gate used `/preset-models`: CPU boundary checks
  passed 43 cases; GPU session 59658 passed real layers 0–2 at 64K+1K in 35.36 s.
  All 1024 extend hidden and prefix/extend last-token logits were bitwise equal.
  The 1212-file source manifest before/after was
  `a6dcea67735358d25f796290344f7db721e3de3db8b41b08a5a8f382b25e2e95`,
  with no changed file during the gate. External MFU edits resumed afterward;
  formal measurements subsequently moved to a frozen detached worktree, where
  the gate was repeated. This live-tree pass is not re-signed for later sources.
- A global CPU regression during parallel edits found only one stale resource
  boundary expectation (1828 passed, 718 GPU/opt-in skips). That expectation was
  corrected; the later 2,338-test complete CPU result supersedes this incomplete
  regression milestone.
- The obsolete 61-layer acceptance was removed following the user's explicit
  correction and current `AGENTS.md:102–113`. Its loading-only process PID 715200
  / exec session 70544 was stopped, terminal exit 143, and the PID is absent.
  It produced no numerical or performance evidence. The required acceptance is
  real checkpoint layers 0–2, sequential hidden/residual propagation, independent
  empty resident/offload prefixes, and all 1024 extend hidden states at 64K+1K.

## Completed P6 publication

- The default review（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/echo_chunks/default_review.json`）
  selected C1024/W1024 for the declared 16-user, two-pass 64K+128 workload,
  NH/P, budgets and H200 runtime identity. Four accepted memory runs are bound
  to the formal workload through the
  [verified identity record](echo_memory_formal_workload_binding.json).
- Report run `20261003_echo_gr_chunks_publication_01` and the GR README are
  published. The report retains C2048's lower cold-request/p95 tradeoff and
  ECHO's 15.08% higher full-trace time than same-C serial sparse; choosing the
  chunk does not establish a fused-prefetch speedup or a universal optimum.
- Prepared03 cleanup was applied by root session `35272`, exit 0: 93 files
  installed, 486 DeepSeek-only files removed and 1,590 retained NOSA files
  verified. The [ledger](echo_cache_publication_ledger.json) identifies the
  receipt and preserved scope. All planned P6 deliverables are complete.

The corrected external auditor passed 111 CPU tests. The report-layer workload
identity binding fix passed two targeted groups of 37 and 20 tests, as recorded
by the executing agents/root. These are validation/reporting changes; no new
runtime source change, GPU execution or repeated raw audit was introduced by
this documentation closeout.

## Controlled serving configuration

The accepted GR matrix fixes NH=1,050,624 (16 padded 65,664-token sessions),
P=32,768, HBM=4 GiB, DRAM=64 GiB, 16 sequential users and two complete traversals.
The fixed-workspace comparison reserves 2048 queries for every C; deployment
comparison reserves C. C=2048 is identical in both matrices and is measured once
per repetition. Resource planning gives 15 ECHO sessions at workspace 2048,
and 16 at C=256/512/1024 with their own workspace. Accepted C2048 and C1024 memory
runs directly observed those respective capacities: C2048 had zero second-pass
hits, while C1024 ECHO/serial had all 16. Smaller-C capacities retain the analytic
coverage boundary in the memory audit; C512 is numerically excluded from timing.
These memory/diagnostic observations do not establish formal latency. Every
formal request saves all hidden/logits tensors for independent CPU recomparison; intrusive allocation
and per-layer traffic diagnostics run separately from formal latency.

## Historical first frozen measurement attempt

These failed attempts predate the corrected runtime and accepted replacements.

- Detached worktree:
  `/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-20261003-shared-chunks-01`.
  Its independent `/preset-models` 64K+1K gate passed in 30.01 s (session 36502),
  all extend hidden and last-token logits bitwise equal. Before/after checkpoint
  source SHA was `4530d282a22b30e7bfcc3d0a31a301756c8b61082eb5ecc2817298796526c89c`.
- `20261003_echo_shared_chunks_01` exited before timing (session 18065, exit 1).
  C1024 resident/offload passed, but C256 resident versus C1024 prefix hidden
  failed the existing rtol=0.01 / atol=0.02 gate: 1,622 of 469,762,048 elements.
  The run is not an experimental result and nothing was published. Diagnostics
  were kept outside `experiments/`; exact selections and per-layer arithmetic
  were subsequently compared before the replacement timing protocol.
- `20261003_echo_memory_c1024_01` also stopped before acceptance (session 8746,
  exit 1). The first HBM prefill sample exposed 8 B of CPU scalar staging in
  `exact_topk` → `aten::isfinite` → `aten::ne`, whereas the resident CPU execution
  allowance was zero. Its observed CUDA fixed-plus-active peak was within the
  HBM reservation. CPU scratch ownership and other paths were then audited;
  this failed validation has no published experiment artifact.
- A later main-tree-only preflight fix rejects prefill/extend/workspace Q>P
  before replacing a resident cache. Targeted CPU checks passed 61 cases.
  It was not copied over either frozen diagnostic run; it is included in the
  accepted replacement runtime.

## Historical follow-up before the replacement freeze

- The old freeze's cross-C difference was isolated to batch-dependent FP32
  index-head linear rounding near tied scores. Changing only the index weights
  reproduces the first difference; logical-ID sorting removes that diagnostic
  difference but has **not** been applied to runtime consumption. For every C,
  resident/offload prefix/extend hidden, logits and exact selected sets were
  bitwise equal. Across C, the required extend range passed for 256/1024/2048;
  C512 had one failing hidden element. Full-prefix differences remain reported
  diagnostics under plan §5.4. See [numerical screen](echo_chunk_numerical_screen.md).
  These findings belong to the old computation source. The replacement nonmatrix,
  no-Hadamard and q2 runtime subsequently re-screened every budget-feasible C in
  the accepted standalone sweep.
- `cache/host_allocation.py` now exposes the actual power-of-two pinned allocator
  bin as owned storage, retaining a view of the requested logical shape.
  Ten 1,050,624-token, 576-BF16 host arenas own 20 GiB. Copy counts still use
  logical record bytes. CPU reference allocation remains unrounded. Automatic
  NH selection accounts for the stepwise bin sizes.
- The separate memory observer found pinned scalar handouts inside native
  PyTorch `nonzero` and `unique`, beyond Python scalar conversion wrappers.
  The missing cumulative volume is exactly 5,088 × 4 B + 640 × 8 B = 25,472 B;
  each path synchronizes its D2H scalar transfer before returning. Cumulative
  handouts are not a live-memory peak. This finding led to the completed collector
  and CPU execution reservation used by the accepted memory gates.
- The controlled GR wrapper now runs the independent tensor/evidence audit in
  external staging before publishing any run. A failed audit preserves its
  diagnostic files outside `experiments/`; two shell lifecycle checks passed.
  The combined report keeps fixed-workspace and deployment-workspace matrices
  separate and labels observed repeat ranges without asserting an optimum.

No new freeze or accepted P6 timing is represented by these follow-up fixes.

## Accepted replacement runtime and historical observer follow-up

- Fresh runtime freeze:
  `/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-20261003-shared-chunks-02`.
  Its 745 copied files matched main at creation; overlay SHA256 was
  `ab721f90a3d09bb07e8ec9624552be12a77f4791390274573ec5ceb33b5c5b0b`.
  Official dependencies and the installed environment are shared links with
  explicit before/after identity checks, not independent physical copies.
- GPU 0 real layers 0–2, `/preset-models`, independent empty 64K prefixes plus
  1K extend: **1 passed in 28.60 s**. Complete extend hidden and both last-token
  logits were bitwise equal. The 1,213-file gate source SHA remained
  `428240214945b379f0c3016b845b906d6b5924f236abe5c9b1b05fa21b84ab25`.
- GPU 1 affected model/operator/cache regression on this freeze: session 63696,
  exit 0, **279 passed**, with the separately completed three-layer opt-in gate
  skipped in that invocation. It includes the small ten-block four-scheme
  checkpoint test. A module-scoped compiler-cache reset isolates independent
  linear-test shapes from earlier checkpoint-test compilations; it changes no
  runtime quantization arithmetic or compiler settings.
- Full main-tree CPU regression: session 31915, exit 0, **2,338 passed,
  834 GPU/opt-in skips, 58 subtests**, in 63.88 s. The preceding NOSA collection
  and allocator-check failures were resolved by the parallel NOSA work. The
  DeepSeek runtime matches the freeze; this CPU result is not a claim that every
  unrelated main-tree file matches the earlier snapshot.
- The sweep auditor now reconstructs all requested-C resource plans from saved
  configuration. It rejects a feasible C fraudulently relabelled as a budget
  exclusion, including removal of that C's numeric/timing records. The freeze's
  1,228-file sweep SHA is
  `b727c720864641bc94a7a2cb4cfa69c1c1ddae2831e370feed882a043a716b6d`.
  The accepted standalone run `20261003_echo_shared_chunks_02` later used this
  unchanged sweep source identity.
- Memory runs `20261003_echo_memory_c1024_02` and
  `20261003_echo_memory_c2048_fixed_02` exited 1 and were not published. HBM,
  ECHO and serial_sparse each completed their 32-request numerical/budget checks;
  dense prefill/extend passed before the collector incorrectly rejected an
  allocation-free, in-place truncate. The collector was corrected to permit
  zero allocation events while retaining scope, fixed-inventory and pinned
  handout checks. Complete four-scheme 03 replacements are accepted below.
- Unaccepted full-instrumentation diagnostics were stopped/superseded outside
  `experiments/`. The replacement diagnostic tool sampled predeclared requests
  **0, 15, 16, 31**, covering every layer and actual prefix/extend batch there;
  it executed the whole four-scheme 32-request trace and saved/reopened all
  candidate hidden/logits. Unsampled traffic is unknown, never zero. Actual hit
  versus rebuild classification comes from the trace. This satisfies the plan's
  per-layer mechanism requirements without asserting full-trace traffic from
  the sampled records. The C2048 replacement launch supplied `.venv/bin` in
  `PATH` so FlashInfer could find `ninja`; its earlier failed launch produced
  no result.

The observation tools, complete memory gates, sampled diagnostics and standalone
sweep described as follow-up work at this milestone have since been accepted.
The formal GR matrix, required repeats and serving-default review are accepted;
the final GR report is published and affected-old-result cleanup is complete.
Valid latest MFU reports are separate deliveries;
their retention boundary is in
[publication scope](echo_cache_publication_scope.md).


## Historical observer freeze 04 rollout

The 749-file observer freeze at
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-20261003-gr-tools-04`
has overlay SHA256
`d5c7de6e8dc7b577ac69322cbb3d219510ec116bf71b18edb04dd8b347d77b0a`.
Root compared its complete `gate_sources`, `sweep_sources`, and installed
`backend` fields with observer freeze 03: all identical. The accepted runtime
checkpoint/GPU evidence above applies to these unchanged runtime sources; the
new observer subsequently passed the diagnostic runs listed below.

- `BatchObserver` now wraps the allocate/release callbacks captured by the
  prefix pool before per-request instrumentation. It also attaches to existing
  sessions and restores callbacks afterward. Sixteen CPU checks cover real
  `PersistentGRRunner` cold/revisit/eviction lifetimes and callback restoration.
  Failed sampled diagnostic 03 runs remain external and are not results.
- Corrected sampled diagnostics 04 ran at C1024/W1024 (GPU 1,
  session 25484) and C2048/W2048 (GPU 4, session 82252), using requests
  0/15/16/31 for detailed observation and all 32 requests for tensor comparisons.
- Complete memory runs 03 started from the unchanged runtime in observer
  freeze 03: C1024 all schemes (session 43071), C2048/W2048 all schemes
  (25449), C512 HBM (7779), and C256 HBM+dense (70186). The first three were
  accepted; C25603 failed and was replaced by accepted C25605. Final acceptance
  independently inspected actual phase matrices in addition to cases/requests.
- All checkpoint paths in these runs are `/preset-models`. No 61-layer model
  acceptance was required or run. Formal latency waited for these intrusive
  checks; these historical milestones do not supply timing samples.


## Formal timing execution protocol, declared before new timings

This protocol was declared before timing. After the owned intrusive
memory/diagnostic jobs terminated, the standalone sweep completed and the GR
matrix started sequentially on GPU 0 from observer freeze 04.
The standalone run ID is `20261003_echo_shared_chunks_02`; all requested
C=256/512/1024/2048/4096 are screened afresh under plan section 5.4. An earlier
C512 rejection cannot be carried into this source version. The sweep retains
all required numerical evidence, natural-state first extends, interleaved cold
samples and verified common-prefix repeated extends.

The initial GR prefix is `20261003_echo_gr_chunks_01`, using the accepted
sweep's C set. Four schemes execute a complete 16-user, two-pass 64K+128 trace
for every fixed-workspace and deployment-workspace case; C2048 is the same
configuration in both protocols, not a second sample. Each trace must pass the
saved-tensor/source audit before publication. Detailed diagnostics and memory
validation remain separate from latency.

After the first matrix, repeat the leading ECHO deployment configuration and
the C1024/C2048 deployment controls with all four schemes. If the leader is a
smaller C, its own diagnostic is required before selection. Review complete
trace totals, cold prefix, revisit hit/rebuild distributions and p95 together.
If differences are within the observed repeat variation, do not claim an
optimum and prefer the smaller workspace under the declared selection rule.
Additional repeats are justified by unresolved variation, not by discarding
slow samples. Final selection must cite accepted artifacts and budget coverage;
no default has been selected by this protocol declaration.


## Accepted diagnostics and observer freeze 05

Both sampled diagnostics from freeze 04 completed and passed separate CPU
artifact audits, then were copied into main with complete before/after file
hash equality and additive import receipts:

| Run | Complete requests | Observed requests | Layer batches |
| --- | ---: | ---: | ---: |
| `20261003_echo_gr_diagnostics_c1024_04` | 128 | 16 | 7,840 |
| `20261003_echo_gr_diagnostics_c2048_04` | 128 | 16 | 5,280 |

All saved candidate hidden/logits are bitwise equal to their same-C HBM
references. C1024 ECHO/serial revisit samples have only the ten extend batches;
C2048 sampled requests rebuild prefix. These are mechanism diagnostics, not
latency measurements. Their 1,390-file source hash is
`60fc2e1cbd26155a6d347d3a817910c10ed11d8e0b3af21b3dde06ed93025662`.
Freeze 04 passed source/import verification again after both runs.

C256 memory run 03 failed at HBM request 31 because a cross-clock 1-microsecond
allocation-matching window rejected 774 valid short-lived generations. Complete
per-address allocation sequences, including ordinary allocations, had matching
counts. The observer now checks those generations bijectively, validates
sizes/free order and uses allocator-history free-completed times for active
peaks. No tolerance was widened and no visible-only peak was substituted.
The original failure reparsed with no evidence errors; 58 small GPU tests passed.
This does not reclassify the failed run as accepted.

New observer freeze 05:
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-20261003-memory-tools-05`,
755 copied files, overlay SHA256
`b056164fb7ac62760914c8113356a0bb1b433e82e60a1ba5d6a17f1deb0a283a`.
Root verified complete gate/sweep/backend identities equal freeze 04.
`20261003_echo_memory_c256_dense_05` completed the full HBM+dense 32-request run
(session 12361, GPU 6), with all four requested samples, and is accepted. The
successful C512, C1024 and C2048 memory runs retain their original observer-03 identities.

## Temporary storage cleanup requested by the user

Removed the failed old sweep under `/tmp/echo-chunk-sweep-20261003_echo_shared_chunks_01.6kAVpa`
and 66 obsolete DeepSeek output/profiler files (failed source-drift run,
withdrawn candidate01, replaced official/control/NCU outputs and retired local
split-K tensors). About 10.9 GiB of logical file content was removed. Current
accepted MFU source/output copies, active helper scripts and NOSA materials were
not removed by this cleanup. The external receipt is
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-runs/tmp_cleanup_20261003.json`.
At the final check, the overlay had 28 GiB available; concurrent unrelated
cleanup also increased free space. All new large ECHO artifacts use the SSD
TMPDIR. Historical references to the deleted failed sweep describe an earlier
run, not a still-available evidence directory.


## Accepted memory, standalone and formal GR runs

The complete engineering memory ledger is now
[memory_acceptance_summary.json](memory_acceptance_summary.json), SHA256
`5d5de6784c0999fdbdb6bbe74ed61c9199f4a0882278182dee6728dc4f4b4698`.
It accepts 11 directly sampled cases, 352 executed requests and 172 observed
phases across the four memory runs, and explicitly distinguishes shape bounds
from directly measured configurations. All four source/import guards passed;
6,243 original data/log/profile files (25,114,372,182 bytes) were copied into
main with identical hashes. C25605 carries the independent audit tools and
records in its additive `acceptance/` directory, whose manifest SHA is
`873a193806b2ddfc52217f0e8102853199c29133f91348c54f10b2746bfef114`.
The original manifests were not rewritten. These are budget acceptance records,
not formal latency samples.

After every owned GPU/profiler and heavy CPU/SSD task terminated, root started
`20261003_echo_shared_chunks_02` on GPU 0 from freeze 04 (session 94098).
Staging is
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-runs/echo-chunk-sweep-20261003_echo_shared_chunks_02.FOFc6b`.
The fresh allocation-only preflight confirms C256/512/1024/2048 feasible and
C4096 excluded: 5,108,643,392 bytes needed versus the 4,294,967,296-byte cap.
All budget-feasible candidates were screened again before timing; no older
numerical exclusion was inherited. C512 failed one cross-C extend-hidden element
under the unchanged threshold and was excluded. C256/C1024/C2048 passed and
completed the cold-prefill, natural first-extend and common-prefix protocols.
The sweep passed independent artifact acceptance and is published in the
standalone report（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/chunk_sweep/chunk_report.md`）,
with its publication provenance（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/chunk_sweep/provenance.json`）.
Original data and acceptance/import receipts remain in
`experiments/deepseek_v32_echo_prefill/output/data/20261003_echo_shared_chunks_02/`.

The formal GR matrix and required deployment repetitions now comprise 7 accepted
runs / 896 requests from freeze 04 on GPU 0. Their complete trace, source and
saved-output audits are accepted; the first C256 run's original wrapper failure
and corrected external-auditor recovery remain distinguished in the
[recovery record](echo_gr_auditor_recovery.md). Current run identities and
lightweight verification are recorded in the [freeze gate](echo_cache_freeze_gate.md).
The selected C1024/W1024 configuration, published report and applied cleanup
receipt are recorded in the [publication ledger](echo_cache_publication_ledger.json).
P0–P6 are complete within the declared workload and implementation scope.
