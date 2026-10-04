# ECHO cache implementation requirements audit

> **2026-10-03 撤回说明：** 本文涉及的旧 DeepSeek 4 GiB / W / chunk 对照已按用户要求撤回，
> 相关实验源码与运行产物已清理；下文仅保留当时的工程过程，不再证明当前容量或性能。
> 当前入口为[固定 P/NH 容量实验](../../../experiments/deepseek_v32_echo_cache/README.md)。

> Experiment location: [deepseek_v32_echo_cache](../../../experiments/deepseek_v32_echo_cache/README.md).
> Historical commands, source paths and hashes below remain unchanged; use the
> [migration record](echo_cache_experiment_migration.md) to locate moved artifacts.
> This directory migration is not a new measurement.

Date: 2026-10-03. Target: the current
[implementation plan](echo_cache_implementation_plan.md), requirements P0–P6.
**P0–P6 are complete.** This engineering audit records the accepted implementation
and publication scope; performance conclusions remain bounded by the published
experiments.

Follow-up on the same date: the independent allocator investigation found a
further P1 defect: logical host storage understated owned pinned bins. Shared
and DeepSeek dense allocations now expose and charge full bins, and all schemes
reserve 40 B of observed CPU execution scratch. Updated checks passed 114 CPU
cases and 44 GPU2 cases; see the
[correction checkpoint](echo_pinned_allocation_checkpoint.md). Earlier source
identities and checkpoint passes below remain earlier evidence.

The initial review read the live workspace and previously reported terminal results.
It did not run a GPU task or modify/copy the frozen measurement tree. The parent
authorized two local corrections discovered during review: non-destructive
`Q>P` mode-switch admission and diagnostic source-manifest coverage. Their CPU
checks are recorded below. Concurrent MFU work and the measurement freeze have
different source identities; the audit does not transfer a pass between them.

Latest runtime acceptance is complete on freeze 02 and unchanged in observer
freezes 03–05: the real layers 0–2 independent-empty-cache 64K+1K gate passed
(28.60 s, all extend hidden and last logits bitwise equal), followed by 279
affected GPU tests. Main-tree CPU regression passed 2,338 tests with 834
GPU/opt-in skips and 58 subtests. Gate/sweep source identities and their exact
boundaries are recorded in the [implementation checkpoint](echo_cache_implementation_checkpoint.md).
Four memory gates and both sampled C1024/C2048 diagnostics are also accepted.
The standalone sweep `20261003_echo_shared_chunks_02` passed independent
artifact acceptance and is published（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/chunk_sweep/chunk_report.md`）.
The formal GR matrix and required C1024/C2048 deployment repetitions are accepted:
7 runs / 896 requests, including complete source/artifact and saved-output audits.
C1024/W1024 passed serving-default review, the final GR report is published and
affected-old-result cleanup is applied. The
[publication ledger](echo_cache_publication_ledger.json) records completion.

## Final verdict

P0–P5 are implemented, with meaningful unit, integration and checkpoint evidence.
The earlier mode-switch admission, diagnostic provenance and pinned allocation
defects are corrected. The replacement runtime passed its checkpoint/GPU gates;
the observer corrections subsequently passed complete independent captures.
The [memory audit](echo_cache_memory_audit.md) and
[acceptance ledger](memory_acceptance_summary.json) accept four runs covering
11 cases, 352 requests and 172 observed phases, with explicit direct-versus-
analytic coverage and process-memory boundaries.

P6 is also accepted. The
reviewed report（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/echo_chunks/results.md`）
selects C1024/W1024 for the fixed trace, binds the four memory runs through the
[workload identity record](echo_memory_formal_workload_binding.json), and retains
the cold-request/p95 tradeoff and ECHO's 15.08% slowdown versus same-C serial
sparse. The choice does not establish a fused-prefetch benefit. Prepared03
replacement/cleanup completed in root session `35272`, exit 0, with 93 installed,
486 DeepSeek-only removed and 1,590 retained NOSA files verified. No planned
implementation or publication gate remains open.

Source identity remains a constraint throughout these measurements. The accepted
runtime includes the mode-switch precheck, pinned-bin accounting and 40 B CPU
reservation. Observer-only fixes have separate identities and accepted captures;
any later runtime change requires affected checks and new measurement identities.

The failed first sweep is historical: C256 versus C1024 full-prefix hidden
differed at 1,622 / 469,762,048 elements. The replacement sweep re-screened every
budget-feasible C under the clarified protocol: complete same-C prefix/extend
hidden, last logits and exact selections are bitwise equal; across C, complete
extend hidden/logits retain `rtol=.01, atol=.02`. Full-prefix differences are
disclosed diagnostics. C256/C1024/C2048 passed; C512 failed one extend-hidden
element and C4096 exceeded budget. Neither excluded C appears in timing tables.

## P0: official behavior, scope and identity

| Requirement | Implementation and evidence | Audit finding |
| --- | --- | --- |
| Pin the official ECHO reference and distinguish code from paper | `indexer/echo.py` records ECHO `bc1b75c1000010d0ac6f032ebaac283255c050b1`; the plan cites official call sites and discloses local corrections | Satisfied in implementation/documentation. The local reference remains a read-only reference, not a new project dependency. |
| Freeze the existing dirty kernel before cache edits | `echo_cache_native_before.json` and `.patch` preserve pre-edit sources/includes; native checkpoint identifies its validation scope | Historical evidence exists. HEAD alone is not the dirty runtime identity. |
| Identify actual JIT code and local includes | ECHO `build_info()` hashes its model-specific CUDA/header files, wrapper/cache helpers, flags and shared CUTLASS headers; experiment provenance records loaded libraries | Implemented. Final artifact/source checks must follow the actual freeze. |
| Preserve explicit inter-query flags and disclosed policy | Serving enables fused prefetch only for `echo`; `serial_sparse` disables it. No EARLY_EVICT path was added. Offset remains last up-to-four-row mean with finite guard | Implemented. This is not a kth-score EMA and does not reproduce SGLang scheduler/radix/PD/graph behavior. |
| Check official normal-domain policy independently | Actual upstream AST functions and original upstream `topk.cu` are executed in `test_echo_official_policy.py`; post-alloc/priority helpers are also compared in `test_echo_cache_ops.py` | Prior GPU passes confirmed, not merely a handwritten local oracle. Exact terminal boundaries are below. |

The preservation claim is specifically about the cache transplant. Independent
MFU/nonmatrix/rotary/linear edits coexist in the workspace and need their own
numerical evidence; their existence cannot be hidden behind a cache-only claim.

## P1: shared ownership, admission and memory

| Requirement | Implementation and evidence | Audit finding |
| --- | --- | --- |
| Global host IDs, noncontiguous 64-token pages, per-layer shared HBM | `SharedSparseTokenPool`, `SparseTokenSession`, page translation in cache and native prefetch | Implemented. Generic width/dtype remains outside the DeepSeek-specific 576-BF16 native adapter. |
| P excludes slot-zero sentinel; host tail/candidate capacity counts | Shared records use `P+1`; maps/priority/free bitmap include sentinel storage; sessions reserve padded full capacity | Covered by allocator/counter tests and native sentinel checks. |
| Separate fixed shared, per-session and active workspace bytes | `estimate_shared_bytes`, `estimate_session_bytes`, `shared_bytes`, `session_bytes`, `execution_reservation`, and backend/model planners | Corrected to include each pinned bin and 40 B of observed CPU execution scratch. CPU free-page storage, CPU/device page tables and bounded per-layer counters are counted. Shared aliases count once. |
| NH constrained by both DRAM and HBM/private state | Serving planner includes global maps, one-session feasibility, indexer state, metadata workspace and all host-page-table DRAM headroom | Bin-aware monotone search now respects per-layer allocation steps; byte-boundary tests passed. Predicted session counts are not measured reuse rates. |
| Impossible session admission has no side effects | `PrefixSessionPool.acquire` checks empty-pool byte/page feasibility before dropping same-user or LRU state; allocator constructs metadata before changing ownership | Covered by impossible-request, changed-prefix, host quota, byte quota and allocation-failure tests. Allocation failure does not unconditionally clear unrelated users. |
| Offload selection and Q geometry rejected before cache replacement | Constructor/serving planner reject single-query top-k or Q>P; this audit added the missing standalone mode-switch Q/workspace check | **Corrected in this audit.** Previous `set_cache_mode(True)` could release a valid resident prefix before a Q>P runner-construction failure. New CPU tests preserve identities, lengths, contents, generation and continued execution. |
| Reserve pending KV sources before projection and bound all in-flight layers | `reserve_append_source()` precedes `attention.project`; shared pool reaps the write window before new source allocation; full storage identity is charged | Implemented. Delayed nondefault-stream D2H and source-view lifetime were exercised on GPU. |
| Account alignment, temporary overlap and physical execution peak | Pinned tensors expose complete owned bins; GPU fixed-block rounding and dynamic events are independently collected, with deferred frees tracked through `free_completed` | Accepted for the four memory runs' explicitly covered cases/phases; other listed configurations have analytic shape bounds. Cache-owned peaks are not process RSS or retained-allocator physical peaks. Cached unowned blocks remain separate from cache-owned capacity. |

Weights are reported separately by `backend.describe()`, and process CUDA
allocated/reserved peaks are separate metrics. Those process peaks include
weights, cache and ordinary allocations; they must not be labelled an isolated
ordinary-activation peak or cache-only process peak. The memory collector
classifies ordinary operations out of the cache acceptance ledger rather than
establishing an independently complete ordinary-activation peak.

## P2: layer order, direct append and asynchronous dependencies

| Requirement | Implementation and evidence | Audit finding |
| --- | --- | --- |
| Distinct committed, indexer-visible and written lengths | Cache `begin_step`, `declare_indexer_visible`, append and commit maintain separate fields | Implemented; phase-order and failure tests check this separation. |
| Index-K write → historical prefetch → exact top-k → main-KV append → exact recall → MLA | `EchoAttentionRunner._forward` follows this sequence; native receives initialized `history_length` and session page table | Implemented; current suffix is excluded from fused host reads. |
| Direct current KV HBM write plus host backing | Shared `append` uses projected records directly for HBM and a pinned-host D2H write | Implemented; no compulsory new-token H2D. A later genuine eviction/recall remains legal and is counted. |
| Source storage and host content remain valid until all reads/writes finish | `_HostWrite` retains the original source; copy-stream events, `record_stream`, `wait_host`, lease completion and release/drain order establish dependencies | Covered by delayed D2H, immediate host read, nondefault-stream transport and host-ID-reuse tests. No default-stream timing assumption is required. |
| Consumer slots/workspace cannot be reused while active | Exclusive nested layer operation leases and pending-prefetch ownership; next operation waits previous stream/consumer event | Implemented under the explicitly serial request model. Concurrent service and CUDA Graph support are not claimed. |

## P3: actual prefetch claims and FIFO events

| Requirement | Implementation and evidence | Audit finding |
| --- | --- | --- |
| Prepare sorts the full pool without evicting | `prepare_prefetch` sorts `age[1:]`, resets fixed scratch and creates a pending lease | Implemented; all-hit/zero-cap native tests assert no eviction. |
| Live global CAS claim, unique slot rank, actual-victim invalidation | Shared `claim_and_copy_warp` implements MISSING→CLAIMED, bounded atomic rank, ownership-checked old-map CAS and overflow rollback | Native tests cover duplicate IDs, cap exhaustion, nonidentity pages, record bytes, both map directions and no residual CLAIMED. |
| Q>P and cap `min(8192,P-Q)` | Model/cache and native ABI validators reject Q>P before launch; P excludes sentinel | Implemented, including the new earlier mode-switch precheck. |
| Actual copies/evictions distinct from overshooting reservation attempts | Physical-slot allocation log and fixed int64 actual counters; `metrics()` uses accurate totals | Implemented; attempt counter is not reported as copy volume. |
| Official protect/finalize/allocation clocks, including empty/all-hit events | Native protect/finalize advance a stream-ordered clock after data-parallel work; cache append/recall stamp actual allocations | Prior official AST differential and cache event-trace tests passed. |
| Release has no access event; clock rollover preserves ties/order | Owner-checked release clears mappings/free state without a stamp; `_clock_event` drains and rank-compresses before 9,000,000 | Implemented; rollover and stale-owner tests cover the local lifetime adaptation. |

Partial-free recall, immutable consumer protection and clock rollover remain
disclosed correctness adaptations. For tied priorities, the official differential
checks victim priority multiset/count/protection rather than inventing a unique
upstream tie order. Normal-path sort/unique/remap operations still incur PyTorch
allocation and host synchronization; measured scopes include those costs. No
claim that all recall planning is fused or fully GPU-resident is justified.

## P4: exact recall, transactions and snapshots

| Requirement | Implementation and evidence | Audit finding |
| --- | --- | --- |
| Complete logical exact union, padding ignored, existing selection protected | `ensure` validates written range and union capacity, translates global IDs, marks actual misses, protects hits and fills only missing records | Implemented. Partial-free allocation takes free slots first and only evicts the deficit. |
| Oversized union splits query consumption without clipping selection | `_consume` recursively splits query rows only on `WorkingSetTooLarge`; a single impossible query fails | Implemented and tested against complete selections. This is the local fallback, not official whole-batch behavior. |
| Unique reads per group; cross-group rereads reported honestly | Each `ensure` deduplicates its union; opt-in diagnostics record group recalls and previously consumed missing IDs | Synthetic/CPU checks and accepted C1024/C2048 GPU diagnostics cover these records. GPU traffic findings apply to the predeclared sampled requests; unsampled traffic remains unknown. |
| All layers/output/synchronization complete before commit | Both model entrypoints begin owned transactions, validate all written endpoints, synchronize, then commit | Covered by failure-before/after layer, omitted append, failed begin and retry tests. No single-layer check substitutes for checkpoint validation. |
| Failure drains, rolls back only owned steps, restores hints, or poisons | Both entrypoints drain before rollback; failed drain marks unusable state | Implemented and CPU-tested. Generic pool drain errors also poison its state. |
| Truncate/release only the session suffix/owner | Global page ownership plus suffix translation; indexer visibility rewinds; serving restores committed prefix hints | Cross-user A→B→A, release/reuse, changed-prefix and candidate-truncate tests cover this. Evicted historical HBM residency need not be restored on failure. |
| One global pool snapshot, stable ownership and quiescent restore | Snapshot records device records/maps/free/priority/clock and session pages/lengths; model includes hints/generation; topology and prefix invalidation reject stale snapshots | Implemented. Immutable host/indexer prefixes are retained, not duplicated; deterministic overwritten scratch is reset. Diagnostic snapshot bytes are outside serving capacity. |
| Fixed-state performance repeats really restore the same state | Chunk-sweep `restore_verified` re-snapshots and compares complete captured state identity | Accepted in standalone run `20261003_echo_shared_chunks_02`, including 20 common-prefix samples per timed C/mode. Formal GR deployment repetitions are also accepted; real serving evolves cache state across each trace and does not restore snapshots between requests. |

## P5: chunk, output, timing and experiment interfaces

| Requirement | Implementation and evidence | Audit finding |
| --- | --- | --- |
| One model outer chunk runs all chosen layers; no implicit scoring split | Standalone/serving own prefill loops; attention receives their exact batch; each block receives `chunk_size=len(hidden)` | Implemented and CPU-tested. The MLP geometry bug found earlier is no longer present. |
| Extend independent of prefill C; exact-consumption splitting separately counted | Explicit `extend_chunk_size`, default full candidate; separate capacity fallback | Implemented. Resource validation precedes transaction mutation. |
| Correct model scope and complete outputs | Standalone real 0–2 propagation; serving 10 independent source-input copies; embedding/MLPs/final norm/last-token head retained; full candidate hidden returned | The real three-layer 64K+1K gate and small ten-block four-scheme checkpoint test are accepted; full candidate hidden/head comparisons also passed the memory/diagnostic traces. All seven formal GR runs passed their independent complete-output audits, covering 896 requests. No 61-layer acceptance is needed. |
| Formal timing includes management/write/indexer/fetch/recall/MLP/output/sync | Persistent runner wall interval includes admission, cold prefix, all candidates, truncate and sync; model wall interval includes cache helpers and launch gaps | Implemented. Loading, JIT warmup, source capture, tensor comparison and snapshot restore are outside formal timing. |
| Explicit shared-pool CLI, fixed NH/P/workspace and exclusion reasons | New shared pool/host/workspace/extend flags; old per-session `--deepseek-slots` fails; allocation-free screens and schemas retain every measured/excluded C | Implemented. C=4096 budget rejection is a reported constraint, not a speed result. |
| Every formal scheme output can be independently rechecked | Root `measure.py` saves all actual hidden/logits tensors; `audit_numerical_tensors` reopens them and compares each request to HBM | Implemented after this review's initial finding. This is a correctness gate attached to the latency experiment, not an additional experiment result. |
| Detailed prefix+extend diagnostics are independent of formal timing | `echo_diagnostics.py` records every actual phase/layer/batch for predeclared requests 0/15/16/31, group/residency evidence, copy equations and manager scopes, while executing complete four-scheme traces | C1024/C2048 diagnostic runs 04 are accepted: 128 complete requests and 16 observed requests each. Raw selection matrices are checked live and fully hashed; saved tensors support offline count reconstruction. Unsampled traffic remains unknown. |
| Provenance includes reused experimental scope code and official inputs | This audit corrected diagnostic snapshot to include official sources plus the direct `Scopes` module; source-drift rejection test passes | Corrected before first successful GPU execution; accepted diagnostic artifacts carry their observer source identity. No files were copied into an actively running freeze. |

The formal runner's original per-request cache counters describe the final
forward call, normally extend, and its per-layer dictionary only retains the
last outer batch. `echo_chunks_report.py` now labels that boundary. These fields
must not be added up as full cold-prefix traffic or used as pre-fused token-hit
ratios. The separate diagnostic entry supplies complete actual phase information
for its predeclared sampled requests; it does not measure unsampled traffic.

## Evidence ledger and source limits

The first table preserves historical checks on their original sources. Current
replacement acceptance follows it; earlier passes are not re-signed for later
runtime or observer versions.

| Evidence | Confirmed terminal result | What it establishes |
| --- | --- | --- |
| Generic cache core, earlier source | GPU 0, 17 passed; hashes in `echo_cache_core_checkpoint.md` predate the pinned allocation correction | Earlier ownership, allocator, snapshot, direct append, asynchronous source and native integration checks |
| Pinned allocation and CPU workspace correction | CPU: 114 passed, 9 skipped; GPU2 exec 80702: 44 passed in 3.16 s; Ruff passed | Actual pinned backing bins, independent-layer accounting, automatic NH steps, one-byte budget limits, 40 B CPU workspace, source lifetime and unchanged transfer bytes; no full checkpoint/performance acceptance |
| Native cache/indexer | GPU 1, 24 passed; recorded in `echo_cache_native_checkpoint.md` | Native transport/metadata plus 64K+1K operator parity and upstream post-alloc/priority differential |
| Official free/victim/hint policy | Agent-confirmed exec session 65250, exit 0, **18 passed in 20.36 s**, GPU 3 | Actual pinned torch/graph free helpers, upstream bounded selector, finite hint expressions. This run did not save a complete runtime SHA manifest; do not invent one. |
| Live-tree checkpoint | Exec session 59658, exit 0, 1 passed in 35.36 s; 1,212-file before/after digest `a6dcea67735358d25f796290344f7db721e3de3db8b41b08a5a8f382b25e2e95` | `/preset-models`, real layers 0–2, independent 64K prefixes, all 1K extend hidden and both last-logit outputs bitwise equal on that source |
| Frozen-tree checkpoint | Native agent confirmed 1 passed in 30.01 s; 1,212-file before/after digest `4530d282a22b30e7bfcc3d0a31a301756c8b61082eb5ecc2817298796526c89c` | Same required numerical gate on the measurement freeze; does not include later live-tree preflight edits |
| This audit's Q>P correction | CUDA hidden; exec session 37364, exit 0, **61 passed in 2.02 s** | Model scheduling/admission, cache resource tests and chunk protocol tests; includes non-destructive prefill/extend/workspace rejection |
| Diagnostic entry/source fix | CUDA hidden; exec session 51050, exit 0, **8 passed in 1.95 s**; Ruff passed | Batch completeness, lifetime, output/count auditing and source coverage; no GPU/performance acceptance |

| Current accepted evidence | Confirmed result | Boundary |
| --- | --- | --- |
| Replacement runtime checkpoint | Freeze 02, **1 passed in 28.60 s**; gate source `428240214945b379f0c3016b845b906d6b5924f236abe5c9b1b05fa21b84ab25` | Real layers 0–2, independent empty 64K prefixes, all 1K extend hidden and prefix/extend last logits bitwise equal |
| Affected GPU regression | Session 63696, **279 passed** | Includes small ten-block four-scheme checkpoint checks; separately completed opt-in three-layer gate skipped in this invocation |
| Full CPU regression | Session 31915, **2,338 passed**, 834 GPU/opt-in skips, 58 subtests | Main-tree run with DeepSeek runtime matching the freeze; no claim that unrelated main-tree files match the snapshot |
| Four memory gates | **11 cases, 352 requests, 172 phases**; independently verified and copied | Exact run/source/phase coverage and analytic shape bounds in [memory acceptance](memory_acceptance_summary.json); C512 resident memory coverage does not confer numerical eligibility |
| Sampled C1024/C2048 diagnostics | Runs `20261003_echo_gr_diagnostics_c1024_04` and `20261003_echo_gr_diagnostics_c2048_04`; **128 complete / 16 observed requests each** | All hidden/head outputs match same-C HBM; traffic evidence covers predeclared requests only and supplies no formal latency |
| Standalone sweep | Run `20261003_echo_shared_chunks_02`, independently accepted and published（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/chunk_sweep/chunk_report.md`） | C256/C1024/C2048 timed; C512 numerical and C4096 budget exclusions retained. Real three-layer scope, no serving-default selection |
| Formal GR matrix and deployment repetitions | **7 accepted runs / 896 requests**, with complete source/artifact and candidate-hidden/logits audits | Five first-pass cases plus independent C1024/C2048 repeats; fixed/deployment C2048 is one shared control. [Monitor record](echo_cache_freeze_gate.md) and [first-case auditor recovery](echo_gr_auditor_recovery.md) preserve execution and validation boundaries. C1024/W1024 review and final publication are accepted in the [ledger](echo_cache_publication_ledger.json). |
| Validation and report-layer corrections | Corrected external auditor: **111 CPU tests passed**; workload identity binding: **37 + 20 targeted tests passed**, per executing-agent/root records | Validation/reporting changes only; frozen runtime unchanged. This closeout did not rerun tests, GPU work or raw tensor audits. |

Official-policy command, as confirmed by the executing agent:

```bash
CUDA_VISIBLE_DEVICES=3 MAX_JOBS=2 uv run --no-sync python -m pytest \
  -q --tb=short -x operators/deepseek_v32/indexer/tests/test_echo_official_policy.py
```

The test verifies the pinned official revision and clean helper files, builds
the original upstream `topk.cu` with a temporary binding, and executes extracted
upstream functions. Earlier 18-pass evidence should be retained as earlier
evidence; a final-source recheck is appropriate if those components change.

Historical identities when this audit's local corrections were first checked
(current frozen identities are in the implementation checkpoint):

```text
d06ed92e32b2a150703148d7fa69688af9c085ebcae46321cf3a673a91014191  models/deepseek_v32/echo_infer.py
e915840218f51057e5937474e1922699e43ebf2c53e4603c0372eaf996f9ef7f  models/deepseek_v32/tests/test_echo_infer.py
9d92c722c0a7cb124136b6706b081097d6aad7a89c3d0171efde6db500e9a0c0  experiments/gr_serving/src/echo_diagnostics.py
6de1823608b4ceab3777e1a429035ede07d15b36e50bdacfbf300ea861415285  experiments/gr_serving/tests/test_echo_diagnostics.py
```

## Experiment/reference discipline

- `measure.py` and the controlled chunk report have a genuine experiment target:
  complete serving latency and reuse under fixed hard caps. Numerical tensors,
  source hashes and independent checks are prerequisites attached to those runs.
- `echo_diagnostics.py` can provide mechanism/traffic evidence for that experiment.
  Its intrusive scope timings are labelled diagnostic, contaminated request
  latency fields are removed, and `formal_latency=False` is enforced. A CPU
  test pass or an unpaired diagnostic run is not a published speed result.
- `cache_memory_{audit,capture}.py` is an ancillary hard-budget check and
  engineering acceptance tool. Its schema and shell already say
  `latency_measurement=False` / no published latency samples. Even if accepted
  artifacts are retained alongside a valid serving run, they must remain linked
  acceptance evidence; do not turn a standalone memory-validation run into a
  paper performance experiment. The published README preserves this boundary.
- Fixed/logical bytes, observed rounded PyTorch cache peaks, process
  allocated/reserved peaks and uncovered external allocation categories must
  remain separate. None is automatically interchangeable with another.
- The standalone chunk sweep and GR replacement are accepted and published under
  new run IDs. Affected old DeepSeek results were cleaned after publication;
  retained NOSA evidence and valid slower controls are preserved according to the
  [publication ledger](echo_cache_publication_ledger.json).

## P6 acceptance complete

The runtime gates, four memory captures, two sampled diagnostics and full
standalone protocol are accepted. Standalone acceptance includes the clarified
same-C/cross-C numerical rules, independently audited exclusions, 2 cold warmups
and 5 fresh cold samples, natural first 128-token extends, and 5 warmups plus
20 repeated extends from verified common-prefix state per timed C/mode.

The five-case GR matrix and declared C1024/C2048 deployment repetitions also
passed their independent source/artifact and full-output audits: 7 runs / 896
requests. Serving-default review selected C1024/W1024 within the declared scope;
the final report and README are published, and the prepared03 cleanup receipt is
recorded in the [ledger](echo_cache_publication_ledger.json).

The accepted runtime identities are preserved. A future runtime change requires
affected gates to be repeated; this completed report/validation closeout changes
no runtime code and establishes no broader workload or model claim.
