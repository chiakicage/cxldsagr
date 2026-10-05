# NOSA fixed-serving indexer dispatch plan

Prepared 2026-10-05. This remains an unimplemented proposal; the unified-runtime
refactor does not implement or validate it. The descriptor-copy pilot completed
against frozen Python sources; its [result](nosa_copy_descriptor_result.md)
preserves the original diagnostic boundary. The target is
lower ordinary complete-candidate latency, followed by accepted four-scheme
measurements. Fewer allocation calls alone do not establish that target.

Continue from the current `models/nosa/cache/` and `models/nosa/execution/`
layout after the refactor acceptance finishes. Recheck the allocation and
dispatch assumptions below against that accepted source before implementing
the workspace. The diagnostic run IDs, commands and hashes remain historical;
they do not validate the refactored implementation.

The six original formal numerical-reference files needed by the frozen reviewers
are retained separately with their original bytes and hashes. See the
[reference relocation and reopening instructions](../../acceptance/unified_runtime_20261005/t007_reference.md).
Historical commands below remain unchanged; reopening uses only an in-memory
`FORMAL` path override. The relocation does not rerun these diagnostics.

## Evidence and selected change

The current [motivation report](../../../../experiments/nosa_motivation/README.md)
uses H65536/A128/C1024, P65536/NH16777216 and the full 32-layer checkpoint.
HBM candidate MFU is 73.6–75.1% of the separately measured FA3/matrix composition.
The [prequeue diagnostic](nosa_hbm_prequeue_diagnostic.md) reduced the pre-check
event interval by about 1.8 ms, while the [matched copy control](nosa_guarded_copy_control.md)
gave only 0.016–0.076 ms net median reductions against the original class.
These results motivate reducing remaining eager orchestration; none measures
the benefit of the workspace proposed here.

Current source has three relevant costs:

- `models/nosa/indexer.py::prepare_native_indexer_inputs` allocates IDs, validity
  and normalizers for every checked H64K layer. That is 96 GPU tensor allocation
  calls per 32-layer candidate. The offload joint path allocates the same three
  outputs in `operators/nosa/indexer/_indexer_cuda.py::select`.
- `cache/indexer_cache.py::workspace` creates a CPU scalar tensor solely to read
  its dtype width. Replace that probe with `dtype.itemsize`, preserving dtype,
  length, capacity, alignment and released-cache validation. The constructor's
  identical width probe can use the same expression; distinguish setup savings
  from timed-call savings.
- Selection constructs several active-shape views. Reuse the views and geometry
  obtained in the current invocation when passing them through adapters, rather
  than constructing identical views twice.

The older isolated cProfile records these functions and allocation calls, but
has incomplete call accounting and changes execution latency. Its times are
diagnostic clues, not expected savings. Attention-output pooling, metrics-kernel
fusion and compute-graph copy placement are separate candidates.

## Ownership and API contract

Add a NOSA-specific selection workspace owned by `NosaFixedResources`. Its pure
layout/validation helper belongs under `models/nosa/execution/`; operator adapters consume
explicit output tensors and do not import model configuration. Allocate once
with shared resources, before sessions execute. Sessions keep their existing
K/V/CIS, compressed records, score/preparation slab, ranking and host finite flag.
The new storage contains only IDs, validity and score normalizers and is charged
once per backend. Session release cannot free it.

Borrowing requires the existing active session lease, matching resource identity
and generation, an active written cache step, the expected layer, device, dtype,
head geometry and a query count within the immutable plan. Validate before any
new output write. Reentry, a second borrower, a different stream within the
borrow, a foreign/stale session and capacity growth fail explicitly. An enabled
fixed route may use the existing allocation path for a dispatch branch that has
not yet received output-buffer support; record those calls rather than claiming
all selection allocations have disappeared.

The borrow spans selection and submission of every attention consumer, including
offload planning, FA3 prepare/repair and cache publication. Normalizers may be
overwritten only after scoring; IDs/mask may be overwritten only after their
last attention consumer. The next layer reuses them on the same stream after
those consumers, without a per-layer device synchronization. Register borrowed
storage on that stream where the existing asynchronous lifetime contract
requires it. Cross-stream/session reuse goes through the existing completed
lease drain and allocation/session readiness events; the workspace does not
create another session scheduler.

Every success and exception path ends the logical borrow. If work was already
submitted, resource ownership and storage stay live through the existing lease
drain. A failed drain poisons the backend and preserves ownership, as
`NosaExecutionResources.lease` does today. Backend close releases the workspace
only after all sessions and execution have drained; setup failure unwinds only
this new allocation.

Public `NosaIndexer.__call__` and standalone operator calls continue returning
independent owned selections by default. Use an explicit internal borrow/output
parameter or internal entry point for the fixed model path. An opt-in caller
must know the selection expires after its current attention submission; a
diagnostic retaining it must copy at that boundary. Preserve existing
indexer/attention instrumentation seams and capture all layer selections before
reuse. Profiling copies stay in separate diagnostic execution, outside ordinary
timing. A replaced/custom callable uses the ordinary owned-result path unless
it explicitly adopts the borrow contract.

The public complete hidden output remains independent across layers, candidates,
users and backend close. Keep the existing graph-mode output clone and final
finite decision in `NosaForCausalLM.forward`. No new selection storage is placed
in opaque cache state or returned as public hidden.

## Dispatch, views and rollback

Start with the existing H64K checked resident route, whose
`_indexer_checked_cuda.select_prepared_out` already accepts all three outputs.
For all three offload schemes, thread explicit outputs through the current
`select_contiguous_blocks` / `_select_blocks` joint branch into
`_indexer_cuda.select`; its native entry already accepts those tensor arguments.
Keep the default allocating path and the exact current dispatch predicate.
Only allocation and adapter plumbing change; kernels, launch order, score
passes, normalization, model-dtype rounding, stable ties and selection policy
remain the same.

New explicit-output adapters must validate tensor type, exact active shape,
contiguity, dtype/device, inference status, alignment and byte-range disjointness
against every input and writable scratch before launching. The allocating
joint wrapper previously made this ownership implicit; its new out path cannot
rely on that assumption. Keep the existing checked native guards as well.

Preserve the resident joint threshold `count >= 2047`. Short guarded selection
continues through `_indexer_deferred_cuda`, with the original standalone score
dispatch and native empty selection on a false flag. Extend explicit outputs
to that adapter and the existing short/native-selection wrappers as a separately
checked part of the same workspace integration. Preserve the short-Q normalizer
split count. Unsupported layouts retain their current supported implementation;
an output workspace must not expand native dispatch eligibility. No cache,
indexer, attention or IO operation enters a CUDA Graph.

Obtain each workspace's active-shape views once per selection invocation and
pass them to existing guards and launches. In `NosaIndexer.__call__`, an already
exact-length K/CIS view can be used directly; slice only when the supplied view
actually covers later tokens. Keep validation at its current points and refresh
it on every call. Do not cache tensor metadata or validation results between
layers, calls or across eager attention. A broader handoff of resident records
from indexer to attention would need a fresh cache owner/step/buffer-identity
check after indexer execution and preserved custom-callable behavior; it is not
part of the first workspace candidate.

`IndexerCache.reserve_layer/finish_layer/abort_layer` and model-wide commit/
abort remain authoritative. Workspace writes carry no committed length. A
failed layer or nonfinite flag aborts the entire step, leaves history bytes and
committed indexer metadata intact, and returns no output. Reusable temporary
bytes need not be restored. Offload may write only its existing uncommitted
derived tail before the decision. Candidate success still discards its tail;
persistent history construction still commits all layers together. No retries
or conversion of failed candidates into successful measurements are added.

## Layout and capacity accounting

Let `Qmax = plan.metadata["max_query_tokens"]`, `K = kv_heads`,
`G = query_heads / K`, and `Cmax = max(0, max_session_capacity // 16 - 1)`.
Use three independent flat allocations and contiguous prefix views:

| Storage | Active shape | Payload capacity in bytes |
| --- | --- | ---: |
| IDs, int64 | `[Q,K,64]` | `Qmax * K * 64 * 8` |
| Validity, bool | `[Q,K,64]` | `Qmax * K * 64` |
| Normalizers, FP32 | `[splits,Q,K,G,2]` | `Rmax * K * G * 2 * 4` |

For supported scored calls, `splits=1` at Q>=128 and otherwise at most
`Smax=max(1,min(4,ceil(Cmax/128)))`. A conservative bound valid for every smaller
query is `Rmax=max(Qmax if Qmax>=128 else 0, min(Qmax,127)*Smax)`.
Take a flat prefix before reshaping, so smaller Q/split combinations stay
contiguous. Actual calls retain their existing split predicate and pass no
normalizer to routes that prohibit it. All buffers remain disjoint from Q/K/CIS,
score/preparation/ranking storage and each other; retain native alias checks.

For current Qmax=1024, K=2, G=16, capacities are 1,048,576 + 131,072 +
262,144 = **1,441,792 B (1.375 MiB)**. This is one backend's logical storage,
not a measured peak, per-layer multiplication, or per-user charge. Charge each
allocation separately with `allocation_bytes`; include it in
`plan_resources`, `allocate_shared`, `storage_tensors`, `shared_bytes`, cleanup
and actual allocator audits. Metadata records capacities, ownership, supported
borrow routes and policy revision. CPU-only planning must not allocate tensors
or inspect CUDA. Query/head/dtype/device/config changes require a new drained
plan, rather than resizing live storage. P/NH admission semantics do not change.

`session_budget_breakdown` currently includes selection in
`max(cis_creation, cis_source + max(selection, finite, boundary))`. For an
enabled route, recompute that liveness maximum after moving the selected
outputs to shared ownership; do not subtract `selection_hbm` from the total.
Keep offload ranking and any still-allocating fallback helper charged. If those
routes require the original bound, name it as a conservative reservation and
separate it from actual storage. Update the documented CPU dtype-probe temporary
after its removal, retaining the independent offload scalar/metric temporaries.
Continue reporting allocated, reserved and device-used separately, including
unchanged graph static/private storage and inactive allocator blocks.

## Verification and performance decision

1. Freeze a new candidate and baseline after the descriptor pilot completes.
   Keep the graph class and all other runtime policy identical in both arms.
   Verify current source, native build inputs and loaded libraries; unchanged
   kernel source does not by itself establish identical compiled artifacts.
2. Extend existing cache/indexer/fixed-resource tests for capacity formulas,
   dtype width, Q=1/127/128/1024 and configured maxima, owner/generation rejection,
   disjoint buffers, unsupported-route behavior and failed setup cleanup.
   Preserve current short/long dispatch and nonfinite semantics. Add CUDA
   checks for sequential layers, nondefault streams, failed partial submission,
   lease drain, two users, session release, backend close and retained public
   outputs. Reuse `test_native_checked_indexer.py`,
   `test_native_deferred_short_indexer.py`, `test_nosa_compute_graphs.py` and the
   fixed-serving suites. GPU skips cannot satisfy explicit CUDA acceptance.
3. Run controlled ordinary complete `extend_candidate` repetitions on the exact
   saved H64K request 0 and 16 histories, then the complete four-scheme trace.
   Keep GPU5/CPU48–55/NUMA1, precision, register8, input order, GC/allocator policy,
   compute graphs, warmups and output boundary matched. Use fresh process pairs
   in both orders and retain every sample, including scan tails. Record any
   extra diagnostic clones and their different memory/cache conditions.
4. Require exact all-token hidden and IDs/masks against independent histories,
   unchanged transfer payloads/LRU/discard, valid memory accounting and no live
   allocation growth. Confirm output-buffer use at the executed branches.
   Report complete candidate and request means, medians, ranges and pairwise
   process changes. Consistent net wall-time improvement, rather than object
   counts or a profiled subtraction, decides promotion. An inconclusive or
   regressing candidate stays outside production.
5. After promotion, publish new formal/profile/API families with new run IDs,
   full32 all-four correctness and measurement audits. Reassess HBM complete
   request and candidate MFU separately, and retain the async overall-latency,
   unique-read and every-sample overlap checks. This optimization does not
   establish compute/IO dominance or fix the current async performance gates
   merely by passing correctness.

## Affected experiments and publication

Keep workspace borrowing enabled only for the fixed-serving path initially.
The generic `dtype.itemsize` edit still changes other cached sparse executions;
review each actual timed call path before production promotion.

| Family | Required impact decision |
| --- | --- |
| `nosa_motivation` | Replace all four formal traces, matching profile and API/capture families after full acceptance; they execute the new fixed path. |
| Generic budget serving | The former `gr_serving` experiment is retired. Preserve model/cache regression coverage for the dtype-probe change; do not queue reruns of the retired scope. |
| `nosa_mfu` sparse model and operators/modules | Cached model/module timings execute `IndexerCache.workspace`; refresh affected native/Triton timings. Audit standalone API timing separately if optional-output plumbing changes its executed wrapper. |
| `nosa_indexer_pattern_65536_1024` | Audit actual sparse propagation and capture consumers for retained selections; refresh affected timed/captured paths. Preserve numerical-only assets only with explicit source/semantic evidence. |
| `nosa_offload_overlap` | Its operator timing uses frozen Q/selection and excludes indexer work. Preserve those timings if source-path/native identity audit confirms no changed timed code; rerun affected full32 checkpoint correctness separately. |
| `nosa_mfu` dense, DeepSeek families | Current dense NOSA and DeepSeek paths do not use this NOSA selection workspace; verify actual call paths before declaring them unaffected. A broad source snapshot alone does not make a path affected. |

Before accepted replacement, retain existing README run IDs and boundaries,
report assets and matching raw output. Mark affected production paths as not
yet remeasured. Replace conclusions and remove superseded affected assets in
the same publication update only after new evidence passes. Diagnostic failures
stay outside experiment deliverables. This plan itself schedules no GPU job or
publication change.

## Separate future kernel option

The joint scorer computes QK twice to avoid materializing Q-head probabilities.
The indexer follow-up（Git `934485b:docs/agents/system/nosa_candidate_efficiency_followup.md`） measured a complete
indexer at 4.79–5.06 ms and two actual-operand QK products at about 4.02 ms; the
published composition's earlier one-QK term is a different reference boundary.
The candidate/API gap therefore cannot be assigned entirely to CPU dispatch.

A future KDA task may compare the existing two-pass schedule with bounded
per-layer logits storage. It must retain causal masking, normalization order,
BF16 rounding, exact selection and full-model acceptance while measuring the
extra HBM traffic and complete API cost. That changes the kernel algorithm and
memory contract; it is excluded from this kernel-preserving dispatch candidate.
