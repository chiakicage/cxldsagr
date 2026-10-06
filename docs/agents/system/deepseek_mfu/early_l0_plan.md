# Private early L0 scheduling plan

Status: root authorized the private scheduling implementation after rejecting
the prebound validator01 screen. Production V9 remains the baseline. This is an
implementation plan, not a performance claim; native builds and GPU work remain
subject to root's separate review and scheduling.

## Decision and scope

The existing projection graph depends on validated hidden input, position,
weights, precision policy and graph storage. It does not read hint contents or
history KV. It can therefore execute before hint snapshots and cache-step
publication, provided all deterministic invalid-state checks run first and all
cache consumption waits until preparation succeeds. Moving independent GPU or
allocation work changes when failures may surface; that is an explicit schedule
change, not a reason to require universal old OOM/error ordering.

The first private candidate covers only the real checkpoint layers 0–2,
single GPU, prepared v4 compute graphs, ordinary persistent extend, H=65,536,
A=128 and one extend chunk. Methods are HBM, ECHO and serial sparse, both
declared cold and warm states. The model owns the usual cache/session and is in
one graph execution lease. Residual is absent for L0. Only its first projection
is submitted early. Prefill, dense prefetch, multi-device placement, transient
candidates, C10 serving, unprepared/missing graph pairs and other shapes keep
their existing dispatch. Dense's initial DMA wait is not changed.

Eligibility is decided before candidate mutation. It is a configuration choice,
not recovery from a failed early launch. A failed check/launch/preparation
propagates and cleans up once; there is no retry or switch to the original path.
No direct-token embedding capture, recapture, bound-MLA, new kernel, clock change,
selection change or validation-success cache belongs in this candidate.

## Concrete code work in an isolated tree

1. Snapshot the baseline chosen after the recall screen, including model,
   compute graph, attention, sparse cache/pool sources and relevant tests. Record
   exact hashes and isolate imports as in the accepted private screens. Read the
   NOSA/shared-module rules before editing the private sparse cache helper;
   this helper remains shared-source work even though the scheduling user is
   DeepSeek. No production file is edited during the prototype.
2. Extract the current ordinary `SparseTokenCache.begin_step` checks into a
   private read-only helper. Keep their exact conditions and messages, including
   session/pool health, active transaction, transient owner and capacity. The
   existing `begin_step` calls that helper and then performs its three original
   state assignments. The early model path calls the same helper on all three
   layers before projection, then still calls normal `begin_step` after replay.
   This deliberately repeats guards; there is no unchecked commit shortcut or
   externally supplied proof. Guard cost remains inside measured forward.
3. Factor the existing graph projection closure into an internal submission
   operation shared by normal and early paths. Keep all current graph-bank,
   stream, attention identity, hidden, position and residual checks; copy the
   same graph inputs and replay the exact same graph. Return graph-owned
   projected tensors without allocating the offload-owned KV clone. Keep the
   clone in the normal attention projection callback, after the original cache
   operation and `reserve_append_source`. The normal path retains its original
   order and clone behavior.
4. Within the existing `_forward` try/cleanup boundary, use this order only for
   the supported early path: token and query checks; all-layer step preflight;
   eager first-chunk embedding; pure L0 graph submission; required hint
   snapshots; normal `begin_step` for every layer; the original layer loop.
   Do not repeat embedding in that loop. The L0 callback consumes the pending
   projection; L1/L2 and finish use the original graph path. Position is taken
   from committed length and checked against every layer's committed length;
   stale `written` is never used as the pre-begin position.
5. Use one local pending-projection record, owned by this forward and the active
   bank. Bind bank execution identity, pair, layer, query count, position,
   hidden/residual identity and cache generation. The callback requires the same
   source and position and consumes it once. No intervening replay may reuse
   the L0 pair or its shared position inputs. This local record owns no new GPU
   storage; hold strong references to IDs, embedding output and graph outputs.
   Reject a mismatched, already-consumed or stale record before cache writes.
6. Keep the original operation lease, cross-stream handoff, source reservation,
   owned-KV clone, index writes, recall, finish, final synchronization and commit.
   Hint snapshots still precede their first mutation. All `begin_step` calls
   still precede every cache/indexer write. No graph or model-wide source hash
   calculation is added to the request path.

The duplicate preflight costs real time and can erase the benefit, especially
for HBM and serial where hint snapshots are absent. That is a screen result to
measure, not a justification for deleting capture/health checks. ECHO has three
actual hint snapshots to overlap; after early graph submission their device
copies remain ordered on the same stream and are not claimed to overlap GPU
projection. A source-only count is not sufficient to rank the methods.

## Failure and ownership plan

Initialize `started`, hint snapshots and pending references before early work.
The full existing model `try` must encompass eager embedding, projection input
copies/replay, hint snapshots and begin calls. A failure after the first GPU
submission uses model-wide device drain even when `started` is empty. Successful
drain permits ordinary rollback of only the layers whose begin returned and
restoration of only snapshots that were captured. Keep all original and cleanup
exception objects, including graph-lease completion failure.

If drain fails, retain the pending record, IDs, embedding output and graph
storage on the model/bank owner, poison the model and disable reuse. The local
exception traceback is not the sole ownership mechanism. A later explicit
`synchronize`/`close` may release those references only after confirmed drain;
it never resumes the request. If rollback or hint restoration fails, retain the
current original-plus-cleanup exception behavior and poisoned state. No new
host-write source is allocated before the original source reservation.

Malformed tokens, invalid shapes/positions, mismatched graph identity, poisoned
or closed resources, stale sessions, active cache steps and insufficient
capacity are checked before early resource use. If state changes between
preflight and normal begin, the repeated normal guard fails and drains the
already-submitted computation. Independent hint allocation or CUDA failures can
now occur after graph submission; record this changed timing explicitly. Error
handling still crashes, drains and retains unsafe owners as required.

## Verification and decision sequence

First complete CPU checks in the isolated tree. Verify supported/unsupported
dispatch, exact ordinary begin behavior after helper extraction, and event
ordering using existing facade patterns. Inject failure at projection, first
and later hint snapshots, each begin, owned-KV clone and pending consumption.
Include zero-started-cache failure, duplicate/stale consume, drain failure,
rollback failure, hint-restore failure and graph completion failure. Assert
original exception identity, grouped cleanup errors, no commit on failure,
correct retained references and no CUDA initialization. Freeze all sources,
tests, loader and plan before root's GPU decision.

If authorized, the first GPU check uses the unchanged prepared graph and exact
baseline weights/input. Compare complete outputs and exact selections, cache
maps/clocks/counters and traffic for all three methods in cold/warm states.
Exercise cross-stream handoff, replay then preparation failure and terminal
cleanup. Check actual storage allocated/reserved and retained source limits.
Run relevant shared/cache tests because of the guard-helper refactor. A failed
validation stops the screen; do not retry or relax comparisons.

Only after checks pass, run a short independent balanced AB/BA screen over the
same full synchronous forward boundary. Preserve every sample, baseline and
candidate order, observer record and actual loaded source identity. Both arms
use equivalent private loaders and the same warmed graph bank; do not compare
a new candidate import path with a differently prepared control. Report the
preflight overhead and actual schedule, without subtracting observation or
validation cost. Reject if no repeatable useful gain or any path regresses
materially. Do not repeatedly tune sample subsets.

A useful screen permits a separate root decision about full independent
check/bench/profile and publication. It does not pass the full/L0/L1/L2 gap gates
or replace formal MFU. Re-run complete prefill controls and the unchanged dense
control when producing a new full-model result. Account for all affected
shared-source experiment identities before promotion or report replacement.

## Evidence

Accepted V9 read-only bounds are in
`/tmp/deepseek_early_l0_schedule_review_20261006_01/schedule_bounds.json`, SHA
`c222646e9db7d1ce7b93c65f247e2430a2bd3f062f9efd5be60647ea5b58bc95`.
The token-H2D-API-return to embedding-launch region has uncovered gaps of
57.399/109.451/87.608 us for HBM/ECHO/serial. It encloses more than the work that
can move. The first graph API itself still contributes 82.822/90.694/95.670 us
of exposed gap. These are single intrusive-profile observations and are not
predicted savings. The implementation plan supersedes the initial memo's
unnecessary requirement to preserve universal independent-failure precedence.
