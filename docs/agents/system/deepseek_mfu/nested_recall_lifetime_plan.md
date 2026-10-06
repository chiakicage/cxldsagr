# Nested full-recall lifetime: conditional private plan

Source-only proposal. Implement nothing unless the frozen standalone candidate
passes state/fault acceptance and the fixed paired screen shows a convincing
complete-call improvement. Candidate01, both state harnesses and production stay
unchanged. This proposal covers the owned diagnostic model; guarded serving
sessions require a separate lifecycle contract.

## Why the depth guard cannot simply be removed

`models/deepseek_v32/attention.py::_forward` holds `cache.operation()` across
projection, indexer, append, recall, MLA and attention output. The inner
`_ensure_from_topk` therefore exits its own lease before the real outer lease
exits. Current `retain_failed(frame)` already durably retains a synchronous
native failure, even while the outer depth remains positive. It does not retain
a *successful* inner frame if later MLA/output or outer stream tracking fails.
Those successful frames must also remain owned through outer exit.

The production pool decrements depth before recording its final stream; that
recording can fail with depth zero and a stale `_active`. Depth zero alone is
not proof that a new private wrapper has completed its own exit bookkeeping.
The CPU progress ledger remains a host-return record, never a completion fence.

## Smallest private extension

Bind a private `operation` method, alongside the existing private
`_ensure_from_topk` binding, to the exact production cache instances used by the
diagnostic model. Delegate to the saved, unmodified production
`SparseTokenCache.operation` method. This observes the true outer boundary
without editing attention arithmetic, providers or the shared pool class.
Keep the bound method an ordinary context factory: construct the original
operation context once, return it directly for nested calls, and add exactly
one ownership wrapper for the true outer call. Do not stack an extra generator
context on every reserve/append/ensure operation. Frame registration is a
bounded direct token lookup and owner append, with no per-stage callbacks or
additional tensor exports. Include the outer wrapper, token registration and
release in all timing; a small standalone win does not provide a free overhead
budget. In particular, measure the changed cold path and certified controls
again rather than subtracting the new control cost.

1. Before entering a new outer operation, allocate an ownership token and place
   it in a private module-level registry keyed by the exact pool. A strong
   registry is necessary; a pool/frame reference cycle can be collected after
   a traceback is dropped. Nested calls join the same token only when pool,
   session, layer and current lease identity match. Owner registration failure
   occurs before any new recall submission.
2. Register each full-recall frame in the token **before `prepare`** can allocate
   temporaries or submit host-write waits. The token retains frame owners,
   transported tensors, cache/session/pool, and the original exceptions. Keep
   all existing type/provider/metadata checks. A nested eligible call additionally
   requires a live token from this declared private operation binding; arbitrary
   positive depth does not enable the route.
3. Preserve the existing inner native call, ledger application and immediate
   `retain_failed` behavior. Never drain, dispose, clear frame owners or remove
   the token at inner exit. A failed invocation poisons the pool immediately, so
   subsequent checked operations cannot reuse it.
4. The private outer wrapper retains the token while the original pool context
   exits. After an entirely successful body and original exit, release registry
   ownership using established same-stream tensor lifetimes, without adding a
   CUDA synchronize or event. A stream change *between* leases retains the
   original pool dependency handling. The first model trial requires one
   caller stream throughout each lease; broader in-lease stream switching needs
   its own lifetime and ordering proof.
5. If an entered recall, later attention work, or original outer cleanup fails,
   mark the token terminal and leave its existing strong registry entry in
   place. Retain **every participating frame**, including earlier successful
   calls. Keeping the already-installed entry avoids requiring a fresh owner
   allocation on the failure path. Mark outer-exit bookkeeping complete only
   after exception aggregation. An exception caught inside the body must not
   turn a poisoned token into a successful outer return.

Holding frames extends scratch lifetime. The initial H <= P model trial has one
eligible full recall per attention lease. Before supporting multiple such calls
in one lease, bound their retained chosen/workspace bytes and reserve that total;
the existing single-invocation scratch bound is not automatically sufficient.
Report actual allocated/reserved/used peaks in the eventual model trial.

## Exception and terminal-release contract

Propagate the original native/FFI exception object when it is the only failure.
If the original pool exit already produced a group containing it, preserve that
group and its objects. Add distinct outer/body/bookkeeping failures without
replacing the primary or dropping any cleanup exception. If body code swallowed
a recorded native failure, outer exit re-raises that recorded failure. No retry,
fallback after failure, unpoisoning or normal return is allowed.

Do not call `terminal_dispose` from either operation wrapper. Extend that private
helper to collect terminal lease tokens and all their frames in addition to
`FAILED_INVOCATIONS`. Require both pool depth zero and completed private outer
exit, then retain the full owner graph before device drain. Remove both token
and frame registries only after successful drain and irreversible CPU teardown.
A failed attempt retains them and cannot be repeated. Do not relax the current
exact-type/provider and no-release-guard restrictions for serving sessions.

The ordinary model error path synchronizes and attempts rollback. Once the
private full route has poisoned its pool, rollback can fail through `_check`;
those additional exceptions must remain grouped with the original. The private
trial's outer driver may perform terminal cleanup only after `model.forward`
and all its cleanup frames have unwound. It must mark/discard the failed model;
successful terminal cleanup does not make that model reusable.

## Gates before any model timing

- CPU lifetime probes: nested depth, native failure before/after each stage,
  success followed by a body failure, success followed by outer stream-tracking
  failure, primary plus cleanup failures, swallowed inner failure, and registry
  retention after traceback/ordinary references are dropped. Disposal while an
  outer frame is active must fail; no failure permits a second submission.
- Actual-kernel nested state/fault checks with both allocators, pending D2H,
  stream changes between leases, exact ledger semantics and post-terminal
  owner evidence. Reuse the saved-state oracle; do not infer GPU progress from
  depth, ledger or host return. Build/GPU execution remains root-owned.
- Repeat the independent fixed screen including the new registration/exit
  costs. Only then run a fresh complete model correctness check and paired
  trial. Selection, recall and IO remain eager outside compute-only graphs.
  Full extend and every layer must separately meet gap < 10%, with gap equal
  to the window minus union(compute, actual IO), and compute stages labeled.

This document establishes no native, GPU, performance or model acceptance.
