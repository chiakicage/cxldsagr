# Temporary no-op cleanup audit reuse candidate

Status: deferred at the user's requested C10 closure, 2026-10-04. C11 cleanup
is not promoted. The three core files are restored byte-for-byte to accepted
frozen C10, and the new cleanup fixture/tests are archived and removed. The
archive and hashes are under
`/tmp/deepseek_cleanup_production_20261004/deferred_source/core_manifest.json`.
The following is the retained candidate plan, not an active execution task.

Historical status: temporary CPU candidate completed. Focused parity,
unchanged CPU regression and two root-authorized complete-segment timing runs
are recorded in [the checkpoint](deepseek_cleanup_audit_candidate_checkpoint.md).
The timing window has ended. Production files and existing experiment results
remain unchanged until root selects and validates an integration.

## Source-proven opportunity

`PersistentGRRunner.execute` synchronizes candidate execution, performs a full
`PrefixSessionPool.audit`, calls `backend.truncate`, synchronizes again, then
performs the same full audit. Both before/after metric fields are populated
from those observations. The first audit checks shared allocation, every
session's actual whole storage allocation and reservation, actual/reserved
host pages and HBM tokens, and global configured limits.

`DeepSeekServingBackend.truncate` has an existing successful-transient return
that performs no mutation. It first validates session owner/release/scheme,
inactive execution and an in-range boundary. Its return predicate then checks
the successful-transient flag, equality of requested/session/prefix lengths,
and every layer's committed/written/indexer boundary, inactive step,
inactive transient state and absent pending prefetch. The successful forward
has already synchronized, discarded all transient steps, restored offsets,
and drained the execution lease. The normal DeepSeek `synchronize` performs
only the CUDA completion barrier; it does not modify measured storage/quotas.

The flag alone is insufficient. A shorter prefix, ordinary append, unfinished
layer, pending prefetch, custom cleanup or unknown synchronization operation
must retain the second full audit. Reuse is valid only under the existing
serial admission-owner contract, with no concurrent mutation of the pool or
backend resources. It is not a concurrent serving protocol.

## Temporary contract

The prototype lives under `/tmp/deepseek_cleanup_audit_20261004/` and extracts
current methods into isolated functions. AST guards limit modifications to
the explicit receipt path; no import-time monkeypatch affects production.

- A fresh typed `CleanupAuditTicket` binds the runner/owner, session and prefix
  to one call after the first successful full audit. Ticket equality is by
  identity. A typed `CleanupUnchanged` holds that exact ticket.
- An optional `cleanup_after_candidate(session, prefix, *, ticket)` extension
  executes the actual truncate and existing synchronization. It exposes the
  receipt only after both complete successfully. The existing no-op branch is
  the only branch allowed to issue it; no mode/capacity flag substitutes for
  the predicate.
- The temporary DeepSeek adapter declines certification when the called
  truncate or synchronization implementation is unknown/overridden. It still
  calls that operation normally and preserves its exceptions. A normal
  mutating truncate returns no certificate. The owner/session/prefix binding
  must agree with the actual call.
- The selected runner path accepts only the exact receipt type whose ticket
  is the fresh local object. Only then may `actual = before_cleanup` replace
  the second full audit. Unknown, truthy, stale or mismatched returns take the
  ordinary audit. Every failed operation still reaches the existing session
  discard/release path; a failed release preserves ownership as before.
- Base `PersistentGRRunner.execute`, normal DeepSeek callers and NOSA remain
  untouched in the prototype. A possible later production integration needs
  an explicit selected path so default execution stays unchanged.

The temporary implementation may add a private ticket argument to its copied
truncate method so the receipt is issued directly at the original no-op
return. This is an implementation probe, not a commitment to a public API.
Do not optimize the arithmetic inside the first audit or cache measurements
across requests; those are separate candidates.

## Profiler parity

Current `InstrumentServing` wraps backend `truncate` and `synchronize` on the
instance. Treating those unknown wrappers as certified would weaken the proof;
refusing them without changing instrumentation would make formal execution
reuse one audit while profiling takes two.

For a later selected integration, the profile should instead wrap the actual
`cleanup_after_candidate` hook as one cleanup scope and leave its called
truncate/synchronize methods unmodified. The selected runner must resolve that
hook at call time, so the wrapper observes the real operation. Default profile
instrumentation remains unchanged. A transparent outer-hook wrapper must
preserve the receipt and audit count in prototype tests; unknown inner
truncate/synchronize overrides must retain audit two. No profiler production
edit is authorized at this stage.

## Exactness and measurement gates

1. Check the copied default/no-op predicates and request wrapper against the
   current source AST. Record source identity before and after every run.
2. Compare complete runner outputs and all metric fields using deterministic
   clock fixtures. Cover first visit, revisit, candidate change and eviction.
   Count every first-audit session/shared measurement and all quota checks.
3. Exercise stale receipts, wrong types/owner/session/prefix, mutating cleanup,
   unknown synchronization, shared/inactive-session storage growth, real
   storage resizing behind small/empty aliases, host-page/HBM-token overrun,
   cleanup/synchronization/release failures and serial owner retention.
   Generic/NOSA and all noncertified paths retain their ordinary audit behavior.
4. Reuse relevant existing persistent, transient, prefix-pool, DeepSeek cleanup
   and storage-accounting CPU tests. Keep the native-token/default validation
   selection unchanged for this candidate.
5. When root grants a CPU timing window, build one- and sixteen-session
   fixtures with ten layers and real CPU tensor storages. Use the actual
   DeepSeek `session_bytes` / `_storage_bytes` measurement and unchanged pool
   audit, rather than synthetic constant byte counters. Record allocation
   geometry, aliasing and logical storage capacity separately from process
   physical DRAM usage. No model inference or pinned/CUDA storage is implied.
6. Time the complete audit/cleanup/barrier/audit segment in randomized paired
   blocks, retaining all samples and reporting generic-path overhead. Include
   the receipt checks and mandatory first full audit. No component result may
   be reported as full serving latency or subtracted from a formal run.

Production promotion and any affected GPU/serving reruns remain root-owned.
