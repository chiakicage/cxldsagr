# Cleanup audit V2 integration plan

Status: deferred at the user's requested C10 closure, 2026-10-04. This is an
archived plan, not an active integration task. C11 cleanup is not promoted;
the worktree core files now match accepted frozen C10. Archived C11 source and
removed test/fixture hashes are recorded in
`/tmp/deepseek_cleanup_production_20261004/deferred_source/core_manifest.json`.

2026-10-04. Root selected the reviewed V2 prototype after its corrected CPU
tests and two separate complete-segment measurements. Production integration,
CPU validation and actual integrated-code timing are complete; evidence is in
[the production checkpoint](deepseek_cleanup_audit_production_checkpoint.md).
GPU/formal validation remains root-owned.
V1 performance is evidence of the opportunity, not V2 acceptance.
V1 source/evidence remain under `/tmp/deepseek_cleanup_audit_20261004/`.
V2 is isolated under `/tmp/deepseek_cleanup_audit_v2_20261004/`.
Prototype evidence is in
[the V2 checkpoint](deepseek_cleanup_audit_v2_checkpoint.md).

## Fixed contract

The first audit remains the unchanged full `PrefixSessionPool.audit` and
therefore traverses all sessions, shared allocation and quotas. Receipt reuse
is limited to the existing exact DeepSeek successful-transient no-op and the
current serial admission owner. Every mutating, absent, unknown, stale or
uncertified path performs audit two. Cleanup/synchronization exceptions retain
the existing discard/release behavior; failed release retains ownership.
Native token validation, input preparation, output selection and all request
metric fields remain unchanged.

V1 trusted every outer hook that returned the right typed receipt. V2 closes
that gap at the caller: an explicit canonical unbound function must match the
current backend hook before and after the call. Both guards require exact
`types.MethodType`, matching bound self and matching function identity. No
`__wrapped__` traversal or copied callable metadata is accepted. The backend
also checks its outer hook and every inner method against stable canonical
references. The explicitly supplied canonical function is trusted code; this
is not a sandbox against arbitrary mutation of Python functions or modules.

## Production changes after selection

1. `executor/serving_backend.py`: add frozen, identity-equality
   `CleanupAuditTicket(owner, session, prefix)` and
   `CleanupUnchanged(ticket)` types, plus a small exact bound-method matcher.
   Keep the existing mandatory backend protocol unchanged. This avoids a
   model-to-serving import and avoids a registry of backends or wrappers.
2. `models/deepseek_v32/serving_backend.py`: preserve the exact truncate
   predicate and mutation path. Add only a private keyword-only ticket
   argument; the existing no-op return produces a receipt when that argument
   is supplied, and still returns `None` for ordinary calls. Add
   `cleanup_after_candidate(session, prefix, *, ticket)`: validate exact ticket
   type, owner/session/prefix, and canonical method identities; call truncate;
   execute the original synchronization; recheck identities and ownership;
   return the receipt only on success. Stable references cover `truncate`,
   `synchronize`, `_check_session`, `_execution_active`, and the outer hook.
3. `serving/persistent.py`: add an optional constructor argument
   `cleanup_audit_hook=None`, accepting an explicit unbound Python function.
   Validate its type before ownership/resource allocation. `None` keeps the
   default two-audit helper. Extract only the existing four cleanup statements
   into `_cleanup`; bind the receipt-aware helper once when explicitly
   selected. Keep one `execute` body and replace only that four-statement
   segment with the helper call. No AST rewriting enters production.
4. The selected helper captures the configured canonical function, performs
   audit one, resolves the current backend hook, and evaluates its exact
   method identity before calling it. It resolves/checks the current hook
   again afterward. Reuse requires both identity checks plus the exact
   receipt type and fresh ticket identity. Unknown hooks run normally but
   always require audit two, including a hook that restores the canonical
   method before returning or fabricates a current typed receipt. An absent
   hook uses the original truncate/synchronize/audit sequence.
5. Add constructor-level `cleanup_audit_identity` metadata describing the
   requested policy, canonical module/qualified name and contract version.
   This describes configured selection, not whether a particular request
   skipped audit two. Preserve existing request metrics and native-validation
   identity. No extra per-request counter is needed for implementation
   selection; tests count the actual pool audit calls externally.

The default helper extraction adds a method call. V1's extraction probe found
only small/noisy effects; integrated-code validation must verify its actual
CPU overhead against the frozen baseline. Default NOSA retains the ordinary
two-audit behavior and does not supply a canonical hook.

## Experiment and profiler wiring

- Motivation measurement: extend the existing selected runner factory in
  `experiments/deepseek_v32_motivation/src/measure.py` with
  `cleanup_audit_hook=DeepSeekServingBackend.cleanup_after_candidate` and
  record the runner's cleanup identity beside token-validation identity.
- Official comparison: use
  `OfficialDeepSeekServingBackend.cleanup_after_candidate` in
  `experiments/deepseek_v32_echo_official/src/measure.py`. It inherits the
  canonical base function, so recognized behavior is shared without a model
  import in the generic runner. Wire both regular and resident-repeat metadata.
- Profiling: all selected runner constructions in
  `experiments/deepseek_v32_motivation/src/profile.py`, including warmup and
  graph-setup paths, receive the same canonical hook. Record its identity.
  For the selected policy, `InstrumentServing` wraps `runner._cleanup` as the
  complete cleanup scope and leaves backend truncate/synchronize/outer-hook
  methods untouched. Keep existing actual `PrefixSessionPool.audit` scopes,
  so one/two audit calls remain visible. Default instrumentation stays as it
  is. No trusted-wrapper exception or registry is introduced.
- Existing source snapshots already cover `executor`, `models`, `serving`
  and the relevant experiment sources. Verify the modified files are included
  and retain loaded source identity through the standard frozen-run workflow.
- Mark affected experiment implementations as changed but not yet run, and
  preserve accepted reports/artifacts until root publishes validated
  replacements. Do not relabel old timing as this implementation's result.

## Validation and gates

1. Final V2 focused parity includes 48 CPU tests: original output/metric and
   ownership coverage; unknown outer forwarded/fabricated receipts;
   canonical restoration/replacement; copied `__self__`/`__func__` metadata;
   shared/inactive-session storage growth; and transparent runner-level
   profiling. The final MethodType revision passed along with 393 unchanged
   CPU regression tests (one optional NOSA checkpoint skip, five CUDA tests
   deselected). Its independent callable-spoof reproduction also passes.
2. Independent review must accept the final guard implementation and rerun
   its two outer-hook/storage-spoof reproductions. Resolve any blocker before
   production edits. V2 CPU timing waits for root's explicit scheduling grant.
3. After integration, rerun focused/default/native/selected runner parity,
   NOSA ownership/resource-drain checks, and actual profiler wrapper/audit-count
   checks. Check public truncate still returns `None` and default constructor
   selection cannot activate reuse. Verify bad option types fail before owner
   binding. Compare every original request metric and full output.
4. Measure the final complete CPU cleanup segment, including new guards and
   helper binding; keep certified/fallback/default cases and all samples.
   Report storage capacity versus process RSS and the mocked-barrier boundary.
   Do not subtract these values from existing formal measurements.
5. Root schedules appropriate actual GPU numerical, lifetime and profiler
   parity validation, then new formal measurement/report identities. Preserve
   current valid artifacts until those replacements are accepted. The fixed
   P/NH and complete-request semantics remain unchanged.
