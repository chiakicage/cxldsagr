# C6 sparse recall lifecycle review

2026-10-04. Independent CPU-only source review by the graph validation agent.
No production code, CUDA compilation, GPU execution or performance measurement
was performed by this reviewer. Root's 20,000-state ordering audit was not
repeated. This review covers the current orchestration and bindings as well as
the initial plan; it does not replace the implementation agent's GPU acceptance.

## Findings corrected during review

1. **Tombstone before record overwrite.** The initial draft deferred victim
   eviction until publication after gather. If gather completed but a later
   host/publication-launch error occurred, old-owner maps could still name
   overwritten records. Releasing only the failed user's session would not
   repair other sessions. Current compaction clears old h2d/d2h mappings,
   priority and free-state before gather; publication installs new maps only
   after the same-stream copy. This preserves safe unmapped holes on partial
   failure, matching the checked path.
2. **Commit each FIFO event before subsequent fallible work.** Classification
   now writes device clock t+1 and Python increments immediately after that
   launch. Publication, including M=0, writes t+2 and Python commits it before
   remapping. The final map kernel is read-only. Deferring either clock update
   until full success could leave occupied priority equal to the next event's
   timestamp after a recoverable error, invalidating the victim-order proof.
3. **Count successful recall separately from classified misses.** The first
   implementation derived recall/transfer bytes from selected minus resident.
   Classification precedes copy, so a failed copy or publication could report
   transferred records that never became usable. Current code increments the
   existing Python recall counter from the already-read miss count only after
   publication succeeds. No new persistent counter is needed. Selection/hit
   counters still update at classification and evictions at tombstoning.
4. **Invalidate map proofs at the mutation boundary.** The first implementation
   incremented map_generation before waiting for host writes or sorting victims.
   Current code clears append_owner once M>0 is known, then waits and computes
   the victim/prefix buffers, and invalidates residency immediately before
   compaction. Recoverable waiting/allocation/sort failures do not invent a map
   mutation. Successful recall invalidates once; M=0 preserves all proof fields.

The implementation agent incorporated all four changes before this review
closed. No additional source-level lifecycle blocker was found in the inspected
snapshot. This is conditional on the runtime checks below.

## Consistency checks

- Eligibility uses `host_written_end <= P`. In transient execution this is H,
  while written is H+A. Thus H+A may exceed P safely: candidate physical rows
  start at P+1, never enter host translation, victim ranking or host writes,
  and fit the explicit candidate capacity. The logical bitmap's existing
  P+1+A storage covers the selected logical domain.
- All-hit calls without a residency certificate, empty tensors and negative
  padding retain both FIFO events. A selected subset cannot create a new
  all-history certificate. The limit-1 rollover case falls back before
  classification, preserving normalization between checked events.
- Flags and prefix buffers retain full P capacity even when H<P. Their
  compaction rank follows ascending logical position; host page translation
  occurs afterward. Fragmented/nonmonotonic page IDs therefore do not reorder
  misses. Chosen physical IDs exclude the sentinel and candidate tail.
- Existing free_slots, miss_scratch, allocation_log, counter and union scratch
  are reused under the enclosing exclusive operation lease. No new persistent
  allocation was added. argsort/output tensors remain execution temporaries
  covered by the existing reservation; acceptance must still inspect allocated,
  reserved and device-used memory rather than infer a physical peak from sizes.
- `wait_host` precedes every actual host gather, including pending writes for
  a persistent suffix. Native metadata and generic gather both use the caller's
  current Torch stream. Outer operation completion records the event that
  protects shared records/scratch from another layer, owner or stream.
- The new header is covered by the existing `echo_*.cu*` source fingerprint.

## Required acceptance boundaries

Run full-state differentials against checked ensure, including history H<P,
H=P with A>0, mixed hit/miss, all-hit without a certificate, empty/padding-only
IDs, fragmented host pages, partial-free victim sets, equal-priority ties and
clock limit-2/limit-1/limit. Check both clocks, both maps, bytes, priorities,
free flags, proof fields, counters and output IDs. Keep root's ordering audit
as supporting evidence rather than another execution test.

Inject recoverable errors after classification, victim selection, gather and
publication. Verify the exact completed event count, mutation/proof boundary,
other-session maps and records, and absence of phantom recall counts. Then
exercise a subsequent valid cache operation to expose stale clock/proof state.
Failed GPU dependencies or completion events must retain ownership and prevent
unsafe reuse according to the existing pool/backend failure contract.

GPU tests must include same-stream and cross-stream reuse, delayed host writes,
pending-prefetch rejection, candidate-tail ownership and release/rollback.
Successful source review is not permission to publish unmeasured C6 latency.

## Reviewed source identities

| File | SHA-256 |
|---|---|
| `cache/sparse_token_cache.py` | `011e6990e9a496b1f6af5ea1ae0afc1d466731ca370978d98daa6f8218d5abfb` |
| `operators/deepseek_v32/indexer/cache_ops.py` | `8df0c741ecc343537729f5862790b9f3ac1520e9da792aef5ce6e2b22bad0224` |
| `operators/deepseek_v32/indexer/csrc/echo_sparse_recall.cuh` | `ddad6791b1205aa931689805018ff86a285601788c620fe4fc592846e74fd39e` |

The implementation agent was still connecting exports and preparing tests at
review time. Later source identities and GPU evidence belong in its checkpoint.
