# C7a dense-ticket lifecycle review

Date: 2026-10-04. Reviewer: `graph_validate`.

This is an independent CPU source review of the C7a dense preparation path.
It does not claim CUDA compilation, numerical acceptance, measured speedup or
full serving validation. Those checks belong to the implementation owner's
separate execution records. No production code was changed by the reviewer.

## Finding

No unresolved source-level blocker was found in the reviewed ticket lifetime,
shared-scratch reuse, clock sequencing, mapping updates or failure cleanup.
The review requested direct metadata-ready/copy-ready Event construction and
recording failure coverage; the implementation owner added all four cases.
Their source was reviewed, but the reviewer did not run their GPU process.

## Shared scratch and ticket ownership

The serving loop waits for the current layer's ticket, prepares the next
layer's prefetch, then computes the current layer. Consequently the next
layer's copy can read its ID arrays while the current layer overwrites pool
metadata scratch. `record_stream` alone would protect allocation lifetime,
not the values of a shared scratch tensor.

The native path avoids that hazard. It allocates separate int64 miss and
destination arrays, each with M elements and no storage offset. The binding
rejects aliasing with the prefix, sort output or other ticket array, and
rejects oversized borrowed backing storage. The P-element sort output and
shared prefix remain confined to caller-stream metadata work; the copy stream
reads only the ticket's owned IDs plus the layer's host/device records.

Both ID arrays are retained in `_tickets` before copy submission. The
metadata-ready event orders the private copy stream after ID production and
map reservation. `wait(ticket)` orders that layer's consumers after its copy
event. `drain()` joins even an unconsumed speculative copy before invalidating
tickets or releasing their ID references. A failed synchronization preserves
the tickets and marks the helper failed.

The reviewed delayed-copy test overwrites allocation_log, miss_scratch,
free_slots, union_bitmap and the counter while the copy stream is deliberately
delayed. It compares native/checked mappings, records, FIFO metadata and
metrics for two layers, fragmented host pages, partial/full misses and multiple
record dtypes/widths. The test asserts the delay is still active after the
caller finishes overwriting scratch.

## Mapping and event semantics

Classification walks exactly initialized host history H, bounded by P. It
protects resident mappings at timestamp t, writes miss flags for all P logical
positions including the zero-filled tail, and advances the GPU clock to t+1.
The Python clock advances after successful submission of that event.

Compaction follows logical history order through the page table. Stable
priority sorting selects distinct physical victims below timestamp t, so
protected history cannot be evicted. Miss global IDs cannot equal any old
victim ID because they were unmapped during classification; per-rank map
updates therefore do not collide. The new reservation replaces both map
directions, updates free/priority metadata and advances the second event to
t+2. Candidate-tail rows and host IDs outside initialized history are excluded.

This dense path intentionally reserves maps before asynchronous copying.
Its ticket-readiness contract differs from synchronous sparse recall's
post-copy publication. The new native header and binding document that
difference. Callers must still drain the helper before rollback, truncate or
session/storage release; pool metadata serialization alone does not join the
helper's separate copy stream.

All-hit and empty calls retain two FIFO events. The native shortcut falls
back at the existing normalization boundary. Map-generation invalidation
occurs after host wait, sort, prefix sum and private-ID allocations succeed,
immediately before map reservation; generation-guarded readiness certification
remains unchanged.

Dense preparation does not add consumer selection/resident/max-working-set
counts. It records only native evictions plus the existing dense request and
successful recalled-byte metrics. Successful copy submission increments the
Python recalled count; a pre-submission failure cannot claim completed recall.

## Failure boundaries

The outer prefetch failure guard now covers metadata preparation, metadata
event creation/recording, copy submission and completion event creation/
recording. It permanently disables helper reuse after any such failure.
The backend's failure path drains the helper and synchronizes before rolling
back or releasing sessions. This is cleanup, not automatic request recovery.

The new direct Event tests distinguish metadata-ready failures, where no copy
or ticket has been submitted, from copy-ready failures, where ID arrays must
remain retained until drain even though the completion event is unavailable.
They preserve real pool lease-completion events, verify reuse rejection and
check another user's records after cleanup. Existing injected post-native and
post-copy failures cover their separate submission boundaries.

## Reviewed source identities

| File | SHA-256 |
|---|---|
| `models/deepseek_v32/pool_prefetch.py` | `f27260feaa6b21d302232e29325a461a31d158d6ef8671f7dfa164c2c3a62a7b` |
| `operators/deepseek_v32/indexer/cache_ops.py` | `9283837189aeb5b8a0e55e2e6e6a25e24e08a1bb2f545edf2bf11fec2b93175c` |
| `operators/deepseek_v32/indexer/csrc/echo_dense_prefetch.cuh` | `30374a97873ecd7b7d87730ba40c400e111d0e3ba32feee3982efc8c508bb59d` |
| `operators/deepseek_v32/indexer/csrc/echo_indexer.cu` | `86793e7375d8204a1fe445f42ecb4efd35d7b6083fa82ee2bf4e3e0a12da3eab` |

The include/export wiring reaches the new header through the existing indexer
extension. Its build fingerprint scans the component's csrc files, including
this local header. These identities cover this review scope, not every
transitive dependency or a performance-run snapshot.
