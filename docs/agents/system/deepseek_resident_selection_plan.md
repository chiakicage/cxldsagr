# DeepSeek trusted resident selection

Status: implementation design only. Production source is frozen while the C1+C2
64K screen runs. This file does not claim new kernel or end-to-end measurements.
The parent task and completion criteria remain in
[the optimization plan](deepseek_motivation_optimization_plan.md).

## Purpose and scope

The C1 residency certificate proves that every initialized host-backed token of
the active session remains in its layer's HBM pool. `SparseTokenCache.ensure`
still sorts the full selected-ID matrix with `torch.unique`, filters tensors,
reads GPU scalars for validation, and maps the matrix through several PyTorch
operations. At the target geometry the input is up to `[1024, 2048]` int32 IDs.

Add a private path from the model's exact-top-k producer to a native all-resident
consumer. Keep public `ensure(indices)` unchanged as the checked API and as the
fallback when the proof is absent. The native path must return exactly the same
physical IDs and exact distinct-selection statistics, refresh exactly the same
history priorities, and contribute the same two FIFO events.

The path covers persistent history construction and a transient candidate whose
entire retained history is resident. It does not recall misses, allocate slots,
change selection, split candidates, add host IDs for candidates, or establish a
new residency certificate from capacity alone.

## Admission and trust boundary

The model route is private, for example `_ensure_from_topk(indices)`. Its only
production caller is `EchoAttentionRunner._consume`, after `_forward` has:

1. Constructed scores with visible length `end = position + query_count`.
2. Called `exact_topk` on precisely those visible scores. That producer returns
   contiguous int32 IDs in `[0, end)`, or negative padding.
3. Appended current KV successfully, so `cache.written == end` before consumption.

Slicing rows while recursively splitting a checked miss path preserves the
producer's range guarantee, but certified all-resident calls cannot exceed the
history pool's distinct-union capacity and need no split.

Inside `cache.operation()`, after lease acquisition can invalidate ownership,
check all of the following using host metadata:

- Shared CUDA cache with the native method available; CPU and unsupported native
  adapters call public `ensure`.
- No pending fused prefetch and an active/live session.
- `all_history_resident` remains true after acquiring the lease.
- Input is contiguous 2-D int32 on the cache device. Do not silently copy or
  narrow an arbitrary public input into the trusted route.
- Persistent mode has `written <= P`. Transient mode has
  `history = transient_start <= P`, `written - history <= candidate_slots`, and
  this session still owns the transient tail.

No public `trusted=True` switch is needed. The positive-ID bounds are guaranteed
by the identified producer, not discovered by reading the tensor. Public
`ensure` continues to reject malformed IDs before mutation. Native code must
branch on negative padding before looking up a page or map. A private positive
ID outside the producer contract is an implementation defect; it must never be
silently accepted as a smaller valid selection. A native invariant guard may
fail the CUDA operation for an out-of-range ID or nonresident physical mapping;
it must not add a scalar read or a recovery/retry path to valid execution.

## Native API and storage

Add the wrapper in `operators/deepseek_v32/indexer/cache_ops.py` and implement the
entry in the existing `csrc/echo_indexer.cu` compilation unit, optionally placing
the kernels in a local header covered by its source fingerprint. Suggested API:

```python
resident_selection(
    indices, physical, page_table, host_to_device,
    priority, clock_tensor, union_bitmap, union_count, selection_totals,
    *, written, history, transient, slots, candidate_slots, timestamp,
)
```

`indices` and `physical` are matching contiguous `[Q, K]` int32 tensors.
`selection_totals` is a contiguous three-int64 per-session/layer view with
`[selection_records, resident_selection_records, max_working_set]` contributed
by native calls only. Every tensor is caller-owned and validated before launch;
the native entry allocates no tensor or auxiliary CUDA storage. Use the current
torch stream through the existing TVM FFI stream mechanism.

Reserve one shared bitmap and count for the pool's exclusive serial workspace:

```text
D = P + 1 + candidate_slots
union_bitmap: ceil(D / 32) uint32 words
union_count:  1 uint32
additional shared bytes = 4 * ceil(D / 32) + 4
```

Bit zero corresponds to the padding sentinel and is never set. Bits `1..P`
cover history slots; later bits cover candidate rows. For `P=65536`, `A=128`,
the added shared storage is 8,216 bytes. It is shared across every session and
layer, not multiplied by their counts. `P+A < MISSING` already bounds the union
below the uint32 counter limit.

Reserve three additional int64 counters per session and layer. At ten layers
this is 240 bytes per session, or 60 KiB for 256 retained sessions. Prefer one
`[layers, 6]` counter slab holding the existing three prefetch counters followed
by the new three counters; retain compatible three-element views for official
cache adapters. Then `reset_stats()` can zero one six-element row and `metrics()`
can perform one bounded read per layer. This must be an explicit allocation
change in `estimate_session_bytes`, actual `session_bytes`, and release paths.

When implementing the planned-append companion below in the same candidate,
reserve a fourth native counter for its eviction total and use a `[layers, 7]`
slab from the start. Total extra storage then becomes 320 bytes per ten-layer
session, or 80 KiB for 256 sessions. The first three entries remain the compatible
prefetch view; the next three hold selection statistics; the final entry holds
native planned-append evictions.

Include shared scratch in `estimate_shared_bytes` and `shared_bytes`. CPU
reference estimates classify the same allocated tensors as DRAM. No new
query-sized persistent buffer is needed: the returned physical-ID tensor replaces
the result already produced by public `ensure`; retain the model's existing
query-dependent execution-workspace reservation and verify its coverage.

## Two-kernel algorithm

Call `_clock_event(layer_id)` first, preserving existing rare timestamp rollover
handling. Read its Python integer `timestamp = layer.clock`, which is strictly
newer than all live priorities. Pass that scalar by value to every CTA.

Kernel 1 resets every word of `union_bitmap` and sets `union_count[0] = 0`.
The following kernel launch on the same stream is the grid-wide ordering boundary.
No CTA in the mapping kernel may reset the bitmap itself.

Kernel 2 processes the flat selected-ID matrix. A bounded grid of 256-thread
CTAs with a grid-stride loop is a straightforward first candidate; choose and
measure its CTA cap rather than assuming the first launch geometry is optimal.

```text
local_unique = 0
for each assigned selected ID:
    if logical < 0:
        physical_output = -1
        continue

    if transient and logical >= history:
        physical = P + 1 + logical - history
    else:
        global = page_table[logical // 64] * 64 + logical % 64
        physical = host_to_device[global]  # certified in [1, P]

    write physical_output
    word = physical // 32
    mask = uint32(1) << (physical % 32)
    first = (atomicOr(union_bitmap[word], mask) & mask) == 0
    if first:
        local_unique += 1
        if physical <= P:
            atomicExch(priority[physical], timestamp)

block_unique = block_reduce_sum(local_unique)
if block leader:
    old = atomicAdd(union_count, block_unique)
    atomicAdd(selection_totals[0], block_unique)
    atomicAdd(selection_totals[1], block_unique)
    atomicMax(selection_totals[2], old + block_unique)

if global thread zero:
    priority[0] = MISSING
    clock_tensor[0] = timestamp + 2
```

Use unsigned 64-bit atomics on the nonnegative int64 counter storage and the
existing compatible cast for priority exchange. All bitmap shifts use uint32;
the high bit must not depend on signed left-shift behavior. Block reduction must
include every thread, including padded lanes and lanes with no loop iterations.
For an empty matrix, launch at least one mapping CTA so the two FIFO events still
advance; all-negative inputs similarly produce zero union without skipping them.

There is no need for a third finalization kernel. Atomic adds to `union_count`
produce cumulative nondecreasing totals. The last nonzero block contribution
observes exactly the final distinct-union size; atomically taking the maximum of
all block endpoints therefore updates the exact per-invocation maximum. Prior
calls' maxima remain intact. The two cumulative totals are sums of disjoint
first-set contributions, so both equal the exact distinct count for all-resident
input. History and candidate physical domains are disjoint and injective within
this session, hence physical deduplication equals logical deduplication.

Writing `clock_tensor` early in one CTA is safe because every CTA uses the
by-value `timestamp`; none reads a clock scalar that another CTA changes. Only
selected history slots receive that timestamp. Candidate rows have no priority,
free bitmap, host ID, or page-table entry. The subsequent logical empty allocation
event advances the clock again without changing a priority, matching public
`ensure` exactly. Set `layer.clock = timestamp + 2` on the host after successful
submission. Do not call `protect()` or `stamp()` again.

The kernel modifies no ownership map or allocator state. Preserve C1's append
plan and residency certificate: all priority refreshes belong to this session's
already-consumed slots. No host-read dependency is required. The operation lease
must cover the following MLA consumer, as it does now.

## Counters, transactions, and lifecycle

`metrics()` returns, without mutating cumulative state:

```text
selection_records = Python fallback count + native selection count
resident_selection_records = Python fallback count + native resident count
max_working_set = max(Python fallback maximum, native maximum)
```

All other existing counters retain their current meanings. A mixture of public
fallback calls and private native calls must sum exactly once, including repeated
`metrics()` reads. Direct Python `CacheStats` fields contain fallback contributions
only for these three fields; audit existing readers and use `metrics()` wherever
a total is required. Current serving aggregation already reads `metrics()`.

- `reset_stats()` zeros both sets after the current execution is ordered, as in
  the existing serving lifecycle. Do not reset per-session counters from the
  per-invocation bitmap clear kernel.
- Ordinary rollback and transient discard retain attempted-work statistics just
  as the existing counters do; neither promotes candidate records to history.
  Shared scratch is overwritten by the next invocation.
- Snapshot restoration already resets statistics and invalidates the residency
  certificate. Clear the new per-session counters too; scratch needs no snapshot
  because every call clears it. A restored session initially uses public ensure
  until a current residency proof exists.
- Session release must drop every view of its counter slab so released Python
  handles do not retain storage. Pool close releases shared bitmap/count.
- Timestamp rollover still uses `_clock_event` before dispatch and its existing
  synchronization/rank normalization. The bitmap has no timestamp or epoch, so
  rollover, statistics reset, and host-page reuse cannot make old bits current.
- On a failed launch or failed consumer, follow the existing drain/rollback/poison
  behavior. Do not expose or retry a partially mapped output.
- The global pool lease serializes bitmap reuse across layers, sessions, and
  caller streams. Same-stream FIFO or the retained cross-stream event dependency
  must order the entire previous mapping kernel and MLA use before reuse.

## Implementation sequence and validation gates

1. Finish and inspect the currently running C1+C2 screen before production edits.
   Record its source/run identity; do not re-label its results as native-selection
   evidence later.
2. Add allocation/accounting and counter lifecycle changes. Check actual shared
   and session storage equals estimates on CPU and CUDA, including all retained
   sessions, candidate capacities, close, and released handles.
3. Add the native helper and an independent CUDA differential against public
   ensure using identical cloned pool/session state. Check returned physical IDs,
   selected records, all maps, priority/free arrays, exact counters, and host/device
   clock values. Use both int32 high-bit bitmap words and adversarial duplicates.
4. Route only the model's private exact-top-k path; retain checked API tests and
   unsupported/unknown-proof fallback tests. Confirm formal execution performs
   no tensor scalar reads in the successful private path.
5. Run complete ten-copy checkpoint comparisons from independent empty caches,
   then a fresh complete 64K+128 multiuser four-scheme screening run. Verify every
   candidate hidden/logit output, exact traffic/counter equations, transaction
   boundaries, source identity, and allocated/reserved/device-used budgets.
6. Profile the promoted source to verify removal of unique/filter/scalar-sync
   work and to measure total first-request and candidate time. A fast synthetic
   helper cannot establish the user objective. Retain the required full 16x2
   publication, independent output audit, and updated matrix/E2E MFU comparison.

Differential cases must include:

- Empty and all-negative input; repeated padding values; one token and one slot;
  IDs crossing physical bitmap bits 31/32 and 63/64; repeated IDs across CTAs.
- `[1024, 2048]` selections with exact union near 64K; repeated sink/local patterns;
  duplicate-heavy selections where most entries map to one physical word.
- Fragmented host pages and permuted/noncontiguous physical slots, including a
  later user's full cold build over the previous user's occupied pool.
- Transient history plus candidate selections, candidates repeated across queries,
  an empty retained history, and candidate capacity greater than P where the
  normal backend contract allows it. Candidate bytes and host maps stay unchanged.
- Mixed native/check-before-recall fallback/native calls; repeated metrics reads;
  reset statistics; truncate, rollback, transient discard, snapshot/restore, host
  page reuse, owner interleave, and forced priority rollover near `PRIORITY_LIMIT`.
- Delayed nondefault-stream consumers and alternating streams/layers so shared
  bitmap reuse cannot race. Public malformed-ID calls still reject before mutation.

## Companion: native planned append

The accepted C1+C2 screen still misses the user's latency target. Its C1 append
plan avoids repeated sorting, but each append still creates logical IDs, performs
map gathers and boolean filtering in `_evict`, reads the dynamic eviction count,
copies records, and launches several map/free/priority writes. Fuse these steps
only when C1's exact append plan is valid. Keep arbitrary/fallback allocation and
transient candidate append on their existing paths initially.

Suggested native entry:

```python
planned_append(
    source, records, page_table, append_order,
    host_to_device, device_to_host, priority, free, clock_tensor,
    native_evictions,
    *, start, count, timestamp,
)
```

The call receives the full preallocated append-order array and scalar `start`;
it needs no GPU arange, chosen-slot tensor conversion, or logical-ID tensor.
Each logical row `i` has a unique planned physical slot:

```text
logical = start + i
global = page_table[logical // 64] * 64 + logical % 64
slot = append_order[start + i]
old = device_to_host[slot]
if old != MISSING:
    host_to_device[old] = MISSING
    contribute one exact native eviction
copy every byte of source[i] to records[slot]
host_to_device[global] = slot
device_to_host[slot] = global
free[slot] = false
priority[slot] = timestamp
```

One thread also restores sentinel priority and writes `clock_tensor=timestamp+1`.
Pass the timestamp by value and update the Python clock after submission. Reduce
eviction contributions per CTA and atomically add them to the fourth native
per-session counter. `metrics().evicted_records` becomes the sum of Python
fallback evictions, existing fused-prefetch evictions, and these native evictions;
repeated metrics reads must not fold or double-count any component.

The intermediate `MISSING`, priority `-1`, and free-true states currently written
by `_evict` are unobservable inside the exclusive operation and need not be
materialized. The final state and eviction total must match exactly. New logical
IDs are unwritten, planned slots are unique, and no selected old ID can be another
new ID in the same valid append. Consequently no thread may erase a newly
published mapping belonging to another row. Verify these preconditions against
the C1 state-machine oracle rather than replacing the allocator policy.

The record copy is opaque byte movement: width and dtype come from source and
destination tensors. Do not bake 576 BF16 values into the shared cache contract.
A warp-per-record first implementation with aligned vector movement and exact
tail handling can fuse metadata without copying or reimplementing numerical
model operators. If the transport helper is factored out, genuinely generic byte
movement belongs under `operators/common`; model-specific metadata dispatch can
stay in the DeepSeek cache adapter.

Keep the existing source reservation, bounded writeback retention, and
`write_host(session, layer, start, source)` after the native append. D2H still
copies original source values, remains timed, and follows its existing event
dependencies. The append plan is consumed and the residency proof extended only
after successful submission. Existing failure/rollback handling remains in force.
Written-record counts stay on the host because `count` is already known.

No extra persistent storage is needed beyond the one eviction counter described
above. The plan's int64 slot array is already reserved by C1. Tests must compare
all metadata, records, eviction totals, clocks, and host copies against the old
append path, including occupied pools, fragmented pages, unaligned record widths,
zero versus nonzero actual eviction, rollback, and source-lifetime ordering.

## Companion: native all-history protection for dense prefetch

For a proven all-resident dense ticket, C1 still maps `arange(history)` and invokes
separate protect and empty-stamp helpers. Replace that certified branch with one
native range-protection kernel using scalar `history`:

```python
protect_resident_history(
    page_table, host_to_device, priority, clock_tensor,
    *, history, timestamp,
)
```

Each logical history ID occurs once, maps through its actual page table and live
HBM map, and receives priority `timestamp`. One thread restores the sentinel and
writes `clock_tensor=timestamp+2`, representing protection and empty allocation.
Every CTA uses the scalar timestamp. Launch at least one CTA for history zero so
the clock events are retained. There is no bitmap, output mapping, dynamic count,
or GPU scalar read in this helper.

The caller must recheck the C1 certificate inside the operation lease. The ticket
still reports exact host-known values `requested=resident=history`, `fetched=0`;
it keeps ownership/caller-stream/generation validation and requires no copy event.
Do not call generic protect/stamp afterward and do not invalidate the append plan.
Unknown-residency tickets retain the current scan, allocation, copy, and ready
event path. Tests compare exact priorities and both clocks to the old full-range
protect plus empty stamp, including noncontiguous maps, history zero, rollover,
interleaved owners, and append-plan continuation after ticket consumption.

Implement and measure the three helpers as one source candidate only after the
parent releases the active profile freeze. The accepted C1+C2 run remains a
separate source identity and cannot validate these native implementations.

The first candidate should use the simple two-kernel bitmap implementation.
Warp aggregation or epoch-tag designs are subsequent measured candidates if
atomic contention or the clear launch is still material. They are not required
to establish this implementation's correctness and are not assumed faster.
