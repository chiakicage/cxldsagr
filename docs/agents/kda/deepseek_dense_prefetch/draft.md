# C7 draft and evidence

## Baseline and observed scope

Current helper: models/deepseek_v32/pool_prefetch.py. Its uncertified path builds
_global_range, filters actual misses, protects hits, filters/ranks victims, writes
maps and priorities, then records metadata_ready on the caller stream. A private
copy stream waits for that event, runs generic gather and records ticket._ready.
wait(ticket) adds the consumer dependency and certifies full-history residency
only if map_generation still matches. drain joins even unused lookahead copies
before releasing ticket references or permitting another caller stream.

The backend order is wait(current ticket), prefetch(next layer), then compute
current layer. Current-layer cache/indexer work can overwrite all pool-wide
metadata scratch while next-layer copy is still active. A ticket must therefore
retain private host IDs and physical slot IDs; record_stream prevents allocator
reuse, not in-place overwrite of a borrowed shared buffer.

Observed C4 run motivation_c4_20261004_u16_r2_02, source helper SHA256
e9391575306b7d3c102f96b606ae3fb502700481765665ab88117a9f52bc57fa:
dense revisit mean request 40.330498 ms, candidate execute 34.310174 ms. First
visits average 2470.469757 ms and copy zero historical misses during cold build.
These are C4 observations; C5/C6/graph combined results are pending. The older C3
profile recorded ten dense revisit gather kernels totaling 14.710032 ms. That
is actual GPU activity, not an overlap fraction or a current C7 latency result.

## C7a native whole-history metadata

Keep the existing certified all-hit helper. For H <= P without a certificate:

1. Native dense classify scans logical [0,H), translates pages, protects every
   hit at t, writes full-P miss flags, counts M, and commits protection clock
   t+1. No input selection tensor or bitmap construction is needed. It must not
   increment C6 selection/resident/max counters: dense requested/resident/fetched
   counts belong to the ticket/backend; later attention owns consumer selection.
2. Read M once to preserve exact ticket counts and zero-miss proof behavior.
   If M>0, clear append_owner, wait_host, stable-sort priority[1:], and prefix-scan
   flags in existing shared scratch. Preserve limit-1 rollover fallback.
3. Allocate ticket-owned int64 host IDs[M] and physical slots[M] before map
   mutation. Compact ascending logical misses into those private tensors from
   the prefix and sorted victims, with fragmented/nonmonotonic page translation.
   Increment map_generation immediately before mutation; tombstone old-owner
   maps before assigning new-owner reservations. Commit allocation clock t+2,
   including M=0. All metadata preparation completes on the caller stream.
4. Record metadata_ready only after compaction/reservation, append the ticket
   before any copy submission, then let the private stream read only the ticket's
   own ID tensors and the stable layer host/record allocations. Retain the same
   readiness, caller-stream, record_stream, map-generation and drain contracts.

Dense reservations intentionally publish new-owner mappings before copies, as
in the current ticket contract; consumers must wait for ticket readiness. This
is distinct from synchronous C6, which publishes only after its gather. Do not
call C6's post-copy API and cite its comment as dense ordering. Use an explicit
reservation helper or clearly factored internal kernel with separate contracts.
Failure after reservation must disable helper reuse until cleanup; tombstoned
old-owner mappings cannot be restored to overwritten records. Broaden the
submission guard to cover new metadata mutations and event creation, while
retaining tickets for every potentially submitted copy. Backend drain and GPU
synchronization still precede rollback/release; failed completion keeps the
existing poisoned/owner-retention behavior.

A P-sized argsort slice would retain the entire P allocation in every ticket
until drain. Prefer compact private M-sized physical IDs instead, matching the
current helper's two owned M-sized ID tensors. Shared prefix/allocation_log views
must not escape. The raw argsort tensor stays temporary on the caller stream.
This adds no persistent storage or full KV staging. Validate actual ticket
storage bytes across all live layers and the existing pool/indexer execution
reservation, including allocated/reserved/device-used peaks in full acceptance.

Reuse C6's checked ordering proof and low-level guard/reduction utilities, but
keep C7 helpers separate so the accepted C6 route need not change. A new dense
range producer can share internal classification logic only if its metrics and
event semantics remain explicit. No code is changed at this planning stage.

## C7b bounded transport hypothesis

The current generic operators/common/csrc/kv_transfer.cu launches one 128-thread
CTA per record. A 64K miss call therefore launches 65,536 CTAs. BF16 x576 is 1152
bytes or 72 uint4 elements; only 72 of 128 threads perform one vector load/store
on aligned rows. The source uses ordinary vector/byte loads, not explicit .cv.
Host L2 behavior must be measured or controlled rather than assumed absent.

A bounded grid needs a grid-stride record loop; reducing gridDim alone would
silently drop records. Candidate caps can be selected around 1/4, 1/2, 1 and 2
CTAs per SM after resource inspection, retaining the unrestricted baseline.
A cap does not guarantee reserved SMs: CUDA block placement and simultaneous
GEMM residency depend on resources and scheduler behavior. Stream priority also
does not establish overlap. Changing vector/warp layout is a separate candidate
if cap-only copy throughput becomes insufficient.

Keep the default generic gather behavior unchanged initially and apply any new
option only to dense lookahead. Its source is shared by checked recall, C6 recall
and reference prefetch, so all generic width/alignment tests and affected cache
correctness must rerun even if default launch behavior is retained. Do not change
host-cache policy in the same cap candidate; test it separately if investigation
requires it. A faster isolated gather alone is not a serving promotion gate.

## Required evidence

C7a: complete helper prefetch/wait/drain wall latency, actual metadata launches,
GPU work, transferred records and byte counts, and allocation ownership/peak.
Compare against the checked helper with identical maps/records/snapshot. Include
cold all-hit, uncertified all-hit, partial-free, partial/full misses, H<P, H=P,
H=0 before first append, nonmonotonic pages and transient candidates.

C7b: compare unrestricted and capped transport with identical complete compute
and full history, plus a serial control. Capture actual gather and GEMM/attention
kernel intervals per layer and the request dependency chain. For a kernel whose
blocks wait internally, use internal work intervals rather than its full launch
window; mere interval intersection does not establish useful overlap. Promotion
requires lower complete candidate/request latency while preserving exact bytes
and outputs, not just a visually overlapping trace. Root's complete serving
trajectory and full-model numerical acceptance remain mandatory.

Independent CPU ticket review by graph_validate confirms private ID lifetime,
caller-only metadata ordering, no borrowed scratch escape, no selection-metric
increments, and the delayed-copy overwrite test described in the executable plan.
