# Staging integration draft

The local coarse-prefetch baseline writes directly into sorted pool slots and
logs allocations before `SparseTokenCache.finalize_prefetch`. The official Q1
kernel instead copies at most 64 records into a staging tensor and writes
temporary h2d tags `P + 1 + stage_slot`. Passing these tags to local exact recall
would be invalid. The official threshold can also select records outside exact
top-k, so the old coarse-eligibility certificate cannot certify this policy.

The selected candidate promotes every stage record before existing finalize.
It reuses the prepared free/FIFO order and allocation journal and leaves exact
recall unchanged. This adds at most 73,728 B of D2D traffic per invocation and
preserves actual H2D accounting. A shadow h2d alternative would add NH-dependent
storage and copying; the parent selected direct tags with explicit cleanup.

Main risks are leaked tags on exceptions, host-zero handling, duplicate or
out-of-range page IDs, a suffix read before host KV exists, overshooting atomic
attempt counters, and cache-graph lifetime. Restrict this entry to one pending
query, validated local page tables and 64-slot headroom. Register the callback
before launch and retain all stage storage through finalize. Separate actual
successful copies `min(attempts, 64)` from rejected attempts. Fully validate
stage/victim ownership before mutation in the promotion CTA.

Implement a small independent TVM FFI CUDA adapter with source-bound build info.
The prepare kernel expands history token IDs and initializes stage IDs/counter;
current-token and trailing padding entries contain safe zero and are excluded
by the official kernel. A single promotion CTA verifies 64 bounded metadata
entries, copies exact BF16 bytes, then publishes mappings/journal/statistics.
A separate idempotent cleanup kernel clears leftover tags only.

Validation uses the exact command in `integration_task.md`. First run synthetic
stage-transition cases independent of official score computation, then the raw
official bridge with a CPU-derived exact promotion/reference state. Record
source identities, test result and limitations in `integration_checkpoint.md`.
The parent's complete-API cold benchmark must include packing, scheduling,
staging, promotion, cleanup and finalization; no standalone kernel speedup is a
production performance acceptance result.
