# R1 executable plan

Parent: current C10 implementation, unchanged native resident selection from
C9. [Draft and contract](r1_draft.md); [independent review](r1_independent_review.md).
Status: deferred at the user's 2026-10-04 request to close on C10. Temporary
CPU proof, offline SM90 compilation and low-level GPU correctness completed
under `/tmp/deepseek_resident_r1_20261004/`; lifecycle and timing gates did not
run. No production edits or performance promotion. The remaining steps below
are a future-work plan, not ongoing closure work.

## Implementation

1. Save the exact current cache entry, native wrapper and resident CUDA source
   hashes under `/tmp/deepseek_resident_r1_20261004/`. Copy only a bounded
   validation wrapper into the temporary module; structural comparison must
   confirm unchanged argument validation and argument order. Add an explicit
   capture rejection to the new private API. Leave the existing native API
   available and unmodified.
2. Clear the entire existing bitmap and count in a first Triton kernel. In a
   second kernel map every selected occurrence, with always-enabled traps for
   invalid positive IDs, pages and nonresident mappings. Negative padding maps
   to -1. Use 64-bit address/index arithmetic where the existing API does.
3. For history, perform naturally aligned `ld.acquire.gpu.global.u64` of the
   priority word. If it differs from the fresh timestamp, use a GPU-scope atomic
   64-bit exchange. Only an exchange returning an older value wins; only that
   winner atomically ORs the original bitmap bit and contributes a count.
   Candidate rows retain ordinary bitmap election and never access priority
   beyond P. Restore priority[0] and publish clock=t+2 exactly once.
4. Per-program reductions update count, both accumulated totals and maximum
   working set using the original nonnegative endpoint equation. No consumer
   reads the final union inside this kernel. No new scratch or persistent storage,
   host scalar read, hidden synchronization or production torch.compile.

## CPU preparation and correctness

Implement a CPU event model that interleaves acquire loads, exchanges and
winner publication for duplicate slots, including deliberately stale older
loads. Verify exactly one winner per selected history slot, candidate bitmap
deduplication, final priorities and full bitmap/counters. Inspect scoped PTX
for every compiled SM90 specialization and record register/shared/local usage;
offline compilation is not GPU acceptance.

Use the original native API and checked cache as independent GPU oracles.
The low-level fixtures must include Q=0, all negative IDs, repeated hot slots,
uniform and clustered selection, permuted physical IDs, fragmented pages,
small/nonpower-of-two P, P65536/Q1024/K2048, transient Q128 with history65536,
larger candidate ranges where valid, and nonzero accumulated totals/max.
Compare every output, original scratch word, priority, clock, count and total;
unchanged maps/inputs and candidate guards must remain intact. Invalid positive
IDs and broken maps run in isolated subprocesses after a successful warm call.
Repeat fresh calls, alternate streams, force clock rollover through the actual
private cache entry, and verify explicit fresh-API capture rejection plus the
unchanged original API behavior. Do not use repeated fixed timestamp calls as
valid R1 inputs.

## GPU scheduling and performance

Root must grant correctness and timing windows separately. Use both compiler
paths before imports and an explicit `CUDA_VISIBLE_DEVICES`; PyTorch's H200 and
nvidia-smi's M403 names are both recorded. No other GPU process or CPU timing
benchmark may run during performance sampling.

Only after exactness, compare complete validated APIs with identical initial
state restored outside timing. Use ten warmups, forty alternating paired
samples, separate wall/event intervals, and retain all raw samples. Include
small/tail/adversarial controls as well as all three real captured selections.
Record actual loaded Triton PTX/CUBIN/module identity and unchanged native
oracle identity after measured loops. A proposed dispatch restriction must
follow measured shape evidence and preserve the full fallback.

Promotion requires a cold-shape API improvement without material revisit
regression, reviewed implementation/proof, then current-source private-cache
and complete-model gates. Any accepted change needs fresh formal and matching
profile run IDs before public replacement. C10 formal numbers and C9 API
numbers remain attached to their own implementations.
