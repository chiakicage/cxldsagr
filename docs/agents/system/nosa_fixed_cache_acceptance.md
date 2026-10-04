# NOSA fixed-history cache implementation and acceptance

Validated on 2026-10-04. This is engineering evidence, not a performance report.

## Contract

`models/nosa/fixed_serving.py` exposes `NosaFixedServingBackend`, preserving the
ordinary budget-mode `NosaServingBackend`. The new backend accepts fixed
`sparse_pool_tokens=P` and `host_arena_tokens=NH`. P and NH, histories, and history
chunks are multiples of 64; a complete history must satisfy H <= P.

HBM-only admits retained history through an independent P-token session quota.
The other schemes allocate one per-layer `[P + Amax, KV heads, D]` K and V pool
and admit host histories through a global NH-token quota. Host storage is lazy,
session-owned pinned storage with exactly H logical tokens per record. NH is
not a claim that all host storage has been allocated or that filling NH fits
the 512 GiB planning budget. For the 32-layer checkpoint, filling NH=16,777,216
already requires 512 GiB of logical main K/V payload before allocator or other
DRAM overhead.

Every `(layer, logical page, KV head)` slot has a session identity tag. Conflicts
replace only the requested logical page. This direct-mapped cache policy is
explicitly different from token LRU. Session identities are never reused during
the resource owner's lifetime. Selection, CIS, causal masking, and exact
attention are unchanged.

Dense prefetch copies next-layer misses into that layer's P slots on a copy
stream, then publishes the tags. Sparse synchronous and asynchronous execution
use the same native sparse-union fetch with hit pages omitted. Candidate suffix
writes invalidate any tags they overwrite, including historical pages belonging
to a longer session. Candidate pages never receive a retained identity.

Candidates run once, do not write main K/V to host, and discard pending KV/CIS/
indexer cursors. Retained history is unchanged. Offload main candidate capacity
belongs to the backend pool. The current implementation still reserves CIS and
derived candidate tails per session; HBM-only has a per-session resident tail.
Those capacities are reported and counted rather than described as shared.

## Ownership and failure handling

All cache writes and candidate transactions require the existing exclusive
backend execution lease. Lease exit, abort, release, and shared-allocation drain
synchronize the CUDA device, which includes the additional dense copy stream.
The existing poison/owner-retention behavior applies when completion cannot be
established. Releasing one session does not release backend pool storage.

The fixed constructor uses the existing native allocation model and reserves
all pool tensors, page tags, attention/fetch scratch, trace capacity, and the
reusable dense-transfer counter. Session reservations additionally cover
session-owned histories, CIS, compressed records, pending appends, and helpers.
These storage/allocation bounds do not replace allocated/reserved/device-used
measurement of process memory.

## Executed checks

- CPU: 15 fixed-mode tests pass, including all four schemes against resident
  output, candidate-length changes without rebuilding history, session LRU
  quota eviction, direct-page conflicts, shorter-history suffix invalidation,
  failed-candidate release, and shared pool lifetime.
- GPU: seven native integration tests pass on SM90 with two random layers,
  H=8192, A=128, and chunk=1024. They cover independent histories, users
  `0,1,0,0`, miss then hit payload counters, unchanged host histories, zero
  candidate D2H, different H/A conflicts, and failure after an unconsumed delayed
  next-layer prefetch. These random-layer tests are not checkpoint acceptance.
- Full checkpoint: all 32 actual NOSA-8B layers pass H=65536, A=128, chunk=1024,
  P=65536, NH=16777216. Each of the four schemes runs users `0,1,0,0`; every
  candidate hidden element from every offload scheme is bitwise equal to the
  corresponding independent-empty-cache HBM result. Candidate D2H is zero;
  the offload revisit after the other user fetches history, and its immediate
  repetition transfers zero main K/V. Retained KV/indexer length remains H.
  This test completed in 39.78 seconds including setup; that duration is not a
  serving latency measurement.

Reproduce the checkpoint check from the repository root:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=2 \
  CXLDSAGR_SM90_BACKEND=native \
  NOSA_FIXED_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  .venv/bin/python -m pytest \
  models/nosa/tests/test_fixed_serving_cuda.py::test_cuda_fixed_full_checkpoint_independent_histories_and_all_candidate_hidden \
  -x -q -s
```

The native different-history tests ran after the dense suffix invalidation
fix. Full-checkpoint validation above uses equal H and therefore does not
exercise that edge. Performance, CPU launch overhead, MFU, and per-sample
internal overlap remain separate experiment gates owned by the main task.

## Complete-checkpoint graph acceptance

The same test subsequently passed with `NOSA_FIXED_COMPUTE_GRAPHS=1` after the
combined Q/K/CIS finite-check helper, device-only wrapper validation changes,
incremental allocator guard, and deferred finite decisions before commit.
It first obtains independent eager HBM results for users `0,1,0,0`, then creates
fresh backends and histories for all four graph-enabled schemes. All candidate
hidden elements are bitwise equal to the eager results, verified both with
zero numeric tolerance and by comparing their uint8 views. The transfer-counter
and retained-length checks above also pass. Capture is enabled for query sizes
128 and 1024 before runner planning; model weights are loaded outside inference
mode so graph invalidation can inspect their version counters.

This GPU 2 check passed in 93.52 seconds including setup and waiting for a shared
native build, with one upstream FlashInfer deprecation warning. That duration
is not a serving measurement.
The runtime source inventory covering models, operators, layers, cache,
executor, and serving was identical before and after execution, with SHA256
`ec453e213f755712a774204b859bf6fdfc1d8f56ca3343ef84885e748f33e9d4`.
The engineering evidence at
`/tmp/nosa_full_graph_preserved_dispatch_acceptance_20261004/evidence.json`
records that inventory,
native and fused build manifests, and hashes of the nine NOSA shared libraries
actually mapped by the test process. The test does not claim cryptographic
checkpoint shard identity, performance acceptance, or general arbitrary-H pool
support. Add `NOSA_FIXED_COMPUTE_GRAPHS=1` to the command above to reproduce the
graph comparison.

The deferred resident path preserves the previous `compressed_count >= 2047`
joint-indexer dispatch condition. Shorter prefixes keep their existing scoring
and synchronous preparation paths. Extending the joint native path to shorter
1024-query prefixes changed rounding relative to the original Triton score
path; an independent checkpoint check detected the first changed selection at
user 1, query start 20480, layer 5. That dispatch expansion was removed before
the successful final check above. It is not an accepted implementation or
performance baseline.

The final deferred-validation failure suite passes 25 GPU checks in 18.87
seconds on GPU 2. Twenty-four cases cover all four schemes, both layers of a
small random model, and Q/K/CIS corruption using NaN, positive infinity, and
negative infinity, respectively. Each case fails a candidate after H=65536 and
a prefix chunk starting at 32768, so resident checks enter the retained native
joint path. Before the runner releases the failed session, the tests verify
that committed K/V/CIS and derived indexer bytes, lengths, and layer metadata
are unchanged; pending states/reservations are empty and the validation binding
is cleared. No output is returned, the session is released, and a fresh session
produces the original exact output. A separate case checks that rejecting a
model call does not clear another validation owner's binding. These tests
exercise failure handling; they do not replace the real 32-layer checkpoint
comparison or establish serving performance.

Reproduce these failure checks with:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=2 \
  CXLDSAGR_SM90_BACKEND=native \
  .venv/bin/python -m pytest models/nosa/tests/test_nosa_compute_graphs.py \
  -k 'nonfinite_graph_prefix or rejected_model_call' -x -q
```
