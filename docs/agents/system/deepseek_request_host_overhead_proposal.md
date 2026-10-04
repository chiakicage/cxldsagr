# Request validation and cleanup overhead after C9

Status: CPU source analysis only, 2026-10-04. No production changes, new
benchmarks, test runs or CUDA work. C9 profile audits are still running;
profile values below prioritize candidates and are not formal latency claims.
The separate SFA/projection and history-hint proposals remain available.

## What the current path actually does

C4 introduced a private packed array, not a `PackedTokenInput` type. Both
current and frozen C9 `serving/persistent.py` have SHA-256
`e07b33d5192372b64042f650c308baf99a9e944d3ba7ed54abc597162fad63f7`.
Both copies of `cache/prefix_pool.py` have SHA-256
`92080f5fd8eac4081be6bd4128501bf6ba07b74945c550b338016f304664e7e0`.

The request admission sequence is:

1. `_validate` reads a list/tuple, makes an owned list snapshot, checks every
   element with exact `type(...) is int`, then scans `min(ids)` for negatives.
   It validates prefix, context and planned history/candidate bounds.
2. `_prepare_token_input` encodes the complete snapshot into `array('q')`,
   hashes the history's little-endian int64 bytes and makes a CPU tensor that
   owns the array. The unusual-ABI/int64-overflow fallback preserves the old
   prefix `struct.error` versus suffix tensor-overflow errors.
3. The complete H+A tensor is copied to the backend device before admission.
   Only afterward are reservation/quota estimates and `pool.acquire` run.
4. On a hit, only the candidate slice is used. The full history transfer has
   already happened. The prefix signature is still checked against the current
   request; request metadata hashes are never trusted as content evidence.

Relevant sites are `serving/persistent.py:27`, `:36`, `:201` and `:227`.
The profiler wraps only `_validate` as `request_validation`, so packing,
hashing, CPU tensor creation and device transfer belong to request-exclusive
work. Its metadata helper does not traverse the request's token list.

The C9 profile's `analysis/stage_costs.csv` reports one offload revisit
validation call at 2.748–2.775 ms and two pool audits totaling 1.946–2.000 ms.
Request-exclusive host duration is 3.048–3.091 ms; it includes approximately
1.25 ms of CUDA API intervals and one approximately 0.0124 ms GPU activity.
Thus the exclusive remainder is not entirely Python validation. These are
instrumented observations from
`experiments/deepseek_v32_motivation/output/data/motivation_c9_triton_profile_20261004_01`,
not a prediction of removable latency or a replacement for formal timing.

## Exact input validation options

A fresh mutable input list can differ anywhere in H, even when UID, length,
list identity and supplied digest are unchanged. Current content identity and
strict-type semantics require reading all H+A elements. Caching a successful
validation by user/list identity, validating only A, trusting workload hashes
or moving preprocessing outside the timed request would change this contract.
An exact tuple containing validated exact ints is immutable, but the current
workload supplies fresh mutable lists; tuple caching is not this optimization.

The most isolated substantial candidate is a small CPU native validator for
the existing private list snapshot. It would check exact Python-int type and
nonnegative value in one C-level pass, returning the same result as the current
type scan plus `min`. Keep the snapshot, metadata checks, `_prepare_token_input`,
digest, device transfer ordering and rare overflow fallback unchanged at first.
Positive arbitrary-precision ints must still pass this validation and reach
the existing later overflow handling; a validator must not silently impose
int64 range at a different error boundary. It must not invoke a token's
`__index__`, equality, hash or user metaclass methods. This adds a CPU build
surface, so root must choose whether its measured benefit justifies that cost.

A smaller Python/NumPy candidate could merge preparation with validation and
check negativity in the already-owned int64 buffer. NumPy is an existing
dependency. This only removes the Python `min` pass; exact-type traversal
remains. It also changes the ordering of packing relative to invalid-prefix
and overflow errors unless handled explicitly. Therefore it should not be
assumed faster or simpler than the isolated native scan. A single Python
generator combining type/sign tests likewise needs CPU timing before selection.
No proposed validator has been implemented or measured here.

## Reuse an audit only across proven no-op cleanup

`PersistentGRRunner.execute:265` audits all sessions after synchronized
candidate execution, calls `backend.truncate`, synchronizes, then audits all
sessions again. `PrefixSessionPool.audit:232` freshly checks shared storage,
every session's actual whole allocations, each reservation, host pages, HBM
tokens and global limits. With sixteen retained users and ten layers, each
audit revisits at least 640 runner tensors plus per-session and shared pool
metadata. This explains the growth relative to the one-session HBM case.

The current DeepSeek `truncate:1193` already returns without mutation when
successful transient execution has restored all lengths/indexer boundaries,
discarded every transient step and cleared pending prefetch. The predicate
also requires the requested boundary to equal both session and prefix length.
This branch was introduced in C4 and is covered by
`test_completed_transient_cleanup_keeps_history_without_truncating_again`.
`synchronize:1353` only synchronizes CUDA; it does not resize cache storage or
change page quotas.

A bounded candidate can communicate that exact no-op result to the runner
and reuse `before_cleanup` as `actual` after the existing synchronization.
Keep the first full audit, no-op predicate, completion barrier, failure cleanup
and both metric fields. The result must be tied to the actual cleanup call;
`candidate_persistence='gpu_transient'` or a stale boolean alone is insufficient.
The existing generic `truncate` protocol returns `None`; add an explicit
optional cleanup result/hook rather than treating any return truthiness as
proof. Backends without that contract and every mutating/truncating path keep
the second full audit.

This removes repeated observation across a source-proven state-preserving
operation. It does not replace measured storage with reservation estimates,
skip other users in the first audit or cache allocation sizes across requests.
The serial owner contract must exclude concurrent pool/storage mutation during
this interval. Tests that inject a resizing cleanup must take the ordinary
second-audit path and still fail on reservation overflow.

Independently, `audit` can sum actual/reserved bytes and quota counts using
local integers during its one existing entry traversal, constructing the
return `CacheFootprint` at the end. This avoids repeated `CacheFootprint`
additions and later reservation-property traversals while still reading and
validating every current measurement. Keep the same checks and failure cleanup.
This is a smaller pure-Python candidate with an unknown, likely limited payoff.

Do not initially replace full auditing with active-session-only checks or
cached tensor byte counts. Generic backends may lazily grow even inactive
storage; current tests deliberately change `untyped_storage().nbytes()` while
keeping small/empty alias views. Any future incremental ledger would need an
authoritative allocation-lifecycle contract that the current generic pool lacks.

## Candidate-only token transfer after a read-only reuse check

A separate candidate can avoid copying H to the GPU on a verified prefix hit.
It must still snapshot, strictly validate and hash the full current request.
Factor the exact `PrefixSessionPool.acquire:194` reuse predicate into a shared,
read-only check: ready entry, matching signature, sufficient retained capacity,
byte reservation and page/token quota. A prospective hit transfers only A;
a miss transfers H+A. Perform the ordinary acquire/LRU mutation only after
the required transfer succeeds, preserving the existing failure-order guarantee.

The probe must not mark ready, touch LRU order, release a session or allocate.
If its entry is invalidated before acquisition, a candidate-only copy is no
longer sufficient: finish any required full-input transfer before entering a
mutating admission path, or fail before changing the pool. Reuse one predicate
instead of duplicating slightly different hit rules. Keep this change separate
from input validation and cleanup-audit reuse for attribution.

This optimization removes an unused transfer on a hit, not history validation
or first-visit input transfer. All work remains within the same request timer.
The profile's CUDA API intervals only motivate a test; they are not the expected
savings because allocation, runtime and profiler costs may remain.

## Cache scope names

`pool_operation` is the NVTX scope around
`SharedSparseTokenPool.operation:353`, including its yielded body. It is not
one kernel or one isolated lease-check cost. Nested scopes are subtracted in
exclusive CPU attribution, but unwrapped work inside the body still belongs
there. Examples are native `resident_selection`,
`sparse_selection_classify/compact/publish/map`, cumsum/temporary allocation
and the host `counter.item()` synchronization in
`SparseTokenCache._ensure_sparse_from_topk:572`. Outer lease completion also
records stream/event state.

`cache_torch_argsort` wraps `torch.argsort` globally during the diagnostic.
Current relevant call sites are:

- `cache/sparse_token_pool.py:487`: initial cold-build append victim order.
- `cache/sparse_token_cache.py:680`: fused-prefetch victim order.
- `cache/sparse_token_cache.py:605`: exact-recall miss victim order.
- `cache/sparse_token_cache.py:451`: generic checked allocator fallback.
- `models/deepseek_v32/pool_prefetch.py:114`: dense-history miss victim order.

These are scope-to-source mappings. Exact CUDA sort kernel names and time
fractions must come from the corresponding captured node, not from the scope
label or a guessed CUB implementation.

## Verification before promotion

Reuse the current token-input, persistent-runner, prefix-pool, DeepSeek cleanup
and storage-accounting suites. Add focused cases for the selected behavior:
list mutation between requests with the same object/UID, mutation after snapshot,
bool/int subclasses/foreign metaclasses, huge positive and negative ints,
overflow position and error precedence, changed prefix/suffix, device-transfer
failure before LRU mutation, no-op versus resizing cleanup, shared growth,
page/token overrun, release failure and probe invalidation. Preserve little-
endian digest identity and private CPU/device-buffer lifetime.

CPU timing should first isolate complete snapshot+validation+packing+hash,
then use the full CPU runner with realistic H/A and one versus sixteen users
for audits. Record randomized baseline/candidate pairs after exactness.
Transfer and full-serving gains require root-scheduled GPU measurements with
the complete request boundary; do not extrapolate CPU-only or profiler time
into accepted E2E savings. Generic/NOSA callers need their existing regression
coverage because `PersistentGRRunner` and `PrefixSessionPool` are shared.
