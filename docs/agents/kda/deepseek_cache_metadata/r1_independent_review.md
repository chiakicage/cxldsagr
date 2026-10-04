# Independent R1 design review

The timestamp election is sound for the private serial eager cache path if
the implementation preserves its existing guards and event ordering. This is
a source/design review only. No candidate code, GPU correctness or performance
has been accepted.

## Timestamp invariant

Immediately before a permitted R1 invocation, every live nonsentinel history
slot must have `priority < passed_timestamp`. The current cache path preserves
this invariant:

| Event | Priority assignment | Next host clock |
| --- | --- | --- |
| Initialization | Free slots `-1` | `0` |
| `stamp`, native protect, planned append, prefetch finalize | Current `t` | `t+1` |
| Existing all-resident selection / dense all-hit protect | Current `t` | `t+2` |
| Sparse or dense classify plus publish | Protected `t`, newly filled `t+1` | `t+2` |
| Fused host fetch | No priority assignment; finalize performs the event | Advanced by finalize |
| Release | Removed slots become `-1` | Unchanged |
| Rollover | All live distinct groups ranked `0..u-1` | `u` |
| Valid snapshot restore | Restores priorities and both saved clocks together | Saved host clock |

The two-event fast paths decline at `PRIORITY_LIMIT-1`. Rollover drains prior
work before ranking. The private entry rejects pending fused prefetch, uses
the host all-resident proof and holds the existing exclusive operation lease.
Cross-stream operation ordering is established by the pool's events. These
conditions matter; range validation on an arbitrary timestamp does not prove
freshness, and H <= P alone does not prove residency or freshness.

Relevant sources are `cache/sparse_token_pool.py`,
`cache/sparse_token_cache.py`, `models/deepseek_v32/pool_prefetch.py` and the
resident, sparse-recall, dense-prefetch and finalize/protect native helpers.
The fused fetch helper changes maps/allocation logs and copies records, but
does not assign slot priority itself.

## Election and memory ordering

Given initial priority less than `t`, exactly one atomic exchange of a slot
to `t` returns an old value less than `t`; subsequent exchanges return `t`.
A coherently scoped atomic-memory load may skip an exchange when it sees `t`.
All concurrent accesses to these history-priority words must use compatible
GPU scope, atomic width and semantics. An ordinary unsynchronized load or
non-atomic store racing an exchange is not covered by this proof.

The unique winner still atomically ORs the original physical bitmap bit and
contributes once to count/totals/max. Different slots sharing a bitmap word
still need atomic OR. Candidate slots remain disjoint from history priorities
and use the existing bitmap election. Physical output is written for every
valid input occurrence, including duplicate selections; negative padding and
the sentinel remain excluded from the union.

Seeing priority `t` does not prove that the winner has already written bitmap
or counters. This is harmless only because no thread consumes their completed
values during the mapper. Completion of the mapping kernel orders the final
state before later consumers. All CTAs must use the passed immutable timestamp;
early publication of the device clock by one CTA cannot become an input to
another CTA's election.

This reasoning was checked against NVIDIA's current [PTX ISA memory model](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model)
on 2026-10-04, particularly [morally strong operations](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#morally-strong-operations),
[atomicity](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#atomicity-axiom)
and [sequential consistency per location](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#sc-per-loc-axiom).
For naturally aligned full 64-bit priority words,
`ld.acquire.gpu.global.u64` and `atom.relaxed.gpu.global.exch.b64` use the same
generic proxy, overlap completely and include all device CTAs in their scopes.
They satisfy the per-location atomicity/coherence requirements. A stale older
load only causes an extra exchange; the exchange still distinguishes the
single winner. Relaxed exchange suffices for election because the algorithm
does not consume another location after observing priority `t`. Narrowing
either operation to CTA scope, using `ld.global.nc` or a weak ordinary load,
or mixing access widths is outside this proof.

## Required graph boundary correction

`SparseTokenPool._check` explicitly rejects CUDA Graph capture. Its host
`_clock_event` does not run on graph replay. A direct low-level captured call
with scalar timestamp `t` can therefore replay against priorities already
equal to `t`, incorrectly skipping history-union contributions. The current
bitmap-based API does not need timestamp freshness for union election, so it
must remain available for such unknown/nonfresh callers.

R1 graph testing must state its setup. Restoring a complete compatible
priority/clock state before each replay can test compiled execution and input
ownership, but does not establish autonomous fresh-timestamp graph execution.
Alternatively the candidate dispatch must retain the original API during
capture. A dynamically generated per-replay epoch would require a separate
design and is outside this proof. The existing full-model compute graphs
capture projection and finish islands; they do not make cache selection itself
graph-captured.

No additional design blocker was found for the stated private eager path.
Generated PTX/CUBIN inspection and current-source GPU correctness remain
required to establish that an implementation actually follows this proof.

The parent task has accepted the graph-boundary correction: the proposed
integration uses a separate private fresh-timestamp entrypoint that rejects
capture and leaves the original low-level `resident_selection` available for
direct or captured callers. This resolves the graph design concern without
claiming new cache Graph support.
