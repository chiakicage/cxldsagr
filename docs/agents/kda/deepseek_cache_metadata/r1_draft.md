# R1: use the fresh resident priority timestamp for union election

Status: deferred on 2026-10-04 when the user selected C10 for closure. The
temporary implementation passed CPU proof, offline compilation and low-level
GPU correctness. Lifecycle and timing gates remain unrun. Production keeps
the accepted native resident selector. The [implementation plan](r1_implementation_plan.md)
records the remaining work; this candidate is outside C10.

## Evidence and hypothesis

The accepted C9 captures contain 640 history resident-selection kernels per
offload cold request, totaling 61.079--64.355 ms. C10 did not change that kernel,
but these are C9 measurements. The current large-input path clears a shared
bitmap in every CTA, atomically accumulates physical bits locally, merges those
bits globally and refreshes the selected history priorities. Exact physical
mapping, union statistics and both FIFO events remain required.

For the private all-resident cache path, `_clock_event` supplies a timestamp
strictly newer than every live slot priority. Ordinary events advance the host
clock after stamping; rollover ranks the old distinct priorities into
`0..unique_count-1` and sets the next clock to `unique_count`. The branch at
`PRIORITY_LIMIT-1` already falls back. This freshness invariant must be reviewed
across every native/fused event before implementing R1; scalar range validation
in the low-level helper alone does not prove freshness.

If freshness holds, the first atomic exchange of a selected history slot to
the current timestamp is also a unique-selection election. A GPU-scope acquire
load may avoid subsequent atomic exchanges once the slot already has this
timestamp. The winning exchange returns an older value. Only that winner sets
the original union-bitmap bit and contributes to the existing counters.
Other selected occurrences still write their exact physical output. Candidate
tail rows have no priority allocation and retain ordinary atomic bitmap election.

The proposed Triton mapper keeps the original clear/count-reset launch, exact
bitmap contents, count/totals/max equations, sentinel value and device/host clock
advancement. It introduces no persistent storage. It avoids per-CTA bitmaps and
repeated shared-memory atomic work, but adds global priority reads and may lose
on small or contended inputs. No speedup is presumed.

## Concurrency and failure requirements

The acquire load must be an actual scoped PTX atomic-memory operation, not an
ordinary non-atomic load racing priority writes. Exchanges use appropriate GPU
scope/semantics. Kernel completion orders the winning bitmap and counter updates
before later consumers; acquiring a priority value is not a promise that those
other updates have completed. No thread consumes their final values inside the
mapping kernel. A CTA may publish clock only because every lane uses the passed
timestamp, not the mutable clock tensor.

Keep guarded negative padding, written-length bounds, fragmented page lookup,
live slot bounds and transient-tail mapping. The sentinel is never part of the
union. Empty/all-negative input still resets scratch, restores sentinel and
advances the two events. No CPU scalar read, hidden synchronization, cache
policy or transaction change belongs to this candidate.

Unknown/nonfresh timestamp callers must retain the original API. Do not silently
broaden this proof to an arbitrary direct metadata invocation. A fresh timestamp
does not follow merely from H<=P or from an allocated cache.

## Candidate gates

1. Independently review the freshness and scoped-load/atomic proof. Reject or
   revise the design if any production path can reuse the timestamp.
2. Implement only under `/tmp/deepseek_resident_r1_20261004/`. Compare complete
   final state with the unchanged native selector and checked cache reference;
   include exact scratch bitmap/count, cumulative totals/max, priorities,
   physical IDs, clocks, input preservation, candidate guards and invalid-ID
   failure behavior. Use all three captured real selection batches as well as
   adversarial duplicates, empty input and varied pool/page geometry.
3. Inspect actual SM90 PTX/CUBIN and run Hopper correctness before timing.
   Check repeated calls with fresh timestamps, forced rollover through the
   actual cache path and nondefault streams. The pool already rejects CUDA
   Graph capture. Keep the existing low-level resident API available for its
   original capture behavior; a new private fresh-timestamp API must reject
   capture explicitly. Replaying a fixed timestamp over the already-stamped
   priorities is invalid for R1 and must not silently return zero union counts.
4. In an exclusive root-assigned window, measure complete APIs using identical
   restored state outside timing, alternating orders, declared warmups/repeats
   and source/runtime hashes. Require a cold-target gain without material
   revisit regression; otherwise reject or narrowly dispatch by proven shape.
5. Any selected production integration receives combined cache/model correctness
   and new formal/profile evidence. Component results cannot replace the full
   latency/MFU goal or update the existing public reports.

## Initial independent source review

Initialization uses priority -1 and clock zero. Stamp, protect, planned append
and finalization stamp the current time then advance it. Sparse/dense two-event
paths stamp t or t+1 and leave the host clock at t+2. Fused copy does not stamp
priority itself; finalization does. Rollover and snapshots preserve freshness
at the private eager entry. The reviewer identified the graph-replay exception
above, which narrows the candidate to the existing eager pool lease rather than
changing or removing the original low-level API. Scoped-memory proof is still
under review; no implementation has run.
