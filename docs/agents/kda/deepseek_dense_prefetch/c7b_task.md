# C7b bounded gather task contract

Status: root reviewed the plan and authorized CPU implementation in an isolated
`/tmp` workspace. No production edits or CUDA work are authorized. Start with
generic exact coverage, then the real Q128 projection graph and official MLA
serial/concurrent controls. Other matrix diagnostics are conditional. Any
later GPU work requires a quiet-window grant.

## Objective

Test whether an optional CTA cap with a grid-stride whole-record loop reduces
the complete latency of next-layer mapped-host gather concurrent with current
GEMM/attention work. Preserve the existing record transport, dtype/layout,
ordinary load policy and stream/lifetime contract. A cap does not reserve SMs
or establish that C6's observed delay was caused by SM contention.

This extends [the C7 task](task.md) and is independent of C7a native metadata.
The first prototype and probe remain in a separate `/tmp` workspace; the live
C7a source and accepted C6 source stay unchanged. Production integration and
the final combined C7 serving experiment are separate decisions.

## Inputs and outputs

The generic operation receives contiguous two-dimensional pinned CPU records,
matching CUDA records, and equal-length contiguous CUDA int64 host/device ID
vectors. For M work rows, row r copies the complete caller-specified record
`host[host_ids[r]]` to `device[device_ids[r]]`. Preserve every byte, including
arbitrary bit patterns; no arithmetic or selection change is permitted.

The target is BF16 x576, 1,152 bytes per record, M=65,536 (72 MiB per gather),
with Q=128 current-layer work, H=65,536 and top-k=2,048. HBM source records used
by current-layer attention and the next-layer gather destination must be
separate allocations. Generic width, dtype, offset, sparse-ID and count
contracts also remain valid. Repeated host IDs to distinct destinations retain
their existing per-input-row copy behavior; the cap must not deduplicate them.

## Implementation constraints

- Keep 128 threads per CTA and the aligned uint4 / unaligned byte copy paths.
  Check both actual row addresses and width inside each grid-stride iteration.
  Keep ordinary loads; do not introduce `.cv`, cache hints, shared staging,
  warp specialization, new atomics in the timing kernel, or a vector-layout
  change in this candidate.
- Launch `min(M, cap)` CTAs and process
  `r = blockIdx.x; r < M; r += gridDim.x` with int64 row/address arithmetic.
  Every requested work row belongs to exactly one CTA. Preserve per-row bounds
  checks and the existing device trap for invalid host/device IDs.
- Public integration, if later approved, uses keyword-only `max_ctas=None` on
  `gather_host_records`. None preserves the original route; an explicit cap must
  be a positive integer excluding bool. Reject malformed values before launch,
  including for empty input. Empty valid input maps/launches nothing.
- Only dense `PoolHistoryPrefetch` may explicitly opt in through its own
  construction/config. Ordinary checked recall, residual recall and other
  generic callers keep the default. No global cap, backend monkey patch or
  change to ordinary recall semantics is allowed.
- Keep private ticket-owned host/slot IDs, `record_stream`, metadata-ready,
  gather-ready, caller-stream and drain/rollback/release ordering. Mapped-host
  and target allocations must live through completion. A cap adds no persistent
  storage, HBM staging layer or change in map publication.

## Correctness and evidence

Before timing, compare every target/guard byte and compute output against the
same unrestricted operation. A diagnostic-only build additionally counts each
work-row visit and each vector/byte copy unit. Require one visit per work row
and one copy per unit; dense unique-ID tickets account for exactly
M x record_bytes. This proves work coverage in the instrumented kernel, not
physical PCIe traffic or absence of hardware cache effects. Timed kernels have
no counters. Invalid-index traps run in disposable subprocesses.

The [draft](c7b_draft.md) records the evidence and risks; the
[executable plan](c7b_implementation_plan.md) specifies controls, acceptance
commands, timing and promotion. Record code/build/input hashes, actual GPU and
stream properties, dependencies, precision, run ID, warmups/repeats, all raw
samples and exact measurement boundaries. No data preparation, snapshot
restore, weight load, compile or fixture creation may enter a transport-only
timing window; actual API helpers and output allocation remain inside any
measurement labeled complete API time.

## Promotion

Component screening may select a cap only after all byte/output/lifetime gates
pass and its complete gather-plus-compute latency improves against the
unrestricted control under the same inputs, stream topology and launch order.
An overlap fraction, faster isolated gather or favorable single sample cannot
promote the implementation. Report isolated gather and compute slowdowns too.

After review, a selected cap must pass complete C7a helper lookahead checks and
root's fresh combined serving numerical/performance run. Compare final C7
against the accepted C6 baseline using all samples and the same P/NH workload.
First-visit/all-hit behavior, candidate/request latency, memory, exact outputs
and counters remain required. C7a and C7b component results cannot be added to
predict a serving speedup. Root owns experiment publication and old-result
replacement.
