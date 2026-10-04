# C7b executable plan and probe design

Status: root reviewed the plan and approved CPU implementation in isolated
`/tmp` only. Start with generic byte/coverage gates, then the real Q128
projection graph and official MLA cap sweep; individual GEMM/BMM diagnostics
are only needed for unexplained behavior or a surviving cap. No GPU command
below has run. Production source stays locked for combined C7a validation.
C7a metadata work and C6 accepted runs remain separate. The isolated generic
prototype and `prepare`/`check` CLI now exist at
`/tmp/deepseek_c7b_transport_20261004_01/`. `prepare_02` checked 59 count/cap
partitions without CUDA or native imports and defined 1,145 fixture cases.
CUDA compilation, byte/coverage execution and all pair/timing modes remain
pending. Contract:
[c7b_task.md](c7b_task.md); hypotheses/risks: [c7b_draft.md](c7b_draft.md).

## 1. Freeze a temporary component workspace

After root approval, create a separate `/tmp/deepseek-c7b-transport-*` tree
with the original generic wrapper/native source, a cap-only prototype with a
distinct FFI module name, tests and `probe.py`. Keep live production files
untouched. Copy only necessary source; retain paths and SHA-256s. Hash the
actual DeepGEMM/FlashMLA interfaces and loaded native libraries, local includes,
compiler flags, GPU capabilities and environment settings affecting CUDA
connections or launches. Do not infer an environment value from its absence
in C6 metadata. Source-drift checks run before/after every measurement.

Provide `--help`, `--check-only`, `--mode transport|pair|lookahead`, `--compute`,
`--caps`, `--count`, `--launch-order`, `--address-regime`, `--warmups`,
`--repeats`, `--profile`, `--output-dir` and `--run-id`. Reject an existing
output run ID before CUDA initialization. A selected GPU must be present,
SM90, and have the required compiled dependencies; missing GPU cannot pass
via skip. The executable CLI and exact frozen paths are recorded at approval.

## 2. Implement only the cap candidate

Use the unchanged unrestricted entry as the reference. The prototype adds a
grid-stride record kernel with 128 threads and a separately named FFI entry.
Factor the byte/vector row-copy body only within the prototype, preserving the
baseline source. Keep per-row ID bounds/trap and actual pointer alignment.
No shared state/barrier is needed for disjoint destination rows. The initial
cap sweep is None, ceil(S/4), ceil(S/2), S, 2S, clamped to M for positive M;
S is the queried SM count. No production auto-tuning or per-call device query
is introduced. Compile once, inspect actual registers/shared memory and save
PTX/SASS or resource output. Check that host loads retain their ordinary
vector/byte policy; changes in code generation must be disclosed.

The future public API, after a selected component and separate integration
approval, is `gather_host_records(..., *, max_ctas=None)`. Validate explicit
caps as positive integers excluding bool; default callers keep the original
entry. Expose opt-in only through dense helper construction/config. No change
is made to sparse recall defaults, record semantics or publication timing.

## 3. Byte coverage, failure and lifetime gates

Before timing, compare the unrestricted and every capped kernel against an
independent CPU byte-scatter oracle. Check all destination storage, guard
regions and host storage after synchronization. Keep unique destination IDs;
include shuffled/fragmented host IDs and repeated host IDs to different slots.

| Dimension | Required cases |
|---|---|
| Counts | 0, 1, cap-1, cap, cap+1, 2cap+1, 1,024, 65,535, 65,536, 65,537; deduplicate valid values |
| Cap validation | None; positive integer; bool, float, zero, negative rejected |
| Records | BF16 widths 8/576; FP32 width 19; uint8 widths 1/15/16/17 |
| Alignment | Host/device storage offsets (0,0), (1,0), (0,1), (1,1), with byte-safe guards |
| IDs | Full and partial scatter; first/last valid IDs; sparse gaps; repeated host IDs with distinct targets |
| Protected storage | Untouched HBM rows, slot-0 sentinel, candidate tail and host guards remain byte-exact |
| Invalid IDs | Negative and upper-bound host/device IDs in fresh subprocesses; synchronized failure required |

For byte-pattern tests initialize raw storage as bytes before typed views, so
NaNs and payload bits do not turn float equality into the oracle. Test both
the default and cap routes. Empty input must not map host memory or launch a
kernel. Do not reuse a CUDA process after an expected device trap.

Build a diagnostic variant with row and vector/byte-unit visit counters at the
actual copy sites. Every input row and unit must count exactly once, and dense
unique-ID fixtures must total M x record_bytes. Record instrumentation identity
and remove counters from all performance kernels. These counters establish
algorithmic copy coverage, not hardware memory transactions.

For later helper integration, rerun the existing delayed-copy/private-ID
scratch-overwrite checks with a cap. Exercise drain of unused copies,
same/changed caller streams, stale/foreign tickets, failed submission and
drain-before-rollback/release. Use private IDs and existing record_stream rules;
the cap cannot shorten allocation lifetime or permit premature map consumption.

## 4. Compute fixtures that preserve the current workload

Prepare deterministic checkpoint-derived inputs once through the existing
model APIs. Fixture construction, H=65,536 prefix build, exact top-k selection,
device upload, snapshot restore, hashing, compilation and graph capture are
setup, each completed before timing. Save exact input/selection hashes and
actual shapes. Never extract or clone inputs inside a measured pair.

The first compute screen is the current projection graph and official sparse
MLA below. Run other rows only to diagnose unexplained behavior or a surviving
cap; do not require every micro-GEMM before rejecting an unhelpful cap.
The initial target is Q=128 and a full 65,536-record gather. Use disjoint HBM
storage for current-layer resident attention records and next-layer gather
output. Keep all weights/input tensors alive through both streams. Core
component fixtures are:

| Fixture | Exact work and API boundary |
|---|---|
| FP8 GEMM projection | `fp8_linear` q_a shape M/N/K=128/1536/7168 and q_b=128/24576/1536, measured separately; quantization and API helpers included |
| FP8 MLP | Current gate/up/down APIs with 128/18432/7168, 128/18432/7168 and 128/7168/18432; report separate APIs and the actual dependency sequence without artificial repetitions |
| BF16 BMM | Actual q_absorb B/M/N/K=128/128/512/128 and value expansion=128/128/128/512 with current output layouts |
| Official sparse MLA | `sparse_mla` Q=128, heads=128, D=576, V=512, H+A=65,664 records, exact valid 2,048-selection rows; include existing index conversion/layout/output work |
| Current projection graph | Existing compute-island behavior using one complete projection replay; capture once outside timing and preserve all real helpers |

These are component workloads. A projection fixture calling only the captured
graph must be labeled replay-only; it excludes production replay-input copies
and cannot be called complete compute-island API time. A complete replay API
measurement includes its real per-call input copies. Keep current output
references after each invocation; a setup-time output reference must not stand
in for newly computed output. Retain output through the final stream join and
hash it after synchronization outside the timing window.
No duplicated GEMM repetition may lengthen
compute merely to manufacture overlap. Do not call a stripped kernel complete
API time, and do not claim component fixtures are full serving. If a public
API allocates outputs or converts inputs internally, include that work in its
API measurement; only explicit fixture setup is excluded. If graph or library
reuse requires static buffers, document that distinct replay boundary.

For each fixture, compute-only repetition must establish a stable exact output
reference first. Serial and concurrent modes must preserve every output byte
and all input/weight/record bytes. If the unchanged vendor baseline itself is
not reproducibly byte-exact, stop this gate and report it rather than assigning
the difference to the cap or silently weakening the contract.

## 5. Paired schedules and timing

Keep the C6 topology for the primary comparison: default/null main stream and
one private non-blocking gather stream, both at the same priority. Use the same
events and work in both controls. After setup synchronization, record a start
event on the main stream and make the gather stream wait on it before work.
Preserve GPU submission times and stream IDs. Record the end event on the main
stream only after its compute is complete and it has waited for gather-ready;
this gives a causally ordered cross-stream makespan. CPU enqueue order alone
does not. A third control stream is unnecessary.

| Mode | Required dependency | Completion |
|---|---|---|
| Gather only | None beyond fixture readiness | Gather-ready event |
| Compute only | None beyond fixture readiness | Compute-done event |
| Serial pair | Main compute waits for gather-ready before the complete compute API | Both branches joined |
| Concurrent pair, gather first | Gather submitted, then independent current compute; no gather-ready wait before compute | Main joins gather-ready after compute |
| Concurrent pair, compute first | Same work with reversed submission order; diagnostic control | Both branches joined |

The serial control uses an explicit cross-stream event dependency, not a CPU
synchronize between branches. This preserves streams while enforcing true
serial GPU execution. Concurrent mode must not read the gather destination.
Start/end event creation/records and common waits must be identical across
pair schedules except the additional serial dependency. The CPU wall timer begins after fixture restore
and ends after the final join synchronization. CPU wall includes enqueue/API
cost; GPU event makespan includes device gaps. Neither subtracts measured
setup, sync or allocation time after the fact. Do not use a busy-spin or sleep
kernel to force the branches to queue together.

For each cap and fixture, use five warmups and thirty measured repetitions in
three deterministic randomized ten-sample blocks. Retain every sample and
report mean, median, p10/p90, pair differences and launch order. Interleave
unrestricted serial/concurrent anchors in each block. Require fresh-output
checks outside timing at least before and after each block; verify all outputs
when storing them does not enter the measured window. A selected cap receives
a second independent run ID with the same protocol before integration.

Use two separately labeled address regimes: fixed-address reuse and rotating
disjoint pinned-host source slabs. Record source/destination bytes, slab count,
reuse distance, L2 size as reported by hardware and stream properties. Choose
the same slabs/order for paired variants, with four source slabs at the primary
72 MiB shape as the initial rotating fixture. Neither regime proves no L2
reuse. Do not flush caches inside the timed region or change load policy.

Collect three profiled repetitions per selected unrestricted/capped pair after
unprofiled timing, using a separate profile run ID. Export raw Nsight SQLite.
Independently reconstruct gather and matrix-primary/API GPU intervals and
all GPU work, event edges and launch gaps. Preserve actual API input shapes,
selection count and bytes. Report gather/core and gather/API interval
intersections and uncovered gather time; no gather gives a null ratio.
Kernel envelopes alone establish observed temporal overlap, not sustained
useful execution. Claims of actual work overlap need separate diagnostic
per-CTA copy start/end timestamps and suitable matrix activity evidence.
Diagnostic instrumentation is excluded from latency promotion.

## 6. Complete lookahead and final integration are separate gates

Only after a component cap survives, replay the existing complete current-layer
work with C7a next-layer `prefetch/wait/drain`, preserving actual checkpoint
independent source-layer inputs and candidates. This phase includes next-layer
metadata planning/reservation, all actual current-layer graph/indexer/top-k/
MLA/MLP helpers, replay input copies, indexer workspace copies, ticket events,
joins and cleanup. Snapshot restoration and fixture construction remain outside
timing. Use the same C7a source for cap=None and the selected cap. Also run a
serial helper control that waits for next-layer gather before current compute.
Preserve startup target 0 separately and pair actual target j with actual j-1.

This complete layer-pair diagnostic is still not the final serving result.
Root later freezes a combined C7 source, reruns full independent-empty-cache
numerical acceptance and the complete fixed-P/NH sixteen-user/two-round
trajectory under a new formal run ID. Fresh profile/API-MFU work must match
that final source. Recheck first visits/all-hit calls, all candidate and request
latencies, exact tensors, maps/counters and allocated/reserved/device-used HBM.
No old C6 profile time or C7a isolated delta may be substituted or added to it.

## 7. Prospective commands and evidence layout

After root approves implementation, the temporary workspace will provide the
following interface. `C7B_WORKSPACE` is assigned to the newly frozen task tree;
`C7B_GPU` comes from a later explicit quiet-window grant. Neither is set or
executed by this planning task.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python "$C7B_WORKSPACE/probe.py" --help
CUDA_VISIBLE_DEVICES="$C7B_GPU" .venv/bin/python "$C7B_WORKSPACE/probe.py" \
  --check-only --caps none,33,66,132,264 --output-dir "$C7B_WORKSPACE/output" \
  --run-id c7b_transport_correctness_01
CUDA_VISIBLE_DEVICES="$C7B_GPU" .venv/bin/python "$C7B_WORKSPACE/probe.py" \
  --mode pair --compute all --caps none,33,66,132,264 --count 65536 \
  --launch-order both --address-regime both --warmups 5 --repeats 30 \
  --output-dir "$C7B_WORKSPACE/output" --run-id c7b_transport_pair_01
```

The literal cap list above is for 132 SMs; regenerate and record it if the
granted device differs. The future profile wrapper must preserve process exit
status, use separate stdout/stderr, create raw profile and data subdirectories,
and retain source/build/input identities. Engineering probes stay in `/tmp`,
not `experiments/`. A future paper experiment must use the repository's
experiment layout and result replacement rules.

Required output is a source/build/fixture manifest, full correctness JSON,
per-sample timings CSV, cap/compute/address-regime summary, allocation/storage
records, profiling SQLite plus independent overlap/event audit, and a candidate
ledger with parent candidate and selection/rejection reason. Record failures
as engineering evidence only. No failed/rejected probe is a valid experiment
result.

Component selection requires all exactness gates and repeatable improvement of
the gather-first complete pair over unrestricted concurrency in the realistic
current-graph/MLA controls. Inspect both mean and median across all blocks and
the confirmation run; changes within baseline repeatability are inconclusive.
Report regression in isolated gather, compute slowdown and serial-pair latency.
A cap winning only compute-first or a synthetic fixture remains diagnostic.
Production promotion additionally requires complete lookahead and final
serving acceptance; root decides after reviewing those fresh results.
