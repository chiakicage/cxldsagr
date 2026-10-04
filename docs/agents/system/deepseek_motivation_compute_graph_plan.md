# Compute-island graph integration plan

Status: implementation proposal, 2026-10-04. Production/model/operator and
motivation sources remain frozen while the C1+C2 screen runs. Only temporary
GPU 1 probes and this execution document were added for this planning task.

## Objective and boundaries

Replace repeated Python submission of pure model computation with two CUDA
Graph replays per layer. Keep the current eager cache operations, leases,
session state, exact indexer/top-k/recall, transient candidate handling, and
request synchronization. This does not enable cache or whole-request capture;
`SharedSparseTokenPool._check` must continue rejecting capture of cache calls.

Use the same independent ten checkpoint block copies, Q=1024 history chunks,
Q=128 candidates, H=65536, P=65536, and NH=16777216. Do not pad candidate
batches, reduce exact selection, share weights between copies, or change the
FP32 normalization/residual equations. The final criterion remains complete
four-scheme request latency/MFU, including copies around graph boundaries.

## Feasibility evidence

`/tmp/deepseek_c3_compute_islands_probe.py` makes a temporary in-memory copy of
the current projection function. It changes only position construction from
`arange(start,start+Q).float()` to `(arange(Q,int64)+device_start).float()`.
No production module is modified or monkeypatched. It then captures:

1. `residual_rms_norm(hidden,residual)` followed by projection of normalized
   input, returning the six `Projected` fields and the saved BF16 residual.
2. Absorbed attention output expansion/projection, post-attention residual norm,
   and the existing dense MLP, returning hidden and residual.

With real layer-0 checkpoint weights, Q=1024 and Q=128, both incoming-residual
branches, and changed hidden inputs, graph 1 matches all six fields and the
saved residual byte-for-byte at starts 0, 1024, 64512, and 65536. Graph 2 is
byte-exact against eager for three changed attention/residual inputs at each Q.
The probe is not full-model, cross-session, or performance acceptance.

Median graph replay times, 20 warmups and 100 event-timed repetitions:

| Q | Input norm + projection, no residual | Input norm + projection, residual | Output + norm + MLP |
|---|---:|---:|---:|
| 1024 | 0.443552 ms | 0.497920 ms | 1.222896 ms |
| 128 | 0.170048 ms | 0.171520 ms | 0.235568 ms |

These timings exclude inter-island copies and eager cache/attention work.
The Q=1024 finish timing varied from 1.1496 to 1.2384 ms at p10/p90, so it is
not a forecast for the integrated pipeline. Evidence, memory snapshots, and
source hashes are in `/tmp/deepseek_c3_compute_islands_probe.json`.

## Ownership and graph keys

Add `models/deepseek_v32/compute_graphs.py` with a backend-owned
`DeepSeekComputeGraphs` resource. It holds static input tensors, graph objects,
borrowed output views, capture metadata, and measured allocation capacity.
Sessions hold no graph, weight, or private compute workspace. The backend's
existing single-execution lease also serializes this resource.

The graph key is `(independent_layer_id, Q, island_kind, residual_present,
precision_configuration, source_identity)`. It is not `(source_layer % 3)`:
all ten copies retain distinct weight addresses. Only projection uses the
residual-present branch. In this workload it is false for layers 0,3,6,9 and
true for the other six layers, in both history and candidate execution.

The initial bank contains two islands for each independent layer and each of
Q=1024/128: forty graphs. It contains no session pointer, KV/indexer pointer,
host page number, sequence length, or ECHO offset. Increasing the candidate
limit or executing another supported candidate length must not rebuild a
retained history. For an unprepared Q, retain an explicitly reported eager
fallback; do not capture or compile a graph inside a timed request. The formal
target workload must have 100% expected island coverage, without such fallback.

Initially use independent graph-private pools. Their lifetimes are easier to
audit and their replay order can follow any valid session trajectory. Sharing
a private pool is a later measured optimization requiring proof that all
shared graphs replay in a compatible order, never overlap, and cannot overwrite
still-live outputs. Do not introduce pool sharing merely to improve an estimate.

## Model integration without duplicated cache logic

Keep the ordinary `CheckpointBlock.forward` path for standalone calls and
unsupported graph shapes. Add a serving-only dispatch for its existing
single-chunk case. The compute resource closes over immutable references to
`backend.attentions[layer]`, MLP, and norm weights. It must never capture a
lookup through mutable `block.attention`, `block.cache`, or a session runner.

Factor `EchoAttentionRunner._forward` so its current eager body can accept two
explicit internal callbacks, while the public ordinary path keeps its existing
projection and output callbacks:

```text
runner's existing operation lease
  reserve_append_source()                  # before any graph output is overwritten
  p = projection_callback(position)        # graph 1; also retains saved residual
  index cache write / declare visible
  prepare / indexer / finalize / exact top-k / offset
  append p.kv
  exact recall / raw absorbed MLA
  result = finish_callback(raw_attention)  # graph 2 returns hidden, residual
```

The serving block adapter supplies a projection callback that takes the raw
hidden/residual inputs, runs graph 1, and remembers its saved residual for
graph 2. This avoids a third norm-only boundary. The finish callback copies
absorbed MLA and that saved residual into graph 2's inputs, replays, and returns
the pair. Do not fork or reimplement the cache/indexer body in the graph module.

The callbacks preserve current capture hooks, exception handling, and scope
names at their eager boundaries. An exception propagates through the existing
request rollback/poisoning rules. `block.attention` and `block.cache` continue
being installed and cleared by the backend; they are not part of graph state.

## Inputs, positions, and numerical identity

Graph 1 static inputs are BF16 `[Q,7168]` hidden, optional same-shaped residual,
and a device int64 start scalar. Precompute its constant int64 `[0..Q-1]` base.
Inside capture, add the start scalar in integer arithmetic, then convert to
FP32 and construct angles/trig exactly as today. All supported positions fit
exactly in FP32, and the probe verifies that this ordering matches the original
`arange(...).float()` path. This requires one graph per Q/layer, not 64 graphs
for the history positions. No full-context trig cache is needed initially.

Refactor projection into a checked ordinary entry point and a pure internal
entry accepting the explicit positions tensor (or equivalent device start and
base). The eager callback validates `position == cache.written` and
`0 <= position`, `position+Q <= cfg.max_seq_len` before replay. Do not embed a
permanent start=0 validity check and treat that as validation of later positions.

Copy hidden/residual and update the device scalar on the backend's execution
stream, then replay there. The scalar can be written with one `fill_`; its
launch is timed. Do not read it back or introduce a scalar D2H. No computation
changes to rounding, activation quantization, RoPE pairing, FP32 head weights,
or either residual norm are permitted.

Graph 2 static inputs are BF16 `[Q,128,512]` absorbed attention and BF16
`[Q,7168]` saved residual. Its operations use the current operand order:
`attention.output`, then `residual_rms_norm(saved, projected_attention, ...)`,
then dense MLP. Copy-in and replay costs both remain in the layer/request time.

## Output and asynchronous-copy lifetime

Graph outputs are borrowed buffers overwritten by that graph's next replay.
A Python reference or `record_stream` does not prevent those bytes from being
overwritten. The following rules are part of implementation correctness:

- Index Q/K/scales, weights, and packed Q stay valid through the eager indexer,
  cache-write and MLA operations on the same stream. Do not launch the same
  graph again before those consumers are ordered ahead of it.
- **Persistent main-KV append requires an owned source.** Before passing a
  graph-owned `p.kv` to asynchronous host writeback, copy it into an ordinary
  bounded owned source tensor. Existing writeback tickets retain that tensor
  until their events complete; existing `max_inflight_writes` and source
  reservation still apply. Its allocation and D2D copy stay timed and charged.
  At Q=1024 each source is 1.125 MiB; do not clone the 144 MiB query tensor.
  A later two-slot source ring is allowed only with explicit per-slot completion
  before reuse. Generic source-count headroom alone is not that proof.
- Transient candidates have no D2H KV persistence. Their GPU tail copy and
  subsequent MLA are ordered on the same stream; no owned host-writeback
  source is needed, and the existing candidate counters remain unchanged.
- The saved residual from graph 1 must survive until graph 2's input copy.
  Separate private pools and stable output references provide that lifetime.
- Graph 2's hidden/residual outputs may feed the next layer and the existing
  independent replay-input clones. They must remain valid through those copies.
  Per-independent-layer graph keys prevent later layers from overwriting the
  three saved source inputs during the same chunk. Keep the explicit clones
  for layers 3–9, even when their values are equal.
- Final normalization/LM head and `output_parts` remain eager initially.
  Their returned tensor is owned and survives future requests; never return a
  borrowed graph output directly from `extend_candidate` or `prefill`.
- Diagnostic hooks that retain outputs must receive owned snapshots or complete
  their existing snapshots before reuse. Test this explicitly; a callback's
  retained reference must not silently become the next request's output.

## Planning, capture, and actual HBM capacity

Extend the existing pure `plan_resources` metadata with exact static input
shapes and a declared upper limit for graph-private allocation. Include that
new backend execution allocation in the shared reservation once, never in each
session. Ordinary model compute versus cache scratch stays separately reported;
neither category is excluded from the physical HBM check.

The private-pool limit must be stated as a chosen planning bound, not an
observed fixed requirement. Derive/tighten it from the allocation inventory and
calibration for these pinned libraries. Allocation rejects any graph bank that
exceeds it. An allocation-free planner cannot discover opaque vendor/allocator
sizes by capturing on the GPU. Do not silently allocate first and change the
plan afterward, or subtract an unrelated old allocator gap from fixed P/NH.

In `allocate_shared(plan)`, warm the exact operator variants on a private
warmup stream, join it, and capture before accepting sessions or starting
request timing. Use initialized BF16 dummy inputs and legal positions. Warm
DeepGEMM, FlashInfer, the FP8 quantizer, and all residual branches before capture;
the prototype demonstrates they can be captured after warmup. Record setup
latency separately. A cold user request still builds its own history from an
empty session; prepared compute graphs are model/runtime initialization.

For each graph, identify its `graph.pool()` in the allocator snapshot's
`segment_pool_id`. Sum the **total segment capacity**, including inactive cached
blocks, not merely live graph output storages. Keep three measurements:
PyTorch allocated, PyTorch reserved, and total device used. Keep graph-private
segments separate from ordinary allocator cached segments and driver/cuBLAS
allocations. Count explicit static inputs outside the graph pools separately,
deduplicating by owning storage. Weights are already counted with the model.

The temporary one-layer probe observed these private capacities:

| Q | Projection, no residual | Projection, residual | Finish |
|---|---:|---:|---:|
| 1024 | 342 MiB | 324 MiB | 290 MiB |
| 128 | 42 MiB | 42 MiB | 42 MiB |

For example, the 290 MiB finish pool had only 28 MiB active after capture;
reporting the latter as its HBM cost would be wrong. Extrapolating the measured
per-graph pools to 4 no-residual and 6 residual copies gives about **6.89 GiB**
private reserved capacity, plus about **1.81 GiB** of separate per-key input
buffers for the simple design. These are planning estimates from one copied
source layer, not an observed full-backend capacity or mandatory fixed reserve.
The actual bank and total device usage must be measured before admission.

`shared_bytes`/resource diagnostics must expose graph storage and actual private
capacity without double-counting their outputs. Repeated replays, new sessions,
and all candidate sizes must not grow the bank beyond its plan. Destroy graphs
and their input/output references only after all compute/copy consumers drain.
If completion cannot be established, poison and retain ownership as the current
backend does. Allocation failure rolls back only this allocation attempt's new
resources, preserving the existing owner/rollback contract.

## Profiling and MFU integration are required

`InstrumentOperators` currently records Python wrapper calls and NVTX scopes.
Those wrappers run during capture, not replay. Ordinary runtime scopes alone
would omit matrix work from the new ledger, and eager operator timings cannot
be substituted for serving graph timings.

Build a capture-time structural ledger for every graph: source and independent
weight identity, graph key, each matrix API's shapes/dtypes/useful FLOPs,
primary and auxiliary graph-node membership, and immutable graph/node IDs.
At runtime record every graph key and replay count under a lightweight eager
island scope, including the request/chunk/layer. Multiply the frozen structural
ledger by actual replays and add the still-eager indexer, MLA, and LM-head calls.
Verify it independently against the static workload FLOPs and replay sequence.
Do not infer ten independent copies from one graph execution.

Prototype Nsight Systems `--cuda-graph-trace=node` and inspect its exported
graph/node identifiers. A concrete node-labeling path is to query the active
capture graph with `cudaStreamGetCaptureInfo_v2`, take node-set differences
around each instrumented API (`cudaGraphGetNodes`), and resolve those nodes to
CUPTI graph/node IDs. Check the installed APIs/export schema on a small graph
before implementation; do not assume a node handle equals an exported ID.
If that mechanism is unavailable, retain explicit labeled child-graph/operator
nodes or another verified equivalent mapping. This mapping work proceeds with
the optimization; it is an acceptance deliverable, not a reason to keep the
production compute path eager.

Attribute each replay's real GPU node activities to those captured API groups.
Use activity interval unions per invocation, including quantizers, layout
copies, reductions, and other API helpers, exactly as in the existing report.
Replayed primary and helper counts must reconcile with the captured node
inventory. Preserve parent/child API membership without double-counting, and
do not truncate GPU activity at the enclosing CPU replay scope's end.

The first graph-enabled profile must prove: zero missing matrix calls/nodes,
no duplicate attribution, matching precision-weighted FLOPs, actual matrix API
GPU timings inside replay, full request stage totals, and graph coverage for
all expected Q/layer/branch keys. Publish weighted matrix API MFU and wall MFU
from that new source/run. Separate isolated graph measurements remain engineering
evidence and never replace this graph-enabled serving profile.

## Implementation and validation sequence

1. Add the pure position-input interface and capture resource with one layer/Q;
   compare all fields at changing positions and incoming residual states. Keep
   eager entry points and CPU references intact. Test immutable inputs and
   output alias behavior, then expand to all independent weight copies.
2. Factor runner callbacks without duplicating cache logic. Integrate the two
   islands and owned persistent-KV source boundary. Test delayed D2H, a single
   layer, repeated replay of the same key, multiple sessions, failure/rollback,
   and diagnostic output retention. These tests target the new lifetime risks.
3. Integrate allocation, graph-private reserved accounting, setup, teardown,
   owner failure paths, Q coverage, and source manifests. Audit repeated replay
   for memory growth. A missing graph in the formal target workload fails the
   coverage gate; general eager fallback remains explicitly reported.
4. Implement/reconcile captured matrix ledgers and node timing attribution.
   Run a small annotated/control equivalence check before the full profiler.
5. Run the current real-checkpoint four-scheme numerical suite and full H+A
   screen with independent empty caches and eviction/revisit cases. Compare
   every candidate hidden/logit value and retained history against eager C1+C2.
   Count zero candidate D2H and unchanged logical selection/cache behavior.
6. Rerun the full sixteen-user/two-round trajectory and graph-enabled profile,
   recompute MFU, inspect remaining launch gaps, and measure all boundary copies.
   Continue optimizing if latency/MFU convergence remains unproven. Replace
   published results only through the existing fresh-run acceptance gate and
   audit shared-consumer experiments affected by the model changes.

Temporary probe command (repository root):

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=1 .venv/bin/python /tmp/deepseek_c3_compute_islands_probe.py
```
