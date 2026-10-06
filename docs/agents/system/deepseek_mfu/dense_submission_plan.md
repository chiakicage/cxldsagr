# Private dense post-clear submission

## Scope and decision

Prepare one private cache-manager candidate that joins the copy-stream wait,
original contiguous H2D copy, and original dense-map publication in one host
native entry. Keep metadata clear, its CPU clock commit, both event-record
boundaries, the consumer wait, and terminal drain in their current positions.
This is a source-based proposal, not a measured improvement or implementation
authorization. No production edits, native builds, CPU tests, or GPU runs were
performed for this plan. Root schedules each later gate.

The baseline is current production **append02 + resident mask01 V10**, including
the prepared recall bridge. This candidate neither depends on nor duplicates
the separate private full-recall candidate. It affects contiguous dense history
prefetch only. It cannot independently satisfy the four-method gap requirement.

The opportunity source is
`/tmp/deepseek_v10_gap_opportunity_review_20261006_01/memo.md`. In the intrusive
V10 capture, dense embedding-to-first-history-H2D has 223.967 us of gap: 25.021 us
inside the clipped CUDA API union and 198.946 us outside it. The latter is not
a measurement of removable Python time. The required first copy is 75,497,472 B
and lasts 1,373.438 us; clear and publication GPU work also remain. The proposal
does not assign the entire observed gap to these wrappers or predict a saving.

## Current sequence and exact crossing counts

`models/deepseek_v32/cache/prefetch.py::_submit` currently does the following for
an uncertified, nonempty CUDA history:

1. Inside `cache.operation()`, create and retain the ticket with host/device
   owners. Recheck residency and provider support. Wait for host writes,
   normalize the clock, invalidate residency, and save the new map generation.
2. Call `dense_history_clear` on the caller stream. After its successful host
   return, increment `layer.clock` once. Create and record `metadata_ready` on
   the caller stream, then exit the operation lease and track its stream.
3. Make the helper copy stream wait for `metadata_ready`. Enter a Torch copy
   stream context. Call `copy_host_records_async`, whose private device/FFI
   scope calls the original common `copy_contiguous` Function.
4. Call `dense_history_publish`, with another device/FFI scope. Create and
   record `ticket._ready` on the copy stream. Exit the Torch stream context.
   Only then increment `cache.stats.recalled_records` and return the ticket.
5. `wait()` checks ticket ownership, activity, map generation, and caller stream;
   it waits for `_ready` before certifying residency. `drain()` confirms work
   completion before releasing borrowed storage.

The counts below concern explicit source-level entries for one nonempty miss,
not a claim about every C extension crossing or CUDA API visible in a trace.

| Item | Current | Candidate |
| --- | ---: | ---: |
| Post-clear Python-to-native packed Function calls | 2 | 1 |
| Complete miss packed Function calls, including clear | 3 | 2 |
| Post-clear transported tensors | 2 + 6 = 8 | 8 |
| Complete miss transported tensors | 5 + 2 + 6 = 13 | 5 + 8 = 13 |
| Post-clear explicit Python device scope pairs | 2 | 1 |
| Post-clear explicit Python FFI stream scope pairs | 2 | 1 |
| Python Torch copy-stream enter/exit pairs | 1 | 0 |
| Python `copy_stream.wait_event(metadata_ready)` calls | 1 | 0 |
| Actual copy-stream CUDA event waits | 1 | 1 |
| Metadata-ready and ready event creations/records | 2 / 2 | 2 / 2 |
| Actual contiguous H2D calls and publication launches | 1 / 1 | 1 / 1 |

Clear transports only h2d, d2h, priority, free bitmap, and evictions; its Python
page-table and clock checks do not export those two tensors. Copy exports host
and records, while publication exports six metadata tensors. There are no
duplicate post-clear tensor exports to remove.

The candidate adds raw stream/event handle reads and one native FFI stream
rebinding. It retains fresh metadata inspection. These costs, eligibility,
scope restoration, and all event work must be timed. Do not present the removed
public `wait_event` call as one net fewer C extension crossing: extracting its
event handle is itself additional host work.

The main countervailing cost is earlier export: six publication tensors move
from after H2D submission to before the combined call can start it. Their total
number is unchanged, but their placement can delay the first copy. The new
handle/flag work and native validation can also erase scope savings. Preserve
first-copy launch latency as a profile diagnostic and require a complete-cycle
wall-time win; do not infer one from the crossing counts.

Simply joining copy and publication while retaining the existing Torch stream
context mostly removes work after H2D has started. The proposed native wait and
explicit copy-stream binding additionally remove the Torch stream switch before
H2D. The metadata-ready record remains inside the original lease; moving it and
copy into one call would cross the intervening lease-exit failure boundary.

## Private candidate layout and dispatch

After authorization, create an isolated directory such as
`/tmp/dense_post_clear_candidate_20261006_01/` containing a private prefetch
module, host-C++ dispatch source/loader, and independent check/screen harness.
Load it under a private module name. Do not modify production modules or patch
shared globals to install the candidate.

The loader prepares one native Function per explicit device before helper or
pool execution. Bind the original common `copy_contiguous` and original ECHO
`echo_dense_history_publish` Functions from their actual native library modules,
using the existing library/C++-target checks. The original DSOs continue to own
all CUDA code and the copy implementation. No duplicate CUDA copy code, new
kernel, scratch tensor, graph capture, request-time build, or backend retry is
introduced. Fingerprint the real host source/include closure, compiler, FFI
runtime, original Functions, and loaded DSOs.

Choose the route before clear or any ticket mutation. The first candidate is
restricted to the ordinary exact helper/cache/pool/session layout, exact native
provider and original clear/publish/copy wrapper identities, ordinary scalar
types, and exact Torch tensors without instance metadata/transport overrides
or active TorchFunction/TorchDispatch modes. All CUDA tensors belong to the
helper device; the actual copy stream belongs to that device. Unsupported
providers, subclasses, modes, layouts, CPU paths, and empty history keep their
existing route. The private gate is normal dispatch, never fallback after a
candidate failure. Internal class/global monkeypatch authentication is not a
new API requirement.

Certified-resident tickets keep their existing protect/stamp path. They do not
call the continuation or allocate its events. Preserve the current generic
zero-history behavior explicitly, including no H2D copy.

## Submission ABI, stream binding, and validation

The proposed continuation accepts **8 tensors and 7 scalars**:

| Tensor order | Value |
| --- | --- |
| 0 | records on the helper CUDA device |
| 1 | pinned host records |
| 2 | page table |
| 3 | host-to-device map |
| 4 | device-to-host map |
| 5 | priority |
| 6 | free bitmap |
| 7 | clock tensor |

Scalars are `history`, `host_start`, saved pre-clear `timestamp`, the actual
copy-stream handle, the initialized metadata-ready event handle, a fresh
six-bit metadata `requires_grad` mask, and a fresh `host.is_pinned()` result.
`device_start` remains the private constant 1. The last two values transport
per-invocation owner metadata that TensorView does not expose; they are never
cached validation-success tokens. Host/record `requires_grad` remains accepted
where the original generic copy accepts it. No CPU tensor progress ledger is
needed because the only CPU clock advance stays before the continuation.

Keep one explicit Python device scope plus `tvm_ffi.use_torch_stream()` around
the continuation. Torch's current stream remains the caller stream. After all
argument conversion, the native body binds the FFI device slot to the supplied
copy-stream handle, calls `cudaStreamWaitEvent(copy, metadata_ready, 0)`, validates
copy metadata, invokes original `copy_contiguous`, validates publication
metadata, then invokes original `echo_dense_history_publish`.

The stream rebinding must be **inside** the native entry. The installed FFI
Torch argument setter observes the first CUDA tensor's producer stream, and
`FuncCall` binds that stream immediately before calling the Function. An outer
`use_raw_stream(copy)` alone would therefore be overwritten by the caller
stream. The explicit Python FFI scope remains the authoritative restoration
owner: it unconditionally restores its saved stream after the native call,
including failures. No Python callback, PyTorch computation, or further native
dispatch occurs between continuation return and that scope's exit. The entry
is private to this wrapper; direct unscoped native calls are unsupported.

Do not rely on automatic FFI restoration alone. Its current helper restores
only when its saved stream differs from the producer stream; those streams
usually match in this wrapper. A native change could otherwise escape. Probe
both equal and different enclosing FFI streams, and verify the final FFI and
Torch stream plus current device exactly. Scope construction and restoration
must preserve the original error object and every cleanup error in Python,
following the existing private manual-exit pattern rather than exception-
throwing C++ destructors.

Validation remains fresh and staged:

- After the native copy-stream wait, reproduce the generic copy requirements:
  nonnegative integer spans, CPU pinned source, CUDA target, equal dtype and
  positive row width, rank 2, contiguous storage, and both allocation bounds.
  Use the freshly read pin bit; adding a new hot-path pointer-attribute CUDA
  query is not part of this candidate. Keep offset contiguous views supported.
  Original native `copy_contiguous` retains its own span check and actual DMA.
- Only after original copy host return, reproduce `_dense_history_maps` and
  `_resident_maps`: bounded history, the original timestamp domain, page-table
  coverage, host-span bounds, exact metadata dtype/device/shape/contiguity, and
  the transported fresh `requires_grad` bits. Then call original publication,
  retaining its GPU map assertions and clock write.
- List any validation failure that moves in the private path. In particular,
  all eight tensor exports happen at one entry, so a later publication tensor
  conversion failure can occur before the copy-stream wait/copy. It still occurs
  after successful clear, CPU clock commit, metadata event record, and lease
  exit. No second attempt, proof restoration, or recovery is allowed. Exact
  scalar/provider/type eligibility is checked before mutation; export success
  must never be cached between calls.

## Clock, ticket, ownership, and failures

Keep the original clear call and `layer.clock += 1` statement. Clear does not
write the GPU clock; successful publication writes `timestamp + 1`. A post-clear
failure must not undo the CPU event or advance it again. CPU/GPU clock equality
is required after successful publication, not inferred after partial enqueue.

Attach `metadata_ready` to the ticket before recording it, and retain the actual
caller/copy stream objects plus the continuation tensor frame on that ticket.
The helper already retains the ticket before clear. Retain `_cache`, which owns
the session, pool, maps and counter slab, plus the full host/record storage.
Retain the frame through FFI/device scope exit and ready-event record. Do not
infer async completion from a Function return or from a recorded ready event.

Ready-event creation and record remain Python operations after publication and
successful scope exit. Recalled statistics advance only after that record
succeeds. Keep the original map-generation update immediately after residency
invalidation. `wait()` still rejects a stale generation or foreign/expired
ticket and requires the original caller stream before waiting and certifying.
This first candidate does not reuse event pairs or add a new generation counter.

| Failure point | Required state and cleanup |
| --- | --- |
| Clear body before successful return | Original clock behavior; ticket already owns storage; helper failed. |
| Metadata event record or lease exit | CPU clock already advanced once; continuation not entered; helper failed. |
| Export, FFI rebind, or copy-stream wait | No completion inferred; ready absent; retain ticket, event and all owners. |
| Copy validation/DMA or publication | Keep actual submitted prefix; no ready or statistics success; maps uncertified. |
| Scope exit or ready-event record | Publication may be enqueued; helper failed; retain owners until drain. |
| Consumer wait or completion check | Preserve failure state and original errors; never certify on a failed wait. |

Keep existing helper failure propagation and any pool poison produced by its
lease; never reset either state to allow execution. Existing enclosing request
rollback/close still runs. Drain both the actual copy stream and caller stream
on failure, preserving all completion errors. If any completion check fails,
release no ticket-owned tensors, events, or storage. Successful drain marks all
tickets inactive, clears the new frame/event references with the existing
borrowed owners, and resets the execution caller binding. A successful drain
does not clear `helper.failed`. Preserve the original error together with
drain/rollback/restoration errors; do not replace it.

`pending_bytes` and tensor scratch reservation remain 0: the added references
borrow existing storage. The same two lazily created CUDA events remain per
nonresident ticket, and now have explicit ticket lifetime. Report event handles
and ownership separately from tensor capacities. No event allocation is moved
out of the timed request in this first candidate.

Event reservation/reuse is deliberately a separate possible candidate. It
would need a declared maximum live-ticket capacity, distinct event pairs,
generation-checked leases, successful drain before reuse, and separately
accounted construction cost. Do not include it in the first screen.

## Independent gates and measurements

1. Root reviews this plan, then authorizes isolated implementation. Freeze
   baseline/current source, fixture identity, and both loaded CUDA DSOs. Extend
   the host bridge only; keep the production common and ECHO CUDA sources intact.
2. In a scheduled CPU-only slot, check fresh validation and dispatch, exact
   clock positions, stage order, ticket retention, map-generation checks, scope
   restoration, and primary-plus-cleanup exception identity. Mock CUDA only
   for these boundary tests and label that limit. No fake timing claim follows.
3. Root separately builds the host bridge and probe. Counting original-style
   native targets verify the 8-tensor/7-scalar ABI, actual pointer offsets,
   exactly one wait/copy/publication in order, and no copy after a lease failure.
   Real Torch conversion and CUDA tests must verify that copy/publication use
   the supplied copy stream when Torch and enclosing FFI streams differ.
   Cover different current devices, both caller-stream kinds, conversion and
   native-stage faults, and restoration. Save actual before/after evidence.
4. Run original kernels on independent dense pools. Cover cold, certified,
   unproven all-hit, partial/permuted history with another session's unrelated
   suffix, nonzero host offsets, generic record width/dtype and contiguous
   offset views, empty history, H=P, pending D2H writes, multiple layer tickets,
   stale generation, caller changes, and copy/publication/ready/drain failures.
   Require exact consumed KV, direct maps, free bits, legal eviction counts,
   clocks, ticket state and actual H2D bytes. Scratch mutation must not affect
   DMA. Sticky CUDA faults run separately; no automatic rerun after failure.
5. Screen a three-layer standalone helper using current captured KV fixtures:
   H=65,536, P=NH=65,664, BF16 width 576, normal persistent-append configuration.
   Use cold history and certified-resident controls. Reset each arm independently
   with production operations; reset and validation remain outside timing.
   Time the full three-layer `prefetch/wait` cycle plus helper drain, with
   enqueue separately. Retain the production lookahead submission order and
   include every per-ticket check, event creation/record, handle/flag read,
   bridge call, statistics update, and completion check. Native preparation and
   helper/pool construction remain declared setup, with no new event prewarm.
6. Use three warmup cycles per arm/state, then 40 balanced AB/BA pairs per state:
   160 measured cycles and 480 measured layer tickets. Keep every sample, pooled
   and both order-stratum paired wall deltas, and resident controls. Cold cycles
   must transfer 226,492,416 B; resident cycles transfer no history H2D. Neither
   enqueue-only improvement nor subtraction of control time is a wall-time win.
   Root owns GPU/NUMA/CPU isolation, observation, and new SSD run IDs. Reject a
   null or regressive result without repeating an unchanged candidate.
7. Only a convincing standalone wall-time win permits matched complete-model
   validation/timing and minimal profile on current V10. Require full outputs,
   cache state, actual traffic, and request benefit. The user gate remains
   complete extend **and every layer** strictly below 10% under the established
   compute/actual-IO union gap definition; diagrams must label compute stages.
   Cache-manager, MFU, and affected fixed-P/NH motivation experiments require
   fresh formal runs before promotion/publication. Shared common code remains
   unchanged; do not infer NOSA or generic dense-staging validation from this
   DeepSeek helper check. Check their dependency paths if the final scope grows.

## Source identities at proposal time

All repository paths below are relative to the repository root. Installed FFI
sources are under `.venv/lib/python3.12/site-packages/tvm_ffi/`.

| Source | SHA-256 |
| --- | --- |
| `models/deepseek_v32/cache/prefetch.py` | `854a3d5081ba396a900b233ad1d1c3678d73cf6de68a0aed0a5d85315351ec31` |
| `cache/sparse_token_pool.py` | `b34c8cfbc43e58881bd14aa2af314ce7fe612cbbd906ca8573f4259d937e5650` |
| `operators/deepseek_v32/indexer/cache_ops.py` | `2aaccd6639e17b605f691833bfac426c35fa7ccba10d440b716822f9f215af33` |
| `operators/deepseek_v32/indexer/csrc/echo_dense_prefetch.cuh` | `05805ae0aa5c5335de74958445269b9585edd97b707dcd2864af60f584960f38` |
| `operators/common/kv_transfer.py` | `787cd38330ea263c5948bcb4510257d3b43fb789e93a6bee207c14a9ab5a480f` |
| `operators/common/csrc/kv_transfer.cu` | `71821370997ee6afb92b0aa0cc7da4c813d17e407b621e2b29aeb343747e525b` |
| FFI `cython/function.pxi` | `258330bb95815fbbeb15a3190cba8999b24b57aed8211a50821d4d2e2b53a450` |
| FFI `include/tvm_ffi_python_helpers.h` | `663869afb23c886145c40d5ecadb7445a7c77d71826c0b29696c3f2e9a81ca6e` |
| FFI `stream.py` | `586413f5558bd8682f31280743cf7fe6581a17b7e2eb270e19ffcceba12c2f0c` |
| Opportunity memo | `995473674d88277d98a4b5db5e2db869579d9ecea5e0d0ca419544628c61a490` |
