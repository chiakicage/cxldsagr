# Private full eager recall entry

## Task and decision boundary

Prepare one private host-dispatch candidate for uncertified exact-top-k recall
with `H <= P`. It should submit the original classify, allocate, gather, publish
and map Functions through one Python-to-native call. Python preflights owner
facts and allocations; that same native entry checks deduplicated TensorView
metadata before classify. A CPU ledger preserves completed clock events on failure.
This deliberately changes private synchronous-error staging; it is not a
drop-in preservation of every earlier Python boundary.

Root approved private implementation and narrow CPU checks after reviewing this
plan. Native builds and GPU execution remain root-owned. Current production is
**append02 + mask01 V10**, including all current source and loaded native
identities. `echo.py` currently hashes to
`de20ebc3b4905b3dddb646167fb770d27a92146e76934be3160ad1b5b149c562`.
Do not substitute old V9 snapshots or rejected stream03. Equal CUDA payloads do
not make different host ELFs identical baselines.

The opportunity is three stage invocations / 32 tensor exports becoming one
invocation / **22 tensor exports plus four scalars**: the 21 distinct recall
tensors plus the CPU ledger; scalars are history, written, timestamp and candidate
capacity. This removes ten exports, before counting scalar checks and ledger
work. It also deduplicates repeated metadata requirements across the three
wrappers while retaining every required condition. Neither change predicts a
speedup; validator01 and combined02 remain rejected.

Source basis: `cache/sparse_token_cache.py::_ensure_sparse_from_topk`,
`cache/sparse_token_pool.py::{operation,wait_host,_clock_event,invalidate_residency}`,
`operators/deepseek_v32/indexer/{cache_ops.py,recall_dispatch.py}` and
`csrc/echo_recall_dispatch.cpp`. The bounded feasibility memo is
`/tmp/full_recall_entry_feasibility_20261006_01/memo.md`; its callback proposal and
old staged-error constraints are superseded here only for this private route.

## Draft contract

**Applicability.** Keep the existing outer operation lease. Use a bounded set of
checks before mutation: exact built-in cache/pool/session types, their owner
association and `pool._depth == 1`; native provider module identity plus the five prepared wrapper
identities (`classify`, `allocate`, `allocate_free`, `complete`, `map`); exact
`torch.Tensor` type with an empty instance attribute dictionary for all transported
tensors, excluding overridden metadata/transport methods; no active
TorchFunction/TorchDispatch mode; ordinary supported
scalar types. The private `RecallFixture` method binding is declared at setup.
Current session count selects the original free-only/general allocator proof.
Generic or overridden provider routes remain segmented; tensor subclasses and
override modes also take the original route. Arbitrary monkeypatching of internal
pool/class code is not an additional supported API: do not add per-call class
dictionary scans or source authentication. Metadata checks remain fresh. Failed
preflight or native execution never triggers fallback or another invocation.
Certified-resident, public wrappers, `H > P` splitting, append and clock-boundary
handling retain their existing routes. Eligible transient suffixes retain the
original geometry and direct suffix mapping.
Nested leases retain segmented recall in this initial standalone candidate.
The caller-local frame cannot outlive a genuinely enclosing attention lease;
nested support requires a distinct owner-lifetime extension before any model trial.

**Preflight.** After original clock normalization and the existing boundary
fallback, save `timestamp = layer.clock`. Form and retain an invocation frame
containing the actual owners and views. Python checks exact integer domains,
the original `requires_grad` rules, allocation capacity and whole-backing-storage
aliases freshly. The native entry checks geometry/dtype/device/contiguity once
per distinct requirement before classify, without an added validator FFI. Offset
data pointers do not establish distinct storage. Preserve the accepted input
set; this change does not demand an ordering among simultaneous invalid fields.
Allocate `physical`, `chosen[P]`, workspace and a fresh zeroed CPU `int64[1]`
ledger before classify. Use the original cached workspace sizing and enforce
`workspace_bytes + 8*P <= 64*(P+1+NH)`; do not introduce a request-time build or
an extra warmed sizing FFI call. Validate newly allocated outputs too.

Choose free-only allocation from the current sole-session and `H <= P` proof;
otherwise bind the original general FIFO allocator. Preserve its GPU proof
check. Run `wait_host` before native entry, including ticket reaping and matching
event waits. Python checks/allocations/waits may fail before any recall launch,
invalidation or clock advance. Native metadata failure occurs after conservative
CPU proof invalidation but before classify. Clock normalization and lease acquisition can already have
effects, and `wait_host` can reap tickets or submit dependencies: preflight is
not described as globally side-effect-free.

After preflight succeeds, call the existing `invalidate_residency` immediately
before FFI submission. This clears append/resident/dense proof fields and advances
`map_generation`, including when conversion or classify subsequently fails.
This conservative early invalidation is an intentional private semantic change.
It never changes GPU maps, records, FIFO priority, free bits or traffic counts.
Do not retain stale proofs merely because the ledger remains zero.

**Native body.** Prepare two native-library bindings, one per allocator, before
pool/session use. Bind original native symbols with the existing C++-target
authentication. The input list has each semantic tensor once; keep distinct
counter slices as distinct views even when they share a backing slab. Use a CUDA
tensor first, followed by other inputs and the CPU ledger. Keep the explicit
device and FFI stream scope around this single call; no stream03 omission,
cached stream, Python stage callback or CUDA Graph is introduced.

The body invokes original classify, writes ledger `1` only after its successful
return, invokes original allocate/gather/publish, writes ledger `2` only after
publish returns successfully, then invokes original map. Reuse the original
complete bridge or its exact forwarding sequence without re-exporting tensors.
Its copy/pinned-pointer validation remains after allocation and publication
validation after gather. Original native error sites remain; Python preflight
does not imply every native error is detected before classify. No kernel,
selection, transport bound, record bytes or FIFO/tie policy changes.

**Finally and ownership.** The frame is local to the private `_ensure_from_topk`
caller and remains alive through its outer `operation().__exit__`. The ledger is
host-written, private to this invocation,
never passed to a CUDA kernel and not a device completion flag. Python `finally`
reads it once and commits `layer.clock = timestamp + progress` before outer
lease cleanup, including when native map fails. Validate the `0/1/2` domain and
require `2` on a normal native return. There is no second clock update elsewhere.
Native enqueue return does not establish asynchronous completion; even ledger
`0` can accompany a classify call that queued work before raising.
CPU/GPU clock equality is required on success. Failed targets are checked against
their actual submitted prefix, without guessing GPU progress from this ledger.

| Failure point | Final clock | Additional private state |
| --- | --- | --- |
| Python fact guard/allocation/host wait before invalidation | normalized timestamp | No recall launch; proofs remain as after lease acquisition. |
| FFI conversion, native metadata preflight or classify before successful return | timestamp | Residency proofs invalidated; partial enqueue remains possible after target entry. |
| Allocation, pointer validation, gather or publication before successful return | timestamp + 1 | Retain actual partial GPU state and all owners needed for cleanup. |
| Map after successful publication | timestamp + 2 | Publication returned; no second native recall. Enclosing request rollback/cleanup remains. |

Keep owner references through native return, ledger application and outer lease
cleanup. Preserve established same-stream allocation lifetimes and host-write
ticket ownership. On failure, the enclosing terminal cleanup drains as required;
if completion cannot be confirmed, retain the frame, cache/pool/session and
storage in the unsafe-owner mechanism and disable reuse. Preserve enclosing
request rollback/cleanup; do not add an unconditional success-path drain.
Re-raise the original exception object; ledger, stream or
drain failures are additional cleanup exceptions, never replacements. Native
forwarding must preserve the original TVM Error object too.
The implementation retains failed frames and poisons the pool. Private
`terminal_dispose` requires the exact private provider/pool, poisoned state,
zero host lease depth and exact standalone sessions without release guards.
It retains the complete owner graph, permanently closes the pool, drains the
entire device while preserving sync and restoration errors, then performs
irreversible CPU-only teardown. It never unpoisons, submits metadata kernels or
makes a request reusable. Any drain/restoration/teardown failure retains the
saved graph and blocks a repeated disposal attempt. A stale active identity at
depth zero is cleared only after successful teardown. Ordinary `close()` is not
used. CPU owners and mocked-CUDA failures passed; actual CUDA drain/restoration
acceptance remains a promotion gate.

## Executable implementation and acceptance plan

1. After root approval, freeze the current baseline and create a separate
   `/tmp/full_eager_recall_candidate_20261006_01/`. Add the private cache route,
   shared preflight, prepared factory and full-entry C++ forwarder. Adapt isolated
   `RecallFixture` wiring to bind the private `_ensure_from_topk` caller; keep public bodies and unrelated
   paths unchanged. Freeze the actual dependency closure and loaded DSOs, not
   just version strings. Review the 22-tensor/four-scalar ABI and original call counts.
2. Run narrow CPU checks with CUDA hidden, cores 32–39 and eight threads: fresh
   scalar and `requires_grad`/whole-storage facts; ordinary vs
   overridden provider dispatch; each preflight failure; early invalidation;
   ledger clock application and primary-plus-cleanup exception identity.
   Existing staged tests remain valid for public/generic paths. New private
   expectations follow the table above, not their old wait-failure clock value.
3. Root builds the host-only forwarder and native probe. Counting native targets
   verify ABI values/pointers, exact order, both allocators, no callback and
   ledger writes after successful returns. The same-entry metadata probe covers
   shape/dtype/device/stride, offsets, empty geometry and fresh descriptor changes;
   no original target may run on invalid metadata. Inject original native errors at
   classify/allocate/gather/publish/map and compare TVM Error identity. Actual
   Torch conversion probes cover offsets, the CPU ledger, default/nondefault
   streams, different enclosing FFI stream/device and restoration on failure.
   Descriptor stubs alone do not validate real owner metadata or lifetime.
4. Run original kernels on independent real cache states. Save actual before
   and after records, page tables, maps, free bits, priorities, GPU/CPU clocks,
   counters, scratch, all CPU residency fields and host-write owners. Derive
   expected selected KV/traffic from saved logical selections and the actual
   before-state. Cover zero/all-padding IDs, duplicates, restored all-hit,
   partial/cold misses, transient suffix, generic offset record layouts,
   sole-session free-only and multi-session
   eviction/ties. Require free-first legal FIFO allocation, unchanged selected
   hits, byte-exact consumed KV and actual gather/recalled-byte accounting.
   Include pending host writes, stream changes, preflight failures and each
   native fault boundary; use separate processes for sticky CUDA failures.
   Check weak-reference/storage retention when cleanup fails. An independent
   reader reconstructs state and traffic; a receipt alone is insufficient.
5. Run the fixed independent recall check and screen in a new private harness:
   current production `RecallFixture`, H=65,536, A=128, chunk=1,024,
   P=NH=65,664, three captured layers and three states
   (`certified_resident`, `restored_all_hit`, `cold_sparse_miss`). Check reference,
   baseline and candidate with two independent resets each; save all 54
   before/after observations. Untimed route proof must observe one actual bound
   full entry for each uncertified call and none for certified resident.
   Native/profile proof retains the original stage/kernel sequence and traffic.
6. Separate timing requires that matching check: three warmups per arm/case,
   40 balanced AB/BA pairs, **720 invocations**, all samples retained. Time the
   complete `invoke()` plus synchronization, with enqueue reported separately.
   Include every per-call applicability check, preflight allocation, host wait,
   invalidation, ledger read and clock update. Reset/append/host drain used by
   the fixed fixture, setup, validation and route tracing stay outside timing;
   no sample filtering or certified-control subtraction. Root's observer uses
   the assigned GPU, CPUs 24–31, NUMA 0 and the standard environment. Save results
   under a fresh SSD run directory, with matching source/native/provider/input
   identities. Review pooled and both order-stratum paired wall deltas. Reject
   null/regressive results; no repeated unchanged screen or model run follows.
7. Only a convincing standalone win permits extending lifetime across the real
   enclosing model lease, validating that extension, then root's separate
   full-model check and fixed paired trial against current production. Require complete request
   outputs/state and full end-to-end benefit; standalone enqueue savings do not
   establish MFU or gap improvement. Production promotion would additionally
   require affected cache-manager/MFU/motivation and shared-model impact checks,
   fresh formal results and the project publication process. No existing report
   or production implementation changes during this private investigation.

The main risks are added preflight/ledger overhead, waiting earlier reducing
overlap, conservative proof invalidation affecting later work, and unsafe
generic-provider bypass. The specified independent screen measures the first
two; state/continuation checks and strict eligibility address the others.

## Current private source evidence

The private CPU evidence wrapper passed 43 checks: 31 dispatch/lifetime cases
and 12 terminal-disposal cases, with CUDA hidden and uninitialized, cores 32–39
and eight Torch threads. Scoped Ruff checks passed. The source checker confirms
all six production source hashes, exactly two isolated dispatch import rewrites,
only `_ensure_from_topk` changed in the copied cache, the byte-preserved original
complete bridge prefix and the 22-tensor/four-scalar forwarding order.

The authored descriptor probe uses native stage stubs that ignore forwarded
arguments. Its intended boundary is metadata rejection, stage order, progress
and native error identity; it does not verify exact ABI values or pointers.
Root's separate probes must establish real transport, offsets and stream use.
Native build/runtime evidence, original-kernel state/traffic acceptance and the
fixed standalone screen are separate from this Python/source freeze. No model
harness or production promotion is part of the current candidate delivery.
