# NOSA persistent GR serving adapter

## Scope and interface

`models/nosa/serving.py` provides `NosaServingBackend(model, scheme,
chunk_size=1024)` and strict `from_pretrained(checkpoint, scheme=..., device=...,
max_seq_len=...)`. The latter retains all 32 NOSA-8B layers and all checkpoint
parameters, including CIS A/delta. It returns all normalized candidate hidden
states, without the vocabulary projection or autoregressive decode.

The backend implements `executor/serving_backend.py`. Root serving owns the
cross-request LRU and admission budgets. A retained session has a committed
stable prefix; candidate suffixes are truncated before the next request.
Eviction discards the complete user's HBM and CPU backing. There is no tier
migration or hidden snapshot copying on a cache hit.

## Schemes

- `hbm`: `NosaKVCache`, with complete K/V/CIS and derived indexer records in HBM.
- `serial_sparse`: `NosaOffloadCache(overlap=False)`, one complete query batch's
  exact sparse union fetched before the existing FA3 attention.
- `overlap`: existing cooperative native `NosaFetchWorkspace(overlap=True)`;
  no shared kernel implementation changes.
- `dense_prefetch`: `NosaDensePrefetchCache`, retaining full historical K/V in
  pinned DRAM and two full logical layer slots in HBM. Layer zero's history is
  queued at append begin. After the current layer writes its suffix, the
  current stream waits for history readiness and the next layer's full history
  copy is queued on a separate copy stream. Copy stream dependencies protect
  the alternating slot from earlier attention consumers. The next layer copy
  may overlap current-layer indexing, attention and MLP. Selection, causal mask,
  CIS and sparse attention arithmetic remain the same as the other schemes.

Dense prefetch deliberately does not instantiate the sparse fetch workspace.
It records `last_prefetch_bytes` as all layers' historical K+V payload; this is
a logical payload counter, not measured PCIe link traffic.

## Cache budget accounting

Admission is bounded by `estimate_session_bytes(capacity, prefix_tokens)`.
It includes full-capacity K/V/CIS, all compressed K/CIS and pooled CIS buffers,
native indexer scratch, per-user layer staging, sparse metadata and native FA3
scratch, plus every layer's largest owned pending K/V append. Scratch growth
reserves both old and new allocations during replacement. All users are
charged their own buffers; staging is not silently shared across users.

The scope excludes weights, hidden activations, and operator temporary outputs.
`session_bytes()` returns actual owned cache tensor allocations, not process
peak GPU memory. HBM-only uses no CPU KV backing and leaves its equal allowed
CPU DRAM budget unused. Small host synchronization flags are execution control,
not CPU KV cache capacity.

## Validation status

2026-10-02: CPU tests passed (10 passed, one opt-in checkpoint test skipped).
These cover four independent-prefix policies, candidate revisit rollback,
failure/retry with preserved prefix, double-buffer layer ordering, and cache
reservation upper bounds. Ruff check/format passed for both new source files.
Single-GPU SM90 full-checkpoint validation passed at 8192+128 (11.12 seconds)
and 65536+1024 (23.52 seconds). Each geometry used four independently built
sparse prefixes and two different candidate visits. Every candidate hidden
element from serial sparse, dense prefetch, and overlap was bitwise equal to
HBM, with max_abs=0. These durations describe test execution, not serving latency.
No serving performance or overlap conclusion is claimed here.

Checkpoint location: `/mnt/ssd-wlcb/chenkaiqi/NOSA-8B`.

Correctness command, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=1 PATH="$PWD/.venv/bin:/usr/local/cuda/bin:$PATH" \
  NOSA_SERVING_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
  .venv/bin/python -m pytest -s -q -p no:cacheprovider \
  models/nosa/tests/test_serving.py::test_cuda_serving_checkpoint_independent_prefixes_and_revisits
```

Default GPU correctness geometry is 8192 + 128, two different suffix visits.
`NOSA_SERVING_PREFIX_TOKENS` and `NOSA_SERVING_SUFFIX_TOKENS` select the full
64K + 1K check. Each policy constructs its own sparse prefix from empty cache;
all candidate hidden elements are compared against HBM. This is a code check,
not an experiment result or a performance run.

## Dense synchronization and budget review

Read-only review after the successful GPU checks found the following ordering:

1. `begin_step` queues layer 0 history before the model starts its layers.
2. `write_layer(i)` first publishes owned suffix records. The current compute
   stream then waits for the ready event associated with slot `i % 2`, and
   writes this layer's suffix to the same slot.
3. `_prefetch(i + 1)` makes the copy stream wait for the current stream's work
   submitted so far. That includes the previous consumer of slot `(i + 1) % 2`,
   so the copy cannot overwrite a slot still used by an earlier attention.
   Current-layer indexer/attention/MLP are submitted after this dependency,
   allowing the next-layer copy to overlap them without forming a stream cycle.
4. Attention consumes only its matching staged layer. Commit, abort, truncate,
   and release synchronize before backing or staging can be reused or freed.

CUDA correctness covered all 32 layers, alternating both slots repeatedly,
multiple prefix chunks, and candidate rollback. CPU failure injection covered
abort/retry. There is no dedicated per-copy timing evidence yet.

The reviewed estimate covers the complete cache-owned tensor set: HBM K/V
or pinned K/V backing; resident CIS; per-layer compressed K/CIS and pooled CIS;
indexer score/validation/ranking workspace; double dense staging or sparse
staging, first-use/ready/queue/tile counters and native FA3 scratch; and all
layers' maximum pending suffix K/V clones. Lazy scratch growth reserves both
old and new capacities. Optional profiling trace storage is excluded from
formal latency allocations because profiling is disabled there; a separate
profile must report its added storage and must not reuse the formal budget
occupancy numbers as a claim about the profiled process.

Source freeze: `models/nosa/serving.py` and its tests are final. Subsequent
changes in this handoff are documentation only while formal serving runs.

## Post-run native work-interval checks

The current rerun uses 16 synthetic users visited in ID order twice, 32 requests,
without a heat distribution. Seed 42 controls content only. Cache budgets are
4 GiB HBM / 64 GiB DRAM. The current user priority is 65536 + 128; do not start
other lengths or restart the stopped industrial 1024/4096 run before this group.
The new formal run and its diagnostic are not yet measured as of this source
update. Prior accepted results remain in the experiment until replacements
have passed the complete numerical and provenance checks.

Run `experiments/gr_serving/scripts/profile.sh` only after the matching formal
latency process finishes and its run is accepted. Use a distinct run ID and the
same physical SM90 GPU. Pass `--latency-data` for that run and `--num-users 16`.
The entry point reads prefix, suffix and chunk size from accepted metadata.
It authenticates every saved token row and workload identity in request order,
using a stream so a 4096-request, 64K-history trace is not retained in memory.
Only the first three real rows with `is_revisit == True` are retained for
execution: request IDs 16/17/18 in the current sequential case. Repeated users are allowed, and every later trace row is still
verified. Formal measurement rows must cover all saved request IDs once per
scheme; profile metadata copies the exact configuration, observed counts,
heat identity and access-trace identity.

For each selected request, preserve its original request, user and visit IDs,
input/prefix/candidate hashes, saved tokens, and formal HBM hidden reference.
Each `serial_sparse` and `overlap` sample constructs its complete sparse prefix
from an independently empty session, performs an unprofiled candidate control,
truncates to the same prefix, and executes the profiled candidate through all
32 layers. Its new session is a diagnostic reconstruction, not an observed
formal LRU hit. All candidate hidden elements must exactly match both the
unprofiled control and the saved formal HBM result.

Only actual layer-31 attention is instrumented. The wrapper retains consumed
selection IDs/masks, enables native schema-3 work tracing around the original
attention call, synchronizes, and restores the wrapper and tracing flag in
`finally`. Layer 31 must be the last attention call. Export selections and raw
work intervals before any subsequent shared-workspace use. There are three
requests times two modes, with one sample each, yielding six work records.
Independent serial/overlap executions must consume exactly the same selection.
The profiler saves complete control/profiled hidden tensors, selection tensors,
raw native traces, original request identities, numerical attestations,
source/build/checkpoint identities, and the extra profile storage allocation.

Existing `expected_fetch_rows`, `prefix_transfer_bytes`, `new_work_profile`,
`native_build_metadata`, `work_interval_metrics`, `stripe_interval_metrics`, and
`work_profile_metadata` helpers from `nosa_offload_overlap` retain their complete
checks. They authenticate the unique historical page union, first-read bytes
per 128-query statistics group, every nonempty 8-token stripe, and exact page
envelope = min(stripe start)/max(stripe end). Overlap ratios intersect independent
page-envelope and stripe-copy unions with actual consumer softmax intervals,
never the fused kernel's full execution window. Each of the three overlap
samples must satisfy both ratios >= 0.9 before stating a 90% claim. Valid
below-threshold samples remain published. Serial traces retain page-copy
windows and report unmeasured math overlap as null. These intervals do not
measure physical link traffic or full-attention latency hiding.

Native schema-3 evidence is required. Optional `--nsys` adds a supplemental
CUDA/NVTX capture and SQLite export; the existing overlap SQLite CLI expects a
third resident mode and is not used for this two-mode diagnostic. Publication
uses `--verify-only` to independently reanalyze raw intervals, selections and
complete hidden tensors before copying accepted results into the experiment.
Overall latency and budget-occupancy conclusions come from the separate
uninstrumented formal run.

Source update validation: the profile module and its CPU tests were adapted for
streaming 4096 requests and rejecting duplicate or incomplete formal trace IDs.
No model, cache or kernel implementation changes were made by this update.
The GPU diagnostic remains pending the new accepted formal run.
