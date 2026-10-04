# NOSA shared workspace operator checkpoint

Date: 2026-10-03. Scope: operator portion of
[the shared cache plan](nosa_shared_cache_implementation_plan.md), P1–P3.
This is engineering validation, not a model/checkpoint or performance report.

## Delivered interfaces

- `NosaAttentionWorkspace(max_queries, kv_heads, head_dim=128, *, device, dtype)`
  eagerly allocates bounded fallback/pages/members/counts. `slices(queries,
  kv_heads=None, *, device=None, dtype=None)` returns contiguous actual-shape
  views. Counts always reserves `2 * ceil(max_queries / 8) * kv_heads`; the
  actual view doubles only at 256 batches. `estimate_capacity_bytes(...)` is
  pure; `tensors()` enumerates complete storage and `capacity_bytes` inspects
  actual storage capacity.
- `NosaFetchWorkspace` adds `max_queries=None`, `bounded=False`, and
  `trace_capacity=0` (four-int64 rows). Bounded construction eagerly reserves
  scratch and trace. `reserve(queries, trace_rows=0)` never grows a bounded
  instance. Query/trace excess fails before preparation or launch.
  `estimate_trace_rows(C,Q,H)` is available both on the class and in its module;
  it includes page envelopes, math intervals and eight stripe rows per page.
  `estimate_capacity_bytes(C,H,D, *, dtype, query_tile_size=128,
  max_queries=None, trace_capacity=0)` includes all staging, metadata, scratch
  and trace. `tensors()` excludes transient statistic views.
- Device-only `api.py`, `_cuda.py`, `_fa3.py` accept optional keyword
  `workspace`. Shared CUDA requires native SM90/BF16/D128/GQA16 and matching
  K/V strides. Disabled native, incompatible layouts/dtype, excess q/H and
  graph capture fail instead of selecting an unreserved fallback. Returned
  output remains an independent activation allocation.
- Both workspace classes permit explicit CPU storage layouts for accounting
  tests; fetch requires `bounded=True` for CPU construction. CPU workspaces
  cannot run attention. The model adapter must use its reference path before
  requesting the CUDA workspace/provider. Importing the new storage module or
  offload API does not import FA3, TVM FFI or Triton.
- Execution leases, cross-session ownership and final synchronization of the
  device-only provider are the backend resource owner's responsibility. Fetch
  retains its per-call completion event. No model, serving, or public cache
  files were changed by this operator task.

## Validation

GPU used: physical device 2, NVIDIA M403, capability `(9, 0)`. The initial GPU
attempt lacked `.venv/bin` in `PATH`, so extension compilation could not find
`ninja`; it did not validate numerical execution. With the existing environment
on `PATH`, the following completed successfully:

```bash
.venv/bin/python -m pytest \
  operators/nosa/attention/tests/test_workspace.py \
  operators/nosa/attention/offload/tests/test_nosa_offload.py \
  -k 'not cuda' -q -p no:cacheprovider
# 22 passed, 30 deselected (before extra bounded GPU parameterizations)

PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=2 \
.venv/bin/python -m pytest \
  operators/nosa/attention/tests/test_workspace_cuda.py \
  operators/nosa/attention/offload/tests/test_nosa_offload.py \
  -q -p no:cacheprovider --tb=short --maxfail=1
# 43 passed in 143.29 s, including native compilation
# Final suite after adding bounded serial and cross-stream cases:
# 47 passed in 3.48 s

PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=2 \
.venv/bin/python -m pytest \
  operators/nosa/attention/tests/test_workspace_cuda.py \
  -q -p no:cacheprovider --tb=short
# 3 passed in 2.00 s, repeated after final shared validation tightening

.venv/bin/python -m pytest operators/nosa/attention/tests/test_workspace.py \
  -q -p no:cacheprovider
# 16 passed in 2.74 s
```

Ruff passes for all eight changed Python files. CPU tests independently sum
actual storage and prevent allocation/CUDA inspection during estimation;
they exercise exact q/H view shapes and rejection without mutation. GPU tests
cover `Q=1032`, actual `q=1024`, varying q/H including zero q, unchanged storage
pointers, shared resident results bitwise equal to the owned route, and only
the output allocation in the shared attention call. The existing full offload
suite passes; bounded parameterizations directly verify exact unique reads,
serial/overlap equality, tail/cross-stream behavior, poisoned metadata reset,
counts work-order boundaries, and trace reuse.
Trace tests verify page envelope start/end equals the min/max of nonempty copy
stripes; they do not establish a 90% overlap performance claim.

The parent integration must include `attention/tests/test_workspace.py` in CPU
collection and `attention/tests/test_workspace_cuda.py` in GPU collection.
No full 32-layer checkpoint or serving performance run was performed here.

### Resource integration review follow-up

Read-only review found that a failed staging drain originally skipped the
device drain and permitted direct resident session mutation after poisoning.
The integration owner repaired that resource code. Additional independent tests
were added in `models/nosa/tests/test_resource_drain.py` and
`models/nosa/tests/test_cache_allocation_audit.py`:

- Three staging/device drain failure combinations verify that both drains are
  attempted, shared/session allocation dependencies are waited, poisoned owner
  aliases and storage survive, and direct release/reset/truncate are rejected.
- For all four schemes, an allocation hook observes a newly allocated indexer
  byte slab before it replaces the old slab. It independently enumerates
  storage at that instant, checks both slabs against the session reservation,
  and checks that shared storage is counted once. These are actual CPU storage
  transitions at the checked-indexer reservation geometry, not native execution.
- An isolated GPU IndexerCache audit grows 1 MiB scratch to 3 MiB and checks
  allocated and reserved allocator peaks. The allocated peak includes both
  live slabs, while post-allocation storage contains only the replacement.
  No model weights, outputs or activations occur in this audit, so it does not
  confuse full-process allocation with cache tensor capacity.

The final resource code passes the two new files on CPU with `-k 'not cuda'`:
**8 passed, 1 deselected in 2.72 s**. Physical GPU 2 passes
`test_cache_allocation_audit.py -k cuda`: **1 passed, 4 deselected in 1.74 s**.
Ruff passes both files. These follow-up model tests are outside the operator
diff snapshot below and are collected by the existing recursive model test
entrypoints. Full model allocation and performance acceptance remains with
the integration owner.

## Default-path and experiment impact

No native CUDA source, attention mathematics, selection, mask, CIS, numerical
repair or scheduling order changed. The `workspace=None` device-only branch
retains the original allocation sequence and native/Triton dispatch. The
default unbounded offload branch retains lazy scratch/trace allocation and
the original native launch arguments/shapes. Its `capacity_bytes` now sums
full storage, which equals the former element-byte sum for these owning
tensors; no buffer is added to the default layout. The new bounds/slices are
active only for the explicit shared path. Python source identity changes cause
a new FA3/fused JIT key, even though CUDA sources are unchanged.

Under plan section 8.2 this operator change alone does not require replacement
of the valid standalone `nosa_offload_overlap` or default resident/indexer
reports. Their original run/source identity must remain explicit. GR serving
uses the new bounded path and requires the plan's full fresh correctness,
budget, profile and performance acceptance before replacing its old reports.
Any further integration change to default run/prepare/synchronization or
measurement semantics must revisit this impact assessment.

## Source identity

Working tree base: `1a9aa455ef101a64da255cb1828bfa8f6d9a3f0c`; the worktree has
other independent changes. Only this component's diff, including new files,
is saved in [nosa_shared_workspace.patch](nosa_shared_workspace.patch), SHA-256
`9ce98f69f4f2e48f18dc14f456c170d2f57d39f9b6cb0bef6b94a4e902c4fe09`.

| File relative to `operators/nosa/attention/` | SHA-256 |
|---|---|
| `workspace.py` | `3bfca7f13321a2e90a8f5942ac86956af62b1003d2967463c847908ad06fa301` |
| `offload/api.py` | `cfa95a8208219a915de450cd062387113bf7cfaedd48432e8044707dcb34d23a` |
| `device_only/api.py` | `a8c483019ae6632767e7c33c5ffd8d78b46f4f06b7d8387eaa59aa1950926d21` |
| `device_only/_cuda.py` | `2f9de5fdb75e3a87ee98e46ab93ddcd18b0482b872e5f1e114238510858e5742` |
| `device_only/_fa3.py` | `9ee7135e8778f8f07da35948f2581c56c3aa7d12cc2f990c65d4a0342a731e73` |
| `tests/test_workspace.py` | `f0d8c841b5637df412974e3ad909b1220a1197ad761f850aec66f419cbaaa2e5` |
| `tests/test_workspace_cuda.py` | `a838053d73c2ee49ff7e8b76fad5d830f2cad9314db539d3e8a13e1f04e170ae` |
| `offload/tests/test_nosa_offload.py` | `f9d200354c2bf45f67f91086db23ec46f00b66dab210716065e9f8ece22f64a7` |
