# C7a combined full-model validation plan

Status: executed after root's exclusive GPU 2 grant, with the integrated exact
ECHO hint and its targeted test module. The combined command passed 176 tests
with no failures or skips. See [the combined checkpoint](../../system/deepseek_motivation_c7_hint_validation.md)
for source identity and the actual coverage. This plan's command below now
includes the hint suite as executed. CUDA was released after completion.

## Command

From the repository root, after an explicit GPU grant:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 \
DEEPSEEK_GRAPH_CHECKPOINT=/preset-models DEEPSEEK_GRAPH_HISTORY=65536 \
DEEPSEEK_GRAPH_CHUNK_SIZE=1024 DEEPSEEK_GRAPH_CANDIDATE=128 \
.venv/bin/python -m pytest -s -q \
  models/deepseek_v32/tests/test_compute_graphs.py \
  models/deepseek_v32/tests/test_pool_prefetch.py \
  models/deepseek_v32/tests/test_pool_prefetch_native.py \
  operators/deepseek_v32/indexer/tests/test_prefetch_hint.py \
  serving/tests/test_token_input.py serving/tests/test_transient_candidate.py \
  models/deepseek_v32/tests/test_serving_backend.py \
  tests/integration/test_storage_accounting.py \
  cache/tests/test_sparse_token_pool.py cache/tests/test_host_allocation.py
```

The physical device selector may be changed only to the device assigned by root;
all tests still address the selected device as `cuda:0`. Explicit checkpoint
selection must fail if hardware/dependencies are absent. No result artifacts
belong in tests or experiments. If temporary logging is useful, use fresh `/tmp`
paths and preserve pytest's exit status.

## Coverage and boundaries

`test_checkpoint_graphs_match_eager_across_users_and_candidate_shapes` loads
real checkpoint tensors into ten independent dense-block copies, preserving
the project's copied source-input semantics. It tests all four schemes with
H=P=65536, NH=131072, C=1024, and A=121/128. It builds independent empty eager
and graph histories, interleaves another user to evict offloaded history, then
checks all candidate hidden values and last-token logits exactly against HBM.
A121 deliberately selects eager fallback; A128 uses captured compute graphs.
It also checks returned-prefix/hook-output lifetime, separate block weights,
retained host KV/indexer state, zero candidate D2H, graph identity and unchanged
shared allocation. These are workload-surrogate checks, not complete 61-layer
DeepSeek validation or a full-NH formal serving trajectory.

The same module's packed-admission runner check uses H2304/C256/A16,23 and all
four schemes. It verifies owned uploaded inputs after caller mutation, full
live-session storage accounting, same-history reuse across candidate lengths,
changed-history rebuild/revisit classification, candidate failure release and
poisoned graph-completion retention of session/buffers/admission owner. Its
borrowed-output test covers both residual branches, stream changes, delayed
owned D2H and transient borrowing. These smaller lifecycle geometries are
reported separately from the H64K numerical comparison.

The two prefetch modules cover fixed-pool C7a private ticket ownership and exact
FIFO/map/clock behavior. Delayed copies must still be pending during shared
scratch overwrite; full-state and candidate-tail checks follow the wait.
Native reservation, copy submission, metadata-ready and copy-ready event
constructor/record failures must disable reuse and retain enqueued copy inputs
until drain. Foreign, expired and unconsumed tickets, empty/all-hit/rollover,
fragmented pages and partial-free slots are included. The serving-backend tests
also exercise speculative-copy drain and poisoned borrowed staging ownership.

The ordinary token-input, transient-candidate and storage-accounting checks
retain the public lifecycle and hard-reservation contract. Their passes cannot
substitute for real-checkpoint output comparison.

## Evidence gate

Before launch, save a CPU-only source manifest for the final combined code,
including the 16 C7a component sources, `serving/persistent.py`,
`models/deepseek_v32/serving_backend.py`, `compute_graphs.py`, `echo_attention.py`,
`echo_model.py`, and all test files in the command. Include new hint source/tests
if integrated. Recheck all hashes after completion. Preserve the native build
fingerprint and checkpoint path separately. A test-file edit after collection
requires an explicit scoped rerun; do not combine counts into an unrun suite.

Accept only a zero exit with the required GPU/checkpoint cases actually run,
exact complete outputs and lifecycle/storage assertions passed. Record test
counts, runtime, source identity and actual device/dependencies. Release CUDA
explicitly after the process exits. Formal performance requires a fresh run ID,
matching attribution and independent output/measurement audit owned by root.
