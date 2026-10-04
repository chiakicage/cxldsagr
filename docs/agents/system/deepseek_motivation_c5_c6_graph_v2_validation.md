# C5, C6 and graph policy v2 combined validation

Date: 2026-10-04. Validation identity: `20261004-c5-c6-graph-v2`.

This checkpoint validates the combined output-layout change, bounded native
sparse recall and `deepseek-compute-islands-v2` graph policy. It records
correctness, storage ownership and failure handling. It does not publish
serving latency, MFU or full-NH capacity acceptance. The earlier
[C4 validation](deepseek_motivation_combined_validation.md) remains a record
of that earlier implementation and is not evidence for graph policy v2.

## Result

The combined GPU command completed **86 passed**, with no failures or skips,
in **86.87 seconds**. Fifteen dependency deprecation warnings concerned
CUTLASS `Arch` and PyTorch `torch.jit.script_method`.

All four schemes passed the real-checkpoint comparison at H=P=65,536,
NH=131,072, C=1,024 and A=121/128. A=121 deliberately used the eager fallback;
A=128 used captured compute graphs. Independent empty eager and graph
histories produced exactly equal full prefix/candidate hidden states and
candidate logits. Interleaved users, eviction, retained KV/indexer/offset
state, zero candidate D2H, source-copy independence, returned tensor
lifetime, graph identity and shared-memory stability checks all passed.

The real-checkpoint runner test also passed for all four schemes at H=2,304,
P=NH=4,608, C=256 and A=16/23. It verified uploaded input ownership after
caller/CPU tensor overwrite, both full live-session storage audits, changed-A
history reuse, changed-history rebuild with revisit classification, candidate
failure release, and poisoned graph-completion ownership. The synthetic
Event-constructor failure retains graph buffers, the active session and the
admission owner, and rejects unsafe reuse/release. Teardown synchronizes the
healthy device and resets only fixture-injected poison; this does not establish
recovery from a destructive CUDA fault.

Nine directly involved production files and the changed graph test had
identical SHA-256 hashes before and after the command. GPU 0 was released;
the post-run observation was 1 MiB used and 0% utilization.

## Graph v2 review and added coverage

The source review found no unresolved blocker in capture lifetime, stream
ordering or storage accounting:

- Projection warmups discard their outputs. Only projection capture binds
  `pair.saved`; projection replay initializes that tensor before finish
  warmup/capture. A residual-present saved tensor belongs to the projection
  private pool. Without incoming residual, it aliases static hidden.
- Finish uses the captured saved tensor and a contiguous static expanded
  value input. Eager `_expand_values(..., out=expanded)` runs before finish
  replay on the backend execution stream. Existing execution events retain
  the dependency when a later execution switches streams.
- Static planning removes the old saved and absorbed-attention inputs.
  Private-segment accounting includes the projection-owned saved allocation
  through its owning pool, including inactive segment capacity.
- Persistent offload KV still receives an owned clone before asynchronous
  D2H; transient candidates may borrow projection outputs.
- The matrix instrumentation wrapper forwards `torch.bmm` keyword arguments,
  including `out`, and labels the value-weight BMM as `v_expand`. The new
  value-expansion scope is outside replay-template matching, and dynamic
  finish capture contains no expansion BMM. This is source-level review;
  profiler attribution acceptance is recorded separately by its owner.

The existing delayed-writeback graph test is now parametrized with and
without incoming residual. The residual-present fixture independently
allocates two dense blocks from its small checkpoint so layer 1 exercises
the captured residual branch. It verifies:

- `pair.saved` has exactly one active owning allocation in the projection
  private pool, is absent from static input allocations and is not owned by
  the finish pool;
- exact saved values after changed hidden/residual inputs, stable tensor
  identity and storage address across replays and a stream switch;
- exact finish outputs, delayed owned D2H, transient borrowing and unchanged
  graph capacity after replay;
- private reserved bytes equal the sum of the unique graph-pool segments,
  and static logical bytes equal the allocation-free plan.

Saved-value snapshots are cloned on the caller stream and compared after
the delayed copy completes, preserving the intended D2H/replay overlap in
the fixture. This is a correctness test, not an overlap-performance result.

## Command and environment

The command ran from the repository root on physical GPU 0, NVIDIA M403,
SM90/Hopper, with 150,121,545,728 device bytes and checkpoint `/preset-models`.
The environment was Torch 2.12.1+cu130, CUDA 13.0, driver 570.124.06,
FlashInfer 0.6.18, DeepGEMM 2.8.1+057ca59 and FlashMLA 1.0.0+ba89a34.
No test result or profiler artifact was written into experiment or test
directories.

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
DEEPSEEK_GRAPH_CHECKPOINT=/preset-models DEEPSEEK_GRAPH_HISTORY=65536 \
DEEPSEEK_GRAPH_CHUNK_SIZE=1024 DEEPSEEK_GRAPH_CANDIDATE=128 \
.venv/bin/python -m pytest -s -q \
  models/deepseek_v32/tests/test_compute_graphs.py \
  serving/tests/test_token_input.py \
  models/deepseek_v32/tests/test_serving_backend.py \
  tests/integration/test_storage_accounting.py \
  cache/tests/test_sparse_token_pool.py cache/tests/test_host_allocation.py
```

Ruff check and format check passed for the changed test file. The root's
broader CPU regression and the C6 owner's operator/cache tests are separate
checks and are not included in the 86-pass count above.

## Source identities

| File | SHA-256 before and after validation |
|---|---|
| `serving/persistent.py` | `e07b33d5192372b64042f650c308baf99a9e944d3ba7ed54abc597162fad63f7` |
| `cache/sparse_token_pool.py` | `202ba4352118055f2ad3c2e33236487d7f6edd04eb49e5c2cdf2e751deeb2974` |
| `cache/sparse_token_cache.py` | `011e6990e9a496b1f6af5ea1ae0afc1d466731ca370978d98daa6f8218d5abfb` |
| `models/deepseek_v32/serving_backend.py` | `23d63e3fd08b63a9c7afb4fab554deb6f426ffecac22b37939c10cf198d2f67a` |
| `models/deepseek_v32/compute_graphs.py` | `95432e0d32b1f7e52648b5abc17348a8b09638abb5b4b9cfe46137fabf2bfe3b` |
| `models/deepseek_v32/echo_model.py` | `7ccb7ffb0445483569d75074749d1dd4099134288827f3e1f77ed7973ed42e71` |
| `operators/deepseek_v32/indexer/echo.py` | `30492794f10c4bd34d730c0bb926c79827095feb59e7cad35765ff0bced4c219` |
| `operators/deepseek_v32/indexer/cache_ops.py` | `8df0c741ecc343537729f5862790b9f3ac1520e9da792aef5ce6e2b22bad0224` |
| `operators/deepseek_v32/indexer/csrc/echo_sparse_recall.cuh` | `ddad6791b1205aa931689805018ff86a285601788c620fe4fc592846e74fd39e` |
| `models/deepseek_v32/tests/test_compute_graphs.py` | `6328a344acb40bad6901c0fc3b81112548a68b7aab5c9ce7ab785ada0edaf2e1` |

These identities do not cover every transitive dependency or checkpoint
tensor. Formal performance publication requires its own source snapshot
and run ID.

## Large-shape memory observations

All values are bytes. Graph-private accounting includes inactive private
segments; static inputs are charged separately at actual allocator block
capacity. The values remained stable across interleaved users and both
candidate shapes.

| Scheme | Private reserved | Static allocated | PyTorch allocated at setup | PyTorch reserved at setup | Device used at setup |
|---|---:|---:|---:|---:|---:|
| HBM-only | 7,176,454,144 | 644,534,272 | 23,626,525,696 | 28,603,056,128 | 29,398,990,848 |
| ECHO | 7,176,454,144 | 644,204,544 | 26,579,250,688 | 32,069,648,384 | 32,876,068,864 |
| Serial sparse | 7,176,454,144 | 643,887,104 | 26,503,221,760 | 31,677,480,960 | 32,485,998,592 |
| Dense prefetch | 7,176,454,144 | 644,036,096 | 26,503,289,856 | 31,962,693,632 | 32,771,211,264 |

These are snapshots from a correctness process holding eager and graph
backends, diagnostic tensors and allocator caches. They are not isolated
single-backend peaks or physical capacity acceptance at NH=16,777,216.
The 12 GiB graph-private limit remains a chosen upper bound, not an observed
fixed reservation. No latency or MFU conclusion follows from this validation.
