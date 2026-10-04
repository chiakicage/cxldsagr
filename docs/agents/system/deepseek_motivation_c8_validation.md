# C8 combined output, graph-memory and lifecycle validation

Date: 2026-10-04. Run identity:
`deepseek_c8_combined_validation_20261004_01`.

The integrated typed-I/O norm and packed MLP implementation passed the combined
checkpoint and lifecycle gate: **176 tests passed**, no failures or skips,
15 dependency deprecation warnings, **92.39 seconds**. This validates the
recorded C8 source at the geometries below. Formal C8 serving and profiler
measurements are deferred while the native CUDA activation quantizer is being
developed. C8 retains the compiled official linear-quantization helper and is
the frozen reference for that work.

The component records are the [norm checkpoint](../kda/deepseek_norm_io/checkpoint.md)
and [packed MLP checkpoint](../kda/deepseek_mlp_packing/checkpoint.md).
Their component timing results do not establish a serving speedup. A separate
broad synthetic MLP suite encountered Dynamo's recompilation limit after
23 passes and 14 failures; that attempt was not a passing numerical gate.
The test-isolation follow-up and its acceptance are separate from this run.

## Execution and evidence identity

Root ran the combined driver in an exclusive physical GPU 2 window, exec
session 8306. Driver PID 1882205 and pytest PID 1882265 exited 0. The driver
measured **94.669962 seconds** for the pytest process lifetime, separately from
the pytest duration above. Stderr was empty. Hardware was NVIDIA H200,
SM90/Hopper, with 150,121,545,728 device bytes. The recorded environment used
`/preset-models`, PyTorch 2.12.1+cu130 and CUDA runtime 13.0.

The real-checkpoint workload has ten independent copies of the first three
dense blocks, using each source block's corresponding hidden/residual inputs.
It includes embedding, final norm and candidate last-token logits. It is the
checkpoint workload surrogate, not a trained ten-layer model or full
61-layer DeepSeek validation.

The original evidence is in
`/tmp/deepseek_c8_combined_validation_20261004_01/`: `run.py`,
`source_before.json`, `source_after.json`, `run_identity.json`, `stdout.log`
and `stderr.log`. The before/after manifests are identical and cover 119
source, test and configuration files, the native indexer build identity and
the typed norm's 380 dependency labels. They do not hash every installed
package or checkpoint tensor.

| Evidence | SHA-256 |
|---|---|
| Before/after source manifest | `2a90548f9f2235cf11fc55db5fb24a61822d04a3d0e743cb44b2dcf2a467e826` |
| Run identity | `7d1d7affb10c3a694dee1ace61c2367b6910a5f91f75edaad8da4413897fac44` |
| Stdout | `2df1a93521c9359cdba98847b9134809452ac2005be9a9758446235c5aa15761` |
| Independent saved-record audit | `369e3e105029ddcca5c5e691498ef64a019ac0196bafd5a81e32fa0ddca5ca19` |
| Typed norm fingerprint | `8b8a3e6128f23d6d65f6aca3993723c09c74fc9cf0c4da75f21742f1a5630b26` |

The independent CPU audit is `independent_combined_audit_graph_validate.py`
and `.json` in the same temporary directory. It rechecked source identities,
all four numerical/memory and all four runner completion records, memory
inequalities, exact graph-planning bounds and the matched C7 comparison.
It inspected the executed assertions and saved completion evidence; it did
not rerun the GPU tests or compare saved raw output tensors independently.
CUDA remained uninitialized.

At audit time, all recorded production sources and all ten executed test
files still matched. Two other files had later test-isolation edits:
`models/deepseek_v32/tests/test_packed_mlp.py` and
`operators/deepseek_v32/linear/tests/test_deepseek_linear.py`. Neither was
collected by this command. The audit records both old and current hashes.

After that audit, `models/deepseek_v32/official_serving.py` gained optional
common-compute callbacks and a new `test_official_serving.py` was added.
The later official checkpoint parametrization and bounded callback graph test
are also outside this C8 run. Its execution identity remains the
saved before/after manifests and frozen C8 source; the current official
adapter requires the separate gate in the
[official graph plan](deepseek_echo_official_graph_plan.md).

The reference snapshot is
`/tmp/deepseek-motivation-c8_torch_reference-frozen-rb0_2eqx/`.
The audit verified all 483 files in its manifest. It matches 112 of the
119 C8 manifest entries; five integration tests were not copied into the
snapshot and were instead verified against their unchanged current files,
while the two unexecuted tests above contain the later isolation edits.
The frozen manifest file SHA-256 is
`db23d88d3848013e3d5d7527611260d12b5209e7f6f9111467079d1e8c7e4231`.

## Output and lifecycle acceptance

The HBM, ECHO, serial sparse and dense prefetch schemes passed at
**H=P=65,536, NH=131,072, chunk=1,024**. Eager and captured backends
constructed prefixes from independent empty caches. A second user displaced
the retained offload history before the candidate comparisons. A=121 used
the eager fallback; A=128 used the captured compute graph bank.

The test compares every prefix hidden value between eager and captured
execution, every candidate hidden value and last-token logit between them,
and each offload candidate result against HBM at zero tolerance. It also
checks independent copied weights, copied source-block inputs, retained
prefix and hook outputs, unchanged host KV and indexer records, no candidate
D2H, stable graph identity and allocation size, and shared allocation within
the resource plan after each candidate.

A separate real-checkpoint runner check uses **H=2,304, P=NH=4,608,
chunk=256, A=16 and 23**. It covers uploaded token ownership after caller
mutation, both live-session storage audits, changed-A history reuse, history
replacement/revisit classification, candidate failure release and retained
ownership after a synthetic graph-completion failure. These are not H64K
failure-injection cases. Smaller graph tests separately check both residual
branches, caller-stream changes, delayed owned D2H, transient borrowing,
inactive graph segments and partial-bank retention after a capacity failure.

The command also runs the existing pool-prefetch/hint, transient candidate,
token input, backend, host allocation and storage-accounting checks. The
complete original command and environment remain in `run_identity.json`.
From the repository root, the selected pytest files were:

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

## Graph planning and measured storage

No graph-planner change is required for C8. The planner, allocation-bound
helper and graph tests are byte-identical to the
[matched C7 combined validation](deepseek_motivation_c7_hint_validation.md).
The static inputs remain hidden, start, positions and expanded attention
values, with incoming residuals for layers whose source index is nonzero.
C8 norm/MLP intermediate and output allocations occur inside capture and
are covered by the graph-private pool audit.

For ten blocks, H=65,536, chunk=1,024 and A=128, the allocation-free plan is:

| Quantity | Value |
|---|---:|
| Captured query sizes | 128 and 1,024 |
| Graphs | 40 |
| Logical static input storage | 641,820,832 B |
| Static input allocation upper bound | 696,356,864 B |
| Chosen graph-private upper bound | 12,884,901,888 B (12 GiB) |
| Combined graph reservation | 13,581,258,752 B |

The private bound remains a chosen upper limit, not an observed fixed
overhead. Allocation checks it against free HBM, then audits after each
graph pair and after final synchronization. The audit counts total private
segment capacity, including inactive blocks, and deduplicates graph pools.
Static storage is charged through its owning allocator blocks; captured
outputs are not added to static storage again. The backend charges this
shared graph bank once, without a per-session multiplier.

The following are setup observations in bytes. Each process holds eager
and captured backends, diagnostic tensors and allocator caches. They are
not isolated single-backend peaks or acceptance at formal NH=16,777,216.

| Scheme | Graph-private reserved | Static allocated | PyTorch allocated | PyTorch reserved | Device used |
|---|---:|---:|---:|---:|---:|
| HBM | 5,771,362,304 | 644,534,272 | 23,635,700,736 | 27,197,964,288 | 27,993,899,008 |
| ECHO | 5,771,362,304 | 643,958,784 | 26,588,687,872 | 30,893,146,112 | 31,699,566,592 |
| Serial sparse | 5,771,362,304 | 643,773,952 | 26,512,380,416 | 30,547,116,032 | 31,355,633,664 |
| Dense prefetch | 5,771,362,304 | 643,773,952 | 26,512,183,296 | 30,379,343,872 | 31,187,861,504 |

Graph-private reserved storage decreased from **7,176,454,144 B** in the
matched C7 combined run to **5,771,362,304 B** in C8, a reduction of
**1,405,091,840 B (1.30859375 GiB, 19.58%)**. Both runs use the same test
command, geometry and graph-test source. This comparison applies to the
combined C8 changes and does not isolate an individual component's effect.
Process allocated/reserved/device-used observations remain distinct. The
reduction does not justify lowering the chosen 12 GiB bound, changing P/NH
admission, or claiming formal serving latency, MFU or capacity improvement.

The original C7 formal report and its valid run artifacts remain in place
until a newly validated implementation receives a new formal run and report.
