# Combined C7a and ECHO hint validation

Date: 2026-10-04. Validation identity:
`deepseek_c7_hint_combined_validation_20261004_01`.

The combined C7a dense-prefetch metadata and integrated exact ECHO hint passed
**176 tests**, with no failures or skips, in **93.97 seconds**. The process
reported 15 dependency deprecation warnings. This is model/output, storage and
lifecycle acceptance; it does not establish serving latency, MFU, lookahead
overlap or capacity at the formal NH=16,777,216 workload.

## Execution and source identity

Root granted an exclusive physical GPU 2 window after the hint component gate.
Exec session 52237, pytest PID 1804770 and driver PID 1804754 all exited 0.
The driver measured 96.393 seconds for pytest process lifetime, separately from
the pytest-reported duration above. CUDA was explicitly released to root after
completion and source/measurement checks. Stderr was empty.

Hardware was NVIDIA H200, SM90/Hopper, with 150,121,545,728 device bytes.
The selected checkpoint path was `/preset-models`; PyTorch was 2.12.1+cu130
and CUDA was 13.0. The real-checkpoint workload uses ten independent copies of
the first three dense blocks with their corresponding source-block inputs,
embedding, final norm and candidate last-token logits. It is the project's
checkpoint workload surrogate, not a trained ten-layer model or full 61-layer
DeepSeek validation.

The CPU-only driver recorded 109 source, test and configuration files before
launch and after completion, including the new hint module/test and all test
files below. Every hash matched. It also rechecked the full native build info,
including its shared-header hashes. The manifests do not hash every installed
vendor package or checkpoint tensor.

- Before/after manifest SHA-256:
  `66fc72be416d8e9393806aa0de16b64a819e601d840966ec75527f5b81f12dc2`.
- Native build-info SHA-256:
  `c2b3905b211ef59fe56032de605818aa877acc403431dcd2cbc6c45703ed23bf`.
- Temporary evidence directory:
  `/tmp/deepseek_c7_hint_combined_validation_20261004_01/`.
- Evidence files: `run.py`, `source_before.json`, `source_after.json`,
  `run_identity.json`, `stdout.log` and `stderr.log` in that directory.

| File | SHA-256 before and after |
|---|---|
| `models/deepseek_v32/pool_prefetch.py` | `f27260feaa6b21d302232e29325a461a31d158d6ef8671f7dfa164c2c3a62a7b` |
| `operators/deepseek_v32/indexer/cache_ops.py` | `9283837189aeb5b8a0e55e2e6e6a25e24e08a1bb2f545edf2bf11fec2b93175c` |
| `operators/deepseek_v32/indexer/csrc/echo_dense_prefetch.cuh` | `30374a97873ecd7b7d87730ba40c400e111d0e3ba32feee3982efc8c508bb59d` |
| `operators/deepseek_v32/indexer/csrc/echo_indexer.cu` | `86793e7375d8204a1fe445f42ecb4efd35d7b6083fa82ee2bf4e3e0a12da3eab` |
| `operators/deepseek_v32/indexer/prefetch_hint.py` | `0ba70bdc2daf4be2c385b60af69309974394ab8d237f9aaf5519fb9350b6a07c` |
| `operators/deepseek_v32/indexer/tests/test_prefetch_hint.py` | `c1c8cd9e63f0c86f1a5539ead8cea2874f5d688f1e06e23dea1164a5f6c8d267` |
| `models/deepseek_v32/echo_attention.py` | `cddf8aef94f2c3f3520bc8152aea6eaed4411ad9b5d6ad26c0303d2facd1766e` |
| `models/deepseek_v32/serving_backend.py` | `23d63e3fd08b63a9c7afb4fab554deb6f426ffecac22b37939c10cf198d2f67a` |
| `models/deepseek_v32/compute_graphs.py` | `95432e0d32b1f7e52648b5abc17348a8b09638abb5b4b9cfe46137fabf2bfe3b` |
| `models/deepseek_v32/tests/test_compute_graphs.py` | `6328a344acb40bad6901c0fc3b81112548a68b7aab5c9ce7ab785ada0edaf2e1` |
| `serving/persistent.py` | `e07b33d5192372b64042f650c308baf99a9e944d3ba7ed54abc597162fad63f7` |
| `cache/sparse_token_pool.py` | `202ba4352118055f2ad3c2e33236487d7f6edd04eb49e5c2cdf2e751deeb2974` |
| `cache/sparse_token_cache.py` | `011e6990e9a496b1f6af5ea1ae0afc1d466731ca370978d98daa6f8218d5abfb` |

No production or test file was edited during this command. C7a's unchanged
generic transport is covered in its [component checkpoint](../kda/deepseek_dense_prefetch/checkpoint.md).

## Model and lifecycle acceptance

All four schemes passed at H=P=65,536, NH=131,072 and C=1,024. Eager and graph
backends constructed their histories from independent empty caches, then an
interleaved user displaced the retained offload history. For each scheme,
A=121 used eager fallback and A=128 used captured compute graphs. Every
candidate hidden value and last-token logit matched HBM at zero tolerance;
full prefix outputs matched between eager and graph execution.

The same test verifies source-copy independence, retained host KV/indexer
state, zero candidate D2H, returned prefix/hook-output lifetime, stable graph
identity and shared allocation. The hint module's 43 tests also passed,
including byte-exact filtering/reduction outputs and score/offset ownership,
subnormals, nonfinite/overflow cases and stream ordering.

A separate real-checkpoint runner check passed for all four schemes at
H=2304/C=256/A=16,23. It checks uploaded input ownership after caller mutation,
both live-session storage audits, history reuse when A changes, rebuild and
revisit classification when history changes, candidate failure release and
retained admission/session/buffer ownership after synthetic graph completion
failure. The graph borrowed-output checks cover both residual branches,
changed caller streams, delayed owned D2H and transient borrowing. Their
smaller geometries are not presented as H64K failure-injection cases.

The fixed-pool C7a tests verify owned private ticket IDs during a deliberately
pending copy and shared-scratch overwrite, exact pool/maps/clocks/counters,
candidate-tail preservation, fragmented/partial-free cases, empty/all-hit and
clock rollover. Reservation, enqueued-copy and metadata/copy-ready event
constructor/record failures preserve copy-input lifetime until drain and
prevent unsafe reuse. Additional backend/transient/token/storage tests retain
the existing public lifecycle and hard-reservation requirements.

## Command

The driver executed this pytest command from the repository root, with the
same environment below. The shell called the driver to save before/after
manifests and preserve pytest's exit status; it redirected only to the fresh
temporary evidence directory.

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

## Memory observation boundary

The following setup snapshots are bytes. Each process holds eager and graph
backends, diagnostic tensors and allocator caches; these are not isolated
single-backend peaks or full-NH capacity acceptance. Graph-private reserved
storage was 7,176,454,144 bytes for every scheme and stayed stable across the
interleaved users and both candidate lengths.

| Scheme | Static allocated | PyTorch allocated | PyTorch reserved | Device used |
|---|---:|---:|---:|---:|
| HBM | 644,534,272 | 23,626,525,696 | 28,603,056,128 | 29,398,990,848 |
| ECHO | 644,204,544 | 26,579,250,688 | 32,069,648,384 | 32,876,068,864 |
| Serial sparse | 643,641,344 | 26,502,464,512 | 31,700,549,632 | 32,509,067,264 |
| Dense prefetch | 643,922,944 | 26,503,140,864 | 32,052,871,168 | 32,861,388,800 |

The earlier C5/C6 graph-v2 validation remains evidence for its recorded source.
This combined result supplies the new C7a/hint correctness gate; formal
performance and report replacement still require root's new measured run.
