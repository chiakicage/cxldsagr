# C3 metadata checkpoint

Component result: C3b is promoted for integrated screening. This does not prove
the user's complete first-visit latency or end-to-end MFU objective. Root owns
complete-request, full-trajectory, publication and affected-experiment reruns.
Production source frozen for root's integrated screen on 2026-10-04.

## Implementation

The starting tree already contained C3 fused planned append, resident selection,
dense history protection, private top-k dispatch, a shared exact-union bitmap,
and per-session native counters. Those paths had no C3-specific evidence.
This work reviewed and validated them, then changed `echo_resident.cuh` to use
CTA-local bitmap aggregation for selections with at least 1M entries and bitmaps
of at most 4096 words (16 KiB). All other shapes retain the global bitmap path.
Each CTA merges nonzero words with atomicOr, counts only newly claimed bits,
and only the winning CTA stamps each historical slot. Candidate bits contribute
to the union without owning priority or mapping state. This preserves exact
selection, logical-to-physical mapping, FIFO events, and unique counters.

No new persistent storage is added by C3b. The existing C3 bitmap/count and
56-byte per-layer session counter slab are included in pool accounting and
`models/deepseek_v32/capacity.py`; views are measured by their complete storage.
`echo.build_info()` fingerprints `echo_*.cu*`, covering the changed header.
The public arbitrary-ID ensure remains checked; the model consumes trusted
exact-top-k through the private entry, guarded by conservative residency proof.

Files changed during this subtask: `echo_resident.cuh`, new
`cache/tests/test_resident_metadata.py`, the dense-helper unit case in
`models/deepseek_v32/tests/test_pool_prefetch.py`, and these KDA documents. The preexisting
cache/wrapper/prefetch scaffolding was validated without additional edits.

## Correctness

67 tests passed on GPU 2, including the existing cache/pool/transient/prefetch
and native metadata/transport tests plus eight new C3 differential cases.
The new cases compare records, both maps, priorities, free bitmap, GPU and host
clocks, and all cache metrics against checked operations. Coverage includes
uint8 width 7, FP32 width 7, BF16 width 576; two layers and three owners;
fragmented host pages; duplicates and negative padding; empty selections;
transient candidates; rollback; competing-user fallback; public invalid IDs;
clock values LIMIT-2, LIMIT-1 and LIMIT; dense all-hit protection; and repeated
Q1024/top-k2048 random, all-duplicate, all-padding unions across CTAs.

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest -q --tb=short cache/tests/test_sparse_token_pool.py cache/tests/test_sparse_token_cache.py cache/tests/test_transient_suffix.py cache/tests/test_resident_metadata.py models/deepseek_v32/tests/test_pool_prefetch.py operators/deepseek_v32/indexer/tests/test_echo_cache_ops.py
```

The first invocation lacked `.venv/bin` in PATH and could not launch installed
ninja; after correcting PATH, all original 59 tests passed before C3b and all
67 passed after it. Ruff check/format check passed for the new test file.
No failed GPU numerical case is hidden by that environment correction.

The complete `cache/tests`, `operators/deepseek_v32/indexer/tests`, and model
`test_pool_prefetch.py` suites subsequently passed all 236 tests (one upstream
DeprecationWarning). After relocating the dense-helper case to its model unit
test directory, the two changed test files passed all 16 collected tests.

Read-only resource review of the built native module found no new concern.
`cuobjdump --dump-resource-usage` reports 25 registers/thread, 2048 static shared
bytes and zero stack/local bytes for the CTA-local kernel; its extra dynamic
bitmap is at most 16384 bytes (8208 bytes at P65536/A128). All launches use
256 threads. The kernel has no cross-CTA waiting, cooperative residency need,
or dynamic register redistribution; its barriers are reached by every CTA
thread. The direct path uses 30 registers/thread and planned vector append46.

## Component measurements

Run IDs: `deepseek_cache_metadata_c3_20261004_01` (direct global bitmap) and
`deepseek_cache_metadata_c3b_20261004_01` (CTA-local candidate).
GPU 2 reports `NVIDIA H200`, capability `[9, 0]`;
PyTorch `2.12.1+cu130`, CUDA `13.0`. P=65536, H=65536,
record BF16×576, candidate capacity128. The checked comparison disables only
C3 helpers via a metadata adapter, retaining C1 append-order/residency behavior.
Selections use a seeded random history selection, repeated first128 IDs, and
negative padding; these are metadata harness inputs, not a model latency trace.

Five warmups and twenty synchronized wall samples per selection/dense case:

| Case | Checked C1 mean ms | C3b mean ms | C3b median ms |
|---|---:|---:|---:|
| Q128 × top-k2048 resident ensure | 0.67789 | 0.06251 | 0.06226 |
| Q1024 × top-k2048 resident ensure | 0.78717 | 0.10012 | 0.09998 |
| Dense all-hit H65536 preparation/wait/drain | 0.16460 | 0.05675 | 0.05631 |

The first C3 direct-global Q1024 implementation measured 0.16608 ms mean /
0.16567 ms median. C3b reduces that mean by about 40%.

The one-layer 64×1024 cold append harness includes required D2H writeback and
commit synchronization, with allocation outside the timed scope, three warmups
and five measured builds: checked C1 28.25059 ms mean versus C3b 11.57089 ms.
Both show substantial bimodal variation (checked 23.037–31.764 ms; C3b
6.371–15.063 ms); the cause is not established. These are component results.

CUDA activity evidence uses ten additional Q1024 selections per path after
five warmups. Sum of actual kernel durations per call is 366.3611 us for
checked C1 (63 kernel launches) versus 57.4779 us for C3b (2 launches):
0.9185 us bitmap reset plus 56.5594 us selection. Checked C1 has eight runtime
stream synchronizations per call; C3b has none. Both traces include one explicit
harness device synchronization per call. Instrumented GPU durations are reported
separately from the uninstrumented API wall measurement.

Temporary reproducibility materials (engineering evidence, no experiment
publication): `/tmp/deepseek_c3_probe.py`, `/tmp/deepseek_c3b_probe.py`,
`/tmp/deepseek_cache_metadata_c3_20261004_01.json`,
`/tmp/deepseek_cache_metadata_c3b_20261004_01.json`,
`/tmp/deepseek_c3_gpu_profile.py`, `/tmp/deepseek_c3b_gpu_profile_summary.json`,
and `/tmp/deepseek_c3b_{checked_c1,c3b}_profile.json`.

## Frozen source identities

- `cache/sparse_token_pool.py`: `0aaf188e272d5f227c08406f74d60c3cc6e80a80e9441c7f480ef55e53cc4aa4`
- `cache/sparse_token_cache.py`: `ccd597c7a4cd97bbcf8eca4601aea53a502aae83532a14c2b493a8d8ceeedb64`
- `models/deepseek_v32/pool_prefetch.py`: `e9391575306b7d3c102f96b606ae3fb502700481765665ab88117a9f52bc57fa`
- `operators/deepseek_v32/indexer/cache_ops.py`: `09c9354105d7c49f7b11dfec726c64a9b516ee0900c5a4dc369a5498704c5483`
- `operators/deepseek_v32/indexer/csrc/echo_resident.cuh`: `670f6b3807d642bd10653dcd36d848c9b2f9e6f643b5b037b5aec2d5993b2490`
- `operators/deepseek_v32/indexer/csrc/echo_indexer.cu`: `2c6c9512e9d3e17c171a15de3425d84d35296d01f107e8c5194af1f78e3331b8`
- `models/deepseek_v32/capacity.py`: `ddf749ed733b39b2742b54bf3df0225fc3de90f78fe638f97918ba493d936daa`

## Remaining work

Integrated screen/full-shape output comparisons, all sixteen users/two rounds,
current-source end-to-end MFU analysis and final publication remain with root.
Residual metadata work includes the still-checked miss/revisit path, one initial
FIFO sort per owner/layer, stream/event management, and candidate/host lifetime
synchronizations. The resident helper's 57 us GPU work and roughly100 us wall
remain measurable; additional tuning should follow integrated evidence.
