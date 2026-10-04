# ECHO pinned allocation and CPU workspace correction

> Experiment location: [deepseek_v32_echo_cache](../../../experiments/deepseek_v32_echo_cache/README.md).
> Historical commands, source paths and hashes below remain unchanged; use the
> [migration record](echo_cache_experiment_migration.md) to locate moved artifacts.
> This directory migration is not a new measurement.

Date: 2026-10-03. Scope: shared sparse pools, DeepSeek dense backing and their
admission plans. This is engineering acceptance; no performance result is
published by these checks. NOSA and the active measurement freeze were not
modified.

## Defect and correction

The project's PyTorch `2.12.1+cu130` CUDA CachingHostAllocator rounds each host
allocation to `PowerOf2Ceil(size)`. The installed
`ATen/core/CachingHostAllocator.h` allocation path and the `host_memory_stats`
documentation explicitly state this rule. Admission's independent GPU probe
confirmed that a 4,097-byte request takes an 8,192-byte pinned block.

The previous shared/dense ledger used the logical tensor storage size. For
NH=1,050,624 and BF16 width=576, one layer requested 1,210,318,848 bytes but owned
a 2,147,483,648-byte block. Ten layers therefore own **20 GiB** of pinned backing,
before metadata and execution workspace. Dense 64K+128 sessions own ten
independent 128-MiB host blocks, rather than their smaller logical record bytes.

`cache/host_allocation.py` now reserves each allocation's bin and explicitly
requests a backing tensor of that full size. The returned view keeps the original
record shape and contiguous layout; its storage exposes the full owned bin.
Allocation checks verify storage capacity and pinned state against the reservation.
Shared-pool construction also reconciles its complete fixed storage with its
estimate. CPU reference storage remains unrounded. Accounting deduplicates the
complete storage identity, including aliased views.

The shared planner rounds each layer separately. Automatic NH search retains
binary search over 64-token pages, using the discontinuous bin-aware estimate
at each candidate and retaining CPU ownership-table headroom. DeepSeek dense
session admission uses the same per-allocation rule. Transfer counters continue
to use 1,152 bytes per BF16 record, irrespective of storage padding.

## CPU execution reservation

`ExecutionReservation.dram` conservatively sums three independently observed
scratch bounds supplied by the admission investigation:

| Metadata field | Bytes | Observed operation |
| --- | ---: | --- |
| `workspace_cpu_indexer_bytes` | 8 | Pageable scalar in indexer finite/comparison checks |
| `workspace_cpu_scalar_bytes` | 8 | Pinned synchronous CUDA scalar; internal nonzero and unique scalar buffers also fit this bound |
| `workspace_cpu_metrics_bytes` | 24 | Pageable copy of the three-int64 metrics counter |
| `workspace_cpu_bytes` | 40 | Conservative sum of the three terms |

The shared serving reservation includes 40 bytes for every scheme, including
resident HBM. Resident has no host KV backing, but does use CPU execution scratch.
The standalone planner includes the same terms per device and in total DRAM.
The independent serving audit subtracts this explicitly declared workspace when
checking that resident host backing is zero. These observations do not prove a
complete process CPU allocator peak; the full memory collector remains a separate
acceptance gate.

Admission also found that this PyTorch version can leave `active_bytes.current`
and `.peak` inflated after no-stream free/reuse. They cannot establish current
owned bytes. Object inventory measures owned bins; allocator cached blocks with
no cache tensor owner remain a separate physical-retention statistic. Do not
infer a per-object allocation from a global active-stat delta.

## Validation on the updated code

CPU command, with CUDA hidden:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  cache/tests/test_host_allocation.py cache/tests/test_sparse_token_pool.py \
  cache/tests/test_sparse_token_cache.py models/deepseek_v32/tests/test_cache_resources.py \
  models/deepseek_v32/tests/test_serving_backend.py models/deepseek_v32/tests/test_echo_infer.py \
  experiments/deepseek_v32_echo_prefill/tests/test_chunk_sweep.py -q -p no:cacheprovider
```

Terminal result: **114 passed, 9 GPU checks skipped in 2.50 s**. Coverage includes
bin boundaries, independent-layer rounding, one-byte budget rejection, automatic
NH across a bin step, alias deduplication, unrounded CPU storage, all four schemes'
40-byte declaration, and resident rejection at a 39-byte DRAM limit.

Authorized actual GPU2 command:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=2 MAX_JOBS=2 \
  .venv/bin/python -m pytest cache/tests/test_host_allocation.py \
  cache/tests/test_sparse_token_pool.py cache/tests/test_sparse_token_cache.py \
  models/deepseek_v32/tests/test_serving_backend.py -q -p no:cacheprovider
```

Exec session `80702`, exit 0: **44 passed in 3.16 s**, with no skips. The real
pinned checks exercised 137-byte logical storage in a 256-byte block, a shared
147,456-byte layer in a 262,144-byte block, and dense 65×576 BF16 records in a
131,072-byte block. Nondefault-stream D2H content checks passed; the dense copy
counter remained 1,152 bytes for one appended record. Existing shared-pool/cache
CUDA ownership, recall, snapshot and asynchronous-lifetime checks also passed.
Ruff lint and format checks passed for all ten touched Python files.

Post-check identities below identify the reviewed source; no complete runtime
before/after manifest was produced by this short regression. The next checkpoint
freeze must record its own full identity. Its source list now includes the new
`cache/host_allocation.py` dependency.

```text
6c6c65e166fc6a2289547482a9bc51c1da136190b9105423edd23f9732567150  cache/host_allocation.py
4b1a65503cc97340e0e2e688e3b7fec4dd3b12575cbed7f5856ef24244b651c4  cache/sparse_token_pool.py
dbe0d623cd4eccce38a6f285bbb88b9e9ece6930b55b18fb33137d2b38404d9f  models/deepseek_v32/cache_resources.py
48b77dec8e7af782c3b4844dbf9dc5f8d7700c79a160736a7acb270fcacbfd90  models/deepseek_v32/serving_backend.py
84e974f0323e053bd95808c56b861e626e18895b96f9dda12be02802c06a60bc  models/deepseek_v32/echo_infer.py
c5dca03f5d3910b3a57982b5a7919faef25b25d53475df4744bb0d747ce7fb9a  models/deepseek_v32/tests/test_echo_checkpoint.py
```

The full three-layer 64K+1K checkpoint gate, accepted complete memory capture,
and affected performance runs are still required on the new freeze. Earlier
checkpoint/performance evidence does not apply automatically to this allocation
change; retain valid old reports until accepted replacements are published.
