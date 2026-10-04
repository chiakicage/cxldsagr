# C4a and admission cleanup combined validation

Date: 2026-10-04. This checkpoint validates the combined suffix-only resident
indexer mask, packed runner input, exact storage-query helpers and guarded
completed-transient truncate. It records correctness and lifecycle checks;
it does not publish serving latency, MFU or full-NH capacity results.

## Result and scope

The combined GPU command completed **84 passing checks** and one failure in
the newly added test wrapper. All four schemes passed the full real-checkpoint
H=P=65,536, NH=131,072, C=1,024, A=121/128 eager/graph comparison, including
exact prefix/candidate hidden, candidate logits and retained KV/indexer state.
The wrapper incorrectly passed a Python list through its own CUDA-tensor-only
observer during the final poisoned-reuse assertion. Calling the saved backend
entrypoint fixed the test; the isolated four-scheme runner test then passed
in **12.29 seconds**. No production source changed to resolve this failure.

The six directly involved production files listed below had identical hashes
before the combined GPU run, after it, and after the focused runner rerun.
The C4 owner confirmed that the captured indexer hash is the final aligned
suffix implementation. GPU 0 was released after validation and reported
1 MiB used, 0% utilization.

New coverage is limited to two tests:

- `serving/tests/test_token_input.py` transfers packed int64 input to CUDA on
  a non-default stream, overwrites the host tensor and caller list, drops the
  host tensor reference, and verifies exact device values after synchronization.
  This includes the largest valid signed int64 value.
- `models/deepseek_v32/tests/test_compute_graphs.py` exercises
  `PersistentGRRunner` with the real ten-copy checkpoint and compute graphs,
  H=2,304, P=NH=4,608, C=256 and A=16/23. Two histories fit in the independent
  HBM history quota or host arena so the runner can audit an inactive session.
  These smaller shapes cover lifecycle invariants; the separate 64K test
  supplies the full-shape numerical comparison.

The new runner test checks all four schemes with exact candidate hidden/logit
agreement against HBM-only. Before GPU model execution, it overwrites the
packed CPU tensor and caller list at admission; observed GPU prefix/suffix
values and retained input views remain exact. Changing A reuses the same
history-only session. Changing history rebuilds it while preserving the
user's revisit classification. On a history hit, the observed sequence is
`audit -> truncate -> audit`; both audits measure every live session, including
the inactive user, and produce the exact shared-plus-session storage totals.
Metrics agree with these measurements, and candidate D2H is zero.

An injected layer failure after real candidate computation discards only the
failed user's session, leaves the other user present, does not increment the
failed request's visit count, and clears active execution/transient owners.
A separate injected graph-completion Event-constructor failure sets graph and
backend poison, retains graph buffers, the active session and the admission
owner, and rejects reuse, session release and shared-storage release. Failed
runner close keeps the owner bound. The test synchronizes the healthy device
and resets its synthetic fixture flags only for teardown; this is not a
production recovery mechanism or validation of a destructive CUDA fault.

The pre-existing graph tests also cover delayed owned D2H across replay,
cross-stream dependencies, immutable precision policy, partial allocation
failure, private allocator segments, returned tensor lifetime and source-copy
independence. Existing storage/pool/backend tests cover complete owning
storage, aliases, device separation, pinned bins, quotas and truncate guards.

## Commands

All commands run from the repository root. Hardware and dependency scope is
the same as the [C3 validation](deepseek_motivation_compute_graph_validation.md),
except physical GPU **0** was used here: NVIDIA M403, SM90, 150,121,545,728
device bytes; checkpoint `/preset-models`. No test results or profiler
artifacts were written into experiment or test directories.

CPU regression: **227 passed, 12 GPU/opt-in skips**, 3.12 seconds.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  serving/tests tests/integration/test_gr_persistent.py \
  tests/integration/test_storage_accounting.py cache/tests/test_prefix_pool.py \
  cache/tests/test_sparse_token_pool.py cache/tests/test_host_allocation.py \
  models/deepseek_v32/tests/test_serving_backend.py \
  models/deepseek_v32/tests/test_compute_graphs.py
```

Combined GPU command: **84 passed, one test-wrapper failure**, 85.89 seconds.

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

Corrected runner test: **1 passed**, 12.29 seconds.

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
DEEPSEEK_GRAPH_CHECKPOINT=/preset-models .venv/bin/python -m pytest -s -q \
  models/deepseek_v32/tests/test_compute_graphs.py::test_checkpoint_graph_runner_packed_admission_audits_and_failure_ownership
```

Ruff check and format check pass for both changed test files.

## Source identities

| Production file | SHA-256 before and after both GPU commands |
|---|---|
| `serving/persistent.py` | `e07b33d5192372b64042f650c308baf99a9e944d3ba7ed54abc597162fad63f7` |
| `cache/sparse_token_pool.py` | `202ba4352118055f2ad3c2e33236487d7f6edd04eb49e5c2cdf2e751deeb2974` |
| `cache/sparse_token_cache.py` | `ccd597c7a4cd97bbcf8eca4601aea53a502aae83532a14c2b493a8d8ceeedb64` |
| `models/deepseek_v32/serving_backend.py` | `23d63e3fd08b63a9c7afb4fab554deb6f426ffecac22b37939c10cf198d2f67a` |
| `models/deepseek_v32/compute_graphs.py` | `9f609674b458476a096f9941ad3cbfae118a762cbfcd44a6cbef0445fb00eb77` |
| `operators/deepseek_v32/indexer/echo.py` | `30492794f10c4bd34d730c0bb926c79827095feb59e7cad35765ff0bced4c219` |

The final test hashes are
`46e8d593b24d3c99b148e8906ba3e10fb07b5cbd040bbad773b9ec8d7dc9cc03`
for `test_compute_graphs.py` and
`7760f34611cb2f9586554a8ee4973606136d9dec909465e1bd5fe7f27a685f91`
for `test_token_input.py`. The former was
`a70108ede6d660de50718d8ab3589b124e621b6a3fe21b426a6ab961c0b90b2f`
during the initial combined command; its only subsequent change was the
new runner test's poison assertion and guaranteed synthetic-fault teardown.

This is not a hash of every transitive dependency or checkpoint tensor.
Performance publication must retain its separate source snapshot and run ID.

## Large-shape memory observations

All values are bytes. Graph-private capacity includes inactive private
allocator segments; static inputs are counted separately. Values remained
stable after interleaved users and both candidate sizes.

| Scheme | Private reserved | Static allocated | PyTorch allocated at setup | PyTorch reserved at setup | Device used at setup |
|---|---:|---:|---:|---:|---:|
| HBM-only | 7,260,340,224 | 1,943,457,792 | 24,860,961,792 | 29,909,581,824 | 30,705,516,544 |
| ECHO | 7,260,340,224 | 1,944,079,872 | 27,815,015,424 | 33,797,701,632 | 34,604,122,112 |
| Serial sparse | 7,260,340,224 | 1,943,354,368 | 27,738,576,896 | 33,443,282,944 | 34,251,800,576 |
| Dense prefetch | 7,260,340,224 | 1,943,500,288 | 27,738,722,816 | 33,359,396,864 | 34,167,914,496 |

These are snapshots from a correctness process holding eager and graph
backends, diagnostic tensors and allocator caches. They are not isolated
single-backend peaks, physical capacity acceptance at NH=16,777,216, or
mandatory storage reservations. The graph-private 12 GiB limit is a chosen
upper bound, not measured preallocation. No latency or MFU conclusion follows
from these tests; the quiet formal serving and profiling runs remain required.
