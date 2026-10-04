# Motivation admission and cleanup plan

## Observed cost and measurement boundary

Formal source run: `motivation_c3_20261004_u16_r2_01`, H=65,536,
A=128, 16 users, two rounds, ten independent blocks. Mean revisit times (ms):

| Scheme | End to end | Admission | Candidate execution | Cleanup |
|---|---:|---:|---:|---:|
| ECHO | 43.412 | 7.208 | 33.170 | 3.033 |
| Serial sparse | 37.905 | 7.338 | 27.511 | 3.056 |
| Dense prefetch | 84.533 | 7.371 | 74.024 | 3.138 |

The measurements below are CPU engineering probes, not replacement serving
results. `CUDA_VISIBLE_DEVICES=''` prevented GPU use. Input probes use saved
request 16's actual 65,664 Python token IDs. Five warmups and 100 samples were
run per item. GPU copies, CUDA synchronization and actual device allocation
were excluded. Artifacts are `/tmp/motivation_admission_probe/probe.py`,
`measurements.json`, `audit_cprofile.txt`, `fast_validation.py` and
`fast_validation_result.json`.

| CPU operation | Median ms | p90 ms |
|---|---:|---:|
| Current complete `_validate` | 2.694 | 2.729 |
| Existing type/nonnegative validation plus list copy | 2.708 | 2.732 |
| Exact type set/map, C `min`, private list snapshot | 1.308 | 1.344 |
| History slice plus current SHA-256 encoding | 1.143 | 1.152 |
| Full list to CPU int64 tensor | 2.581 | 2.661 |
| Candidate list to CPU int64 tensor | 0.008 | 0.008 |
| Full list to owned `array('q')` | 0.826 | 0.840 |
| SHA-256 over packed prefix view | 0.336 | 0.340 |
| `torch.frombuffer` plus suffix view | 0.002 | 0.002 |
| Current complete CPU input path | 6.499 | 6.603 |
| Packed path with existing validation | 3.867 | 3.902 |
| Packed path with faster exact validation | 2.508 | 2.662 |
| Implemented packed path with safe type-identity mapping | 3.443 | 3.472 |

The alternative snapshot, prefix digest and tensor values matched all 32
saved requests. Empty, bool, float and negative-ID examples retained the
current errors. The `set(map(type, ...))` prototype was not selected for
production: it can invoke user-defined metaclass hashing/equality on an
invalid token type. The implemented path uses C `operator.is_` mapping and
`min`, so invalid types cannot run those callbacks. This retains exact-type
semantics at some additional CPU cost. `candidate_result.json` records the
actual implementation's latest timing and source hashes.

## Why the work repeats

`PersistentGRRunner.execute` validates all tokens, copies the list, copies the
history slice, encodes it with `struct.pack`, and separately traverses the
whole list again to construct `torch.tensor` on every hit. Full-history
identity and type validation are required. The repeated encodings are not.
The saved request's declared hashes must not replace recomputing identity.

The actual hit `PrefixSessionPool.acquire` is already short: 0.0014 ms median
in the CPU fixture. Session LRU mutation is not the observed admission cost.

The cleanup fixture uses real `PrefixSessionPool.audit`,
`DeepSeekServingBackend.session_bytes`, `SharedSparseTokenPool.shared_bytes`
and `PoolSession.session_bytes` with 16 sessions and ten layers each. Tensor
capacities were reduced to 64 tokens and width 1 for CPU storage, while the
number of per-layer/session storage objects matched the real path. CUDA
synchronization was stubbed for the separate truncate measurement.

| CPU metadata operation | Median ms | p90 ms |
|---|---:|---:|
| One session's actual storage scan | 0.046 | 0.047 |
| Shared actual storage scan | 0.096 | 0.099 |
| Full 16-session audit | 0.885 | 0.892 |
| Both current cleanup audits | 1.772 | 1.783 |
| Same full audit with one storage query per tensor | 0.654 | 0.662 |
| Both exact audits with one storage query | 1.300 | 1.308 |
| Truncate after completed transient discard, CPU only | 0.504 | 0.513 |

Both original storage helpers obtained `tensor.untyped_storage()` to dedup,
then obtained it again through `storage_allocation_bytes`. They also stringified
the device on each scan. The implemented exact-query variant kept all session and shared
scans and produced identical physical byte totals. cProfile over 100 full
audits recorded 154,000 `untyped_storage` calls and 76,200 allocation-size
queries. cProfile durations include profiler overhead; the table does not.

## Approved candidate and invariants

Root authorized the following CPU-tested candidate before the next source
freeze. It is not yet a measured GPU improvement.

1. In `serving/persistent.py`, validate a private list snapshot with exact
   `type(value) is int` semantics using
   `all(map(operator.is_, map(type, ids), repeat(int)))`, then `min(ids) >= 0`.
   Keep all user-ID, prefix, context and planned-capacity
   checks. Invalid inputs must still fail before admission.
2. Encode the complete validated list once into an owned native int64 buffer.
   Hash only its history view using the existing little-endian SHA-256
   contract. Create the private CPU tensor view with `torch.frombuffer`, then
   preserve the existing full device transfer **before** `pool.acquire`.
   This first candidate does not change failure ordering or the transfer
   boundary. On big-endian hosts, hash a byte-swapped prefix copy while
   retaining native byte order for the tensor. Preserve the original prefix
   `struct.error` versus suffix tensor-overflow errors on invalid int64 data.
3. In `DeepSeekServingBackend._storage_bytes` and
   `SharedSparseTokenPool._tensor_bytes` only, reuse the queried storage for
   its size and use the device object in dedup keys. Preserve complete owning
   storage size, pinned allocator bins, alias dedup, empty-storage behavior
   and CPU/device separation. No cached byte totals or skipped audit.

The implemented CPU input path saves about 3.06 ms in this probe. The two complete audit
scans save about 0.47 ms in the scaled CPU fixture. These differences must not
be presented as achieved GPU serving speedups or subtracted unconditionally
from formal latency.

## Separate proposals

- DeepSeek transient candidate execution already discards all candidate
  state, restores offsets and synchronizes before setting
  `last_candidate_transient=True`. A guarded `truncate` no-op could remove
  the subsequent ten cache truncations/offset copies. Root owns this model
  change separately. It must verify committed length, written/indexer-visible
  lengths, no pending step, no active execution and preserved prefix hints;
  it must not skip truncation for persistent append or other backends.
- Uploading only candidate IDs on a history hit could remove the remaining
  full-history input transfer, but moving transfer across admission changes
  failure ordering. Defer until its measured cost warrants that additional
  lifecycle change. The first candidate retains the current ordering.
- Do not replace actual allocation inspection with cached accounting or skip
  inactive-session checks. That would require a new explicit mutation/owner
  contract across backends and is outside this candidate.

## Acceptance

CPU checks must cover exact token snapshot/digest/tensor equality; endian and
int64 boundaries; bool/int-subclass/float/negative rejection; buffer lifetime
after caller mutation; first/revisit distinction; unchanged-history reuse
under changing A; changed-history rebuild; eviction and failure release;
actual storage alias/size and pinned-bin accounting; and all existing shared
budget/quota checks. Run GPU checks only after root schedules a device.

Then freeze the candidate source and rerun the affected formal and diagnostic
workload. Report admission, cleanup, total latency and graph/operator MFU from
the same new implementation. Preserve C3 and older published artifacts until
replacement acceptance completes.

## Completed implementation and CPU verification

The authorized input and storage-helper changes are implemented. The final
CPU-only probe checked all 32 saved requests with exact digest and token
equality. Its 100-sample production-path timing was 3.443 ms median and
3.472 ms p90. Evidence is
`/tmp/motivation_admission_probe/candidate_result.json`; no device transfer,
synchronization or GPU execution is included.

The following regression completed with **221 passed and 7 skipped** in
3.04 seconds. The skipped cases require GPU hardware; this result is CPU
acceptance only. Ruff also passed for the changed files.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  serving/tests \
  tests/integration/test_gr_persistent.py \
  tests/integration/test_storage_accounting.py \
  cache/tests/test_prefix_pool.py \
  cache/tests/test_sparse_token_pool.py \
  cache/tests/test_host_allocation.py \
  models/deepseek_v32/tests/test_serving_backend.py
```

Source identities recorded by the final CPU probe:

| File | SHA-256 |
|---|---|
| `serving/persistent.py` | `e07b33d5192372b64042f650c308baf99a9e944d3ba7ed54abc597162fad63f7` |
| `cache/sparse_token_pool.py` | `202ba4352118055f2ad3c2e33236487d7f6edd04eb49e5c2cdf2e751deeb2974` |
| `models/deepseek_v32/serving_backend.py` | `23d63e3fd08b63a9c7afb4fab554deb6f426ffecac22b37939c10cf198d2f67a` |
| `serving/tests/test_token_input.py` | `7f30f8d9c5baf622a8c14b7cc30f0476ca428b179fcbab3aa72044b4a9fbf410` |
| `tests/integration/test_storage_accounting.py` | `3b34ce79d83df9a00dd8ded73e1fa1b8a2ef484c9ad5811a167f470cee40886f` |

The model file is shared with root's independent cleanup work; these hashes
identify the CPU-tested snapshot and do not replace a later combined source
freeze. No GPU checks or replacement experiment have run for this candidate.

`PersistentGRRunner` also serves `experiments/gr_serving`,
`experiments/deepseek_v32_echo_cache/src/capacity_probe.py` and
`experiments/deepseek_v32_echo_official/src/measure.py`. The storage helpers
also affect DeepSeek backend consumers and pool capacity tools. Review their
measurement impact when publishing a new implementation; retain existing
valid artifacts until affected replacement measurements pass acceptance.
