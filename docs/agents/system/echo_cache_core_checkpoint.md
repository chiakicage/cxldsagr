# ECHO shared cache core checkpoint

Date: 2026-10-03. Scope: generic P1/P2/P4 cache implementation plus the cache side
of the P3 native ABI. This is engineering validation, not a performance result or
completion claim for the full implementation plan.

## Implemented ownership and API

- `SharedSparseTokenPool(host_capacity, width, layers, slots, ...)` owns fixed
  per-layer pinned host arenas and HBM pools. `host_capacity` is a multiple of 64;
  `slots=P` excludes the extra sentinel record at physical slot zero. The CPU
  free-page stack is a fixed int32 tensor. Noncontiguous pages are supported.
- `allocate_session(capacity)` reserves padded host pages and CPU/device page
  tables; `session.layer(i)` returns an owner-checked view. Shared `host_to_device`
  and `device_to_host` use global token IDs, translated by global page numbers.
  Session release drains operations, invalidates only that owner's IDs, clears
  private tensor aliases, and returns pages. Backend close drops shared storage.
- `cache.operation()` provides a nested, exclusive layer lease. Its completion
  event includes attention submitted by the caller. A pending fused prefetch
  retains workspace ownership until finalization, including when the caller did
  not use an outer context. A different session or layer cannot overwrite it.
- `begin_step` / `declare_indexer_visible` / `prepare_prefetch` /
  `finalize_prefetch` / `append` / `ensure` / `commit` implement the staged API.
  Prefetch runs before main KV append and excludes the unwritten suffix through
  `history_length`. Prepare sorts all usable slots without evicting records.
- Append copies the projected records directly into HBM and writes the same
  records to host. Adjacent allocated host pages form one D2H span; arbitrary
  noncontiguous pages require multiple spans. Input records must be contiguous.
  All source storage remains referenced through a D2H event. `reserve_append_source`
  must be called before projection to bound the model's source storage at the
  configured global in-flight count (default two). Standalone append accepts an
  already-created tensor and reports its complete underlying source storage.
- Every prefetch/recall waits host write events, including the current suffix.
  Consumer events gate subsequent leases and pool reuse. Drain errors poison the
  pool and prevent further use. Rollback drains and finalizes any completed
  prefetch allocations before invalidating only the session suffix.
- Exact recall deduplicates the full selected union, checks its capacity, protects
  resident selections, uses empty slots first, and evicts only the deficit. A
  union larger than P raises `WorkingSetTooLarge`; the model owns exact query
  consumption splitting. Cache statistics include the model's split count.
- Priority follows explicit FIFO events: append allocation, prefetch finalization,
  recall protection, and recall allocation; empty protection/finalize events also
  advance the clock. Standalone map lookups do not stamp. Rank compression before
  9,000,000 preserves ordering and ties while retaining free/sentinel meanings.
- `metadata_ops` is injected by the DeepSeek model. CUDA uses the native
  `protect`, `mark_misses`, `finalize_prefetch`, and `release_ids` helpers. CPU uses
  the generic reference. Slot selection and full exact-union construction still
  use PyTorch operations and their synchronization/allocation costs belong in
  measured execution; no claim is made that all recall planning is fused.
- `snapshot()` copies each shared pool once, including records/maps/free-state/
  priority/clock. Restore requires quiescence and unchanged session ownership and
  prefix content. Truncation below a saved prefix invalidates that snapshot;
  release/reallocation changes the topology epoch and prevents stale restore.
  Prefix host/indexer contents must not be mutated outside the cache API. Model
  code owns independent indexer tensors and offset snapshots.

## Accounting

Pure `estimate_shared_bytes` matches the fixed tensor allocations: host arenas,
records, global maps, priorities, bitmap, clock tensors, sorted slots,
allocation log, miss scratch, counters, and CPU free-page stack. The session
estimator matches CPU and device page tables plus bounded per-layer counters;
CPU reference aliases the two page tables and counts their storage once.

`shared_bytes()` reports fixed allocated storage. `pending_source_bytes` separately
reports outstanding source storage, deduplicated by full underlying storage
identity. The serving backend adds both. Model index-K/scales/offset and indexer
execution reservations are outside this generic component. Metadata temporary
allowance is exposed by `estimate_execution_workspace_bytes(NH, P)` as
`64*(P+1)+64*NH`; the backend must add it before allocation. Diagnostic snapshots
report `diagnostic_bytes` separately and are not serving cache capacity.

No per-chunk GPU counter tensor list remains. Accurate fused copies/evictions/
capacity failures accumulate in a fixed three-element counter per layer/session;
the potentially overshooting reservation counter is not used as actual traffic.

## Verification on this source

Hardware: GPU 0, NVIDIA M403, compute capability 9.0. No test in this core work
used GPU 3. Python, PyTorch, TVM FFI, CUDA and Ninja come from the existing project
environment. Command run from the repository root:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m pytest cache/tests/test_sparse_token_cache.py \
  cache/tests/test_sparse_token_pool.py -q
```

Result: **17 passed in 2.70 s**, including CPU reference and actual CUDA execution.
`ruff check` passed for the same four Python files. The checks cover:

- noncontiguous page reuse, impossible host allocation without eviction,
  exact persistent estimator/storage equality, shared layer ownership;
- A→B→A content recall, cross-owner eviction/release, stale handles and reused IDs;
- partial-free deficit eviction, duplicate selections, invalid padding, exact
  capacity rejection, no compulsory H2D for newly appended records;
- fixed official event trace for empty/all-hit protection/finalization, sentinel
  exclusion, overflow attempts versus exact copy counts, lifetime clock rollover;
- nested leases and retained pending-prefetch workspace ownership;
- global snapshot restore, stale-prefix/topology rejection, release/close dropping
  otherwise-retained tensor ownership;
- non-default CUDA streams, immediate host reads, native prefetch/protect/finalize/
  release integration, full record content and bidirectional map checks;
- a deliberately delayed D2H stream proving that the source stays referenced
  while pending, accounting includes the entire storage of a view, and
  `reserve_append_source` drains before the next projection source is allocated.

Native normal-domain comparisons with the fixed official checkout are recorded
by the native operator agent in `echo_cache_native_checkpoint.md`. The generic
tests alone do not establish official kernel equivalence. Full checkpoint,
budget/peak, chunk sweep, and serving trace gates remain owned by the parent task.

Frozen SHA256 identities at the above verification:

```text
d1dcf55289cda1f0445d5e323fd60e0f3f20e4ef333bdbf9f3cb1243fa0df16b  cache/sparse_token_cache.py
5b578d06d94e67e0d8131951d364346dbbb008b14f3233ff8dea15d74a7d5193  cache/sparse_token_pool.py
e1f8ec7d3598d902ac85ce7573b77fa46eea4a77c62aed39ce82df6b979137f2  cache/tests/test_sparse_token_cache.py
b696a9d89f94059bb5b6523cb994bacfbb423d46dd1fc90d96d3ee098f121642  cache/tests/test_sparse_token_pool.py
```

## Real checkpoint layers 0–2 numerical acceptance

The user clarified that acceptance covers the real first three checkpoint layers.
The earlier 61-layer attempt was cancelled for this scope correction and exited
with status 143; it supplies no validation evidence.

The current opt-in check is
`models/deepseek_v32/tests/test_echo_checkpoint.py`. It executes checkpoint layers
0, 1 and 2 in sequence, including embedding, the three dense MLPs, final norm,
and the final-token LM head. It does not copy transformer blocks. Both resident
and ECHO offload construct their own sparse prefix from independent empty caches;
only model weights are reused across policies.

Before the final checkpoint run, the standalone and serving CPU suites passed
**43 tests in 1.96 s** with `CUDA_VISIBLE_DEVICES=''`:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  models/deepseek_v32/tests/test_echo_infer.py \
  models/deepseek_v32/tests/test_serving_backend.py -q -p no:cacheprovider
```

The new cases reject `P=8 < min(topk=16, capacity)` before changing an existing
resident model or an ECHO/serial-sparse serving plan. They check cache ownership,
stored prefix records, valid lengths, plan identity and subsequent candidate
outputs, so a rejected offload admission cannot invalidate a usable session.

Command run from the repository root on 2026-10-03:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
  DEEPSEEK_ECHO_CHECKPOINT=/preset-models \
  .venv/bin/python -m pytest models/deepseek_v32/tests/test_echo_checkpoint.py \
  -q -s --tb=short -p no:cacheprovider
```

Final result: **1 passed in 35.36 s**, process exit 0 (terminal session 59658).
This final rerun includes the admission precheck, current nonmatrix/FlashInfer
runtime and both new source-manifest entries. An earlier transient import failure
while those runtime files were being edited did not execute the model and is not
validation evidence. This is numerical verification, with no
timing comparison, persistent test output, or performance experiment. The
checkpoint input is deterministic GR text (seed 314159), with 65,536 prefix
tokens and 1,024 extend tokens. Model prefill chunk is 2,048; extend runs as one
1,024-token batch. Shared per-layer usable P is 32,768, excluding sentinel zero;
NH is 66,560 tokens, including candidate capacity. Hardware is physical GPU 0,
NVIDIA M403 / SM90.

Earlier checks used `/mnt/user-ssd/chenkaiqi/DeepSeek-V3.2`; the latest result
above was actually rerun after the user selected `/preset-models`. It is not a
path-only relabelling of the earlier execution.

The unchanged `profile_layers.comparison` contract is `rtol=0.01, atol=0.02`.
All comparisons were also bitwise equal (`max_abs=0`, `relative_l2=0`):

| Compared output | Complete shape |
|---|---|
| Every extend hidden vector | `[1024, 7168]` |
| Extend final-token logits | `[1, 129280]` |
| Prefix final-token logits | `[1, 129280]` |

All outputs were finite. Offload actually performed 13,535 fused prefetches and
16,263 exact recalls; total H2D was 34,327,296 bytes and D2H was 230,031,360 bytes.
The capacity fallback count was zero. These counters establish exercised paths;
they are not a performance result.

Source SHA256 before and after execution was identical across 1,212 scoped
model/cache/operator, third-party source/include, input/comparison helper, and
test files, including `models/deepseek_v32/nonmatrix.py` and
`operators/flashinfer.py`; `changed=[]`:

```text
dependency manifest digest: a6dcea67735358d25f796290344f7db721e3de3db8b41b08a5a8f382b25e2e95
input token IDs: 819951f2696d92fe65132e9a0aa7f85a1e7f207483260188364fc5844775a881
config.json: c7fa8b191e9936d8e6a57d864baab82b792fae16a116416cdd3a75ba76bc5af1
model.safetensors.index.json: 2a150b2af4aba7b037edc9fdba70f3b41abf758ee926279e7701d954298f9884
tokenizer.json: cd050be35cae877f8f0aa847f45aa87e23835a56ca32b29b28545597852784e5
```

These metadata hashes identify the selected local checkpoint configuration and
index; they are not hashes of every weight byte. Without the opt-in environment
variable, the heavyweight checkpoint test skips. Once explicitly selected,
missing CUDA/checkpoint files/dependencies fail rather than skip.
