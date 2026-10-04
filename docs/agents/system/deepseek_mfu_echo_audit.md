# DeepSeek ECHO and sparse MLA operator audit

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

Updated: 2026-10-03. This is implementation evidence, not a new experiment report.
The original `20261002_echo_layers3_mfu_01` was replaced after acceptance of
`20261003_echo_layers3_official_01` and fresh control `20261003_echo_layers3_control_01`.
The superseded report and outputs have been removed; historical paths below identify
the diagnostic inputs at the time, not retained current outputs. The current user-directed scope
is non-GR, actual checkpoint layers 0–2, with the original input token IDs.
It is not a 61-layer validation or a serving run. This audit changes no cache
policy. The earlier ECHO measurements below predate the official-backend
transition; their old resident path is not evidence for the new implementation.

## Official FlashMLA adapter and validation

`operators/deepseek_v32/attention/device_only/mla.py` now calls the official
`flash_mla.flash_mla_sparse_fwd` API, with no custom Triton fallback. The current
contract is SM90, BF16, 64 or 128 query heads, D576 and V512. The offload adapter
uses the same backend after physical remapping. The independent reference
remains importable without FlashMLA or Triton.

The adapter makes Q/KV contiguous and 16-byte aligned, masks invalid int64 IDs
before narrowing to int32, and pads the selection width to a multiple of 128
using `-1`. It preserves duplicate slots and all-padding zero rows. All copies
and padding are part of the operator call. No optional `topk_length` truncation
is used. Unsupported geometry or dtype fails explicitly.

The parent integrated FlashMLA revision
`ba89a3466e9470ad08ab39738d4e7bb66989e1e7` with shared CUTLASS `f3fde583`.
The loaded package reports version `1.0.0`; its native library is
`/root/.venv/lib/python3.12/site-packages/flash_mla/cuda.cpython-312-x86_64-linux-gnu.so`,
SHA256 `1c9969551e8ef7ca96ca1182ab4f9347aa5402586a1ba5a5c82f86cf4b2fb9a6`.
`build_info()` records the actual package, interface and native-library hashes.

On physical GPU 2, 29 tests passed in 6.05 seconds:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest -q \
  operators/deepseek_v32/attention/device_only/tests/test_deepseek_mla.py \
  operators/deepseek_v32/attention/offload/tests/test_deepseek_offload_mla.py \
  operators/deepseek_v32/attention/reference/tests
```

These checks cover supported heads, selection tails, duplicates, negative and
out-of-range IDs, int64 wraparound sentinels, unused NaN KV rows, all-padding
rows, strided and misaligned views, empty inputs, large logits, explicit
unsupported-input errors, independent CPU imports, physical remapping and
bitwise equality after query splitting.

The original layer-2 capture (SHA256 below) was then replayed unchanged:
Q `[1024,128,576]`, KV `[66560,576]`, indices `[1024,2048]`, BF16 and scale
`0.1352337788608801`. Full-batch output is bitwise equal both to concatenated
64-query calls and to a compact reversed physical mapping of all selected KV.
An independent FP32 PyTorch oracle, with TF32 disabled and FP32 output retained,
gives maximum absolute error `2.0075589e-4`, mean absolute error `9.8124565e-7`,
normalized RMSE `0.001686989`, and cosine `0.99999857`. Every element passes
`atol=0.004, rtol=0.02`.

After ten warmups, 40 complete API CUDA-event samples have median `0.911856 ms`
and range `0.907968–0.926464 ms`; synchronized host-wall median is `0.924986 ms`.
This is a same-input isolated adapter diagnostic, excluding selection, transfers,
cache management and model propagation. It establishes neither a whole-model
speedup nor new three-layer numerical acceptance. No older custom-MLA timing is
relabelled as official FlashMLA performance. The machine reports NVIDIA M403 via
`nvidia-smi`; PyTorch reports H200/SM90. GPU 2 UUID is
`GPU-72f87cfe-1a24-f69c-8b5d-71f1d8414a34`.

Temporary harness and JSON are
`/tmp/deepseek_mfu_echo_audit/flashmla_validate.py` and
`/tmp/deepseek_mfu_echo_audit/flashmla_validate.json`; they include all samples,
capture identity, adapter hashes and actual binary identity. These validation
artifacts are not experiment deliverables. The integrating agent owns the new
official FlashMLA / DeepGEMM three-layer run and its publication.

## Observed bottleneck

The published layer-2 NCU replay uses real extend activations with 1,024 queries,
66,560 index keys, 64 index heads, D128 FP8 inputs, a 16,384-record BF16 main-KV
pool, and an 8,192-record prefetch budget. Historical pool residency is reset to
empty before each replay; this is not the warm model cache state. In the existing
full NCU report, resident logits take 1.119 ms, have zero local-memory requests,
and reach 51.64% elapsed tensor-pipe active. Fused prefetch logits take 2.423 ms,
have 21,450,007 local load/store requests, and reach 23.10% elapsed tensor-pipe
active. Long-scoreboard stalls per issued instruction rise from 0.623 to 7.261.
These NCU values are diagnostic observations, not independent wall-time samples.

The existing source report confirms the cause rather than only suggesting it:

- `extend_logits_offsets_val[16]` is written with four `STL.128` instructions
  and accessed by dynamic `LDL` instructions in the histogram/recall loops.
- The top-k branch deallocates to 32 registers even though it scans logits,
  maintains the histogram pipeline, reserves slots and copies 1,152-byte host
  records. SASS spills its Q/KV loop counters, flags, and scheduling variables.
- The fused implementation puts math behind a 96-stage logits producer/consumer
  ring. A slower top-k consumer eventually stalls the math producer at its
  empty-logits barrier. The resident math path has no such consumer.

Authoritative old inputs: the `full_indexer-offload.ncu-rep` and
`source_indexer-offload.ncu-rep` files under
`experiments/deepseek_v32_echo_prefill/output/profile/20261002_echo_layers3_ncu_indexer_offload_01/`.
The parsed SASS and metric evidence is in the matching `output/data/` run's
`ncu_analysis.json`. The input capture is
`output/data/20261002_echo_layers3_mfu_01/kernel_inputs_layer_2.pt`, SHA256
`8e5a5e5eebbce6d6338b6279339111575ab71269e53cad9d188a982b4747a8f6`.

## Implemented candidate

`operators/deepseek_v32/indexer/csrc/echo_logits.cuh` now gives the top-k
warpgroups 96 registers and loads only the current two queries' scalar offsets
outside the KV loop. The two offsets have compile-time indexing. FP8 WGMMA,
reduction order, causal masking, coarse-bin construction, exact-top-k selection,
host record copying, slot reservation, and the public API are unchanged.

The dynamic register requests are `256*112 + 256*96 + 128*32 = 57,344`.
The actual candidate NCU launch allocation is 96 registers per thread for 640
threads, a 61,440-register CTA allocation. Thus the dynamic requests fit the
actual CTA register allocation; this does not rely only on the SM-wide limit.
Shared memory remains 227,972 bytes per block. The launch still uses one CTA per
SM and keeps the existing logits ring and synchronization protocol.

## Validation and temporary measurements

The isolated candidate was run on physical GPU 3. The machine reports M403 via
`nvidia-smi`; PyTorch reports an H200, CC 9.0, 132 SMs. Hardware naming must be
reported under the existing experiment hardware-identity policy, not inferred
solely from PyTorch's name.

Temporary work is under `/tmp/deepseek_mfu_echo_audit/` and has not been promoted
into `experiments/`. `baseline.json` and `candidate96.json` include actual source
SHA256 values, verification results, hardware information and all 30 API CUDA
event samples. `bench.py` is the exact harness. Both use five warmups, alternating
resident/offload execution order, identical real captured inputs and a freshly
reset historical pool before each sample. The reset is synchronized before the
start event and is excluded. Timings include the logits API's metadata and mask
kernels and any device-observed enqueue gaps; they are not pure GEMM timings.

| Same-GPU API event median | Before | Candidate |
| --- | ---: | ---: |
| Resident indexer | 1.264 ms | 1.275 ms |
| Fused offload indexer | 2.620 ms | 2.206 ms |

Fused API time falls 15.8%. The candidate's cold-pool `--set full` NCU capture
takes 1.987 ms, reports zero local load/store requests and zero local SASS
instructions, and reaches 28.33% elapsed tensor-pipe active. Long-scoreboard
stalls remain 5.786 per issued instruction. Full/source NCU files use the
`candidate96_full` and `candidate96_source` names in the temporary directory.
The NCU harness is `profile.py`; `--cache-control all --clock-control none` is
used. The old-vs-new NCU comparison is not a same-session timing speedup claim.

The real layer-2 fused logits remain bitwise equal to resident logits. Before
and after the timing samples, all copied historical records and protected
current records match host bytes, and reservation count agrees with the map.
The CUDA unit suite passed all 15 tests, including the independent FP32 logits
oracle, bitwise fused/resident outputs, exact-top-k subset and mapping checks,
nonzero offsets across persistent Q blocks, tied scores, zero budget, repeated
budget exhaustion, rollback, generic record width/dtype and nondefault stream.

Command:

```bash
PATH="$PWD/.venv/bin:$PATH" CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m pytest -q \
  operators/deepseek_v32/indexer/tests/test_echo_indexer.py \
  cache/tests/test_sparse_token_cache.py
```

This is operator correctness and isolated performance evidence. Full model
64K+1K resident/offload numerical acceptance, restored-prefix timing, and updated
three-layer operator MFU publication remain the integrating agent's work.

## Cache policy findings and separate optimization path

`SparseTokenCache.prepare_prefetch` first makes the new chunk resident, then
calls `_available_slots(current, min(8192, slots-new_count))`. That helper
invalidates every offered slot before the fused kernel knows whether the slot
will be used. Empty slots are ranked first, but warm history can still be
evicted eagerly even when the histogram finds few useful misses. This preserves
mapping safety but can increase later exact recall and waste reuse.

The old three-layer extend run illustrates the mismatch between fixed capacity
and actual prefetch use: layer 1 copied only 3 prefetched records, recalled 1,054
records including the 1,024 newly appended records, and evicted 1,098 records.
Layer 2 filled all 8,192 prefetch slots and evicted 13,094 records while its
maximum exact working set was 13,005. These counters establish policy activity;
they do not prove which individual eviction caused a later miss. A per-record
reuse/recall trace is required for that causal conclusion.

Other source-level costs are explicit:

- New KV records are copied GPU→host by `append`, then immediately host→GPU by
  `ensure(current)` in `prepare_prefetch`. Keeping the original new records in
  their protected HBM slots would eliminate this round trip while retaining
  host backing for future history.
- Exact `ensure(indices)` performs `torch.unique` on up to 2M indices, creates
  masks/maps, sorts eviction age, and reads a GPU scalar to validate the maximum
  index. This introduces many kernels and CPU synchronization outside indexer
  MFU, while still contributing to end-to-end latency.
- Prefetch can only reserve slots after eviction completes, which prevents a
  concurrent invalidation/re-prefetch race. Publishing a map before record copy
  is safe only because all consumers and subsequent mutations are ordered after
  the fused kernel on the same stream. Neither invariant may be removed.

A future cache change should compare empty-only opportunistic prefetch and a
measured bounded reservation policy against the current policy, preserve exact
selection and residual recall, and measure combined indexer + recall + attention
time. Lazy invalidation inside the current fused kernel needs a separate
race-proof reservation/publication protocol; merely skipping `_evict` is not
safe. Per-record transfer/hit/eviction traces can identify whether reservation
budgets should depend on existing residency. Any new state must be included in
`snapshot_prefix`/`restore_prefix`, and all variants must restore identical
prefix residency before extend timing.

The existing full-model implementation snapshots record contents, both maps,
ages, clock and offsets after synchronization, and restores all of them before
each extend sample. It commits only after all layers and device synchronizations
succeed; only transactions started by the failing invocation are rolled back.
The register/offset candidate leaves these contracts intact.

## Earlier serving proposal: outside the current task

The latest scope is the non-GR three-layer benchmark above. The following
earlier serving proposal and static accounting findings are retained as internal
context; they are not a current launch instruction. No serving run was launched
by this audit.

2026-10-03 scope correction: the old 4K/16K/64K seven-population matrices are
an inventory of affected code consumers, **not a rerun instruction**. The user
explicitly deferred 4K/16K and rejected the earlier population-scaling workload.
Do not rerun those 84 DeepSeek cases. The current authorized serving workload is
64K history + 128 candidate, user IDs `list(range(16)) * 2`, 32 requests,
HBM 4 GiB / CPU DRAM 64 GiB, no heat distribution, prefix chunks of 1,024 and
32,768 sparse slots. The detailed prior contract is
[the sequential rerun record](gr_serving_sequential_rerun.md).

Only DeepSeek's four schemes consume the changed MLA, FP8 linear and ECHO
indexer implementations. For this optimization, rerun those four schemes on
the authorized sequential workload: 128 request rows, 128 correctness records,
96 non-HBM all-candidate comparisons, and 32 saved HBM references. Each scheme
starts empty after independent warmup. Retain all four schemes regardless of
their ranking. The sequential workload is a new controlled workload; its
numbers cannot be presented as a same-input speedup over the old heat trace.

The existing detached tree
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-gr-serving-worktrees/sequential-u16-t32-h64k-02`
exists at `16dc058` with copied dirty first-party sources. Read-only inspection
found no live serving process and no accepted sequential run-02 output. Its
`experiments/gr_serving/output` resolves to the main checkout's output tree.
It predates these operator changes and must not be launched as the optimized
implementation without refreshing and freezing an exact current-source copy.
Do not modify or re-sign the failed run-01 artifacts. Use a fresh isolated
source tree, include the new MLA/linear configuration modules, preserve the
strict final source guard, and keep that tree frozen through independent audit.

The following is the planned command from that frozen tree's repository root;
it has **not** been launched. Recheck physical GPU 3 is idle before launch.
The existing launcher stages failures outside experiments and promotes data
only after its measurement checks complete.

```bash
CUDA_VISIBLE_DEVICES=3 \
TMPDIR=/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-gr-serving \
  bash experiments/gr_serving/scripts/run.sh \
  gr_serving_sm90_20261003_mfu_sequential_deepseek_01 \
  --models deepseek_v32 --users 16 --requests 32 --sampling sequential \
  --history-tokens 65536 --candidate-tokens 128 --seed 42 \
  --hbm-budget-gib 4 --dram-budget-gib 64 \
  --deepseek-slots 32768 --deepseek-layers 10 --chunk-size 1024 \
  --deepseek-path /preset-models --device cuda:0 --warmup 2 --atol 0 --rtol 0

.venv/bin/python -m experiments.gr_serving.src.audit \
  experiments/gr_serving/output/data/gr_serving_sm90_20261003_mfu_sequential_deepseek_01 \
  --repo "$PWD" \
  --expected-run-id gr_serving_sm90_20261003_mfu_sequential_deepseek_01 \
  --reference-check cpu \
  --json experiments/gr_serving/output/data/gr_serving_sm90_20261003_mfu_sequential_deepseek_01/audit.json
```

The CPU auditor supports a DeepSeek-only model subset and derives its expected
case/request matrix from metadata. It independently checks sequential order,
null heat identity, token hashes, every numerical record, saved HBM tensor
shape/dtype/finiteness, declared reservations, sampled allocations, LRU outcomes
and report arithmetic. It cannot rerun comparisons from saved non-HBM outputs,
because those outputs are not persisted; the in-process per-request numerical
gate remains required. MLA full/split equality must hold at the 64-query dispatch
boundary before this exact-tolerance serving run is accepted.

### Hard-budget limitation that must remain explicit

The current reservation/audit code does not prove the requested hard cache
budget throughout execution. `estimate_session_bytes` counts persistent KV,
resident index keys/scales, maps/ages, retained counters and dense stages;
`session_bytes` measures those retained tensors. `Cache.ensure` unique/sort/remap
scratch and fused-prefetch `free_slots`, `allocations` and `page_table` storage
are not reserved separately before allocation or sampled while live. The
runner audits after extend has returned and after truncate, when these
temporaries have already been released. Ordinary activations and GEMM workspace
are separate, but cache-management scratch cannot be reclassified as ordinary
activation to satisfy the cache contract.

At the authorized geometry, the existing formula reserves 472,036,520 HBM bytes
per ECHO session. Nine retained sessions leave 46,638,616 bytes under 4 GiB;
the formula alone gives no proof that live cache scratch fits that remainder.
This is a static accounting gap, not a measurement of an actual overrun. A
successful exact numerical run and CPU artifact audit cannot close it. Before
publishing hard-budget serving conclusions, bound and reserve cache scratch
before allocation, verify its live footprint, and rerun if doing so changes
admission/LRU. Until then the planned run can only establish implementation
numerics and latency under the explicitly disclosed reservation/boundary-sample
measurement limit. No serving or cache-policy change is made by this audit.

### NOSA provenance and artifact retirement

No accepted NOSA sequential 16/32 result exists to reuse: run 01 failed the
source-identity guard, and the accepted older NOSA 64K run used a different heat
trace and 4/16 GiB budgets. Unchanged NOSA code does not make those measurements
valid for the new sequential workload. Preserve its old data and layer-31
profiles with their original run IDs and boundaries; do not place those numbers
in a new sequential comparison table or assign them the new DeepSeek source ID.

Publish an accepted DeepSeek-only replacement as its own four-case sequential
section with its own sources, tokens, audit and limits. The previously requested
complete eight-case sequential experiment remains pending NOSA's first valid
four-case run; if completed separately, its run/source/input identities must be
explicit in the combined navigation and per-model tables. A new NOSA native
profile is needed only for a new NOSA workload report, not merely because a
DeepSeek implementation changed. Do not delete the mixed old `report/h64k/` or
the old latency/profile run directories while they still supply retained NOSA
evidence. Retire affected DeepSeek report entries only with accepted replacement
publication; keep 4K/16K marked with their old implementation and pending status
until their replacement is authorized. None of this reinstates the withdrawn
old user-scale conclusions.
