# Official ECHO independent publication audit

This checklist applies to
`deepseek_v32_echo_official_20261004_p65536_nh16777216_u16_r2_01`.
It supplements the [validation contract](deepseek_echo_official_validation.md).
It is an internal execution document, not a performance report.

## Audited state

The formal run completed successfully in the frozen tree
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-official/20261004-01`.
The root agent observed unified session 83884 exit code zero and the run script's
`Completed` message; PID 1620428 exited. The initial temporary data directory was
`/tmp/deepseek-echo-official-deepseek_v32_echo_official_20261004_p65536_nh16777216_u16_r2_01-a56c1j1i`.
Accepted data now resides in the frozen experiment's `output/data/<run_id>/`.
The independent frozen report audit and additional aggregate/source/memory
checks passed. Publication copies in the main worktree were then independently
verified: all 2,426 data files and both log files match the accepted frozen run,
and all six published report files match its generated report byte-for-byte.

Initial formal source identity:
`b98ac0b4384e5b58f6f3c62cf9fe8ae341ddd57c4b5293d6a7d66e202fbcefed`.
The observed workload identity matches the motivation reference:
`7e4c737a86464c12238191933e707344426231bf671a29658837e676ff5284ae`.

## Completed independent checks

The frozen `src.report.audit_run` returned `passed`: all 64 formal requests,
32 independent resident-repeat requests, six warmup requests and all 96 saved
output payloads were rechecked. A separate read-only pass verified all 1,254
live frozen source hashes, 885 official source hashes, eight native/JIT build
artifacts, all four summary groups, and every official request's layer capacities
and shared/session memory ledger. Report metadata matches final raw metadata.

The physical device was H200 GPU 3, visible as cuda:0, UUID
`80ff95c3-176e-fd8a-728f-9c5577c4a779`, with 132 SMs and
150,121,545,728 bytes of visible HBM. Python 3.12.13, PyTorch 2.12.1+cu130
and CUDA 13.0 were recorded. Both schemes report 7,827,793,408 parameters and
9,864,625,792 weight bytes. FP32 `highest`, disabled matmul TF32, enabled
BF16/FP16 reduced-precision reduction and enabled cuDNN TF32 remained unchanged.

Independent aggregation gives:

| Scheme | First mean ms | Revisit mean ms | Revisit p95 ms | Full trace s | Revisit retained histories |
|---|---:|---:|---:|---:|---:|
| Fresh official resident HBM | 2341.777291807 | 2339.174924178 | 2346.279142264 | 74.895235456 | 0/16 |
| Official ECHO adaptation | 4908.212345934 | 66.321755010 | 67.847446480 | 79.592545615 | 16/16 |

Candidate ECHO H2D is 1,246,440,960 bytes: 809,583 prefetched records plus
272,397 residual-recall records, each 1,152 bytes. Candidate D2H is zero for
both schemes. These counters do not include history construction. No internal
overlap measurement or selection hit-ratio claim is established by this run.

| Comparison | Output | Maximum relative L2 | Minimum element fraction | Maximum absolute error |
|---|---|---:|---:|---:|
| Resident repeat / primary HBM | hidden | 0.0024777187163 | 0.9999291556222 | 0.1796875 |
| Resident repeat / primary HBM | logits | 0.0040085095065 | 0.9999922648515 | 0.0625 |
| Official ECHO / primary HBM | hidden | 0.0024654522294 | 0.9999291556222 | 0.1796875 |
| Official ECHO / primary HBM | logits | 0.0032308607742 | 1.0 | 0.0625 |

Every tensor comparison passes the frozen relative-L2 and 99.9% element
fraction gate. None is bitwise equal across independent runs. The resident
repeat has 29/32 allclose hidden outputs and 31/32 allclose logits; official ECHO
has 29/32 and 32/32 respectively. The numerical observations do not prove that
all differences arise from rounding rather than ordering or cutoff ties.

### Memory reconciliation

`PrefixSessionPool.audit()` starts with actual shared storage and adds every
session. Therefore `cache_hbm_bytes` and `cache_dram_bytes` already include
`shared_cache_*`; adding the shared fields again would double-count them.

At the final official request, actual cache HBM is 2.666889845394 GiB:
1.377743660472 GiB shared plus 16 sessions of 86,513,136 bytes each. Actual
cache DRAM is 320.001037597656 GiB: 320.0009765625 GiB shared plus 16 CPU
page tables of 4,096 bytes. The official extra shared allocation is 25,428,897
HBM bytes and is counted. The shared reservation is 3.512757300399 GiB;
execution allowances make it larger than currently owned shared storage.
Reservations must not be described as already allocated buffers.

CUDA allocated/reserved peaks are 11.903460026/18.353515625 GiB for HBM and
13.765183926/19.103515625 GiB for ECHO. Maximum device-used memory among
request-boundary samples is 19.024475098/20.469787598 GiB respectively;
these samples are not continuous device peaks.

The ten pinned record backings allocate 320 GiB, while their logical NH records
occupy 180 GiB. Final `allocated_bytes.current` in the host allocator is
343,597,383,692 bytes (320 GiB plus 12 bytes of small allocator blocks).
The host allocator reports `active_bytes.current=687,194,771,832` with
`active_bytes.freed=0`. The installed PyTorch
`ATen/core/CachingHostAllocator.h` explains why this statistic can accumulate:
the no-event `free()` branch returns a block to its free list without decrementing
active stats, and `get_free_block()` increments active stats on reuse; the event
completion branch does decrement them. This is consistent with warmup/formal
reuse doubling reported active bytes. It is not evidence of 640 GiB physical
backing. Owned-storage accounting and allocator allocated bytes agree; no
memory-accounting blocker was found.

The separate mirror-test result was confirmed directly by its executing agent:
the original command ran `pytest -s -q --tb=short` on
`models/deepseek_v32/tests/test_official_checkpoint.py` with physical GPU 2 and
the explicit `/preset-models` opt-in. Its terminal ended
`1 passed, 15 warnings in 221.57s (0:03:41)` with exit code zero; selected KV
and same-ordered FlashMLA errors were zero. No persistent test artifact was
created. Calibration and checkpoint top-k JSON hashes also match the validation
contract; the top-k audit's request-0 token hash matches formal request 0 even
though its whole-workload manifest has a different sampling scope.

### Main-worktree publication verification

The published README reports the correct full-trace direction: 79.593 s for
ECHO versus 74.895 s for the fresh resident control, a 6.27% increase. It limits
the 35.270 revisit ratio to end-to-end requests with different history-rebuild
requirements. It identifies the frozen source, official resident/top-k control,
unmeasured subsequent main-worktree optimizations, actual-versus-planned memory,
and the separate numerical and supplementary-test boundaries.

The independently checked published report hashes are:

| File | SHA-256 |
|---|---|
| numerical_summary.json | `30e239d0485c00dabe39ebc1015adbfed7b45cffb9735b7a29117ae85442c6f3` |
| per_request.csv | `7793e5a875761d8564d297876353e3bbdfbc5de4a58aab83952dd74445612b61` |
| report_provenance.json | `f19c8710256835a24d69d62d989508f7c931d2037b78521e686d8f2381171396` |
| results.md | `84f93bf7363e37e842eb8c6e36457d74c3d6e16d21f451b8c47c131b91cfd04b` |
| summary.csv | `530299306254ef68849e9769a4106b2f92dff2e093ece0bb5d5ce435127114d4` |
| summary.json | `34c462d718bfc30c2ca9f4ade18be6e6d3ae64f54872d9297d639cf6a3fcb199` |

No numerical, identity, timing-summary, memory-accounting or publication-copy
blocker remains in this independent audit. This does not certify arbitrary
inputs or performance of subsequent main-worktree code changes.

## Acceptance evidence required

1. **Terminal successful execution.** Inspect the real process/session status,
   successful command exit, final `metadata.json`, and the accepted output
   directory. A running metadata file, partial report, or missing observation
   handle is insufficient. Do not restart because a polling call times out.
2. **Identical requested model and capacities.** Verify H=65,536, A=128,
   chunk=1,024, P=65,536, NH=16,777,216, 16 users, two sequential rounds,
   seed=42 and ten independent checkpoint source-block copies. Confirm
   7,827,793,408 parameters, source input replay, FP8 linears, all candidate
   hidden and the last-token LM head. Inspect the source/model inventory,
   backend description, resource plans, per-layer capacities and request data.
3. **Actual official execution.** Confirm pinned ECHO revision
   `bc1b75c1000010d0ac6f032ebaac283255c050b1`, original resident/fused logits,
   original top-k, allocator and residual recall, with no local fused-kernel
   fallback or all-hit bypass. Check runtime source and native identities, not
   labels alone. Preserve the explicit single-GPU GR adaptation, no-Hadamard
   projections and shared FlashMLA boundary.
4. **Fresh cache lifecycle.** Inspect both three-request warmup traces, including
   real ECHO host recall; then verify fresh formal allocation and all 64 formal
   request rows in scheme-major order. Require 16 HBM misses on revisit and 16
   ECHO retained-history hits for this trace. Visit identity must remain separate
   from retention. Every candidate is one batch, writes zero host KV and leaves
   only H tokens. Host and HBM quotas must match the specified P/NH.
5. **Independent resident repeat.** Require all 32 `hbm_repeat` outputs, an empty
   initial cache, the same admission/resource plan, and matching request IDs and
   token hashes. Its validation rows must not publish latency. Formal HBM peak
   memory must have been captured before the repeat; official ECHO resets its
   peaks after its own warmup.
6. **All saved output values.** Recompute CPU FP64 errors from all 96 saved
   output payloads. Check finite values and matching shapes/dtypes. Every repeat
   and offload request must have relative L2 <=0.005 for hidden and <=0.01 for
   logits, with at least 99.9% of elements satisfying
   `abs(actual-reference) <= 1/32 + abs(reference)/64`. Recheck stored output
   hashes and reported metrics. All-element closeness, maximum absolute error
   and bitwise equality are separate observations. Show repeat and offload
   error envelopes separately; do not attribute every difference to rounding.
7. **Independent data-path evidence.** Inspect the full-checkpoint mirror-test
   code, confirmed terminal result and source identity for all 1,310 calls, 2,650,296,320 selected-record
   occurrences and same-ordered FlashMLA comparisons. Verify correspondence to
   the actual frozen runtime modules. Inspect the checkpoint top-k audit of 20
   calls / 11,520 rows, including unique indices, exact selected-score multisets
   and maximum cutoff bucket occupancy 5,513 versus capacity 16,384. Do not
   generalize this sampled top-k audit to every possible input. The mirror test
   has no persistent result artifact, in accordance with the project's test
   rules; do not invent an artifact path or turn its terminal output into an
   experiment result. Its reported session 56228 exited zero, with one test
   passing in 221.57 seconds. The separate formal tensor audit is mandatory.
8. **Timing and counters.** Recompute summaries directly from formal rows;
   check nonnegative finite stages and their sum to request wall latency.
   Include GPU counter accumulation/reductions inside forward. Host readout,
   tensor saving/comparison, compilation and warmup stay outside latency.
   Candidate transfer counters exclude history construction. Do not infer
   unmeasured selection hit ratios or internal overlap from end-to-end time.
9. **Physical memory.** Reconcile per-scheme peaks with raw boundary samples;
   report allocated, reserved and device free/used separately. Check actual
   shared/session storage accounting and official extra metadata against plans.
   P/NH equality does not mean equal total bytes. No empirical headroom or byte
   subbudget may be subtracted from this fixed-capacity experiment.
10. **Precision and immutable identity.** Require FP32 `highest`, matmul TF32
    disabled, recorded BF16/FP16 reduction flags and cuDNN TF32, and unchanged
    final settings. Verify all source snapshot hashes and complete official
    source/native snapshots. Warmup may append JIT artifacts; final native
    inventory must remain unchanged after official warmup. Audit with the
    frozen implementation because the main worktree is changing concurrently.

## Publication requirements

- Run the frozen `src.report.audit_run` and an independent aggregate check after
  successful completion. Record which evidence is directly checked and which is
  inherited from the separate full-checkpoint validation.
- Preserve accepted formal logs, outputs, manifests and official source/native
  copies. Keep calibration provenance and the terminal/code basis of the
  supplementary checkpoint test explicit in internal records; do not present
  standalone correctness tests as experiment results. Only accepted formal
  output belongs under the new experiment's `output/`.
- Copy report artifacts only after acceptance. Published numbers must derive
  from this run, and report hashes must match copied artifacts. Include the
  actual GPU, dependencies, run ID, source identity, workload, precision policy,
  sample count and numerical criteria.
- Update the new experiment README and index together. Explain that the HBM
  control uses official resident logits/top-k, while weights, projections and
  FlashMLA retain the motivation scope. Any separate old motivation timing keeps
  its own run ID and is not a matched new control.
- Preserve existing motivation reports. State that concurrent changes in the
  main worktree were not measured by the frozen official run. Do not label
  unmeasured newer production code as the measured implementation.
- Record the final independent audit result here only after the checklist has
  authoritative evidence; a checklist alone does not establish completion.

## Additive comparison of existing results

The user subsequently requested that the report include the existing local
implementation's performance and explicitly ruled out another performance run.
The added [comparison report](../../../experiments/deepseek_v32_echo_official/report/existing_implementation_comparison.md)
therefore uses the accepted motivation run
`deepseek_v32_motivation_20261004_p65536_nh16777216_u16_r2_02` and the accepted
official `_01` run audited above. It preserves each run's own HBM control.
No GPU execution, new timing, or cross-run tensor comparison was performed for
this addition. The local row describes its published pre-optimization source,
not the unmeasured C3 implementation in the current worktree.

Independent raw aggregation confirms all four rows:

| Source / scheme | First mean ms | Revisit mean ms | Revisit p95 ms | Full trace s |
|---|---:|---:|---:|---:|
| Motivation / HBM | 2872.185398373 | 2874.508268247 | 2885.521579985 | 91.947098666 |
| Motivation / local ECHO | 4792.370115563 | 70.038766189 | 71.553977978 | 77.798542108 |
| Official / HBM | 2341.777291807 | 2339.174924178 | 2346.279142264 | 74.895235456 |
| Official / ECHO | 4908.212345934 | 66.321755010 | 67.847446480 | 79.592545615 |

The local source manifest is
`006cdcddae8276b39ede88f55dd4e282e543f938127f7a35abdf0e081c336cb9`.
All 1,240 saved local source hashes and all 128 saved output-payload hashes
were rechecked. Each of the 96 local offload payloads is file-identical to its
corresponding HBM payload. This verifies retained artifacts; it does not rerun
the numerical experiment. All eight original local summary groups and four
official groups agree with raw measurements, including exact stage sums.

Both runs' report-provenance input hashes match their original raw files.
Their `workload/workload.json` and `workload/requests.jsonl` are byte-identical;
all 32 request identities, history/prefix/candidate/input hashes, model inventory
fields, checkpoint stat inventories and model dimensions match. The common
workload hash is the one recorded above. Each scheme contains 16 first visits
and 16 revisits with the same three-request warmup and an empty formal cache.

The physical GPUs differ: local UUID
`1bdee8b4-22ac-536c-208b-bfb4ed38b878`, official UUID
`80ff95c3-176e-fd8a-728f-9c5577c4a779`; both report H200, 132 SMs and the same
visible capacity and runtime versions. Local HBM uses mainline DeepGEMM
resident logits; local ECHO uses the local fused indexer/prefetch on each
offload call. Both use deterministic sorted FlashInfer top-k with its SMALL
tie rule. The official pair uses original resident/fused logits and original
top-k. The snapshots also differ, and local metadata does not record the full
FP32/TF32 policy. Neither run uses compute graphs. These are separate accepted
trajectories, not a controlled estimate of the cache implementation alone.

The additional CSV/JSON figures, source and generator hashes, GPU identities,
allocator peaks, and observation-only percentage differences were independently
verified. Local ECHO has 2.360% lower first-visit mean and 2.254% lower total
time, but 5.605% higher revisit mean, using official ECHO as the denominator.
The report makes no statistical-stability or new cross-run numerical claim.
All three published additions match the corresponding files under
`output/data/<official_run_id>/existing_comparison/` byte-for-byte. Their hashes
are:

| File | SHA-256 |
|---|---|
| existing_implementation_comparison.csv | `4a24b41d30465008df662bb1d8339e2307125a7f3fe9e23919dda92f4d5c9076` |
| existing_implementation_comparison.json | `b97637db44a5db46f9cc01ee0a7a469f139bd4f8fcf44e716f5282e9fa6767ac` |
| existing_implementation_comparison.md | `13ced73f138b94001c76b7fbfc003efa3ec0001cb5cd9277ca41668d76dbc3da` |

The CPU-only generator `src/compare_existing.py` has SHA-256
`f1296dc4fd8b3ea6f8c56abc5b83274fe83510dea78c9dcce64e547ac27de387`.
All six original official publication files still have the exact hashes listed
earlier in this audit. The addition is accepted within the stated existing-run
comparison scope.
