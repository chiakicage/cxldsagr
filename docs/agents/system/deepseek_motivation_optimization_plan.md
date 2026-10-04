# DeepSeek motivation launch and cache optimization

## Task contract

User objective: bring all three offload schemes' first-visit latency to the
HBM-only level, and continue improving every scheme's end-to-end MFU toward
its measured, precision-weighted matrix API MFU. This is an active optimization
task, not a diagnostic-only task. No smaller milestone constitutes completion.
After the MFU optimization is complete, rerun and update the experimental
comparison in `experiments/deepseek_v32_echo_official/`, as explicitly requested
by the user on 2026-10-04.

The user prefers avoiding `torch.compile` and explicitly switched the requested
kernel implementation to handwritten Triton on 2026-10-04. Write needed kernels
through KDA. The linear activation quantizer has been replaced with a local
Triton candidate, with production acceptance in progress. Retain the
official helper as an isolated correctness and complete-API timing oracle.
Do not treat `cute.compile` of a manually authored CuTe kernel as a new
`torch.compile` dependency.

Workload stays at ten independent checkpoint dense-block copies, H=65536,
A=128, chunk=1024, P=65536, NH=16777216, sixteen users in two sequential
rounds. All candidate hidden and last-token logits remain in scope. History
writeback, admission, cleanup, and required synchronization stay timed.
Preserve exact selection, transaction safety, session LRU, token priorities,
host pages, independent weights/indexer state, and transient candidate semantics.
Do not reduce work, increase the pool, or charge an invented memory reserve.

Authoritative baseline: formal
`deepseek_v32_motivation_20261004_p65536_nh16777216_u16_r2_02` and diagnostic
`deepseek_v32_motivation_profile_20261004_01`. First visits: HBM 2872.185,
ECHO 4792.370, serial 4099.002, dense 5285.434 ms. Matrix-weighted API MFU
for a first request is 53.322/41.117/53.206/53.223%; end-to-end MFU is
33.682/20.186/23.601/18.303%. Revisit candidate matrix API MFU is
27.862/17.468/28.473/28.493%. These are profile observations, not fixed roofs
or a promise of achievable latency. Recompute them after implementation changes.

## Draft: evidence and candidate order

1. Cold history has zero H2D but repeated allocation sorting, unique, scalar
   synchronization and whole-history dense scans. Eliminate zero-count work,
   same-stream event dependencies, and waits before HBM-only reads. Add an
   explicit conservative residency proof so ECHO can use the official resident
   indexer and dense can omit all-hit scan/copy work. Reuse allocation order only
   while its ownership and priority assumptions remain true.
2. Remove common redundant causal masking and single-chunk block output copies.
   Keep FP32 norm/residual arithmetic and independent replay inputs. Consider
   fused metadata/selection helpers and launch replay after the safe baseline.
3. Profile the remaining launch gaps and top-k/nonmatrix cost. Optimize actual
   dominant components; measure revisit fetch/compute overlap and total latency.
   Full CUDA Graph support cannot be presumed from resident capture alone.

Risks: other sessions invalidate residency; a selection refresh changes FIFO
priorities; releasing/reusing a host page must not leave certificates alive;
skipping a wait must never expose incomplete D2H/H2D data; candidate counter
definitions must remain exact. A larger chunk or smaller experiment cannot
stand in for the requested workload. Profile instrumentation inflates wall time.

## Executable plan

- [x] Inspect current tree, accepted reports and process state (2026-10-04).
  No motivation measurement process or CUDA workload was live at inspection.
- [x] C1: cache/all-hit launch reduction. Test CPU state transitions and Hopper
  ordering/eviction/recall, then screen complete H+A requests of all four schemes.
- [x] C2: shared projection/block/selection launch reduction, independent of C1.
  Compare all outputs; measure complete requests with fresh run ID.
- [ ] C3+: use new traces to eliminate remaining cache and common launch costs,
  preserving operator API semantics and workload. Iterate to the full objective.
- [ ] Run full 16x2 trajectory, all-output checks, source identity verification,
  Nsight profile and precision-weighted MFU analysis for the promoted source.
- [ ] Publish new report and evidence, then replace affected prior reports and
  outputs in the same publication change. Check shared-module dependent
  experiments; mark them awaiting rerun until their own evidence is replaced.

Commands (from repository root, native Hopper environment):

```bash
.venv/bin/python -m pytest cache/tests/test_sparse_token_cache.py cache/tests/test_sparse_token_pool.py cache/tests/test_transient_suffix.py models/deepseek_v32/tests/test_pool_prefetch.py
CUDA_VISIBLE_DEVICES=0 bash experiments/deepseek_v32_motivation/scripts/run.sh --run-id <new_screen_id> --num-users 2 --rounds 2
CUDA_VISIBLE_DEVICES=0 bash experiments/deepseek_v32_motivation/scripts/run.sh --run-id <new_full_id>
CUDA_VISIBLE_DEVICES=0 bash experiments/deepseek_v32_motivation/scripts/profile.sh --help
```

Screening is engineering evidence, not final 16-user acceptance. Keep rejected
or failed runs outside `experiments/`. Freeze production source while a run is
active. Existing valid publication remains until replacement is validated.
Each candidate record must identify parent, source hash, correctness, measurement
boundary, hardware, warmup/repetition count, performance, and next decision.

## Completion audit

Compare all offload first-visit distributions against contemporaneous HBM-only;
report ratios and variability, not only a best sample. Compare E2E and weighted
matrix API MFU separately for first history and revisit candidate. Account for
remaining top-k, norm, fetch and gaps rather than defining them out of E2E.
No completion claim while any scheme is materially launch-bound or the requested
latency/MFU convergence remains unproven. Retain the full objective on continuation.
