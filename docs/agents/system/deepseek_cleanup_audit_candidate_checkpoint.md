# Cleanup audit reuse: temporary CPU candidate

Status: deferred at the user's requested C10 closure, 2026-10-04. C11 cleanup
is not promoted. Core code is restored to accepted frozen C10, and the new
cleanup fixture/tests are archived and removed. The archive manifest is
`/tmp/deepseek_cleanup_production_20261004/deferred_source/core_manifest.json`.
The evidence below describes the historical candidate only.

2026-10-04. V1 passed the listed tests and CPU timing, but independent review
then found an outer-hook trust gap: an unknown hook can mutate storage after
obtaining a current receipt and bypass audit two. Production integration is
held. V1 files and measurements are preserved; a separate V2 prototype adds
caller-side canonical-hook checks before and after invocation. The V1 timing
is not V2 performance evidence. Root owns selection, GPU validation and formal
reruns. No production cleanup code or published experiment result was changed.

The design contract and source proof are in
[the candidate plan](deepseek_cleanup_audit_candidate_plan.md).
All executable probes and engineering evidence are under
`/tmp/deepseek_cleanup_audit_20261004/`; they are not paper experiments.

## Correctness and source boundary

The copied `truncate` issues a typed receipt only at the existing successful
transient no-op return. AST restoration proves that its guards, exact no-op
predicate and mutating path are otherwise unchanged. A fresh identity-bound
ticket ties the receipt to the current owner, session, prefix and cleanup
call. The adapter returns it only after the original synchronization succeeds
and only for recognized truncate, synchronize and validation methods.

The copied runner changes only the cleanup segment. It always performs the
first complete audit. Exact fresh receipts reuse that result; absent hooks,
unknown methods, ordinary mutations, wrong types and stale receipts perform
the second audit. The existing exception, discard, failed-release and owner
retention paths remain unchanged. This proof depends on the current serial
owner contract, not concurrent serving.

- 34 focused CPU tests compare complete outputs and every metric field with
  deterministic clocks; cover fixed/budget modes, first/repeat visits,
  candidate changes, eviction, first-audit storage/quota checks, real storage
  growth behind empty aliases, unknown cleanup mutations, stale certificates,
  cleanup failures and failed-release ownership. Per-phase results and source
  identities are in `parity_run.json`, with separate stdout/stderr logs.
- 393 unchanged CPU tests passed with the temporary methods installed only
  inside the test process. Five CUDA tests were deselected; one optional NOSA
  checkpoint test was skipped because `NOSA_SERVING_CHECKPOINT` was unset.
  This includes actual small NOSA model eviction/reconstruction tests,
  ownership/resource-drain tests, DeepSeek cleanup, and storage accounting.
  `regression.json` retains all 1,182 phase reports and 121 before/after source
  identities. No CUDA initialization occurred.
- The regression driver received only import blank-line formatting after
  execution. `regression_executed_source.txt` preserves the executed bytes;
  `regression_driver_format_followup.json` proves AST equality and binds both
  hashes. The production/prototype implementation did not change.
- Root's initial source review found no structural blocker; the subsequent
  independent review found the gap above. Reproductions are retained in
  `/tmp/deepseek_cleanup_audit_outer_hook_repro.py` and its JSON. The earlier
  test suite did not cover a mutating outer hook and does not discharge this
  issue. Independent benchmark arithmetic/identity checks are retained in
  `/tmp/deepseek_cleanup_review_20261004/benchmark_independent_audit.json`.

## CPU storage geometry and timing

Both benchmark processes used H=65,536, ten layers, one or sixteen independent
sessions, and fixed-pool plus byte-budget modes. Each layer has actual CPU
FP8 `[65536,128]` index keys, FP32 scales, offset and prefix-offset storages.
Actual `SparseTokenSession` page tables and full counter backing storage are
charged by the unchanged production accounting methods. The shared fixture
contains 88 real small tensors totaling 5,632 bytes.

Per-session storage is 86,521,648 bytes. Sixteen sessions plus shared storage
total 1,384,352,000 bytes per runner. Indexer tensors are allocated but not
fully touched. These are real backing-allocation capacities, not process
resident or pinned DRAM. The scaled shared fixture is not a physically filled
NH host arena. The processes peaked at 533,004 and 536,068 KiB RSS. No model,
checkpoint, host transfer or CUDA barrier executes; the barrier is a plain
Python no-op while its original DeepSeek wrapper remains in the call chain.

The CPU is an Intel Xeon Platinum 8558P, with Python 3.12 and the project Torch
environment. Each process uses ten warmups, then 41 randomized paired blocks
of ten calls per implementation. Seeds are 20261004 and 20261005. GC remains
enabled. All 1,312 paired blocks are retained, totaling 26,240 measured complete
cleanup calls. Each call includes the first full audit, actual truncate,
barrier wrapper, and either audit two or all ticket/receipt checks. Population,
description, count checks and closure are outside timing. Source identities
match before/after each run, and storage results and audit counts agree.

The table reports median microseconds as baseline → candidate, followed by
the median paired saving. Negative savings mean candidate overhead. `Absent`
has no optional hook; `Unknown` wraps the inner truncate and requires audit
two; `Extracted` models the added bound method call if production extracts its
default cleanup body. The latter is an integration probe, not production code.

| Users | Budget mode | Path | Run 1: median / paired saving (µs) | Run 2: median / paired saving (µs) |
|---:|---|---|---|---|
| 1 | Fixed | Certified | 256.325 → 132.805 / +123.806 | 264.853 → 135.867 / +128.748 |
| 1 | Fixed | Absent | 258.669 → 259.101 / −0.375 | 264.253 → 264.876 / −0.125 |
| 1 | Fixed | Unknown | 257.167 → 259.171 / −1.279 | 263.828 → 265.691 / −1.783 |
| 1 | Fixed | Extracted | 255.788 → 257.143 / −1.282 | 261.512 → 262.324 / −0.831 |
| 1 | Bytes | Certified | 261.096 → 135.564 / +125.721 | 266.427 → 137.864 / +128.672 |
| 1 | Bytes | Absent | 260.626 → 262.484 / −1.712 | 265.263 → 264.816 / +0.407 |
| 1 | Bytes | Unknown | 262.235 → 262.667 / −0.409 | 267.849 → 268.858 / −0.543 |
| 1 | Bytes | Extracted | 260.734 → 262.209 / −1.375 | 267.918 → 266.917 / +1.106 |
| 16 | Fixed | Certified | 1636.188 → 826.762 / +810.012 | 1655.528 → 835.823 / +820.591 |
| 16 | Fixed | Absent | 1619.872 → 1618.626 / +0.392 | 1659.449 → 1654.602 / +3.659 |
| 16 | Fixed | Unknown | 1626.839 → 1628.427 / −3.804 | 1657.489 → 1654.690 / +3.515 |
| 16 | Fixed | Extracted | 1621.129 → 1626.311 / −3.911 | 1659.105 → 1664.703 / −1.266 |
| 16 | Bytes | Certified | 1641.122 → 828.616 / +812.106 | 1678.788 → 845.920 / +831.445 |
| 16 | Bytes | Absent | 1639.459 → 1641.113 / −2.464 | 1680.815 → 1676.847 / +3.308 |
| 16 | Bytes | Unknown | 1641.125 → 1647.025 / −5.051 | 1679.988 → 1683.224 / −1.042 |
| 16 | Bytes | Extracted | 1638.621 → 1626.418 / +10.195 | 1678.534 → 1679.547 / −1.657 |

Certified sixteen-user cleanup saves 0.810–0.831 ms in these CPU fixtures,
with paired speedups of 1.981–1.985. One-user speedups are 1.927–1.948.
Fallback/default-helper paired changes remain below 0.7% in magnitude and
change sign in several cases. This supports a small overhead boundary, not
zero-cost claims or a precise isolated method-call cost. Outliers remain in
the raw samples. These values cannot be subtracted from a formal GPU run or
presented as full serving/MFU gains.

Raw data: `benchmark_h64k_01.json` and `benchmark_h64k_02.json`, each with
separate stdout/stderr logs. `benchmark_summary.tsv` retains all per-case
medians, paired deltas and speedups. `driver_prepare_02.json` verifies all
sixteen fixture/call combinations without timing at H=128.

## Provisional integration shape, held for V2 review

1. Put the typed optional contract beside `SharedCachePlan` in
   `executor/serving_backend.py`, avoiding a model dependency on the serving
   implementation. The ticket binds an opaque admission owner, session and
   prefix; the receipt binds that exact ticket. Do not change the mandatory
   backend protocol for NOSA or generic backends.
2. Add an explicit runner option, disabled by default. Extract the four
   cleanup statements into a small default helper and bind the selected
   receipt-aware helper once during construction. Keep one `execute` body;
   do not ship AST rewriting or duplicate the full request wrapper. The
   extraction adds a default-path call, so recheck that actual production
   integration against the saved baseline. The `Extracted` rows only measure
   the temporary model of this change.
3. Add the DeepSeek optional hook and issue the receipt directly at the
   existing exact no-op return. Preserve public `truncate` behavior when no
   private ticket is passed. Keep the trusted-method/owner checks and only
   return after synchronization succeeds. No flag-only certification, audit
   arithmetic changes, cross-request cached measurements or weakened quotas.
4. Resolve the selected outer hook at cleanup time. For the selected profiler
   path, instrument that hook while leaving its inner truncate/synchronize
   methods unchanged. Test one-audit parity under the transparent outer
   profiler wrapper; unknown inner wrappers must still use two audits.
   Default profiler instrumentation remains as it is.
5. Rerun complete CPU outputs/metrics/failure parity against the integrated
   code, then root's applicable GPU and complete-serving checks with new
   source/run identities. Keep NOSA's option disabled by default. Preserve
   existing experiment artifacts until accepted replacements are published.

The above V1 proposal is superseded where it allowed transparent wrappers of
the backend cleanup hook. V2 supplies an explicit canonical unbound hook at
the integration boundary and checks its bound self/function at the caller
before and after invocation. Unknown outer hooks execute normally but always
take audit two. Selected profiling wraps the runner cleanup helper, leaving
all guarded backend methods unchanged. No trusted-wrapper exception or
registry is needed. V2 must pass the new adversarial tests before integration
can proceed; its timing remains to be measured separately.

## Core identities

| Source or artifact | SHA256 |
|---|---|
| `candidate.py` | `966071d754063f35ecdab51e59b99f9e0b250dd632ae68a3a673cd2b7d410993` |
| `fixtures.py` | `21df0be523b5b28f170b581ea279ae1ab57de1487dce7287bd9c7d3de0e32857` |
| `benchmark.py` | `09e78212459e964ff00c6b8db1da6e9910e692425455d04173da9d297885347a` |
| Current `serving/persistent.py` | `c4bfb300e4ef98d0d7bb94825d26770c4d2b600eb430549b726a1941b78ae572` |
| Current DeepSeek `serving_backend.py` | `23d63e3fd08b63a9c7afb4fab554deb6f426ffecac22b37939c10cf198d2f67a` |
| `benchmark_h64k_01.json` | `1a11beb6bb0e7ff816d49de46ac38c844dfe814fe0d85520033e4bc67dfef983` |
| `benchmark_h64k_02.json` | `0ac8a4f6d23a6d9ecc74e916522c29031dfcb4db0daf1db627ae9bece178f1eb` |
| `regression.json` | `19b5cf4c56bb685e3b6b6d0ba7bcdaaa61e4b9d503079dd03d83738671d03217` |

Each result file carries the full runtime/prototype source mappings. The
timing window has ended and was returned to root; there are no remaining
benchmark or GPU processes from this candidate.
