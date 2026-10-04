# ECHO allocator memory audit

> Experiment location: [deepseek_v32_echo_cache](../../../experiments/deepseek_v32_echo_cache/README.md).
> Historical commands, source paths and hashes below remain unchanged; use the
> [migration record](echo_cache_experiment_migration.md) to locate moved artifacts.
> This directory migration is not a new measurement.

Implementation: `experiments/gr_serving/src/cache_memory_audit.py` and
`experiments/gr_serving/src/cache_memory_capture.py`. The collector is independent
of runtime code and formal latency sampling. All four declared memory gates
have passed: complete four-scheme C1024 deployment and C2048 fixed-workspace,
C512 resident coverage, and C256 resident+dense-prefetch coverage. Independent
raw-artifact checks passed for all 11 cases, 352 requests and 172 sampled phases.
All original data/log/profile files were copied to main with SHA256 equality;
reproduction scripts and audit inputs are preserved as separately hashed sidecars.

The collector and CPU-reservation suite has 58 passing tests in 2.92 seconds,
including actual CUDA scalar and nonzero/unique transfers and an in-place
zero-allocation truncate, plus generation-order and missing-evidence rejection
tests (terminal session 28974, GPU 6, 2026-10-03). Another 95 audit
tests pass, including the new CPU scratch versus host-backing distinction and
the four-dependency source audit (terminal session 58562). These are correctness
tests, not experiments.

## Workload and capture boundary

The CLI defaults to `/preset-models` and executes the real 10-block checkpoint
workload surrogate with source-layer input replay, all candidate hidden states
and the last-token LM head. It does not claim full-61-layer validation. The
workload is the complete sequential 16-user, two-pass, 64K+128 GR trace.
C defaults to 256/512/1024/2048 and all four schemes. A declared `--schemes`
subset must contain `hbm` as its independent same-C numerical reference, contain
no duplicates, and uses canonical scheme order. `configuration.json` and the
manifest record the full arguments and expected case matrix. Acceptance requires
the exact completed case sequence and all 32 requests for every case; missing,
extra or short cases fail. NH is 1,050,624 and P is
32,768. Cache limits are 4 GiB HBM and 64 GiB CPU DRAM. Each C reserves its own
execution workspace by default. `--workspace-query-tokens 2048` instead reserves
one workspace across C values; that protocol retains at most 15 ECHO users.

Pure planning found all 16 scheme/C combinations feasible for at least one
session. These byte/page-ledger counts are predictions, not measured residency.
The revised pinned-bin screen preserves the counts at this explicit NH because
they remain HBM-limited; automatic NH selection is directly affected. Shared
ECHO host capacity is now 20 GiB plus CPU metadata and execution scratch; each
dense-prefetch session owns 1.25 GiB host capacity.

| C | hbm | echo / serial_sparse | dense_prefetch |
| --- | --- | --- | --- |
| 256 | 4 | 16 | 15 |
| 512 | 4 | 16 | 14 |
| 1024 | 3 | 16 | 12 |
| 2048 | 2 | 15 | 7 |

After one complete warmup request, each case starts with independent empty cache
state. Requests 0/15/16/31 are profiled for prefill (only on actual miss), extend,
truncate and session metrics. Remaining requests build the actual LRU and device
pool state normally. Diagnostic timings are discarded. Every request compares
all candidate hidden states and the last-token head exactly against its same-C
resident run. Admission and release retain existing reservation checks; these
profiler phases cover execution, truncate and metrics.

## Allocation evidence

The fixed-cache inventory explicitly enumerates pools, host pages, session page
and counter tables, indexer keys/scales/hints and dense staging. Shared storage
aliases count once. CUDA snapshots provide allocator block padding. CPU pinned
storage is independently charged at the next power of two of owned storage
bytes; ordinary CPU storage keeps its reported size. The revised runtime exposes
a complete pinned-bin backing storage through a logical tensor view. The
collector does not import the runtime's rounding helper.

The baseline is not a peak measurement. Available temporary bytes equal the
full admitted reservation minus rounded fixed capacity, including reserved hints
not yet materialized. CUDA logical storage and CPU charged storage reconcile
against the backend ledger. Diagnostic continuation may record an old CPU ledger
deficit, but it cannot turn that mismatch into a passing result.

Chrome trace `[memory]` events provide pageable CPU allocation bytes and CUDA
allocator block bytes. CPU launch scopes assign ownership; frees remain global
after scope exit. Projected `p.kv` markers retroactively charge full storage from
its original allocation through free. An ordinary result marker excludes final
prefix/candidate hidden concatenation. Named projection, MLA, MLP and output
stages remain ordinary activations. Unscoped work within a captured phase is
conservatively charged to cache helpers.

CUDA allocator history records `alloc`, `free_requested` and `free_completed`.
The current observer pairs complete per-address allocation generations in order,
including ordinary CUDA allocations. Every address must have identical profiler
and history generation counts; history requested bytes must fit the profiler's
rounded block; requested-free states must agree. Invalid allocation/free order,
reuse before completed free, missing generations or bytes, and truncated history
fail acceptance. Active peaks use history timestamps exclusively. A requested
free without a completed event remains live through the capture boundary.

The earlier observer used a one-microsecond cross-clock containment tolerance.
That was sufficient for accepted 03 runs but failed in the longer C256 trace:
profiler allocation timestamps could lag history allocation timestamps by about
12 microseconds and occur 1–3 microseconds after history requested free. The new
observer does not widen that tolerance or substitute profiler-live peaks for
history-active peaks. CPU and CUDA peaks remain separate device timelines.

Malformed traces, ambiguous generations, mismatched frees, missing projected
source markers and potentially truncated history fail acceptance. Category peaks
explain attribution; the simultaneous combined peak is checked because the
logits, selection and metadata reservation terms can cover different scopes.

Only truncate and session-metrics phases allow zero allocation events. They
still require cache scopes, fixed-inventory reconciliation and complete pinned
handout reconciliation. A completely empty CUDA phase additionally requires
allocator history with no `alloc` events. Prefill and extend still require
allocation evidence. Real dense-prefetch truncate performs ten in-place
`aten::copy_` operations and can therefore legitimately have no `[memory]`
events. The CUDA test rejects missing history or hidden allocator allocations
for an allegedly empty phase.

## Pinned host capacity and CPU execution scratch

Installed PyTorch is `2.12.1+cu130`, git
`7269437d655783a26cba32aa88195b741ff496aa`.
`ATen/core/CachingHostAllocator.h` uses `PowerOf2Ceil(size)` before
`allocate_host_memory(roundSize)`. A CUDA probe confirmed that a 4097-byte pinned
tensor owns an 8192-byte block while profiler `[memory]` emits no pinned event.

Each of the ten BF16/576 arenas at the fixed NH has 1,210,318,848 logical bytes
but owns a 2 GiB block. Total owned arena capacity is 20 GiB. The old logical
ledger omitted 9,371,648,000 bytes (8.73 GiB). Dense prefetch's 65,664-token host
buffer has 75,644,928 logical bytes per layer and a 128 MiB block, or 1.25 GiB per
ten-layer session. Both require reservation before allocation, including
stair-step changes during automatic NH planning.

`host_memory_stats()['allocated_bytes.current']` reports global active plus
cached blocks, including blocks no longer owned by a cache tensor. It is reported
separately from owned fixed storage. The documented `allocated_bytes.peak` sums
per-bin peaks and does not establish exact concurrent physical retention.

The installed allocator also omits an active-stat decrement when a pinned block
is freed without stream dependencies. Three repeated allocations of the same
4097-byte pointer produced `active_bytes.current` 8192/16384/24576, while
`allocated_bytes.current` remained 8192. The no-event free branch only returns
the block to the free list; active decrement occurs in event processing. Thus
neither active current nor active peak is used as an ownership or peak assertion.

The cumulative `active_bytes.allocated` counter still records each handout.
The observer temporarily wraps `Tensor.item`, `__int__`, `__float__` and
`__bool__`, restoring all four on exit. Every CUDA conversion records its input
dtype bin, call interval and handout delta. Its trace must contain
`aten::_local_scalar_dense`, `cudaMemcpyAsync` and `cudaStreamSynchronize` on the
same thread inside the call. The matching
[pinned-scalar implementation](https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/aten/src/ATen/native/cuda/CUDAScalar.cu)
allocates one pinned scalar, performs `memcpy_and_sync`, reads it and releases the
local tensor before returning. The observer charges the entire enclosing call
interval, preserving overlap with pageable allocations. It does not substitute
cumulative handouts for a live peak. Ordinary scopes remain ordinary; unscoped
input-validation conversions are conservatively cache helpers. Phase-wide
handouts must exactly equal recorded Python and native handouts; unexplained dynamic pinned
allocation fails acceptance.

Native CUDA operations also create pinned scratch without calling the Python
conversion entry points. The parser joins GPU `Memcpy DtoH (Device -> Pinned)`
bytes to `cudaMemcpyAsync` by correlation ID and finds the enclosing native
`aten::nonzero` or `aten::_local_scalar_dense` interval on the launch thread. One
contained `cudaStreamSynchronize` is required. Actual copied bytes determine the
allocator bin; the enclosing native call provides a conservative lifetime.
Python-wrapped scalar calls are excluded from this second inventory to avoid
double counting. Native buffers and pageable allocations share one peak timeline.

At the same pinned PyTorch revision,
[Nonzero.cu](https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/aten/src/ATen/native/cuda/Nonzero.cu#L173)
uses one int32 count below INT_MAX elements and retains the pinned count array
through the native function return. Its transfer is synchronous. Larger count
arrays would be charged from the actual copied bytes, not assumed to be 4 bytes.
[UniqueCub.cu](https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/aten/src/ATen/native/cuda/UniqueCub.cu#L71)
uses an int64 length and calls its native `.item()`. The CUDA synchronization
contract is defined by
[CUDAFunctions.h](https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/c10/cuda/CUDAFunctions.h#L105).
The minimal nonzero/masked-index/unique/unique-consecutive probe observed
4/4/8/8-byte handouts, with a combined 8-byte live peak. Its parser result also
included a Python int conversion and exactly reconciled all 32 handout bytes.

The real CUDA item/int/float/bool probe made four int64 handouts totalling 32
bytes; their nonoverlapping call intervals had an 8-byte peak. Runtime reservation
uses these separately justified CPU terms:

| Term | Bytes | Evidence |
| --- | ---: | --- |
| Pageable indexer scalar | 8 | `isfinite` → `aten::ne`; 1280 allocations in 640 layer calls |
| Synchronous pinned scalar/count | 8 | int64 scalar bin and smaller nonzero count bin, including token validation |
| Pageable metrics transfer | 24 | Three int64 totals via `.tolist()`, with real +24/-24 events |
| Conservative additive bound | 40 | Sum of justified terms, not a measured simultaneous peak |

The four-scheme observer must still check for additional CPU scratch. Explicit
snapshot `.cpu().clone()` operations and reference prefetch are diagnostic/test
paths outside production serving. CPU page ownership tables are persistent
inventory; in-place page release does not allocate a second tensor stack.

## Coverage limits

Direct library CUDA/CPU allocations outside PyTorch allocators, pageable malloc
usable-size overhead, Python objects and pinned allocator internal bookkeeping
are unobserved. Pinned tensor capacity is not total process RSS. CUDA process
peak allocated/reserved bytes are reported separately and include weights and
ordinary activations. Pinned retained blocks are likewise process-wide and can
outlive cache ownership. Owned fixed capacity plus measured temporaries must fit
the admitted cache reservation, itself below the hard budget; this does not prove
that total process physical retention equals cache-owned memory.

## Diagnostic checkpoint evidence

`20261003_echo_memory_c1024_01` stopped at resident 64K prefill because its CPU
reservation was zero while the observed pageable scratch peak was 8 bytes. HBM
dynamic headroom was 1,214,514,048 bytes and active dynamic peak 672,740,352 bytes.
Fixed plus dynamic was 1,529,004,032 bytes against a 2,070,777,728-byte admitted
reservation. All 640 projected-source markers and the returned-hidden marker
were present. Frozen source aggregate was
`6fdcc59fdc6ce5886c067aed53c5adf6b44fddb463bb4ad47b85a68b0a41a269`.

`20261003_echo_memory_c1024_observe_01` initially stopped on seven CUDA history
timestamp matches. Re-parsing unchanged evidence with the documented tolerance
removed every parser error and recovered the same HBM and CPU peaks. Its real
zero-byte CPU reservation violation remains. This is a diagnostic parser
correction, not a new runtime performance result.

`20261003_echo_memory_c1024_observe_02` completed all 32 resident requests, then
stopped at ECHO's first prefill because the first parser revision did not yet
explain native pinned scratch. Python wrappers explained 7,056 bytes; 5,088
nonzero calls at 4 bytes and 640 native unique scalar calls at 8 bytes explain
the remaining 25,472 bytes. Re-parsing the original trace now exactly reconciles
all 32,528 handout bytes, with no evidence errors. Combined CPU peak remains 8
bytes and HBM dynamic peak remains 672,740,352 bytes. The real old reservation
and fixed pinned-ledger deficits remain failures. No new GPU execution was used
for this re-parse. The diagnostic is not publishable. Temporary root:
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-runs/echo-memory-20261003_echo_memory_c1024_observe_02.osCSbQ`.

`20261003_echo_memory_c1024_02` and
`20261003_echo_memory_c2048_fixed_02` both exited 1 on the frozen tree
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-20261003-shared-chunks-02`.
They used corrected full-bin capacity, the 40-byte CPU reservation and the
complete pinned observer, with no diagnostic continuation flag. Overlay SHA was
`ab721f90a3d09bb07e8ec9624552be12a77f4791390274573ec5ceb33b5c5b0b`.
Both completed all 32 requests for hbm, echo and serial_sparse with successful
budget audits and exact same-C hidden/head comparisons. Dense-prefetch's first
prefill and extend passed, but its first zero-allocation truncate exposed the
collector's overly strict requirement for `[memory]` events. The error was
`ValueError: trace lacks cache scopes or allocator memory events`. The unchanged
traces now re-parse under the strict zero-phase guard with cache scopes present,
no profiler memory events, no CUDA history allocations, and zero CPU/CUDA dynamic
peaks. These partial runs remain failed engineering diagnostics outside
`experiments/`; they are not accepted complete results and will not be published.

`20261003_echo_memory_c1024_03` (GPU 3) and
`20261003_echo_memory_c2048_fixed_03` (GPU 0) are complete four-scheme reruns on
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-20261003-gr-tools-03`.
The latter fixes workspace capacity at 2048 query tokens. The 748-file overlay
SHA256 is `2b4ccc9bd8469760dcef03a120d7579fa2d792281d7364025188ddf9313951c8`.
Runtime and installed-backend source identities are unchanged from freeze 02;
only GR validation tools changed. Collector, capture and audit SHA256 values
were checked against main before launch:

- collector: `ea60a13ed462ffc479f67635cae25e48df025caec0d9119cc1e656af4e7ee86c`
- capture: `623acf4bf2465ff4c39d8b19e85476dba9286399a064b003bb09cac1e7dbfe7e`
- audit: `76aa40c2339b6fc71cf37bf3ba76462e8fe9673786df2dbae30e989f78fe0c01`

Both full 03 gates are accepted. Independent verification checked 60 phases at
C1024 (16 hbm, 14 echo, 14 serial_sparse, 16 dense_prefetch) and 64 at C2048
(16 per scheme), including actual-hit-dependent phase coverage. All 32 requests
per case passed exact same-C resident hidden/head comparisons. The source
manifest has 1,388 files and aggregate SHA256
`4d79651da05cd485fa4a9464ae3985f7c345004c42af49dcf198dc2b1418ee74`;
this differs from the overlay digest above. The C512 resident-only supplemental
run `20261003_echo_memory_c512_hbm_03` also passed its 32 requests and 16 phases.
It is memory coverage, not an offload numerical comparison or permission to rank
C512 in a performance sweep.

All three accepted runs were copied into main's corresponding
`experiments/gr_serving/output/{data,log,profile}/<run_id>` directories after
confirming the targets did not exist. Source/destination SHA256 matched for every
file: 1,612 files / 6,336,973,973 bytes for C1024; 1,624 / 5,270,478,786 for C2048;
1,477 / 2,788,233,924 for C512. Original manifest fields and source identities
remain unchanged. No observer exemption or diagnostic continuation was enabled,
and none of these timings enters formal latency results.

`20261003_echo_memory_c256_dense_03` exited 1 at hbm request 31 prefill after 31
requests, before dense-prefetch started. The history was not truncated (967,085
events below 4,000,000). All 774 errors were zero temporal matches, not multiple
candidate matches. Independent counting found exactly 322,369 allocations over
4,517 CUDA addresses in both event streams, with identical counts at every
address. The complete-generation matcher re-parsed this original phase without
errors: CUDA active peak 169,122,816 bytes versus a 305,177,088-byte temporary
reservation, and CPU peak 8 bytes versus 40. This re-parse diagnoses the observer;
it does not accept or publish the incomplete run. Failed artifacts remain outside
`experiments/` under the SSD staging directory.

The complete replacement `20261003_echo_memory_c256_dense_05` is accepted on GPU 6
from `/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-echo-20261003-memory-tools-05`.
Its 755-file overlay SHA256 is
`b056164fb7ac62760914c8113356a0bb1b433e82e60a1ba5d6a17f1deb0a283a`.
The observer SHA256 is
`b195636f72950f7d77827324c9b1cfc29cb968103a130ae3c65172b2ecb3ef86`.
Root verified the runtime gate/sweep source dictionaries and installed backend
identity are unchanged; differences are engineering tools and tests. Both schemes
completed all 32 requests and all four sampled request IDs, followed by independent
verification of all 32 phases. The 1,392-file captured source SHA256 is
`c268d9b14e60ae381662c544d03eec5b0a0e34ca6197bfcb9778eaaeb38a17be`.
The three prior gates keep their original 03 source identities; no manifest was
rewritten or re-signed.

## Accepted memory gates and coverage

The complete engineering record is [memory_acceptance_summary.json](memory_acceptance_summary.json).
It contains each independent audit, actual request/phase matrix, fixed and pinned
inventory, per-phase observed/reserved bytes, source identities and all original
copy hashes. The 6,243 original files total 25,114,372,182 bytes; every copied file
matched its frozen source SHA256.

Peaks below are maxima over sampled phases; each phase was checked against its
own reservation. GiB means 2^30 bytes. CPU scratch is bounded by the separate
40-byte additive reservation; fixed pageable metadata is included in the JSON.

| C / workspace Q | Scheme | Requests / phases | Users / second-pass hits | Cache HBM peak GiB | Max HBM reservation GiB | Owned pinned GiB | CPU temporary peak B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 / 1024 | hbm | 32 / 16 | 3 / 0 | 3.0189 | 3.5235 | 0.00 | 8 |
| 1024 / 1024 | echo | 32 / 14 | 16 / 16 | 2.3149 | 2.8839 | 20.00 | 24 |
| 1024 / 1024 | serial_sparse | 32 / 14 | 16 / 16 | 2.3149 | 2.8839 | 20.00 | 24 |
| 1024 / 1024 | dense_prefetch | 32 / 16 | 12 / 0 | 3.4329 | 3.9374 | 15.00 | 8 |
| 2048 / 2048 | hbm | 32 / 16 | 2 / 0 | 2.8471 | 3.8552 | 0.00 | 8 |
| 2048 / 2048 | echo | 32 / 16 | 15 / 0 | 2.8598 | 3.9323 | 20.00 | 24 |
| 2048 / 2048 | serial_sparse | 32 / 16 | 15 / 0 | 2.8598 | 3.9323 | 20.00 | 24 |
| 2048 / 2048 | dense_prefetch | 32 / 16 | 7 / 0 | 2.8893 | 3.8973 | 8.75 | 8 |
| 512 / 512 | hbm | 32 / 16 | 4 / 0 | 3.5029 | 3.7564 | 0.00 | 8 |
| 256 / 256 | hbm | 32 / 16 | 4 / 0 | 3.3474 | 3.4741 | 0.00 | 8 |
| 256 / 256 | dense_prefetch | 32 / 16 | 15 / 0 | 3.6650 | 3.7921 | 18.75 | 8 |

C1024 ECHO and serial_sparse have 14 phases because sampled revisits hit and
therefore do not prefill. Every other listed case has 16. All seven nonresident
cases compare all candidate hidden states and the final-token head exactly to
independent same-C resident outputs (224 requests). The four resident cases are
references, including the C512 supplemental memory check; they do not establish
cross-C numerical equivalence or performance eligibility.

Direct coverage is exactly the table. The C2048 fixed-workspace run executes the
same C2048/Q2048 geometry and reservation as per-chunk C2048, without claiming a
second run. For unsampled configurations, the source shape bound is
`execution HBM = 1,184,000 * Q + 2,101,248` bytes at capacity 65,664, top-k 2,048,
record width 576 and at most two in-flight writes. Sparse metadata workspace is
fixed by NH/P; per-session and pinned capacity depend on capacity and scheme,
not C. The JSON records all 32 analytical planner rows.

- Per-chunk C256/C512 ECHO and serial_sparse have the same sixteen-user fixed
  inventory as measured C1024 and smaller Q reservations. These are analytical
  budget bounds, not observed allocator peaks at those C values.
- Per-chunk C512 dense-prefetch has fourteen sessions, below measured C256's
  fifteen-session fixed inventory, and Q512 between measured Q256 and Q1024.
  Its own ledger is checked; endpoint peaks are not interpolated.
- Fixed-Q2048 C256/C512/C1024 share the measured C2048 reservation and capacity
  counts (2/15/15/7 users). Q does not exceed the declared workspace bound, but
  these smaller-C fixed-workspace traces were not directly profiled.

These arguments apply to this workload, geometry, precision and runtime identity.
They do not establish unseen allocation paths, arbitrary histories, cross-C
numerical equivalence, process-memory limits or latency.

The new sidecar directory is
`experiments/gr_serving/output/data/20261003_echo_memory_c256_dense_05/acceptance/`.
It preserves the independent checker, copy and summary helpers, planning data,
and all four verification/copy records. Its own `manifest.json` hashes fourteen
files; it was added after original-copy verification and leaves original run
manifests and copied files unchanged. The checker can rerun against retained
artifacts without GPU access. Its README includes reproduction commands.

## Commands and outputs

CPU-only feasibility with a new output directory:

```bash
uv run --no-sync python -m experiments.gr_serving.src.cache_memory_capture \
  --model /preset-models --plan-only --run-id memory_plan \
  --output-dir /tmp/echo_memory_plan
```

After coordination with the formal measurement owner:

```bash
CUDA_VISIBLE_DEVICES=3 bash experiments/gr_serving/scripts/cache_memory_audit.sh \
  echo_memory_RUN_ID --model /preset-models
```

The script stages under TMPDIR, leaves failed diagnostics outside experiments,
and publishes accepted results only under normal `output/data`, `output/log` and
`output/profile` run directories. Each sampled phase includes a trace, CUDA
history and JSON audit; request numerics and workload/source identities accompany
it. No source in an executing freeze may change.

Observer-only continuation uses `--cpu-observation-limit-bytes 67108864`. This
ceiling does not alter runtime admission or waive evidence failures and does not
allow fixed plus temporary CPU capacity above the 64 GiB hard budget.
