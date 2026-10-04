# NOSA full-checkpoint allocation acceptance

Updated 2026-10-03. **Independent NOSA allocation acceptance passed both 64K+1K
and 64K+128 geometries: all 48 phases fit their separate bounds, and all 32
candidate hidden comparisons are bitwise equal.** A strict test-only generation
join replaces the failing timestamp join in these fresh captures. The preceding
captures remain failed with their original evidence errors and limits.
This is independent correctness
work under P5 of [the shared-cache plan](nosa_shared_cache_implementation_plan.md),
while public serving interfaces remain unhanded-off. It does not produce serving
latency, LRU-capacity or paper-experiment results. The allocator guard's blanket
`gc.freeze()` rejection has been repaired, but integrated acceptance remains
pending and the default-pool observation limits below still apply. Current
arithmetic is conditional on that allocator contract.

Storage cleanup on 2026-10-03, explicitly requested by the user: the latest
accepted evidence now resides at
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-nosa-validation/nosa-allocation-generation-xdgzyq7t`.
The original `/tmp/nosa-allocation-generation-xdgzyq7t` is a compatibility symlink.
All recorded source/artifact hashes, 565 regular files and seven symlinks were
verified during the move. The historical paths/commands below remain unchanged.
Raw traces and CUDA histories from the three superseded allocation attempts
were deleted; their reports, manifests, source snapshots and valid CPU/GPU
regression logs remain. Earlier statements that these raw traces were retained
describe their status at the time, before this cleanup. Bulky join-diagnostic
projections were also removed, retaining five small summaries. The cleanup
ledger is
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-nosa-validation/tmp-cleanup-20261003.json`.
No current accepted evidence or DeepSeek artifact was deleted. Future large
validation captures should use an explicit SSD-backed pytest `--basetemp`.

## Scope

`models/nosa/tests/test_cache_allocation_checkpoint.py` adds an explicit
`NOSA_CACHE_AUDIT_CHECKPOINT` opt-in. It loads all 32 layers, builds two independent
resident prefixes for numerical references, releases those reference resources,
then audits all four shared schemes. Each scheme warms a separate session and
discards it before two new empty audited sessions are created. Shared storage
identity must remain unchanged across warmup and capture.

Each scheme captures two complete 64K prefills (all 64 chunks and all 32 layers),
then A/B/A/B candidate execution plus truncate in four further traces. Exact
`(query_start, query_length, layer)` sequences cover write, indexer and attention;
every candidate hidden element is compared bitwise with its independent resident
reference. The two candidate geometries are 1024 and 128 tokens. The first run
did not capture `session_metrics`; v2 adds that normal request boundary at the
end of each phase under `cache_maintenance` so its CPU counter copy is included.

The test independently inventories shared and session storage, reconciles logical
bytes, and uses CUDA allocator block sizes including rounding and unsplit tails.
Cache temporaries are tracked through `free_completed`. The first run checked
both the aggregate reservation and shared-plus-active-session reservation, without
borrowing from inactive sessions. Its original v1 reservation remained the limit;
observed overhead did not increase it. A replaced pre-capture allocation remains
in the parser's fixed baseline, making that case a conservative upper bound rather
than a precise live total. The stricter v2 active-session check below also excludes
unused shared reservation.

NOSA's combined QKV projection remains an ordinary activation. The independently
cloned K/V suffix, indexer finite checks/CIS/derived work, boundary copies and
attention helper allocations are charged to cache. Attention outputs are marked
ordinary. Ordinary allocation peaks, model weight bytes and whole-process CUDA
allocated/reserved peaks are reported separately. Inactive CUDA segments, direct
library allocations and CPU malloc overhead are outside the observed cache
allocation bound; active blocks' rounding and absorbed tails are included. The
generic parser is reused read-only from
`experiments/gr_serving/src/cache_memory_audit.py`; its DeepSeek projection wrapper
and allowance formulas are not used.

Budget, evidence and numerical failures remain failures. The test collects all
phases for a scheme before asserting, so a DRAM failure does not hide the HBM
evidence. All traces and reports stay in pytest/system temporary directories.

## First complete audit: failed, numerically consistent

The complete 64K+1K audit finished with **4 failed cases / 24 failed phases**:
four schemes, each with two full prefills and four extend+truncate captures.
All **16 candidate hidden comparisons were bitwise equal**, covering every
candidate element. All 24 cache and ordinary-activation parser evidence-error
lists were empty. These facts establish neither allocation acceptance nor public
admission integration: every phase failed at least one budget check.

The run used physical GPU 1 (`GPU-a2226185-cb05-a411-80da-f365154128fe`), NVIDIA
M403 / SM90 with 143771 MiB reported memory, the full local NOSA-8B checkpoint,
32 layers, BF16, 2 KV heads, 32 query heads and D128. Each session capacity was
66560 tokens; prefix chunks and candidates were 1024 tokens. This is a correctness
capture with an independent warmup session, not a repeated latency measurement.
Runtime versions were PyTorch `2.12.1+cu130`, FlashInfer `0.6.18`, TVM FFI
`0.1.13.post3`, Triton `3.7.1`, nvcc `13.2.78`, and CUTLASS
`f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`.

The immutable project-source copy is
`/tmp/nosa-allocation-checkpoint-24bwnwhm/source`; virtualenv and unchanged third-party
checkouts are shared. Its 201-file manifest digest is
`047ab1b0596fe25ff0cb2669e74bccb18d88e9fca8a7d38881cff6f1eafaf1b9`.
The copy was created at `2026-10-02T17:46:37.825217+00:00`. The manifest and runtime
identity remain at `/tmp/nosa-allocation-checkpoint-24bwnwhm/source_manifest.json`
and `/tmp/nosa-allocation-checkpoint-24bwnwhm/runtime_identity.json`. The frozen
plan policy was `nosa_backend_workspace_v1`; fused native key was
`cff17752fdd73139`. Relevant SHA256 identities in that copy are:

| Source | SHA256 |
|---|---|
| `models/nosa/serving.py` | `11a35de376ca9968a7e3d6ed65293acef6fac8baa1301c694afff53d1196ecf8` |
| `models/nosa/serving_resources.py` | `e56f5cb163c4603e4cf08d113f14611a2fe535ab7059eccc5e7d76b62e93dea4` |
| `models/nosa/tests/test_cache_allocation_checkpoint.py` | `51982c8c460c2c186d3153a9a8ff1c5ff033050210424229645dda16b7311e17` |
| `experiments/gr_serving/src/cache_memory_audit.py` | `a9a8ca6f80fec129b01b7801bd2ab3435ffe79559ba0310e2c5750d0997a6d84` |

The following command ran from the frozen source directory:

```bash
CUDA_VISIBLE_DEVICES=1 \
PATH="/mnt/ssd-wlcb/chenkaiqi/cxldsagr/.venv/bin:$PATH" \
NOSA_CACHE_AUDIT_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
NOSA_CACHE_AUDIT_SUFFIX_TOKENS=1024 \
/mnt/ssd-wlcb/chenkaiqi/cxldsagr/.venv/bin/python -m pytest \
  models/nosa/tests/test_cache_allocation_checkpoint.py::test_checkpoint_cache_allocation_lifetimes \
  -q -s --tb=short -p no:cacheprovider \
  --basetemp /tmp/nosa-allocation-checkpoint-24bwnwhm/pytest-1024
```

Reports and traces remain only under
`/tmp/nosa-allocation-checkpoint-24bwnwhm/pytest-1024/`. Numbered case directories
`test_checkpoint_cache_allocati0` through `test_checkpoint_cache_allocati3` map to
`hbm`, `serial_sparse`, `dense_prefetch`, and `overlap`; the pytest `current`
symlink is not another run. Each contains `user{0,1}_prefill` and
`visit{0,1}_user{0,1}_extend_truncate` `.json`, `.cuda-memory.json`, and
`.report.json` files. These failed correctness diagnostics are not stored in
`experiments/` and are not experiment baselines.

### Allocation topology established by the first run

All byte counts below come from that frozen v1 execution, not the modified code.

| Capture / allocation group | Observed allocator bytes | v1 temporary limit | Diagnosis |
|---|---:|---:|---|
| hbm, user 0 prefill peak | 78,906,880 | 77,333,760 | Derived blocks plus old/new indexer slabs exceed the reservation |
| hbm, user 1 prefill peak | 81,629,184 | 77,333,760 | Identical logical records received larger allocator tails |
| serial sparse, user 0 prefill peak | 128,771,584 | 110,885,600 | Pending clones and Q finite-check allocations remain live together |
| overlap, final user 1 candidate peak | 55,447,552 | 33,327,840 | Pending clones, Q finite-check allocations, CIS and one absorbed tail |

- **Independent derived allocations matter.** Each of 32 layers owns compressed
  K, compressed CIS and pooled CIS, with payloads 2,129,408 / 16,636 / 4,156 B.
  Their combined logical payload is 68,806,400 B. Resident user 0 allocated
  70,571,520 B for them, including two large K blocks of 3,048,960 and 2,952,192 B.
  Resident user 1 had still larger tails. Rounding a combined payload once cannot
  bound these 96 independent storage allocations.
- **Resident growth has two slabs.** User 0's prefix peak consists of derived
  blocks 70,571,520 B, old W 4,132,864 B, new W 4,198,400 B, and CIS source 4,096 B.
  IDs and normalizers are absent at this peak. The first candidate's new
  allocations instead peak at 5,709,824 B: W 4,263,936 B, IDs 1,048,576 B,
  validity 131,072 B, normalizers 262,144 B, and CIS source 4,096 B. The second
  candidate reuses W and peaks at 1,445,888 B. Retaining an already freed old W
  in the audit baseline is conservative, but a separate `free_completed`
  reconstruction still found real resident candidate overruns of 1,704,192 B
  and 4,426,496 B for the two users. The failure is not only a baseline artifact.
- **Offload owns every pending clone until commit.** For Q1024, 64 independent
  K/V clones contribute 33,554,432 B across 32 layers. The combined QKV projection
  is an ordinary activation and is not counted again as cache storage.
- **`isfinite` is more than its final mask.** In this pinned PyTorch version,
  `(x == x) * (x.abs() != infinity)` retains BF16 abs plus three Boolean outputs.
  Q has 4,194,304 elements, so the payload peak is 20,971,520 B. Serial sparse
  user 0 prefill combines that group with derived blocks 70,047,232 B, pending
  clones 33,554,432 B, W 4,194,304 B and CIS source 4,096 B. Final overlap has the
  same allocation categories; one 4,194,304 B Boolean tensor receives a
  5,111,808 B allocator block, adding 917,504 B. No new category was found there.
- **Small CPU allocations are real budget terms.** Resident revisits retain a
  1 B pinned host flag while a 1 B uint8 dtype-size probe runs. The flag appears
  in independent storage inventory, but not in positive profiler memory events;
  the CPU profiler alone does not establish its allocation lifetime. Offload
  `aten::isfinite` creates an 8 B wrapped Double infinity scalar. A separate
  BF16 `IndexerCache.workspace` probe had earlier observed 2 B, but those helper
  probes are not full-model peak measurements. The v1 resident CPU reservation
  left no transient space on revisits, and offload left none in any phase.

Read-only follow-up found an additional allocation outside those first-run
captures: `models/nosa/cache.py::_NosaTransferMetrics.snapshot` calls `.tolist()`
on two int64 GPU counters through `models/nosa/serving.py::session_metrics`.
Pinned PyTorch's
[`tensor_list.cpp`](https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/torch/csrc/utils/tensor_list.cpp#L59)
copies that tensor to CPU first, allocating a 16 B tensor while the session's host
K/V remains live. This affects `serial_sparse` and `overlap`;
`dense_prefetch` has no sparse counter tensor. The earlier 8 B sparse temporary
formula covered the recorded execution phases but missed this request boundary.
The current estimator reserves 16 B and the subsequent GPU audits include metrics
observation; no first-run trace establishes that added boundary's measured peak.

## v2 implementation and reservation contract

[The pure session estimator](../../../models/nosa/session_budget.py) charges each
independent allocation through
[the allocator helper](../../../models/nosa/allocation_budget.py). The current
resource policy is `nosa_backend_workspace_v2`; it is not the identity of the
failed frozen run. For one device payload of n bytes, the checked native policy
uses `r = ceil(n / 512) * 512`, then charges r for `r <= 1 MiB` and `r + 1 MiB`
otherwise; zero remains zero. Each independently allocated pinned host K/V tensor
instead reserves its next-power-of-two bin; CPU role budgets charge payload bytes. The bound
requires the pinned native allocator, default 512 B rounding, unrestricted block
splitting, non-expandable segments, and absence of private/no-split pools. The
runtime guard rejects incompatible effective settings, GC-visible Python
MemPools, and recorded private segments. Normal GC freezing is allowed. Python
GC cannot expose frozen empty MemPools, and PyTorch lacks a getter for all direct
C++/private-binding pool routing; these external paths and allocator/environment
mutation are unsupported. The guard does not prove absence of every external
pool. The current NOSA GPU regression covers the independent backend lifecycle,
and the fresh captures below establish its controlled allocation acceptance.
Public integration remains pending. This is not a guarantee for arbitrary
allocator configurations or direct library allocations.

Let A(x) be the per-allocation bound, C session capacity, Q the maximum actual
batch, L layers, H KV heads, G query-heads/H, D head dimension, s dtype bytes,
`m=max(C//16-1,0)`, `p=max((C-16)//64,0)`, and `P=ceil(C/64)`:

```text
derived = L * (A(m*H*D*s) + A(m*H*s) + A(p*H*s))
resident fixed = A(L*C*H*s) + derived + 2*A(L*C*H*D*s)
offload fixed = A(L*C*H*s) + derived
                + A(16) for serial_sparse/overlap counters only
offload pending = 2*L*A(Q*H*D*s); resident pending = 0

resident W payload = align256(align256(Q*H*P*s) + 3328 + H*64*4)
offload W payload = align256(Q*H*P*s) when P>64, otherwise 0
phase = max(CIS creation peak, A(Q*H*s) + max(selection, finite, boundary))
session HBM = fixed + pending + 2*A(W payload) + phase
```

`2*A(W) + phase` is intentionally conservative: two maximum indexer slabs remain
reserved alongside the largest helper phase, even where actual lifetimes are
sequential. It also covers the current audit's fixed pre-capture old-slab
treatment. CIS creation bounds the model-dtype projection, FP32 conversion,
softplus/scaling and final dtype conversion's individual liveness groups. Its
returned model-dtype CIS source remains charged during subsequent selection and
attention.

Selection charges IDs, validity and normalizers independently, plus a separate
`A(H*64*4)` ranking tensor for offload. It evaluates both the Q>=128 one-split
case and q=min(Q,127), whose normalizers can use up to four splits. A maximum-Q
one-split calculation alone does not cover smaller admitted batches. Resident
finite preparation already occupies W. Offload finite checks bound BF16 abs and
three masks, scalar reductions, retained Boolean results and their allocator
tails; Q, K and CIS validation phases are sequential. Boundary compression
reserves both the copied history of at most 31 tokens and its concatenation
with Q pending tokens. Existing pending clones are not charged twice.

Resident session DRAM reserves the fixed 1 B pinned flag plus a 1 B temporary
probe. Offload reserves two separate contiguous host K/V tensors plus the maximum
of sequential infinity-scalar, dtype-probe, host-observation and request-metrics
temporaries: 8 B for `dense_prefetch`, 16 B for `serial_sparse` / `overlap`. CPU
reference results keep their original storage-role budgets, without CUDA host
temporaries; they are not guarantees of the PyTorch reference implementation's
real CPU allocation peaks.

Shared workspaces expose ordered `allocation_sizes(...)` tuples corresponding
to their independently owned tensors, including empty entries and excluding
aliases. Shared reservation sums A over that tuple; it does not round an aggregate
payload. V2 tests separately check actual shared blocks against shared reservation,
active-session fixed blocks plus its temporary peak against that session's own
reservation, and the total shared-plus-all-sessions bound. **The active session
cannot borrow padding or unused reservation from shared storage or another
session.** This is stricter than the first run's shared-plus-active check.

### Intermediate budget table and retrospective diagnostic

This intermediate estimate preceded pinned-bin accounting and is not the current
DRAM reservation. For C66560 / Q1024 / L32 / H2 / G16 / D128 / BF16, the figures
below are bytes and exclude shared reservation:

| Scheme | v1 session HBM | v2 session HBM | v2 session DRAM | Minimum HBM margin against old traces, no shared credit |
|---|---:|---:|---:|---:|
| hbm | 2,266,891,520 | 2,307,158,016 | 2 | 34,398,208 |
| serial_sparse | 119,407,888 | 181,293,568 | 2,181,038,096 | 35,573,760 |
| dense_prefetch | 119,407,872 | 181,293,056 | 2,181,038,088 | 35,573,760 |
| overlap | 119,407,888 | 181,293,568 | 2,181,038,096 | 35,573,760 |

The resident total is fixed 2,295,087,104 + two W 10,625,024 + phase 1,445,888 B.
Serial sparse/overlap total is fixed 111,952,384 + pending 33,554,432 + two W
10,616,832 + phase 25,169,920 B; dense has one fewer 512 B counter allocation.
The offload phase includes a 25,165,824 B allocator bound for the finite-check
group plus 4,096 B CIS source. These terms are allocation-derived bounds, not
added percentages fitted to the observed deficit.

Applying these formulas to the 24 old reports fits every recorded active bound.
The margin calculation removes the independently recorded shared payload and
shared allocator padding from each active-plus-shared baseline before adding
the observed temporary peak. Minimum CPU slack is 0 B for resident candidates
and dense offload phases, and 8 B for the recorded sparse phases because their
16 B metrics copy was not captured. **This is read-only retrospective arithmetic
for diagnosis, not a rerun, new allocation acceptance, or performance evidence.**
Frozen reports remain failed and retain their original limits and source identity.

## Validation and outstanding acceptance

- Original wrapper/storage checks: **5 CPU tests passed, 4 opt-in cases skipped
  (4.14 s)**; Ruff check and format passed. These checks did not establish a
  complete-model allocation bound.
- Session-estimator unit checks after the 16 B sparse metrics correction, using
  the real allocator helper: **24 passed (1.27 s)**, with CUDA hidden; Ruff check
  and format passed. Coverage includes
  independently charged records, q=127/128 normalizer dispatch, all pending
  clones, finite/boundary/CPU transients, two slabs, invalid native geometry and
  smaller admitted batch sizes. Sparse metrics reserve and unchanged CPU role
  semantics are asserted separately. This is preliminary formula coverage only.
- Source identity for that 24-test result: `models/nosa/session_budget.py`
  SHA256 `fb203c8bb750d7d2ae78fc4bb74e6ec068556dd6b6a88dfaf78ea07ff848d255`;
  `models/nosa/tests/test_session_budget.py` SHA256
  `91b38d6d5381f5fccb4523708e466655cd0c720749a8a100a03d228121984e1f`.
- The intermediate CUDA-padding correction was frozen in
  `/tmp/nosa-allocation-v2-kkz8r6jj/source`, source SHA256
  `3db509419ca6ececf4b18fe49bc69f7ca152a60672d4fa5e729d25e5a098e9ef`.
  Both 64K+1K and 64K+128 ran: each had **1 passed, 3 failed**, because
  the new generic inventory correctly counted pinned allocator bins. Each
  resident case passed all six phases and all four candidate comparisons; the
  offload cases failed at the fixed baseline before capture. Their numerical
  outputs and 16 B metrics boundary were not accepted by these attempts.
  No frozen report limits or outcomes were rewritten.

The preliminary session checks used:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest \
  models/nosa/tests/test_session_budget.py -q --tb=short -p no:cacheprovider
```

No affected performance report is accepted, replaced or deleted by this
correctness work. Previous post-forward storage checks cannot substitute for
transient allocation acceptance, and direct-backend checks cannot substitute for
public PrefixPool/LRU admission integration.


## Pinned DRAM correction and timestamp-join failure

The pinned CUDA host allocator rounds each independent request to the next power
of two and reuses only the same bin. At C66560 each K/V tensor requests
1,090,519,040 B; at C65664 it requests 1,075,838,976 B. Both require a 2 GiB bin,
so two K/V tensors own **4,294,967,296 B per session** in either geometry.
The intermediate payload-only DRAM estimate was therefore insufficient even
before any offload computation. This is actual allocator capacity, not a
percentage added to match a measurement.

`pinned_allocation_bytes(size, device)` now reserves each bin independently.
Sparse session DRAM is 4,294,967,312 B including the 16 B metrics temporary;
dense session DRAM is 4,294,967,304 B including its 8 B temporary. HBM estimates
are unchanged. CPU reference role accounting still uses payload bytes.
`backend.session_bytes()` reports owned pinned capacity; the separate
`session_storage_bytes()` preserves the logical storage ledger. The HBM field
continues to describe tensor storage; its allocator padding and transient bounds
are checked by the intrusive audit rather than inferred from post-forward stats.

The audit additionally surrounds each empty session constructor with isolated
`host_memory_stats()` observations. It requires exactly two host handouts for
offload and compares their cumulative `active_bytes.allocated` increase with
the independently inventoried bins. Reused bins also increase that cumulative
counter. No other pinned allocation or statistics reset occurs in this window.
`active_requests.allocated` checks the handout count. The CUDA runtime must be
initialized; empty statistics are not accepted. Host `current` and `peak` fields
are not used as a per-tensor capacity oracle: the pinned upstream implementation
has release-accounting limitations, and bucket peaks are not a common instant.

The allocator policy is now
`nosa_torch_native_512b_1mib_tail_pinned_power2_v2`. Explicit
`pinned_reserve_segment_size_mb` is rejected because the preallocated segment and
its 4 KiB cursor rounding exceed the per-bin ownership model. Previously latched
configuration later hidden by external mutation, frozen empty Python pools and
low-level pool routing remain unsupported and cannot all be detected by the
observable-state guard. Process inactive cached segments, fragmentation and
CPU malloc overhead remain separate from the active owned-cache bound.

Final candidate sources are frozen at
`/tmp/nosa-allocation-pinned-3v_cm548/source`: **347 files**, manifest SHA256
`0c826be422887988c2cec5c78770270fccae50b25b8e73988735a17f331e9333`.
The runtime identity, manifests, test stdout/stderr, and pytest traces are in
that system temporary directory, not in experiments. The virtualenv and four
unchanged third-party checkouts are symlinked and separately identified.

- Allocator helper: 53 CPU tests; session formula: 27 CPU tests.
- Complete final-source CPU scope: **681 passed, 671 GPU/opt-in skips,
  1 profiler warning (34.60 s)**. This includes NOSA, bounded operators,
  staging/integration and diagnostic-profile checks.
- Targeted GPU session accounting and lifecycle: **22 passed (20.00 s)** after
  pinned-capacity reporting changed.
- Final-source full NOSA GPU regression: **1230 passed, 5 skipped, 2 warnings
  (394.24 s)**, recorded in `/tmp/nosa-allocation-pinned-3v_cm548/nosa-gpu.stdout`.
  The skips are the four separately executed allocation-audit opt-ins and the
  `NOSA_MODEL_PATH` metadata opt-in; they are not missing-hardware skips. This
  regression does not establish complete allocation acceptance.
- Complete 64K+1024 allocation audit: **4 failed (689.61 s)**; complete 64K+128
  allocation audit: **4 failed (659.59 s)**. Logs are
  `/tmp/nosa-allocation-pinned-3v_cm548/run-1024.stdout` and
  `/tmp/nosa-allocation-pinned-3v_cm548/run-128.stdout`. In each geometry, all
  **16 candidate phases pass their budget checks and all-candidate bitwise
  comparisons**, but all **8 full-prefill phases have unknown CUDA peaks**
  because the profiler/history join cannot uniquely match allocation generations.
  These evidence failures remain failures; an unknown peak is not zero or proof
  that the reservation fits.

Across both geometries, all **32 candidate hidden comparisons are bitwise
equal**, all **48 phase CPU bounds pass**, and all **12 audited offload session
constructors** independently report **4 GiB from exactly two pinned handouts**.
These are numerical, CPU-bound and constructor-capacity evidence. They do not
establish the missing full-prefill CUDA allocation bound, serving latency or
multi-user admission behavior. The failed reports and their original limits
remain unchanged.

A **test-only strict full-callback allocation-generation join is implemented**
for these long captures. It validates all CUDA allocation/free-request callbacks,
including ordinary activations, in exact single-thread encounter order before
applying ownership. Counts, addresses, requested/block sizes, baseline frees and
allocation/request/completed-free state must agree. Pending frees remain active
through the capture end; no timestamp tolerance or clock adjustment is used for
identity. Modified histories and malformed or duplicate baseline owners fail.
The shared parser in
`experiments/gr_serving/src/cache_memory_audit.py` remains read-only, and the
public serving interface is still not handed off; this work does not relax
either boundary.

The adapter's 46 adversarial tests passed. In the new frozen source its tests
plus the existing checkpoint-wrapper tests passed **51 tests, 4 opt-in skips,
1 profiler warning (4.03 s)**. Fresh complete captures passed under
`/tmp/nosa-allocation-generation-xdgzyq7t`: **349 files**, manifest SHA256
`9eeed836edf1e0bc58f1e4b906634e68b55b35a377f66aa37316de5d5cec95e5`.
Only the checkpoint test and the new adapter/test files differ from the preceding
347-file freeze. Production sources and the read-only parser
(`3eaf0d43f82a9ba81024261365b2cda51bbf1b1bd6dba9f40f9b2da48f1d3118`)
are unchanged. The absolute workspace virtualenv executable preserves fused
build key `cff17752fdd73139`; source relocation still triggers JIT rebuilds.
All four schemes passed all six phases in each geometry.
Five representative old traces also pass retrospective diagnostic parsing, but
their original reports remain failed; only fresh captures can establish the new
acceptance result. An initial command lacking the opt-in produced four skips
(1.27 s), retained separately as `run-1024-missing-optin.*`; it is not acceptance.

The cross-phase lifetime argument is specific to this test: before every
`_capture`, it constructs the next user's CUDA history or suffix tensor, after
the previous capture has synchronized the device. In the pinned PyTorch source,
[`DeviceCachingAllocator::malloc`](https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/c10/cuda/CUDACachingAllocator.cpp#L1559)
calls `process_events` at line 1577 before acquiring a new allocation block.
Thus completed prior suffix-clone frees are processed during that input
allocation, outside the next recorder, before the new block becomes active.
The join retains pending clones through the preceding capture's end; the next
fixed baseline can then omit those released clones without losing an overlap
with a new allocation. Device synchronization or a snapshot alone does not drain
the allocator queue. Standalone `_capture` calls and arbitrary callers do not
inherit this guarantee: without the preceding input allocation, an orphan
`free_completed` in the next recording must fail closed rather than be ignored.

The regression command uses the existing owned-test request explicitly through
`NOSA_OFFLOAD_REQUEST`; the preceding source copy omitted this ignored fixture,
producing one FileNotFoundError. The isolated correction with that same request
passed (16.96 s). An earlier CPU copy also omitted two imported experiment
modules; the complete dependency copy passed. Neither missing-file attempt is
represented as successful acceptance.

Under the planned 64 GiB cache DRAM cap, this geometry's reservation permits at
most 15 offload sessions, since 16 full 4 GiB histories leave no room for the
required temporary. This is a formula implication for the future admission
run, not measured LRU behavior. Keep the specified 16-user sequential workload;
do not alter its population or budget to manufacture revisit hits. The public
owner/lifecycle integration and new formal performance runs remain outstanding.

## Accepted independent allocation result

The fresh 349-file freeze completed both opt-in checkpoint tests successfully:
**64K+1024: 4 passed, 2 warnings (585.15 s); 64K+128: 4 passed, 2 warnings
(573.68 s)**. These are test durations, including compilation and intrusive
profiling, not serving latency. The warnings are an upstream FlashInfer
deprecation and the profiler's single-cycle event-lifetime notice; each capture
uses one cycle and verifies the entire callback stream against allocator history.

All **48 phases** have zero cache and ordinary-allocation evidence errors and
fit both the aggregate and active-session bounds. All **32 candidate hidden
tensors** equal their independent resident references bitwise. The minimum
active-session HBM margin across all six phases of each case is:

| Candidate tokens | hbm | serial_sparse | dense_prefetch | overlap |
|---|---:|---:|---:|---:|
| 1024 | 34,398,208 B | 35,573,760 B | 35,573,760 B | 35,573,760 B |
| 128 | 35,168,256 B | 38,965,248 B | 38,965,248 B | 38,965,248 B |

No case borrows shared padding or inactive-session reservation. Minimum CPU
margin is 0 B for every case. Every phase observes the expected temporary CPU
peak: hbm 1 B, dense 8 B, serial sparse/overlap 16 B. All 12 offload constructors
independently confirm 4,294,967,296 B and two pinned allocator handouts per
session. The four resident constructors own no host buffer; their lazy pinned
flag is checked separately during execution.

Each offload prefill matches 81,052 allocations and 80,955 free requests;
80,891 completed frees are recorded. Its final 64 pending generations remain
charged through capture end. Every offload candidate also retains 64 pending
generations. The 1024-token first visits additionally prove the old indexer
slab's baseline free; the 128-token visits need no slab replacement. None of
these generations is assigned by timestamp proximity.

[The source and evidence inventory](nosa_cache_allocation_source.json) records
31 component identities, all 349 frozen dependency identities, native/toolchain
metadata, the exact commands, per-case bounds and hashes of all 48 traces,
allocator histories and reports. [The component diff](nosa_cache_allocation.patch)
is against the recorded Git revision and includes untracked implementation/test
files. All frozen files and the 31 current workspace component files matched
after acceptance; runtime headers, allocator libraries and four third-party
checkouts were also rechecked. Source relocation caused rebuilds but did not
change fused build key `cff17752fdd73139`.

Production code is identical to the preceding frozen regression, which passed
681 CPU tests and 1230 GPU tests. Only three audit test files changed for this
fresh capture. The public parser remains its recorded frozen read-only
dependency; later concurrent edits to public files are outside this acceptance.
No formal performance artifact was published, replaced or deleted. Public
runner ownership, admission/LRU, formal loop measurement and DeepSeek dense
integration still require the pending interface handoff.
