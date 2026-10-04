# NOSA shared cache: entrypoint and acceptance audit

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

Date: 2026-10-03. Engineering audit for P0/P4 of
[the implementation plan](nosa_shared_cache_implementation_plan.md). No GPU work
was launched, no performance measurement was made, and no shared/DeepSeek file
was changed by this audit. The working tree remains the authority; the hashes
below identify this inspection, not a frozen final implementation.

## Interface status and admission owner

The ECHO core/native/implementation checkpoints describe ongoing implementation,
not a handed-off frozen serving ABI. The observed shared protocol has pure
`plan_resources`, `allocate_shared`, `shared_bytes`, and host-page hooks.
`PersistentGRRunner` currently has no admission-owner binding. It allocates shared
resources before building the pool, then audits; failure after allocation does
not clean up. Two empty runners can each reserve the full backend budget.

The smallest owner extension is backend `bind_owner(owner)` and
`unbind_owner(owner)` with identity checks, plus explicit failed-construction
cleanup. Binding must happen before plan/allocation and last until successful
pool close, including when the pool is empty. Rebinding the same owner may be
idempotent; a distinct live owner must fail without changing buffers or plans.
Unbinding a foreign owner must fail. Execution leases are a separate mechanism.
`runner.close()` releases sessions and its binding, retaining backend storage;
the outer owner calls `backend.close()`. Failure to drain must not clear ownership
and make poisoned storage reusable.

The constructor cleanup also needs to distinguish a newly allocated plan from an
identical pre-existing valid plan: a failed new runner must release its own new
allocation, but must not close an already valid reusable plan. An atomic
allocation/rollback hook or a binding-scoped allocation checkpoint can implement
this without changing ECHO policy. This point cannot be proved by bind/unbind
alone. Public file integration should be performed by the agreed single owner
after interface handoff.

## Entrypoint changes

| Entry | Observed gap and required change |
|---|---|
| `serving/run_multi_user.py` | Pass both C=`history+candidate` and A=`candidate` to the runner. Close backend in an outer `finally`, including emission/request failures. Its DeepSeek constructor still passes removed `slots`; let the ECHO owner unify CLI options, preserving the explicit rejection of obsolete per-session semantics. |
| `measure._warmup` and formal runner | Currently pass C only. Add A, preserve one same-scheme plan across warmup/session close/formal populations. |
| `measure._select_scheme` | NOSA replaces the backend while leaving old shared ownership alive. Close the previous backend after its sessions are closed before constructing the next scheme around the same model. Add `finally` cleanup around each backend, including warmup and numerical/source-check failure. |
| `measure` metadata | Existing case shared/session reservation fields are useful. Add NOSA policy revision, workspace/host scope, exact C/A/Q, staging count, zero formal trace reservation, fixed actual shared bytes, and current session bytes/actual transfer metrics. Keep weights, ordinary activations and process allocator peaks distinct. |
| `profile._profile_request` | Direct session creation currently occurs without plan/allocation. Use a declared diagnostic plan before session allocation, bounded reserved trace storage, and explicit resource lifetime. Obtain the workspace under a valid session lease or a backend diagnostic accessor; `session.attention_workspace` outside a lease must no longer be a back door. |
| `profile` trace checks | The current gate requires positive lazy trace growth. Replace with equality to the diagnostic reservation and no-growth through execution. Separate shared, session and trace bytes; compare the diagnostic plan to a pure formal plan estimate. Export trace/selection evidence before another lease can overwrite it. Preserve same GPU UUID, source, full-hidden and unique-union checks. |
| `profile` scheme loop | It also overwrites backend variables and needs close before replacement and final cleanup. Every sample constructs an independent sparse prefix. Diagnostic scope still has no multi-user LRU admission/capacity claim. |
| `audit` | The current LRU oracle already subtracts shared reservation and enforces session/page capacity. Add NOSA identity, C/A/Q, stage count, backend-workspace/session-host scopes, trace zero in formal runs, and shared-actual equality across rows. Host pages remain zero for NOSA; continuous host K/V remain session DRAM. Keep strict source coverage unchanged. |
| `report` | Existing generic rows retain extra fields. Add gates for any fields the new report uses and summaries separating revisit hit from revisit rebuild; current scopes only distinguish all/first/revisit. Show shared/session reservation and actual bytes independently. Do not relabel existing old rows as the new policy. |
| `tests/integration/test_gr_persistent.py` | Current one-session budget excludes shared. Plan C/A first and set caps to shared plus exactly one session; outer backend close is required. Keep A/B/A output and miss semantics. |

`source_snapshot` includes both models, all runtime cache/operator/shared code,
GR, and experiment code/tests even for NOSA-only runs. Changing ECHO during a
NOSA run invalidates the source gate. Freeze the integrated tree or use a truly
isolated source checkout; do not narrow the gate. `audit_sources` also checks new
eligible files, preventing omission of newly added workspace/staging modules.

## Test and measurement gates

Existing `models/nosa/tests/test_serving.py` direct calls require explicit shared
planning. Its full checkpoint check currently has only one user with two suffixes
and tolerance 0.016. It must instead exercise two distinct users A/B/A under the
shared mode and compare every normalized candidate hidden element bitwise when
the unchanged numerical path permits it. A short 8K default check does not prove
64K acceptance. Per-layer hooks currently inspect session bytes only; add actual
shared storage and transient allocation coverage.

Add owner-lifetime tests for two empty runners, construction failure before/after
allocation and audit, successful close/rebind, foreign unbind, and retention of a
pre-existing identical plan. CPU resource tests must cover one/two-session exact
budgets, one-byte-under failures, host-only shortage, C/A/Q validation before any
cache mutation, no growth, released/foreign sessions, A/B/A with differing sizes
and tokens, tail blocks, failure recovery and hidden output independence. CPU
accounting proves storage/state contracts, not actual HBM capacity.

The GPU command currently collects all NOSA and operator tests, but only the two
named sparse cache test files under `cache/tests`; new `cache/tests/test_staging.py`
must be included explicitly. Missing GPU/dependencies on explicit GPU invocation
fail during preflight. Checkpoint opt-ins must be set, or those tests skip.

Required commands after implementation and source freeze (select an idle Hopper
GPU immediately before execution):

```bash
bash scripts/run_tests.sh cpu

CUDA_VISIBLE_DEVICES=0 \
NOSA_SERVING_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
NOSA_SERVING_PREFIX_TOKENS=65536 NOSA_SERVING_SUFFIX_TOKENS=1024 \
bash scripts/run_tests.sh gpu

CUDA_VISIBLE_DEVICES=0 \
NOSA_OFFLOAD_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
.venv/bin/python -m pytest models/nosa/tests/test_offload_checkpoint.py \
  -s -q -p no:cacheprovider

CUDA_VISIBLE_DEVICES=0 \
NOSA_SERVING_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
NOSA_SERVING_PREFIX_TOKENS=65536 NOSA_SERVING_SUFFIX_TOKENS=128 \
.venv/bin/python -m pytest \
  models/nosa/tests/test_serving.py::test_cuda_serving_checkpoint_independent_prefixes_and_revisits \
  -s -q -p no:cacheprovider

CUDA_VISIBLE_DEVICES=0 \
TMPDIR=/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-gr-serving \
bash experiments/gr_serving/scripts/run.sh NEW_NOSA_SHARED_RUN_ID \
  --models nosa --sampling sequential --users 16 --requests 32 \
  --history-tokens 65536 --candidate-tokens 128 --chunk-size 1024 \
  --hbm-budget-gib 4 --dram-budget-gib 64

.venv/bin/python -m experiments.gr_serving.src.audit \
  experiments/gr_serving/output/data/NEW_NOSA_SHARED_RUN_ID \
  --expected-run-id NEW_NOSA_SHARED_RUN_ID \
  --json experiments/gr_serving/output/data/NEW_NOSA_SHARED_RUN_ID/audit.json

CUDA_VISIBLE_DEVICES=0 \
TMPDIR=/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-gr-serving \
bash experiments/gr_serving/scripts/profile.sh NEW_NOSA_SHARED_PROFILE_ID \
  --latency-data experiments/gr_serving/output/data/NEW_NOSA_SHARED_RUN_ID \
  --num-users 16
```

Replace each new run ID with a unique value. The standalone offload test's default
65536+1024 request exists at inspection time under
`experiments/indexer_block_sparse_profile/output/data/sparse_native_h200_gpu1_20260929_01/request.json`.
Its existence does not replace the strict shape/model/token checks. Both the NOSA
checkpoint directory and `.venv/bin/python` exist. `nvidia-smi` reported eight
NVIDIA M403 devices with 143771 MiB each; occupancy is transient and no device
was reserved or launched by this audit. Profile must use the formal run's actual
GPU UUID. The temporary filesystem can be too small; the existing script supports
the shared-mount `TMPDIR` shown above.

The controlled 16-user two-pass run is an execution check, not completion of
T-006 capacity validation. A complete fixed heat IID trace must cross actual
shared/session capacities and show HBM miss/offload hit and joint misses, without
forced revisits or seed selection. Report actual observed population and reuse
distance. The overlap 90% claim still requires both per-sample page-envelope and
nonempty-stripe-copy ratios; lack of a 90% claim must not discard valid data.

## Existing result replacement inventory

Immediately when NOSA implementation changes, revise the GR README statement
that NOSA is unaffected to mark shared-cache changes as not rerun. Keep its old
numbers and original implementation/source identity until new acceptance.

| Existing formal run | Existing profile run | Report container |
|---|---|---|
| `gr_serving_h200_20261002_h4k_01` | `gr_serving_h200_20261002_h4k_profile_02` | `report/h4k` |
| `gr_serving_h200_20261002_h16k_01` | `gr_serving_h200_20261002_h16k_profile_01` | `report/h16k` |
| `gr_serving_h200_20261002_h64k_01` | `gr_serving_h200_20261002_h64k_profile_01` | `report/h64k` |

All formal containers mix NOSA and DeepSeek. Replace NOSA four-scheme claims and
profile claims with accepted new runs, but do not rewrite/filter old raw mixed
runs. Keep a mixed container only while it still holds the unreplaced other model;
coordinate final cleanup with ECHO. The failed sequential `_01` attempt is already
excluded and remains outside experiment results. The operator overlap experiment
needs a separate impact decision: optional disabled workspace branches alone can
preserve its old valid evidence, while changes to default prepare/run/sync/stats
require main, confirmation and internal profile reruns. Optional device-only
injection alone need not trigger resident/indexer reruns. DeepSeek non-serving
prefill is unaffected by a dense-serving-only P6 change.

## Inspected source identities

| File | SHA-256 |
|---|---|
| `executor/serving_backend.py` | `dec9f8e93158476d8eac88fa0fff6e5e7dbc13dde7438b414984a11a4fe319c8` |
| `serving/persistent.py` | `e732204c18edda915b7e6c7c06ae7ebe644577bc81c567beb9e1baa4b9f47dab` |
| `serving/run_multi_user.py` | `073a65de18d18ee95a4ad27bc025417ecaaa808fdf8b0c807b9f1e3c2145cbc6` |
| `experiments/gr_serving/src/measure.py` | `b280074564e9fb522779e488871224c352b42671a7325b0980b7c4eb51d57444` |
| `experiments/gr_serving/src/profile.py` | `318c85798986d5e0c532ff068979ba324e651299cf16969501111c442a615f80` |
| `experiments/gr_serving/src/audit.py` | `b3834fd4d0a1692eea7f3115529b44ff3657204a05409a1ef2c680c87b36ee5d` |
| `experiments/gr_serving/src/report.py` | `fb592130cf3540a599c3111d727eec09c2bef2e48e287f23ced8f82d71178078` |
| `models/nosa/serving.py` | `3e901b7251e9a131634932b5cff84b3a2d19a85abb6a38ee0ad5c850ffa4b2f5` |
| `models/nosa/tests/test_serving.py` | `fa525d25c8f714472e2e53100f1d95a292237ce6417b83b013fef70521e47fe0` |
| `scripts/run_tests.sh` | `d4ab5b41ba42799db6c1014e0da73c1dbdffb6731b844d0c7d42782db56bf70e` |

## Follow-up CPU verification

The assigned follow-up added `models/nosa/tests/test_serving_resources.py` and
migrated `tests/integration/test_gr_persistent.py` to exact shared-plus-session
budgets with explicit C/A and backend cleanup. The independent storage walk does
not use production `storage_tensors()` or byte estimators. It checks shared storage
identity, disjoint per-session storage, and actual pending storage after every
model layer. Other cases cover owner/lease separation, initial allocation failure,
changed-plan rejection before allocation, stale/foreign sessions, prefix state,
output lifetime, execution failures and one/two-user capacity boundaries.

CPU resident reference intentionally recomputes compressed records; it does not
materialize the native resident indexer cache. The test therefore verifies the
same committed `IndexerLayerState` before and after suffix/truncate, preserving
the original reference behavior. Offload continues to preserve materialized
per-layer derived records. This is not a numerical-policy change.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  models/nosa/tests/test_serving_resources.py \
  tests/integration/test_gr_persistent.py -x -q -p no:cacheprovider
```

Observed result: **123 passed in 9.81 s**; Ruff check/format passed for those two
test files. This is CPU storage/state verification during integration. The
production implementation is still changing, so final source-stable CPU/GPU and
complete checkpoint/experiment gates remain required. No GPU test was launched
by this follow-up, and no experiment result was replaced.

## NOSA-only profile adaptation

With public serving/executor/measure/audit files still awaiting handoff, the
authorized follow-up changed only `experiments/gr_serving/src/profile.py` and
added `experiments/gr_serving/tests/test_profile_shared.py`.

New collection requires an accepted formal NOSA bounded shared-cache run. It
reconstructs the formal C/A/Q plan, checks its exact plan/session reservations
against the accepted formal case, then adds explicitly reserved trace storage to
a separate diagnostic cap before allocation. Each sparse scheme keeps one plan
across its independent sample sessions. Scheme replacement and final cleanup
close the backend. Trace reservation never grows during execution.

Instrumentation now validates candidate IDs first, enters the backend execution
lease, captures the adapter installed by that lease, and wraps it for layer 31.
It calls the existing model forward and exports selection, counters, raw trace,
page/stripe intervals and memory snapshots before releasing the lease. This
avoids losing the wrapper when the backend temporarily replaces attention and
avoids a later request overwriting shared diagnostics.

New metadata uses schema 2, with explicit formal/diagnostic shared reservations,
session reservations and actual bytes, trace capacity/extent, C/A/Q and ownership.
Publication verification checks these against the saved accepted formal metadata,
rejects workspace growth or over-budget storage, and requires the complete BF16
candidate hidden extent. Legacy schema-1 verification remains available for old
reports; new GPU collection rejects those old per-session formal runs. Source
gate coverage was not narrowed. Unique-union and both interval-ratio checks stay
unchanged; no whole-request session traffic is inferred from the layer-31 trace.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  experiments/gr_serving/tests/test_profile.py \
  experiments/gr_serving/tests/test_profile_shared.py -q -p no:cacheprovider
```

Observed result: **89 passed in 6.94 s**. Tests include legacy artifact validation,
schema-2 artifact tampering, pure diagnostic planning, wrapper/export failure
cleanup, and a real tiny CPU shared NOSA forward with every layer observed inside
the execution lease. Ruff, `git diff --check`, and CPU-only CLI `--help` passed.
This does not validate native trace collection on GPU; formal collection remains
unrun until a new source-stable formal latency result is accepted.

## Current public integration gaps (2026-10-03 follow-up)

The user reconfirmed that the public interface has not been handed off. These
findings are deferred to its owner; this inspection did not change public files.

- Both NOSA measure runners still pass only C. For 65536+128 this gives A=Q=65664
  and 134,771,272 B of sparse shared storage; explicit A=128 gives Q=1024 and
  68,321,352 B. The latter matches the intended bounded diagnostic plan. The new
  profile correctly rejects formal metadata using the conservative C-only plan.
- `audit_metadata` requires zero session DRAM for hbm. Native NOSA hbm correctly
  reserves one byte for its pinned indexer validity flag; a CPU metadata fixture
  reproduces `inconsistent host backing reservation`. Fix the audit's model-aware
  expectation, preserving that real allocation in the budget.
- NOSA-only `measure` writes `backend_provenance=None`. Its source audit fallback
  allows only three submodules, whereas the current tree has four including
  FlashMLA. A temporary source snapshot reproduces this failure. The subsequent
  HEAD gitlink comparison also needs to respect the explicitly recorded current
  dependency identities, including staged updates; adding only a fourth name is
  insufficient. Keep full source coverage and the newly added numerical evidence.

The current formal/profile source snapshots do contain the new transfer-metrics,
allocation-audit, shared-profile and cache-memory modules (194/196 eligible files
at inspection). The engineering component snapshot's omission of one test is a
separate issue. `cache_memory_capture` remains DeepSeek-specific: its `.project().kv`
access and ten-layer samples do not constitute NOSA allocation acceptance.

The follow-up also reproduced and fixed two NOSA profile validation defects.
Deleting schema/resources/export fields can no longer downgrade a shared artifact:
the validator also reads the saved formal NOSA plan and model policy. The source
gate now checks the full current/saved/manifest sets and authenticates all saved
files, so additions and filtered manifests fail. Optional official dependencies
are included; the existing two diagnostic content-drift exceptions are retained.

Final profile verification: **115 CPU tests passed (7.85 s)**; Ruff check/format,
diff check and CLI help passed. CPU `--verify-only` also passed on each real
retained profile (`gr_serving_h200_20261002_h4k_profile_02`,
`gr_serving_h200_20261002_h16k_profile_01`,
`gr_serving_h200_20261002_h64k_profile_01`). No historical artifacts were modified
and no new GPU profile was collected. Current source identities and the full
independent follow-up acceptance are recorded in
[the checkpoint](nosa_shared_cache_checkpoint.md#independent-follow-up-ownership-and-profile-verification).
