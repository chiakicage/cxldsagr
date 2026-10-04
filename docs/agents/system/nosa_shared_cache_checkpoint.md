# NOSA shared execution cache checkpoint

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

Updated 2026-10-03. **Both implementations, public integration and correctness/
allocation acceptance are complete.** The user's latest
instruction is correctness only, with no further performance tests. The formal
measurements, profiles, default selection and publication/replacement stages in
the original plan are canceled, not outstanding requirements.
Current evidence and limits are in
[the integration acceptance record](nosa_shared_cache_integration_acceptance.md).
The earlier stage-by-stage entries below retain their original source boundaries.

## Ownership and work coordination

The user confirmed **DeepSeek cache implementation complete**, lifting the
earlier handoff barrier. Root owned sequential changes to
`executor/serving_backend.py` and `serving/persistent.py`; separate agents handle
NOSA owner hooks, public measurement/audit callers, and DeepSeek dense migration.
The integrated runner binds one admission owner before allocation, rolls back
failed construction, and preserves preexisting shared plans. A failed release
keeps the owner bound and disables requests. CPU/GPU lifecycle, numerical and
allocation checks validate the implementation. No new serving latency result or
NOSA complete capacity-loop measurement is claimed.

NOSA's accepted independent evidence is preserved while integration proceeds.
The existing worktree has extensive independent DeepSeek changes; preserve them.
No reset/stash or blanket output cleanup has been performed. Source identity for
formal GR runs includes both models and shared files; do not narrow that gate to
work around parallel development.

## Implemented and under validation

- `models/nosa/serving_resources.py` provides pure C/A/Q planning, shared storage
  allocation and independent storage deduplication, exclusive execution leases,
  generation/session identity checks, backend owner binding hooks, poison handling
  and close. An identical plan is reusable. A different allocated plan requires
  explicit close first, avoiding old/new simultaneous allocation and preserving
  the old valid plan on failure.
- Shared `serial_sparse`/`overlap` use one bounded full-address fetch workspace.
  Shared `dense_prefetch` uses common `cache/staging.py` double buffering.
  Shared `hbm` and dense inject a bounded device-only FA3 scratch provider.
  Backend session creation requires a prior explicit plan/allocation; ordinary
  model-owned caches keep their existing standalone default behavior.
- Borrowed cache begin/write/commit/abort require the owning execution lease;
  foreign/stale sessions and conflicting lifecycle calls are rejected. A prefill
  lease spans all chunks; extend includes the full transaction and output.
  Lease return conservatively synchronizes pending work, including speculative
  dense copies and commit-time indexer compression.
- Sessions own K/V histories, CIS/derived records, pending append, indexer scratch
  and their independent transaction state. Shared staging is not counted again
  or released with a session. Shared native resident accounting includes its
  pinned one-byte indexer validity flag; standalone stats retain their prior scope.
- Operator default paths preserve their native math, selection, numerical repair,
  launch arguments and lazy allocation behavior. Shared paths fail for unsupported
  native configuration or query/trace bounds. CPU workspace allocations test the
  storage role ledger, not actual GPU memory.
- Last-request transfer accounting separates prefix and candidate main-KV H2D,
  main-KV D2H and indexer boundary H2D. Dense/D2H payload is counted after each
  successful copy submission; sparse fetch accumulates native counters into two
  session-owned int64 values (16 bytes, included in its reservation and stats).
  Failed operations are marked failed, not reported as successful requests. These
  are tensor payload bytes, not measured physical PCIe link traffic.
- Shared and session allocation-ready events fence initial use on a different
  caller stream. Failed cleanup attempts both staging and device drains, keeps
  ownership/aliases, poisons resources and rejects subsequent mutation or reuse.
- The NOSA diagnostic profile now declares a bounded trace plan before allocating,
  wraps attention within the backend execution lease and exports trace before
  returning it. New schema 2 checks shared/session/trace bounds and fixed storage;
  old schema 1 remains verify-only. Its 89 CPU tests passed; no new formal profile
  is claimed without the matching new latency run.

Component details and source identities:
[operator workspace](nosa_shared_workspace_checkpoint.md),
[dense staging/GPU tests](nosa_dense_staging_checkpoint.md),
[entrypoint audit and CPU tests](nosa_shared_entrypoint_audit.md).

## Verification observed in this turn

These are correctness checks, not paper experiments. Subsequent affected source
changes require their corresponding acceptance to be repeated on final sources.

- CPU targeted model/cache tests: 120 passed, 4 GPU/checkpoint skips.
- Resource ownership/budget plus integration tests: 123 passed. They cover pure
  planning, independent actual storage sums, exact one/two-user admission and
  one-byte-under rejection, A/B/A with different capacities and tails, stale/
  foreign sessions, exclusive leases, allocation failure and explicit replan.
- Double staging: 17 passed on physical GPU 3 (SM90), including delayed copies,
  nondefault streams, delayed consumers, exceptional unconsumed prefetch and
  poisoning. Small four-layer native shared serving: 12 passed on GPU 3; all
  candidate hidden outputs equal independently prefixed HBM controls bitwise.
- Operator bounded/default suite: 47 passed on GPU 2; actual q=1024 under Q=1032,
  varying head/query views, unique reads, serial/overlap equality, stripe/page
  envelope checks and cross-stream metadata reuse are covered.
- Complete 32-layer shared checkpoint, physical GPU 1, four schemes, two distinct
  independently built user prefixes per scheme and two interleaved visits:
  **65536+1024 passed (45.49 s)** and **65536+128 passed (36.06 s)**. Every candidate
  normalized hidden element is bitwise equal to HBM, max_abs=0. Timing here is
  pytest duration, not a serving latency result.
- Standalone owned resident/offload 32-layer 65536+1024 test: **1 passed (16.88 s)**,
  bitwise equality and max_abs=0, confirming the independent default model path.
- An earlier `bash scripts/run_tests.sh cpu` attempt had 2008 passed, 745 skipped,
  **4 failures and 14 setup errors**, all in the concurrently introduced
  `operators/deepseek_v32/indexer/tests/test_echo_official_policy.py`. Its explicit
  GPU fixture fails under the CPU entrypoint. The DeepSeek-owned test was not
  modified. A later run by the concurrent DeepSeek workline completed with
  **2089 passed, 824 skipped, 1 warning, 58 subtests passed (60.55 s)**, observed
  in `/tmp/deepseek-nonmatrix-cpu.stdout`. That resolves the observed collection
  problem, but is not a run against this workline's final frozen source. Repeat
  global checks after public integration.

Checkpoint commands used the existing environment and weights:

```bash
CUDA_VISIBLE_DEVICES=1 PATH="$PWD/.venv/bin:$PATH" \
NOSA_SERVING_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
NOSA_SERVING_PREFIX_TOKENS=65536 NOSA_SERVING_SUFFIX_TOKENS=1024 \
.venv/bin/python -m pytest models/nosa/tests/test_serving.py::test_cuda_serving_checkpoint_independent_prefixes_and_revisits \
  -q -s --tb=short -p no:cacheprovider
# Repeated with NOSA_SERVING_SUFFIX_TOKENS=128.

CUDA_VISIBLE_DEVICES=1 PATH="$PWD/.venv/bin:$PATH" \
NOSA_OFFLOAD_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
.venv/bin/python -m pytest models/nosa/tests/test_offload_checkpoint.py \
  -q -s --tb=short -p no:cacheprovider
```

An early targeted command omitted `.venv/bin` from PATH and unintentionally
collected GPU tests; native compilation failed to locate `ninja`. Corrected
explicit CPU/GPU invocations are recorded above. This was not a numerical failure.

## Previous independent NOSA batch verification

After traffic accounting, allocation-stream fences and poison cleanup were in
place, the implementation/test sources were captured in
[nosa_shared_cache_source.json](nosa_shared_cache_source.json) and the recorded
component [diff](nosa_shared_cache.patch). The manifest covers 23 changed source/
test files, their SHA-256 identities, the observed unhanded-off public interfaces,
dependencies, GPU identities and native build metadata. Fused build key:
`cff17752fdd73139`. The files covered by the manifest remained unchanged during
that verification. This was not a complete dependency freeze: it omitted the
already-tested `models/nosa/tests/test_transfer_metrics.py`, the separate offload
prepare source, and unchanged runtime dependencies. It is an engineering component
snapshot, not a formal experiment run ID. Later ownership/profile corrections
must be identified separately; the results below describe the earlier revision.

```bash
CUDA_VISIBLE_DEVICES='' PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pytest \
  models/nosa/tests operators/nosa cache/tests/test_staging.py \
  tests/integration/test_gr_persistent.py \
  experiments/gr_serving/tests/test_profile.py experiments/gr_serving/tests/test_profile_shared.py \
  -q --tb=short -p no:cacheprovider
# 532 passed, 667 CUDA/checkpoint skips in 46.23 s.

CUDA_VISIBLE_DEVICES=1 PATH="$PWD/.venv/bin:$PATH" \
NOSA_SERVING_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
NOSA_SERVING_PREFIX_TOKENS=65536 NOSA_SERVING_SUFFIX_TOKENS=1024 \
NOSA_OFFLOAD_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
.venv/bin/python -m pytest models/nosa/tests operators/nosa \
  cache/tests/test_staging.py tests/integration/test_gr_persistent.py \
  -q --tb=short -p no:cacheprovider
# 1109 passed, 1 optional NOSA_MODEL_PATH check skipped, 1 empty-graph warning;
# 96.17 s. Includes complete shared 64K+1K and standalone 64K+1K checks.

CUDA_VISIBLE_DEVICES=1 PATH="$PWD/.venv/bin:$PATH" \
NOSA_SERVING_CHECKPOINT=/mnt/ssd-wlcb/chenkaiqi/NOSA-8B \
NOSA_SERVING_PREFIX_TOKENS=65536 NOSA_SERVING_SUFFIX_TOKENS=128 \
.venv/bin/python -m pytest models/nosa/tests/test_serving.py::test_cuda_serving_checkpoint_independent_prefixes_and_revisits \
  -q -s --tb=short -p no:cacheprovider
# 1 passed in 37.03 s; all 16 scheme/user/visit outputs equal bitwise, max_abs=0.
```

The isolated slab check observes replacement immediately after allocation while
the previous slab remains owned. It proved only that this pair fit the earlier
logical allowance, not that all simultaneous cache allocations fit. The later
full-checkpoint audit below found real underreservation. A separate GPU fixture allocates
only indexer byte scratch and confirms that PyTorch's allocated/reserved peaks
include both old and new slabs. Per-layer model storage audits remain distinct
from whole-process activation/weight peaks; formal serving must report both.
The final targeted Ruff and `git diff --check` passed.

## Independent follow-up: ownership and profile verification

The user reconfirmed that the public interface is not handed off. This follow-up
changed only NOSA resource ownership, its tests and the independent NOSA profile
validator. Public serving, measure/audit and DeepSeek implementation files were
not edited by this workline.

- `attach()` now rejects foreign or previously owned sessions before changing
  identity. A foreign backend cannot overwrite the original owner's registry
  contract; the original owner can still extend, truncate and release.
- Session registration now owns readiness-setup failure cleanup. Event creation,
  stream lookup or event-record failure drains the device before releasing only
  the newly created session. Failed drain/cleanup preserves ownership, poisons
  resources and rejects reuse. The old session and shared plan remain intact.
  CPU fault injection covers all three phases, both drain outcomes and all four
  schemes through `backend.create_session`.
- Profile verification recognizes shared NOSA evidence in the saved formal plan
  and model policy, preventing multi-field removal from downgrading it to schema
  1. It checks the complete current source set, saved source set and every saved
  digest, including optional official dependencies. The existing two diagnostic
  file content-drift exceptions remain unchanged.

Current evidence is in [the follow-up source inventory](nosa_shared_cache_followup_source.json)
and [component diff](nosa_shared_cache_followup.patch). It includes 24 component
files, including the previously omitted transfer-metrics test; 204 conservative
repository/environment/entrypoint identities; and 120 NOSA correctness-source
identities checked before and after GPU acceptance. No changes or source-set
changes occurred in those 120 files. This inventory does not claim that observed
DeepSeek code was validated or replace the later formal run's full source gate.
Native fused build key remains `cff17752fdd73139`. The external FlashInfer helper
change adds only DeepSeek's `rotary_pair`; all existing functions remain unchanged.

Final checks on this follow-up revision:

- Ownership, readiness failure, drain and transfer CPU checks: **155 passed**.
- Profile CPU checks: **115 passed**. All three retained real schema-1 profiles
  (`h4k_profile_02`, `h16k_profile_01`, `h64k_profile_01` under the original
  `gr_serving_h200_20261002_` prefix) passed CPU `--verify-only`; no artifacts changed.
- The same full NOSA GPU command recorded above: **1137 passed, 1 optional
  `NOSA_MODEL_PATH` skip, 1 empty-graph warning (95.85 s)**. It includes the full
  shared and owned 32-layer 65536+1024 checkpoints, storage/async checks, unique
  sparse reads and internal interval-consistency tests.
- The separate shared 65536+128 checkpoint command above: **1 passed (36.01 s)**,
  with one upstream FlashInfer deprecation warning. All 16 scheme/user/visit
  candidate hidden tensors equal HBM bitwise, max_abs=0.
- Targeted Ruff check/format, CLI help and diff checks passed.

These are correctness and artifact-validation results. No new serving latency
or native diagnostic run has been published; existing performance results remain
identified with their original source and run IDs.

## Current pinned-capacity revision and outstanding requirements

The current allocator policy is
`nosa_torch_native_512b_1mib_tail_pinned_power2_v2`. NOSA offload K and V each own
a separate pinned allocation: at 64K+1024 and 64K+128 capacity, each payload is
slightly above 1 GiB and owns a 2 GiB bin. Host K/V therefore reserve 4 GiB per
session, plus 16 B sparse or 8 B dense host temporaries. CPU reference role
accounting retains payload bytes. `session_bytes()` reports owned pinned
capacity; `session_storage_bytes()` separately reports logical storage.

The frozen source at `/tmp/nosa-allocation-pinned-3v_cm548/source` passed the full
NOSA GPU regression: **1230 passed, 5 skipped, 2 warnings (394.24 s)**. Four skips
are allocation-audit opt-ins executed separately; one is optional checkpoint
metadata. The initial timestamp-join allocation audits failed: **64K+1024:
4 failed (689.61 s); 64K+128: 4 failed (659.59 s)**. Each geometry's 16 candidate
phases passed their budget checks and all-candidate bitwise comparisons; each
geometry's 8 full-prefill captures have unknown CUDA peaks because allocation
generation matching failed. Across both geometries, all 48 CPU phase bounds
pass, and 12 offload constructors independently confirm 4 GiB from two pinned
handouts each. Those original reports remain failed. Source identity and logs are recorded in
[the allocation checkpoint](nosa_cache_allocation_checkpoint.md).

A test-only strict full-callback generation join subsequently passed 46
adversarial tests and fresh full captures: **48/48 phases passed, 32/32 candidate
hidden tensors bitwise equal, zero cache/ordinary evidence errors**. Both
geometries passed all four schemes: 64K+1024 in 585.15 s and 64K+128 in 573.68 s
(test durations, not latency measurements). All active-session bounds fit
without shared or inactive-session credit. CPU temporaries and every 4 GiB
offload pinned allocation are independently covered. See the
[accepted source/evidence inventory](nosa_cache_allocation_source.json) and
[component diff](nosa_cache_allocation.patch). Production sources are unchanged
from the 1230-test GPU regression; only the three audit test files changed.
At that independent checkpoint the shared parser was read-only and public
interfaces had not yet been handed off; the current integration supersedes that boundary.

1. Preserve the independent NOSA source/tests and rerun affected acceptance when
   public integration changes their execution or accounting. Independent
   allocation acceptance is complete for the recorded controlled native setup.
2. Run the bounded diagnostic profile against the future matching accepted formal
   latency run. The independent migration is implemented; no new run is published.
3. Complete the in-progress bind/unbind and constructor cleanup integration,
   C/A in CLI and measure warmup/formal paths, backend lifecycle, formal schema/
   audit/report fields, and the new staging tests in the global GPU entrypoint.
4. After integration, run global CPU/GPU checks and repeat any affected allocation audit;
   complete the formal four-scheme 16-user/two-pass 64K+128 run and new layer-31
   diagnostic run. Keep full source freeze, unique-read and overlap gates intact.
   Broader capacity conclusions still require the plan's stated workload.
5. Only after accepted new run IDs, update README/report conclusions and replace
   affected old artifacts. Mixed DeepSeek/NOSA raw containers remain while their
   unreplaced counterpart still depends on them. No performance artifacts were
   deleted; GR README now explicitly marks NOSA shared changes unmeasured.
6. Complete and independently verify the in-progress
   DeepSeek dense shared double buffering, all-candidate hidden and last-token
   head, formal GR comparison and publication. No new measurement is yet claimed.

NOSA still uses full logical-address staging, with no finite HBM slots/eviction,
no cross-user value sharing and no CUDA Graph or concurrent-service support.

## Historical handoff wait after independent acceptance

At this earlier checkpoint the explicit user instruction was that shared interfaces had not been
handed off and work should remain within independent NOSA scope. That scope is
now complete. Remaining implementation depends on the handoff; source changes
by the concurrent executor do not by themselves grant it. No additional GPU
rerun or independent implementation is needed without a new change or finding.
The next continuation rechecked the 31 component hashes and current public
interfaces: independent evidence remains unchanged, while owner binding and
constructor-failure cleanup are still absent. C/A request checks already exist;
no new handoff was found. This is the second post-acceptance observation of the
same handoff blocker. The user additionally requested `/tmp` cleanup in that
turn, which was completed: superseded raw captures were removed and the latest
accepted evidence was moved, content-verified, to SSD storage with its original
path retained as a symlink. See the allocation checkpoint's storage note.

A third consecutive observation rechecked the same component/interface hashes,
accepted evidence directory and compatibility symlink. No handoff or new
independent work is available; the public interfaces still lack owner binding.
The full goal was therefore blocked on the explicit interface handoff, not
complete. The later user confirmation has resumed the remaining integration,
formal measurement and DeepSeek dense requirements.

## Integration started after handoff

The pre-edit shared interface identities are:

- `executor/serving_backend.py`: `dec9f8e93158476d8eac88fa0fff6e5e7dbc13dde7438b414984a11a4fe319c8`.
- `serving/persistent.py`: `e732204c18edda915b7e6c7c06ae7ebe644577bc81c567beb9e1baa4b9f47dab`.
- `serving/run_multi_user.py`: `073a65de18d18ee95a4ad27bc025417ecaaa808fdf8b0c807b9f1e3c2145cbc6`.
- `experiments/gr_serving/src/measure.py`: `445b60d7ec1c6cd5911ae65d1b187268f86831fb9f8ec3d4eae465d8db3f8151`.
- `experiments/gr_serving/src/audit.py`: `775e0d4b4b15ece3dd1fdbe5f6b8b612cf43a7684b7f3a6a006e597ed79a4b54`.
- `models/deepseek_v32/serving_backend.py`: `48b77dec8e7af782c3b4844dbf9dc5f8d7700c79a160736a7acb270fcacbfd90`.
- `models/deepseek_v32/cache_resources.py`: `dbe0d623cd4eccce38a6f285bbb88b9e9ece6930b55b18fb33137d2b38404d9f`.
- `cache/staging.py`: `5d96ef67d5cd726887f32ac2baa8baa6dfa29be20d9e07325ec2f1688a9b2fd9`.

The shared owner extension uses `bind_owner(owner)` and
`unbind_owner(owner, rollback=False)`. The backend records its plan at binding.
Rollback drains/frees only resources newly allocated under that binding;
ordinary runner close preserves shared storage for reuse. Failed cleanup must
retain ownership. The outer caller closes the backend after closing the runner.
This separates admission ownership from the shorter execution lease.

Initial runner fault/lifecycle suite: **27 passed (1.33 s)**, covering two empty
runners, allocation/pool/audit construction failure with and without an existing
plan, failed rollback, and release failure followed by close retry. These are
CPU contract tests; model integration and final-source regressions are pending.

The current research workload follows the researcher's complete sequential loop
choice in `docs/agents/research-supervisor/loop_motivation.md`; the older IID
requirement in this implementation plan has been corrected to match it. The
specified 16-user/two-pass point remains a controlled acceptance requirement.

The integration's public CLI/measure/audit/report CPU suite now has **435 passed,
4 CUDA-dependent skips (12.41 s)**. It covers explicit C/A in warmup and formal
runners, backend finalization, schema-2 NOSA fixed/temporary DRAM accounting,
complete sequential rounds, and authenticated per-round summaries. Profile
source coverage now includes the newly migrated `deepseek_v32_echo_cache`
experiment, matching the formal snapshot; its targeted suite has **118 passed
(7.63 s)**. Independent DeepSeek memory inventory now enumerates backend dense
staging and pending sources; targeted tests have **13 passed, 1 CUDA skip
(1.61 s)**. These are integration CPU checks, not final frozen-source GPU gates.

Allocation-free NOSA planning on the current native allocator, C=65664, A=128,
Q=1024, L=32, H=2, D=128, BF16, gives:

| Scheme | Shared HBM (B) | Per-session HBM (B) | Per-session DRAM (B) | Maximum sessions |
|---|---:|---:|---:|---:|
| hbm | 1,052,672 | 2,276,651,008 | 2 | 1 |
| serial_sparse / overlap | 70,420,992 | 180,146,688 | 4,294,967,312 | 15 |
| dense_prefetch | 137,629,696 | 180,146,176 | 4,294,967,304 | 15 |

The 4 GiB / 64 GiB limits include every byte: sixteen sparse sessions exceed
DRAM by 256 B, and sixteen dense sessions by 128 B. Tensor allocation APIs were
forbidden during this planning check and CUDA allocated/reserved bytes stayed
zero. The sequential workload predicts all revisit misses at U=16; measured
LRU and latency were not measured. The user subsequently canceled these runs;
the original [publication plan](nosa_shared_publication_scope.md) is inactive.
