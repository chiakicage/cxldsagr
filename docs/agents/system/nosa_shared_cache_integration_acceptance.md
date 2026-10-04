# NOSA / DeepSeek shared-cache integration acceptance

Updated 2026-10-03. **Completed under the user's correctness-only scope.**
Scope: implementation, numerical correctness, ownership,
asynchronous lifetimes and cache allocation bounds. The user explicitly canceled
further performance tests. No formal serving run, chunk timing screen, performance
profile, default-selection review or replacement report was started in this batch.

## Implementation

- The public runner binds one admission owner before allocating shared resources.
  A second runner is rejected even while the first has no sessions. Construction
  rollback releases only newly created resources, preserving a preexisting plan.
  Failed cleanup retains the owner and disables reuse; close can be retried.
- NOSA backend owns its bounded sparse workspace, dense double staging and
  hbm/dense FA3 scratch. Sessions retain independent histories, CIS, derived
  records, pending append and indexer state. Explicit C/A/Q limits apply before
  allocation or mutation. Owned standalone model calls remain supported.
- DeepSeek dense uses one backend-owned `[2, C, record_width]` staging allocation.
  Sessions own independent host records, maps and indexer state; a layer borrows
  its staging view only during the execution lease. Slot reuse waits for its
  previous consumer, and lease return drains speculative copies as well.
- The CLI, memory observer and measurement/profile entrypoints use the shared
  lifecycle and separate shared/session accounting. Their correctness tests do
  not establish new performance results.

## Frozen identity and test environment

Frozen source: `/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-shared-integration/freeze03`.
Evidence sibling: `freeze03-evidence`. All commands use its `run.sh` launcher,
which sets the frozen working directory and Python import path.

| Identity | Value |
| --- | --- |
| Freeze ID | `7c12bfb334dee401bfdb66d904787962e49ef59e70811a3701249b26db06ed21` |
| Full project inventory | 793 entries |
| Project SHA256 | `d31697e4ebd55e3dff8c3b076fe0d95d19992e517be5013a43bb04d28bb6cd3e` |
| Manifest SHA256 | `754d15f982d88b0d19da2f3eb388575038f1f62ebdef4f073d8423a0518754a3` |
| Device reported by driver | NVIDIA M403, Hopper/SM90, 143771 MiB |
| Driver | 570.124.06 |

The four official gitlinks are materialized through explicit shared checkout
links. Their HEADs, tracked source bytes and prepared include links are verified
separately; only those four gitlinks have skip-worktree flags. No ordinary source
is skipped. The initial GPU attempt lacked the ignored ECHO test reference:
7 failures and 14 setup errors all reported that missing directory. It was then
supplied at clean commit `bc1b75c1000010d0ac6f032ebaac283255c050b1`, with all
2,884 tracked entries and two uninitialized nested gitlinks verified separately.
The full GPU rerun passed without changing project runtime or test source.

The combined dependency verifier is
`freeze03-evidence/echo-test-dependency/verify.sh`; its manifest SHA256 is
`72eba4b08276d3fea3360263a7eb97e36e94e1fe8522ad4eba6700bcf43f0e80`.
Test commands, environments, exit statuses and logs remain outside the repository
under the evidence directory. Test durations are not serving latency samples.

## Correctness results

| Check | Result |
| --- | --- |
| `bash scripts/run_tests.sh cpu` | 2672 passed, 836 CUDA/opt-in skips, 58 subtests, 1 warning |
| Full GPU rerun, physical GPU 1 | 1639 passed, 6 opt-in skips, 16 warnings; combined pre/post verification passed |
| CLI and import-origin checks | 11 checks passed |
| NOSA shared full 32-layer 65536+1024 | Passed in the full GPU suite, independent empty prefixes and interleaved users |
| NOSA standalone owned 65536+1024 | Passed in the full GPU suite, independent resident/offload prefixes |
| NOSA shared full 32-layer 65536+128 | Passed separately; all 16 scheme/user/visit records equal bitwise, max_abs=0 |
| NOSA intrusive allocation, both candidate lengths | 8 cases, 48/48 phases accepted; 32/32 complete candidate hidden comparisons bitwise equal |
| DeepSeek real first three checkpoint layers, 65536+1024 | Passed; complete hidden and last-token logits equal bitwise |
| DeepSeek ten independent dense-block serving surrogate | Passed; all four schemes' candidate hidden/logits and independent storage/shared staging checked |
| DeepSeek intrusive allocation matrix | 10 cases, 320 requests; all 150 sampled phases passed independent CPU reconstruction |

The six GPU opt-in skips are the four NOSA allocation cases and the two DeepSeek
checkpoint cases; they were explicitly run in separate processes. Skips are not
counted as passes. Warnings are from upstream profiler, CUDA-graph or deprecation
paths. The checks cover shared staging cross-stream readiness, delayed consumers,
failed speculative copies, rollback, poisoning, close retry and exact admission.

NOSA allocation captures ran on physical GPUs 2 and 3. Each geometry has four
schemes, two independent 64K prefixes and two interleaved visits per user. Every
phase fits both total and active-session temporary reservations; spare shared or
inactive-session capacity is not credited to the active session. The minimum
active-session HBM margin is 34,398,208 B; the minimum DRAM margin is 0 B. Every
offload session's two pinned handouts total exactly 4,294,967,296 B. Temporary CPU
peaks remain charged: hbm 1 B, dense 8 B, sparse/overlap 16 B.

The independent NOSA review hashes 144 raw phase artifacts and eight constructor
files, checks the arithmetic, complete layer/chunk schedules, pending-free
retention and allocator-history identities. Review SHA256:
`886b4d0c57fc27787db4977bed21cc810b2e0046545c2f946b1fdf5d3819801d`.
Its original result is `freeze03-evidence/validation/nosa-allocation-review.json`;
the detailed command/log index is `nosa-engineering-evidence-index.json` beside it.

## DeepSeek allocation acceptance

The frozen GPU 4 capture accepted all ten cases: C1024/C2048 with all four schemes,
plus C256 hbm/dense. Each executes 16 users twice at H65536+A128, giving 320
complete requests. All candidate hidden/logits comparisons were exact, max_abs=0.
Independent CPU replay reconstructed all 150 actual allocation phases sampled at
request IDs 0/15/16/31 and matched every report field, with no evidence errors or
reservation overflow. Both final audits and the correctness-only aggregate exited
0 with empty stderr. The aggregate binds 3,374 original file records, all raw
replays, complete source/backend/workload identities and the successful driver.

The capture performs complete numerical comparisons during execution. Nonresident
output tensors are not retained, so the independent CPU audit does not claim to
reopen those outputs and repeat their numerical comparison. The memory result
covers observed phases, not a whole-process peak over the entire trace. No recorded
duration is used as a performance result, and no new default configuration is selected.

Evidence root:
`/mnt/ssd-wlcb/chenkaiqi/.cache/cxldsagr-shared-integration/deepseek-freeze03-validation`.
Its `exit_ledger.json` binds the two checkpoint tests, both allocation commands and
the before/after freeze checks. Independent raw replay and the final engineering
aggregate are under `freeze03-evidence/memory-publication-tools/`.
The final `memory_acceptance_summary.json` SHA256 is
`eb4fa4ce25fd1c795d5bd3fa6dd7c9d208fc9b7464454bff043e65235535a123`;
the complete GPU `exit_ledger.json` SHA256 is
`14de83aac189d5df8cef1b206ef14304b945562ca128ce099784cc4ecfe37943`.
Current command/result and evidence identities are indexed in
[the integration evidence record](nosa_shared_cache_integration_evidence.json).

## Delivery/source consistency

An independent final comparison found all 352 runtime, test and configuration
files unchanged between the delivered main tree and freeze03, with no extra
ignored execution source. The broader second check covered 349 source files, or
359 including fixtures/config, with the same outcome. Main documentation updates
and the known post-freeze `token_memory.py` report-only addition are recorded
separately; the latter was not imported or executed by this acceptance batch.
Receipt: `freeze03-evidence/validation/main-freeze03-consistency.json`, SHA256
`4b083624c79ef5b14ca0e4718d33bd749374aedff0a6f819a675ab17db92317c`.

## Scope limits and canceled work

NOSA still has full-address staging, without finite HBM slots/eviction, CUDA Graph
capture or concurrent serving. Allocation audits keep cache, weights, ordinary
activation and process allocated/reserved peaks separate. NOSA's direct-backend
fixture does not prove the unrun 4 GiB/64 GiB complete user-capacity loop. The
DeepSeek surrogate is ten independently stored checkpoint dense blocks, not a
trained 8B model or a complete 61-layer validation.

All 11 prepared performance job specifications were marked canceled before any
ran; their launcher rejects canceled jobs. This work generated no new performance
report and deleted no old experimental report or raw run. Old numbers retain their
original run/source identities, including the separately added token-memory
analysis; none is relabeled as new shared-serving performance. The current task
does not require the inactive publication matrix to be executed.

About 17 GiB of unused `/tmp` contents were removed across this work, including
337 MiB of inactive pytest directories after checking for live references. Current
validation temporaries and raw evidence use SSD storage; the root filesystem has
about 36 GiB available.
