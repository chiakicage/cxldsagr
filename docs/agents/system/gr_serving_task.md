# Persistent GR serving implementation and experiment contract

Latest status, 2026-10-02: the user identified problems with the ECHO implementation's
MFU and cache strategy, and with the current DeepSeek MFU. The old model-local
baseline comparison is therefore not established. Numerical and accounting audits
below do not validate implementation efficiency or the cache policy; DeepSeek/ECHO
rankings must await correction and fresh measurements. Temporary cache scratch
coverage also needs review; boundary tensor accounting is not a full peak audit.
Preserve original run identities and artifacts until accepted replacements exist.
This is a status/commit handoff, not a claim that those problems have been fixed.

The later 16-user / 32-request sequential run 01 failed source-identity validation;
run 02 remains planned, with no accepted replacement. See
[sequential rerun](gr_serving_sequential_rerun.md). The completion statements below
refer only to the original short-trace execution and its historical checks.

Follow-up correction: the user rejected the nominal population sweep and its
lack of revisit misses. The measurements below remain old-trace records, not
evidence of valid active-user scaling. The proposed replacement is documented
in [workload redesign](gr_serving_workload_redesign.md); it has only CPU schedule
and LRU checks so far, with no production implementation or new GPU run.

Completed measurements on 2026-10-02. Research entries: 1.2, 1.4, 1.5,
2.1–2.5, 3.1–3.2 and 4.1–4.3. No measurement or GPU profile process is active.
The former live-run notes are superseded by this accepted-run record.

## User scope and execution choices

The user requested single-GPU sequential GR requests with persistent fixed user
history and changing candidates. DeepSeek V3.2 runs first: independently copy
the real first three dense layers and their corresponding hidden/residual inputs
to approximately 8B physical parameters, with no MoE. Compare HBM, ECHO, serial
sparse and dense layerwise prefetch. Then use full NOSA-8B and compare HBM,
serial sparse, dense layerwise prefetch and native sparse-fetch/attention overlap.
All schemes receive identical allowed HBM and CPU DRAM cache budgets.

Later explicit choices were populations 1/8/32/64/128/256/512, heat-driven draws,
at most eight revisits per user (excluding first visit), and separate fixed
histories of 4096/16384/65536 tokens. Revisit classification must remain independent
of cache hits; evicted revisits still include reconstruction in their latency.
The final requested step is to apply the current repository Research Supervisor.

The user targeted about 30 minutes. Root disclosed and proceeded with about
30 minutes per history after an optional scope clarification remained unanswered;
this is an execution assumption, not a confirmed user answer. The 128-token
candidate, common 4 GiB HBM / 16 GiB DRAM quotas, request caps 32/32/6 and
DeepSeek sparse slots 4096/8192/32768 are the disclosed runtime arrangement.
Actual durations were 8.258/31.983/32.797 minutes, excluding separate profiling.
The three-run total does not meet a combined 30-minute limit.

## Accepted runs

| History | Latency run | Cases / requests | Native profile run |
|---|---|---|---|
| 4K | `gr_serving_h200_20261002_h4k_01` | 56 / 1608 | `gr_serving_h200_20261002_h4k_profile_02` |
| 16K | `gr_serving_h200_20261002_h16k_01` | 56 / 1608 | `gr_serving_h200_20261002_h16k_profile_01` |
| 64K | `gr_serving_h200_20261002_h64k_01` | 56 / 336 | `gr_serving_h200_20261002_h64k_profile_01` |

All launchers exited 0. Independent audits accepted input/heat/cap replay,
request pairing, revisit identity, exact LRU victims, budget reservations and
actual allocations, correctness records, saved CPU references, source/dependency
identity and report calculations. All 2664 non-HBM comparisons were bitwise equal.
Each profile has six records for three actual saved revisits, independent prefix
construction and complete hidden comparisons. All nine overlap samples are below
the per-sample 90% criterion; none are discarded for that reason.

The 4K latency source digest is
`67f91c34ead43d4056500661caa1c67e8ef8a50bc25327e081a0561bbab43060`.
The 16K/64K digest is
`45511acf6b7ec0e0a9bf3e21a4f9b1c9b74f88a46f03303f3f93cc9460dfd44c`.
Only profile.py and its test changed between them, to ignore synthetic module
file paths during diagnostic snapshots. The 4K profile records and authenticates
this exact diagnostic-only difference; no performance source drift is accepted.

Original data remains under `experiments/gr_serving/output/{data,log,profile}/`
by run ID. Selected evidence is in the experiment's history-specific report
folders. See [experiment report](../../../experiments/gr_serving/README.md) and
[independent acceptance review](gr_serving_review.md) for publication checks.

## Implementation and interpretation

Global whole-user LRU releases both memory tiers on eviction. Admission reserves
model-declared conservative cache capacity before allocation and audits actual
storage. Budgets include resident indexer/derived records, main KV pools,
staging, mappings, scratch and pending append. All model layers and GPU work
complete before commit; successful candidates truncate back to fixed history.
Exact prefix identity changes or insufficient cache capacity cause reconstruction.

Weights and ordinary activations are outside the cache quota. Reports distinguish
weight bytes, output payload, cache samples/reservations and process allocator
peaks. The first NOSA case's process peak includes transient prior DeepSeek weights
at model transition; it is not an isolated NOSA activation or weight peak.
The artificially small cache quota is not the H200 physical capacity.

Every model/population shares an authenticated token trace across its schemes.
Two discarded warmup requests precede each scheme; measured population cases
start with independent empty caches. The timer includes admission/eviction, cold
prefix construction, complete candidate work, synchronization and truncate.
Weight loading, compilation, workload generation and between-request numerical
checking/serialization are outside each request timer.

DeepSeek has 7,827,793,408 parameters with independent dense-block weights/cache
and source-input replay; this is a checkpoint workload surrogate, not a trained
8B model or full 61-layer validation. DeepSeek computes all candidate hidden and
the last-token LM head, while NOSA computes hidden only. Dense prefetch changes
transfer scheduling, not sparse attention semantics. NOSA still has full-address
layer staging, not a finite token/page-slot cache. No network, queueing, arrival
replay, concurrency or task-quality claim is made.

Heat sampling keeps original eligible-user weights, removes users after nine
visits and stops at the request cap or exhaustion. No initial tour or equal
quota is forced. At 64K, observed users are 1/4/6/5/6/6/6 and revisits are
5/2/0/1/0/0/0. Empty scopes remain null; configured 512 users does not mean
512 users were accessed. Short traces cannot establish stable tail latency.

16K NOSA has 13 HBM revisit misses among 63 revisits versus zero for each offload
scheme; HBM still has the lowest all-request mean in every population. All
DeepSeek, 4K and 64K measured revisits hit. NOSA overlap lowers the all-request
mean versus serial sparse in 20/21 configurations, but does not consistently
lower revisit means. These results distinguish conditional capacity benefits,
cold-history execution and retained-prefix execution. Internal ratios cover actual softmax work,
not the full kernel lifetime or complete attention execution.

## Validation and research handoff

The CPU suite after shared generator changes passed 1653 tests (688 skipped,
58 subtests). Subsequent auditor/profile fixes passed 53 and 58 focused checks;
CLI/measurement/integration checks passed 22. Relevant Ruff checks passed.
Independent full-checkpoint prefix/candidate tests preceded the formal runs;
formal candidate comparisons and native profiles provide the current GPU evidence.
No repeat of unrelated GPU tests was used as a substitute for the experiment.

Root read the current repository `skills/research-supervisor/SKILL.md` after
formal measurement completion and updated status, roadmap, sources and updates.
T-004/T-005 first-round exploration is complete; T-006 coverage/budget sensitivity
and T-007 serving-cost attribution remain suggestions, not automatically started
work. Researcher edits about the tentative GR scene and HBM/DRAM scope are retained.

## Rejected execution attempts

Earlier runs 01–05 are not valid experiment results. Run01 was withdrawn because
an intrusive startup diagnostic might have overlapped timing. Runs02/03 were
interrupted; root /tmp capacity was exhausted, though the exact cause of run03
was not recoverable. Run04 failed at mktemp before measurement. Run05 was stopped
when the user changed the matrix. Failed profile01 stopped before GPU execution
on a synthetic module file path. Calibration source-gate failures correctly
prevented publication while code was changing.

Those diagnostics remain outside experiments; no rejected data or performance
number is used as a comparison. Formal staging used the SSD-backed TMPDIR to
avoid the root filesystem's 20 GiB limit. All current accepted results retain
source snapshots and original data; no unaffected operator experiment was replaced.
