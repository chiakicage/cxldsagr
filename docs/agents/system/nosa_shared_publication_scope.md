# Shared serving integration: publication scope and replacement matrix

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

> **2026-10-03 撤回说明：** 本文涉及的旧 DeepSeek 4 GiB / W / chunk 对照已按用户要求撤回，
> 相关实验源码与运行产物已清理；下文仅保留当时的工程过程，不再证明当前容量或性能。
> 当前入口为[固定 P/NH 容量实验](../../../experiments/deepseek_v32_echo_cache/README.md)。

## Current status, 2026-10-04

The canceled plan below is historical. Its proposed H64K/64-GiB loop and
instruction to end the capped heat traces were not the scope of the later
authorized reruns. The actual GR replacement retained the original capped
H4K/H16K/H64K workloads with 4/16-GiB admission caps and is now independently
audited, published and cleaned; see the
[旧 GR 实验（已结束）](experiment_organization.md#retired-gr-serving) and
[execution receipt](nosa_gr_selected_source_execution.md).

The separate fixed P/NH source `a73ad32b...` has accepted and published formal
`_03`, matching profile `_03` and independent reference `_02` evidence, with
exact superseded-output cleanup complete. See the
[fixed report](../../../experiments/nosa_motivation/README.md) and
[publication/cleanup receipt](nosa_motivation_publication.md). Candidate MFU,
tails and compute/IO dominance remain unresolved; fixed async failed its
latency and 90% overlap gates. Acceptance of the evidence does not complete
the efficiency goal or revive the withdrawn DeepSeek scope.

## Historical canceled scope

Date: 2026-10-03. **Execution canceled by the user's later instruction: correctness
only; no more performance tests.** No formal run, fresh chunk screen, performance
profile, default selection or report replacement described below was started.
The tables are a retained design, not an active task list. Existing reports and
raw evidence retained their original source identities; at that checkpoint the
new shared-serving implementation had no new performance result. Correctness and allocation checks
are recorded in [the engineering checkpoint](nosa_shared_cache_checkpoint.md).

This was an execution scope decision, not a measurement result.
It covers NOSA shared execution resources, public admission ownership, and the
DeepSeek dense shared-staging integration. Existing accepted evidence remains
under its original source identity at that checkpoint. The replacement matrix
below was canceled, not subsequently executed. No GPU experiment was run while
preparing that audit; current accepted replacements are identified above.

## Scope decisions

NOSA's current experiment is the complete fixed-history user loop specified in
[the implementation plan](nosa_shared_cache_implementation_plan.md), section 8.2.
The earlier capped heat traces at H4K/H16K/H64K are ended as experimental scopes.
They need not be repeated to publish the current H64K loop. The replacement must
remove their history-length and heat-population claims, rather than present the
new workload as a before/after speedup. The old DRAM budget was 16 GiB; the new
budget is 64 GiB. All four reasonable cache schemes remain in the new comparison.

DeepSeek's seven physical GR runs remain useful: they distinguish fixed W=2048
from deployment W=C and provide two independent repeats for C1024/W1024 and
C2048/W2048. Replacing the entire report with only a default-C1024 run would lose
that evidence. The new generic Python execution guard is inside every scheme's
request timing, even though non-dense math, CUDA launches and C/W byte formulas
are unchanged. Consequently all four schemes are remeasured at all seven physical
run positions. Do not splice new dense timings into old-source control rows.

DeepSeek artifacts now belong to
[`experiments/deepseek_v32_echo_cache`](../../../experiments/deepseek_v32_echo_cache/README.md).
The [migration record](echo_cache_experiment_migration.md) and
artifact mapping（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/artifact_locations.json`）
preserve original locations and identities. Directory migration itself does not
trigger GPU remeasurement and its historical records must not be rewritten.

## Freeze and engineering gates

1. Finish both models' owner/rollback contracts, shared staging and public
   runner/CLI/measurement integration. Run targeted CPU lifecycle, failure and
   admission tests, then the required CPU regression.
2. Freeze the complete actual source tree and installed backend identity before
   GPU acceptance. `measure.source_snapshot` includes both `gr_serving` and
   `deepseek_v32_echo_cache`; NOSA profile's formal-source enumeration must match
   this exact set. Do not weaken source/hash gates to bridge old measurements.
3. Run the shared staging GPU cases and affected model/operator regressions.
   NOSA full-checkpoint acceptance uses independent empty sparse prefixes for
   resident/offload, H65536+A128 and H65536+A1024, all candidate hidden states,
   separate cache allocation/ordinary activation evidence, and current source.
   DeepSeek requires the new dense path's complete hidden and final-token logits,
   execution/transfer lifetimes, independent sessions and shared-stage budgets.
   These are engineering checks, not paper performance samples.
4. Refresh DeepSeek engineering memory evidence before default review. The old
   source-bound observer summary cannot certify current dense staging or new
   runtime source. Recompute the full analytic C/W planning matrix and directly
   observe its changed allocation/lifetime and largest-inventory anchors. At
   minimum cover C1024/W1024 and C2048/W2048 all four schemes plus C256/W256
   HBM+dense if those C remain accepted. If fresh screening admits another
   configuration or its retained-user maximum changes, add the necessary anchor;
   do not interpolate endpoint peaks into a measured peak. Store engineering
   evidence outside experiment performance outputs; publish only its source-bound
   evidence index in the default review.

The accepted earlier NOSA allocation generation on SSD remains original-source
evidence. New source includes owner rollback and parser-label/test updates; old
GPU results must not be relabeled as new integration acceptance.

## NOSA formal runs

Common configuration: complete 32-layer NOSA-8B, BF16, H65536+A128, prefix chunk
1024, planned C65664/A128/Q1024, 4 GiB HBM / 64 GiB local CPU DRAM, sequential
users, exactly two complete rounds, seed 42, two excluded warmup requests per
scheme. Each U has its own unique run ID and workload; all four schemes consume
that same workload from independent empty caches. These runs return all candidate
hidden states and do not execute an LM head.

| New run suffix | U | Requests per scheme | Four-scheme requests | Purpose / expected admission regime |
| --- | ---: | ---: | ---: | --- |
| `nosa_shared_h64k_u1` | 1 | 2 | 8 | All schemes fit; a hit reference point |
| `nosa_shared_h64k_u8` | 8 | 16 | 64 | HBM rebuilds; offload retains histories |
| `nosa_shared_h64k_u15` | 15 | 30 | 120 | Exact offload session-capacity edge |
| `nosa_shared_h64k_u16` | 16 | 32 | 128 | Mandatory original control; all schemes rebuild |
| Total | | 80 | 320 | Measured requests; warmups are separate |

Use a fresh common run prefix; these suffixes are descriptive templates, not
already executed run IDs. The actual planner on H200/native PyTorch gives these
byte reservations without tensor allocation:

| Scheme | Shared HBM | Per-session HBM | Per-session DRAM | Session limit |
| --- | ---: | ---: | ---: | ---: |
| HBM | 1,052,672 | 2,276,651,008 | 2 | 1 |
| Serial sparse / overlap | 70,420,992 | 180,146,688 | 4,294,967,312 | 15 |
| Dense prefetch | 137,629,696 | 180,146,176 | 4,294,967,304 | 15 |

At U16, sparse DRAM reservations exceed 64 GiB by 256 B and dense by 128 B.
The exact hard budget must be respected. These are planning bounds, not new
observed capacity or latency results. Check actual loop hits, evictions, round
identity and both tier ledgers against them after execution.

Run one separate NOSA layer-31 profile bound to the accepted **U16** formal run.
The current tool takes its first three revisits, request IDs 16/17/18, yielding
six profiled samples (three requests times serial/overlap) and six uninstrumented
controls. Each builds its own empty sparse prefix and compares all candidate
hidden states to the saved formal HBM output. Reserve trace storage separately
before allocation; export before returning the execution lease. U8 profiling is
optional follow-up. U1 with two requests cannot satisfy the tool's three-revisit
requirement.

Report whole-query-batch serial sparse-union fetch as the latency comparator,
unique physical host reads, actual internal copy/attention intervals, and both
page-envelope and nonempty-stripe overlap ratios for **each** sample. A 90%
claim requires both ratios >=0.9 in every sample. A valid negative measurement
remains a result; it does not pass that performance claim. A direct-session
profile is not evidence of multi-user LRU occupancy.

Per-run outputs follow `experiments/gr_serving/output/{data,log,profile}/<run_id>`.
The new report should contain metadata/source identity, independent audit,
per-request and per-round tables, hit/rebuild coverage, shared/session budget and
actual-byte tables, latency distributions with sample counts, and the accepted
profile metadata/analysis. Proposed tracked location: `report/shared_h64k/`.

## DeepSeek refresh matrix

First rerun the fresh-source three-layer screen at C={256,512,1024,2048,4096},
H65536+A128, P32768, 4 GiB HBM / 64 GiB DRAM. Reapply allocation feasibility and
the complete original numerical gates; do not assume old C512/C4096 exclusions.
Use the existing cold 2-warmup/5-sample and extend 5-warmup/20-sample protocol for
accepted C. This gives a fresh authenticated screen for the strict report gate.
The old screen's standalone compute is unaffected by serving-only P6 changes,
but its signed source set includes the changed serving planner and executor
contract, so it cannot be relabeled or silently rebound to fresh GR source.

If accepted C remain 256/1024/2048, run this exact matrix on one unchanged frozen
source/hardware/backend identity:

| Position / new suffix | C | W | Independent full traces | Protocol use |
| --- | ---: | ---: | ---: | --- |
| `fixed_c256` | 256 | 2048 | 1 | Fixed workspace |
| `fixed_c1024` | 1024 | 2048 | 1 | Fixed workspace |
| `fixed_c2048` | 2048 | 2048 | 1 | Fixed and deployment |
| `deployment_c256` | 256 | 256 | 1 | Deployment |
| `deployment_c1024` | 1024 | 1024 | 1 | Deployment |
| `repeat_deployment_c1024` | 1024 | 1024 | 1 | Independent repeat |
| `repeat_deployment_c2048` | 2048 | 2048 | 1 | Independent repeat, both protocols |

Every physical position measures HBM/ECHO/serial sparse/dense prefetch, U16,
32 ordered requests per scheme, H65536+A128, NH1,050,624/P32,768, 4 GiB/64 GiB,
two excluded warmups per scheme, and all candidate hidden plus last-token logits.
This is seven physical runs, 28 scheme traces and 896 requests. Sharing C2048
between two protocol views does not add a replicate. If fresh screening changes
the accepted C set, update the matrix and all claims explicitly rather than
silently dropping valid slow configurations or retaining a failed one.

Collect fresh independent diagnostics for C1024/W1024 and C2048/W2048 after their
formal runs: execute all 32 requests of all four schemes, sample layer/batch
evidence only for request IDs 0/15/16/31, and compare full hidden/logits. These
are 256 intrusive requests total, not formal latency samples. Add another
diagnostic if a different default candidate becomes relevant. The report requires
diagnostic/formal runtime and request identity, so old-source diagnostics cannot
be reused as current-source mechanism evidence.

Refresh `echo_chunks` aggregation and default review only after formal,
diagnostic, screen and engineering-memory gates pass. Retain the conclusion that
ECHO may be slower than serial sparse if the new measurements show it; shared
capacity and fusion speed are distinct comparisons. Store new runs in
`experiments/deepseek_v32_echo_cache/output/{data,log,profile}/<run_id>` and replace
the corresponding `report/chunk_sweep/` and `report/echo_chunks/` material only
with the newly accepted publication.

## Replacement transaction

Before new acceptance, retain all old README numbers, report materials and their
raw evidence, with a clear current-integration-unmeasured banner. On acceptance
and publication, update README results/claims and remove replaced artifacts in
the same report update. Never retain a superseded result as `legacy`, merely
rename it, or clean early to make space.

| Old scope | Replace / end after acceptance | Exact affected old artifacts |
| --- | --- | --- |
| NOSA capped H4K/H16K/H64K traces | End old workload/length scopes; publish current H64K loop | `gr_serving/report/h4k`, `h16k`, `h64k`; formal runs `gr_serving_h200_20261002_h4k_01`, `...h16k_01`, `...h64k_01` |
| NOSA old layer-31 profiles | Replace with fresh U16-linked profile | `gr_serving_h200_20261002_h4k_profile_02`, `...h16k_profile_01`, `...h64k_profile_01` in all three output categories |
| DeepSeek GR seven-run publication | Replace all seven four-scheme physical runs | Five `20261003_echo_gr_chunks_01_{fixed_c256,fixed_c1024,fixed_c2048,deployment_c256,deployment_c1024}` runs and two `20261003_echo_gr_chunks_repeat_01_deployment_c{1024,2048}` runs, all under the new experiment |
| DeepSeek aggregated publication | Replace materialized results and review | `report/echo_chunks/` and `output/data/20261003_echo_gr_chunks_publication_01` |
| DeepSeek mechanism diagnostics | Replace both source-bound diagnostic runs | `20261003_echo_gr_diagnostics_c1024_04` and `...c2048_04`, all output categories |
| DeepSeek fresh-source screen | Replace after new complete accepted screen | `report/chunk_sweep/` and `20261003_echo_shared_chunks_02`, all output categories |
| DeepSeek old engineering-memory report dependencies | Replace with current engineering evidence index; remove superseded experiment-hosted raw dependencies after references are updated | `20261003_echo_memory_c1024_03`, `...c2048_fixed_03`, `...c512_hbm_03`, `...c256_dense_05`; update default-review evidence binding |

For every formal run in the table, cleanup includes its matching data/log/profile
directories, not only the tracked chart. Preserve independently valid and
unaffected experiments, the user-authorized migration record, and compact source
or engineering provenance as appropriate. Update navigation to current outputs;
do not alter original historical commands or hashes in the migration record.

The independent NOSA operator/default-owned-cache and DeepSeek MFU/nonmatrix
experiments are outside this matrix unless a final source-impact review finds a
changed execution or measurement path there. Their directory names alone do not
justify either rerunning or deleting them.
