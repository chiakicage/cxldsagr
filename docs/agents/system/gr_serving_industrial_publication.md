# Industrial 64K publication inventory

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

2026-10-02. This is a read-only inventory and future acceptance checklist, not a performance
report. The industrial 1024-user / 4096-request run was stopped before measurement when the
user changed the immediate request to a 16-user / 32-request sequential group with 4 GiB HBM
and 64 GiB DRAM. Its launcher exited 143; it has no accepted new result. Do not apply the
cleanup below on the strength of that interrupted attempt. Source work for the newly requested
group is independent of this deferred industrial publication plan.

## Exact replacement inventory

Once a valid replacement for the old 64K latency and profile has been fully accepted and
published, remove these six old output directories together; do not retain them as legacy data:

- `experiments/gr_serving/output/data/gr_serving_h200_20261002_h64k_01/`
- `experiments/gr_serving/output/log/gr_serving_h200_20261002_h64k_01/`
- `experiments/gr_serving/output/profile/gr_serving_h200_20261002_h64k_01/`
- `experiments/gr_serving/output/data/gr_serving_h200_20261002_h64k_profile_01/`
- `experiments/gr_serving/output/log/gr_serving_h200_20261002_h64k_profile_01/`
- `experiments/gr_serving/output/profile/gr_serving_h200_20261002_h64k_profile_01/`

The old latency data directory contains `analysis/`, `reference/`, `source/`, `workloads/`,
`audit.json`, `correctness.jsonl`, `measurements.jsonl`, `metadata.json`, `render_report.py`,
and `source_manifest.json` (about 175 MiB at inventory time). The old profile data directory
contains `formal_inputs/`, `samples/`, `source/`, `analysis.json`, `metadata.json`,
`source_manifest.json`, and `work_intervals.json` (about 27 MiB). The two profile-category
directories are empty; the log directories hold the original measure/profile/verify logs.

All 19 current files in `experiments/gr_serving/report/h64k/` belong to those two old runs:

```text
audit.json
cache_and_memory.csv
metadata.json
per_request.csv
per_request.png
per_request.svg
per_request_readable.png
per_request_readable.svg
population_coverage.csv
profile_analysis.json
profile_metadata.json
provenance.json
source_manifest.json
summary.csv
summary.json
summary.png
summary.svg
summary_readable.png
summary_readable.svg
```

Replace the directory's contents as one accepted publication update. New material may retain
the same descriptive filenames, but must identify the new run, inputs, budgets, source and
derivation. Reusing a renderer requires saving and signing the actual renderer used for the new
result before the old output copy is removed; old figure bytes are not new result evidence.

Preserve `report/h4k/`, `report/h16k/`, and all `output/{data,log,profile}/` directories for:

- `gr_serving_h200_20261002_h4k_01`
- `gr_serving_h200_20261002_h4k_profile_02`
- `gr_serving_h200_20261002_h16k_01`
- `gr_serving_h200_20261002_h16k_profile_01`

Those geometries have not been rerun. Preserve their original workload/budget/source boundaries;
do not combine their numbers with a new 64K trace as a controlled history-length sweep. No other
NOSA operator, full-model 64K+1K, DeepSeek 61-layer, or legacy SM120 experiment is replaced by
this serving publication.

## Documentation touch points after new evidence exists

| File | Required publication review |
|---|---|
| `experiments/gr_serving/README.md` | Replace the old entire 64K section, its five tables, profile request IDs 1/2/3, 4/16 GiB quota, six-request boundary, commands, renderer/source hashes and links. Update the opening run status, result index and workload section. Remove stale "改动后未运行 GPU" statements once actual execution is accepted. The old 168-case/3552-request/2664-comparison and 20/21 aggregates include removed 64K evidence and cannot survive as current combined conclusions. |
| `README.md` | Update the GR serving status near lines 11–16; it currently says only old short traces exist and no new comparison ran. Preserve model and scene limits. |
| `experiments/README.md` | Update only the GR serving entry to distinguish the accepted new 64K case from pending 4K/16K replacements. |
| `models/nosa/README.md` | Review the opening old-three-trace status near lines 13–15. Preserve independent full-checkpoint correctness results and operator reports. |
| `docs/agents/system/gr_serving_review.md` | Replace current-acceptance claims and old 64K evidence links/aggregate counts. Its old coverage, all-hit, profile and publication sections cannot authenticate new data. Historical decision context may remain clearly labeled without treating deleted artifacts as available results. |
| `docs/agents/system/gr_serving_task.md` | Update active scope, completion statement, accepted-runs row, old no-active-process statement and aggregate numerical claims. Distinguish old execution choices from the user's latest requested budgets/trace. |
| `docs/agents/system/gr_serving_nosa.md` | Update post-run profile handoff status only after actual new profile acceptance; keep independent prefix and softmax-window boundaries. |
| `docs/agents/system/gr_serving_industrial_rerun.md` | Record terminal outcome of this interrupted run and its lack of measurements. Do not publish its planning estimate as latency. |
| `docs/agents/system/gr_serving_workload.md` and `gr_serving_workload_feasibility.md` | Retain old cost calculations only as labeled planning history; remove any implication that their deleted source outputs remain present or support new results. |
| `docs/status.md` | Supervisor-owned: reconsider 1.3–1.5, 2.2–2.5, 3.1–3.2, 4.1–4.3 and the "CPU only/no new GPU" status using accepted observations. Do not declare scene/quality/general capacity conclusions solved from one trace. |
| `docs/roadmap.md` | Supervisor-owned: update the CPU-only introduction and T-006 using the actual scope completed; retain only current pending work. T-002/T-003/T-007 are not automatically solved by this run. |
| `docs/agents/research-supervisor/sources.md` | Supervisor-owned: S-008–S-010 contain old three-run identities/aggregates/profile links; add new verified sources and mark replaced 64K evidence appropriately. S-011–S-013 describe later workload corrections and must not be overwritten as if they were new measurements. |
| `docs/agents/research-supervisor/updates.md` | Supervisor-owned: append the evidence-based understanding change after acceptance; historical user corrections remain intact. |

`models/deepseek_v32/README.md`, `cache/README.md`, `serving/README.md`, `executor/README.md`,
`docs/agents/system/implementation-status.md`, and `research-context.md` were also inspected.
Their independent full-model/operator measurements are outside this replacement. The standalone
serving CLI's Beauty default is a separate interface and must not be rewritten merely because a
formal experiment selects industrial or sequential requests. Recheck links, but do not perform
unrelated edits. Root will read the current project Supervisor skill when final evidence exists;
this inventory does not invoke Supervisor or modify its files.

## Acceptance and publication sequence

1. Confirm terminal success of the actual replacement process and complete metadata. A launch,
   growing log or passing CPU tests is not measurement completion. Preserve the old 64K report
   and outputs throughout any still-running/incomplete replacement. Source stays frozen through
   formal audit and native profile.
2. Run the CPU auditor against the complete successful new latency directory with a new audit
   JSON destination, the exact expected run ID and independently recorded source digest.
   For an industrial run, `--expected-trace-directory
   GR/generated/industrial_10m_pv_share_t4096_seed42` is mandatory. Use the default CPU reference
   check: existence-only returns structural status and cannot accept the run. Audit every saved
   request, all four schemes in each model, all correctness records, all HBM tensor references,
   actual LRU victims/hits, reservations, sampled allocations and report calculations.
3. For the deferred industrial 1024/4096 scope, expected complete counts would be 8 cases,
   32768 measurements/correctness rows, 24576 non-HBM comparisons and 8192 HBM reference files.
   The original CSV has 751 first visits and 3345 revisits per case. These expectations are not
   transferable to the new sequential 16/32 request. Workload canonical `access_trace_sha256`
   signs access records; top-level measurement `access_trace_sha256` signs copied CSV bytes.
   Do not confuse those two distinct hashes.
4. After formal acceptance, run a fresh native NOSA layer-31 profile on the same physical GPU,
   with a distinct run ID and `--latency-data` pointing to that accepted run. Select the actual
   saved population using `--num-users`. For industrial 1024/4096, the predetermined first three
   CSV revisits are request IDs **57/59/61**, users **731/353/632**, previous IDs **31/34/54**;
   authenticate these against saved model rows rather than copying the old 1/2/3 profile.
   The profile launcher verifies all six serial/overlap samples and 18 full-hidden comparisons,
   unique reads, selections, nonempty stripes, page envelopes and source linkage before copying
   accepted outputs. Do not add a GPU run merely to repeat a passed unrelated correctness test.
5. Retain every valid profile sample, including ratios below 0.9. A 90% claim requires both
   page-envelope and stripe-copy ratios >=0.9 in every overlap sample. Ratios use actual consumer
   softmax work, not the full fused kernel window. Independent empty-cache diagnostic sessions
   are not the formal multi-user LRU state; profile timing is not formal serving latency.
6. Build new report tables/figures from accepted raw data, with copied-artifact and derived-file
   hashes, renderer identity, model/input/heat/schedule/budget identities and both run IDs in
   `provenance.json`. At minimum retain summary JSON/CSV, full per-request CSV, coverage,
   cache/memory evidence, formal metadata/source/audit, and native profile metadata/analysis.
   One population is not a population sweep: use plots that show scheme and request behavior
   clearly without presenting an invented scaling axis. Inspect every rendered panel.
7. Independently recompute README table values, all/first/revisit sample counts, mean/p95/p99,
   miss/hit groups, byte-to-GiB conversions and capacities. Match copied files to accepted hashes,
   check the complete report file inventory and ensure all local links resolve. Ignored outputs
   are plain code paths; tracked `report/` assets are links. Disclose output boundaries: DeepSeek
   workload surrogate includes a last-token LM head, NOSA returns hidden only.
8. Publish the accepted new README/report/provenance and retire exactly the old 64K directories
   listed above in the same update. Recheck untouched 4K/16K provenance and fix references that
   would point at deleted old 64K evidence. Root then runs Supervisor to update research meaning
   and pending tasks, preserving researcher corrections and avoiding paper/meeting narrative.

## Still unverified

No accepted industrial latency, native profile, measured large-budget occupancy, new figure,
independent publication audit or Supervisor result exists from the interrupted attempt. The
new sequential workload also requires its own complete run and audit before any claims.
The latest steering prioritizes one small 64K group; neither this inventory nor a future 64K
acceptance proves the original three-geometry rerun complete. Nothing here validates production
traffic, recommendation quality, concurrency/queueing, general NOSA finite page slots, CXL/RDMA,
isolated activation peaks or cross-model task-equivalent latency.
