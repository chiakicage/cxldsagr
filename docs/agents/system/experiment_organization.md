# Experiment organization and independent numerical acceptance

User objective: organize the experiments by the agreed four purposes, then
commit the result. Do not run new experiments. This is implementation and
migration work, not a new performance result. Existing unrelated worktree edits
must be preserved.

## Required end state

1. The experiment index and each retained entry identify motivation, baseline
   implementation efficiency, sparse-pattern exploration, or design microbench.
2. Retire the old capped-popularity `gr_serving` experiment as a complete scope;
   first extract its actively reused workload, provenance and memory-audit tools.
   Preserve the generic serving/model APIs and necessary correctness tests.
3. Merge dense and native/Triton resident performance into
   `experiments/nosa_baseline_performance`, with explicit independent scopes.
   Preserve valid reports, run IDs, raw data, snapshots and measurement meanings.
   Record byte-preserving artifact migration; do not rewrite historical hashes.
4. Keep actual sparse-pattern observations as design exploration. Remove only
   the discontinued idealized dense-MFU/bandwidth threshold/overlap branch;
   preserve QA observations and decomposition/distribution used for exploration.
5. Keep operator implementation-efficiency checks and offload microbench useful
   for their declared purposes; no claim that an operator MFU proves serving
   efficiency or that A1024 overlap proves A128 serving benefit.
6. Separate numerical check, clean bench and intrusive profile. Repeated bench
   samples must not run external reference/equality checks, save every full
   output or automatically launch profiler. Independent successful checks are
   bound to actual source/native/input/configuration and reused by compatible
   bench runs. Runtime finite/repair/allocator/transaction/synchronization remain.
7. Preserve historical report readers and evidence. New harness paths are
   explicitly unmeasured. A migrated old receipt does not certify changed code.
8. Update active imports, module commands, test collection, source inventories,
   report links and navigation. Historical snapshot contents remain immutable.
9. Verify via CPU regressions, parser/help checks, migration inventories and
   dependency/link audits only. No GPU tests, measurement, profile, synthetic
   timing, new analysis run or numerical experiment is authorized.
10. Commit a coherent implementation with its required dependencies and report
    navigation, keeping unrelated work out of the commit and all ignored raw
    experiment outputs out of Git.

## Work ownership

- Root: shared `evaluation/` tools, `GR/workload.py`, GR retirement, external
  imports, global docs/tests, integration and completion audit.
- `organize_motivation`: NOSA and DeepSeek motivation check/bench/report/profile.
- `organize_microbench`: resident operator/modules and offload check/bench/profile.
- `organize_baseline_pattern`: resident merge and pattern scope cleanup.

## Implementation and verification

The four-purpose index is in `experiments/README.md`; each retained experiment
states its scope and historical measurement boundary. `GR/workload.py` and
`evaluation/` hold the extracted shared code. The baseline merge and pattern
scope reduction are complete. Numerical acceptance, ordinary measurement and
profiling are separate in both motivation experiments, the resident baseline,
operator/module efficiency and offload microbenchmark.

Check receipts bind sources, input/configuration, backend/native artifacts and
relevant precision/runtime settings. They are published only after successful
final checks. Dense resident acceptance now covers every candidate row, rather
than only the final token. Clean bench branches do not perform full reference
comparisons or store per-request output tensors. Runtime model/operator checks
were not disabled by this organization work. New GPU harness execution remains
unverified and unmeasured, explicitly recorded in each README.

Verification on 2026-10-05:

- `bash scripts/run_tests.sh cpu`: 3195 passed, 1273 skipped, 58 subtests passed.
  CUDA was hidden; skips include GPU/checkpoint cases and one local tokenizer
  dependency. The single warning is PyTorch's CPU profiler cycle notice.
  This is not GPU acceptance. CPU collection now also includes DeepSeek
  motivation and official-adapter tests, with importlib mode for duplicate
  test basenames.
- Migration inventory: 489 entries, including 455 preserved report/raw/snapshot
  files totaling 142,679,265 bytes. Every preserved file's size and SHA-256
  match the pre-move inventory. The original run IDs are unchanged.
- An additional 154 publication-manifest file checks passed across the
  motivation, kernel/module and offload report/raw artifacts. Updated report
  readers reopened both historical motivation runs on CPU: each checked all
  128 saved requests and 96 offload comparisons, preserving the old schema.
  These were read-only checks of saved evidence, not new model executions.
- All 59 retained pattern-report hashes match the scope inventory, and every
  path declared retired by that inventory is absent.
- Shared and owned Python lint, shell syntax and CLI help checks passed.
  516 local README/navigation links resolved. Live Python/shell references to
  removed modules were migrated; remaining old names are historical snapshot
  keys, historical fixtures or a profiler output basename.
- `git diff --check` passes outside immutable report assets. Existing generated
  SVG whitespace and CSV CRLF in pre-task report updates are preserved.

The migration inventory is at
`experiments/nosa_baseline_performance/output/data/layout_migration_20261005_01/migration.json`.
The pattern scope inventory is versioned at
`experiments/nosa_indexer_pattern_65536_1024/report/scope_manifest.json`.
No new performance data, figures or GPU validation runs were produced.

## Commit scope

The preexisting worktree had not yet committed the NOSA fixed-capacity/shared
serving implementation, allocator and pool adapters, deferred finite checks,
cached-fetch support or their accepted reports. The organized entrypoints
depend on these APIs, and the reviewed resident source graph requires their
exact files. They and their corresponding tests are therefore included as
prerequisites, alongside the experiment organization and linked acceptance
documentation. This does not relabel their historical GPU results as tests of
the new harness.

The unrelated, unpromoted external prototype note
`docs/agents/system/nosa_indexer_output_reuse_result.md` remains uncommitted.
Concurrent follow-up changes to its planning notes also remain outside this
commit.
Raw `output/` data, local weights and environment files remain ignored. The
final CPU regression covered the staged implementation. The commit hook later
sorted one DeepSeek official-harness import and reformatted two GPU-test
decorators. The GPU-test AST is unchanged, and all 107 official-adapter CPU
tests passed after the import change. No model/operator runtime code changed
after the full regression.

## Retired GR serving

The user authorized the four-purpose cleanup and no new experiments on
2026-10-05. The capped-popularity `gr_serving` experiment is retired as a whole,
including all four methods and H4K/H16K/H64K outputs. This is a scope decision,
not a claim that an unfavorable result was numerically invalid or replaced by
fixed P/NH measurements. The short traces did not establish user-population
scaling or a physical whole-process 4-GiB capacity limit. Their local findings
included 13 H16K HBM revisit misses and none for offload, HBM lowest all-request
mean in 21 groups, and nine overlap samples below the stated 90% gates.

The last published formal IDs were `gr_nosa_poolscan_h4k_20261004_01`,
`gr_nosa_poolscan_h16k_20261004_01`, and `gr_nosa_poolscan_h64k_20261004_01`,
with corresponding `*_profile_20261004_01` diagnostics. Their performance
outputs are removed rather than kept as a competing legacy experiment.
Historical engineering documents may still name these runs; this note is a
retirement record, not a replacement evidence archive or new performance report.

Reusable requests move to `GR/workload.py`; source/numerical evidence helpers,
pool-provider provenance, and the still-used memory audit move to `evaluation/`.
Generic byte-budget serving remains implemented in `models/nosa/serving.py` and
`serving/persistent.py`, with its required regression coverage. Retirement of
this experiment does not remove those model/cache capabilities.
