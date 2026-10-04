# NOSA native validation changes: affected experiments and rerun queue

> Historical GR scope (2026-10-05): the `gr_serving` experiment is retired and its
> experiment outputs have been removed. Its commands, result-retention rules and
> rerun instructions below are historical and no longer active. See the
> [retirement scope](experiment_organization.md#retired-gr-serving). Other
> implementation and experiment findings retain their stated scope.

The [selected-source fixed sequence](nosa_motivation_selected_source_commands.md)
completed on GPU5/CPUs48–55/NUMA1/register8: formal
`nosa_motivation_sm90_20261004_03`, matching profile
`nosa_motivation_profile_sm90_20261004_03` and independent APIs
`nosa_attention_reference_sm90_20261004_02` all passed independent audits at
runtime digest `a73ad32b0a3322cfb06c66121f50345d2ef6004b465733115fee6006a0f2f18a`.
No Q128 native candidate was selected. The accepted package is published and
exact superseded-output cleanup is complete; receipts belong to the
[fixed report](../../../experiments/nosa_motivation/README.md) and
[fixed publication record](nosa_motivation_publication.md).

GR's H4K/H16K/H64K formal/profile `_02` families completed on
GPU0/CPUs0–7/NUMA0/register8 at source `8dc00e...`, were independently audited,
published and cleaned using the exact old NOSA inventory. The
[GR execution receipt](nosa_gr_selected_source_execution.md) records completion.
The 444-request identity proof remains valid. The failed H4K `_01` attempt and
its lazy-host-flag reporting diagnosis remain external, without performance claims.

Candidate-only MFU, tails and compute/IO dominance remain open. Fixed async
failed its overall-latency comparison and all 96 applicable layer samples'
two 90% overlap gates. GR's nine layer-31 overlap samples also failed 90%.
These valid outcomes do not complete the requested efficiency goal.

This is an execution handoff, not a performance report. The prior formal
measurement `nosa_motivation_sm90_20261004_02` was accepted at its original digest
`6e3dfd17a86dd87be4ec89f0bfccc9bb25a52c5af773f5a5f2ed0753ba5c3e0c`.
The older checkpoints below retain their original identities and boundaries;
they do not describe the current run or override the final replacement receipt.

The prior integrated short-prefix/private-allocator source was
`02b5d0f9589c5e49257b40b73a5ddce4b2811f0614cc79d4afe242a2aa09f08b`.
It passed actual full 32-layer fixed-serving correctness on GPU5/NUMA1 with
register8: H65536/A128, two users over two visits, independent eager and four
graph backends, all hidden bitwise equal, real host misses, output lifetime and
owner cleanup. The process used the production `private_cpp` adapter without
fallback and retained its source/build/binary/allocator identities. These are
correctness checks at that earlier source. The later selected-source trio above
now supplies register8 measurement/profile/API evidence; it does not relabel this
earlier gate.

The original joint-native predicate remains `count >= 2047`. In measured `_02`,
short prefixes still used synchronous preparation. The current guarded short
path preserves the original standalone score schedule: all 31 affected Q1024
prefix lengths passed exact operator checks, followed by full 32-layer acceptance.
The earlier failed attempt that expanded joint-native scoring below 2047 is
rejected; it is distinct from this accepted dispatch-preserving implementation.
The earlier statements that no formal run existed and all short prefixes were
synchronous described the pre-`_02` queue state, not the current implementation.

The commands below mix earlier templates and preserved execution records.
Their presence does not establish execution; the updated dependency table and
linked receipts identify completed work. The offload and GR replacements are
published, as is the accepted fixed trio. Their exact superseded outputs have
been removed after publication checks. Any later
implementation change requires a fresh source freeze and affected correctness
and measurement gates.
See the [latest checkpoint](nosa_motivation_plan.md) for the separate source
identities and [allocator guard record](nosa_allocator_guard.md) for its parity
and engineering evidence.

## CPU and provenance readiness

At the earlier matrix-reference checkpoint, the complete
`experiments/nosa_motivation/tests` suite passed 83 tests after the
matrix-reference integration, graph-storage and cross-method-library tests.
`check_graph_replays` now requires graph policy/finite-validation metadata and
checks that graph static/reserved storage does not change within a request.
Profile and audit CLI checks and Ruff passed. These are CPU harness checks, not
new GPU numerical or performance results. This historical count is not the
latest integrated-source regression count.

`measure.main` finishes each method's complete prefix/candidate warmup before
constructing its measured backend. `run_case` snapshots native libraries after
that construction and compares the same set after all measured requests. A new
method may legitimately load additional libraries during its own warmup; the
final inventory is a superset of those method-specific snapshots. The snapshot
is not taken before the first warmup. Cached TVM-FFI modules retain their loaded
libraries when a backend closes. New libraries or changed file hashes during a
measured method are rejected. Profile subset runs compare their loaded subset
against the reference run's final library inventory.

The default warmup covers H=65,536 in 1,024-token chunks, A=128, first-user
construction, a second user replacing page tags, and a revisit requiring sparse
or dense recall. Compute graphs are prepared for both query sizes before the
warmup and again for the independent measured backend. Provenance collection
and hashing remain outside request latency. The accepted fixed run and its
independent provenance review checked the measured method-specific library
inventories. This remains a required check for any future source/shape change.

## Dependency scope and retained evidence

| Experiment | Directly affected work | Retained evidence and required replacement |
|---|---|---|
| `nosa_motivation` | Guarded short-prefix validation and production private allocator snapshot, on the existing fixed P/NH/graph/offload path | Selected source `a73ad32b...` passed the fresh full32 gate and accepted formal `_03`, profile `_03` and independent reference `_02` audits. Publication and exact superseded-output cleanup are complete; the [fixed publication record](nosa_motivation_publication.md) preserves receipts. Candidate MFU, tails, compute/IO dominance and failed async gates remain unresolved. |
| `nosa_kernel_mfu` | Native score/select compilation and complete resident indexer API | Completed and published `kernel_flags_synthetic_20261004_01`, `kernel_flags_real_20261004_01`, `modules_flags_native_20261004_01` and `modules_flags_triton_20261004_01` on GPU5/NUMA1. All 105 captured source fingerprints match across the four runs; 28 operator rows check all 1,024 queries, 12 module rows pass exact repetition/composition checks, and all 36 traces pass integrity checks. Root independently audited all 40 rows and 2,436 sample intervals. Replaced the previous report and removed only the three superseded measurement runs; preserved `kda_inputs_baseline_20260928_1345`. This remains resident A1024 evidence, separate from motivation A128/offload. |
| `indexer_block_sparse_profile` | Ordinary resident native indexer and public model forward | Completed and published `sparse_flags_native_20261004_01` and `sparse_flags_triton_20261004_01` on GPU5/NUMA1. The 109-source pair shares exact request bytes; independent wall timing and nsys/module capture passed raw SQLite recomputation plus 364 independent report checks. Both superseded runs were removed after publication. The checkpoint test default now selects the identical request copy in the new native run. No fixed-pool graph path is enabled. |
| `nosa_offload_overlap` | Current cached-fetch branches and FA3 API; current whole-model checkpoint path also uses changed indexer/finite checks | Completed and published `nosa_cached_fetch_20261004_01`, `nosa_cached_fetch_confirm40_20261004_01` and `nosa_cached_fetch_profile_20261004_01` on GPU5/NUMA1 at integrated source `02b5d0...`. The fresh ordinary full32 H65536/A1024 checkpoint gate passed with all normalized hidden bitwise equal. Independent audit passed 657 timing checks and all 27 measured profile ranges; every one of nine fused samples passed both 90% overlap thresholds with exact stripe identities, bytes and page envelopes. Current SASS/resource/read-once receipt and both reports are published. After acceptance, exactly the three superseded 20260930 runs' nine directories/294 files were removed. Frozen historical inputs are retained; this measures the default cold-union complete operator API, not finite P/NH hits or serving performance. This completed row supersedes the earlier unexecuted offload plan wording in this handoff. |
| `gr_serving` | Budget-mode shared FA3 workspace, offload wrappers and native score/select compilation | Published all three `gr_nosa_flags_h*k_20261004_02` formal runs and matching profiles at `8dc00e...` after independent audits. Exact old NOSA report/output cleanup completed; [receipt](nosa_gr_selected_source_execution.md). Sampled HBM storage, DRAM bins, reservation bounds and allocated/reserved remain separate; no whole-process physical HBM claim. Fixed P/NH results remain distinct. |
| `nosa_gr_65536_1024` | Public forward control flow; native sparse/finite/cached-fetch paths are not called | Completed and published `dense_wrapper_20261004_01` on GPU5/NUMA1 after SQLite/MFU recomputation and 80 independent report checks. Timing remains within an nsys process with capture disabled, not an independent unprofiled process. Its old run was removed only after all dependent pattern estimates were replaced and published. |
| `nosa_indexer_pattern_65536_1024` | Full-NOSA sidecar/sparse dispatch can use the changed native score/select kernels | Completed and published `nosa_pattern_flags_20261004_01`; 2,514 independent checks cover raw unions, 80 report assets, estimates, overlap windows and threshold optima. Six new CPU runs replace the QA64 environment comparison and dense-dependent time chain. Seven superseded pattern/analysis runs were removed. Preserved QA32, QA64, QA64 reanalysis, both decompositions and distribution with unchanged full-file hashes. The `triton` model label remains a dispatcher label; figures now say SM90 dispatcher. |

The resident operator/module, full-model, dense and pattern replacements above
were accepted and published on 2026-10-04. Their publication records list exact
source runs, audits and removed artifacts. Kernel publication removed three old
measurement runs; full-model publication removed two; pattern publication removed
seven old pattern/analysis runs and its superseded dense timing input. All current
raw captures and required unchanged inputs remain available. The separate offload
and GR replacements are also published; fixed measurement/profile/API acceptance,
publication and exact superseded-output cleanup are complete. Resident A1024,
budget-serving and fixed P/NH results retain their separate boundaries.

## Runnable replacement commands

Commands run from the repository root. GPU0 and `YYYYMMDD` in the older
templates below are historical planning values, not a current schedule. Completed
runs used the assignments and IDs in their receipts; use fresh IDs and a new
authorized assignment for any later reproduction. These templates do not reopen
the completed replacement queue.
The original APIs keep their documented 989 TFLOP/s denominator; motivation uses
its separately declared 989.5 TFLOP/s H200 value. Do not silently combine them.

First run the fixed P/NH replacement measurement and its matching three-sample
diagnostic replay. The template does not enable register8 automatically; record
and use the same explicitly selected allocator configuration in both processes.
Do not attach a new-source profile to the old `_02` measurement:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/nosa_motivation/scripts/run.sh \
  --run-id nosa_motivation_updated_YYYYMMDD_01 --device cuda:0 --compute-graphs
CUDA_VISIBLE_DEVICES=0 bash experiments/nosa_motivation/scripts/profile.sh \
  --run-id nosa_motivation_updated_profile_YYYYMMDD_01 \
  --reference-run experiments/nosa_motivation/output/data/nosa_motivation_updated_YYYYMMDD_01 \
  --device cuda:0 --repeats 3
```

The following four commands record the completed resident measurements; choose
new run IDs to reproduce them. They used GPU5, NUMA1 memory, CPUs48-55 and eight
OMP/MKL/OpenBLAS/PyTorch intra-op threads, as saved in each launch environment.
The operator comparisons keep the accepted frozen L0/L15/L31 input capture
to isolate implementation effects. A fresh trajectory capture is separately
needed if a report claims those inputs represent the current model trajectory.

```bash
CUDA_VISIBLE_DEVICES=5 bash experiments/nosa_kernel_mfu/scripts/run.sh \
  kernel_flags_synthetic_20261004_01 --device cuda:0 --peak-tflops 989 --reference-all
CUDA_VISIBLE_DEVICES=5 bash experiments/nosa_kernel_mfu/scripts/run.sh \
  kernel_flags_real_20261004_01 --device cuda:0 --peak-tflops 989 --reference-all \
  --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345
CUDA_VISIBLE_DEVICES=5 bash experiments/nosa_kernel_mfu/scripts/modules.sh \
  modules_flags_native_20261004_01 --device cuda:0 --kernel-backend native --peak-tflops 989 \
  --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345
CUDA_VISIBLE_DEVICES=5 bash experiments/nosa_kernel_mfu/scripts/modules.sh \
  modules_flags_triton_20261004_01 --device cuda:0 --kernel-backend triton --peak-tflops 989 \
  --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345
```

The completed resident full-model pair used unchanged request bytes. The request
copy in the new native run has the same SHA256 as the removed original source.
These are reproduction commands; choose fresh run IDs and retain the actual
GPU5/NUMA1/eight-thread settings recorded in their launch environments:

```bash
CUDA_VISIBLE_DEVICES=5 bash experiments/indexer_block_sparse_profile/scripts/run.sh \
  sparse_flags_native_20261004_01 --kernel-backend native --peak-tflops 989 \
  --request-file experiments/indexer_block_sparse_profile/output/data/sparse_flags_native_20261004_01/request.json
CUDA_VISIBLE_DEVICES=5 bash experiments/indexer_block_sparse_profile/scripts/run.sh \
  sparse_flags_triton_20261004_01 --kernel-backend triton --peak-tflops 989 \
  --request-file experiments/indexer_block_sparse_profile/output/data/sparse_flags_native_20261004_01/request.json
.venv/bin/python -m experiments.indexer_block_sparse_profile.src.compare \
  --native-data-dir experiments/indexer_block_sparse_profile/output/data/sparse_flags_native_20261004_01 \
  --triton-data-dir experiments/indexer_block_sparse_profile/output/data/sparse_flags_triton_20261004_01 \
  --output-dir experiments/indexer_block_sparse_profile/output/data/sparse_flags_native_20261004_01/comparison
```

The offload replacement needs a main timing run, independent confirmation and
internal-work profile; all three consume identical frozen inputs:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/nosa_offload_overlap/scripts/run.sh \
  nosa_cached_fetch_20261004_01 \
  --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345 \
  --layers 0 15 31 --queries 1024 --tile-size 128 --fetch-ctas 96 \
  --warmup 5 --repeats 20 --reference-all
CUDA_VISIBLE_DEVICES=0 bash experiments/nosa_offload_overlap/scripts/run.sh \
  nosa_cached_fetch_confirm40_20261004_01 \
  --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345 \
  --layers 0 15 31 --queries 1024 --tile-size 128 --fetch-ctas 96 \
  --warmup 5 --repeats 40 --reference-all
CUDA_VISIBLE_DEVICES=0 bash experiments/nosa_offload_overlap/scripts/run.sh \
  nosa_cached_fetch_profile_20261004_01 --profile \
  --input-dir experiments/nosa_kernel_mfu/output/data/kda_inputs_baseline_20260928_1345 \
  --layers 0 15 31 --queries 1024 --tile-size 128 --fetch-ctas 96 \
  --warmup 5 --repeats 3 --reference-all
```

For the retained budget-mode request configurations, explicitly select NOSA and
the original cap; do not pass the removed `--deepseek-slots`. Check generated
request hashes against the old trace before treating the new numbers as an exact
trace replacement. The current uncapped IID trace is a different experiment.

```bash
CUDA_VISIBLE_DEVICES=0 GR_AUDIT_BEFORE_PUBLISH=1 bash experiments/gr_serving/scripts/run.sh \
  gr_nosa_flags_h4k_20261004_01 --models nosa --users 1 8 32 64 128 256 512 \
  --history-tokens 4096 --candidate-tokens 128 --requests 32 --max-revisits 8 \
  --hbm-budget-gib 4 --dram-budget-gib 16
CUDA_VISIBLE_DEVICES=0 GR_AUDIT_BEFORE_PUBLISH=1 bash experiments/gr_serving/scripts/run.sh \
  gr_nosa_flags_h16k_20261004_01 --models nosa --users 1 8 32 64 128 256 512 \
  --history-tokens 16384 --candidate-tokens 128 --requests 32 --max-revisits 8 \
  --hbm-budget-gib 4 --dram-budget-gib 16
CUDA_VISIBLE_DEVICES=0 GR_AUDIT_BEFORE_PUBLISH=1 bash experiments/gr_serving/scripts/run.sh \
  gr_nosa_flags_h64k_20261004_01 --models nosa --users 1 8 32 64 128 256 512 \
  --history-tokens 65536 --candidate-tokens 128 --requests 6 --max-revisits 8 \
  --allow-empty-revisits --hbm-budget-gib 4 --dram-budget-gib 16
CUDA_VISIBLE_DEVICES=0 bash experiments/gr_serving/scripts/profile.sh \
  gr_nosa_flags_h4k_profile_20261004_01 \
  --latency-data experiments/gr_serving/output/data/gr_nosa_flags_h4k_20261004_01 --num-users 8
CUDA_VISIBLE_DEVICES=0 bash experiments/gr_serving/scripts/profile.sh \
  gr_nosa_flags_h16k_profile_20261004_01 \
  --latency-data experiments/gr_serving/output/data/gr_nosa_flags_h16k_20261004_01 --num-users 8
CUDA_VISIBLE_DEVICES=0 bash experiments/gr_serving/scripts/profile.sh \
  gr_nosa_flags_h64k_profile_20261004_01 \
  --latency-data experiments/gr_serving/output/data/gr_nosa_flags_h64k_20261004_01 --num-users 1
```

The dense and full-NOSA recapture commands below are completed GPU5 records.
For a new run, choose fresh IDs; both used CPUs48-55, NUMA1 and eight threads.
The retained new dense request is byte-identical to the original request source:

```bash
CUDA_VISIBLE_DEVICES=5 bash experiments/nosa_gr_65536_1024/scripts/run.sh dense_wrapper_20261004_01
CUDA_VISIBLE_DEVICES=5 CXLDSAGR_SM90_BACKEND=native \
  bash experiments/nosa_indexer_pattern_65536_1024/scripts/sparse_compare.sh \
  nosa_pattern_flags_20261004_01 \
  --request-file experiments/nosa_gr_65536_1024/output/data/dense_wrapper_20261004_01/request.json
```

The dependent CPU replacements are `qa64_environment_compare_flags_20261004_01`,
`estimate32_denseflags_20261004_01`, `estimate64_denseflags_20261004_01`,
`overlap32_64_full_denseflags_20261004_01`, `overlap32_64_half_denseflags_20261004_01`
and `threshold32_64_half_denseflags_20261004_01`. Their new dense timing source is
`dense_wrapper_20261004_01`; the retained QA inputs and offline assumptions are
unchanged. All 22,792 recomputed fields and 74 source/input checks passed.
Reproduction commands and source mappings are in the pattern experiment README.

Report renderers should write to a new temporary destination until all checks
pass. Use each README's existing report/rebuild entry point with the new IDs;
do not overwrite the retained report directories before acceptance.

## Scheduling estimates

These are deliberately broad planning allowances, not observed runtimes or
performance claims. They assume one idle H200/SM90, prepared dependencies,
locally available checkpoint/input files and successful validation. Cold native
compilation, large profiler exports, a failed gate or corrective development can
exceed them. They do not authorize dropping samples to meet a deadline.

| Work | Single-GPU planning allowance |
|---|---|
| New fixed P/NH four-method measurement | 15–40 minutes |
| Matching first/revisit profile, three samples, four methods | 30–90 minutes |
| Synthetic/real operator and native/Triton module comparisons | 15–45 minutes total |
| Resident full-model native/Triton benchmark + nsys pair | 20–60 minutes total |
| Offload timing + confirmation + profile | 10–30 minutes total |
| Three retained budget-mode traces and their profiles | 60–180 minutes total |
| Dense baseline and full-NOSA pattern recapture | 15–45 minutes total |

The old budget experiment documents 31.983 and 32.797 minutes for two mixed
DeepSeek/NOSA processes, including loading, warmup and reporting. Those are not
NOSA-only costs and are not used as estimates for the current implementation.
Independent experiment jobs may use separate idle GPUs after freeze; a latency
run and its diagnostic profile must not overlap on the same GPU.
