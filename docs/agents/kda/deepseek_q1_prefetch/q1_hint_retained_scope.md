# Exact Q1 hint: retained-result scope

The incremental CPU audit passed on 2026-10-08. It compares current production
with the accepted pre-hint source archive from
`q1_hint_model_profile_20261008_01`, rather than with Git HEAD. It does not
relabel any earlier measurement as a run of the new binary.

Evidence:

- Helper: `/tmp/cxldsagr-checks/q1_hint_retained_scope_20261008_02.py`, SHA-256
  `8edbb2d5b4c6444680bfbad5a64ca927794115b7e16d54722ab98b6ae6e72ff0`.
- Result: `/tmp/cxldsagr-checks/q1_hint_retained_scope_20261008_02.json`, SHA-256
  `7da75091c85a826ad3132557d2622f39eb02048320d07af26feb0a8553ce4a6e`.
- Accepted pre-hint model analysis:
  `experiments/deepseek_v32_echo_official/output/data/q1_hint_model_analysis_20261008_01/result.json`,
  SHA-256 `ee0246a98ca8c5bf9f5bf5757cffdb34864cd0f1bf7a729098db2630d54d834b`.

All 1,380 archived source files were rehashed against their recorded identities
and compared with live files. Of these, 1,378 remain byte-identical. The two
changed files are the hint dispatcher and the experiment's source/native
identity collector. The production adapter and CUDA source are new files.
Removing the exact `(1, 65537)` aligned-input branch and the changed function
docstring makes the complete dispatcher AST identical to the pre-hint AST.
Reference validation, generic mean, scorer, top-k, decode EMA, attention,
chunking, cache operations, resource declarations and capacity formulas are
unchanged within this incremental comparison.

`EchoAttentionRunner._forward` updates the hint from `scores[:, :end]` for the
original query batch before calling `_consume`. Exact-union overflow only
recurses inside `_consume` with slices of already selected IDs; it never
re-executes the scorer or hint. Consequently a Q1 attention leaf produced by
capacity splitting cannot activate the exact-mean branch.

| Retained scope | Evidence and decision |
| --- | --- |
| MFU A128+ matrix | Actual result files from all 12 bench/profile pairs bind H=4096/16384/65536, A=128/256/512/1024, history chunks of 1024, complete A-token extend and P=NH=H+A. No hint batch has Q=1. |
| Prior A128 MFU | The retained `deepseek_dense_late_wait` bench, stage profile and operator profile bind H=65536, A=128, C=1024 and extend chunk 128. Preserve their original identities. |
| Motivation matrix | All 12 accepted metadata files and 1,536 actual request rows bind the stated H/A values. Actual graph query sizes are `{1024, A}`. Candidate storage is transient; the unchanged backend executes candidates as one batch. |
| Prior motivation and DMA | The original rerun bench/profile and `deepseek_dma_c10_bench_20261006_01` metadata bind H=65536, A=128 and C=1024. The original single-point profile remains distinct from the newer matrix, and the DMA trace remains an original request-memory observation. |
| Cache manager | The accepted post-top-k profile uses Q=128. Its source fixture is unchanged; prefix construction uses direct append. The retained full-model transition profile also binds H=65536/A128/C1024. Independent recall and native append directly execute cache operations and never invoke hint. |
| Five capacity plans | The production CPU planner recomputed each complete plan, including selected, useful-P and next-infeasible ledgers. All 1,222 scalar comparisons match the original plans. Formula/resource sources are byte-identical to the pre-hint archive; the capacity entry is identical to the previously published static audit. |

All five plans still reserve Q=1024 at N=65664. The reservation remains
1,212,157,952 B for indexer/selection plus 2,359,296 B for two append sources,
or 1,214,517,248 B in total. The attention increment remains zero. Planning
still conservatively covers reachable Q1 tails; removal of a temporary mean
tensor does not reduce these reservations or establish any new user capacity.

The audit used CPU only and checked that CUDA remained uninitialized. Source
and input bytes were checked again after computation. Earlier official-Q1,
split16 and bounded64 changes keep their separate prior scope audits. This
audit establishes neither native payload equivalence nor a GPU rerun, graph
reserved reduction, physical memory peak, or capacity-fill trajectory.
Affected H65536/A1 local measurements are excluded and require the separate
production correctness, clean timing and profile cohort before publication.
