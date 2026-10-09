# Bounded-free preparation integration scope

This record describes the earlier bounded-free integration. Its formal A1
publication was superseded after stateless preparation fusion; the current
cohort is `deepseek_h64k_a1_fused_prepare_20261009_01`. See `checkpoint.md` for
current selected reports and the signed input that remains retained.

Prior publication: `deepseek_h64k_a1_isolated_20261008_01` passed all 16
independent method/role processes and the cohort publication audit on GPU 0 /
CPUs 0–7. Local timeline, startup, complete-stage MFU and operator MFU assets are
consolidated in `experiments/deepseek_v32_mfu/report/h64k_a1/`; the separate
`report/h64k_a1_mfu/` tree is superseded. Each child prepares only its selected
method, and offload checks read accepted HBM outputs on CPU. This changes the
measurement boundary, not the production computation. New H64K/A1 publication
uses `scripts/method_isolation.sh` and `src.method_isolation_report` as described
below. Official comparison/gap reports retain their separate source identities.

Historical integration status: root applied the reviewed bounded-preparation
patch after the private complete-model check, timing and profile gates passed.
The first production refresh was `deepseek_h64k_a1_free_20261008_01`, with
operator profile `deepseek_h64k_a1_free_mfu_profile_20261008_01`. Its audit and
original cleanup records below describe that publication, not the current
selected reports. Root owns production/cache changes; the formal observer and
transition/comparison validators remain part of the MFU workflow.
The seven integration mirror files remain frozen under
`experiments/deepseek_v32_mfu/output/data/q1_free_prepare_integration_20261008_02/source/`.
Before application, `integration.patch` passed `git apply --check`; both frozen
manifests matched their files. The component candidate passed its independent check, 100
AB/BA pairs per state, NSYS and cold/scattered full/source NCU. Integration has
its own native identity because its grid arithmetic and reset-loop indices use
int64. The private model check validates that entry independently of the original
component receipt.

## Dispatch and ownership

The native provider reuses the exact official Q1 eligibility predicate. Cache
dispatch additionally requires one active session, the current exclusive pool
operation, ordinary nontransient append, and `P-H>=64`. Provider eligibility
requires one query, actual indexer-visible columns equal to `query_start+1`,
`N>=32768`, initialized history equal to query start, and at least 64 permitted
prefetch slots. Then `L<=H` live records implies at least `P-H>=64` empty slots.
This is capacity evidence, not residency certification.

The candidate prepares exactly 64 ascending free slots, publishes MISSING in the
unused slot-list tail, resets the complete journal/counter/stats, and retains
request metadata. It checks all priority ages and selected bitmap/reverse-map
owners. Candidate runtime failures propagate. Other support conditions retain
normal full preparation. The lease and distinct `_BoundedPreparedPrefetch`
token expose the prepared bound. Official Q1 consumes 64 explicitly; the
unchanged generic consumer rejects this distinct token through its existing
exact-type guard. The original full token and its positional consume API are
unchanged. Existing finalization and rollback invalidate either token.

## Dependency audit

Both NOSA rule files and the DeepSeek rules were read. Runtime Python searches
found no SparseTokenCache/SharedSparseTokenPool use in `models/nosa/`,
`operators/nosa/`, or NOSA experiment source. Their historical broad provenance
manifests mention the shared file, but that does not establish an executing
dependency. No NOSA computation, allocator, request path or persistent storage
changes are proposed.

| Experiment/path | Effect of current mirrors | Required publication treatment |
| --- | --- | --- |
| DeepSeek MFU, ECHO H64K/A1 | New preparation algorithm in the complete extend graph | Private full-model correctness and A/B gate, then formal independent check/bench/profile and source/native audit before replacing results |
| Local official-comparison figures | Local ECHO measurements depend on MFU | Regenerate affected local data/figures only after formal publication; official raw SGLang/ECHO artifacts retain their own unchanged provenance |
| DeepSeek MFU HBM, serial sparse, dense prefetch | They do not call fused prefetch preparation | Preserve only after source/path scope audit; do not attribute the component gain to these methods |
| Cache-manager post-top-k interval, H64K/A128 | Full sort/token remain; preparation and token consumption occur before the declared interval | Check exact source/native coverage of the interval and retain valid unaffected measurements; targeted runner correctness checks the normal dispatch |
| Cache-manager native append/recall components | No changed algorithm or scratch layout | Source/native impact audit; the new helper is not called by these components |
| DeepSeek motivation, transient A128/256/512/1024 | New free path is ineligible; Q!=1 short-circuits added eligibility checks, and the original full token/consumer remain unchanged | Scope audit and targeted correctness; do not attribute bounded Q1 timing to these requests |
| Cache management static plans and dense DMA memory observations | No persistent allocation/estimator change; existing scratch reused; dense does not consume this preparation | Recompute/compare affected static plans if publishing a new source identity; retain unchanged dense observations with a scope audit |

Root selected the narrow Q1-only token design before the full-model gate. The
new `_02` mirror leaves `_PreparedPrefetch`, `echo.logits` and
`_validate_prefetch` source segments exactly equal to production. The private
binder also leaves those objects and the original cache_ops aliases/native call
function untouched; only the new free-prepare function receives a private native
namespace. The added storage/stream/bound validation executes only for the
distinct bounded token. `_01` was superseded before any GPU gate or performance
measurement of the integration mirror.

No cache budget reduction is claimed. Removing one CUB workspace allocation
does not prove a lower device-used or allocator-reserved peak, and the existing
conservative execution reservation is unchanged. Full-model process snapshots
with both models loaded are not per-arm memory comparisons.

## Evidence

- Component check receipt:
  `/tmp/cxldsagr-checks/q1-free-prepare/q1_free_prepare_check_20261008_02/receipt.json`.
- Component benchmark and complete artifact audit:
  `experiments/deepseek_v32_mfu/output/data/q1_free_prepare_bench_20261008_01/`.
  Audit verifies seven phase archives, each containing 2,735 source files, exact
  input/native identities, all 800 balanced pairs and profiler kernel coverage.
- The first component check invocation failed before producing a check directory
  because PATH omitted `.venv/bin/ninja`. The corrected invocation changed PATH only and
  passed. Failed logs remain only in the system temporary check workspace;
  no failed experiment result was published.
- Frozen mirror CPU checks: 28 passed, 50 deselected. These cover the token and
  provider predicate and do not replace full-model or new-native GPU checks.
- Independent reviewer confirmed the component evidence, actual baseline DSO,
  mirror alias binding/restoration, one-use bounds and exception preservation.
- The separate model check `q1_free_prepare_model_check_20261008_01` completed
  its compute, diagnostic proof and clean graph checks, but cleanup failed because
  it ran outside `torch.inference_mode()`. It issued no accepted receipt. Root
  fixed cleanup in the private harness; this failure is unrelated to the
  superseded integration mirror `_01` and the earlier component check `_01`.
- Model check `q1_free_prepare_model_check_20261008_02` passed after that fix.
  Receipt:
  `/tmp/cxldsagr-checks/q1-free-prepare-model/q1_free_prepare_model_check_20261008_02/data/q1_free_prepare_model_check_20261008_02/receipt.json`.
  Its receipt digest is
  `d239de3b5755aa6e4cc742bfe90a45c66eb9f20f2000fe927eb797a1cc3ca5e6`.
  Independent CPU verification checked the receipt signature, all 1,418 artifact
  hashes and equality of the recursive file inventory. This verifies stored
  evidence integrity; it does not repeat the GPU computation.
- The accepted independent audit is
  `/tmp/cxldsagr-checks/q1_free_prepare_model_check_20261008_02_independent_audit.json`,
  produced by the adjacent `.py` helper. It reread 38 saved tensors and passed
  37 bitwise comparisons, reconstructed all 12 compact eager/graph stage proofs,
  and checked six cache comparisons. It matched 1,383 current/archived source
  files, the six baseline and seven candidate mirror files, seven mapped native
  libraries, input/checkpoint metadata and 8,370 concrete runtime files. CUDA
  remained uninitialized. Original scores and KV payloads omitted from compact
  evidence were checked during GPU acceptance and are not reconstructed by this
  CPU audit. The same limit applies to clean graph map/free/resident-record
  checks whose arrays were not saved. In-memory CuTe bytecode and compiler
  fingerprints without retained paths remain source-bound runtime evidence.

## Private full-model gate

The entry is `experiments/deepseek_v32_mfu/src/q1_free_prepare_model.py`, invoked
as `python -B -m experiments.deepseek_v32_mfu.src.q1_free_prepare_model` with
separate `--mode check`, `--mode bench` and `--mode profile` runs. Bench and
profile require the accepted check's `--receipt`. The process-local binder is
`q1_free_prepare_binding.py`; `q1_free_prepare_validation.py` supplies the
independent bounded preparation proof adapter. `-B` avoids adding bytecode files
to the frozen mirror inventory.

All three modes bind the request, checkpoint, sources, mirror manifests, GPU,
runtime kernels and loaded native artifacts. The model scope is two independent
ECHO instances of the real checkpoint's first three layers, cold H=65,536 and
A=1, P=NH=65,600, ordinary persistent append, `return_hidden=True`, chunk and
query workspace 1,024. This is the complete declared three-layer model path,
not a 61-layer measurement.

1. **Check, completed:** exercise the new integration CUDA entry on all eight
   saved/synthetic component states; compare eager and diagnostic captured
   execution for token IDs 111090, 111091 and 111092 in each arm. Actual-stage
   CPU proofs retain the truthful preparation capacities, 8,192 for baseline and
   64 for candidate, with official effective prefetch cap 64 in both. Scores,
   exact top-k, hints and outputs agree; scheduling-dependent residency,
   priority and traffic differences require each execution's legal stage proof.
   After destroying the diagnostic graphs, check the actual clean timing graphs
   against eager outputs and cache invariants for all three inputs, and compare
   clean outputs across arms. The clean graphs have no observer or timing event
   nodes; their actual-stage oracle comes from the separate diagnostic graphs.
   Save outputs and all proof files, finish cleanup successfully, then issue the
   source/native/input-bound receipt over the complete file inventory.
2. **Bench, completed:** `q1_free_prepare_model_bench_20261008_01` ran on
   GPU 0 / CPUs 0–7 after the independent audit passed. Complete-model wall
   medians were 2.7214065 ms for baseline and 2.641556 ms for candidate. The run
   uses 100 alternating AB/BA pairs after five warmups
   per arm. Every sample restores the same cold prefix and synchronizes before
   measurement. Prefix restoration and binding changes stay outside the timer;
   the timer surrounds complete `model.forward` plus CUDA synchronization. Both
   arms retain independent clean graphs and buffers. Inspect all 200 samples,
   the balanced order, paired differences and source/native identity before
   interpreting the medians. Component latency gains do not establish a model
   speedup.
3. **Profile, completed:** `q1_free_prepare_model_profile_20261008_01` records
   one pair under the same accepted identity. Its independent source audit is
   `/tmp/cxldsagr-checks/q1_free_prepare_model_profile_20261008_01_source_audit.json`;
   it froze source/native/input identities before production edits. The adjacent
   `_trace_audit.json` verifies the prepare episodes and 260 unchanged non-prepare
   GPU node signatures per arm. It attributes 51.008 us less summed prepare GPU
   activity; the intrusive graph span difference is not the clean wall gain.
   The profile saved neither outputs nor per-sample transport counters; it relies
   on the matched independent check and does not establish a traffic split.

Root selected promotion after the complete-model gate; the affected-path
acceptance and first formal refresh were completed under the FREE identities
below. Later changes require matching acceptance, independent timing/profile,
measurement audit and new run IDs. Process memory snapshots with
both models loaded must not be presented as per-arm cache capacity differences.

## Current formal entrypoints

These are current entrypoints, not commands executed during the historical
private gate.
Use new run IDs and fresh output/report destinations throughout. Freeze the
promoted source before check, bench and profile; the private receipt cannot
certify a changed production execution identity.

The formal observer now derives a `bounded-free-q1-v1` descriptor from the actual
returned bounded token, lease, buffers and exclusive pool owner. It stores each
eager/captured execution separately and verifies consumption by official Q1.
`prefetch_transition_audit.py`, `extend_graph_validation.py` and
the method-isolation report validate the descriptor and its actual preparation
cap64 through compact evidence, each layer, common scope and receipts.
`audit_shape_matrix.py` retains this validation for its existing matrix path.
Existing full proofs without the descriptor retain `min(8192, P-A)`; cap64 alone does not select
the new proof. At bounded-preparation integration, the four focused CPU test
modules passed 170 tests and Ruff passed.
Independent reviews are
`/tmp/cxldsagr-checks/bounded_formal_validation_independent_review_20261008_01.json`
and `/tmp/cxldsagr-checks/bounded_matrix_gate_independent_review_20261008_01.json`.
They pass 33 validator checks and three matrix-gate checks, including exact
reconstruction of archived full proofs and rejection of cap64 evidence without
the descriptor or outer scopes inconsistent with compact evidence.

At that integration, the production native eight-state check, 136 existing and
three targeted GPU checks passed; ten cache dispatch cases were included in CPU
coverage. Global CPU regression passed 4,884 tests and 58 subtests, with 1,548
GPU/optional cases skipped. Those skips are not GPU acceptance. The current
isolated cohort has its own 13 standard comparisons, 44 complete-graph checks,
six ECHO stage proofs and independent minimal/operator output rereads.

The private AST proof adapter and model harness are retained only for their
immutable archived acceptance/timing/profile identities. They must use their
archived validator sources for historical reproduction; they do not provide an
adapter around the new formal validators or authorize new production runs.

| Purpose | Existing entry and required arguments | Publication / retention scope |
| --- | --- | --- |
| Production cache/provider correctness | `.venv/bin/pytest cache/tests/test_sparse_token_pool.py operators/deepseek_v32/indexer/tests/test_echo_indexer.py operators/deepseek_v32/indexer/tests/test_official_prefetch.py`; include the promoted bounded-token cases | Check leases, normal full preparation, cleanup and the actual new native entry; unit checks do not replace formal model measurements |
| Formal H64K/A1 cold check, bench, minimal profile and operator profile | With a fresh `MFU_RUN_ID`, run `bash experiments/deepseek_v32_mfu/scripts/method_isolation.sh --model /preset-models --request experiments/deepseek_v32_mfu/output/data/deepseek_h64k_a1_isolated_20261008_01/request.json`; the script fixes GPU0/CPU0–7 and the documented H64K/A1 configuration | Produces 16 fresh children, `<cohort>_<method>_{check,bench,profile,operators}`. Checks live under `/tmp/cxldsagr-checks/deepseek_v32_mfu/data/`; each method has its own receipt and actual runtime/native identity. HBM reference outputs enter offload checks only on CPU |
| Cohort integrity and consolidated report | `CUDA_VISIBLE_DEVICES= .venv/bin/python -m experiments.deepseek_v32_mfu.src.method_isolation_report --manifest <new-cohort>/manifest.json --output-dir <new-report-run>` | Audit all child receipts, command/process identities, saved outputs, source/native files, traffic, original graph/window ownership and FLOPs before publishing the reviewed assets to `report/h64k_a1/` |
| H64K/A1 warm correctness | `bash experiments/deepseek_v32_mfu/scripts/run.sh --mode check --prefix 65536 --extend 1 --chunk-size 1024 --extend-chunk-size 1 --sparse-pool-tokens 65600 --host-arena-tokens 65600 --compute-graphs --extend-graph --extend-residency warm`, with explicit model/device and a new `MFU_RUN_ID` | Separate warm receipt; it does not authorize cold timing or establish a new warm latency result |
| Per-operator MFU | `method_isolation.sh` invokes `scripts/profile_layers.sh` separately for each selected method, bound to its own check and benchmark; `src.method_isolation_report` joins the results | `final_mfu.csv`, `operator_mfu.csv`, `operator_mfu.svg` and graph audits now live in `report/h64k_a1/`. Do not synthesize a common profile or use a schema-3 four-method receipt for isolated children |
| Official/local comparison figure | `python -m experiments.deepseek_v32_echo_official.src.compare_decode_timeline --official-report experiments/deepseek_v32_echo_official/report/sglang_decode --mfu-report <new-mfu-report> --output-dir <new-comparison-run> --publish-dir <new-comparison-report>` | Replaces `report/combined_decode/` after local publication; official SGLang/ECHO raw runs and numerical/residency limitations stay unchanged |
| Updated gap attribution | `python -m experiments.deepseek_v32_echo_official.src.diagnose_decode_gap --comparison <new-comparison-run> --official-report experiments/deepseek_v32_echo_official/report/sglang_decode --local-report <new-mfu-report> --output-dir <new-gap-run>` | Replaces affected `report/decode_gap/` selected data after raw activity/source audit; profile windows remain distinct from clean wall timing |
| Normal A128 cache-manager dispatch | `bash experiments/cache_manager_performance/scripts/run.sh --mode check --run-id <new-check-id> --capture-dir experiments/cache_manager_performance/output/data/input_fixture_20261006_02 --warmup 2` | The bounded path is ineligible. Preserve post-top-k/append/recall results after source/native interval audit; if an executing measured component changed, use `scripts/profile.sh` with the new receipt and `src.report` instead of retaining its old numbers |
| Capacity plans | `python -m experiments.cache_management.src.capacity_plan` with each published plan's exact fixed-P/fixed-NH, supplied HBM/model-memory, DRAM and headroom arguments, under new run IDs; compare complete plans and source identities | Refresh the static impact/publication record only if needed for the promoted identity; no request-memory or capacity-trace claim follows from this CPU calculation |

The A128–A1024 MFU and C10 motivation paths keep their existing full token and
consumer. Audit their executing source paths and retained native payloads;
do not rerun unaffected experiments solely because broad manifests list the
shared file. If this audit finds an actual dependency change, use the affected
shape through `experiments/deepseek_v32_mfu/scripts/shape_matrix.sh` and/or
`experiments/deepseek_v32_motivation/scripts/run.sh` / `scripts/profile.sh`, with
fresh matching check and performance runs. NOSA runtime paths have no dependency
on the changed cache class. Existing isolated attention, packing, promotion and
top-k component measurements remain separate from the new preparation/model
claim and retain their valid provenance after a source-scope check.

## Publication cleanup boundaries

Keep current README figures, reports and successful raw runs until their
replacement passes correctness and measurement-integrity review. For the
isolated publication, the replaced HINT material is MFU `report/h64k_a1/`,
the former `report/h64k_a1_mfu/`, official/local
`report/combined_decode/`, and any local gap values in `report/decode_gap/` and the
two experiment READMEs. Use their manifests to enumerate replaced files and raw
run dependencies; do not delete by a broad `q1_*` pattern.

Current source IDs are `deepseek_h64k_a1_isolated_20261008_01` and its
`<method>_{check,bench,profile,operators}` children. The consolidated report run
is `deepseek_h64k_a1_isolated_report_20261008_01`; comparison and gap runs are
`combined_decode_isolated_20261008_01` and `decode_gap_isolated_20261008_01`.
Only superseded, unreferenced HINT `output/{data,log,profile}/<run_id>` directories
belong to this cleanup. Original CUB/FREE check, benchmark and profile artifacts
used by active controls remain at their signed source/request/native paths.
Preserve independent official decode/Engine runs, unaffected A128–A1024 results,
valid component evidence and any explicitly retained correctness/input fixture.
If an old implementation is still required as a performance comparison, create
a new accepted measurement rather than retaining superseded result numbers.

The historical bounded-preparation private model gate binds its request to
`experiments/deepseek_v32_mfu/output/data/deepseek_h64k_a1_cub_20261008_01_h65536_a1_profile/request.json`.
Its component fixture is
`/tmp/cxldsagr-checks/deepseek_v32_mfu/data/deepseek_h64k_a1_cub_20261008_01_h65536_a1_check/echo_default_graph_prefetch_evidence.pt`.
These remain dependencies of retained signed check/bench/profile evidence;
their original paths must not be rewritten. New isolated and fused-preparation
work uses the retained isolated cohort's `request.json`, whose SHA256 is
`d776856368e64c931805f2181c41417609d55082866cc71e7468dfb6d57b380f`.
The isolated manifest's old HINT `source_path` and invocation record its
original input provenance; its own retained request has identical bytes.
No extra request copy is needed for HINT cleanup. Do not edit signed paths,
manifests or historical commands to conceal an original dependency.

## Superseded mirror cleanup audit

`q1_free_prepare_integration_20261008_01` contained ten files, 159,991 bytes, and
no symlinks. It was superseded before any integration GPU gate or measurement.
Before adding this cleanup note, `rg --no-ignore` found no reference to that
exact directory in current runtime source, relevant experiment output/source,
KDA documents, or component/model check artifacts. The current binder selects
`_02`, whose seven source files
match the frozen manifest; its runtime and signed evidence do not depend on
`_01`.

Two temporary generation scripts referenced `_01`:
`/tmp/build_free_prepare_mirrors.py` creates it, and
`/tmp/build_narrow_free_prepare.py` reads it while generating `_02`. They are
generation-time dependencies, not runtime inputs. After root confirmed that no
regeneration remained, the obsolete directory and both scripts were removed.
The 12 file hashes and 178,510-byte inventory were recorded before deletion in
`/tmp/cxldsagr-checks/q1_free_prepare_obsolete_mirror_cleanup_20261008_01.json`.
Accepted component/model evidence and `_02` remain intact. Signed source
identities and historical commands were not rewritten.

## Historical FREE publication and incremental retention

This section records the earlier `deepseek_h64k_a1_free_20261008_01`
publication and its original audit/cleanup paths. It does not describe the
current selected report or authorize deleting dependencies retained by later
controls. Its four-method formal clean medians were HBM 1.9263660069555044 ms,
ECHO 2.6545010041445494 ms, serial sparse 2.488470054231584 ms and dense
prefetch 5.548551096580923 ms. The HBM batch difference is not a preparation
effect: only ECHO executes the new bounded path. The separate private paired
gate remains the evidence for the preparation optimization's complete-model
effect. Formal node counts are 197 / 269 / 221 / 203; the ECHO preparation
uses nine GPU kernels and no preparation memsets.

The independent selected-report review was
`/tmp/cxldsagr-checks/deepseek_free_selected_reports_review.json` and its
same-named Python script. It checks 46 manifest assets, 32 clean timing samples,
96 layer traffic samples, eight raw operator captures, MFU formulas and
conservation, and every displayed GPU interval in all six comparison panels
against the corresponding SQLite rows. Official numerical acceptance remains
false. A final copy/document review,
`/tmp/cxldsagr-checks/deepseek_free_published_docs_review.json`, verifies all
44 selected files against the reviewed raw reports and 15 archived audit
artifacts. Original audit bytes are retained under
`experiments/deepseek_v32_mfu/output/data/deepseek_h64k_a1_free_20261008_01/publication_review/`;
its `publication.json` records the publication and audit identities.

The incremental retained-source audit is
`/tmp/cxldsagr-checks/bounded_retained_source_scope_20261008_01.json` with a
same-named Python script. It verifies five archived baseline runtime sources
and all seven frozen candidate/live files. The baseline test file is absent
from the private bench runtime archive and is explicitly excluded. Only
`SparseTokenCache.prepare_prefetch` and official `_prepare` differ among
existing Python callables. Generic logits, full token and validator, cache
append/recall functions and module executable statements are unchanged.
The native translation unit adds only one include and one FFI export. Six
live candidate files match exactly; the new native header differs only by
removal of its last empty line, with `live == frozen[:-1]`.

All added cache eligibility checks are under `new_count == 1`. Retained
A128–A1024 MFU shapes and transient C10 candidates keep their original path;
their history chunks are all 1,024 tokens. Existing cache-manager measured
append/recall bodies and static allocation formulas do not change. This is
incremental source-scope evidence, not an assertion that the new DSO matches
an old compiled payload, a rerun of retained experiments, or a capacity claim.

Root reviewed `/tmp/cxldsagr-checks/deepseek_free_cleanup_inventory.json` and
deleted exactly 1,531 superseded raw files, 391,892,494 bytes, after checking
their existence, sizes and hashes. The deletion record and inventory copies
were recorded in that matrix's `publication_review/cleanup.json` and companion files.
The old CUB bench and stage profile remain because live
formal-reproduction, compute-collection, compute-inspector, event and replay
controls bind them. The old operator profile has no outside top-level data
JSON dependency in the inspected two experiments; its `request.json` remains
at its original path, and its superseded remaining files were removed. Five
protected request files were rehashed unchanged after cleanup. Every
signed request, accepted component/private run and mirror `_02` remains.
At that publication, stable report paths received the accepted FREE bytes.
They have since been replaced by later accepted publications. No signed source
identity or historical command was rewritten in the recorded cleanup.
