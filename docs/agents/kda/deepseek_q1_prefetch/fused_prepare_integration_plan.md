# Stateless Q1 preparation: production integration plan

Status: applied on 2026-10-09 at 00:56:21 Asia/Shanghai after root released the
source freeze and the independent reviewer approved the minimal implementation.
The nine-file patch SHA256 is
`6d35542bb9aea59a50ad29b6a9ce5637096d6702996fe9467b038281fe3bf624`.
Focused main-tree CPU acceptance passed (261 tests; 24 GPU skips are excluded
from GPU acceptance), and scoped Ruff plus `git diff --check` passed. Focused
GPU1/H200 correctness acceptance passed all 151 tests with zero skips in 56.12 s.
The only warning was FlashInfer's upstream `Arch` import deprecation. The rebuilt
generic CUBIN is byte-for-byte identical to the preserved formal baseline,
including all 44 kernel text sections and all metadata sections.
The accepted component and private model evidence remains bound to its original
sources and actual native artifacts. Fresh production check, benchmark and both profile roles completed in all 16
children of `deepseek_h64k_a1_fused_prepare_20261009_01`; audited publication is
`deepseek_h64k_a1_fused_prepare_report_20261009_01`. This plan is complete;
current outcomes and cleanup are recorded in `checkpoint.md`.

## Minimal change

1. Add the candidate's page64 packing, identity block table, logical-to-host token
   table and stage reset kernel to
   `operators/deepseek_v32/indexer/csrc/official_prefetch.cu`, in namespace
   `official_prefetch`. Retain the original `prepare`, promotion validation,
   record publication and tag clearing exports. Do not modify or copy official
   ECHO headers, scoring, scheduler or causal-cleaning code.
2. Add a current-key entry in `official_prefetch.py`, for example
   `logits_from_keys(q, k, weights, scales, context_lens, prefetch)`. It allocates
   the same packed/table/stage tensors, launches fused preparation, calls the
   existing official scheduler and scoring bridge, then the original promotion.
   Keep the explicit packed-input `logits(...)` API and `_prepare(...)` as the
   reference/control entry. A small shared staging-allocation/registration helper
   may remove duplication, provided every existing validation and ownership step
   survives. Do not import experiment code into production.
3. Switch only the existing `use_official_q1` branch in `echo.logits`. Preserve
   the public signature, bounds creation/validation, output padding/view and all
   other branches. No model, cache, serving, capacity or generic-loader refactor
   is needed. Leave previously checked private producer/loader/analyzer files
   untouched; fresh production evidence is required after relocation.

The official branch predicate stays exactly: non-None lease, Q=1,
`N=query_start+1>=32768`, `history_length=query_start`, and integer
`max_prefetch>=64`. Resident Q1 retains the original paged predicate based on
causally visible length; other resident calls remain nonpaged. Other offload
shapes/capacities keep the existing prefill branch. Validation, initialization,
build and execution failures propagate; none selects another backend.

The separate bounded free-slot preparation predicate also stays unchanged:
compatible provider, one active session, exclusive operation, ordinary append,
`P-H>=64`, and the provider's official-Q1 support check. Multi-session/transient
leases can still reach official Q1 through the existing full preparation; do
not mistake the bounded-token predicate for the indexer dispatch predicate.

## Invariants and storage

- Preserve all Python ABI checks and native bounds assertions, including
  `context_lens[0]==history+1` before reads, valid host pages, signed-int32 tag
  bounds, at least 64 legal slots, contiguous/no-grad tensors and SM90. Keep the
  packer's 4-byte K alignment restriction; use the candidate's scalar-word path
  for valid 4-byte-but-not-16-byte alignment and its vector path when aligned.
- Rebuild from current K/scales and page-table bytes on every call/replay. Copy
  raw FP8/FP32 bits, including NaNs, signed zero and subnormals; zero page padding.
  The suffix has no host record. No retained packed history or content identity
  is introduced.
- Consume the original prepared token once (bounded consumption explicitly 64);
  preserve full journal/stat reset for unprepared leases. Register the original
  `_Staging` owner and cleanup callback before any launch that can publish tags.
  Preserve `POLICY`, strict `score>offset[1]`, cap64, all offset bits, promotion,
  finalize, exact top-k/ties, complete recall and failure cleanup, including host
  ID zero. An uncertain CUDA completion retains storage/owner and disables reuse.
- Reuse the current counter/journal/stat buffers. At N=65537, packed storage is
  8,659,200 B, block table 4,100 B, host/stage storage 336,132 B; scheduler and
  score storage keep their current sizes. No new persistent or temporary storage
  is required. Allocation order can affect the graph allocator: independently
  inspect complete private segments, allocated, reserved and device-used values.
  Equal tensor sizes alone do not establish equal reserved memory or capacity.

## Native and measurement binding

Keep the existing immutable `official_prefetch` loader and SM90 flags. Its
`build_info()` already binds the Python adapter, CUDA file and loader; add a
separate preparation-version field while retaining the predictive policy ID.
Record the actual mapped new ELF, build record, source closure and toolchain in
fresh check/bench/profile identities. Never accept old receipts by dropping
changed fields or normalizing ELF bytes.

`echo.build_info()` includes `echo.py`, so the branch edit also changes the
generic prefill module name despite unchanged CUDA sources. Do not fix that
fingerprint or loader in this optimization. Record the scoped Python/CUDA diff;
compare the old and newly built generic device-code payloads if asserting binary
continuity. Preserve original identities for retained non-Q1 reports.

Before release, all 12 offload check/bench/profile/operator results in accepted
cohort `deepseek_h64k_a1_isolated_20261008_01` were verified to bind the same
2,559,304 B generic ELF, SHA256
`559763db1915507709a379f928907b84e8a5bae1203c1d92af9e26a02e12ba01`.
The matching runtime ELF and all 13 archived generic source files were preserved
under `/tmp/cxldsagr-checks/q1-fused-prepare-production/generic_baseline_20261009_01/`.
`baseline.json` hashes to
`a6b5ee6d9c47bd602a9ab4e9505751bb8df859d58c016dc58821e1d4633728a5`;
each archived source also matched the current pre-release file. CPU-only
`cuobjdump --extract-elf all` produced one SM90a CUBIN, SHA256
`0131578b8d4ab4e55e9cf1cd728387cec7eff2d36032835344d7847173410e43`,
with 44 `.text.*` sections. Their exact hashes and all other section identities
are in `device_payload.json` (SHA256
`c2551dba81b6d70ebd4e1867311e837114d3a3521c23a91d08e9a08f3e42c23b`).
The cohort itself was not changed. Saved runtime Ninja metadata is explicitly
an observation now, not a historical receipt assertion. No binary was built
or launched for this baseline capture.

After production correctness acceptance, an explicit adapter load captured the
new generic ELF `cxldsagr_echo_indexer_2a106c9b8c39ee24.so`, SHA256
`afafc5d3bee190963b64821cce3b18ef852ff00be425cf77015916480c6b08f1`.
Its complete extracted 1,218,840 B SM90a CUBIN has the same SHA256 as the baseline;
all sections match. The only changed source hash in the generic build identity
is `echo.py`; all other source, header, flags and revision fields match. This
supports unchanged generic device-code scope, without merging the distinct ELF
identities or establishing performance for the new official preparation path.

The new immutable official-prefetch artifact is
`cxldsagr_official_prefetch_a960ded27efe4b1f.so`, SHA256
`4c8afb1790d194fbdc5c0b70d49654c1e3298d9a5c3ee50fc969cff8ec8bc320`.
Native copies, observed process maps, complete build identities and exact CUBIN
comparison are under
`/tmp/cxldsagr-checks/q1-fused-prepare-production/production_native_20261009_02/`.
The first metadata-capture script stopped on an `AttributeError` from an incorrect
Python import namespace; the corrected standalone capture imported
`_native_cache` directly. No production source, test or loader changed. These
maps belong to the explicit post-acceptance adapter load; pytest's own process
maps were not persisted. The fresh formal cohort retains its own runtime gates.
Focused commands and terminal results are recorded in
`/tmp/cxldsagr-checks/q1-fused-prepare-production/focused_acceptance.json`, with
the explicit boundary that results were transcribed from tool responses.

The private model gate needed immutable generic/FlashInfer loaders because
Ninja rebuilt identical sources into different ELF bytes. Production formal
runs must pass their own strict actual-ELF gates. Do not import the private
monkeypatches or broaden this integration into dependency management. Identity
drift is a hard failure with saved diagnostics and a separately scoped fix.

## Profiling and report impact

`InstrumentOperators` and `ColdPrefetchObserver` wrap public `echo.logits`, not
`_pack_q1_keys` or `_prepare`. Therefore the new kernel remains inside one
`indexer_fused` operator call and the existing `indexer_prefetch` model stage.
Keep these wrappers and native graph lineage. No extra nested matrix scope or
FLOPs are needed. Complete indexer time includes fused preparation, scheduler,
official core/clean and promotion; top-k, pool preparation and recall retain
their existing boundaries.

The `timeline.activity_lane` rule already marks `official_prefetch::` kernels as
GPU control, before broad graph/operator ownership. Name the fused kernel in
that namespace and test both alignment specializations. It reads HBM keys and
device page metadata, so it adds no H2D payload. Only the original official core
owns staged H2D; promotion stays D2D. `extend_gap_sources` will classify this work
as indexer layout. Official comparison/diagnosis readers inherit these labels.
Add classification and exclusive-attribution regression cases; no production
hook change is currently indicated by the inspected wrappers.

One formal check must change: `run_contract.validate_runtime_participation`
currently demands a Triton page64 specialization for every large Q1. An isolated
cold ECHO process will no longer execute it. Require the versioned fused
preparation identity and its actual mapped adapter for that declared path;
retain page64 requirements for resident paths and old report identities.
If necessary, expose an optional loaded-adapter runtime record through
`backend_provenance`, without loading unused providers. Bind an observed
current-key entry marker after that entry returns normally; use a stable policy
identity, not invocation counts that differ by measurement phase. The retained
packed-input API loading the same ELF must not claim fused preparation executed.
Do not satisfy the gate
by warming an unused packer. Tests must reject missing/wrong fused records and
continue to reject a missing resident page64 specialization. Any additional
reachable Q1 path must declare its actual adaptation rather than infer residency
from capacity. Keep the frozen private profile analyzer's exact node sequence
bound to its private run; do not rewrite it to accept production names.

## Focused acceptance and formal refresh

Add `operators/deepseek_v32/indexer/tests/test_q1_fused_prepare.py`. Compare the
new preparation with CPU byte/table references and the retained packed-input
bridge, not a baseline call to the now-fused public dispatcher. Cover page
boundaries, both alignment paths, raw bit patterns, randomized page permutations,
same-address K/scale/page changes during replay, nondefault streams, unchanged
source tensors/maps, output stride/tails, both prepared-token types, unprepared
leases, duplicate owners and injected failure cleanup. Malformed context/page
device assertions run in isolated subprocesses. Cover cold, partial, resident,
unsaturated, saturated and empty predictions; every actual saturated execution
needs its own legal transition proof. Scores/top-k/consumed KV stay exact.
Include the existing full-preparation/transient lease contract at the operator
boundary; the single-session cold full-model exception does not extend to other
residency/session states.

After source release, use these focused commands from the repository root.
The GPU1/CPU8-15 commands require the parent's explicit scheduling release.

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  operators/deepseek_v32/indexer/tests/test_q1_fused_prepare.py \
  operators/deepseek_v32/indexer/tests/test_native_cache.py \
  experiments/deepseek_v32_mfu/tests/test_method_isolation.py \
  experiments/deepseek_v32_mfu/tests/test_backend_provenance.py \
  experiments/deepseek_v32_mfu/tests/test_operator_instrumentation.py \
  experiments/deepseek_v32_mfu/tests/test_full_graph_profile.py \
  experiments/deepseek_v32_mfu/tests/test_launch_gap.py \
  experiments/deepseek_v32_mfu/tests/test_operator_report.py \
  experiments/deepseek_v32_mfu/tests/test_method_isolation_report.py

PATH="$PWD/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 8-15 .venv/bin/python -B -c \
  'import torch; torch.cuda.init(); assert torch.cuda.get_device_capability() == (9, 0)'
PATH="$PWD/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 8-15 .venv/bin/python -B -m pytest \
  -q -p no:cacheprovider --import-mode=importlib \
  operators/deepseek_v32/indexer/tests/test_q1_fused_prepare.py \
  operators/deepseek_v32/indexer/tests/test_official_prefetch.py \
  operators/deepseek_v32/indexer/tests/test_q1_dispatch.py \
  operators/deepseek_v32/indexer/tests/test_q1_packing.py \
  operators/deepseek_v32/indexer/tests/test_echo_indexer.py
```

CPU skips are not GPU acceptance. Run scoped Ruff and `git diff --check`; add
the changed runtime-contract test to its existing suite. Then freeze the final
source tree and run a fresh 16-child isolated cohort (all four checks first,
then each method's independent bench, minimal profile and operator profile).
Use the existing request SHA256
`d776856368e64c931805f2181c41417609d55082866cc71e7468dfb6d57b380f`.
The formal script fixes GPU0/CPU0-7, H65536/A1, real L0-L2, P=NH=65600,
cold offload, FP8/BF16, one warmup, three prefill and five step samples.
Coordinate quiet CPU/DRAM and GPU ownership before execution; do not preload
other methods. A proposed fresh cohort ID is shown below and must be unused.

```bash
MFU_RUN_ID=deepseek_h64k_a1_fused_prepare_20261009_01 \
  bash experiments/deepseek_v32_mfu/scripts/method_isolation.sh \
  --model /preset-models \
  --request experiments/deepseek_v32_mfu/output/data/deepseek_h64k_a1_isolated_20261008_01/request.json
CUDA_VISIBLE_DEVICES= .venv/bin/python -B -m \
  experiments.deepseek_v32_mfu.src.method_isolation_report \
  --manifest experiments/deepseek_v32_mfu/output/data/deepseek_h64k_a1_fused_prepare_20261009_01/manifest.json \
  --output-dir experiments/deepseek_v32_mfu/output/data/deepseek_h64k_a1_fused_prepare_report_20261009_01
```

Independently audit saved outputs, each actual transition/traffic record,
source/native binding, process ownership, graph nodes and private memory,
complete sample retention and all wall/profile denominators before publication.
The formal cohort refreshes the current implementation; candidate speedup is
supported by the predeclared private 500-pair comparison and all of its order/time
strata, not by subtracting two formal-cohort medians. Inspect the production
profile for one fused preparation per eligible layer, removal of its three
predecessors, and unchanged downstream policy/ownership. No official SGLang
rerun is required for an unchanged reference, and numerical acceptance remains
false across implementations.

## Result boundary and publication

| Dependent result | Required action |
| --- | --- |
| MFU H64K/A1 and local official comparison/diagnosis | Fresh cohort, complete independent audit, new report run IDs, then replace selected reports and superseded affected outputs. Rebuild `combined_decode` and `decode_gap` from the new per-method artifacts. |
| MFU A128 single point and H/A matrix | Audit actual indexer query sizes and unchanged dispatch. Q>=128 and chunk1024 do not enter the new branch; MLA-only capacity splitting does not turn the indexer into Q1. Keep valid original reports with original identities. |
| Motivation C10 A128/256/512/1024 | Same dispatch/source audit; ordinary C10 candidate batches and chunk1024 remain outside the change. No new C10 performance or GR quality claim. |
| Cache-manager A128 post-top-k, append and exact recall | Retain after dependency audit: query128 uses old indexer dispatch, and isolated recall/append do not execute this preparation. Their original timing boundaries and receipts stay intact. |
| Cache-management static plans and A128 request memory | Recheck the unchanged allocation ledger and Q1 active-byte floor. Keep prior static/request results as their original evidence; new graph-reserved observations do not refresh C10 request memory or capacity-fill acceptance. |
| Resident page64, promotion, top-k, hint, attention component controls; official Engine/HTTP | Preserve valid unchanged component/reference evidence with its exact boundary. Fused preparation is a new combined boundary; do not relabel old component timings as its result. |

Until new publication passes, retain existing selected reports and mark the
production change as pending remeasurement. After publication, update the MFU,
official, model/operator and third-party entry points, clean only superseded
affected results, and send the research implication to Supervisor. Existing
unrelated user changes, valid controls and frozen private evidence are preserved.
