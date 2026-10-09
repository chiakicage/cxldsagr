# Private full-model gate for stateless preparation fusion

Status: private component and full-model correctness, the 500-pair model
benchmark, separate paired NSYS and source-bound preparation NCU diagnostics
passed their independent CPU audits. The accepted model receipt is check_02;
actual outputs, hint bits, transitions and both 60 MiB graph pools are bound to
that identity. The model analyzer/review suites passed 49 CPU tests; the
profile/NCU suites passed 23. The historical packing NCU and new preparation
captures are individually verified diagnostics, not a same-run NCU baseline
or summed latency pair. Detailed results, exact capture limitations and
retention decisions are in `fused_prepare_checkpoint.md`. The parent was
notified that the source-freeze gate is satisfied. Production integration and fresh formal acceptance/measurement subsequently
passed under `deepseek_h64k_a1_fused_prepare_20261009_01`; see `checkpoint.md`.
Private results retain their own boundary and do not replace production timing.

Contract: two independent real checkpoint L0-L2 models, ordinary persistent
H65,536/A1 append, P=NH=65,600, cold main KV with the same retained prefix and
hint per sample, FP8 projections and BF16 main KV. Only calls satisfying the
existing `echo._uses_official_q1` predicate use the private fused preparation.
All resident, Q>1, small-context and unsupported-capacity calls invoke the exact
original dispatcher with their original arguments. The official scoring,
prediction, exact top-k, hint update, recall and attention policies are unchanged.

The current production module has no `_logits_sm90` helper. A harness-owned
stable public trampoline is installed once before any observer. Its private arm
state chooses the original or candidate path; changing arms never replaces
`echo.logits` or an observer wrapper. Checks assert that the observer remains
installed, and teardown restores the original public function even on errors.

Reuse the accepted `q1_hint_model` helpers for independent model creation,
cloned outputs, current cache snapshots/transition comparison, graph-memory
description, cleanup, source archival and paired summaries. The private driver owns
its binding, component gate, explicit dispatch checks, observation loop and
sampling. The new launcher adds source/runtime/provenance hooks around that
frozen driver. Computation and original helper source files remain unchanged.

Independent model correctness covers three tokens (111090/111091/111092), eager,
diagnostic graph and clean timing graph. Save actual per-layer scores, selections,
all hint bits and compact transition proofs; every actual saturated prediction
must satisfy its own proof before comparing allowed residency differences.
Clone all saved outputs because complete graph results can borrow graph storage.
Destroy diagnostic graphs before preparing clean graphs. CPU dispatch tests also
prove exact delegation of resident/Q>1/unsupported calls and restoration on failure.

The component check and benchmark must match source/native/input identity. The
benchmark gate requires all five policies and all three real layer inputs,
100 balanced pairs, complete sample retention and negative median paired deltas
in both AB/BA strata. Missing evidence is a hard error, not a fallback.

Clean model timing uses 5 warmups and 500 balanced AB/BA pairs, predeclared
after correctness and before any model timing sample. Analyze the overall
paired results, both order strata, and all five consecutive 100-pair blocks;
retain every sample and do not select favorable blocks. The receipt binds the
minimum of 100 pairs, fixed shape/runtime and warmups, allowing this larger
predeclared sample count. Each wall sample
includes the complete `forward(return_hidden=True)` and synchronization. Prefix
restore, cold eviction, binding, metrics and validation remain outside timing.
Record each sample's actual traffic and actual private graph reserved storage,
plus allocated/reserved/device-used snapshots separately. Profile uses independent
process/receipt, native graph ownership and capture/replay labels; profile node
durations are not clean latency samples.

The commands below record the completed independent runs. Reproduction must use new output directories. The request is the active isolated cohort's saved H64K/A1 input, SHA256
`d776856368e64c931805f2181c41417609d55082866cc71e7468dfb6d57b380f`.

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 8-15 .venv/bin/python -B -m \
  experiments.deepseek_v32_echo_official.src.q1_fused_prepare_model_pinned_run check \
  --request experiments/deepseek_v32_mfu/output/data/deepseek_h64k_a1_isolated_20261008_01/request.json \
  --component-receipt /tmp/cxldsagr-checks/q1-fused-prepare/check_20261008_02/receipt.json \
  --component-bench experiments/deepseek_v32_echo_official/output/data/q1_fused_prepare_bench_20261008_02/summary.json \
  --output-dir /tmp/cxldsagr-checks/q1-fused-prepare-model/check_20261008_02

PATH="$PWD/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 8-15 .venv/bin/python -B -m \
  experiments.deepseek_v32_echo_official.src.q1_fused_prepare_model_pinned_run bench \
  --request experiments/deepseek_v32_mfu/output/data/deepseek_h64k_a1_isolated_20261008_01/request.json --pairs 500 \
  --receipt /tmp/cxldsagr-checks/q1-fused-prepare-model/check_20261008_02/receipt.json \
  --output-dir experiments/deepseek_v32_echo_official/output/data/q1_fused_prepare_model_bench_20261008_02
```

Promotion remains a parent decision after production integration and a fresh
formal cohort. The accepted private timing result is
limited to these two independently loaded model instances in one paired run.
It does not establish production, complete 61-layer or serving performance.
