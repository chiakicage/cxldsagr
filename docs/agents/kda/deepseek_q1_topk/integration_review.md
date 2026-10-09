# CUB top-k production integration review

Status: integrated and formally remeasured. The accepted patch was applied
after the component and full-model paired gates passed. Actual production
tests passed 65 cases, including dispatch, numerical boundaries and backend
provenance. Candidate experiment files and `q1_qkv.py` remain unchanged.

The authoritative production cohort is `deepseek_h64k_a1_cub_20261008_01`,
with operator profile `deepseek_h64k_a1_cub_mfu_profile_20261008_01`.
Independent correctness, clean timing, saved-output comparisons and source/
native/input audits passed before publication. The five updated report trees
are bound by `/tmp/cxldsagr_cub_publication_20261008_01.json`. Synchronized
full-step timing is approximately unchanged; the private paired improvement
below is not a demonstrated end-to-end gain in the formal matrix.

The inherited task contract and implementation plan remain in `task.md` and
`implementation_plan.md`. This integration changes only Q=1, count=2048 and
N>=32768. It reuses the unchanged FlashInfer SMALL Filtered selection, then
calls the same CUB CUDA translation unit as the accepted private candidate.
Other shapes retain the original sorted operation and invalid-index mask.

## Implemented scope and review checklist

1. Add a lazy immutable CUB wrapper and its CUDA file under the model indexer;
   bind the installed FlashInfer CCCL include closure, wrapper and loader.
2. Parameterize the existing official int32 adapter's two sorting flags with
   their original defaults. Gate only the accepted Q1 geometry in `exact_topk`.
   Preserve validation, contiguous conversion, output ownership and failure
   propagation. Do not catch runtime errors or select an alternative backend.
3. Extend backend provenance with build/source identity and observed loaded
   CUB native identity. Reading runtime provenance must not invoke factories.
4. Add dispatcher/ABI/failure tests and meaningful GPU cases at the new length
   threshold, for ties, signed zeros, causal tails, layouts and changed graph
   inputs on a nondefault stream. Add identity mutation tests.
5. Retain the reviewed changed-file mirrors and applied patch with their
   base/patch hashes. Formatting, actual GPU tests and formal four-method
   remeasurement completed after integration. The new production module and
   native identity have their own acceptance; the private receipt remains
   evidence only for its archived preintegration sources.

Risks are dispatch boundaries, signed-zero ordering, native stream selection,
first-use capture, and an incomplete source closure. The accepted CUB source
should remain byte-identical. The added helper uses existing FlashInfer/CUB,
Torch and TVM-FFI dependencies; no dependency pin or environment change is needed.

Independent CPU audits are at
`/tmp/cxldsagr-checks/q1_topk_cub_independent_review.json` and
`/tmp/cxldsagr-checks/q1_selection_model_independent_review.json`. They verify
saved evidence and hashes, and do not rerun GPU execution. The latter confirms
100 paired model samples per arm: layer medians 1.046336/1.038720 ms, full graph
1.484304/1.476480 ms, wall 1.830168/1.820790 ms. Paired median gains are
6.976/6.320/6.541 us; layer and full-graph comparisons win 86/100 pairs. Wall
AB and BA medians have opposite signs, so wall improvement is less consistent.
The saved prefix/default-query outputs match bitwise; the harness reports its
eager/graph and two changed-token checks but did not save those changed outputs.
