# DeepSeek norm I/O task contract

Date: 2026-10-04. The plain/fused full component validations and complete-API
benchmark passed independent audits. Root accepted the component and authorized
production integration. The narrow BF16 adapter passed focused integrated GPU
tests and live source-identity checks; a temporary postprocessing-only field
error was resolved by independent saved-data audits. Full-checkpoint/serving
acceptance remains with root. This contract grants no additional CUDA window.

## Objective

Reduce DeepSeek norm conversion/allocation work while preserving the current
official CuTe FP32 computation, checkpoint FP32 norm weights, unrounded FP32
residual sum, exact normalized output and existing ownership/stream behavior.
Prefer existing vendor computation. Do not edit installed/upstream files.

The full mixed-type direction must avoid materializing full FP32 activation
copies. A simpler packed-buffer candidate is evaluated separately: it retains
those copies and only removes an output conversion launch. It cannot be
reported as accomplishing the mixed-type objective.

## Contract and scope

- Hopper SM90; record the actual device and software identities on every run.
- Plain norm widths 512, 1536 and 7168; fused residual norm initially width
  7168. Q=1/9/121/128/1024; contiguous and actual row-strided layouts.
- Actual source-layer 0–2 input/post-attention/query/KV weights plus final norm
  remain FP32. Epsilon 0 for nonzero rows, checkpoint 1e-6, 1e-5 and 1e-3.
- BF16 and FP32 inputs as required by the current API. A BF16 KV latent is a
  `[Q,512]` view with row stride 576; contiguous-only success is insufficient.
- Compare normalized FP32 intermediates bitwise before accepting final BF16
  agreement. For fused norm, compare both the unrounded FP32 sum and separately
  rounded residual. Zero tolerance; BF16 agreement cannot hide FP32 drift.
- Caller input/residual/weights stay unchanged. Outputs must not alias caller
  storage. Any shared backing between returned views must be declared, have
  disjoint extents and be charged at complete owning-storage capacity.
- Preserve FP32 constructor geometry, reduction operand shape/order, official
  `row_reduce_sum_multirow`, epsilon placement, `rsqrt(fastmath=True)`, final
  multiplication order, PDL and stream dependencies. Changes to these are
  separate candidates rather than repairs hidden under an I/O label.
- CPU/autograd behavior, indexer LayerNorm, SiLU, RoPE and GEMMs are outside
  this task. No tolerance relaxation, weight cast or Hadamard change.

The two direct unmodified-class mixed-type candidates have already failed
compilation. They are recorded in `checkpoint.md`; their FP32 controls passed.
The earlier vendor Triton candidates failed FP32 exactness. These outcomes
do not authorize broader arithmetic changes.

## Promotion gates

1. Independently review the scoped source delta before CUDA execution.
2. Pass strict screen, full numerical matrix, ownership, non-default-stream
   and graph replay/lifetime checks on the actual changed implementation.
3. Measure complete API costs, including allocation, packing, casts and any
   fallback, in a quiet window. Kernel-only timing is supplementary.
4. Revalidate all real-checkpoint hidden/logit outputs and graph resource
   accounting before integrating with serving. Changed returned storage can
   affect private graph pools even when static input planning is unchanged.
5. Root runs quiet full serving/MFU measurements with a fresh source snapshot
   and run ID before publishing a performance claim or replacing old results.

Initial probes belong under `/tmp/deepseek_norm_adapter_20261004/`. If later
promoted, a model-specific local kernel belongs under
`operators/deepseek_v32/norm/`, with its adapter in
`models/deepseek_v32/nonmatrix.py` and adjacent tests. Do not add `__init__.py`
or copy the complete vendor module. JIT identity must cover the local source,
the actual imported vendor helpers, relevant DSL version and geometry.
