# Consumer-layout scale candidate

Status: private correctness and first component screen completed on 2026-10-06;
no production promotion. This is a
new layout-only candidate following integrated Triton T1. Earlier component
tasks and measurements remain historical evidence with their original scope.

The objective is to avoid DeepGEMM's SM90 activation-scale transpose by directly
emitting FP32 scales with logical shape `[M, ceil(K/128)]` and stride
`(1, ceil(M/4)*4)`. The current production T1 quantizer and `fp8_linear` are the
performance baseline. The isolated compiled official DeepGEMM helper is an
exactness oracle; old C8 snapshots are not a performance baseline.

The first private implementation adds an explicit layout opt-in to ordinary
SM90 `fp8_linear` and its quantizer. The public quantizer defaults to contiguous
scales. Grouped/MoE and indexer paths retain their current dispatch. No arithmetic,
FP8 conversion, GEMM API, GEMM recipe, weight, prepared-activation ownership,
one-use validation, output-alias check, input padding or error behavior is
removed. A prepared activation is consumed in its existing layout without an
extra conversion. CPU behavior remains the existing reference behavior.

Correctness requires exact FP8 bytes and logical FP32 scale bits on the existing
213 adversarial fixtures, full input-owner and output-padding preservation,
retained outputs, nondefault streams and changed-input graph replay. Ordinary
linear outputs must match current T1 plus unchanged DeepGEMM, including partial
K/N padding and the existing prepared/output contracts. Unsupported hardware
continues to fail explicitly. No numerical tolerance replaces these checks.

The full-API matrix is BF16 M=1,121,128,1024 crossed with eight current-model
`(K,N)` pairs: `(7168,1536),(1536,24576),(7168,576),(1536,8192),
(7168,128),(16384,7168),(7168,18432),(18432,7168)`.
M128/M1024 represent extend/prefill; the repeated gate/up shape remains two
separate GEMM consumers in prepared-activation correctness checks.
Separate quantizer-only timing is diagnostic. The promotion metric is complete
quantize-plus-original-GEMM latency in eager and owned graph replay APIs, with
stable borrowed replay reported separately. Profile must show the activation
transpose disappears and the original GEMM remains.

All implementation and harness files stay under
`/tmp/deepseek_linear_scale_layout_candidate_20261006_01/`. Only the scoped
linear sources are frozen, so unrelated resident-mask integration does not
invalidate this candidate. This agent may prepare and run CPU checks on cores
32–39 with eight threads. Root owns every native build, CUDA check, benchmark,
profile and full-model run; preparation alone does not authorize those actions.

Root may advance the candidate only after independent correctness and identity
review, fixed balanced AB/BA complete-API measurements, a confirmation window
when a component win is claimed, and separate profile evidence. A null or
regressive complete API excludes the affected shape/API from deployment.
An unrestricted layout opt-in is not accepted when a required cell regresses.
Component acceptance permits a
root-owned full-model trial; it does not establish full-model gap/MFU gates or
authorize production promotion or report replacement.

The first screen covers all 32 shapes, while the current complete-model task
uses Q128 and Q1024 only. All their eager and owned-graph paired medians improve
in pooled and both order strata, for both timers. M1/K16384/N7168 graph APIs
regress, excluding unrestricted deployment; Q1024/K7168/N1536 borrowed replay
has an order reversal and is not accepted as a demonstrated benefit. Keep both
findings visible. A fresh unchanged fixed matrix confirmation is permitted to
confirm the prospective Q128/Q1024 deployment scope, followed by a distinct
profile. Any later model opt-in must be restricted explicitly to verified query
shapes; other inputs retain the existing contiguous path. This narrows the
candidate's deployment scope without changing the four-method complete/layer
gap gates or claiming a general quantizer speedup.
