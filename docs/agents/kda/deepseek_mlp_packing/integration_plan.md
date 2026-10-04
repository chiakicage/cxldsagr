# Measured packed/shared dense MLP integration

Root selected `packed_shared` after independently auditing the exact component
screens, eager run and complete graph benchmark. The prototype improves both
complete eager and graph APIs; sharing adds a smaller repeatable graph gain to
packing. Production integration is authorized, with CPU checks and independent
review first. GPU validation and new production performance runs need a later
root scheduling grant. Frozen temporary probes/results remain unchanged.
Production source review, focused CPU checks, 37 production GPU tests and nine
actual-checkpoint-weight cases with 27 changed-input graph replays passed.
The test-only compiler-cache isolation fix and accepted rerun records are in
[checkpoint.md](checkpoint.md). Production serving performance remains unmeasured.

## Production contract

Extend `fp8_linear` and `CheckpointLinear.__call__` with optional `out`,
`quantized` and `return_quantized` arguments. With no options, retain the current
CPU, padding, BF16 output and official DeepGEMM behavior. Options form a narrow
SM90 inference contract: nonempty BF16 rank-two input, unit inner stride,
aligned K/N and a valid BF16 row-major output view. Reject unsupported explicit
options instead of silently allocating or copying a compact result.

Gate receives its packed half and `return_quantized=True`. Its normal matrix
API performs the official activation quantization and GEMM, then returns the
output plus a call-local prepared record. Up receives the other half and that
record. The record binds source tensor identity, shape, stride, pointer and a
version counter where available, and permits one consumption. It is never
stored on a model/linear instance or reused across MLP calls. The caller must
keep inference tensor contents unchanged between these adjacent calls; such
tensors do not carry mutation version counters. Down remains unchanged.

Out views must have the exact logical shape, BF16 dtype, same device, unit
inner stride, nonoverlapping rows and 16-byte pointer/pitch alignment. Reject
storage aliases with activation, weights/scales or prepared quantization.
The packed gate/up views intentionally share only their fresh output owner.
Keep both official GEMM shapes, recipe and independent weights/scales unchanged;
retain both official scale-transpose calls. No grouped GEMM or weight merging.

Enable packing explicitly for dense blocks only. Shared experts keep the
existing `CheckpointMLP` path. The runtime dispatch requires nonempty aligned
BF16 SM90 inference input and compatible independent gate/up FP8 weights.
CPU, autograd, BF16 weights, empty/padded/unaligned and other unsupported input
forms retain the old path. The MLP creates one `[Q,2N]` allocation, invokes
gate and up separately, then calls `silu_mul_packed` on the owner.

`graph_validate` owns every edit to `nonmatrix.py` and its tests. Requested
helper: contiguous `[Q,2N]` to owned BF16 `[Q,N]` through the unchanged
FlashInfer SiLU/multiply; no cat or hidden packing copy in the supported path.
Reference fallback preserves existing FP32 SiLU/product and final BF16 rounding.

## Attribution

Update the existing `CheckpointLinear` instrumentation wrapper to forward
optional arguments. Gate/up/down remain separate matrix records with unchanged
individual shapes and FLOP counts. Gate owns the one gate/up quantization;
up owns only its GEMM/scale-layout work, and down still quantizes independently.
Wrap the new packed activation helper under the existing `silu_mul` label.
The packed allocation/views stay inside the complete MLP call and outer graph
capture; no preparation is moved outside the measured complete API.

## Checks and rollout

- CPU: preserve existing dense/shared/MoE reference results and unsupported-path
  dispatch; reject invalid explicit option use; verify optional argument
  forwarding and all three matrix scopes, one gate quantization and independent
  down quantization using an instrumented adapter stub.
- CUDA after grant: byte-exact quantized data/scales, gate/up/SiLU/down versus
  the old sequence; guard/other-half boundaries, rejected aliases/layouts,
  source mismatch and repeat-consumption rejection; retained output lifetime,
  non-default streams and changed-input graphs using production adapters.
- Independent review must examine fallback dispatch, prepared-record lifetime,
  output stride validation, capture behavior and attribution before GPU work.
- Root-owned actual checkpoint propagation and full serving/graph validation
  follow component checks. The integrated API introduces validation/record
  overhead, so prototype timings do not establish its measured performance.
  New formal runs and matching attribution precede experiment replacement.

Files owned here: `operators/deepseek_v32/linear/fp8.py`, `echo_model.py`,
`echo_block.py`, their nearest tests and shared operator instrumentation/tests.
Norm/nonmatrix files belong to `graph_validate`; all other source edits remain
with root or their assigned agent. No publication/cleanup occurs in this step.
