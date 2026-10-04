# Measured norm integration plan

Date: 2026-10-04. CPU implementation/review and the focused production GPU
correctness gate are complete. Root's independent audit accepted the saved
test/runtime evidence despite a temporary postprocessing field-name error;
details are in [checkpoint.md](checkpoint.md). Full-checkpoint/serving acceptance
remains with root. Component timing is not a model or serving speedup result.

## Accepted component and changes

The frozen `benchmark_measure_01` uses benchmark SHA-256
`98daa59211e240cdb8869ab71c8340c353ae90085f431c9acb0039ffab4bd7dc` and result
`946f63dea505c95732dc4bfca1c23df7fde020fa44c66b3d6c90771a2e6a16dd`.
All 64 before/after check records passed; 32 comparisons retain 3,840 raw
samples and 64 separate traces. Independent review verified every sample key,
alternating order, summary, trace identity and source/validation identity.

Add a model-specific operator at `operators/deepseek_v32/norm/`, without
`__init__.py`. Move the measured plain/fused kernel methods into local modules,
retaining attribution and the imported official constructors, launch geometry,
predicate helper and `row_reduce_sum_multirow`. Preserve complete compile-time
layout equality checks. Remove temporary printing/file-output instrumentation;
kernel arithmetic, operand types and owned output ABI remain identical.

The public adapter imports only ordinary Python/Torch on CPU. Load CuTe,
FlashInfer kernel classes and TVM FFI lazily on an eligible GPU call. Compile
one specialization per plain width or fused geometry, device and PDL setting,
with dynamic row count. Keep the measured Float32 constructor, vec_size=4,
copy_bits=128 geometry and launch shared-memory budgets. BF16 operand copies
still transfer four values, and FP32 weights/reduction remain unchanged.

Root owns existing `nonmatrix.py` integration through this agent. Use the local
operator only for nonempty 2D BF16 CUDA inference on SM90, actual norm widths
512/1536/7168 (fused 7168), contiguous or validated row-strided views, FP32
contiguous aligned weights and compatible residuals. Explicit same-dtype
packing remains part of strided execution. Preserve existing ordinary paths
for FP32, CPU/autograd, unsupported geometry, dtype, alignment or layout.
No kernel failure silently changes algorithm or falls back after a partial
write. Local outputs are freshly allocated per call and use the caller's
current CUDA stream through the TVM FFI stream context.

Plain norm returns one independently owned BF16 tensor. Fused norm returns
independently owned normalized and saved tensors; the FP32 `h=x+r` remains
live through normalization and is separately rounded for saved output.
The `residual is None` model branch retains its existing returned-input
ownership behavior. No graph-private buffer is added to the production API.

## JIT identity

Record a deterministic identity covering the actual local adapter/kernel
sources, imported vendor plain/fused modules and reduction/layout helpers,
relevant installed CuTe/TVM FFI implementation and package versions. Do not
scan other models or all repository operators. Feed the fingerprint into the
compiled callable's process-memoizer key. The supported compile-scoped
`trace_finalize_hooks` API also adds a module StringAttr before IR hashing when
caching is enabled. Direct `cute.compile` sets `no_cache=True`, so this path
does not claim a DSL disk-cache hit. Independent review verified that behavior.
No global environment/cache directory mutation or upstream edits are allowed.
Source identity is fixed for each process's loaded implementation; changed
source/dependencies in a fresh process must select a different specialization.

## Packed SiLU coordination

Cache-C3 owns MLP packing/shared quantization changes in FP8 and model files.
This agent adds `nonmatrix.silu_mul_packed(packed)` independently of norm:
contiguous rank-2 BF16 CUDA inference `[Q,2N]` uses unchanged official
`flashinfer.silu_and_mul(packed).bfloat16()`, with positive even width and
half-width 16-byte alignment. It returns owned BF16 `[Q,N]`, without input
mutation or redundant packing. CPU/autograd/unsupported inputs split into
views and use the existing `silu_mul` arithmetic. Malformed odd width fails.
Cache-C3 adapts profiling so the existing SiLU stage still measures this call.

## Verification gates

1. CPU: import without initializing CUDA or importing CuTe/vendor kernels;
   fallback dispatch and packed SiLU exact CPU/autograd behavior; source
   dependency identity checks; Ruff and existing relevant CPU tests.
2. Independent CPU source review: compare relocated kernel method ASTs to
   frozen measured methods, inspect BF16 dispatch and stream/allocation behavior,
   verify compiler identity handling and packed SiLU helper contract.
3. In a root-assigned correctness window, compare the integrated plain/fused
   outputs bitwise with a direct independent official FP32 CuTe oracle, then
   its BF16 cast. Include widths, awkward rows, actual KV stride 576, caller
   guards, output retention, nondefault streams and changed-input graph replay.
   Explicit GPU mode must fail when hardware or dependencies are unavailable.
4. Root owns combined real-checkpoint hidden/logit comparison, runner fault and
   lifetime tests, graph resource accounting and new full serving/MFU runs.
   Engineering correctness evidence does not replace affected experiment
   measurements. Preserve old valid reports until replacements are accepted.

All temporary benchmark sources and results stay frozen. Existing source locks
were lifted by root only after independent component result/source audit.

The implementation is frozen at fingerprint
`8b8a3e6128f23d6d65f6aca3993723c09c74fc9cf0c4da75f21742f1a5630b26`.
`production_prepared_01.json` records 380 manifest labels and identical plain/
fused class ASTs against the measured prototypes. Computing the identity before
and after importing `_compile` produced the same fingerprint without initializing
CUDA. CPU targeted checks passed seven tests, with 64 GPU-only cases explicitly
skipped. The subsequent explicit GPU gate passed all 71 tests with zero skips.
Root's global CPU regression separately passed 2,778 tests and 58 subtests,
with 1,005 explicit skips; this does not replace the pending combined GPU gate.
