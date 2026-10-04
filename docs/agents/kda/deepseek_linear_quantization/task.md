# DeepSeek linear activation quantization

Status: the user switched the implementation preference to handwritten Triton
on 2026-10-04, while retaining the request to avoid production `torch.compile`.
Native V1/V2/V3 passed exactness but were rejected for large-batch Graph
regression. V4 has offline preparation only and is no longer the active
candidate. Handwritten Triton T1 passed exactness and two independent timing
windows, is integrated, and has passed the production API, consumer and
actual-checkpoint MLP gates. Fresh full-serving latency/MFU measurements remain
the root task's next gate; component evidence is in [checkpoint.md](checkpoint.md).
The current execution plan is
[triton_implementation_plan.md](triton_implementation_plan.md).

Replace production `torch.compile` activation quantization with a handwritten
Triton kernel and a lazy local binding. Preserve the public
`quantize_fp8_activation(x)` contract and every FP8 data byte / FP32 scale bit of
the isolated compiled DeepGEMM oracle. CPU reference behavior stays unchanged.
The user's latest implementation preference is Triton.

Inputs are BF16, FP16 or FP32 `[M,K]`, M >= 0 and K > 0, on Hopper SM90 or CPU.
CUDA covers every supported nonnegative row/column stride, storage offsets,
overlapping read-only views, partial 128-element groups and empty M. Return
independently owned contiguous E4M3FN `[M,K]` and FP32 `[M,ceil(K/128)]`.
Keep current-stream ordering, retained-output lifetime and changed-input graph
replay semantics. Do not add a production eager fallback whose nonfinite
conversion differs from the former compiled helper. Preserve non-SM90 rejection.

The exact arithmetic and adversarial cases are specified in the independent
[semantics review](semantics_review.md). In particular, use the compiled
FP32 reciprocal-448 constant, bitwise exponent ceiling/clamp, NaN-propagating
maximum, FP32 reciprocal/product and the validated SATFINITE E4M3 conversion.
No tolerance replaces bitwise acceptance. CPU reference arithmetic is a
separate retained contract, not the complete GPU oracle.

Primary performance matrix: BF16 M=1,121,128,1024 crossed with
K=1536,7168,16384,18432. K=2048 remains in functional coverage. Preserve dense,
grouped and packed/shared MLP consumers. Do not change scale output layout,
GEMM recipes, weights, gate/up sharing, indexer quantization or transport.

Baseline source is `/tmp/deepseek-motivation-c8_torch_reference-frozen-rb0_2eqx`,
483 files, mapping SHA-256
`cae2da8b9a58907ae35f99b5eeda017348b1a0a1ce1e1421fd6ca7703c956bb6`.
The official helper is DeepGEMM commit
`057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7`.
Only isolated oracle validation/timing may compile that helper.

Production files are `operators/deepseek_v32/linear/quantization.py`,
`_quantization_kernel.py` and `_quantization_identity.py` in that component.
`build_info()` must be CPU safe and record policy revision, dtype/layout and
launch dispatch, SM90 target/options, actual local source hashes, and Triton /
CUDA toolchain identity. It must not compile or initialize CUDA. Launch on the
active PyTorch stream; expose observed live JIT specializations separately from
stable build identity. The JIT identity must include local dependencies and
build options, without scanning another model or unrelated operators.

Earlier native prototype evidence remains in
`/tmp/deepseek_linear_native_quantization_20261004/` and its versioned siblings.
The Triton candidate uses its own workspace and run identity. Root schedules CUDA work;
no GPU run is authorized by writing this plan. Commands and promotion gates
are in [triton_implementation_plan.md](triton_implementation_plan.md). Promote only after
exactness, independent review, complete-API measurements and a confirmation
window. Integrated consumer/full-serving checks and fresh formal results remain
root-owned; a component win cannot complete the four-scheme/MFU objective.
