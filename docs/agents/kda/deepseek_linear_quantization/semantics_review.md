# Independent activation quantization semantics and acceptance gates

Status: CPU source/PTX review only. No new kernel was compiled or launched for
this review. `cache_c3` owns the task, draft, implementation plan and candidate.
Production must remove `torch.compile`. The user subsequently selected
handwritten Triton; this supersedes the original native-CUDA-only constraint.
The scalar, layout and exactness rules below apply to either implementation.
The compiled official helper is allowed only in an
isolated validation/timing oracle. C8 formal/profile execution is on
hold until the replacement passes its gates. Root separately tracks the affected
local dependent experiments.

## Source identity and oracle boundary

The current frozen reference is
`/tmp/deepseek-motivation-c8_torch_reference-frozen-rb0_2eqx`, with root's
483-file mapping SHA-256
`cae2da8b9a58907ae35f99b5eeda017348b1a0a1ce1e1421fd6ca7703c956bb6`.
Its `operators/deepseek_v32/linear/fp8.py` SHA-256 is
`ecd3dbf40a7d479271d2d3e066c92a3d62edb5119c753e761f12c10d97ebb82e`.

The mathematical source is pinned DeepGEMM 057ca596:
`3rdparty/DeepGEMM/deep_gemm/utils/math.py`, SHA-256
`9cdecd0c0a0a8b3c64af87107464c91510876110eb80bdb485ff1f74eda8ea0e`.
The production call fixes `use_ue8m0=True`, `gran_k=128` and
`use_packed_ue8m0=False`. Output scales remain FP32 values with zero mantissas;
they are not packed exponent bytes.

Existing isolated compiled-source evidence was inspected without importing or
executing it:

- Aligned BF16 generated wrapper:
  `/tmp/deepseek-linear-main-fixed-20261003/inductor/el/cel6rlrwwbizx2fwkz3tygec664gwzrczhgk52mxuhfkiabym63p.py`,
  SHA-256 `8355a8eb4debd226c8f6f521b9703b564219527b8aabf35a616bc858010f4b66`.
  Its saved example input is `[1024,7168]`.
- Corresponding PTX is under that run's
  `inductor/triton/0/INGIJZG2TPH4IQMT35PNMO4OHIYJR223ALJSL7WWHMMWYS3S2HFQ/`,
  named
  `triton_per_fused___lshift_____rshift____to_copy_abs_add_amax_bitwise_and_clamp_div_mul_reciprocal_unsqueeze_view_0.ptx`,
  SHA-256 `7c7c6a7c6658ad88c9d0192c12a966922909256c062535d17e73115dc718f387`.
- Tail-aware generated reduction:
  `/tmp/deepseek-linear-quant-isolate-20261003/inductor/gb/cgb4wunq36vxru6apfvaolbm4esrwgf3xjxm527ex3c6usdkzp52.py`,
  SHA-256 `43e47a2253de14347795df5ab9131e27870c4063a8b79026eb1d3886cc059304`.

These files document historical compiler lowering. They do not establish the
exact generated variant used by a future oracle run. New validation records must
identify the imported official helper, compiler options, generated artifacts,
input layouts and unchanged source before/after execution. Do not import a
generated cache module merely to inspect it: those modules can initialize CUDA
or compile on import.

## Exact scalar and layout contract

For input `x[M,K]`, define `G=ceil(K/128)` and pad the final logical group with
zeros. Each row/group is independent; an incomplete final group must not read
the next row, allocation guard or stride padding.

1. The upstream helper materializes an input-dtype zero-padded tensor, takes
   absolute values, widens to FP32, and reduces the 128-element maximum. The
   compiled aligned path eliminates the materialization, loads BF16 values and
   widens to FP32 before arithmetic. Maximum and the `1e-4` floor propagate NaNs.
   `torch._inductor.runtime.triton_helpers.maximum/max2` implements this using
   comparisons plus an explicit NaN condition.
2. Let `a=max_nan(max_nan(abs(x)), float32(1e-4))`. The inspected compiled path
   computes `u=float32(a * float32(1/448))`. The reciprocal constant has bits
   `0x3b124925`; PTX uses `mul.f32`, not a newly introduced correctly rounded
   division by 448. Replacing this with division or log-based scale rounding
   needs independent equivalence evidence at rounding boundaries.
3. Let `b=bits_uint32(abs(u))`,
   `e=((b >> 23) & 255) + int((b & 0x7fffff) != 0)`, and
   `s=bits_float32(clamp(e,1,254) << 23)`. This is the actual
   `ceil_to_ue8m0` rule. It is not `exp2(ceil(log2(u)))` for nonfinite/extreme
   inputs. The inspected PTX folds some of these integer operations while
   retaining the same source rule.
4. Compute FP32 reciprocal `r=1/s`, then FP32 product `x*r`, then convert to
   E4M3FN. The inspected PTX uses `div.full.f32`, FP32 multiplication and
   `cvt.rn.satfinite.e4m3x2.f32`. There is no BF16/FP16 intermediate rounding
   and no explicit normalized-value clamp in the official source.
5. Return independent contiguous E4M3FN data `[M,K]` and contiguous FP32 scales
   `[M,G]`. Padded logical elements are excluded from returned data. The first
   candidate must retain this public ABI and normal output lifetime.

All-zero and sufficiently small groups use scale `2^-22`. For finite FP32 or
BF16 inputs the largest scale is `2^120`, so the reciprocal is still a normal
FP32 value. An Inf or NaN maximum reaches the exponent clamp and produces scale
`2^127`, whose reciprocal `2^-127` is subnormal. This last case specifically
tests `div.full` versus `div_rn`, compiler FTZ behavior and conversion semantics.
Do not infer its output bits from the mathematical formula alone.

The compiled E4M3 conversion is saturating. Eager PyTorch conversion and the
existing CPU reference can differ for overflow, Inf or NaN. NaN payload/sign
canonicalization, signed zero and Inf conversion therefore require the compiled
GPU oracle, with outputs compared as bytes. A NaN group has a finite clamped
scale under the official rule; replacing the scale with NaN would also change
the other finite elements in that group.

The existing indexer quantizer and CPU linear reference use log/exp scale
rounding and explicit normalized clamps. They remain useful independent checks
for their documented finite domains; neither defines the complete GPU linear
quantizer contract. Keep the CPU reference behavior separate and retain the
existing explicit rejection of non-Hopper CUDA execution.

## Native CUDA conversion and arithmetic constraints

The inspected CUDA 13.2 headers are
`/usr/local/cuda-13.2/targets/x86_64-linux/include/cuda_fp8.h` and
`cuda_fp8.hpp`. Their SHA-256 values are respectively
`ef06c30cd4199f9c342bf948dc71480c5c87309040a7c0657598a17558043d4c` and
`6dcc7d10e4ed4d3458429f0da99837ce338ef1b417034422a3f2d36c2a184c7f`.
Other installed copies exist under the Python environment's NVIDIA include
directory. The candidate must record the headers and toolchain actually used;
this review does not assume those copies are identical.

For `__CUDA_ARCH__ >= 890`,
`__nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3)` emits
`cvt.rn.satfinite.e4m3x2.f32` and returns its low byte. This is the instruction
seen in the historical compiled baseline. Keep the normalized value in FP32
until that conversion. The packed `__nv_cvt_float2_to_fp8x2` interface may be
used only after checking its x-low/y-high byte order and tail-store bounds.

`__NV_NOSAT` follows a software conversion path in these headers. It
canonicalizes FP32 NaN bits before widening to double, and E4M3 overflow/Inf
becomes NaN rather than signed maximum finite. It is not an equivalent
replacement for the observed baseline. Header host fallback behavior also
does not establish the device instruction's NaN sign/payload output; the
compiled GPU oracle must check those bytes.

The CUDA `__nv_cvt_float_to_e8m0` family is not a direct replacement for the
manual scale-bit rule. Its documented minimum scale is `2^-127`, and NaN
converts to an encoded NaN. DeepGEMM clamps the constructed FP32 exponent to
1 through 254, with a NaN maximum producing the finite scale `2^127`. Preserve
the integer rule, NaN-propagating reduction and floor explicitly. Plain
`fmaxf` alone drops a single NaN operand and is insufficient.

Inspect the candidate PTX for the exact reciprocal-448 constant, FP32 multiply,
reciprocal and E4M3 conversion. Do not enable fast-math or FTZ. The nonfinite
group case must test whether the reciprocal remains `2^-127`, whether
multiplication consumes that subnormal correctly, and how any underflowed
product is converted. Replacing reciprocal-then-multiply with direct division
can change these cases even if ordinary finite samples agree.

## Supported inputs and fallback decisions

The existing public linear quantizer accepts rank-two BF16, FP16 and FP32,
positive K, and zero or more rows. GPU callers can supply contiguous tensors,
pitched row slices, unit-stride slices with offset pointers, column strides or
other representable nonnegative strides. Dense and grouped linear adapters
also pad/reshape before quantization; their full adapter cost remains visible.

A generic templated native CUDA kernel can cover BF16, FP16, FP32, strides and
partial K using the same arithmetic, while a separate aligned BF16 dispatch
is tuned for checkpoint shapes. An eager official-helper fallback is not
automatically equivalent to the former compiled path: an Inf input produces
scale `2^127`; if the reciprocal survives, the normalized Inf reaches E4M3
conversion, where eager nonsaturating conversion and compiled SATFINITE can
diverge. A NaN group also exposes reciprocal underflow, signed zero and NaN
canonicalization. Overflow conversion must be checked independently even
though finite normalized values should stay within the selected scale range.

Any restricted native domain therefore needs an explicitly identified,
independently verified remaining path. Do not silently delegate unsupported
dtype/layout inputs to the eager helper. No fallback may invoke production
`torch.compile`. Empty input should return
correct empty outputs without a launch. Native import/compilation stays lazy;
CPU/reference imports must not initialize CUDA or load the native backend.

Primary checkpoint widths, read from `/preset-models/config.json` and current
projection construction, are:

| Quantizer K | Actual consumer examples |
|---:|---|
| 1536 | q_b and index_q projections |
| 7168 | q_a, kv_a, index_k and dense gate/up |
| 16384 | attention output projection after value expansion |
| 18432 | dense MLP down projection |

K=2048 also belongs to the model's expert/shared-expert paths; their retained
correctness tests must continue to pass, although the ten-dense-block serving
substitute does not measure MoE. Small widths and partial K belong to correctness
coverage, not replacements for checkpoint-shape performance measurements.

## Correctness gates before timing

- Compare every FP8 byte and every FP32 scale bit to the isolated compiled
  official helper. No tolerance, finite-only norm or floating `allclose`
  comparison substitutes for this. Save shape, dtype, strides, offsets,
  generated-source identity and per-case results.
- Cover M=0,1,2,7,8,15,16,17,31,32,33,63,64,65,121,128,129,1024; primary K
  above; partial/small K such as 1,24,127,128,129,257,1025,7167,7169.
  Test masks with invalid tail storage filled with a large value or NaN.
- Exercise every BF16 and FP16 encoding, both signs of zero, subnormals,
  largest finite values, Inf and several NaN payloads/signs. Place exceptional
  values at different reduction lanes and in partial groups. Exhaustive encoding
  coverage alone is insufficient: vary group maxima independently from the
  values being quantized.
- For FP32, include random magnitudes, adjacent representable values around
  `448*2^e` scale boundaries, `1e-4`, normalized FP8 rounding ties, and the
  normal/subnormal transition. Construct ties with a separate group maximum
  so the value under test does not itself redefine the scale.
- Check full backing storage, row/column guards, unchanged inputs and stride
  padding. Hold earlier data/scales outputs across subsequent calls, varied
  shapes and allocator churn; verify separate ownership and unchanged bytes.
- Test a nondefault stream with dependent work and a waited consumer. Warm up
  before graph capture; replay changed inputs while keeping addresses/shapes
  fixed; retain cloned results across later replays. Use independent graph pools
  for the oracle and candidate. Never reset compiler state within such a case.
- Exercise a mixed sequence of dtype/shape/stride/mode cases without production
  Dynamo. Isolate the compiled oracle's cases if its eight-entry guard cache
  would otherwise saturate; that isolation cannot mask a native candidate
  failure or replace an uninterrupted candidate sequence.
- Recheck direct `fp8_linear`, packed gate/up sharing, SiLU/down, grouped paths
  and full-model candidate outputs from independent caches. Component exactness
  does not replace new full H64K serving and affected experiment validation.

The CPU fixture builder delivered for this screen is
`/tmp/deepseek_linear_native_quantization_20261004/adversarial_fixtures.py`,
SHA-256 `3ad364c7fc2161b3e8c46c36bc2aabe7be5b3ee3eae7d87de8af97b7e7973b20`.
Its `iter_cases(suite="full", seed=20261004)` yields 115 cases with owned flat
CPU storage, logical shape/strides/offset, tags and metadata. Each fixture's
`materialize(device)` copies the complete backing storage and reconstructs the
same view; `manifest()` records its backing-byte hash. All 65536 BF16/FP16
encodings are checked both densely and individually with scale anchors at
one, two and sixteen times each finite subject magnitude. Further cases cover
threshold neighbors, all 126 finite-positive E4M3 midpoint intervals with both
signs, nonfinite payloads, subnormals, guards and the supported layout matrix.

`fixtures_cpu_audit_01.json` in the same temporary directory records a passing
115-case integrity audit with CUDA uninitialized; Ruff also passed. This is
input-fixture validation only, not quantizer correctness. Empty strided tensors
can retain a non-unit inner stride after `.contiguous()`, so compare their
logical bytes through same-size integer views or explicit contiguous storage;
blind `.contiguous().view(torch.uint8)` can fail before reaching the kernel.

## Complete-API performance gates

Measure the public quantizer call, including output allocation and all necessary
layout/copy work, against the isolated compiled official helper. Primary BF16
cases use M=121,128,1024 crossed with K=1536,7168,16384,18432; include M=1 as a
small-input check. Confirm the exact checkpoint source-0/1/2 activation samples
where available. Keep FP16/FP32 and awkward-layout correctness domains distinct
from this primary target matrix.

Report eager CPU wall and joined CUDA-event duration. Add complete graph
copy/replay/owned-output timing as a separate boundary; replay-only kernel time
is diagnostic and must not be presented as a complete allocating API. Use
matching stream dependencies, independent outputs, precompile/warmup outside
measurement, randomized/interleaved variant blocks and an independent ordering
seed. Keep all samples and disclose launch/resource metadata. Confirm a winner
under a separate exclusive window rather than selecting a one-block fluctuation.

Finally measure complete packed MLP/projection APIs and matching full serving.
Replacing the quantizer may change allocation, capture and helper behavior even
when the arithmetic is exact. Production timing must include the actual new
adapter; a standalone kernel result does not authorize formal-report replacement.

## DeepGEMM scale layout: separate future candidate

For the current SM90 recipe `(1,128,128)`,
`csrc/apis/layout.hpp::transform_sf_pair_into_required_layout` sends activation
SFA through the FP32 `(gran_mn=1,gran_k=128)` transformation. Weight SFB has
`(gran_mn=128,gran_k=128)` and only undergoes layout validation. Its allowed
SM90 layouts are contiguous or contiguous after transposing the last two
dimensions; the current wrapper supplies contiguous checkpoint scales.

`csrc/jit_kernels/impls/smxx_layout.hpp::get_mn_major_tma_aligned_tensor`
requires logical `[M,G]` FP32 SFA with exact strides `(1,align(M,4))` for its
two-dimensional no-copy return. The 16-byte TMA alignment gives four FP32
elements, not 128 rows. For grouped `[B,M,G]`, also require batch stride
`align(M,4)*G`. A different noncontiguous layout triggers an allocation and
`copy_`; ordinary contiguous row-major input triggers `transpose_fp32`.

The saved packed/shared MLP trace
`/tmp/deepseek_mlp_packing_probe_20261004/graph_bench_01_checkpoint_layer0_q128_full_mlp_graph_borrowed_packed_shared_trace.json`
contains two `transpose_fp32<512u,64u,56u,128u,57u>` launches and one with
`144u`/`145u` in the scale-width/padded-shared-width positions. The first two
transform gate/up SFA (`7168/128=56`); the third transforms down SFA
(`18432/128=144`). They are not weight-scale transformations.

Currently the gate's public GEMM internally creates its transformed SFA, while
the prepared activation retains the quantizer's original row-major scales.
The up GEMM therefore repeats the transformation. Two later candidates are
possible after the initial quantizer is validated:

1. Gate explicitly performs the official SFA layout helper inside its measured
   scope, passes the transformed tensor to its GEMM, and retains that tensor in
   the one-use prepared record. Gate/up public GEMMs then take the exact-layout
   no-copy branch. This keeps the public quantizer output ABI unchanged and
   could remove the second transformation.
2. An explicitly different internal quantizer output contract writes SFA
   directly with strides `(1,align(M,4))`. It must preserve logical scale bytes,
   initialize/protect any required padding, account for actual storage, and
   validate every consumer and borrowed lifetime. This must not silently replace
   the initial contiguous public output contract.

Neither candidate is part of the initial replacement or currently measured.
Both require fresh exactness, same-stream/graph lifetime and full-API timing
evidence. Do not change the two independent weight tensors or fold gate/up GEMMs.

## Aligned BF16 unsigned reduction candidate

The v3 prototype changes only the aligned BF16 group's maximum reduction.
It retains each original FP32 value for normalization and converts its bits
to an unsigned magnitude with `bits & 0x7fffffff`. Lane-local unsigned maxima
and `__reduce_max_sync(0xffffffffu, maximum_bits)` replace the branch-heavy
NaN-propagating float reduction. The floor is also an unsigned maximum with
the bits of the same FP32 `1e-4` constant. All remaining arithmetic is unchanged.

For nonnegative finite IEEE FP32 values, unsigned encoding order equals numeric
order, including subnormals and positive zero. Clearing the sign makes both
zero signs reduce identically while the original negative-zero operand remains
available for output conversion. Positive infinity exceeds every finite
encoding. Every NaN magnitude encoding exceeds infinity, so the unsigned
reduction selects a NaN whenever the original reduction propagates a NaN.

The selected NaN's sign, signaling state and payload need not match the old
maximum. They do not affect this quantizer's output scale: after the unchanged
multiply by `0x3b124925`, every NaN has exponent 255 and nonzero mantissa;
the manual exponent increment/clamp produces 254. Infinity also clamps to 254.
Both cases therefore produce scale bits `0x7f000000` (`2^127`) and the same
subnormal reciprocal `2^-127`. NaN selection never replaces the original four
normalization operands. Their signed-zero, finite, infinity and NaN conversion
paths still execute the same multiply and SATFINITE instruction.

The full-warp mask is valid only while every lane in a warp processes the same
complete 128-value group. The current aligned dispatch has no partial K groups,
and its tail returns are uniform across the warp. Do not extend this proof to
a lane-divergent implementation without revisiting the active mask.

CPU proof script:
`/tmp/deepseek_linear_redux_review_20261004/prove_reduction.py`.
Its `scale_equivalence_01.json` has SHA256
`809e816e8337d6eeefb821e5abd3fff537264f632d31fa373c9d36ec711ce70f`.
It checks all 65,536 BF16 encodings, 983,040 pairs with signed zeros,
subnormals, finite anchors, infinities, signaling NaNs and quiet NaNs, and
1,242,408 full groups from the 63 aligned fixtures. All derived scale bits
match. This is CPU evidence only; the actual PTX must show unsigned `redux`
and preserve multiply, reciprocal and conversion instructions, followed by a
fresh 213-fixture/8-lifecycle GPU screen and complete-API timing.
