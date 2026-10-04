# Direct Triton candidate T1

Implementation begins after this plan on 2026-10-04. Temporary sources live in
`/tmp/deepseek_linear_triton_quantization_v1_20261004/`; production is unchanged.
The execution and promotion gates are those in `triton_implementation_plan.md`.

Use one directly launched `triton.jit` kernel with a `[32, 128]` tile of flattened
row/group IDs, two warps and one stage. The installed compiled oracle's primary
contiguous shape uses this geometry. Specialize contiguous full-group addressing
to flattened offsets; retain the same arithmetic for general strides and K tails.
Input masks provide logical zero padding, output masks write only logical K, and
the wrapper allocates independent contiguous data and row-major FP32 scales.

Keep the original arithmetic: NaN-propagating absolute maximum and floor,
multiply by FP32 bits `0x3b124925`, integer UE8M0 ceiling/clamp, explicit
`div.full.f32`, FP32 product and SATFINITE E4M3 conversion. Plain `tl.max` is
unsuitable because installed Triton suppresses NaNs; use `tl.reduce` with a
custom `tl.maximum(..., propagate_nan=tl.PropagateNan.ALL)` combine. Do not
integrate the native V4 unexecuted reciprocal optimization or an SFA layout change.

Bind the transitive correctness oracle to accepted V3 screen SHA
`18d502190bee89e9655c086115601562e6aa578f5cb311b98514b06c0f39cab0`, including
the original driver/fixtures, binary, dependencies and official artifacts.
Reuse all 213 fixture manifests and the unchanged eight lifecycle cases.
Guarded-output checks call the same Triton launch with deliberately offset
output storage. The complete C8 public compiled helper remains the fresh
performance baseline for all 16 shapes and all three timing boundaries.

`quantize`, CPU-safe stable `build_info`, and observed-only `runtime_info` are
the production-facing interface. Root owns the provenance implementation.
No CUDA execution is allowed before an explicit root scheduling grant.
