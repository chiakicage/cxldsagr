# Integer reduction candidate after aligned vector timing

Parent: V2 aligned vector4 / four-groups-per-warp candidate. Its full 213-fixture
and eight-lifecycle correctness screen passed, but large Graph API timing still
regressed. `bench_01` completed 16 shapes and 4800 pairs with no correctness
failure. At M1024/K18432, native/original medians are 65.776/56.000 us for owned
Graph API and 29.072/20.544 us for borrowed replay. V2 is not promoted.

V2 PTX contains repeated branching NaN-preserving FP32 maxima. The next
candidate changes only the aligned kernel's reduction to unsigned absolute
FP32-bit maxima and `__reduce_max_sync`; vector loads/stores, launch geometry,
dispatch, generic fallback and all scale/product/conversion arithmetic stay
unchanged. Files live in `/tmp/deepseek_linear_native_quantization_v3_20261004/`;
V1/V2 source and results remain immutable.

## Equivalence argument

Every reduction operand is the absolute value of a BF16 input widened exactly
to FP32, or a nonnegative initial/floor value. Clearing the sign bit produces
an unsigned word in `[0,0x7fffffff]`. For finite nonnegative FP32 numbers and
positive infinity, unsigned word order is numerical order. Positive and negative
zero both reduce to positive zero; their original signed values are retained
separately for normalization and conversion.

Every absolute NaN word is greater than the positive-infinity word. Therefore
unsigned maximum selects a NaN whenever any input is NaN. It may select a
different NaN payload than the former floating reduction, but every such
payload stays NaN after the reciprocal-448 multiplication. The official exponent
ceiling then has exponent 255 and a nonzero mantissa, and clamps to 254. Thus
every NaN maximum produces exactly scale `2^127`, independent of payload/sign.
An infinity maximum also clamps to that same scale. No original input value
is replaced by its absolute bits for the final FP8 conversion.

The floor is unsigned max against the exact FP32 representation of `1e-4f`.
After warp integer reduction, reinterpret that maximum as FP32 and retain the
same reciprocal-448 constant, `mul.rn`, exponent/mantissa ceiling/clamp,
`div.full.f32`, per-value `mul.rn` and SATFINITE packed conversion. This proof
does not authorize changing CPU reference arithmetic or generic FP16/FP32 paths.

## Execution and gates

Implement after this plan, then compile offline and inspect actual
`redux.sync.max.u32` PTX/SASS, floor bits, register/spill usage and unchanged
local dependency identity. Independently inspect the signed-zero/NaN argument
and the source diff. A fresh root grant is required for the entire 213-fixture
screen plus eight lifecycle cases, including the fastpath and unaligned-output
fallback. Compare all FP8/scale bytes, source identity and ownership.

Only after independent acceptance, run the same 16-shape paired benchmark with
10 warmups and 50 pairs. Keep complete eager/owned Graph APIs and borrowed
diagnostic separate. A selected improvement still needs a different-seed
confirmation window and actual production-loader/consumer validation before
integration or formal performance publication.
