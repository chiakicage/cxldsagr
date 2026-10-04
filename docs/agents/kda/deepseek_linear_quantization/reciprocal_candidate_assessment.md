# Optional reciprocal construction after v3

Status: offline assessment only, requested by root while the v3 screen runs.
No candidate CUDA source, compilation or GPU execution. Keep v3 arithmetic
unchanged until its timing decision; this is a possible later v4 direction.

The scale is a positive FP32 power of two with encoded exponent `e`. Its exact
reciprocal can be represented by `(254-e) << 23` when `e <= 253`, or by
`0x00400000` when `e == 254`. The latter is subnormal `2^-127`, so preserving
no-FTZ arithmetic remains necessary. This construction also works for the full
clamped exponent domain 1–254, although the quantizer's floor makes small
exponents unreachable.

With the current `1e-4` floor and reciprocal-448 constant, finite BF16 inputs
produce exponents 105–247. NaN or infinity produces exponent 254. Exponents
248–253 are not reached by current finite input values, but the requested
105–254 oracle matrix should retain them as a useful superset: 150 scales.

The actual v3 SM90 binary's first group contains nine reciprocal-specific SASS
instructions: a quarter constant, two range predicates, one factor selection,
three predicated rescaling multiplies, `MUFU.RCP`, and one final multiply.
Equivalent sequences recur in the unrolled groups. For exponent 254, the code
rescales the denominator by one quarter to `2^125`, computes its reciprocal,
then multiplies by one quarter to obtain `2^-127`. The lower-range rescaling
predicates are false for this quantizer's positive normal scales. This is an
instruction attribution, not a measured cycle or latency saving.

The saved evidence is
`/tmp/deepseek_linear_redux_review_20261004/reciprocal_assessment_01.json`.
It records the actual v3 binary/SASS hashes, all 150 constructed inverse words,
and their equality to CPU IEEE FP32 division. CPU division does not certify
actual `div.full.f32` behavior: the emitted `MUFU.RCP` is the remaining
hardware-specific equivalence question.

Before implementing a new quantizer candidate, obtain a root-scheduled GPU
oracle witness with the same SM90 target and no-FTZ/no-FMA build flags. Supply
all scales at runtime and use the same inline `div.full.f32` with literal
FP32 numerator one as v3. Retain both the oracle inverse words and constructed
words. Inspect the generated
PTX/SASS to confirm a real divide sequence rather than constant folding.
Require equality for every one of the 150 scale words, including both normal
and subnormal reciprocal endpoints. Include unchanged normalization and
SATFINITE conversion of signed zeros, BF16 subnormals, finite values, infinities
and NaNs; nonfinite groups must retain the exponent-254 route.

Only after that witness passes may a separate candidate replace the aligned
path's reciprocal. Preserve the generic fallback, reduction, original input
values, FP32 normalization multiply and packed SATFINITE conversion. Run a new
213-fixture/8-lifecycle screen and all complete-API timing boundaries. Compare
against both the accepted baseline and v3 under matching conditions; a smaller
instruction sequence alone does not justify promotion.
