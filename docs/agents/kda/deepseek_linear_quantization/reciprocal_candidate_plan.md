# Exact reciprocal candidate and transitive screen

Parent: V3 aligned vector/REDUX candidate. V3 passed all 213 fixtures and eight
lifecycle cases directly against the compiled official helper, but its measured
Graph API still regressed at the two largest M1024 widths. `bench_01` SHA-256 is
`332181e3ce49bd98de3440e8763e1f16e2809eaed82fcffea8e83f60aa08ab82`.
At K18432, the owned-Graph paired speedup is 0.929x and borrowed replay is
22.736 us versus 20.592 us. V3 is numerically accepted, not performance-promoted.

V4 lives in `/tmp/deepseek_linear_native_quantization_v4_20261004/`. Change only
the aligned kernel's reciprocal of its exact UE8M0 scale. Preserve REDUX,
reciprocal-448 multiplication, exponent ceiling/clamp, per-value multiplication,
SATFINITE packing, launch geometry, dispatch and the generic fallback.

For scale exponent bits e, scale is `2^(e-127)`, so its reciprocal is
`2^(127-e)`. At e <= 253, this is normal FP32 with raw bits `(254-e)<<23`.
At e=254 the result is the subnormal `2^-127`, raw bits `0x00400000`.
The actual minimum scale exponent from the `1e-4` floor is 105. The direct
witness will cover every exponent 105 through 254, including exponents not
normally reached by finite BF16 activations. No approximate reciprocal or FTZ
mode is introduced. Original values still multiply by exactly the same FP32
inverse before conversion.

## Direct instruction witness

Keep `reciprocal_full` and the candidate `inverse_from_exponent` as distinct
device helpers. A separate engineering `.cu` witness includes the candidate
source to use these exact helper definitions, while the production-candidate
JIT continues to compile only `activation_quantization.cu`.
The witness takes a runtime exponent tensor, preventing constant-folding of
the tested division. It persists all 150 old/new inverse words and compares
normalized FP32 bits and SATFINITE bytes for all 65536 BF16 encodings crossed
with all 150 exponents. Signed zero, subnormal, extrema, Inf and signed/payload
NaNs are therefore included. Inspect actual PTX/SASS for `div.full`, the new
bit construction, multiplication and conversion. This is direct compiled CUDA
instruction evidence, not a fresh full official-quantizer screen.

## Transitive full quantizer screen

Import the immutable V3 wrapper from its original path under a unique Python
module name. Do not copy or rebuild it under a new source identity. Bind all
V3 source/header/native-binary hashes to its accepted direct official screen:
`/tmp/deepseek_linear_native_quantization_v3_20261004/screen_01.json`, SHA-256
`18d502190bee89e9655c086115601562e6aa578f5cb311b98514b06c0f39cab0`.
Use the exact same 213 fixture manifests and unchanged eight lifecycle input
definitions. Current candidate and frozen V3 receive identical tensors in every
comparison, including guards, streams, changed-input graphs and retained outputs.
Label the new result `oracle_kind=frozen_native_parent_transitive` and link the
full accepted official chain. Do not claim a fresh official compile per fixture.

The old parent source and all its artifacts remain unchanged. The new direct
witness and full transitive screen need independent review and a root GPU grant
after offline compilation. Both must pass before timing. Continue to use the
fresh original C8 public API as the 16-shape benchmark baseline; its cost is
not replaced by V3 timing. An accepted candidate needs a second timing window
with a different order seed, then final production-loader/consumer validation.
