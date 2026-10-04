# DeepSeek typed norm I/O

`api.py` provides BF16 norm I/O around the official FlashInfer Float32 CuTe
arithmetic on SM90. The model adapter selects nonempty rank-2 inputs with
width 512/1536/7168 for plain norm or 7168 for fused norm, aligned FP32 weights,
and aligned contiguous or `width + 64` row-strided inputs. The real MLA latent
view uses width 512 and stride 576. Strided input packing keeps BF16 dtype.
Other model inputs retain the ordinary `nonmatrix.py` path; direct unsupported
operator calls fail.

Plain and fused classes inherit official Float32 constructors. The local
plain kernel retains the inherited launch; the fused launch adds independent
normalized/saved output pointers. Operand-specific copies transfer four logical
values, and compile-time checks preserve complete Float32 TV/register layouts.
The official reduction helper and scalar operation order stay unchanged.
Fused normalization consumes the unrounded FP32 sum and rounds the saved output
separately. Every ordinary call owns its outputs and uses the current stream.

Importing the public API does not load CuTe or initialize CUDA. Eligible calls
load compilation lazily. `_fingerprint.build_info()` records local sources,
resolved vendor helpers, CuTe/TVM FFI sources and headers, the actual loaded
CTK-flavor native extension, and software versions. The process memoizer includes
that fingerprint and the exact specialization signature. A compile-scoped
hook adds the fingerprint to live IR. Direct `cute.compile` currently bypasses
its IR cache, so this is process caching and IR provenance, not a disk-cache
claim. Source changes require a fresh process; run provenance checks reject
changes after the first identity snapshot.

The focused tests compare integrated outputs bitwise with an independent direct
official FP32 oracle, check full caller storage, retained outputs, nondefault
streams, graph replay and fallback dispatch. GPU acceptance requires available
SM90 hardware; the global GPU entry verifies that before collecting tests.
Current implementation and measurement boundaries are recorded in
[the integration plan](../../../docs/agents/kda/deepseek_norm_io/integration_plan.md)
and [checkpoint](../../../docs/agents/kda/deepseek_norm_io/checkpoint.md).
