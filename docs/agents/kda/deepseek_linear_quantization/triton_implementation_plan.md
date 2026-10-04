# Handwritten Triton quantizer draft and execution plan

The user explicitly changed the preferred kernel language to Triton on
2026-10-04. Continue the existing MFU objective and avoid production
`torch.compile`. This supersedes the native implementation choice in the
earlier draft and executable plan. Native evidence remains an engineering
record; no native candidate was promoted.

## Draft

Start from the arithmetic and efficient reduction geometry of the actual
compiled pinned DeepGEMM oracle, then express it as a direct local `triton.jit`
kernel. Keep independently owned contiguous FP8 data and FP32 scale outputs,
all supported dtypes/strides/tails and active-stream/Graph semantics. Do not
change linear recipes, indexer quantization, model weights or cache behavior.

The main risk is rounding and nonfinite behavior: NaN-propagating reduction,
the exact FP32 reciprocal of 448, exponent ceiling/clamping, reciprocal/product
and SATFINITE E4M3 conversion must retain every output bit. Use the original
arithmetic for the first Triton candidate. Exact reciprocal-bit construction
can be considered separately if measurements justify it; unexecuted native V4
preparation does not validate the new Triton code.

Use a single candidate and bounded primary launch geometry first. Avoid adding
autotuning startup or hidden compiler paths to production. Preserve lazy CPU
imports and reject non-SM90 CUDA devices. Record stable source/compiler identity
separately from live PTX/cubin specializations.

## Execution gates

1. Inspect the frozen official helper and its generated Triton/PTX; write the
   candidate-specific launch/arithmetic plan before implementing it.
2. Build the temporary direct Triton wrapper/kernel, CPU-safe identity and
   harness; independently review arithmetic, masks, layouts and launch options.
3. Run exactness before timing. Reuse the 213 fixtures and eight lifecycle
   input sequences from accepted native V3. A transitive screen must pin V3's
   accepted official-screen SHA, original wrapper/native binary, source hashes,
   driver and fixture hashes, and unchanged lifecycle definitions. Label it
   `frozen_native_parent_transitive`; do not claim fresh official compilation.
   New integrated public regression cases additionally use the pinned compiled
   official helper directly.
4. After the correctness audit, run the 16-shape complete-API paired benchmark
   against the immutable C8 public helper. Keep eager allocation, Graph input
   copy/replay/owned outputs, and borrowed replay diagnostic boundaries distinct.
   Save raw paired observations, warmups, repeats, order seed and live artifacts.
   Use a second quiet window with another order seed to confirm a promotable
   candidate. Root serializes all CUDA jobs during performance windows.
5. Integrate the selected direct Triton implementation; remove production
   `torch.compile` quantization and its obsolete metadata. Test public dispatch,
   dense/grouped/packed MLP consumers, actual checkpoint outputs, stream/lifetime
   behavior and four-scheme H64K correctness under the new production identity.
6. Freeze the implementation and run the full 16-user, two-round four-scheme
   trajectory plus matching matrix API profiling. Continue MFU optimization from
   those measurements, then update the official ECHO experiment. Publish accepted
   replacements before cleaning their superseded reports and backing outputs.

The component promotion criterion remains exact output/lifetime behavior and
reproducibly improved or preserved complete API performance on target shapes,
with no material Graph regression hidden by host overhead. Component timing
does not establish a full-model latency or MFU improvement.
