# Native linear quantization draft

The existing wrapper compiles DeepGEMM's per-token helper dynamically. Its
arithmetic is documented by both pinned Python source and historical generated
PTX, but a new isolated oracle must identify its own generated code. The C8
freeze preserves the current packed/shared MLP and all other preceding changes.

The first candidate uses one warp per 128-element group and four warps per
CTA. Each lane loads four values, respecting general row/column strides and
masking K tails before any load. Warp shuffle reduction propagates NaNs. Lane
zero writes the contiguous scale; lanes write only valid FP8 columns. Inputs
stay unchanged. BF16/FP16/FP32 share templated native arithmetic from the outset.
The wrapper allocates outputs on the caller device and launches on its current
stream. Empty M allocates empty outputs without a launch or native load.

The main numerical risks are reciprocal-versus-division rounding at scale
boundaries, subnormal reciprocals for nonfinite groups, signed zeros, and NaN
conversion. Explicit `mul.rn`, the exact `0x3b124925` constant and `div.full.f32`
avoid guessing what native lowering will do. The inspected compiled oracle
uses `cvt.rn.satfinite.e4m3x2.f32`; validate its actual output bytes rather than
relying on an eager PyTorch cast. No fast-math or FTZ option is permitted.

Other risks are unmasked tail reads, stride/address arithmetic overflow,
FFI stream mismatch, premature input/output lifetime, graph capture allocating
outside the intended pool, and source fingerprint omissions. Inspect generated
PTX and native build metadata offline before requesting correctness time.

Ranked follow-up candidates, each after initial exactness:

1. Compare four versus eight warps per CTA and one versus multiple independent
   groups per warp. K18432/M1024 has 147456 groups, so launch packing can matter.
2. Add vector loads and packed stores for aligned unit-inner-stride BF16 while
   retaining the scalar generic kernel. Validate storage-offset and tail dispatch.
3. Consider additional aligned specialization only when complete API cost,
   not kernel-only time, identifies a remaining benefit.

Direct transformed-scale emission or reuse of DeepGEMM's transformed SFA is a
separate future task. The first replacement retains contiguous public scales
and both current gate/up scale transformations.

First create a temporary module, `.cu` source and driver with `prepare`,
`screen` and `bench` subcommands. `prepare` records sources and may compile
offline with explicit SM90 architecture, with CUDA hidden and uninitialized.
`screen` compares all bytes against an isolated compiled official helper,
including exceptional and strided inputs, streams, ownership and graphs.
`bench` refuses missing/mismatched correctness evidence and measures complete
allocating APIs and separately owned graph APIs. Exact commands, matrix and
evidence gates follow in the executable plan. No live dispatch changes occur
before review and measured selection.
