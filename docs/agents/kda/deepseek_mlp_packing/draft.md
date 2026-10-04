# Dense MLP packing draft

The baseline `CheckpointMLP.__call__` executes gate and up through separate
`CheckpointLinear` instances. Each `fp8_linear` validates/pads its own weight,
quantizes the identical hidden activation with the compiled official helper,
allocates BF16 output and calls DeepGEMM. `silu_mul` concatenates gate/up before
calling FlashInfer; down remains another independent FP8 linear.

The historical C3 graph analysis reports 25.216 ms of concatenation and several
cast/copy groups in a full serial-sparse cold request. That is the sum across
its graph groups, not a current isolated dense-MLP or removable-cat measurement.
It motivates inspection but supplies neither a current baseline nor an expected
speedup. The running C7a+hint formal trace must remain undisturbed.

## Source feasibility

Pinned DeepGEMM is `057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7`.
The SM90 1D2D route supports the proposed output layout by source:

- `csrc/utils/layout.hpp` lines 19-39 accepts a rank-two output with inner
  stride one; it does not require `stride(0)==N`.
- `csrc/apis/gemm.hpp` lines 76-137 validates logical `[M,N]` output and routes
  FP32 scaling factors/recipe `(1,128,128)` to `sm90_fp8_gemm_1d2d`.
- `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp` lines 79-135 builds the same
  logical GemmDesc and passes output N and actual row stride into the descriptor.
  GemmDesc/config/JIT specialization has no output-stride field.
- `runtime_utils.hpp` lines 119-164 and 367-381 encodes the actual view pointer,
  logical N/M and row pitch in a rank-two TMA descriptor. Pointer and pitch must
  be aligned to 16 bytes.
- `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh` lines 421-440 uses
  that descriptor in actual `SM90_TMA_STORE_2D` instructions.

For N=18432, a contiguous BF16 `[Q,36864]` allocation gives each half shape
`[Q,18432]`, row stride 36864 elements/73728 bytes, and the second half starts
36864 bytes from the base. Both pointers and row pitch satisfy the descriptor
alignment. Each descriptor's logical N remains 18432, so its bounds describe
one half, even though the backing allocation is wider. Boundary guards still
require GPU verification; a source-accepted descriptor is not a numerical pass.

The FlashInfer activation kernel addresses `token*2*N + column` and `+N` within
the same contiguous input. It is therefore the same packed layout produced by
baseline `torch.cat`. Its scalar FP32 SiLU/multiply and BF16 store stay unchanged.

## Candidate and controls

The first candidate retains both official GEMMs and one shared compiled
quantization. A control retains two quantizations while using the packed stores;
this separates layout savings from quantization reuse. A second control shares
quantization but retains separate output allocations and baseline cat. No fused
or larger GEMM is part of this investigation.

The temporary correctness harness will compare the baseline, both controls and
the combined candidate. It will check real-checkpoint source layers 0-2 when
`--checkpoint /preset-models` is selected, plus small synthetic edge tiles.
The baseline invokes production `fp8_linear` and `silu_mul`; down invokes the
same production function in every path. Inputs and all independent weights are
shared read-only between comparison paths and snapshotted for mutation checks.

Risks: descriptor bounds or alignment at a half offset; hidden extra contiguous
copies; reuse of quantized storage before the second GEMM; shape/padding changes;
allocator lifetime under non-default streams and graphs; higher live memory
from returning packed views; and loss/double-counting of matrix API attribution.
The initial prototype returns owned SiLU/MLP outputs and keeps packed views only
for explicit untimed inspection. Captured graph pools must own their intermediates
and preserve prior output clones across later replays.

## First sequence

Write the executable plan before the temporary adapter. Run CPU prepare/syntax
checks with CUDA uninitialized. A later root grant permits descriptor/guard and
byte-exact screening first. Only an accepted screen permits fresh-run timing.
Include the full helper boundary and unchanged down call, and inspect actual
kernel launches to verify cat/remnant output copies are absent in packed paths.
Reject on any mismatch, hidden packing copy or unexplained changed GEMM kernel.
