# Aligned vector candidate after scalar baseline

Parent: accepted numerical candidate `warp1_cta4`, native source
`cc48e83b961c4ed03e858d48625eea41500fe85d0e142aaf14519ebc49e6c028`.
The parent is not promoted: `bench_01` passed all measurement checks but its
M1024/K18432 graph-owned median is 84.016 us versus 54.592 us for the original
public helper; borrowed replay is 47.552 us versus 19.008 us. These are component
API measurements. Result SHA-256 is
`e3443f85be79021c53c34851fb606b7c171059a0eefff80dc86cef2d14725f4a`.
The native generic kernel has 30 registers and no local/shared storage. Its
M1024/K18432 launch has 36864 CTAs, scalar loads/stores and runtime 64-bit group
division. The observed graph regression justifies a narrow aligned candidate.

Create the candidate in
`/tmp/deepseek_linear_native_quantization_v2_20261004/`, leaving all parent
source/results unchanged. The candidate keeps the generic kernel intact and
adds one fast domain: nonempty BF16 contiguous `[M,K]` with canonical row stride K, K in
1536/7168/16384/18432, input pointer aligned to eight bytes and output pointer
aligned to four bytes. Other dtypes, widths, pitched/column/zero strides,
misaligned offsets and partial groups retain the accepted generic kernel.
Output layout remains contiguous FP8 plus contiguous FP32 scales.

Each warp processes four independent 128-element groups sequentially; a CTA
contains four warps and handles sixteen groups. Each lane loads four adjacent
BF16 values through one `uint2`, widens by exact bit shifts, and writes four
FP8 bytes through one `uint32_t`. Packed conversion uses the native SATFINITE
float2 intrinsic with separately verified low/high byte order. Reduction,
scale rounding and reciprocal/product arithmetic match the parent. Logical
groups are contiguous in flattened input/output space, removing device-side
row/K division entirely. Compile-time width specialization is unnecessary for
that arithmetic; the host still restricts this first candidate to primary K.

Before GPU work, compile offline, verify vector loads/stores and packed
conversion PTX, record actual registers/no spills and source/include/binary
identity. The expanded exact screen preserves all parent fixtures and adds the
sixteen aligned target shapes, aligned nonfinite/rounding fixtures and dispatch
boundaries. Guarded outputs must invoke the same aligned kernel. Independently
review the dispatch and packed byte order. Only then request a correctness grant.

After exactness and audit, repeat the complete paired benchmark unchanged in
meaning: all sixteen BF16 primary shapes, 10 warmups, 50 paired repetitions,
eager allocation and graph copy/replay/owned outputs, with borrowed replay as a
separate diagnostic. Keep source/binary/complete-screen gates and inspect every
shape. A candidate still regressing meaningful graph cases requires another
iteration. A selected candidate needs a separate confirmation window and new
production validation before integration or formal-report replacement.
