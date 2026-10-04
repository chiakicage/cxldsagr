# Norm I/O draft

## What failed, and why a constructor subclass is insufficient

The installed RMSNorm constructor selects Float32 vec_size=4 for the current
widths. Its kernel creates a sync CopyAtom using `mX.element_type`, then reuses
that atom for FP32 weights. One `copy_bits` value also controls input, weight
and output copies. This couples physical bit width to the logical element
partition when the tensor types differ.

The unchanged BF16-input/FP32-output screen failed at the normalization
multiplication with incompatible operand profiles `((4, 2), 8)`. Changing only
copy_bits from 128 to 64 reached an incompatible copy-predicate shape:
expected `((2,14),(1,1))`, actual `(14,(1,1))`. These are compiler rejections,
not approximate numerical successes. The direct all-FP32 control matched all
64,512 first-screen values exactly.

Both vendor plain and fused kernels are monolithic `@cute.kernel` methods.
They expose no overridable load/store/predicate helper. There is no clean
subclass that merely overrides an atom factory while calling the remaining
unmodified kernel body. Changing fake tensor shape/type does not supply the
missing conversion or preserve the canonical reduction partition. Runtime
monkeypatching or AST-rewriting the installed method would hide the source
delta and is excluded.

A plain local subclass can still reuse the vendor constructor and inherited
launch method while overriding only the kernel method. A fused owned-output
adapter needs a small local launch method as well, because the official ABI
only exposes in-place mX/mR outputs. This avoids forking the whole vendor module,
but it is a local kernel specialization, not an unmodified-vendor call.

## Preferred local specialization

Keep the Float32 constructor and its logical TV layout. Give each operand a
copy atom whose **element count** equals the baseline vec_size: BF16 X uses
64 bits, FP32 W uses 128 bits, BF16 Y uses 64 bits, and diagnostic FP32 Y uses
128 bits when vec_size=4. Do not globally replace 128 by 64.

Derive each load/store predicate from that operand's own partitioned logical
coordinates. Explicitly align/re-tile the loaded register fragments into the
baseline FP32 register shape before reduction; verify the shape, stride and
logical element order at compile time. Equal numbers of values per thread are
insufficient because the vendor's `TensorSSA.reduce(..., reduction_profile=0)`
also depends on the nested register shape/order. If matching element-count
copies already produce the canonical shape, avoid extra retiling; establish
that from the actual layouts rather than assumption.

Import the installed `row_reduce_sum_multirow` unchanged. Retain its intra-thread
reduction, warp shuffle, block/cluster reduction and buffer organization.
Keep the scalar sequence `x*x -> sum/H -> rsqrt(mean+eps) -> x*rstd*(w+bias)`.
The short scalar sequence must be expressed locally because the vendor does
not export it separately. Preserve default compiler options and inspect actual
IR/PTX when available; source-level resemblance alone is not exactness proof.

For fused norm, widen BF16 X and R in registers and compute `h = x + r` in
FP32. Use this same unrounded h for squaring, reduction and normalization.
Store a separately rounded BF16 residual to an independent owned output;
never reload that rounded residual to normalize. Diagnostic specializations
must expose FP32 h and FP32 normalized output. The ordinary fused variant
returns independent BF16 normalized/residual allocations.

Keep the constructor's launch shared-memory budget initially, even if typed
BF16 shared tiles need fewer bytes. Changing occupancy or cluster sizing is
not part of the first correctness candidate. For the current widths, the
expected Float32 geometry is:

| Width | vec_size | Threads/row | Rows/CTA | Vector blocks | Plain/fused shared bytes from baseline formula |
|---|---:|---:|---:|---:|---:|
| 512 | 4 | 32 | 4 | 4 | 8,208 / 16,400 |
| 1536 | 4 | 32 | 4 | 12 | 24,592 / 49,168 |
| 7168 | 4 | 128 | 1 | 14 | 28,688 / 57,360 |

These are CPU derivations for cluster_n=1 and async copy enabled. Assert the
actual constructor choices against the scheduled GPU before using them.

## Simpler vendor-only alternatives

**Packed fused output.** Allocate one FP32 `[2,Q,D]` buffer, widen X and R
into its two disjoint contiguous halves, invoke the existing official fused
kernel on those halves, then convert the whole mutated buffer to BF16 once.
Return two disjoint BF16 views. This preserves the official compute kernel
and should remove one output conversion launch per fused norm. It does not
remove FP32 activation storage or nominal tensor traffic.

Using `torch.cat((x,r), out=FP32_buffer)` does not fuse the two input widening
copies in the installed PyTorch. At commit
`7269437d655783a26cba32aa88195b741ff496aa`, TensorShape.cpp compares input dtypes
against the specified output dtype; the CUDA cat fast paths require
`all_same_dtype`. The BF16-to-FP32 case takes a loop over
`narrow(result).copy_(input)`. Two explicit widening `copy_` calls therefore
make the real work clearer. CPU tests confirmed widening cat produces exact
FP32 values, but do not establish its CUDA performance.

The BF16 output halves share one complete allocation. Retaining a residual
can retain the normalized half after its consumer finishes. In graph v2 this
can increase projection/finish private-pool liveness; it is not a change to
the static graph input formula. Account the entire unique backing storage,
test independent view mutation and retained output lifetime, and reject the
candidate if added residency or complete API latency outweighs the saved node.

**Plain output-buffer reuse.** The public plain API accepts `out`. For BF16
input, its widened FP32 activation is already owned. Reusing that activation
as FP32 out could remove one FP32 allocation, while retaining both conversion
launches and all nominal tensor traffic. Do not do this for arbitrary FP32
input, where `.float()` aliases the caller. The inspected cluster_n=1 kernel
loads its row before writing output, but in-place alias support still requires
GPU exactness/lifetime validation. This is a capacity candidate, not a promised
latency optimization.

**Out-only mixed compilation.** Float32 X/W with BF16 Y is a narrower cheap
screen that was not exercised by the previous BF16-input rejection. It may
still fail the coupled output atom/predicate layout. If exact, it could remove
the final plain cast while retaining input widening; do not assume support.

`torch.add(BF16,BF16,out=FP32)` is not an unrounded-sum substitute. A CPU check
already returned 1 for `1 + 2^-8` and 1024 for `1024 + 2^-10`; the required
FP32 sums are 1.00390625 and 1024.0009765625. FP8/FP4 norm-quantization endpoints
also have different output semantics and are outside this exact BF16 task.

## Expected launch and byte changes

Let N=Q*D. The following are nominal activation tensor reads+writes, excluding
unchanged FP32 weight traffic, padding/cache-line effects, temporary workspace,
compiler spills and cache reuse. They are not measured HBM/DRAM traffic.

| Case | Baseline activation bytes | Direct mixed bytes | Potential bytes removed | Conversion kernels removed |
|---|---:|---:|---:|---:|
| Plain BF16 norm | 6N input cast + 8N FP32 norm + 6N output cast = 20N | 4N | 16N | 2 |
| Fused BF16 X/R, BF16 Y/saved | 12N input casts + 16N FP32 norm + 12N output casts = 40N | 8N | 32N | 4 |
| Packed official fused norm | 40N | 40N | 0 | 1 |

At Q=1024, plain widths 512/1536/7168 could remove 8/24/112 MiB respectively;
a fused width-7168 call could remove 224 MiB. Each baseline call materializes
8N logical FP32 intermediate bytes: 4/12/56 MiB at these widths. Summing these
allocations across calls is not their peak lifetime or allocator reservation.

The serving block uses plain query/KV norms at 1536/512, input norm at 7168
(plain without incoming residual, fused otherwise), and fused post-attention
norm at 7168. With a direct row-strided KV implementation:

| Q=1024 block | Potential nominal I/O removed | Conversion kernel nodes removed | Packed-vendor-only nodes removed |
|---|---:|---:|---:|
| No incoming residual; source-0 copies | 368 MiB | 10 | 1 |
| Incoming residual; source-1/2 copies | 480 MiB | 12 | 2 |

There are four no-residual and six residual-present blocks in the ten-copy
workload: the full mixed upper model is 4,352 MiB and 112 conversion kernel
nodes per Q1024 chunk; packing alone removes 16 nodes with no nominal byte
reduction. These totals exclude final norm and indexer LayerNorm. CUDA Graph
host replay launch counts do not fall by the number of removed kernel nodes.

If the first local kernel packs the strided KV latent into BF16 contiguous
storage, subtract one saved node and 2 MiB per block from the mixed estimates
(366/478 MiB and 9/11 nodes). Its 4N packing traffic and allocation belong to
the complete API measurement. Direct row-strided support is a separate
verified specialization, not a free assumption.
