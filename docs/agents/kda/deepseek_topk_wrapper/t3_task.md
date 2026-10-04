# T3 task: bounded-neighbor tie repair with long-row fallback

Status: prototype GPU/API gates passed; production integration selected,
2026-10-04. Root rejected
[T2](t2_checkpoint.md) for large/tail causal API regressions, then requested
one sort-free short-run candidate. Temporary sources and outputs belong under
`/tmp/deepseek_topk_order_t3_20261004/`. Root subsequently authorized the
scoped [production integration plan](t3_integration_plan.md); its public API
and complete-model gates remain separate. Published experiment replacement
is outside this prototype task.

Keep the installed official call with `sorted_output=True`,
`deterministic=False`, SMALL and `dsa_graph_safe=False`. SMALL retains the same
exact deterministic selected set, and the official stable ordered-value sort
leaves every value bit in its final position. T3 changes only index repair
and nonfinite masking. The FP32-only SM90 contract, K1..2048, Q0, K>N,
noncontiguous scores, fresh outputs, stream and graph semantics remain C5's.

The first candidate uses a fixed short-run threshold of four. A lightweight
Triton kernel uses one CTA per output row. A row has a long finite run iff
some valid position and the position four columns later have the same finite
FP32 bits. Each row writes one fresh int32 flag. A flagged row leaves all IDs
untouched for the second kernel. For an unflagged row, each finite item reads
at most three neighbors on each side, counts matching bits to its left and
matching IDs smaller than its own, then scatters its ID to
`position - equal_left_count + smaller_equal_id_count`. IDs whose destination
is unchanged need no store. Nonfinite IDs become -1; values are read-only.
All needed ID reads complete at a CTA barrier before in-place stores.

The fast kernel must contain no `tl.sort` or prefix scan. A second kernel
launches the same row grid, reads the flag first and uses the existing exact
uint32 run/index general repair only on flagged rows. It retains the adjacent
inversion test so long runs already in ascending ID order need only masking.
Every supported run length therefore has exact semantics, independent of the
observed real sample maximum of three.

Use the same packing guard as T2:
`(min(K,N)-1).bit_length() + (N-1).bit_length() <= 32`. Unsupported geometries
call the complete existing C5 API before any selection. Signed zeros are
distinct bit groups. Nonfinite values never participate in short-run ranking
or long-run detection. Arbitrary NaN model logits remain outside the producer
contract, but finalizer tests preserve every NaN value bit.

The only new tensor scratch is a fresh Q-element int32 flag vector, 4Q bytes,
allocated inside the complete API. It has no persistent cache or host reads.
Both launches use the caller's current stream, so the flag vector and new
index output obey ordinary PyTorch stream/capture lifetime. Q=0 allocates no
flag scratch and launches neither kernel. No new stream, host synchronization,
model/cache metadata mutation, or shape-dependent output reuse is allowed.

The additional two kernels are both timed, including flag allocation and
launch gaps. The retained official value sort is also timed. C5's index sort,
value sort and mask are the comparison, so T3 must not be described as reducing
all finalization from three kernels to two: its fast and fallback kernels
are two launches in addition to the retained official value sort.

Promotion requires exact full API outputs, stream/graph/output ownership,
live compiler/runtime identity and improved complete API latency on the real
target geometries without material tail or pathological regression. A CPU
proof or register reduction is insufficient. Root must review results before
any public API integration, source-scoped production identity collector or
complete-model gate.
