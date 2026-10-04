# T2 conditional repair after the official value sort

Status: rejected for production, 2026-10-04. Root selected this narrower
follow-up after [T1 regressed](t1_checkpoint.md). T2 passed its bounded GPU
correctness gate but regressed the large/tail causal API cases; see
[the checkpoint](t2_checkpoint.md). Its exclusive GPU window has ended.
No production code or published experiment report was changed.

Keep official `radix_topk(sorted_output=True, deterministic=False, SMALL)`.
This retains exact SMALL selection and CUB's stable descending ordered-value
sort, while omitting the ascending-index finalizer. A Triton kernel repairs
only rows with adjacent finite equal-bit values whose logical IDs descend.
Rows without such an inversion already have the baseline finite ordering;
they only need nonfinite indices masked to -1. All output values stay untouched.

For a row requiring repair, compute equal-key run starts and sort uint32 keys
`(run_start << index_bits) | original_index`. Here
`index_bits = ceil(log2(N))`, with N the visible input width. Sorting these keys
orders IDs only within equal-value groups. Give each nonfinite position its own
run start and mask its output ID, preserving the existing value-bit vector.
The first candidate has no neighbor-rank implementation and no uint64 kernel.

Use this route only if `ceil(log2(min(K,N))) + ceil(log2(N)) <= 32`. For K=2048,
N<=2**21 fits. Unsupported packing geometries call the complete current C5 API
before selecting anything. Q=0, K>N, K=1, noncontiguous scores and all invalid
retain the existing contract. Input remains FP32 SM90; BF16/FP16 are rejected.
Positive and negative zero belong to different ordered-bit groups. Do not use
numerical equality for tie detection. Infinities/NaNs are nonfinite and have
masked IDs; arbitrary NaN logits remain outside the model input contract.

Outputs remain freshly allocated by the same wrapper/official op. The finalizer
reads and changes only its new int32 index output; values are read-only. One
program owns each row, with all needed index reads completed before in-place
stores. No host scalar reads, new persistent/tensor scratch, extra streams,
cache metadata changes or synchronization. Official row-state ownership and
`exact_topk` timing attribution stay unchanged. Graph/stream/output lifetime
requirements are inherited from T1.

T2 targets the index-finalization cost only: C9 HBM 35.852 ms/history and ECHO
34.409 ms/history, plus the existing mask. The 46.115/43.665 ms CUB value sort
and 108.164/102.469 ms selector remain. The new row scan, branch and conditional
sort have their own cost; no profile sum is an assumed end-to-end saving.

Promotion requires exact full values and indices, actual runtime/compiler
identity, bounded stream/graph/ownership tests, complete API improvement on
real target shapes without material fallback/tail regression, then parent-owned
integration and affected whole-model/formal/profile acceptance. T1's numerical
result cannot validate T2.
