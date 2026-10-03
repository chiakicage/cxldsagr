# NOSA resident indexer: KDA task contract

Optimize the complete resident NOSA indexer toward 40% useful MFU on captured
L0, L15 and L31. The target remains unfinished. Follow the shared workload,
FLOP accounting, measurement boundaries and publication rules in the
[KDA index](../README.md); this task does not measure offload.

The primary shape is BF16, 65536 prefix + 1024 queries, 32 Q heads, 2 KV heads,
D128, 64-token blocks, full NOSA 33/64 selection and CIS. Useful QK FLOPs are
34,616,115,200; the retained 989 TFLOPS denominator gives an 87.503 µs budget.
Validation, incremental derived-cache preparation, ranking, both selection
stages and all helpers belong to complete-indexer time. Score-only results
must be reported separately; QK recomputation is not useful work.

## Semantic and ownership constraints

Model code owns compression, selection policy and cache reserve/finish/abort.
Incremental 32-token / stride-16 compressed records and the stable CIS pool
commit, roll back and truncate with KV. Invalid Q, new K or new CIS must not
modify derived cache bytes; validate writable aliases before any write.
Keep checked-entry capture rejection distinct from capturable async preparation.

Preserve RoPE inputs, per-Q-head normalization, GQA reduction, BF16 rounding,
five-window pooling, full NOSA selection and stable score ties. Exact pruning
may skip a tile only with a valid upper bound strictly below the exact cutoff.
Unknown bounds and unsupported shape/resource cases retain full computation.
Public score-only calls remain unpruned. Do not manipulate selection or use a
weaker numerical tolerance to reach a performance target.

## Optimization evidence

Profile QK recomputation, normalization/pooling, launch/setup, incremental
preparation and both TopK stages. Test tiling and pipelining before considering
removal of a full QK pass; any such change must preserve normalization and
rounding. Real activation/selection distributions can differ from synthetic
inputs, so both are required. Keep source, build, input and trace identities.
A partial-kernel win or private prototype does not establish complete MFU.

See the [accepted checkpoint](checkpoint.md), [current plan](implementation_plan.md)
and [historical investigations](investigation_log.md).
