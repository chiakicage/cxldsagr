# C4 draft

Parent: C3b integrated implementation. Root's C3 HBM history trace attributes
92.46 ms to CompareFunctor<long> and 46.21 ms to masked_fill over 640 history
indexer calls. echo.logits currently masks all Q×N positions even though all
prefix columns <= query_start are valid for every row.

FlashInfer audit: installed top_k returns sorted values/indices with explicit
deterministic/SMALL policy but has no per-row lengths. top_k_ragged_transform
supports lengths/row_starts, returns only indices, lacks sorted=True, and returns
sequential IDs when length<=k. top_k_varlen lacks the current deterministic/tie
policy parameters. Changing selection would also leave logits() invalid tails
uninitialized, violating its causal tensor contract. Keep exact_topk unchanged.

Candidate order agreed with root: first narrow the existing PyTorch mask to
result[:,query_start+1:]. Compare this with the previous full scan. This avoids
all prefix scans and needs no new kernel. If worthwhile after measurement,
compare a lazy Triton in-place suffix fill that computes endpoints from row IDs,
allocates no bool tensor, and launches once over at most Q×(N-start-1) positions.
Preserve physical row stride and avoid importing the ECHO native module from
resident logits. Choose the measured winner, not an assumed launch-only gain.

Risks: off-by-one causal endpoint, short contexts/empty invalid tail, N beyond
start+Q, padded physical stride, preserving finite prefix NaN bit patterns in a
standalone helper test, arbitrary tensor storage offsets, and top-k tie policy.
Current GPU work is reserved to root; plan/static changes only until granted.
