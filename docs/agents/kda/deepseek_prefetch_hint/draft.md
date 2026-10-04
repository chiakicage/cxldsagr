# ECHO prefetch hint draft

The accepted C3 profile attributes 16020 history activities to ECHO exact_topk
versus 7040 for serial sparse over 640 layer calls. Its CPU exclusive scopes are
241.418 versus 139.773 ms. These are historical combined scopes, not an isolated
hint measurement or C6 latency. The same finite-mean hint expression remains in
C6, whose ECHO first-visit gap is 121.554 ms above contemporaneous HBM. A direct
component measurement is needed before assigning any fraction of that gap.

Candidate: fuse finite classification and zero masking into one Triton pass,
also writing exact integer partial counts. Keep torch.sum on a contiguous FP32
masked tensor with the identical shape as the checked expression. A final
single-CTA kernel sums integer counts, clamps to 1, converts to FP32 and divides
with round-to-nearest, then writes only offset[0]. This retains the original
FP32 reduction and reduces surrounding launches. No logit or top-k change.

Risks: ATen true division may differ from the candidate; reject on any scalar
bit mismatch. Output alignment or allocator choice must not change torch.sum
reduction geometry. Noncontiguous/padded score rows need explicit strides.
Scratch lifetime must remain owned through asynchronous completion. NaN/Inf
classification must operate on bits. Positive-zero replacement must match
masked_fill; finite negative zero remains intact.

Start with a temporary isolated prototype, no production changes. Byte-exact
screen first; timing only after root grants an exclusive GPU window. If exact
and faster, integrate under operators/deepseek_v32/indexer and add the narrow
model call, then run model, cache and checkpoint validation before a new formal
trace. Do not combine a changed sum tree with this candidate.
