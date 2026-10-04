# ECHO prefetch hint contract

Optimize the ECHO-only finite-mean threshold update after exact top-k. Current
code computes the mean of finite values in the last min(4,Q) score rows,
writing offset[0] and preserving all other offset elements. It executes even
when a certified resident call skips prefetch, because a later evicted revisit
consumes the retained hint.

Preserve every finite score bit, nonfinite filtering, PyTorch FP32 sum order,
clamp-to-one integer count semantics, offset identity, exact selection and all
cache/transaction behavior. No installed/upstream source edits. GPU inference
uses SM90; CPU and unsupported layouts keep the checked expression.

Primary shapes: Q1024/N=1024..65536 in steps of 1024, Q128/N65664; cover Q<4,
padded row strides, nonfinite values, signed zeros and mixed magnitudes.
Acceptance requires byte-exact masked scratch and scalar outputs, unchanged
score/offset tails, same-stream safety, then complete API timing including all
allocation, mask, reduction and publication. Full serving rerun remains a
separate root gate. Existing reports remain until accepted replacement.
