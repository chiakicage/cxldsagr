# T1 exact top-k ordering after C9

Status: T1 rejected after strict GPU correctness passed and every complete API
timing case regressed, 2026-10-04. See [T1 checkpoint](t1_checkpoint.md).
GPU0 was released; no production edits. C5 remains production. Its evidence is in
[checkpoint.md](checkpoint.md); original task/draft/plan are in `history/`.

Objective: reduce complete `exact_topk` API cost while returning exactly the
C9 FP32 values and int32 indices, including order, ties and value bits.
`topk_order_t1` retains installed FlashInfer SMALL selection and replaces its
two output sorts plus mask with one Triton uint64 lexicographic sort. Code stays
in `/tmp/deepseek_topk_order_t1_20261004/` until parent-controlled acceptance.

The API remains inference FP32 `[Q,N]`, Q>=0, N>=1, Python integer K=1..2048,
SM90, output width `min(K,N)`. BF16/FP16 remain rejected. Noncontiguous input
conversion stays inside the API. Causal/padded columns are already -inf; there
is no ragged/page-table API substitution. Every output value bit is preserved.
Arbitrary NaN logits remain outside the producer contract, but isolated finalizer
checks cover NaN payloads and GPU NaN stress is reported separately. Both
infinities receive index -1 after ordering.

Positive zero precedes negative zero under the official ordered-bit key.
Equal numerical zeros are not one tie group. Equal ordered keys use smaller
logical index first. Nonfinite masking occurs after sorting and never changes
value bits. No approximation, numerical-float tie substitution or dropped sort.

Keep the same official selector specialization, 1 MiB row-state scratch,
current stream, fresh outputs and caller causal visibility. Finalize only the
new outputs in place, with no persistent/tensor scratch, scalar read or host
synchronization. Preserve capture/replay and independent output lifetimes.
All new work remains owned by the `exact_topk` API.

Resident selection is deferred: cache append occurs after top-k and before
physical mapping, so fusion would cross a required lifetime boundary. Its exact
union counters, priorities, two clocks, lease/proof checks, candidate-tail rules,
empty-input events, invalid-ID traps and fallback paths must remain unchanged.

Promotion requires strict all-value-bit/index correctness across every supported
shape class, existing selection tests, changed-input capture/replay, nondefault
stream and output retention, actual kernel identity, no hidden synchronization,
and allocated/reserved accounting. Compare complete APIs with alternating order,
fixed source/compiler identity, declared warmups/repetitions and same hardware.
Require cold-target improvement without material tail/revisit regression. Root
then controls integration, whole-model and affected formal/profile reruns. CPU
proof or component timing alone does not replace those acceptance gates.
