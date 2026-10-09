# Draft: fuse deterministic top-k ordering and mask

Baseline is production exact_topk: Filtered SMALL selection, index-ascending
finalize, stable value-descending sort, then nonfinite mask. Current HBM trace
spends 61.248 us across three layers in the latter nine kernels. This is a
profile observation, not a clean speedup target.

Candidate one calls existing radix_topk with sorted_output=False,
deterministic=False, and SMALL. The installed dispatcher still instantiates
the identical deterministic-selection core because SMALL implies it. A local
Triton kernel sorts uint64 composite `(descending FP32 ordered-key, logical
ID)` keys, restores original FP32 value bits, and masks nonfinite IDs. Keep
valid selection membership untouched. A power-of-two block covers k<=2048.

Risks are FP32 radix signed-zero ordering, invalid/padded slots, stable tie
ordering, Triton uint64 sort cost/register pressure, graph lifetime and
different generated artifacts between checking and timing. First implement
and validate the combined key algebra and complete API. If it is slower,
profile before changing block/warp configuration or using a CUB implementation.

Independent inputs include all-equal values, signed zeros, finite random
values, causal -inf tails, exact k boundaries and three real score rows
generated from the saved Q1 inputs outside the measured range. Cross-check
with both production exact_topk and a CPU ordered-bit/index oracle. Save
inputs and outputs so acceptance is independently rereadable.

The executable plan defines commands, receipts and promotion gates before
candidate implementation begins.
