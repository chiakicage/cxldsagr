# NOSA striped fetch / attention: implementation and validation plan

## Status and scope

The sequence below was completed for accepted build `484e32532ba74fb6`; see
[checkpoint.md](checkpoint.md) for the original run identities and evidence.
It is the revalidation plan for future implementation changes, not a claim
that this documentation move ran those checks. Implementation, ownership,
ordering and overlap definitions are specified once in [task.md](task.md).

## Implementation and validation sequence

1. Preserve full selection, CIS, causal masks, BF16 arithmetic and repair.
   Partition unique pages into eight disjoint token stripes, preserve one
   owner and one `.cv` host load per historical vector, and audit ready=8's
   cross-CTA acq_rel completion chain including empty tail stripes.
2. Match fetch/compute head priority only for two KV heads and 256 batches.
   Preserve within-head compute cost/tie order; fetch keeps page zero first,
   then descending blocks. Retain the original fallback for all other shapes.
   Keep all CTAs computing with 24/240 registers and the CTA pool guard.
3. Merge initialization/staging and use it in both fused and strong serial
   controls. Retain independent first-use planning and all preparation,
   repair, launch gaps and output/completion work in total-call timing.
4. Revalidate CPU transactions, native SM90 numerics/lifetimes, empty/tail
   stripes, repeated scratch/queue use, geometry fallback and full 32-layer
   resident/offload 64K+1K execution built from independent empty caches.
   Compare every extend hidden state. Cache allocations are not process peak
   memory; correctness-test elapsed time is not model latency.
5. Freeze source/input identities; run fresh 20-repeat timing, independent
   40-repeat confirmation and separate profiling, with full FP32 reference
   coverage and exact logical-byte checks. Keep every sample. Every layer's
   complete-call median must beat the new full-query serial control, supported
   by wall/submit and paired timing rather than overlap alone.
6. Independently audit both page-envelope and stripe-window evidence. Require
   both >=90% in every profiled sample; neither median nor envelope evidence
   alone can pass. Check all identities/bytes, exact stripe partition and page
   min/max envelopes, one fused main and zero serial kernel overlap.
7. Publish only after independent correctness/runtime/measurement acceptance.
   Preserve prior accepted reports until the replacement is published, then
   replace affected report assets/output runs in the same update. Unaffected
   resident results retain their identity; failed diagnostics remain in `/tmp`.

## Additional report checks

Retain `page_envelope_only_union_us`,
`page_envelope_only_math_overlap_us` and signed
`page_minus_stripe_math_overlap_fraction`. Report every sample's threshold
result before median/min/max aggregation. Negative fraction differences are
valid and must not be silently clipped.

## Work outside the accepted kernel task

Bounded HBM slots/eviction, cross-request residency, CUDA Graph capture and
CXL/RDMA remain outside the implemented or measured path. The full-checkpoint
resident/offload test establishes numerical equivalence only; fixed-input
single-layer replay does not establish full-model or serving performance.
Follow the [system roadmap](../../roadmap.md) for broader system work.
