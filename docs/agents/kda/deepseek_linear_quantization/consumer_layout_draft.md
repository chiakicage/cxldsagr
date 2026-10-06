# Consumer-layout scale draft

Current T1 assigns 32 quantization groups per CTA with two warps. Its arithmetic
is already accepted. Only the final scale store assumes row-major addressing.
DeepGEMM's `get_mn_major_tma_aligned_tensor` accepts stride `(1, aligned_M)`
without a transpose; FP32 TMA row alignment is four elements. This supports a
small first candidate with no new kernel launch or third-party change.

Implement one candidate: preserve the current scale store for the default
layout; for the opt-in specialization, write group `(row, column_group)` to
`row + column_group * aligned_M`. Allocate owned strided FP32 output and retain
the current contiguous FP8 allocation. Add only the explicit private ordinary
linear binding needed to select this output. Reuse T1's source/runtime identity
helper and existing fixture generator, rather than rebuilding provenance or
validation infrastructure.

Risks are less-coalesced scale stores, graph allocation/padding differences,
and accidentally routing grouped or public quantization into the new layout.
Input-owner/guard tests and scoped source diffs address semantic risks; full
quantize-plus-GEMM eager/graph timing addresses net performance. The old V9
transpose count is a hypothesis source, not a predicted saving or current-model
measurement. The new resident-mask baseline must be used for any later model
comparison.

The initial sequence is scoped freeze, address-only source edit, CPU source and
dispatch checks, independent review, then root-run separate correctness,
complete-API timing and profile. Commands and stopping gates are in
[consumer_layout_implementation_plan.md](consumer_layout_implementation_plan.md).
Do not optimize arithmetic, CTA geometry, quantization reuse or GEMM dispatch
inside this candidate. If direct stores regress, reject this variant before
considering a separately planned store-layout change.
