# Initial compute-bank collection state

The initial inspector-package control leaves 67.584 us measured HBM idle.
Its compute-bank construction still occurs with CUDA-profiler collection
active, whereas reduced HBM controls allocate this bank with collection
stopped. Test that boundary within the complete formal lifecycle.

Use the already uninspected preparation path from
`q1_compute_inspector_profile_20261008_01` as the control. Move only the
initial original `prepare_graphs(..., "check")` call before the first
profiler range. Preserve the range itself with an explicit NVTX marker and
synchronization of the already completed preparation. All-four-method
warmups, cache switches and later `gap_profile.run_profile` stay unchanged.
Later full-extend captures retain their exact native inspectors. No new
model arithmetic or cache policy is introduced.

Run `q1_compute_collection_profile_20261008_01` on GPU0/CPU0-7 with the
same process/compiler environment and NSYS settings as the exact formal
reproduction. Use temporary staging as required by the original parser;
publish only after successful complete execution and 13 exports. Archive
the additional measurement wrapper. The first captured range must contain
the marker and no begin/end capture, graph launch or GPU activities.
Its missing bank-node history is intentional and must not be fabricated.

Reuse the exact original independent numerical receipt only after execution
identity checks pass. Independently check all four saved outputs, source,
runtime/native files, first-range API evidence and all 13 NSYS settings.
Extract HBM warmup and measured windows via later native full-extend node
ownership, including all 197 GPU nodes and the clipped all-process union
of the 192 layer nodes. Compare with both inspected and uninspected inside-
collection preparations. No prefill attribution, generic gap gate or clean
performance claim is made from this diagnostic.
