# Executable plan

1. Refactor the current exact packing into `echo._pack_q1_keys` without changing
   dispatch. Add an independent candidate in `indexer/page64.py` and bind it in
   `build_info`. The official bridge consumes this helper after acceptance.
2. Add a separate check/bench/profile harness. Freeze the tensor-copy reference
   in that harness. Verify byte equality for N=1,63,64,65,127,128,129,32768,
   65536,65537, including integer-encoded special scales. Verify graph updates,
   official paged scores and exact selections on saved real L0–L2 inputs.
3. Run independent correctness, then clean eager/graph packing and full MQA
   timings with matched source/input/device identity and every sample retained.
   Collect full/PM and source-counter profiles for the packing bottleneck.
4. Promote the helper only on a measured complete-API win and passing checks.
   Update permanent regression tests for CPU dispatch and GPU byte layout.
5. Coordinate source freeze with official ECHO/linear work, then run formal
   full-model acceptance and measurement. Preserve current published results
   until the replacement is accepted and measurement integrity is verified.
