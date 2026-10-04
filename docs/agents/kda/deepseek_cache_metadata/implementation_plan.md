# C3 executable plan

- [x] Audit existing C3 kernels/wrappers and native source fingerprint coverage.
- [x] Add meaningful CUDA differentials at nontrivial opaque record widths,
  page layouts, all-negative/duplicate selection, transient suffix and rollover.
- [x] Validate all relevant existing CPU/GPU tests on GPU 2.
- [x] Benchmark identical synchronized operations against checked C1 paths,
  warm up compilation, repeat at target P/C/H/A and publish component evidence.
- [x] Coordinate storage-accounting changes and private top-k wiring with root.
- [x] Freeze source, record candidate and outstanding end-to-end acceptance.

First inspection found all three native helpers implemented in
`operators/deepseek_v32/indexer/csrc/echo_resident.cuh`; their wrappers and
exports are present but no candidate-specific correctness evidence exists.

C3b candidate (after baseline probe): at Q=1024/top-k=2048, the resident
helper takes 0.166 ms synchronized wall versus 0.805 ms checked. Use a CTA-local
bitmap when the working bitmap fits 16 KiB and selection has at least 1M entries.
Merge each nonzero word once into the global bitmap, derive newly set bits from
atomicOr's return, stamp exactly those slots, and preserve the exact global union
count/max with block reductions. Keep the direct global path for small selections
or large P. Validate Q1024 target geometry before CUDA activity/wall comparison.
