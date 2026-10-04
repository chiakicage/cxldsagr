# T3 executable implementation plan

- [x] Freeze [the contract](t3_task.md) and [design reasoning](t3_draft.md)
  before writing temporary code: threshold four, sort-free neighbor ranking,
  Q-element flags and exact general fallback.
- [x] Write temporary fast and flagged-fallback kernels plus the complete API
  wrapper under `/tmp/deepseek_topk_order_t3_20261004/`.
- [x] Prove the long-run predicate, short-run scatter permutation, unique output
  ownership and untouched flagged rows on CPU. Compare every value bit and ID
  to the independent C5 two-pass relation for widths 1..2,048; include runs
  around four, warp/CTA alignments, signed zeros, NaN/infinity masking, long
  finite runs, packing boundaries and all captured real rows in both tie orders.
  CUDA must remain uninitialized.
- [x] Verify official ABI, metadata rejections, pre-selection complete-baseline
  fallback, fresh flag allocation and Q0 behavior using CPU mocks/meta tensors.
- [x] Compile both kernels with explicit SM90 target and no CUDA driver/module
  load. Record source/PTX/CUBIN hashes, registers, shared memory, local/stack,
  and verify no sorting or prefix-scan path is embedded in the fast kernel.
- [x] Adapt the bounded T2 GPU driver to observe both actual live kernels and
  validate flag branching at short/long boundaries, in-place interwarp ties,
  full API exact outputs, graph/stream/ownership and all prior contract cases.
- [x] Request a root GPU window only after CPU evidence and driver are ready.
- [x] Finish GPU correctness and, only with a separate timing grant, API timing.
  In that window, correctness precedes timing. Ten warmups and forty alternating
  complete API repeats use the same six shapes/patterns as T2. Flag allocation,
  both new launches, retained value sort and layout conversion remain included.
  Report all wall/event medians, paired deltas and loaded runtime identities.
- [x] Root reviewed results and selected scoped production integration. Do not claim
  whole-model acceptance, or replace any published report from this screen.

Initial CPU commands, after the temporary programs exist:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/deepseek_topk_order_t3_20261004/cpu_proof.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /tmp/deepseek_topk_order_t3_20261004/screen.py \
  --mode cpu --output /tmp/deepseek_topk_order_t3_20261004/cpu_abi.json
PYTHONDONTWRITEBYTECODE=1 \
  TRITON_PTXAS_PATH=/mnt/ssd-wlcb/chenkaiqi/cxldsagr/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas \
  TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas \
  .venv/bin/python /tmp/deepseek_topk_order_t3_20261004/compile_cpu.py
```

These are engineering checks, not experiment results. CPU implementation and
offline compile became authorized after cache_c3's CPU timing window ended.
Root granted physical GPU1 correctness after the CPU gate, then a separate
exclusive timing window. Both completed successfully and the GPU was released.
Subsequent production work follows the separate integration plan and grants.
