# GPU dispatch findings and implementation checkpoint

This is engineering evidence for the active MFU task, not a published performance result.

## Original measured gaps

Source: `experiments/deepseek_v32_motivation/report/single_layer/intervals.csv`
and `timeline.json`, measured run `refactor_final_deepseek_profile_20261005_01`,
request 16, candidate layer 1. Merge all streams' GPU intervals before subtracting
their union from the reported layer window:

| Method | Window ms | GPU union ms | Gap ms | Gap/window |
| --- | ---: | ---: | ---: | ---: |
| hbm | 1.154433 | 0.914143 | 0.240290 | 20.81% |
| echo | 2.805954 | 1.547713 | 1.258241 | 44.84% |
| serial_sparse | 1.930850 | 1.029090 | 0.901760 | 46.70% |
| dense_prefetch | 2.957123 | 2.484835 | 0.472288 | 15.97% |

These are intrusive profile windows, not independent formal timing and not an
estimate of completely removable latency. Raw HBM SQLite `capture_3.sqlite`
shows that its largest 105.025 us gap covers a `cudaGraphLaunch_v10000` call
lasting 95.666 us (relative CPU start 21.960 us, end 117.626 us; first graph
kernel starts 121.889 us). Its finish launch lasts 40.568 us. Node-level graph
tracing may contribute to this cost; the existing trace cannot establish an
unprofiled overhead or classify the entire interval as removable Python work.

Other HBM gaps include 19.584 us between indexer zero/arange metadata launches
and 45.184 us from arange completion to the DeepGEMM logits kernel. The
resident indexer currently builds identical causal bounds outside each graph.

## Changes

- Bounded private exact recall (`H <= P`) keeps the miss count on GPU. Stable
  `argsort` victim ordering, logical-order compaction, record copy, publication,
  and original selection remapping remain unchanged. Publication accumulates
  successful recall counts in an eighth per-layer int64 counter. The counter
  slab, named allocation, offline estimator, reset, and release all include it.
- A bounded gather uses at most 128 CTAs and consumes only the GPU-count prefix
  of P-sized ID buffers. The generic transport implementation is shared with
  the dense-prefetch work and validates that count does not exceed capacity.
- A count-free CPU cannot distinguish an uncertified zero-miss selection, so
  this path conservatively invalidates append/residency optimization proofs.
  Actual maps, records, FIFO priorities, free bitmap, clocks and traffic counts
  still match the checked path. `H > P` retains the checked exact-union path
  and query splitting; no exact selection is truncated.
- The projection graph now generates exclusive causal ends from the same
  integer position used by RoPE and owns immutable zero starts. The local
  indexer adapter borrows these bounds. This removes two eager
  metadata launches for every graph-backed layer across all four local methods,
  including C10 HBM-only. Static storage increases by 4 bytes per query per
  layer/shape; private end storage is covered by the graph pool audit. Graph
  policy is `deepseek-compute-islands-v3-indexer-bounds`.

## Checks completed on GPU 0 (H200)

All commands run from the repository root with `.venv/bin` and CUDA in `PATH`.
GPU tests used `CUDA_VISIBLE_DEVICES=0`.

1. `python -B -m pytest cache/tests/test_sparse_recall_metadata.py cache/tests/test_resident_metadata.py models/deepseek_v32/tests/test_native_cache_dispatch.py operators/deepseek_v32/indexer/tests/test_echo_cache_ops.py -q`: **43 passed** in 7.50 s. Includes cold/partial/full residency, fragmented host pages, candidate tail, stable FIFO differential, stream changes, counter accounting, five failure stages, and isolated invalid-ID trap. A TorchDispatch test forbids `aten._local_scalar_dense` during both a miss and an uncertified all-hit recall.
2. `python -B -m pytest models/deepseek_v32/tests/test_compute_graphs.py operators/deepseek_v32/indexer/tests/test_echo_indexer.py models/deepseek_v32/tests/test_echo_cache_order.py -q`: **42 passed, 2 skipped** in 29.67 s. The skips are explicitly opt-in full-checkpoint checks. Covers graph dynamic positions and ends, outputs, delayed writeback ownership, capacity, failure cleanup, and causal score tails.
3. Before the official adapter and its dedicated tests were removed, the borrowed-bounds mask checks passed with this command: `python -B -m pytest models/deepseek_v32/tests/test_official_serving.py operators/deepseek_v32/indexer/tests/test_echo_indexer.py models/deepseek_v32/tests/test_echo_cache_order.py -q`: **48 passed** in 7.30 s. This historical command is not a current test entrypoint.

These checks do not establish performance improvement, full three-layer numerical
equivalence, formal MFU, dense overlap, or multi-user capacity. Those require the
independent experiment checks, timing and profiles being scheduled by the main agent.
