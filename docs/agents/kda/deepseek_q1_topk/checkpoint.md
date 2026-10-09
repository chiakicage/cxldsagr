# Q1 top-k checkpoint

## Decision and scope

CUB candidate two was integrated after component and private full-model gates.
Production dispatch is Q=1, count=2,048 and N>=32,768; other shapes retain the
original path. Actual integration passed 65 focused tests, including Hopper
boundary/bitwise/changed-graph checks and native provenance. Formal matrix
`deepseek_h64k_a1_cub_20261008_01` and its independent operator profile
`deepseek_h64k_a1_cub_mfu_profile_20261008_01` passed and were published.
All 36 check and eight profile output tensors were independently reread;
comparisons were bitwise equal. The separate operator profile's eight outputs
also match the same check bitwise.

Private full-model run `q1_selection_model_bench_20261008_01` retained 100
AB/BA pairs. Layer event medians were 1.046336/1.038720 ms and graph medians
1.484304/1.476480 ms, each with 86/100 candidate wins. The synchronized wall
medians were 1.830168/1.820790 ms, but order-specific wall effects differ in
sign. Formal five-sample step medians remain approximately unchanged at HBM
1.868587 ms and ECHO 2.725153 ms. Do not claim the overall optimization goal
has been met. Component report `q1_official_path_report_20261008_06` binds
the retained raw samples, independent receipts, native bytes and inputs.

The installed FlashInfer selection is unchanged: `sorted_output=False`,
`deterministic=False`, `SMALL` still select the same deterministic Filtered
kernel specialization. CUB sorts unsigned composite keys containing inverted
FP32 radix order in the high 32 bits and logical ID in the low 32 bits, then
reconstructs value bits and masks nonfinite IDs. One CTA uses official
`BlockRadixSort<uint64_t,256,8>`. No third-party files were modified.

## Independent acceptance

`/tmp/cxldsagr-checks/q1-topk/q1_topk_check_20261008_02/receipt.json`
binds 115 bitwise comparisons against production and a CPU radix-order oracle,
20 changed-input graph replays and three nondefault-stream cases. Inputs cover
real layers 0–2, logical/padded/strided views, random, all-equal, signed zeros,
-inf tails, fewer finite values than k, ULP ties, 16 k boundaries, short N,
two rows and empty rows. The check saves inputs, outputs and actual native
binary bytes. Source, installed CCCL headers, compiler, TVM ABI and loaded
top-k/CUB library identities are bound before clean timing or profiling.

- Receipt SHA256: `b12886bebc33311bd75512face7a06f61a21fd04f3b4b6ff3d141f1cd8185823`.
- Top-k native SHA256: `a63d4dfe772493fd9cbc6e65ec9fa92ab77564ab30ef8aef4a6f0b1dec47ffa8`.
- CUB native SHA256: `7c6973504d9b1a6ef7b360d2ae6c34a0d7bc5dfe14b808e8be7289a464de8c5a`.
- CUB Python SHA256: `f0aeac20d6cd0d8e07af3676e3ddaf69d035d4b627548f09d906bd72a53d6cd4`.
- CUB CUDA SHA256: `e032304fdb34a01e77792796012f882dd906caf4f1868afc94bd9cbc530aab6c`.

Independent review by `qkv_integration` found no semantic, stream or lifetime
blocker, and requested context-boundary coverage before a broad Q1 dispatch.
Supplement `q1_topk_boundary_check_20261008_01` adds 20 passing strided cases
at N=32,767/32,768/32,769/65,536/131,072, each with random, causal, all-equal
and signed-zero inputs. Production and CUB match the CPU radix oracle
bitwise. The two loaded native identities match `_02`; the frozen candidate
source did not change. Its separate receipt and complete inputs/outputs are
under `/tmp/cxldsagr-checks/q1-topk/q1_topk_boundary_check_20261008_01/`.

## Complete API timing

Run `q1_topk_bench_20261008_02` measures 50 balanced AB/BA pairs per layer per
regime on GPU1, CPU8–15. Each arm includes selection, ordering and masking;
score generation, graph setup, correctness and source hashing are excluded.
All samples remain, including the single losing layer-0 sample.

| Layer | One replay baseline/CUB us | Paired median delta us | Wins | Twenty replays baseline/CUB us | Paired median delta us | Wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 51.584 / 49.568 | -2.048 | 49/50 | 48.2944 / 46.1808 | -2.0992 | 50/50 |
| 1 | 51.680 / 49.680 | -2.016 | 50/50 | 48.3872 / 46.2928 | -2.0792 | 50/50 |
| 2 | 50.976 / 49.280 | -1.728 | 50/50 | 47.7984 / 45.9552 | -1.8424 | 50/50 |

Both order-specific paired medians are negative for every layer/regime.
Raw samples and per-order summaries are in the run's `result.json`.

## Profile explanation

`q1_topk_nsys_20261008_02` observes graph capture and replay. Six formal NVTX
scopes each contain one GraphLaunch. CUDA correlation maps exactly four
baseline kernels and two candidate kernels, with no memcpy or extra kernel.
The first Filtered kernel has identical name, grid, block, registers and
shared memory in both arms. CUB takes 18.496–18.592 us; baseline finalize,
stable sort and mask total 20.384–20.544 us. These profile durations are
separate from the clean timing above. SQLite and `node_audit.json` are under
the run's data directory; the raw NSYS report is under its profile directory.

NCU `q1_topk_ncu_20261008_02` contains independent full/source collections
(`q1_topk_ncu_full_20261008_02`, `q1_topk_ncu_source_20261008_02` execution
receipts). `q1_topk_ncu_analysis_20261008_02` saves all metrics, PM samples,
source attribution and rule output extracted through `ncu_report`.

- Under base-clock, cache-flushed replay, CUB is 25.600 us. This is intrusive
  profiling, not clean API timing. It uses 70 registers/thread and no spills.
- Excessive shared wavefronts fall from Triton's 35,136 to 5,425; short
  scoreboard falls from 6.827 to 2.871 cycles per issue. CUB still has 2,048
  shared-load and 3,377 shared-store bank conflicts (about 1.7 and 2.0 way).
  Source counters locate 3,244 excessive wavefronts at CUB
  `detail/uninitialized_copy.cuh:37`, 2,048 at `block_exchange.cuh:627`, and
  133 at `block_exchange.cuh:614`. PC stall samples are sparse for this short
  kernel and should not be overinterpreted as precise per-line percentages.
- One CTA on 132 SMs gives 12.40% active-SM occupancy, zero tensor-pipe use,
  and only 0.025% peak DRAM read throughput. The small-grid warning reflects
  the Q1 workload; it does not establish a useful multi-CTA replacement.
  Blocked global loads/stores remain inefficient, but the whole-device
  throughput is low. Fifteen PM metrics are retained; no load-imbalance claim
  is made from the single-row workload.
- Six unavailable `ctc__*` metrics were reported by NCU and preserved in logs.
  No required local-memory/shared-memory/stall evidence depends on them.

## Rejected candidate one

Triton uint64 sorting passed `q1_topk_check_20261008_01` but lost all 300 pairs
in `q1_topk_bench_20261008_01`. Its fused sorter took 27.328–27.392 us in NSYS,
compared with the baseline three-node tail near 20.4 us. NCU measured 17-way
shared conflicts, 35,136 excessive wavefronts at Triton `standard.py:309`,
and 38.176 us under intrusive replay. It is rejected for this task.
Its source is still part of the frozen candidate-two harness identity until
root completes the integration decision; rejected run artifacts are not a
published experiment result and are pending task cleanup.

## Reproduction

Use the environment and CPU/GPU binding in `implementation_plan.md` and add
`--candidate cub`. Check, bench and profile use distinct run IDs; bench and
profile require the exact independent receipt. NSYS must trace graph creation
as well as formal replay. NCU uses kernel regex `cub_sort_mask_kernel`, one
eager layer-0 call, kernel replay, cache control `all`, clock control `base`,
then independent `full + PmSampling + PmSampling_WarpStates` and
`source + SourceCounters` collections.
