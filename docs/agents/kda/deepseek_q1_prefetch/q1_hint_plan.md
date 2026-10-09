# Exact Q1 hint: bounded source and measurement plan

Status: baseline acceptance, clean timing and full/source NCU completed under
the `_20261008_03` IDs; see [baseline diagnosis](q1_hint_baseline_ncu_result.md).
The private exact-tree candidate passed its component correctness and paired
API timing gates; see [component result](q1_hint_candidate_result.md). The private
checkpoint L0–L2 complete-call check and 100-pair benchmark also passed under
`q1_hint_model_{check,bench}_20261008_01`; raw NSYS lineage passed independent
review in [the model result](q1_hint_model_result.md). Bounded production
integration is in progress; fresh production acceptance and formal runs remain outstanding. Keep official
ECHO, its threshold policy, exact top-k, cache preparation and FlashMLA unchanged.

## Baseline and the only initial candidate

The FREE formal ECHO trace spends 36.288 us over three layers in the complete
hint chain: finite mask/count 4.640, PyTorch sum 25.280, mean publication 3.616,
decode EMA 2.752 us. These are invasive activity sums. Each sum launches one
CTA of 512 threads, uses 32 registers/thread and 2,048 B dynamic shared memory,
and lasts 8.288/8.416/8.576 us in that formal trace. The new hint-specific NCU
uses a separate cold-cache policy and must not replace those activity times.

After baseline profiling, consider only fusing finite mask/count and mean
publication into the same exact single-CTA sum tree. Replace the three-kernel
mean path with one private candidate; leave `update_decode_hint` and its call
site unchanged. The latter writes only `offset[1]`, with
`enable_fp_fusion=False`. The initial candidate covers SM90 inference, FP32
`scores[1,65537]`, unit inner stride and 16-byte-aligned input, plus the existing
contiguous `offset[16]`. Other shapes/layouts retain the current normal dispatch.
This avoids scratch traffic and two launches, but adds mask/count work to the
already small sum grid. The 8.256 us mask/publication total is measured work,
not guaranteed removable latency. Do not introduce a second candidate before
the first candidate's complete-call evidence is available.

## Exact reduction order

The installed Torch is `2.12.1+cu130`, git
`7269437d655783a26cba32aa88195b741ff496aa`.
`/root/.venv/lib/python3.12/site-packages/torch/include/ATen/native/cuda/Reduce.cuh`
has SHA256 `12ff2238c9d41da9916b2ffdfb761711c9520c218e65d7b2388988896a15f641`.
Relevant code is `setReduceConfig` (1041), vectorized reduction (500), and
`block_x_reduce` (635). The source-derived aligned Q1 schedule is:

- One output, four-value vector loads, `block=(512,1,1)`, `grid=(1,1,1)`;
  no cross-CTA reduction. The actual trace confirms the launch geometry.
- Thread `t` uses four positive-zero accumulators. Accumulator `i` visits
  `scores[4*t + 2048*j + i]`, `j=0..31`, in ascending `j` order. Thread 0
  adds element 65,536 to accumulator 0 before combining its four accumulators
  as `((a0+a1)+a2)+a3`.
- Shared-memory combinations use offsets 256, 128, 64, 32; warp shuffle-down
  combinations use 16, 8, 4, 2, 1. The height-one Y step adds no arithmetic.
- Nonfinite input is replaced by positive zero before entering this tree.
  Count finite values exactly as integers, clamp the denominator to one, and
  use round-to-nearest FP32 division. Preserve finite subnormals/signed zeros
  and the original overflow/NaN result bits; no reassociation or fast-math.

A generic Triton/CUB sum or multiple partial sums changes this order. Matching
the mathematical mean or a numerical tolerance is insufficient. Bind the
actual Torch CUDA binary and the local include closure in the new receipt;
headers alone do not prove how the wheel was compiled. Inspect the baseline
SASS/source correlation before treating the source-derived schedule as fully
verified machine behavior.

## Inputs and independent acceptance

The existing top-k check has a reusable three-layer corpus:
`/tmp/cxldsagr-checks/q1-topk/q1_topk_check_20261008_02/scores.pt`, SHA256
`1e2727034d9b777f4f4c4ca89ed32a70db431be2ed2deb8d503a23799c13f74a`.
Its receipt binds all three `[1,65792]` FP32 tensors. Use `[:, :65537]` for
the hint; all 65,537 logical values are finite and the physical tail is `-inf`.
These came from the unchanged official no-miss raw scorer on saved extra eager
L0–L2 checkpoint inputs, not the formal timed forward. They are representative
component inputs, not a pre-existing hint acceptance receipt. Prepare exact
kth values outside timing through the unchanged current top-k path.

Use the installed GPU PyTorch expression as the independent oracle:
`tail=scores[-4:]`; `finite=isfinite(tail)`;
`offset[0]=tail.masked_fill(~finite,0).sum()/finite.sum().clamp_min(1)`.
Do not test the candidate against a second copy of its reduction tree or a CPU
sum with a different order. Compare byte views of all 16 offsets and input
storage. Reuse the existing prefetch/decode hint tests, adding only gaps in
coverage at N=65,537:

- Mixed/all NaN and ±Inf; ±0; finite subnormals; cancellation/overflow; values
  chosen around vector, warp, shared-tree and final-element boundaries.
- Changed scores/offsets across captured replays and ordered nondefault-stream
  producer/consumer calls. Unsupported alignment/stride/Q keeps the old path.
- A Q1 then Q>1 sequence must preserve the actual later consumer's prefill
  `offset[0]`; retaining only logits is not enough. Compare prediction state,
  exact selection and cache/output state using existing model/cache checks.
- The complete chain preserves `offset[1]` exactly as two separately rounded
  multiplies by 0.5 followed by addition; include cancellation, signed zero,
  subnormals, large finite values and changed graph inputs. The mean candidate
  never writes that slot. No tolerance relaxation, repair or failure fallback.

## Measurement sequence and decision gates

1. **Baseline acceptance and measurement.** After the freeze, add a narrow
   standalone experiment driver with separate check/bench/profile modes. It
   calls the current hint APIs on the saved corpus and archives source/input/
   Torch/native identities. Missing SM90 or profiler support is an explicit
   failure. Use a new check under `/tmp/cxldsagr-checks/q1-hint/`; keep clean
   benchmarks and NCU in the experiment's normal `output/` directories.
2. **Minimal baseline NCU.** On the assigned SM90 GPU/CPU placement, profile
   the installed four-kernel chain for L0 once after warmup: full overview,
   then source counters for the dominant sum, with explicit kernel/range
   selection, kernel replay, cache-control and clock policy recorded. Query
   supported SM90 metrics; parse retained reports with `ncu_report`. Inspect
   achieved issue rate, active warps, load traffic/cache behavior and sampled
   dependencies/barriers. Sparse samples cannot support time percentages.
   If the wheel lacks line information, retain its binary/SASS results; a
   separately checked `-lineinfo` instantiation of the pinned headers is a
   distinct source-diagnostic build, never the measured production baseline.
   Expand to L1/L2 only if the initial evidence indicates input dependence.
3. **Private candidate gate.** Implement only the exact-tree mean fusion if
   baseline evidence supports the opportunity. Run new independent correctness
   first, then a clean complete mean-plus-unchanged-EMA API benchmark: 20 warmups,
   100 balanced AB/BA single-replay pairs per real layer, independently allocated
   output state, reset outside timing, all samples retained. Include complete
   eager calls separately. Compare source-bound receipts; report order strata.
   Only profile the candidate after a clean improvement; NCU latency is not
   the promotion metric.
4. **Complete-model gate.** Use new private bindings and the established
   independent-model check/bench pattern; do not edit frozen existing drivers.
   Both arms use official ECHO, H=65,536/A=1, real L0–L2, matching cold restore
   and the same preparation history. Check tokens 111090/111091/111092, all
   offsets, hidden/logits and cache transitions, then measure 100 balanced pairs
   of complete `forward(return_hidden=True)+synchronize`. Restore prefix/hints
   outside timing; keep observers out of timing graphs. Separate NSYS follows
   only to verify changed hint nodes and unchanged remaining work. Integration
   still requires the affected formal experiments' fresh acceptance, clean
   measurement, audit and publication under new run IDs.

Stop this candidate if exact bits fail or complete-call/model timing regresses.
The fixed tree prevents redistributing the dominant sum arbitrarily; launch
fusion may be too small to offset its added work. A negative result leaves the
current implementation intact. The next evidence would be the retained NCU
instruction/traffic diagnosis and clean deltas, not a relaxed hint contract or
an assumed benefit from deleting the Q1 `offset[0]` update.
