# T2 checkpoint: correct on bounded GPU gate, rejected for production

2026-10-04. T2 keeps the official stable value sort and exact SMALL selector,
then repairs only finite equal-bit groups with descending IDs using packed
uint32 run/index keys. See [the contract](t2_task.md),
[reasoning](t2_draft.md) and [execution plan](t2_implementation_plan.md).
The candidate and all engineering artifacts are temporary files under
`/tmp/deepseek_topk_order_t2_20261004/`. No production code or published
experiment report was changed for T2.

## Decision and complete API results

Root rejected T2 for production after the GPU gate. It passed exact values/IDs,
stream, graph, ownership and packing-fallback checks, but regressed the large
and tail causal target cases. Small, tied and invalid cases improved. All
forty measured samples remain in the results; no case or outlier was removed.

GPU0 reported NVIDIA H200 through PyTorch and NVIDIA M403 through nvidia-smi,
SM90, UUID `GPU-1bdee8b4-22ac-536c-208b-bfb4ed38b878`; PyTorch 2.12.1+cu130,
CUDA runtime 13.0, Triton 3.7.1, driver 570.124.06. Pre/post observations show
1,980 MHz SM and 3,201 MHz memory clocks and empty compute-process lists.
Root granted this isolated GPU0 window after C10 formal completion. These
snapshots are not continuous telemetry. Both configured PTXAS paths were
present before imports. The correctness and timing processes exited 0;
the window was released before later profiling work.

All numbers below are milliseconds. Delta is the median of the forty paired
`T2 - C5` differences at the same alternating-order repetition; it is not the
difference between the separately computed medians. Negative means faster.

| Q / N | Pattern | C5 wall median | T2 wall median | Paired wall delta | C5 event median | T2 event median | Paired event delta |
|---|---|---:|---:|---:|---:|---:|---:|
| 1,024 / 1,024 | causal | 0.103516 | 0.079364 | -0.024381 | 0.090192 | 0.065888 | -0.024560 |
| 1,024 / 32,768 | causal | 0.333323 | 0.364050 | +0.030282 | 0.319280 | 0.350224 | +0.030112 |
| 1,024 / 65,536 | causal | 0.438674 | 0.444639 | +0.007146 | 0.424832 | 0.430672 | +0.007232 |
| 128 / 65,664 | causal | 0.096173 | 0.120760 | +0.024801 | 0.082768 | 0.107296 | +0.024656 |
| 128 / 65,664 | ties | 0.118484 | 0.109381 | -0.008936 | 0.104896 | 0.095648 | -0.009024 |
| 128 / 65,664 | invalid | 0.150871 | 0.141647 | -0.009193 | 0.137392 | 0.128208 | -0.009248 |

The complete-API wall median regressions are 9.22%, 1.36% and 25.57% for
32K history, 64K history and the causal candidate tail, respectively.
CUDA-event medians agree in direction. This screen does not isolate the cause
within selector, value sort, branch behavior or finalization. No profiler or
whole-model acceptance was run for T2.

Evidence is `gpu_correctness.json`, `gpu_screen.json`, their separate logs,
`gpu_observation_before.json`, `gpu_observation_after.json`, and
`identity_and_timing_audit.json`. The independent CPU rehash script
`audit_gpu_results.py` checked all 133 compiler/source dependencies per process,
24 live correctness and four live timing specializations, 120/20 Triton
artifacts and five vendor records per process. All live module/function
handles and retained CUBIN hashes matched; the installed top-k library was
recorded as loaded. It also recomputed medians, means, all paired differences
and order-stratified deltas from the raw samples.

Independent source review by cache_c3 found no blocker and confirmed installed
FlashInfer preserves deterministic SMALL selection while omitting only its
index-finalization pass. This source review and the CPU proof do not replace
the GPU checks described below.

## Completed evidence

The final CPU proof passed 12,293 comparisons: 7,230 no-repair cases, 5,061
repair cases and two complete-baseline fallback cases. It covers all output
widths 1..2,048, random bit patterns, signed zeros, infinities/NaNs, long finite
ties, exact packing boundaries and a real key equal to `UINT32_MAX`. All 3,072
real captured rows were checked with both baseline and reversed tie IDs.
Each comparison checks every value bit and int32 ID against independent
index-ascending then stable ordered-value-descending passes. Evidence:
`cpu_proof.py`, `cpu_proof.json`, and separate stdout/stderr logs.

`screen.py --mode cpu` passed the official call ABI check:
`sorted_output=True`, `deterministic=False`, SMALL, unchanged 1 MiB row-state
buffer and `dsa_graph_safe=False`. It checked 20 invalid metadata/device
rejections across C5 and T2, plus three complete-baseline fallback calls
before any selector access, including Q=0. Fallback control-flow tests use
meta tensors with emulated CUDA metadata, not CUDA execution. The result is
`cpu_abi_final.json`.

The GPU driver's fixture plumbing was executed on CPU with its finalizer
replaced by the CPU key model: 14 sorted synthetic batches and six real
batches passed (`driver_fixture_audit.json`). This catches fixture/driver
errors only. All six Python files pass Ruff and syntax checks. All CPU checks
asserted that CUDA remained uninitialized.

`compile_cpu.py` explicitly compiled `GPUTarget("cuda", 90, 32)` without a
driver or module load. It emitted seven PTX/CUBIN pairs:

| BLOCK | index bits | Registers | Dynamic shared B | Static shared B | Stack/local B |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 8 | 0 | 0 | 0 |
| 32 | 5 | 16 | 0 | 0 | 0 |
| 128 | 16 | 27 | 512 | 1,024 | 0 |
| 1,024 | 10 | 63 | 4,096 | 1,024 | 0 |
| 2,048 | 16 / 17 / 21 | 124 | 8,192 | 1,024 | 0 |

The 2,048-wide PTX variants use 32-bit sorting comparisons, with no 64-bit
predicate instructions. Compared with T1's 122 registers and 16,384 B dynamic
shared memory, T2 halves dynamic shared memory but does not reduce register
pressure. Compilation is not performance or numerical GPU acceptance.
Evidence: `offline_compile.json`, emitted `finalize_b*_i*.ptx`/`.cubin`, and
separate offline compiler logs. `cpu_evidence_audit.json` binds and rehashes
the CPU results, current sources, and all emitted PTX/CUBINs.

## Real sample boundary

Source manifest:
`/tmp/deepseek_c10_combined_validation_20261004_01/topk_captures.json`.
The three `[1024,2048]` FP32 values/int32 ID batches come from correctness-only
official C5 calls at visible width 65,536 and query interval [64,512, 65,536).
The tensor payload totals 48 MiB. Capture IDs do not identify layers.

| Capture ID | Rows with finite ties / 1,024 | Finite tie runs | Maximum run |
|---:|---:|---|---:|
| 0 | 304 | 365 of length 2 | 2 |
| 1 | 138 | 147 of length 2 | 2 |
| 2 | 533 | 733 of length 2, one of length 3 | 3 |

All 6,291,456 values are finite and nonzero. The analysis checked manifest,
payload and raw tensor hashes, bit-key value ordering, and ascending IDs
inside ties (`analyze_ties.py`, `tie_distribution.json`). These sorted
baseline samples bound rows that could need repair. They do not reveal T2's
selector emission order or its actual repair frequency.

## Executed GPU gate

Root granted the GPU window after C10 formal run
`motivation_c10_20261004_u16_r2_01` completed. The following commands ran in
order; correctness passed before the benchmark started. Reproduction must use
new output names rather than overwrite these artifacts.

```bash
export PATH=/mnt/ssd-wlcb/chenkaiqi/cxldsagr/.venv/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export TRITON_PTXAS_PATH=/mnt/ssd-wlcb/chenkaiqi/cxldsagr/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
.venv/bin/python /tmp/deepseek_topk_order_t2_20261004/runtime_screen.py \
  --mode correctness --device cuda:0 \
  --output /tmp/deepseek_topk_order_t2_20261004/gpu_correctness.json \
  > /tmp/deepseek_topk_order_t2_20261004/gpu_correctness.stdout.log \
  2> /tmp/deepseek_topk_order_t2_20261004/gpu_correctness.stderr.log
.venv/bin/python /tmp/deepseek_topk_order_t2_20261004/runtime_screen.py \
  --mode benchmark --device cuda:0 --warmup 10 --repeats 40 \
  --output /tmp/deepseek_topk_order_t2_20261004/gpu_screen.json \
  > /tmp/deepseek_topk_order_t2_20261004/gpu_screen.stdout.log \
  2> /tmp/deepseek_topk_order_t2_20261004/gpu_screen.stderr.log
```

The driver passed complete C5/T2 value-bit and index equality on 41 main
shape/pattern cases, 27 K representatives, overflowing ties, all-invalid and
causal padding, K>N, K=1, Q=0, strided input and the unsupported packing
geometry N=2**21+1/K=2,048. It checks input preservation, fresh retained
outputs, a nondefault stream and four changed-input graph replays. Separate
sorted-input finalizer checks use the real and adversarial fixtures above.
Arbitrary-NaN full-API stress is separately labelled outside the model-logit
contract.

Only after correctness, the complete API comparison uses ten warmups and
alternates C5/T2 order for forty measured repeats on causal Q/N=1,024/1,024,
1,024/32,768, 1,024/65,536 and 128/65,664, plus tied and invalid 128/65,664.
It reports wall and CUDA-event intervals separately. No profiler or whole
model runs belong to this gate. The runtime wrapper records actual vendor
mappings, Triton compiled handles/PTX/CUBIN and compiler/source identity after
measured loops. The driver reuses T1's unchanged synthetic input and timing
helpers, whose source hash is recorded; its sorted-input finalizer checks
are T2-specific.

## Source and result identities

| File | SHA256 |
|---|---|
| `candidate.py` | `93f1ad1128fc6f65df4728ce4b3397f026399762b9a06e8bf0ad699a0ca6b057` |
| `screen.py` | `8564273e0bbe7562e34c17a62fd4a5da72edcd03a33eab22e83ef38bb9ad3061` |
| `runtime_screen.py` | `d51ef5eb24826fa99fc5d76e0feb96c4d22d55d26b4def3931a0c786a626dd1c` |
| `cpu_proof.py` | `41ae225ca7039edbeca1bcab67387b7794a331609ef58b388d1cb2d0700b47dd` |
| `gpu_correctness.json` | `07f2e9216c12ab4eed4aa02bdd65e6d77484041684dec766e03dbebd997972f5` |
| `gpu_screen.json` | `116b1792ce5200892b5c4fe5dc212ff0b100551cfe142e13acdf9b2c963b9928` |

T2 is not promoted. Its bounded component correctness does not establish
whole-model correctness or a model-level performance benefit.
