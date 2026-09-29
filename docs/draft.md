# NOSA SM90 indexer / sparse attention: KDA draft

## Task contract

Optimize the existing resident NOSA indexer and block sparse attention toward
40% useful matrix MFU. The primary workload is the current NOSA-8B sparse
64K-prefix + 1K-query suffix: BF16, 32 Q heads, 2 KV heads, D128, 64-token
blocks, full NOSA 33/64 selection and CIS attention bias. Use the existing
989 TFLOPS nominal dense BF16 peak, with the actual H20Z-reported device,
132 SMs, PCI identity and clocks recorded. This is not an offload experiment.

The user confirmed that the complete indexer and complete block sparse attention
must each reach 40% MFU on this workload. Include validation, incremental cache
updates, selection and auxiliary kernels in the corresponding complete-module
time; publish score-only and full-indexer results under distinct labels.
Never include padding, masked work or QK recomputation
in useful FLOPs. At 64K+1K attention has 68,190,994,432 useful FLOPs and
score/indexer QK has 34,616,115,200; 40% requires 172.374 and 87.503 us.

## Baseline and validation

Existing valid reports remain published during development:

- `kernel_mfu_h200_gpu1_20260928_071840`: seeded synthetic same-input
  native/Triton graph and eager operator timing; all query rows checked.
- `sparse_native_h200_gpu1_20260928_02` and
  `sparse_triton_h200_gpu1_20260928_01`: independent sparse model trajectories,
  64K+1K full-prefill and extend, end-to-end timing and module breakdown.

Current native paths use CUDA/CuTe WGMMA and model-owned incremental compressed
records. Score uses two QK passes; attention uses four-query block unions plus
per-query repair. Native graph synthetic 64K attention/score are 468.68/216.17
us in the existing report. These numbers describe the existing implementation,
not any future candidate. First capture real layer inputs and profile the
unchanged implementation with line information before choosing a candidate.

Relevant existing acceptance commands (set CUDA_VISIBLE_DEVICES to an idle GPU):

```bash
bash scripts/run_tests.sh cpu
bash scripts/run_tests.sh gpu
bash experiments/nosa_kernel_mfu/scripts/capture.sh <new-input-run> --kernel-backend native
bash experiments/nosa_kernel_mfu/scripts/run.sh <new-run> --peak-tflops 989 --reference-all
bash experiments/nosa_kernel_mfu/scripts/run.sh <new-real-run> --input-dir <captured-input-dir> --peak-tflops 989 --reference-all
bash experiments/indexer_block_sparse_profile/scripts/run.sh <new-native-run> --kernel-backend native --peak-tflops 989
bash experiments/indexer_block_sparse_profile/scripts/run.sh <new-triton-run> --kernel-backend triton --peak-tflops 989
```

## Risks and candidate directions

1. Profile score recomputation, per-tile launch/setup and normalization/pooling;
   test tiling and pipelining first, then consider eliminating a full QK pass
   only with unchanged normalization and dtype-rounding semantics.
2. Profile attention union construction, partial membership processing,
   nonfinite-value handling, TMA wait and WGMMA utilization. Improve work reuse
   and overlap without altering per-query selections, CIS or causal masking.
3. Measure full indexer validation, cache update and both TopK stages separately;
   remove redundant work where request/cache contracts justify it.
4. Real activation/selection distributions may differ substantially from
   synthetic inputs. Test both; preserve exact policy and near-tie behavior
   within existing numerical contracts rather than manipulating selection.
5. A target is not a result. Report observed useful MFU and residual bottlenecks
   honestly if a candidate does not reach 40%.

## Evidence and publication

Run Nsight Compute full/PM-sampling and source-counter collections on an
isolated public-operator harness and parse reports through ncu_report. Use
existing -lineinfo compilation. Temporary or failed candidates stay under
system temporary storage. Valid experiment records use the existing experiment
output/{data,log,profile}/<run_id> layout; selected report artifacts are tracked
under report/. Repository AGENTS.md overrides generic KDA artifact suggestions.

Keep old reports and corresponding raw results until replacement runs pass
correctness, source-stability and measurement checks. After publication replace
the affected reports and remove their superseded artifacts in the same update.
Audit all dependent experiments; do not rerun unrelated dense-only work.
