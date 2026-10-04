# DeepSeek MFU inspection and optimization

Status: compute optimization and publication complete, 2026-10-03. Starting revision: `1a9aa45`.

## Contract

Inspect and improve the implementation measured by `deepseek_v32_echo_prefill`.
Keep exact per-query top-2048 selection, BF16 latent/RoPE records, bounded offload
pool, independent empty-cache resident/offload prefixes, and identical restored
prefix residency per extend. Non-GR benchmarks are restricted by the user to the actual first three checkpoint
layers, preserving sequential hidden/residual propagation. GR is a separate workload. Preserve prior reports until fresh affected runs
pass numerical and measurement audits and their replacements are published.

## Initial evidence and work split

- Existing run `20261002_echo_layers3_mfu_01`: resident extend MLA GPU time
  22.066 ms / 44.9 ms summed kernels; useful operator efficiency approximately 8%.
  Replay NCU reports 12.49% occupancy, 12.16% tensor-pipe elapsed activity,
  67.45% L1/TEX throughput, 1.27% HBM throughput. Inspect tile reuse, shared
  memory/register limits and software pipelining first (root).
- FP8 projection audit and measured tile/parallelism improvement (linear agent).
- ECHO fused-indexer spill diagnosis, precise-logit-preserving optimization and
  cache-policy audit (ECHO agent); changing cache policy needs separate evidence.
- MFU audit: distinguish operator useful FLOPs / kernel time from precision-normalized
  ideal compute time / full uninstrumented wall time (accounting agent).

Official backend validation uses physical GPUs 1 (indexer), 2 (FlashMLA), and 3
(linear). The formal three-layer run uses GPU 1 after these tests finish.

## Acceptance and publication

1. Validate changed operators against numerical references and old implementation;
   cover masked/padded/strided inputs, invalid IDs and all-masked attention rows.
2. Measure candidates on captured real workload inputs; include allocations,
   auxiliary kernels, and scratch costs in the final operator comparison.
3. Freeze sources and rerun the 3-layer 64K + 1K experiment with independent
   resident/offload caches; verify every extend hidden value, annotated/control
   equivalence, FLOPs ledger and capture coverage. Add new NCU evidence.
4. Do not run full61: the user explicitly restricted non-GR benchmark scope to
   layers 0–2. Audit shared consumers including GR serving, preserving their
   separate measurement scope and pending cache-budget limitations.
5. Publish fresh run IDs with source and input identity; replace/clean only the
   superseded affected report and output artifacts once replacement is accepted.
   Update research implications through the project Supervisor when established.

Candidate diagnostics are temporary under `/tmp/deepseek-mfu-20261002/` until an
accepted reproducible measurement is promoted. They are not paper results.

## Official backend integration (2026-10-03)

User requirements supersede the earlier custom-kernel candidates: use official
DeepGEMM main (`057ca596`, 2.8.1), official FlashMLA (`ba89a346`, upstream
Hopper/V3.2 compatibility pin), and first three real checkpoint layers only.
The FlashMLA, dense/grouped FP8 GEMM, quantizer and resident indexer adapters
reuse upstream implementations. Custom fused ECHO offload remains where the
official DeepGEMM main API does not provide fusion. ECHO register/spill fixes
remain part of this compute revision. Intermediate Triton measurements are not
the final report.

A concurrent thread is changing shared ECHO cache semantics. Formal compute
measurement is isolated in `/tmp/deepseek-official-mfu-20261003` to preserve
the verified pre-cache-change behavior; actual source snapshots identify it.
The resulting compute evidence must not be attributed to the new shared cache.
DeepGEMM installed JIT headers are byte-checked against pinned sources; stale
`nv_dev` build output discovered by this check is removed before clean rebuild.

## Accepted compute results and handoff

Published `20261003_echo_layers3_official_01` and the freshly rerun original
implementation control `20261003_echo_layers3_control_01`, plus three new
`20261003_echo_layers3_ncu_*_official_01` reports. See the
[experiment report](../../../experiments/deepseek_v32_echo_prefill/README.md).
The accepted official run uses the pre-shared-cache source snapshots, with
DeepGEMM/FlashMLA adapters and ECHO register fixes. Shared-cache work continues
in a separate thread and is not covered by this acceptance.

- Resident prefix/extend: 1174.166 / 24.449 ms, versus fresh original control
  2684.585 / 46.793 ms. Offload: 2000.785 / 40.279 ms, versus 3602.220 / 66.473 ms.
- Useful precision-normalized end-to-end utilization: resident 24.66% / 22.09%,
  offload 14.47% / 13.41%. The original operator ratio is not model wall MFU.
- Three-layer extend FlashMLA: 2.683 ms / 66.01% operator utilization. Exact
  top-k and Hadamard remain significant nonmatrix work; cache overhead is not
  explained by GEMM MFU alone. Fresh NCU reports have zero local spills.
- Frozen-tree regression: CPU 1785 passed, 701 skipped, 58 subtests; GPU 56 passed.
  All eight model checks are bitwise. Independent audit covers 5496 matrix calls
  and 140787 kernels. All three complete MLA outputs pass unchanged independent
  FP32 thresholds (201326592 output elements total).
- Cross-backend model equality is NOT claimed: original tolerance fails for
  331 hidden and 2870 logit elements. Relative L2 is 0.004482 / 0.006040; same
  argmax does not establish task-quality equivalence. See comparison.json.
- Publication replaced 12 superseded full61/old-three-layer/intermediate-Triton
  runs and their report material. The fresh control is retained for the explicit
  implementation comparison. The old full61 scope was retired by user instruction.

Shared consumer handoff: GR currently has a concurrent shared-cache rewrite,
including ownership, ABI and hard-budget changes. Do not rerun the withdrawn
4K/16K matrix or promote this compute run as validation of the pending cache.
The next serving validation must use its latest authorized workload and verified
new cache revision; its source identity cannot be inherited from this run.
