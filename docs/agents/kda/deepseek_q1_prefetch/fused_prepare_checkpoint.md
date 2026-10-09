# Private stateless preparation checkpoint

This file retains the private gate record and its frozen evidence. Production
integration and formal publication have now completed; current status is in
`checkpoint.md`. Private results are not relabelled as production measurements.

Candidate `q1-fused-page-and-stage-prepare-v1` is implemented only in new
`experiments/deepseek_v32_echo_official/src/q1_fused_prepare.py` and `.cu`.
The new `q1_fused_prepare_run.py` provides independent check, clean paired
benchmark, profile and isolated invalid-input modes. No production or existing
experiment source was modified for this candidate.

Independent check `check_20261008_01` completed successfully on GPU1/CPUs8-15:

- 100 preparation byte/layout cases: 10 N/page boundaries, aligned and 4-byte
  offset keys, all raw FP8/FP32 bits including nonfinite/subnormal scale patterns,
  exact zero padding and changed-data/page-table graph replay.
- 210 complete-indexer cases: real L0-L2, both variants, cold/unsaturated/partial/
  resident/empty predictions, eager, nondefault stream, repeated graph and
  changed-input graph execution. Every score and exact top-k is bitwise equal;
  every actual prediction separately passes strict eligibility, uniqueness,
  staging/pool byte, complete map/journal, priority/bitmap and statistics checks.
- Bounded token consumption, duplicate owner rejection, idempotent cleanup of
  host zero, original exception identity and cleanup registration before launch.
- Six malformed-input subprocesses: context mismatch, negative page and
  out-of-range page, each against baseline and candidate, all device-assert.

The receipt is
`/tmp/cxldsagr-checks/q1-fused-prepare/check_20261008_01/receipt.json`, SHA256
signature `9fe7d382a9115ea7e8c86eed90cc1ecf5ded8f5d297347c3598dcc27ea3870ad`.
It binds 2,968 saved artifacts, 2,840 declared sources, 5,883 runtime files,
real input hashes, actual mapped native libraries and packing assembly.
Candidate native ELF SHA256 is
`fb65104075af2df8bc3383a7722b527d98fca9189ac90150864d4e9787f208e7`.
The immutable artifact name is `cxldsagr_q1_fused_prepare_705aa7febb614335.so`.

Ruff and CPU-only CLI help passed. The check's final identity comparison and
receipt artifact verification passed. A later lightweight CPU review verified
the signature, case coverage and current private source hashes, recorded at
`/tmp/cxldsagr-checks/q1-fused-prepare/check_20261008_01_lightweight_review.json`.
An independent full archive/native rehash is deferred while the parent runs its
formal performance cohort. No extra full tensors were read during that review.

The current benchmark commands are in `fused_prepare_plan.md`. Each timed graph
contains the complete indexer plus original cleanup. Pool preparation/reset,
top-k and exact recall are outside that operator boundary. Component results
and the pending full-model gate are recorded below; production promotion also
requires a separate profile and the parent's complete-model acceptance.


The first component benchmark launch (`q1_fused_prepare_bench_20261008_01`)
failed before sampling: strict receipt identity rejected a rebuilt generic
ECHO ELF. All source/runtime/input/packing identities and four immutable native
libraries matched. The mutable generic ELF changed from
`84d76f9c14f16115d60299b6b26a1bb12c9236ce517990ba87d11840da37175f` to
`489c4730d32ea1c65b7485e4d3cba0409b3e46e09ee2315ec1b7556bb459767d`.
An ELF section comparison found exactly five changed bytes in non-ALLOC
`.strtab`: nvcc's temporary process-ID filename; all other bytes, including
code and the GNU build ID, matched. The failed benchmark output was removed.
No result was published and the process was not retried.

A new private launcher `q1_fused_prepare_pinned_run.py` uses
`q1_fused_prepare_native.py` to route only the unchanged generic build call
through the existing source/toolchain/ABI-bound immutable cache. Both new
sources and the immutable build record enter the execution identity; strict
ELF equality remains required. The frozen component implementation and runner
remain unchanged. A fresh independent check `check_20261008_02` is required
before the new benchmark run ID. The isolated formal cohort uses a different
`/root/.cache/tvm-ffi/` path; this helper explicitly confines all builds to
the private official-experiment runtime.

A read-only `ninja -n -d explain` diagnosed the rebuild trigger: the stored
`cuda_0.o` dependency timestamp differed from the object timestamp by 1 ns.
The immutable loader bypasses the mutable incremental rebuild on a validated
cache hit; it does not normalize ELF bytes or relax acceptance.

Fresh independent check `check_20261008_02` passed on the immutable loader:
100 preparation byte/layout cases, 210 complete-indexer cases and all six
malformed-input subprocesses. Receipt signature:
`d7d9a19db115369a9d9bd025f71e5090896f21aa7adb1eb7d695f29b1a25549e`.
It binds 2,970 artifacts, 2,842 declared sources, 5,883 runtime files and five
actual mapped local native libraries. Generic ECHO ELF SHA256:
`2d76e17ce3b28dc6ec0dd339d49315a0e9f1dc37f116fd2f22b532c0314876ae`.
Candidate native bytes remain unchanged. A separate process verified every
receipt artifact, current source/runtime/input and mapped ELF hash; its record
is `/tmp/cxldsagr-checks/q1-fused-prepare/check_20261008_02_independent_review.json`.

The replacement benchmark `q1_fused_prepare_bench_20261008_02` completed under
that receipt, with 20 warmups and 100 balanced pairs for each of 15 layer/state
groups. All 3,000 samples are retained; each group's AB and BA median delta is
negative (range -2.896 to -0.944 us). Paired group medians range from -2.800 to
-0.976 us. Cold L0/L1/L2 baseline/candidate medians are 37.168/35.936,
62.336/60.976 and 75.744/73.536 us; paired deltas -0.976/-1.744/-2.224 us.
The summed cold paired delta is -4.944 us across separate complete-indexer
calls, excluding caller FIFO/reset, top-k and exact recall. This is not a
complete-model latency result. `independent_review.json` in the run directory
verifies the strict source/native/input receipt binding, actual ELF/packing
archive hashes, all medians/order strata, finite positive samples and actual
traffic arithmetic. Parent heavy work was paused throughout clean timing.

Private full-model correctness passed at
`/tmp/cxldsagr-checks/q1-fused-prepare-model/check_20261008_01`, receipt signature
`29c0f795f5ffc79e57aca84bd4e098ec82f37b29d474d60a9f98251d1b687aa1`.
Both independent arms pass eager/diagnostic/clean graph checks for tokens
111090/111091/111092, all hint bits, outputs and each actual transition proof;
bounded preparation remains 64 and the public observer remains installed.
Receipt artifact and current-source/native rehash passed independently. Each
arm retains 62,914,560 bytes of graph private storage; process allocated,
reserved and device-used snapshots are separately saved and include all
models loaded at that point. Clean timing is predeclared as 500 balanced pairs
with overall/order-stratum/all-five-block analysis before sampling. The harness
passed 24 CPU dispatch/loader/evidence checks. A separate agent reviewed the
observer and loader composition, including injected nested loader failures and
observer cleanup failure retention. The active formal motivation is now the
isolated cohort; updated node sums and source hash are in `fused_prepare_draft.md`.

The first 500-pair model benchmark launch
(`q1_fused_prepare_model_bench_20261008_01`) failed at the receipt gate after
both arms prepared and before any samples. Current project sources and all
eight mapped local native ELFs still matched the accepted model receipt, but
three FlashInfer JIT libraries had been rebuilt in the mutable cache:
`rope.so`, `silu_and_mul.so` and `topk.so`. Their source and build.ninja hashes
matched, while actual ELF hashes changed. The frozen producer had not saved
its pre-gate identity, so these known changes are not claimed to be an exhaustive
in-process identity diff. No performance result exists from this launch; its
failed output directory was removed without retrying the process.

A new private FlashInfer loader and model launcher preserve exact
source/spec/toolchain and actual ELF equality, explicitly archive the three
FlashInfer libraries and compiler dependency files, and save the attempted
pre-gate identity before receipt validation. Normal validated cache hits load
the exact ELF without invoking Ninja. The combined native, model, FlashInfer
and envelope suite passes 55 CPU checks; Ruff and CPU-only CLI checks pass.
The independent analyzer passes 36 CPU tests. Installed/vendor and previously
checked sources remain unchanged. The frozen FlashInfer helper SHA256 is
`c9a7538b0fa36d92e2fd2861dc5a18cac50cb1ecb034504c5346f075558ad47c`;
the model launcher SHA256 is
`84236933f5638d0d8c6d73fcb31867e7f760f1943c5f43aa26d19727c0b5112b`.
The component's accepted check and benchmark remain valid. The independent
bounded component NSYS/NCU profiler has released GPU1. Fresh model check
`check_20261008_02` passed on GPU1/CPUs8-15 with receipt signature
`d60a9ee4d574f4c78846d12926d35fd1fa7b454805eb8f1cde420524f949e714`.
It binds 1,440 artifacts, including all three actual FlashInfer ELFs, their
original Ninja recipes, committed manifests and compiler dependency metadata.
All tokens pass the eager/diagnostic/clean graph comparisons, offset-bit checks,
actual transition proofs and bounded-cap checks. Both arms retain 62,914,560 B
of graph private storage. Their allocated/reserved/device-used observations
remain separate and include the process state at sequential preparation points.
The independent review at
`/tmp/cxldsagr-checks/q1-fused-prepare-model/check_20261008_02_independent_review.json`
passed all receipt/current/archive/native/FI bindings, all 12 compact transition
proofs and 36 raw-score eligibility/exact radix-top-k sets, prefix/eager/
diagnostic/clean outputs, all offset bits, GPU UUID and graph memory arithmetic.

Benchmark `q1_fused_prepare_model_bench_20261008_02` then completed under that
receipt with five warmups and exactly 500 balanced pairs (1,000 arm samples).
Root and other agents paused heavy CPU/DRAM work throughout the clean interval.
Baseline/candidate wall medians are 2.646831 / 2.6119945 ms; paired median delta
is -34.6505 us and the candidate is faster in 413/500 pairs. AB/BA paired medians
are -29.8995 / -38.4835 us. Consecutive 100-pair block medians are -31.5545,
-38.4350, -35.2900, -35.6055 and -33.2850 us; both order medians in every block
are negative. All samples, including recurring large tails, are retained.

The independent analyzer and supplemental audit passed at
`experiments/deepseek_v32_echo_official/output/data/q1_fused_prepare_model_analysis_20261009_01/`.
They verify all 500 pairs and 3,000 per-layer traffic rows. Mean deltas are
mixed: AB +3.1391 us, BA -116.4268 us, block 0 +15.1347 us and block 4 +17.9646 us.
There are 81 pairs with absolute delta above 0.5 ms (46 negative, 35 positive).
Baseline/candidate p95 are 3.49745 / 3.31058 ms and p99 are 3.58927 / 3.65829 ms.
Candidate-minus-baseline actual H2D is +2,304 B at the paired median and
+5,603.328 B at the mean; the improvement is not explained as less work.
Both graph private pools remain 62,914,560 B. The reader's strict DeepGEMM
header-directory handling was corrected and the review suites pass 49 CPU tests.

Separate bounded profile `q1_fused_prepare_model_profile_20261009_01` passed
on GPU1/CPUs8-15 with the same accepted receipt. Its new orchestration script
saves exact argv/environment before launch and requires all three NSYS ranges:
baseline setup, candidate setup, then one AB replay pair. Analysis
`q1_fused_prepare_profile_analysis_20261009_01` verifies raw graph ownership/
lineage and all recorded process/device overlap. Complete graph nodes fall from
263 to 257; preparation falls from nine to three nodes, totaling 23.744 versus
18.592 us. All 254 non-preparation signature/owner rows match. Full graph GPU
spans are 1,878.397 / 1,857.405 us; their busy unions are 1,831.451 / 1,813.597 us.
These invasive spans are not clean timing. Actual profile H2D differs by one
1,152 B residual recall record in L1; it is separately recorded. Graph memory
is unchanged. Root independently confirmed the raw node/signature comparison.
The
component cold paired sum (-4.944 us across separate calls) and full-model wall
paired median (-34.6505 us) are different measurements; the wall delta is not
attributed solely to packing. Production sources remain unchanged.

Bounded preparation NCU collection `q1_fused_prepare_ncu_20261009_01` completed
both applications. Its baseline invocation requested three matching actions
with one skipped match, but the report contains only block-table arange
(grid 17) and stage/token preparation (grid 257). Both are retained solely as
individually verified kernel diagnostics. The candidate report contains the
single expected fused preparation action (grid 1,025, block 128, 26 registers).
Neither process is running. The collector's completion message reports intended
counts; only the independently parsed report establishes actual coverage.

A separate zero-skip capture `q1_fused_prepare_ncu_baseline_20261009_02` also
omitted packing. Its three actions are query-bound arange (grid 1), block-table
arange (grid 17), and stage/token preparation (grid 257). `Case.invoke` does not
pass `_bounds`, so the original skip correctly excluded query-bound arange.
The earlier explanation that this skip omitted packing was disproven. The
Triton interception/filter limitation remains unresolved; no additional GPU
capture is needed for this candidate. The redundant zero-skip data, profile,
logs and one-off launcher were removed after this record was saved. Its report
SHA256 was `be9ed7653bc5b3148d9417a13b6e792602c9d71de4e77b8c9a9612d9ae4ea392`.

The source-bound audit instead reuses the retained `fused_full` and
`fused_source` reports from `q1_packing_ncu_20261008_01`. These profile the exact
current `pack_page64` specialization and saved L0 input on H200 GPU0, whereas
the new arange/stage/fused preparation diagnostics ran on H200 GPU1. Both older
reports have identical metadata and all six assembly hashes to accepted
component check_02: CUBIN `372d4a85eda2e9a633e2444524d80020b60c9cf1c78121bdc7f5bc94a20c22e7`,
PTX `110f1082acf0d73446b17431114351d556107b362f599ded42b24eab9785c3b2`.
The saved L0 input hash is
`fc2a2f093b434f658d9ad7ad0bcf1515e8c5426020021c788235740b3cbb073c`.
Production packing source and input loader match byte-for-byte; the old driver
has only formatting differences, with an identical Python AST. Both paths copy
`index_keys` and `index_scales` directly to CUDA. Output shape is [1,025, 8,448].
Both historical reports match all 64 recorded instruction rows to the accepted
CUBIN after accounting for NCU's absolute branch addresses and omitted
register-reuse annotations; raw disassembly and parser output are retained.

Analysis `q1_fused_prepare_ncu_analysis_20261009_01` passed source, runtime,
receipt, archive, exact input and native-owned NSYS launch-signature binding.
It preserves 47 selected analysis artifacts and hashes 15,668 input/evidence
files. All five diagnostic actions retain every available scalar and
per-instance validity flag, SASS/source correlation and original rule text.
Nonzero invalid correlated values occur in every report; they cannot support
quantitative per-PC stall shares. The prebuilt arange has no source mapping.
Some historical aggregate counters are absent and remain unavailable.
`complete_same_run_baseline=false` and `cross_report_latency_sum=null` make the
boundary explicit. No cross-report sum, paired NCU latency or full-model causal
attribution is produced. The matched component benchmark and full-model NSYS
remain the evidence for preparation reduction. Analyzer suites pass 23 CPU
tests; Ruff passes. The parent's source-freeze gate was released after this
CPU audit. Private acceptance/report artifacts retain their original identities;
production integration requires new acceptance and formal measurements.

The two baseline commands below preserve their exact original argv. They are
diagnostic records, not instructions to reuse these run IDs. The first intended
three preparation actions and yielded only page arange/stage; the second
intended three preparation actions and yielded bounds/page arange/stage.

```bash
/opt/nvidia/nsight-compute/2026.1.1/ncu --target-processes all --profile-from-start off --replay-mode kernel --cache-control all --clock-control base --section LaunchStats --section SourceCounters --metrics gpu__time_duration.sum,sm__cycles_active.sum,sm__cycles_active.max,sm__ctas_launched.sum,dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum --import-source yes --nvtx --nvtx-include q1_fused_prepare_complete_baseline/ --kernel-name-base demangled --kernel-name 'regex:pack_page64|official_prefetch::prepare_kernel|at::native::arange_cuda_out' --launch-skip 1 --launch-count 3 --export /mnt/ssd-wlcb/chenkaiqi/cxldsagr/experiments/deepseek_v32_echo_official/output/profile/q1_fused_prepare_ncu_20261009_01/baseline taskset -c 8-15 /usr/bin/env LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64: .venv/bin/python -B -m experiments.deepseek_v32_echo_official.src.q1_fused_prepare_pinned_run --mode profile --physical-device 1 --warmups 20 --variant baseline --policy zero --receipt /tmp/cxldsagr-checks/q1-fused-prepare/check_20261008_02/receipt.json --output-dir /mnt/ssd-wlcb/chenkaiqi/cxldsagr/experiments/deepseek_v32_echo_official/output/data/q1_fused_prepare_ncu_20261009_01/baseline

/opt/nvidia/nsight-compute/2026.1.1/ncu --target-processes all --profile-from-start off --replay-mode kernel --cache-control all --clock-control base --section LaunchStats --section SourceCounters --metrics gpu__time_duration.sum,sm__cycles_active.sum,sm__cycles_active.max,sm__ctas_launched.sum,dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum --import-source yes --nvtx --nvtx-include q1_fused_prepare_complete_baseline/ --kernel-name-base demangled --kernel-name 'regex:pack_page64|official_prefetch::prepare_kernel|at::native::arange_cuda_out' --launch-skip 0 --launch-count 3 --export /mnt/ssd-wlcb/chenkaiqi/cxldsagr/experiments/deepseek_v32_echo_official/output/profile/q1_fused_prepare_ncu_baseline_20261009_02/baseline taskset -c 8-15 /usr/bin/env LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64: .venv/bin/python -B -m experiments.deepseek_v32_echo_official.src.q1_fused_prepare_pinned_run --mode profile --physical-device 1 --warmups 20 --variant baseline --policy zero --receipt /tmp/cxldsagr-checks/q1-fused-prepare/check_20261008_02/receipt.json --output-dir /mnt/ssd-wlcb/chenkaiqi/cxldsagr/experiments/deepseek_v32_echo_official/output/data/q1_fused_prepare_ncu_baseline_20261009_02/baseline
```
