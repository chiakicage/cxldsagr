# Consumer-layout scale executable plan

Parent: current production Triton T1. Candidate ID: `consumer_layout_01`.
Workspace: `/tmp/deepseek_linear_scale_layout_candidate_20261006_01`.
Task and boundaries: [consumer_layout_task.md](consumer_layout_task.md).

1. Freeze current `fp8.py`, `quantization.py`, `_quantization_kernel.py` and
   `_quantization_identity.py` into a private baseline. Record those file hashes,
   the reused 213-fixture generator and the relevant upstream layout/helper
   sources. Do not freeze unrelated model/cache trees or use old C8 performance.
2. Create a separate private package from that baseline. Add a default-false
   consumer-layout option to the quantizer/ordinary linear, allocate FP32 scales
   with upstream-compatible strides, and specialize only the final scale store.
   Update the private identity policy to declare both layouts. Keep grouped and
   public default dispatch contiguous; keep all prepared, alias and error guards.
3. CPU checks: syntax/import without CUDA initialization; AST/source checks for
   unchanged quantizer arithmetic and GEMM call/recipe; mocked dispatch checks
   for public default, ordinary opt-in, prepared reuse and grouped exclusion;
   test scale-address coverage and padding for M0/1/3/4/121/128/1024 and partial K.
   Save CPU results in the private workspace. Send scoped diff for source review.
4. Root-only `--mode check`: reuse all 213 deterministic adversarial fixtures
   and the compiled official helper, compare candidate/default T1 FP8 bytes and
   logical scale bits, preserve full input owners and strided-output guards,
   and run nondefault-stream/retained-output/changed-input graph cases. Compare
   complete ordinary linear outputs and prepared/output contracts against
   current T1 with the same weights and unchanged DeepGEMM. Save full check
   outputs or existing fixture evidence sufficient for independent reread.
5. Root-only `--mode bench`: require a matching successful check. The eight
   current-model K/N pairs crossed with four M values form 32 shapes. Warm each arm
   four times; retain 80 fixed alternating AB/BA pairs per shape and API, without
   trimming. Measure full eager API (allocation + quantization + original GEMM),
   owned graph API (input copy + replay + owned output), and borrowed graph
   replay as a separate diagnostic. CUDA events and synchronized host wall are
   separate observations. No correctness comparisons occur inside timing.
   Report paired B-minus-A medians, Q1/Q3, P05/P95 and AB/BA strata. Reject a
   regressive complete API; repeat a claimed win in a fresh confirmation run.
6. Root-only `--mode profile`: use a distinct process and matching check, label
   A/B complete APIs and graph replays, and trace kernel nodes with NSYS. Verify
   transpose presence for baseline and absence for the candidate, unchanged
   original GEMM work, and no replacement transpose/copy. Profile is diagnostic
   and cannot replace clean timing. Archive small scoped source/runtime records.
7. Return component evidence and limitations to root. Only root can authorize
   integration and then rebuild Q128/Q1024 graphs for fresh complete-model
   hidden/logits/top-k/cache checks, four-method timing and complete/layer gap
   profiles. Existing experiment reports remain until formal replacement gates.

Prepared commands, run only when root grants the corresponding GPU phase:

```bash
source /tmp/deepseek_mfu_env.sh

python3 /tmp/cache_commit_drain_cuda_check_20261006_01_observer/driver.py \
  --output /mnt/ssd-wlcb/chenkaiqi/codex-checks/deepseek_linear_scale_layout_check_20261006_01_observer --selected-gpu 3 -- \
  env CUDA_VISIBLE_DEVICES=3 LINEAR_SCALE_LAYOUT_GPU_AUTHORIZED=1 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  numactl --physcpubind=24-31 --membind=0 .venv/bin/python \
  /tmp/deepseek_linear_scale_layout_candidate_20261006_01/run.py \
  --mode check --output /mnt/ssd-wlcb/chenkaiqi/codex-checks/deepseek_linear_scale_layout_check_20261006_01

python3 /tmp/cache_commit_drain_cuda_check_20261006_01_observer/driver.py \
  --output /mnt/ssd-wlcb/chenkaiqi/codex-checks/deepseek_linear_scale_layout_bench_20261006_01_observer --selected-gpu 3 -- \
  env CUDA_VISIBLE_DEVICES=3 LINEAR_SCALE_LAYOUT_GPU_AUTHORIZED=1 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  numactl --physcpubind=24-31 --membind=0 .venv/bin/python \
  /tmp/deepseek_linear_scale_layout_candidate_20261006_01/run.py \
  --mode bench --pairs 80 \
  --check /mnt/ssd-wlcb/chenkaiqi/codex-checks/deepseek_linear_scale_layout_check_20261006_01/result.json \
  --output /mnt/ssd-wlcb/chenkaiqi/codex-checks/deepseek_linear_scale_layout_bench_20261006_01

python3 /tmp/cache_commit_drain_cuda_check_20261006_01_observer/driver.py \
  --output /mnt/ssd-wlcb/chenkaiqi/codex-checks/deepseek_linear_scale_layout_profile_20261006_01_observer --selected-gpu 3 -- \
  env CUDA_VISIBLE_DEVICES=3 LINEAR_SCALE_LAYOUT_GPU_AUTHORIZED=1 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  numactl --physcpubind=24-31 --membind=0 \
  nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=node --output=/mnt/ssd-wlcb/chenkaiqi/codex-checks/deepseek_linear_scale_layout_profile_20261006_01 \
  .venv/bin/python /tmp/deepseek_linear_scale_layout_candidate_20261006_01/run.py \
  --mode profile --check /mnt/ssd-wlcb/chenkaiqi/codex-checks/deepseek_linear_scale_layout_check_20261006_01/result.json \
  --output /mnt/ssd-wlcb/chenkaiqi/codex-checks/deepseek_linear_scale_layout_profile_data_20261006_01
```

These are preparation commands, not execution receipts. Actual runtime/compiler
environment must match the accepted check, including the current T1 identity
helper's compiler environment. Root supplies isolation and observer receipts.
