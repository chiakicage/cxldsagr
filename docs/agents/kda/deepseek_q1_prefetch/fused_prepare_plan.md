# Executable plan: private stateless preparation

1. Add `q1_fused_prepare.py` and `q1_fused_prepare.cu`. Use the existing immutable
   native loader; retain the original promotion/clear provider. Allocate exactly
   the original packed/table/stage buffers, validate the original lease ABI,
   consume prepared tokens once, and register cleanup before the fused launch.
2. Add `q1_fused_prepare_run.py` with `check`, `bench`, `profile`, and isolated
   malformed-input modes. Reuse the existing saved-input Case and CPU packing
   reference. Validate N/page boundaries, raw key/scale bits, randomized valid
   page permutations, current-pointer changes, graph replay, nondefault streams,
   cleanup injection, and malformed context/page assertions. Compare every full
   real-layer score/top-k bit and independently audit each actual prediction.
3. Run CPU import/CLI/static checks only until the parent assigns the GPU. Freeze
   private source and prepare exact commands before reporting readiness.
4. On the assigned SM90 GPU, run an independent check into a fresh system-temp
   directory. No benchmark starts without its compatible signed receipt.
5. Run 20 warmups and 100 balanced AB/BA pairs for each real L0-L2 input under
   cold, unsaturated, partial, resident and empty-prediction states. Graph event nodes enclose
   complete indexer and cleanup; reset stays outside timing. Record all samples,
   reservation counters and actual copied bytes. Eager wall/API timing is a
   separate boundary if collected.
6. Collect an independent ProfilerAPI/NVTX one-call profile and, if clean timing
   improves, source-bound NCU for old packing/prepare and fused preparation.
   The parent then decides whether a full-model integration candidate is worth
   testing. Do not update the selected reports from private operator results.

The first check passed, but its benchmark was rejected before sampling because
Ninja rebuilt the mutable generic ELF (see the checkpoint). The current commands
use the new source-bound immutable launcher. The independent replacement check
and clean benchmark completed on GPU1/CPUs8-15 with matching source/native
identity. The commands below reproduce their separate entry points. A new run
must use a fresh output directory.

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 8-15 .venv/bin/python -B -m \
  experiments.deepseek_v32_echo_official.src.q1_fused_prepare_pinned_run \
  --mode check --physical-device 1 \
  --output-dir /tmp/cxldsagr-checks/q1-fused-prepare/check_20261008_02

PATH="$PWD/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 8-15 .venv/bin/python -B -m \
  experiments.deepseek_v32_echo_official.src.q1_fused_prepare_pinned_run \
  --mode bench --physical-device 1 --warmups 20 --pairs 100 \
  --receipt /tmp/cxldsagr-checks/q1-fused-prepare/check_20261008_02/receipt.json \
  --output-dir experiments/deepseek_v32_echo_official/output/data/q1_fused_prepare_bench_20261008_02
```

Step 6 completed with a separate full-model NSYS capture and individually bound
NCU kernel diagnostics. Historical isolated packing exactly matches the accepted
CUBIN/PTX, source and saved L0 input; current baseline arange/stage and candidate
fused preparation are independently verified. The baseline NCU capture is
incomplete, and unlike captures are never summed. The checkpoint records all
bindings, missing fields, original commands and the removed redundant capture.
The parent has the completed CPU audit and may proceed to production integration.
