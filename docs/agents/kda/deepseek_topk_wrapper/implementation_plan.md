# T1 implementation and acceptance plan

- [x] Read C5/C9 evidence, installed dispatch/bit traits, cache lifetime and MLA.
- [x] Select exact finalizer; defer resident metadata changes.
- [x] Implement CPU proof, temporary Triton finalizer and screen driver.
- [x] Run CPU contract/ABI and syntax checks; assert CUDA stays uninitialized.
- [x] In root's exclusive GPU0 window, pass strict correctness before timing.
- [x] Time complete APIs sequentially in alternating order; record actual
  kernel/compiler identity and resource use. All six timing cases regressed.
- [x] Reject T1 and release GPU0. Allocation/promotion/full-model gates do not
  proceed for this losing candidate. No production edits.

Prototype directory: `/tmp/deepseek_topk_order_t1_20261004/`. Default mode is CPU.
From the repository root:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  /tmp/deepseek_topk_order_t1_20261004/screen.py --mode cpu \
  --output /tmp/deepseek_topk_order_t1_20261004/cpu_contract.json
```

Only after root's GPU grant (device assignment supplied by root):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  /tmp/deepseek_topk_order_t1_20261004/runtime_screen.py --mode correctness \
  --device cuda:0 --output /tmp/deepseek_topk_order_t1_20261004/gpu_correctness_v2.json
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  /tmp/deepseek_topk_order_t1_20261004/runtime_screen.py --mode benchmark \
  --device cuda:0 --warmup 10 --repeats 40 \
  --output /tmp/deepseek_topk_order_t1_20261004/gpu_screen.json
```

CPU proof covers all K=1..2048. The bounded GPU gate uses 27 representative K
values around powers of two and vendor dispatch boundaries; Q=0/1/small/128/1024;
N=1/17/257/2049/4099/
32768/65536/65664/66560; count=min(K,N); non-power-of-two widths/strides;
causal padding, all invalid, tie overflow, finite extrema/subnormals, signed zero,
infinities and separate out-of-contract NaN stress. Compare all value bits and
indices; retain outputs over further calls, test nondefault stream and changed
input graph replay. Preserve invalid metadata and BF16/FP16 rejection.

Initial timing: Q/N=1024/1024, 1024/32768, 1024/65536 and 128/65664, K=2048;
add ties/all-invalid at 128/65664. Strict equality precedes each case. Report
complete API wall with completion and CUDA event time separately, including
allocation, conversions, selection, sort, mask and launches. Exclude compilation
and warmup. No profiler in the initial screen. A later root-coordinated step
collects kernel-activity/identity evidence if timing justifies it.

Keep accepted C5/C9 evidence and every published report until coordinated
replacement. A new kernel cannot inherit old correctness or performance results.
The completed screen used `.venv/bin` prepended to PATH and the pinned bundled
`TRITON_PTXAS_PATH`, with `TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas`
preseeded before imports. The saved JSON records exact strings and identities.
