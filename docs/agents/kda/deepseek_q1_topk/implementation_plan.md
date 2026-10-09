# Implementation and validation plan

1. Finish the read-only node audit. Add independent candidate
   `experiments/deepseek_v32_echo_official/src/q1_topk_candidate.py` and harness
   `q1_topk.py`. Freeze their source before official acceptance.
2. Prepare actual L0–L2 score rows from independently hashed `q1_inputs`
   tensors. Save score bytes in the independent check directory. Run bitwise
   value/ID comparisons against production and a CPU radix-order oracle for
   all-equal ties, signed zeros, -inf tails, random rows, k boundaries, layouts,
   multiple rows, nondefault stream and changed-input graph replays.
3. Issue an immutable receipt only after all checks. Include saved tensors,
   all executed source snapshots, native top-k identity and each candidate
   compiled PTX/CUBIN identity. Check and bench use GPU1/CPU8–15.
4. In a separate process, verify the receipt before complete-API balanced
   AB/BA measurements. Use 50 pairs per actual L0–L2 row, one graph replay per
   arm/sample, plus a repeated-call confirmation. Report raw samples,
   per-order paired deltas, medians and wins without discarding outliers.
5. If timing passes, collect separate NSYS graph-node traces verifying the
   same Filtered kernel and the intended four-to-two-node change. Collect NCU
   full/source reports for the fused postprocessor on actual layer-0 scores;
   inspect actual Hopper metrics through ncu_report and preserve profiling
   effects separately from clean timing.
6. Record the decision and provide root a production integration patch only
   after correctness, complete-API performance and native/source gates pass.

Commands run from the repository root with PATH including `.venv/bin`,
`PYTHONPATH=.`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=1`,
`OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, and `taskset -c 8-15`.

```bash
.venv/bin/python -B -m experiments.deepseek_v32_echo_official.src.q1_topk \
  check --run-id q1_topk_check_20261008_01 --physical-device 1
.venv/bin/python -B -m experiments.deepseek_v32_echo_official.src.q1_topk \
  bench --run-id q1_topk_bench_20261008_01 --physical-device 1 \
  --receipt /tmp/cxldsagr-checks/q1-topk/q1_topk_check_20261008_01/receipt.json
```

Use a fresh ID after any executed-source change or failed run. Independent
checks stay in system temporary storage, performance data/log/profile under
the experiment output tree. No production dispatch or published report is
changed by this component task.

## Candidate two after profile

Triton candidate one passed 115 comparisons and 20 changed graph replays but
lost all 300 benchmark pairs. NSYS retains the identical Filtered core and
attributes 27.328–27.392 us to its fused sorter. NCU reports 17-way shared
load/store bank conflicts and 35,136 excessive shared wavefronts. Replace that
tail with the installed official CUB `BlockRadixSort<uint64_t,256,8>`, retaining
the same composite key and fused mask. Use the existing immutable TVM-FFI
loader and bind the complete installed CCCL header closure. Do not change the
production entry point. Repeat the full acceptance under `_02` before timing;
candidate-one snapshots keep their own identities.
