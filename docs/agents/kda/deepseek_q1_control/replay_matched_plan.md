# Reduced HBM replay with the formal execution environment

## Question and first decision

The retained reduced timer control has 96 ns median gaps on GPU1/CPU8–15;
the formal pipeline has 416 ns gaps on GPU0/CPU0–7. Their cache paths and
some quantization binaries differ. First match those conditions while keeping
the reduced timer's model/cache/graph lifecycle unchanged. Do not add edge
enumeration yet: it would change the observer before resolving this simpler
comparison. No production optimization or scheduling mechanism is assumed.

## Frozen scope

Reuse `q1_replay_timer.py` without editing it: one current HBM model, H65536,
A1/token111090, real L0–L2, default last-token logits, one full graph with
197 GPU nodes and 192 layer-owned nodes, no internal events. Construction is
collected; five shared plain warmups run with collection stopped. Plain/timed
callbacks differ only by zero/two ordinary replay timing events. Restore and
synchronize the same prefix outside each timed invocation. Keep the separate
changed-token bitwise check, 50 clean AB/BA pairs and two profile pairs.

The new wrapper binds
`deepseek_h64k_a1_free_20261008_01_h65536_a1_profile` as its reference.
Before importing Torch or the timer, it applies the recorded execution/compiler
environment, including the literal three-prefix PATH, GPU0, eight threads,
DG lineinfo and default cache selection. The formal record has no custom cache
overrides; remove those overrides instead of creating a new cache. Require
CPU affinity 0–7 and the same recorded GPU UUID and Torch precision.

## Implementation and acceptance

1. Add only `q1_replay_matched.py` and this plan. Override the timer's environment,
   source identity, runtime identity and receipt-writing adapters in this process;
   preserve its compute, restoration, correctness, timing and capture functions.
2. Require unchanged formal production/source hashes and checkpoint/input identity.
   Bind the wrapper, timer, event helper and formal result/provider hashes in the
   new receipt. Use a distinct receipt kind; an older timer receipt is invalid.
3. Require exact backend provenance. Each actually loaded native library and
   observed CuTe/Triton specialization must exactly match an entry in the formal
   execution record. Formal-only offload entries may remain unloaded; record them
   explicitly. Do not load them to manufacture equality. Missing or changed
   concrete files and different compiled bytes fail directly.
4. Retain the timer's source snapshot and save the reference records, participating
   native files and available compiled Triton artifacts with hashes outside the
   timed ranges. The existing CuTe in-memory MLIR hashes remain identifiable but
   are not a retained-binary proof. DeepGEMM's installed API/headers, environment
   and shared cache selection are matched; its collector does not expose a
   per-launch JIT binary ledger. Preserve that boundary.
5. Root independently reviews and runs check, clean bench, then NSYS profile.
   Match all 197 native signatures/192 layer owners and relevant raw NSYS settings;
   compute each window using clipped all-process activity unions, retaining the
   boundary-crossing final norm. Compare profile gaps separately from clean wall
   time. This agent does not run GPU work.

## Commands after review

Run from the repository root with fresh IDs. The wrapper owns the in-process
formal environment; the shell only selects the interpreter and CPU affinity.

```bash
taskset -c 0-7 .venv/bin/python -B -m experiments.deepseek_v32_mfu.src.q1_replay_matched \
  check --run-id q1_replay_matched_check_new --physical-device 0 \
  --request experiments/deepseek_v32_mfu/output/data/deepseek_h64k_a1_free_20261008_01_h65536_a1_profile/request.json
taskset -c 0-7 .venv/bin/python -B -m experiments.deepseek_v32_mfu.src.q1_replay_matched \
  bench --run-id q1_replay_matched_bench_new --physical-device 0 --pairs 50 \
  --request experiments/deepseek_v32_mfu/output/data/deepseek_h64k_a1_free_20261008_01_h65536_a1_profile/request.json \
  --receipt /tmp/cxldsagr-checks/q1-replay-timer/q1_replay_matched_check_new/receipt.json
```

Profile uses the same receipt and request, `profile` mode and fresh output/log
directories, with `--trace=cuda,nvtx --sample=none --cpuctxsw=none
--cuda-graph-trace=node --capture-range=cudaProfilerApi
--capture-range-end=repeat`. Keep output staging and exporter failures explicit.
If the matched reduced flow still has short gaps, investigate the remaining
full-pipeline lifecycle and only then collect DAG edges in a separate control.
Neither outcome alone identifies a hardware mechanism.

## Independent analysis entry

`analyze_replay_matched.py` will build a fresh formal-reference ledger from the
FREE cohort's capture-3 lineage and capture-4 measured HBM replay. It delegates
unchanged raw NSYS/signature/owner/event checks to `analyze_replay_timer` through
an explicitly scoped receipt adapter that accepts only the distinct matched
kind. No receipt JSON or execution identity is rewritten. The new entry binds
the wrapper and formal-source/runtime references, verifies archived artifacts,
independently recomputes the 50 clean pairs, and checks clipped activity unions
across every process. The original analyzer checks same-process unions despite
some older output field names saying all-process. Its existing behavior remains
intact and the new result records the additional cross-process check separately.
