# Current 64-CTA independent-warp diagnostic

Status: selected diagnostic on 2026-10-04 after independent-warp rejection.
This authorizes a new isolated diagnostic helper, static review, then root-owned
GPU0 collection. It does not reopen candidate acceptance or authorize runtime
changes. The frozen independent-warp native and screen trees remain unchanged.

## Question and evidence

Current representative screen `nosa_q128_independent_warps_screen_gpu0_20261004_01`
passed exact/FP32, payload, unique bytes, queue and lifecycle checks but rejected
its first internal sample. Page and stripe ratios are both 0.8005504587: copy
348.8 us, actual softmax 298.144 us, intersection 279.232 us. Uncovered copy is
20.928 us startup plus 48.64 us internal, with no tail. At the same copy span,
eliminating startup alone would still not reach 90 percent.

Recorded mean stripe duration rose from 5.019 us in the pair-priority sample to
16.669 us with independent warps; marked concurrency rose from about49 to172.
These are separate single instrumented samples, not API-latency comparisons.
They motivate distinguishing host-load-dependent stores, atomic/fence/sync cost,
ready polling, TMA transaction and consumer/pipeline waits on the actual current
64-CTA code. The old32-CTA NCU report cannot answer that question.

In particular, locate PC samples in the independent fetch body, ready-acquire
loop, producer pipeline acquire and consumer TMA barriers. Check eligible/issued
warps, actual registers/occupancy, system-read sectors and memory throughput.
Host payload is 449*32768 = 14,712,832 bytes; HBM reads are not host traffic.
K(next) precedes V(current), but existing traces do not establish its wait cost.
Source attribution may merge those issue sites; report ambiguity rather than
claiming counters identify a dependency they do not distinguish.

## Frozen inputs and scope

- Native stage: `/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_independent_warps_candidate_20261004/`.
- Screen stage: `/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_q128_independent_warps_screen_20261004/`.
- Screen freeze SHA256: `d77b40a6043bfa18231aa1e640bb2bb2fee27f173dd014dc7041febf1d5c06c1`.
- Partial summary SHA256: `643653035c5bf0e2265ac5fadc538d0011a8ac67eaf34d751f2561a345b89aef`.
- Rejected screen result SHA256: `d1cd5615218a0ce1e5d27767ca18378341bdd331920d22580da32fab92f9aaac`.
- Same accepted request16/layer16 Q128, H65536, two KV heads, BF16 operands,
  original selection, CIS, saved output hash and captured hit/miss owner tags.

## Driver and collection

Experiment owns a new sibling diagnostic helper. Reuse unchanged screen
verification, captured-input audit, execute_case and finish_case. Per invocation:
one original-halves1 correctness control, two unprofiled halves2 warmups, then
one halves2 target with internal markers disabled. A driver-owned wrapper around
NosaFetchWorkspace.run pushes/pops a unique NVTX range only for the target and
immediately delegates to the saved original method. No computation, native flag,
workspace, tolerance, restoration or lifecycle policy changes.

Check target exact saved output, fixed FP32 tolerance0.016, poisoned payload,
14,712,832 unique bytes, actual queue/cursor3784, library/resource identities and
cleanup. Each application replay records a distinct invocation identity and
validation file. Freeze helper files and their native/screen/input dependencies
before collection. Perform CPU validation and independent driver review first;
no broad test reruns for unchanged runtime.

Root allocates idle GPU0 with full NUMA0 CPU and memory binding. Preserve one
canonical venv PATH prefix, register8 allocator, GC, eight intra-op/OMP/MKL/OpenBLAS
threads and separately recorded inter-op default. Keep clock-control none and
cache-control all, recording both. No same-GPU overlapping measurements.

Run NCU strict application replay with grid matching, a single exact NVTX range,
`regex:nosa_offload_fused::fused_main`, and one matched action. Collect full+PM
sampling and a SourceCounters report with the installed available set. Retain
NCU version/section inventory, exact commands, stdout/stderr, actual selected
library/source identities and all application validation records. Do not use
kernel replay, which would reuse mutated host-cache state across passes.

Parse reports via ncu_report, inspect rule details, per-PC source attribution,
resources, issue/stall/memory metrics and each time-series clock independently.
Root reviews actual reports; HBM-audit checks collection and evidence integrity.
No timing/overlap promotion follows from intrusive NCU duration. A matching
pair-priority64-CTA profile is conditional on a concrete unresolved attribution
question after current-source analysis, not automatic.

All diagnostic outputs stay under a fresh directory in
`/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/`, outside experiment deliverables. Record
findings in checkpoint/investigation material and select a bounded implementation
only after this diagnostic resolves, or explicitly limits, the hypothesis.
