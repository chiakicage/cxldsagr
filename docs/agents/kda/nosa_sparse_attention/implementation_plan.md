# NOSA resident block sparse attention: implementation plan

## Starting point

The accepted FA3-v3 implementation is described in [checkpoint.md](checkpoint.md).
In `kda_main_bf16_pair_v3_development`, complete L0/L15/L31 kernel sums are
175.391 / 173.247 / 170.879 µs; only L31 meets the 172.374 µs target.
The independent offload path reuses this attention implementation, so changes
must be audited for both resident and offload impact.

## Next implementation steps

1. Freeze current source, installed FlashInfer 0.6.18 header identity and
   build flags. Reconfirm eight-query unions, original-stride TMA, direct
   epilogue output, cost/tie ordering and per-query numerical repair.
2. Profile the current complete call, including prepare/sort/repair. Read the
   [rejected and unmeasured candidates](investigation_log.md) before selecting
   a bounded change; old register or phase diagnoses are not current timings.
3. Check actual CTA register/shared capacity, alignment, TMA/GMMA descriptors,
   barriers and lifetimes before GPU acceptance. Test real captured cases,
   short/tail/empty inputs, independent strides, exceptional values, legal
   all-fallback cases and poisoned graph replay where supported.
4. Admit a candidate only through paired complete-call measurements and
   relevant fallback controls. Preserve numerical tolerances, exact query
   selections, CIS and causal semantics. A removed launch or lower helper
   time does not itself establish a complete-call gain.
5. Rebuild main, validate all affected native paths and exact supported
   attribution graphs, and remeasure both complete resident modules on all
   three layers. Keep kernel sums distinct from event and wall intervals.
6. Rerun affected experiments with fresh run IDs after correctness and source
   stability pass. Synthetic MFU and affected full-NOSA pattern reruns were
   already pending at this checkpoint; audit the published full-model and
   offload results for any additional impact of the new change.
7. Publish accepted results, then replace superseded affected artifacts. Keep
   the 40% goal open until all three complete attention cases pass, alongside
   the separate complete-indexer requirement.

Shared [acceptance and publication rules](../README.md) remain in force.
This document move did not run kernels, tests or experiments.
