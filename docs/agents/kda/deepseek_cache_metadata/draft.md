# Candidate C3 draft

Parent: C1/C2 accepted screen `motivation_opt_screen_20261004_01`.
The subsequent dense profile reports 657 ms pool operations, 547 ms eviction,
297 ms mapping, 230 ms stamping, 165 ms unique CPU-exclusive costs and 113.899 ms
unique GPU work. These instrumentation costs are diagnostic, not formal deltas.

The current tree already contains unvalidated C3 scaffolding. It adds fused
planned append and resident selection kernels, exact bitmap union counters,
and whole-history protection with two FIFO events. Review and validate this
source before performance comparisons. The private top-k call must be wired by
the model owner; public ensure remains fully checked and serves as differential.

Risks: warp bitmap reductions must be converged; all duplicate IDs must count
once; maps and FIFO priorities must match reference; clock rollover between two
events needs checked fallback; per-session bounded counters and shared bitmap
storage must be accounted by all planners; no asynchronous host read may race
with writeback. Planned slots only remain valid under certified append order.

Candidate order: (1) finish/correct C3 metadata and add differential tests;
(2) measure resident selection and append plus complete cold-build metadata on
GPU 2; (3) optimize demonstrated residual bottlenecks; (4) send source freeze
and accounting requirements to root before complete-model measurements.

Record source SHA256, GPU/dependency identity, shape, warmup/repeats, correctness,
wall/GPU metrics in checkpoint. Probe scripts/data live in system temporary paths;
no engineering checks become experiment publications.
