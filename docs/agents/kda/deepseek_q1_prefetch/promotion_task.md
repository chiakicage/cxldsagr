# Official staging promotion task

Optimize only the local `official_prefetch_promote` adapter on SM90/H200. The
parent's formal `_01` timeline attributes 107.488 us over three layers to the
single-CTA baseline. This is a profiling observation, not a clean benchmark.
The fixed official fused score/prefetch kernel and production sources stay unchanged.

Inputs: up to 64 contiguous BF16 records of width 576; uint32 reservation
attempts; temporary host tags; prepared distinct FIFO slots; occupied or empty
slot owners; allocation journal and three exact int64 statistics. Main formal
shape is N=65,537, P=65,600, with both consecutive and random prepared slots.
Every contiguous BF16 base alignment, including a two-byte offset, remains valid.

Correctness is bitwise for records and exact for maps, journal, eviction and
statistics. Host zero is valid. Saturation, 0/1/17/64 records, >64 attempts,
occupied slots, untouched storage and cleanup behavior must remain correct.
Validation must preserve every baseline device assertion. No valid-input data
races, retries, recovery, or new state allocation are allowed. Failed calls retain
the existing asynchronous error and poisoned-owner contract.

Use GPU1 and CPU8-15. Root owns NCU profiling; provide a one-call ProfilerAPI/NVTX
mode usable on GPU2. All candidate CUDA source lives in the experiment's private
output source directory. Deliver evidence and a reviewable patch; do not promote
into production. A new experiment harness under `src/` is parent-authorized and
outside the in-flight model source manifest.

Promotion requires independent check acceptance, the unchanged 22 adapter tests
with only the module swapped, additional random-bit/alignment/occupied-slot cases,
and faster clean full-call graph event timing across both slot layouts. Report
all phases of a split implementation. Harness gains are not full-model gains.
