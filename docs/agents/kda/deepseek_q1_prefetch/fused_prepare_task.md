# Private Q1 fused input preparation

Status: private candidate passes independent GPU1 correctness; clean timing and
profiling await the parent's measurement slot. See `fused_prepare_checkpoint.md`.
Production sources and selected reports remain unchanged.

Objective: reduce the complete local ECHO Q1 indexer API cost while retaining
the pinned official score/prefetch algorithm, exact score/top-k bits and every
local publication/cleanup contract. This is one component of the active objective
to close the SGLang gap, not a redefinition of that objective.

Inputs: FP8 `K[N,128]`, FP32 `scales[N]`, Q1/H64/D128 query and weights, initialized
history `H=N-1`, legal host page table and exclusive local prefetch lease. Formal
component inputs are saved real checkpoint L0-L2 at H65,536/A1. Hardware is SM90;
the parent assigns a GPU and matching CPU NUMA affinity before execution.

Candidate `q1-fused-page-and-stage-prepare-v1` combines current page64 packing,
identity block-table publication and official host-token-table/staging reset.
The official scheduler, fused score/prefetch kernel, causal clean, promotion and
cleanup are reused unchanged. No persistent packed state or new storage class is
introduced. Every invocation reads current K/scales/page bytes.

Correctness requirements: bitwise packed keys/scales, zero page padding, exact
host IDs including host zero, pending token mapped to safe zero, stage IDs -1,
counter zero, unchanged initialized maps/records/priority, full journal/statistic
reset or verified prepared-token consumption, and registration of the original
cleanup owner before launch. Retain context/history and host-page device asserts.
Cold saturated predictions may differ only under the existing official policy;
each call must independently prove strict eligibility, unique IDs, exact copied
record bits and legal promotion. Scores and exact top-k remain bitwise equal.

Allowed files: new `experiments/deepseek_v32_echo_official/src/q1_fused_prepare*`
files, dedicated tests and these component documents. No production, upstream or
existing experiment source edits during the formal source freeze. No GPU launch
before explicit parent scheduling.

Validation and performance use separate `q1_fused_prepare_run` modes. Check
artifacts live under `/tmp/cxldsagr-checks/`; clean benchmark/profile outputs use
new experiment run directories. A receipt binds all source, native/toolchain,
runtime, actual GPU and input identities. Every timing sample restores caller
metadata outside the timed interval and measures the complete indexer call,
including packing, scheduler, stage reset, official logits/clean and promotion.
Cleanup is measured equally for both variants. Top-k, pool preparation and exact
recall remain outside this operator boundary and are stated explicitly.

Promotion requires complete independent check, clean paired complete-API wins,
separate profile, then the parent's full-model check and formal refresh. A
microbenchmark or compilation success alone cannot promote this candidate.
