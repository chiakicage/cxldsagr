# DeepSeek dense MLP packed gate/up task

Status: the temporary candidate passed the granted GPU 3 correctness screen
(17 fixtures, 51 exact variant comparisons). The separately granted exclusive
eager benchmark also completed: the combined candidate improves all nine
full-MLP medians. The extended driver then passed all 72 graph correctness
cases under a separate grant. Complete graph timing also passed its numerical
and record audits. Root selected the combined candidate and authorized its
narrow production integration. Independent source review, 37 production GPU
tests and nine actual-checkpoint-weight cases with 27 changed-input graph
replays passed. GPU grants are closed; integrated performance remains unmeasured.

Reduce the complete dense MLP gate/up/SwiGLU cost by quantizing their identical
activation once and writing the two separate official DeepGEMM results directly
into disjoint halves of one BF16 `[Q,2N]` allocation. The existing FlashInfer
SiLU/multiply consumes that packed allocation. Keep the down projection's own
quantization and official GEMM unchanged.

Primary workload: SM90/Hopper; BF16 input `[Q,7168]`, independent FP8 gate/up
weights `[18432,7168]` with checkpoint FP32 128x128 scales, and independent down
weight `[7168,18432]`. Q=1024 is the history chunk, Q=128 the candidate graph
shape, and Q=121 the eager fallback. Small aligned N/K and irregular Q fixtures
exercise descriptor boundaries before checkpoint-shaped measurements.

Preserve both gate/up GEMM calls, their individual M/N/K, recipe `(1,128,128)`,
weight/scale tensors, accumulation and BF16 rounding. Share only the exact
compiled official per-token activation quantization of the same unchanged
input. Do not concatenate weights, enlarge a GEMM, change the quantizer, share
weights across blocks, change down quantization, or modify norm/transport code.
No installed/upstream source edits are allowed.

Initial prototype domain is BF16 rank-two nonempty CUDA SM90 input with unit
inner stride,
K divisible by 128, N divisible by 64 and equal gate/up N/K. Unsupported or
padded shapes must retain the original production path if this is integrated.
The prototype may reject them explicitly; it must not hide output copies as
successful direct packed stores. Grouped MoE and shared-expert rollout are
outside this first candidate.

Acceptance requires byte-exact quantized activation/scales, both GEMM outputs,
SiLU result and unchanged down result; unchanged inputs/weights/scales;
no cross-half or guard writes; and caller-stream/graph replay ownership.
Only after exactness may an exclusive timing run compare complete helper and
complete MLP cost, including quantization, allocation, views, GEMMs and SiLU.
Record graph replay time separately from eager wall completion. Source checks
must cover the installed official path and temporary harness.

Promote only if measured complete cost improves without correctness or memory
accounting regressions. Prefer optional output/prepared-input parameters that
retain the two `CheckpointLinear.__call__` scopes. Prepared activations must be
call-local, with no object-level cache. Count shared quantization once in a
defined matrix API, or explicitly revise attribution for a paired boundary.
Current instrumentation wraps `CheckpointLinear.__call__`, so bypassing it can
hide gate/up API invocations. Root owns later integration, combined checkpoint
validation, formal serving rerun and report replacement.
