# Q1 control cost contract

Target H64K/A1 on H200 SM90, preserving actual complete graph work, cache state,
precision, input/output, and timing boundaries. Parent task covers reaching the
official SGLang GPU window; this component targets redundant preparation work.

Current accepted full-model evidence is `deepseek_h64k_a1_official_20261008_03`:
HBM L0–L2 window 1.073347 ms and graph-node idle 0.068832 ms. The local control
classification includes useful layout transformations. One graph launch owns
all nodes, so graph idle is not per-kernel Python launch latency. Prior full
model results were superseded; current component A/B evidence is listed in
`checkpoint.md`.

Candidate: Q1 dense FP8 linear writes its scales directly into the exact official
MN-major TMA-aligned layout. Keep the existing quantization arithmetic and
official DeepGEMM matrix API. Public quantization stays row-major by default;
Q>1 and grouped paths stay on their existing paths. No persistent storage.

Correctness requires bitwise FP8, FP32 scale, and GEMM output equality, including
source strides, dtypes, special values, graph replay, and source mutation.
Preserve scoped source/compiled identity. Measure full quantize+official GEMM
API separately from correctness. Promote only if measured Q1 cost improves;
full-model claims require the parent independent acceptance/bench/profile run.
