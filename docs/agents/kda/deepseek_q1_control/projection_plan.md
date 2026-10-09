# Q1 graph projection overlap: task, draft, executable plan

Objective: reduce the current local HBM GPU critical path toward the official
SGLang window while preserving the complete L0–L2 workload and exact outputs.
Baseline is current production including promoted Q1 scale layout and page64
packing. Only Q1 SM90 graph projection is eligible; eager and Q>1 stay unchanged.

Source DAG: normalized `x` independently feeds Q-A/RMS/Q-B/Q-absorb, KV-A/RMS,
and index-K/LayerNorm. Index-Q depends on Q-A/RMS; the two RoPE pairs join these
branches. Index weights require x and query scales. No cache state is involved.

Official evidence: `deepseek_v2.py::forward_absorb_prepare` forks q/kv norms;
`nsa_indexer.py::_get_q_k_bf16` forks index query/key projections, then rejoins
before paired RoPE. It also splits Hadamard and index quantization. The accepted
official HBM trace has 24 kernels on stream142 and112 on stream7. Their actual
GPU interval intersection is42.751 us (13.504/13.696/15.551 us per layer).
No PDL flag is changed by this candidate.

Local prior operator trace has three-layer kernel sums40.000 us for KV-A and
37.856 us for index-K (including their original quantization/layout preparation).
They are serial today; they can precede the Q-path join. These are possible
overlap bounds, not estimated guaranteed speedups. Current scale layout already
removes some of this duration, and simultaneous GEMMs may contend for SMs/HBM.

Draft: create one private side stream owned by each projection object. Fork it
after normalized x is ready. Side stream executes KV-A/RMS and index-K/LayerNorm;
main stream executes Q-A/RMS, Q-B/Q-absorb and index-Q. Join before paired RoPE.
All arithmetic and library APIs remain identical. Explicit stream lifetime
tracking protects x read on side and side results consumed on main. Fork/join
is in the captured graph; no tensor is computed before the timed API to hide cost.

Implementation gates:
1. Private module, not production default. Bind before graph capture. First
   capture correctness replays changed hidden and changed FP32 positions using
   real L0–L2 weights. All six Projected outputs must match production bitwise.
2. Measure complete projection graphs with balanced AB/BA on GPU2, CPU16–23;
   warmup/capture are separate, and graph contains all preparation and joins.
3. Profile graph nodes and verify actual overlap with valid dependencies; do
   not infer benefit merely from two stream IDs.
4. Only a component win permits fresh full HBM three-layer graph A/B check and
   clean timing. Parent owns formal four-method acceptance/publication.

Risks: side-stream allocations must remain owned until both streams complete;
exceptions must preserve primary and cleanup errors. Capture fork/join must
reconverge. The stream owner remains attached to the model; graph destruction
and model close drain CUDA before storage may be reused. No global SM-count
knob or concurrent backend setting mutation is introduced.


2026-10-08 evaluation update: profile `_02` correlates all32 projection kernels
in every layer/arm to its unique graph launch. Cross-stream GPU interval
intersection is0 for all three candidate scopes, including after in-range
warmup. Candidate node-profile windows are111.104/123.200/122.080 us versus
105.313/116.928/114.848 us baseline. This does not establish hardware overlap;
node tracing changes the measured scheduling outcome relative to clean timing.
Keep the measured scheduling change as a candidate on its clean timing merit,
and do not describe its mechanism as measured compute overlap.

Independent complete HBM check `q1_projection_model_check_20261008_01` passed
bitwise prefix logits, eager/graph hidden+logits and two changed graph input
replays. Independent20-pair AB/BA bench `q1_projection_model_bench_20261008_01`
has baseline median1.9262625 ms and candidate1.850425 ms, paired median
-0.0811595 ms with16/20 wins. All samples are retained, including four losses.
Graph private reserved increases from56,623,104 B to62,914,560 B (+6,291,456 B).
This evidence supports production integration for formal four-method validation;
it does not yet establish the final production latency or close the SGLang gap.
