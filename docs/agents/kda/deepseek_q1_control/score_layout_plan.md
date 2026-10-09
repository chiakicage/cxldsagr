# Official Q1 score view: task and plan

The official bridge allocates stride65792 for logicalN65537 and initializes
every physical tail column to-inf, checked by raw receipt `_04`. A logical
width65537 view selects FlashInfer's scalar path; exposing the initialized
physical extent enables vector width4 for k2048. The source-bound component
comparison below measures both views with identical values and selections.

Candidate: expose this existing backing for exact_topk only. Do not copy
scores, change the official kernel or alter its logical causal range. Keep
prefill coarse output unchanged, public defaults logical, and hint reductions
restricted to original logicalN. This should allow the official FlashInfer
vector path while retaining exact values, signed-key tie ordering and IDs.

Independent check covers real L0-L2 official scores, all ties, changed score
inputs and causal-inf columns, with4 changed graph replays each. Benchmark
uses30 balanced AB/BA pairs of one full exact_topk graph replay, with10 warmups,
on GPU2/CPUs16-23. Official scoring occurs outside both arms; no timed score
copy or preparation is hidden. Source/input/native identities must match.
Only a measured win permits integration and fresh full-model revalidation.


Accepted source-bound check `q1_score_layout_check_20261008_04` passes all
four variants and graph replays for L0-L2. Bench `q1_score_layout_bench_20261008_01`
binds the same official decode artifact, loaded FlashInfer topk.so and actual
Triton mask binaries. Logical/padded medians are65.648/51.264 us (L0),
66.096/51.376 us (L1),64.160/50.624 us (L2). Thirty alternating pairs each,
one complete top-k call per sample, all samples retained. Source snapshots
identify the pre-integration harness and producer. Final production matrix
`deepseek_h64k_a1_official_20261008_03` independently checks the exposed physical
view and unchanged logical-N hint, then measures all four methods with cold
restoration before each offload sample. Stage profiles and operator profile
`deepseek_h64k_a1_official_mfu_profile_20261008_04` passed full node attribution
and output comparison. The component and full-model results keep separate
timing boundaries; no component gains are added to predict model speedup.
