# Official ECHO Q1 prefetch bridge

Objective: test direct reuse of the pinned ECHO decode paged fused MQA and
prefetch kernel in the main PyTorch 2.12 environment. The user explicitly
prioritized official ECHO; do not replace it with a new split score/histogram
algorithm. Production dispatch is outside this isolated candidate; integration
follows separate acceptance.

Inputs: one FP8 E4M3 query `[1, 1, 64, 128]`, current FP8 index keys and FP32
scales, FP32 weights, page size 64, H64K plus one current query, initialized
mapped pinned BF16 MLA history records of width 576. Primary GPU: GPU 1,
H200/SM90, CPU affinity 8–15. GPU 7 is excluded.

The kernel is included directly from ECHO revision
`bc1b75c1000010d0ac6f032ebaac283255c050b1`; no upstream source is modified and no
Torch 2.8 binary is loaded. The bridge uses TVM FFI and CUDA driver/runtime APIs.
The current optional checkout is sufficient for this independent validation,
not a claim that the dependency is distributed or integrated into the model.

Correctness requirements:

- Preserve the official kernel and scheduler, launch configuration, FP32 score
  arithmetic, strict `score > decode_topk_logits` predicate, and 64-record
  temporary prefetch cap. Exact top-k remains a separate consumer.
- Compare all valid raw score bits and exact top-k on the three saved real
  inputs; check causal padding independently.
- Verify every staged host ID is initialized, eligible, and unique; the BF16
  staged record must be bitwise equal to its pinned host record. Check encoded
  host mapping, counters, unchanged resident mappings, and buffer canaries.
- Cover cold, warm, unsaturated, saturated, empty-prefetch, and changed graph
  input/threshold cases. Reset the same initial state before each timed call.
- Keep the official staging ABI separate from the local persistent slot pool.
  The existing global coarse-bin prefill proof is not a decode-policy proof.

Quality target: a source-bound, reproducible working bridge plus clean timings
for the native official invocation and the full per-call packing/metadata API.
Do not claim a full-model speedup from the operator measurement.

Validation: `python -m experiments.deepseek_v32_echo_official.src.q1_official_prefetch check --run-id q1_official_prefetch_check_20261008_01 --physical-device 1`.

Evaluation: the same entry with `bench`, a new run ID, and an explicit matching
acceptance path. Correctness and clean timing run in separate processes.

Promotion requires exact score/top-k acceptance, official state-policy
acceptance, measured improvement at a stated boundary, separately accepted
model/cache integration and capacity planning, and remeasurement of affected
experiments. This independent API alone cannot promote production dispatch.
