# C5 official top-k wrapper

Objective: reduce wrapper launches around official FlashInfer exact top-k while
preserving output values/indices bit-for-bit, sorted descending values,
deterministic SMALL tie order, signed-zero radix ordering, and -1 for every
nonfinite selected value. Input remains FP32 [Q,N], K in 1..2048, capacity min(K,N),
noncontiguous inputs keep the existing contiguous conversion, Q=0 remains valid.

Keep the official FlashInfer selection and both official ordering kernels.
Do not modify third-party source, switch to approximate selection, drop sorting,
or change global graph policy. GPU work is forbidden during root's quiet C4 run
until explicitly released. CPU preparation/tests are allowed after this plan.

Candidate scope: operators/deepseek_v32/indexer/selection.py and its unit tests;
new lazy helper in that module if needed. Exact CUDA/eager/capture differentials,
API wall timing, kernel activity and transient allocation measurements required
before promotion. Whole-model acceptance/publication remain with root.
