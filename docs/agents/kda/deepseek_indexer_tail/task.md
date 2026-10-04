# DeepSeek official indexer causal tail

Objective: remove full Q×N compare/masked-fill scans from the official resident
indexer while preserving the returned causal FP32 logits and official DeepGEMM
arithmetic. Preserve exact finite prefixes, -inf for every column >= start+row+1,
and logical [Q,N] with its physical row stride. Typical shape Q=1024,N=H+Q,
start=H; also retain Q=1, ragged dimensions, and N>start+Q.

Hardware: Hopper SM90. GPU usage must wait for root's explicit test window while
its frozen C3 profile runs. No source changes to third-party DeepGEMM/FlashInfer,
selection policy, models, or graph implementation. Source owner: echo.py and a
new helper/indexer tests if measurements justify a kernel.

Promotion: bit-exact logits prefix and invalid tails; unchanged exact-top-k
values/indices, including tied inputs; no writes to physical padding or adjacent
storage; stream/graph behavior verified if adding a kernel; actual GPU activity
and complete API wall cost outperform current full scan at target shapes.
Complete-model acceptance and publication remain with root.
