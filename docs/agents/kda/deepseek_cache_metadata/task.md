# DeepSeek finite-pool metadata optimization

Objective: reduce cold-build append, resident exact-selection, and dense all-hit
metadata CPU/GPU launch cost for `deepseek_v32_motivation`, without changing P,
NH, exact top-k, FIFO priority events, host ownership, or transient semantics.
This is one component of the active end-to-end latency/MFU goal; operator-only
measurements cannot prove that full objective.

Inputs: Hopper/SM90, P=65536, C=1024, H=65536, candidate A=128,
BF16 opaque record width 576 for the target model. Helpers must retain supported
opaque widths/dtypes, fragmented host pages, negative selection padding, and
candidate rows beyond P+1. CPU remains the executable checked reference.

Acceptance: exact records/maps/priorities/clocks/counters against checked paths
under duplicates, fragmented pages, competing users, rollback/reuse, and clock
rollover. GPU 2 only. Improve synchronized metadata harness latency at real
selection/append shapes. Root owns complete-model and full-trajectory acceptance.

Validation commands:

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest cache/tests/test_sparse_token_pool.py cache/tests/test_sparse_token_cache.py cache/tests/test_transient_suffix.py models/deepseek_v32/tests/test_pool_prefetch.py operators/deepseek_v32/indexer/tests/test_echo_cache_ops.py
```
