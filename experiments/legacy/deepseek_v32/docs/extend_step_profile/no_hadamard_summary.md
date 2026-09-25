# Extend step profile

No-Hadamard, V3.2 FP8 QK synthetic attention, new tokens=4096, chunk=4096.

GPU times below are exclusive and summed across chunks. Event times in summary.json are inclusive (parents include children); do not sum them. E2E is measured separately without instrumentation. CUPTI kernel times are from one warmed iteration.

## History 65536

E2E: 69.560 ms; attributed GPU: 69.083 ms.

| Step | Calls | GPU ms (exclusive) |
|---|---:|---:|
| wq_a/quantize | 1 | 0.0966 |
| wq_a/gemm | 1 | 0.2438 |
| wq_a | 1 | 0.0000 |
| q_norm | 1 | 0.0086 |
| wq_b/quantize | 1 | 0.0086 |
| wq_b/gemm | 1 | 0.8304 |
| wq_b | 1 | 0.0000 |
| wkv_a/quantize | 1 | 0.0874 |
| wkv_a/gemm | 1 | 0.0987 |
| wkv_a | 1 | 0.0000 |
| kv_norm | 1 | 0.0102 |
| mla/rope | 1 | 0.3123 |
| wk_b/quantize | 1 | 0.2217 |
| wk_b/gemm | 1 | 0.8303 |
| wk_b | 1 | 0.0009 |
| index_wqi/quantize | 1 | 0.0208 |
| index_wqi/gemm | 1 | 0.2796 |
| index_wqi | 1 | 0.0000 |
| index_wki/quantize | 1 | 0.0862 |
| index_wki/gemm | 1 | 0.0353 |
| index_wki | 1 | 0.0000 |
| index_norm | 1 | 0.0031 |
| index/rope | 1 | 0.1273 |
| index_weights | 1 | 0.0824 |
| projection/layout | 1 | 3.8012 |
| cache/append | 1 | 0.0124 |
| index/q_quantize | 1 | 1.7612 |
| index/mqa_logits | 1 | 19.7871 |
| index/weights_bounds | 1 | 0.0104 |
| index/topk | 1 | 3.5743 |
| mla/prefill | 1 | 30.9654 |
| wv_b/quantize | 1 | 0.9406 |
| wv_b/gemm | 1 | 0.5142 |
| wv_b | 1 | 0.0009 |
| wv_b/layout | 1 | 1.6646 |
| wo/quantize | 1 | 0.2196 |
| wo/gemm | 1 | 2.3556 |
| wo | 1 | 0.0000 |
| wo/layout | 1 | 0.0000 |
| loop/output_copy | 1 | 0.0910 |
