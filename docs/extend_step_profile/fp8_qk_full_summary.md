# Extend step profile

V3.2 FP8 QK (MXFP8 latent QK, BF16 RoPE QK, FP8 PV) synthetic attention, new tokens=4096, chunk=4096.

GPU times below are exclusive and summed across chunks. Event times in summary.json are inclusive (parents include children); do not sum them. E2E is measured separately without instrumentation. CUPTI kernel times are from one warmed iteration.

## History 65536

E2E: 77.800 ms; attributed GPU: 77.304 ms.

| Step | Calls | GPU ms (exclusive) |
|---|---:|---:|
| wq_a/quantize | 1 | 0.0895 |
| wq_a/gemm | 1 | 0.2439 |
| wq_a | 1 | 0.0000 |
| q_norm | 1 | 0.0087 |
| wq_b/quantize | 1 | 0.0086 |
| wq_b/gemm | 1 | 0.8307 |
| wq_b | 1 | 0.0000 |
| wkv_a/quantize | 1 | 0.0862 |
| wkv_a/gemm | 1 | 0.0989 |
| wkv_a | 1 | 0.0000 |
| kv_norm | 1 | 0.0098 |
| mla/rope | 1 | 0.3103 |
| wk_b/quantize | 1 | 0.2197 |
| wk_b/gemm | 1 | 0.8324 |
| wk_b | 1 | 0.0009 |
| index_wqi/quantize | 1 | 0.0206 |
| index_wqi/gemm | 1 | 0.2804 |
| index_wqi | 1 | 0.0000 |
| index_wki/quantize | 1 | 0.0875 |
| index_wki/gemm | 1 | 0.0351 |
| index_wki | 1 | 0.0000 |
| index_norm | 1 | 0.0032 |
| index/rope | 1 | 0.1266 |
| index/q_hadamard | 1 | 8.1535 |
| index/k_hadamard | 1 | 0.0502 |
| index_weights | 1 | 0.0806 |
| projection/layout | 1 | 3.8017 |
| cache/append | 1 | 0.0121 |
| index/q_quantize | 1 | 1.7596 |
| index/mqa_logits | 1 | 19.7948 |
| index/weights_bounds | 1 | 0.0098 |
| index/topk | 1 | 3.5700 |
| mla/prefill | 1 | 30.9875 |
| wv_b/quantize | 1 | 0.9385 |
| wv_b/gemm | 1 | 0.5169 |
| wv_b | 1 | 0.0008 |
| wv_b/layout | 1 | 1.6651 |
| wo/quantize | 1 | 0.2211 |
| wo/gemm | 1 | 2.3547 |
| wo | 1 | 0.0000 |
| wo/layout | 1 | 0.0000 |
| loop/output_copy | 1 | 0.0934 |
