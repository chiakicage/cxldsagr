# Extend step profile

V3.2 FP8 QK (MXFP8 latent QK, BF16 RoPE QK, FP8 PV) synthetic attention, new tokens=4096, chunk=512.

GPU times below are exclusive and summed across chunks. Event times in summary.json are inclusive (parents include children); do not sum them. E2E is measured separately without instrumentation. CUPTI kernel times are from one warmed iteration.

## History 65536

E2E: 76.849 ms; attributed GPU: 75.959 ms.

| Step | Calls | GPU ms (exclusive) |
|---|---:|---:|
| wq_a/quantize | 8 | 0.0757 |
| wq_a/gemm | 8 | 0.4196 |
| wq_a | 8 | 0.0000 |
| q_norm | 8 | 0.0186 |
| wq_b/quantize | 8 | 0.0141 |
| wq_b/gemm | 8 | 1.0242 |
| wq_b | 8 | 0.0000 |
| wkv_a/quantize | 8 | 0.0776 |
| wkv_a/gemm | 8 | 0.1943 |
| wkv_a | 8 | 0.0000 |
| kv_norm | 8 | 0.0252 |
| mla/rope | 8 | 0.1742 |
| wk_b/quantize | 8 | 0.0886 |
| wk_b/gemm | 8 | 0.7120 |
| wk_b | 8 | 0.0058 |
| index_wqi/quantize | 8 | 0.0276 |
| index_wqi/gemm | 8 | 0.4618 |
| index_wqi | 8 | 0.0000 |
| index_wki/quantize | 8 | 0.0839 |
| index_wki/gemm | 8 | 0.1142 |
| index_wki | 8 | 0.0000 |
| index_norm | 8 | 0.0119 |
| index/rope | 8 | 0.0936 |
| index/q_hadamard | 8 | 2.1405 |
| index/k_hadamard | 8 | 0.2033 |
| index_weights | 8 | 0.1060 |
| projection/layout | 8 | 3.5351 |
| cache/append | 8 | 0.0214 |
| index/q_quantize | 8 | 0.6697 |
| index/mqa_logits | 8 | 25.3851 |
| index/weights_bounds | 8 | 0.0511 |
| index/topk | 8 | 3.4892 |
| mla/prefill | 8 | 31.4783 |
| wv_b/quantize | 8 | 0.6285 |
| wv_b/gemm | 8 | 0.3845 |
| wv_b | 8 | 0.0072 |
| wv_b/layout | 8 | 1.4326 |
| wo/quantize | 8 | 0.0900 |
| wo/gemm | 8 | 2.6749 |
| wo | 8 | 0.0000 |
| wo/layout | 8 | 0.0000 |
| loop/output_copy | 1 | 0.0385 |
