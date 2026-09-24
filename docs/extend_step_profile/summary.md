# Extend step profile

FP8 synthetic attention, new tokens=4096, chunk=512.

GPU times below are exclusive and summed across chunks. Event times in summary.json are inclusive (parents include children); do not sum them. E2E is measured separately without instrumentation. CUPTI kernel times are from one warmed iteration.

## History 4096

E2E: 50.598 ms; attributed GPU: 49.190 ms.

| Step | Calls | GPU ms (exclusive) |
|---|---:|---:|
| wq_a/quantize | 8 | 0.0748 |
| wq_a/gemm | 8 | 0.4196 |
| wq_a | 8 | 0.0000 |
| q_norm | 8 | 0.0184 |
| wq_b/quantize | 8 | 0.0142 |
| wq_b/gemm | 8 | 1.0260 |
| wq_b | 8 | 0.0000 |
| wkv_a/quantize | 8 | 0.0768 |
| wkv_a/gemm | 8 | 0.1938 |
| wkv_a | 8 | 0.0000 |
| kv_norm | 8 | 0.0251 |
| mla/rope | 8 | 0.1721 |
| wk_b/quantize | 8 | 0.0886 |
| wk_b/gemm | 8 | 0.7060 |
| wk_b | 8 | 0.0058 |
| index_wqi/quantize | 8 | 0.0269 |
| index_wqi/gemm | 8 | 0.4605 |
| index_wqi | 8 | 0.0000 |
| index_wki/quantize | 8 | 0.0872 |
| index_wki/gemm | 8 | 0.1131 |
| index_wki | 8 | 0.0000 |
| index_norm | 8 | 0.0115 |
| index/rope | 8 | 0.0922 |
| index/q_hadamard | 8 | 2.1242 |
| index/k_hadamard | 8 | 0.2019 |
| index_weights | 8 | 0.1063 |
| projection/layout | 8 | 3.5522 |
| cache/append | 8 | 0.0204 |
| index/q_quantize | 8 | 0.6698 |
| index/mqa_logits | 8 | 2.4051 |
| index/weights_bounds | 8 | 0.0452 |
| index/topk | 8 | 0.4536 |
| mla/prefill | 8 | 30.7221 |
| wv_b/quantize | 8 | 0.6246 |
| wv_b/gemm | 8 | 0.3989 |
| wv_b | 8 | 0.0074 |
| wv_b/layout | 8 | 1.4394 |
| wo/quantize | 8 | 0.0895 |
| wo/gemm | 8 | 2.6787 |
| wo | 8 | 0.0000 |
| wo/layout | 8 | 0.0000 |
| loop/output_copy | 1 | 0.0378 |

## History 65536

E2E: 77.537 ms; attributed GPU: 76.312 ms.

| Step | Calls | GPU ms (exclusive) |
|---|---:|---:|
| wq_a/quantize | 8 | 0.0755 |
| wq_a/gemm | 8 | 0.4215 |
| wq_a | 8 | 0.0000 |
| q_norm | 8 | 0.0186 |
| wq_b/quantize | 8 | 0.0142 |
| wq_b/gemm | 8 | 1.0344 |
| wq_b | 8 | 0.0000 |
| wkv_a/quantize | 8 | 0.0771 |
| wkv_a/gemm | 8 | 0.1954 |
| wkv_a | 8 | 0.0000 |
| kv_norm | 8 | 0.0248 |
| mla/rope | 8 | 0.1739 |
| wk_b/quantize | 8 | 0.0891 |
| wk_b/gemm | 8 | 0.7010 |
| wk_b | 8 | 0.0058 |
| index_wqi/quantize | 8 | 0.0276 |
| index_wqi/gemm | 8 | 0.4630 |
| index_wqi | 8 | 0.0000 |
| index_wki/quantize | 8 | 0.0855 |
| index_wki/gemm | 8 | 0.1137 |
| index_wki | 8 | 0.0000 |
| index_norm | 8 | 0.0113 |
| index/rope | 8 | 0.0937 |
| index/q_hadamard | 8 | 2.2118 |
| index/k_hadamard | 8 | 0.2048 |
| index_weights | 8 | 0.1072 |
| projection/layout | 8 | 3.5554 |
| cache/append | 8 | 0.0204 |
| index/q_quantize | 8 | 0.6765 |
| index/mqa_logits | 8 | 25.5145 |
| index/weights_bounds | 8 | 0.0483 |
| index/topk | 8 | 3.4874 |
| mla/prefill | 8 | 31.5745 |
| wv_b/quantize | 8 | 0.6401 |
| wv_b/gemm | 8 | 0.3870 |
| wv_b | 8 | 0.0073 |
| wv_b/layout | 8 | 1.4359 |
| wo/quantize | 8 | 0.0902 |
| wo/gemm | 8 | 2.6859 |
| wo | 8 | 0.0000 |
| wo/layout | 8 | 0.0000 |
| loop/output_copy | 1 | 0.0384 |
