# DeepGEMM v3.2 SM120 benchmark results

Date: 2026-07-01 21:34:02 UTC

Environment:

```text
GPU: NVIDIA GeForce RTX 5080
PyTorch: 2.12.1+cu130
PyTorch CUDA runtime: 13.0
Compute capability: (12, 0)
DeepGEMM: 2.5.0
SMs used by DeepGEMM: 84
warmups=3, iters=10, quick=False
```

| Section | Name | Mode | Shape | ms | TFLOP/s | GB/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| projection_gemm | `decode_m1_wq_a` | `fp8` | m=1, n=1536, k=7168 | 0.0373 | 0.59 | 295.36 |
| projection_gemm | `decode_m1_wq_a` | `fp8_fp4w` | m=1, n=1536, k=7168 | 0.0238 | 0.92 | 289.04 |
| projection_gemm | `decode_m1_wq_b` | `fp8` | m=1, n=24576, k=1536 | 0.0366 | 2.06 | 1032.44 |
| projection_gemm | `decode_m1_wq_b` | `fp8_fp4w` | m=1, n=24576, k=1536 | 0.0307 | 2.46 | 769.17 |
| projection_gemm | `decode_m1_wkv_a` | `fp8` | m=1, n=576, k=7168 | 0.0377 | 0.22 | 109.88 |
| projection_gemm | `decode_m1_wkv_a` | `fp8_fp4w` | m=1, n=576, k=7168 | 0.0234 | 0.35 | 110.54 |
| projection_gemm | `decode_m1_wo` | `fp8` | m=1, n=7168, k=16384 | 0.1526 | 1.54 | 769.90 |
| projection_gemm | `decode_m1_wo` | `fp8_fp4w` | m=1, n=7168, k=16384 | 0.0975 | 2.41 | 753.43 |
| projection_gemm | `decode_m1_index_wqi` | `fp8` | m=1, n=8192, k=1536 | 0.0313 | 0.80 | 402.45 |
| projection_gemm | `decode_m1_index_wqi` | `fp8_fp4w` | m=1, n=8192, k=1536 | 0.0207 | 1.22 | 381.66 |
| projection_gemm | `decode_m1_index_wki` | `fp8` | m=1, n=128, k=7168 | 0.0356 | 0.05 | 25.98 |
| projection_gemm | `decode_m1_index_wki` | `fp8_fp4w` | m=1, n=128, k=7168 | 0.0254 | 0.07 | 22.86 |
| projection_gemm | `decode_m1_index_weights` | `fp8` | m=1, n=64, k=7168 | 0.0372 | 0.02 | 12.53 |
| projection_gemm | `decode_m1_index_weights` | `fp8_fp4w` | m=1, n=64, k=7168 | 0.0239 | 0.04 | 12.29 |
| projection_gemm | `decode_m16_wq_a` | `fp8` | m=16, n=1536, k=7168 | 0.0372 | 9.46 | 300.31 |
| projection_gemm | `decode_m16_wq_a` | `fp8_fp4w` | m=16, n=1536, k=7168 | 0.0271 | 13.01 | 260.21 |
| projection_gemm | `decode_m16_wq_b` | `fp8` | m=16, n=24576, k=1536 | 0.0376 | 32.11 | 1025.35 |
| projection_gemm | `decode_m16_wq_b` | `fp8_fp4w` | m=16, n=24576, k=1536 | 0.0270 | 44.68 | 902.76 |
| projection_gemm | `decode_m16_wkv_a` | `fp8` | m=16, n=576, k=7168 | 0.0353 | 3.75 | 120.95 |
| projection_gemm | `decode_m16_wkv_a` | `fp8_fp4w` | m=16, n=576, k=7168 | 0.0228 | 5.79 | 119.12 |
| projection_gemm | `decode_m16_wo` | `fp8` | m=16, n=7168, k=16384 | 0.1500 | 25.05 | 786.34 |
| projection_gemm | `decode_m16_wo` | `fp8_fp4w` | m=16, n=7168, k=16384 | 0.0980 | 38.34 | 753.86 |
| projection_gemm | `decode_m16_index_wqi` | `fp8` | m=16, n=8192, k=1536 | 0.0307 | 13.10 | 418.93 |
| projection_gemm | `decode_m16_index_wqi` | `fp8_fp4w` | m=16, n=8192, k=1536 | 0.0188 | 21.40 | 433.31 |
| projection_gemm | `decode_m16_index_wki` | `fp8` | m=16, n=128, k=7168 | 0.0347 | 0.85 | 30.01 |
| projection_gemm | `decode_m16_index_wki` | `fp8_fp4w` | m=16, n=128, k=7168 | 0.0257 | 1.14 | 27.12 |
| projection_gemm | `decode_m16_index_weights` | `fp8` | m=16, n=64, k=7168 | 0.0347 | 0.42 | 16.71 |
| projection_gemm | `decode_m16_index_weights` | `fp8_fp4w` | m=16, n=64, k=7168 | 0.0269 | 0.55 | 15.12 |
| projection_gemm | `decode_m128_wq_a` | `fp8` | m=128, n=1536, k=7168 | 0.0378 | 74.57 | 326.82 |
| projection_gemm | `decode_m128_wq_a` | `fp8_fp4w` | m=128, n=1536, k=7168 | 0.0248 | 113.80 | 331.91 |
| projection_gemm | `decode_m128_wq_b` | `fp8` | m=128, n=24576, k=1536 | 0.0530 | 182.37 | 835.12 |
| projection_gemm | `decode_m128_wq_b` | `fp8_fp4w` | m=128, n=24576, k=1536 | 0.0381 | 253.97 | 790.70 |
| projection_gemm | `decode_m128_wkv_a` | `fp8` | m=128, n=576, k=7168 | 0.0351 | 30.10 | 148.73 |
| projection_gemm | `decode_m128_wkv_a` | `fp8_fp4w` | m=128, n=576, k=7168 | 0.0257 | 41.16 | 143.07 |
| projection_gemm | `decode_m128_wo` | `fp8` | m=128, n=7168, k=16384 | 0.1731 | 173.69 | 701.72 |
| projection_gemm | `decode_m128_wo` | `fp8_fp4w` | m=128, n=7168, k=16384 | 0.1337 | 224.91 | 579.01 |
| projection_gemm | `decode_m128_index_wqi` | `fp8` | m=128, n=8192, k=1536 | 0.0340 | 94.83 | 438.23 |
| projection_gemm | `decode_m128_index_wqi` | `fp8_fp4w` | m=128, n=8192, k=1536 | 0.0194 | 166.25 | 524.58 |
| projection_gemm | `decode_m128_index_wki` | `fp8` | m=128, n=128, k=7168 | 0.0405 | 5.79 | 46.79 |
| projection_gemm | `decode_m128_index_wki` | `fp8_fp4w` | m=128, n=128, k=7168 | 0.0228 | 10.28 | 67.96 |
| projection_gemm | `decode_m128_index_weights` | `fp8` | m=128, n=64, k=7168 | 0.0359 | 3.27 | 39.60 |
| projection_gemm | `decode_m128_index_weights` | `fp8_fp4w` | m=128, n=64, k=7168 | 0.0232 | 5.06 | 53.82 |
| projection_gemm | `prefill_m512_wq_a` | `fp8` | m=512, n=1536, k=7168 | 0.0552 | 204.21 | 296.51 |
| projection_gemm | `prefill_m512_wq_a` | `fp8_fp4w` | m=512, n=1536, k=7168 | 0.0516 | 218.39 | 237.07 |
| projection_gemm | `prefill_m512_wq_b` | `fp8` | m=512, n=24576, k=1536 | 0.1314 | 294.22 | 485.11 |
| projection_gemm | `prefill_m512_wq_b` | `fp8_fp4w` | m=512, n=24576, k=1536 | 0.1099 | 351.80 | 451.13 |
| projection_gemm | `prefill_m512_wkv_a` | `fp8` | m=512, n=576, k=7168 | 0.0388 | 108.85 | 218.95 |
| projection_gemm | `prefill_m512_wkv_a` | `fp8_fp4w` | m=512, n=576, k=7168 | 0.0260 | 162.37 | 267.11 |
| projection_gemm | `prefill_m512_wkv_b_prefill` | `fp8` | m=512, n=32768, k=512 | 0.0741 | 231.81 | 682.83 |
| projection_gemm | `prefill_m512_wkv_b_prefill` | `fp8_fp4w` | m=512, n=32768, k=512 | 0.0570 | 301.56 | 777.79 |
| projection_gemm | `prefill_m512_wo` | `fp8` | m=512, n=7168, k=16384 | 0.3283 | 366.30 | 406.51 |
| projection_gemm | `prefill_m512_wo` | `fp8_fp4w` | m=512, n=7168, k=16384 | 0.3448 | 348.82 | 259.29 |
| projection_gemm | `prefill_m512_index_wqi` | `fp8` | m=512, n=8192, k=1536 | 0.0544 | 236.80 | 400.38 |
| projection_gemm | `prefill_m512_index_wqi` | `fp8_fp4w` | m=512, n=8192, k=1536 | 0.0464 | 277.71 | 367.78 |
| projection_gemm | `prefill_m512_index_wki` | `fp8` | m=512, n=128, k=7168 | 0.0366 | 25.68 | 132.10 |
| projection_gemm | `prefill_m512_index_wki` | `fp8_fp4w` | m=512, n=128, k=7168 | 0.0231 | 40.61 | 194.06 |
| projection_gemm | `prefill_m512_index_weights` | `fp8` | m=512, n=64, k=7168 | 0.0376 | 12.49 | 114.58 |
| projection_gemm | `prefill_m512_index_weights` | `fp8_fp4w` | m=512, n=64, k=7168 | 0.0249 | 18.89 | 166.38 |
| projection_gemm | `prefill_m2048_wq_a` | `fp8` | m=2048, n=1536, k=7168 | 0.1492 | 302.29 | 217.47 |
| projection_gemm | `prefill_m2048_wq_a` | `fp8_fp4w` | m=2048, n=1536, k=7168 | 0.1465 | 307.92 | 193.31 |
| projection_gemm | `prefill_m2048_wq_b` | `fp8` | m=2048, n=24576, k=1536 | 0.4341 | 356.17 | 326.33 |
| projection_gemm | `prefill_m2048_wq_b` | `fp8_fp4w` | m=2048, n=24576, k=1536 | 0.4043 | 382.44 | 315.36 |
| projection_gemm | `prefill_m2048_wkv_a` | `fp8` | m=2048, n=576, k=7168 | 0.0566 | 298.70 | 382.00 |
| projection_gemm | `prefill_m2048_wkv_a` | `fp8_fp4w` | m=2048, n=576, k=7168 | 0.0534 | 316.59 | 375.88 |
| projection_gemm | `prefill_m2048_wkv_b_prefill` | `fp8` | m=2048, n=32768, k=512 | 0.2430 | 282.84 | 625.94 |
| projection_gemm | `prefill_m2048_wkv_b_prefill` | `fp8_fp4w` | m=2048, n=32768, k=512 | 0.2079 | 330.51 | 701.16 |
| projection_gemm | `prefill_m2048_wo` | `fp8` | m=2048, n=7168, k=16384 | 1.1885 | 404.75 | 152.66 |
| projection_gemm | `prefill_m2048_wo` | `fp8_fp4w` | m=2048, n=7168, k=16384 | 1.1958 | 402.29 | 114.88 |
| projection_gemm | `prefill_m2048_index_wqi` | `fp8` | m=2048, n=8192, k=1536 | 0.1544 | 333.88 | 319.92 |
| projection_gemm | `prefill_m2048_index_wqi` | `fp8_fp4w` | m=2048, n=8192, k=1536 | 0.1399 | 368.42 | 319.26 |
| projection_gemm | `prefill_m2048_index_wki` | `fp8` | m=2048, n=128, k=7168 | 0.0357 | 105.32 | 464.67 |
| projection_gemm | `prefill_m2048_index_wki` | `fp8_fp4w` | m=2048, n=128, k=7168 | 0.0256 | 146.64 | 633.53 |
| projection_gemm | `prefill_m2048_index_weights` | `fp8` | m=2048, n=64, k=7168 | 0.0384 | 48.88 | 412.54 |
| projection_gemm | `prefill_m2048_index_weights` | `fp8_fp4w` | m=2048, n=64, k=7168 | 0.0237 | 79.27 | 661.77 |
| head_grouped_gemm | `decode_b1_wk_b` | `fp8` | groups=128, m/group=1, n=512, k=128 | 0.0481 | 0.35 | 177.45 |
| head_grouped_gemm | `decode_b1_wk_b` | `fp8_fp4w` | groups=128, m/group=1, n=512, k=128 | 0.0350 | 0.48 | 154.17 |
| head_grouped_gemm | `decode_b1_wv_b` | `fp8` | groups=128, m/group=1, n=128, k=512 | 0.0290 | 0.58 | 292.70 |
| head_grouped_gemm | `decode_b1_wv_b` | `fp8_fp4w` | groups=128, m/group=1, n=128, k=512 | 0.0219 | 0.77 | 243.93 |
| head_grouped_gemm | `decode_b16_wk_b` | `fp8` | groups=128, m/group=16, n=512, k=128 | 0.0304 | 8.83 | 353.94 |
| head_grouped_gemm | `decode_b16_wk_b` | `fp8_fp4w` | groups=128, m/group=16, n=512, k=128 | 0.0248 | 10.84 | 307.37 |
| head_grouped_gemm | `decode_b16_wv_b` | `fp8` | groups=128, m/group=16, n=128, k=512 | 0.0301 | 8.92 | 332.23 |
| head_grouped_gemm | `decode_b16_wv_b` | `fp8_fp4w` | groups=128, m/group=16, n=128, k=512 | 0.0218 | 12.33 | 314.66 |
| head_grouped_gemm | `decode_b128_wk_b` | `fp8` | groups=128, m/group=128, n=512, k=128 | 0.0336 | 63.83 | 812.42 |
| head_grouped_gemm | `decode_b128_wk_b` | `fp8_fp4w` | groups=128, m/group=128, n=512, k=128 | 0.0280 | 76.65 | 863.20 |
| head_grouped_gemm | `decode_b128_wv_b` | `fp8` | groups=128, m/group=128, n=128, k=512 | 0.0313 | 68.69 | 679.25 |
| head_grouped_gemm | `decode_b128_wv_b` | `fp8_fp4w` | groups=128, m/group=128, n=128, k=512 | 0.0252 | 85.16 | 717.34 |
| indexer_mqa_logits | `prefill_index_score` | `fp8` | s=512, skv=512, h=64, d=128 | 0.0224 | 96.19 | 208.32 |
| indexer_mqa_logits | `prefill_index_score` | `fp4` | s=512, skv=512, h=64, d=128 | 0.0112 | 192.28 | 237.78 |
| indexer_mqa_logits | `prefill_index_score` | `fp8` | s=2048, skv=2048, h=64, d=128 | 0.1901 | 180.83 | 114.59 |
| indexer_mqa_logits | `prefill_index_score` | `fp4` | s=2048, skv=2048, h=64, d=128 | 0.0782 | 439.64 | 176.35 |
| indexer_paged_mqa_logits | `decode_index_score` | `fp8` | batch=1, skv=8192, h=64, d=128, block=64 | 0.0086 | 15.53 | 128.04 |
| indexer_paged_mqa_logits | `decode_index_score` | `fp4` | batch=1, skv=8192, h=64, d=128, block=64 | 0.0087 | 15.45 | 66.62 |
| indexer_paged_mqa_logits | `decode_index_score` | `fp8` | batch=64, skv=8192, h=64, d=128, block=64 | 0.0602 | 142.58 | 1175.64 |
| indexer_paged_mqa_logits | `decode_index_score` | `fp4` | batch=64, skv=8192, h=64, d=128, block=64 | 0.0216 | 397.51 | 1713.50 |
| indexer_paged_mqa_logits | `decode_index_score` | `fp8` | batch=256, skv=8192, h=64, d=128, block=64 | 0.3200 | 107.38 | 885.41 |
| indexer_paged_mqa_logits | `decode_index_score` | `fp4` | batch=256, skv=8192, h=64, d=128, block=64 | 0.1674 | 205.19 | 884.52 |
| indexer_paged_mqa_logits | `decode_index_score` | `fp8` | batch=64, skv=32768, h=64, d=128, block=64 | 0.3173 | 108.28 | 887.74 |
| indexer_paged_mqa_logits | `decode_index_score` | `fp4` | batch=64, skv=32768, h=64, d=128, block=64 | 0.1664 | 206.55 | 885.03 |

Notes:

- Timings are CUDA-event averages over the kernel call only; tensor creation and FP8/FP4 quantization are excluded.
- Projection GEMMs report `2*m*n*k`; grouped decode projections report `2*groups*m_per_group*n*k`.
- MQA indexer logits report only QK/relu/weighted-sum math as `2*selected_tokens*heads*head_dim`; top-k selection after logits is not included.
- GB/s is a logical tensor-byte rate for the measured kernel call, not a hardware counter.
