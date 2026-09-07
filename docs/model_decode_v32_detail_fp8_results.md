# DeepSeek V3.2 detailed decode benchmark

Date: 2026-07-07 08:43:50 UTC

This benchmark reports concrete projection, indexer, and attention timings for the synthetic one-layer decode path. All tensors stay on GPU; no KV offload path is used.

Environment:

```text
GPU: NVIDIA GeForce RTX 5080
PyTorch: 2.12.1+cu130
PyTorch CUDA runtime: 13.0
Compute capability: (12, 0)
DeepGEMM: 2.5.0
SMs used by DeepGEMM: 84
mode=fp8, post_proj=deepgemm, warmups=1, iters=3
```

Model shape:

- dim=7168, heads=128, q_lora=1536, kv_lora=512
- MLA q_dim=576, v_dim=512, sparse topk=2048
- Indexer heads=64, head_dim=128

Method:

- Projection rows time the DeepGEMM wrapper call, including activation quantization, and exclude surrounding RMSNorm/split/cat/transposes unless the row name says total.
- TFlop/s is throughput computed as math FLOPs divided by CUDA-event time. Rows with no meaningful FLOP model show 0.00.
- Bandwidth is effective bytes divided by CUDA-event time. Bytes include the obvious tensor/cache input, quantized activation/scale buffers, persistent quantized weights/scales, and output; it is not a hardware counter.
- Cache update rows measure paged slot writes of prepacked current-token records. Current-token KV/index packing is excluded from these rows.

Command:

```bash
model_run/deepseek_v32_decode.py --detail --batch-sizes 1,4,16,64 --history-lens 4096,8192,16384,32768,65536 --warmups 1 --iters 3 --mode fp8 --post-proj deepgemm --output docs/model_decode_v32_detail_fp8_results.md
```

## Batch 1, History KV 4096

Output shape: `1x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=1, n=1536, k=7168 | 0.1745 | 0.13 | 63.27 | 0.0110 |
| projection | `wq_b` | m=1, n=24576, k=1536 | 0.1602 | 0.47 | 236.06 | 0.0378 |
| projection | `wk_b_grouped` | groups=128, m/group=1, n=512, k=128 | 0.1851 | 0.09 | 46.32 | 0.0086 |
| projection | `wkv_a` | m=1, n=576, k=7168 | 0.1596 | 0.05 | 26.03 | 0.0042 |
| projection | `index_wqi` | m=1, n=8192, k=1536 | 0.1540 | 0.16 | 81.85 | 0.0126 |
| projection | `index_wki` | m=1, n=128, k=7168 | 0.1548 | 0.01 | 6.07 | 0.0009 |
| projection | `index_weights` | m=1, n=64, k=7168 | 0.1546 | 0.01 | 3.11 | 0.0005 |
| indexer | `index_cache_update` | batch=1, record=132B | 0.0083 | 0.00 | 0.03 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=1, tokens=4097, heads=64, dim=128 | 0.0205 | 3.28 | 27.22 | 0.0006 |
| indexer | `index_topk` | batch=1, logits=(1, 4097), topk=2048 | 0.0267 | 0.00 | 0.92 | 0.0000 |
| indexer | `indexer_total` | batch=1, tokens=4097, topk=2048 | 0.0564 | 1.19 | 10.34 | 0.0006 |
| attention | `attn_cache_update` | batch=1, record=656B | 0.0085 | 0.00 | 0.16 | 0.0000 |
| attention | `sparse_mla_decode` | batch=1, heads=128, topk=2048, qk=576, v=512 | 0.0558 | 10.23 | 29.22 | 0.0016 |
| projection | `post_wv_b_grouped` | groups=128, m/group=1, n=128, k=512 | 0.1814 | 0.09 | 47.55 | 0.0086 |
| projection | `post_wo` | m=1, n=7168, k=16384 | 0.2144 | 1.10 | 548.17 | 0.1175 |
| attention | `attention_total_with_post` | batch=1, topk=2048, includes cache+sparse+W_VB+W_O | 0.4381 | 1.88 | 291.71 | 0.1278 |

## Batch 1, History KV 8192

Output shape: `1x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=1, n=1536, k=7168 | 0.1682 | 0.13 | 65.60 | 0.0110 |
| projection | `wq_b` | m=1, n=24576, k=1536 | 0.1527 | 0.49 | 247.63 | 0.0378 |
| projection | `wk_b_grouped` | groups=128, m/group=1, n=512, k=128 | 0.1994 | 0.08 | 42.98 | 0.0086 |
| projection | `wkv_a` | m=1, n=576, k=7168 | 0.1585 | 0.05 | 26.20 | 0.0042 |
| projection | `index_wqi` | m=1, n=8192, k=1536 | 0.1490 | 0.17 | 84.62 | 0.0126 |
| projection | `index_wki` | m=1, n=128, k=7168 | 0.1560 | 0.01 | 6.02 | 0.0009 |
| projection | `index_weights` | m=1, n=64, k=7168 | 0.1545 | 0.01 | 3.11 | 0.0005 |
| indexer | `index_cache_update` | batch=1, record=132B | 0.0083 | 0.00 | 0.03 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=1, tokens=8193, heads=64, dim=128 | 0.0201 | 6.68 | 55.05 | 0.0011 |
| indexer | `index_topk` | batch=1, logits=(1, 8193), topk=2048 | 0.0332 | 0.00 | 0.99 | 0.0000 |
| indexer | `indexer_total` | batch=1, tokens=8193, topk=2048 | 0.0567 | 2.37 | 20.12 | 0.0011 |
| attention | `attn_cache_update` | batch=1, record=656B | 0.0075 | 0.00 | 0.18 | 0.0000 |
| attention | `sparse_mla_decode` | batch=1, heads=128, topk=2048, qk=576, v=512 | 0.0553 | 10.31 | 29.45 | 0.0016 |
| projection | `post_wv_b_grouped` | groups=128, m/group=1, n=128, k=512 | 0.1722 | 0.10 | 50.06 | 0.0086 |
| projection | `post_wo` | m=1, n=7168, k=16384 | 0.2214 | 1.06 | 530.95 | 0.1175 |
| attention | `attention_total_with_post` | batch=1, topk=2048, includes cache+sparse+W_VB+W_O | 0.4204 | 1.96 | 303.96 | 0.1278 |

## Batch 1, History KV 16384

Output shape: `1x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=1, n=1536, k=7168 | 0.1676 | 0.13 | 65.85 | 0.0110 |
| projection | `wq_b` | m=1, n=24576, k=1536 | 0.1593 | 0.47 | 237.32 | 0.0378 |
| projection | `wk_b_grouped` | groups=128, m/group=1, n=512, k=128 | 0.1962 | 0.09 | 43.68 | 0.0086 |
| projection | `wkv_a` | m=1, n=576, k=7168 | 0.1630 | 0.05 | 25.48 | 0.0042 |
| projection | `index_wqi` | m=1, n=8192, k=1536 | 0.1520 | 0.17 | 82.92 | 0.0126 |
| projection | `index_wki` | m=1, n=128, k=7168 | 0.1539 | 0.01 | 6.11 | 0.0009 |
| projection | `index_weights` | m=1, n=64, k=7168 | 0.1532 | 0.01 | 3.14 | 0.0005 |
| indexer | `index_cache_update` | batch=1, record=132B | 0.0085 | 0.00 | 0.03 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=1, tokens=16385, heads=64, dim=128 | 0.0204 | 13.18 | 108.29 | 0.0022 |
| indexer | `index_topk` | batch=1, logits=(1, 16385), topk=2048 | 0.0510 | 0.00 | 0.96 | 0.0000 |
| indexer | `indexer_total` | batch=1, tokens=16385, topk=2048 | 0.0663 | 4.05 | 34.00 | 0.0023 |
| attention | `attn_cache_update` | batch=1, record=656B | 0.0074 | 0.00 | 0.18 | 0.0000 |
| attention | `sparse_mla_decode` | batch=1, heads=128, topk=2048, qk=576, v=512 | 0.0558 | 10.22 | 29.22 | 0.0016 |
| projection | `post_wv_b_grouped` | groups=128, m/group=1, n=128, k=512 | 0.1727 | 0.10 | 49.92 | 0.0086 |
| projection | `post_wo` | m=1, n=7168, k=16384 | 0.2135 | 1.10 | 550.63 | 0.1175 |
| attention | `attention_total_with_post` | batch=1, topk=2048, includes cache+sparse+W_VB+W_O | 0.4134 | 1.99 | 309.15 | 0.1278 |

## Batch 1, History KV 32768

Output shape: `1x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=1, n=1536, k=7168 | 0.1674 | 0.13 | 65.95 | 0.0110 |
| projection | `wq_b` | m=1, n=24576, k=1536 | 0.1531 | 0.49 | 246.91 | 0.0378 |
| projection | `wk_b_grouped` | groups=128, m/group=1, n=512, k=128 | 0.1901 | 0.09 | 45.08 | 0.0086 |
| projection | `wkv_a` | m=1, n=576, k=7168 | 0.1621 | 0.05 | 25.61 | 0.0042 |
| projection | `index_wqi` | m=1, n=8192, k=1536 | 0.1516 | 0.17 | 83.15 | 0.0126 |
| projection | `index_wki` | m=1, n=128, k=7168 | 0.1585 | 0.01 | 5.93 | 0.0009 |
| projection | `index_weights` | m=1, n=64, k=7168 | 0.1561 | 0.01 | 3.08 | 0.0005 |
| indexer | `index_cache_update` | batch=1, record=132B | 0.0081 | 0.00 | 0.03 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=1, tokens=32769, heads=64, dim=128 | 0.0205 | 26.22 | 214.92 | 0.0044 |
| indexer | `index_topk` | batch=1, logits=(1, 32769), topk=2048 | 0.0486 | 0.00 | 1.69 | 0.0001 |
| indexer | `indexer_total` | batch=1, tokens=32769, topk=2048 | 0.0804 | 6.67 | 55.74 | 0.0045 |
| attention | `attn_cache_update` | batch=1, record=656B | 0.0075 | 0.00 | 0.18 | 0.0000 |
| attention | `sparse_mla_decode` | batch=1, heads=128, topk=2048, qk=576, v=512 | 0.0558 | 10.23 | 29.22 | 0.0016 |
| projection | `post_wv_b_grouped` | groups=128, m/group=1, n=128, k=512 | 0.1712 | 0.10 | 50.38 | 0.0086 |
| projection | `post_wo` | m=1, n=7168, k=16384 | 0.2122 | 1.11 | 553.87 | 0.1175 |
| attention | `attention_total_with_post` | batch=1, topk=2048, includes cache+sparse+W_VB+W_O | 0.4155 | 1.98 | 307.52 | 0.1278 |

## Batch 1, History KV 65536

Output shape: `1x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=1, n=1536, k=7168 | 0.1611 | 0.14 | 68.51 | 0.0110 |
| projection | `wq_b` | m=1, n=24576, k=1536 | 0.1630 | 0.46 | 231.98 | 0.0378 |
| projection | `wk_b_grouped` | groups=128, m/group=1, n=512, k=128 | 0.1806 | 0.09 | 47.46 | 0.0086 |
| projection | `wkv_a` | m=1, n=576, k=7168 | 0.1571 | 0.05 | 26.43 | 0.0042 |
| projection | `index_wqi` | m=1, n=8192, k=1536 | 0.1509 | 0.17 | 83.54 | 0.0126 |
| projection | `index_wki` | m=1, n=128, k=7168 | 0.1573 | 0.01 | 5.97 | 0.0009 |
| projection | `index_weights` | m=1, n=64, k=7168 | 0.1560 | 0.01 | 3.08 | 0.0005 |
| indexer | `index_cache_update` | batch=1, record=132B | 0.0081 | 0.00 | 0.03 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=1, tokens=65537, heads=64, dim=128 | 0.0209 | 51.33 | 420.44 | 0.0088 |
| indexer | `index_topk` | batch=1, logits=(1, 65537), topk=2048 | 0.0480 | 0.00 | 3.07 | 0.0001 |
| indexer | `indexer_total` | batch=1, tokens=65537, topk=2048 | 0.0785 | 13.68 | 113.92 | 0.0089 |
| attention | `attn_cache_update` | batch=1, record=656B | 0.0073 | 0.00 | 0.18 | 0.0000 |
| attention | `sparse_mla_decode` | batch=1, heads=128, topk=2048, qk=576, v=512 | 0.0587 | 9.72 | 27.79 | 0.0016 |
| projection | `post_wv_b_grouped` | groups=128, m/group=1, n=128, k=512 | 0.1721 | 0.10 | 50.09 | 0.0086 |
| projection | `post_wo` | m=1, n=7168, k=16384 | 0.2170 | 1.08 | 541.62 | 0.1175 |
| attention | `attention_total_with_post` | batch=1, topk=2048, includes cache+sparse+W_VB+W_O | 0.4184 | 1.96 | 305.44 | 0.1278 |

## Batch 4, History KV 4096

Output shape: `4x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=4, n=1536, k=7168 | 0.1684 | 0.52 | 65.99 | 0.0111 |
| projection | `wq_b` | m=4, n=24576, k=1536 | 0.1534 | 1.97 | 247.48 | 0.0380 |
| projection | `wk_b_grouped` | groups=128, m/group=4, n=512, k=128 | 0.1622 | 0.41 | 56.18 | 0.0091 |
| projection | `wkv_a` | m=4, n=576, k=7168 | 0.1582 | 0.21 | 26.68 | 0.0042 |
| projection | `index_wqi` | m=4, n=8192, k=1536 | 0.1577 | 0.64 | 80.33 | 0.0127 |
| projection | `index_wki` | m=4, n=128, k=7168 | 0.1566 | 0.05 | 6.42 | 0.0010 |
| projection | `index_weights` | m=4, n=64, k=7168 | 0.1842 | 0.02 | 2.97 | 0.0005 |
| indexer | `index_cache_update` | batch=4, record=132B | 0.0079 | 0.00 | 0.14 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=4, tokens=4097, heads=64, dim=128 | 0.0205 | 13.09 | 108.76 | 0.0022 |
| indexer | `index_topk` | batch=4, logits=(4, 4097), topk=2048 | 0.0298 | 0.00 | 3.30 | 0.0001 |
| indexer | `indexer_total` | batch=4, tokens=4097, topk=2048 | 0.0663 | 4.05 | 35.15 | 0.0023 |
| attention | `attn_cache_update` | batch=4, record=656B | 0.0072 | 0.00 | 0.74 | 0.0000 |
| attention | `sparse_mla_decode` | batch=4, heads=128, topk=2048, qk=576, v=512 | 0.1369 | 16.67 | 47.63 | 0.0065 |
| projection | `post_wv_b_grouped` | groups=128, m/group=4, n=128, k=512 | 0.1640 | 0.41 | 56.82 | 0.0093 |
| projection | `post_wo` | m=4, n=7168, k=16384 | 0.2252 | 4.17 | 522.72 | 0.1177 |
| attention | `attention_total_with_post` | batch=4, topk=2048, includes cache+sparse+W_VB+W_O | 0.4325 | 7.60 | 308.82 | 0.1336 |

## Batch 4, History KV 8192

Output shape: `4x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=4, n=1536, k=7168 | 0.1653 | 0.53 | 67.24 | 0.0111 |
| projection | `wq_b` | m=4, n=24576, k=1536 | 0.1593 | 1.90 | 238.37 | 0.0380 |
| projection | `wk_b_grouped` | groups=128, m/group=4, n=512, k=128 | 0.1638 | 0.41 | 55.65 | 0.0091 |
| projection | `wkv_a` | m=4, n=576, k=7168 | 0.1598 | 0.21 | 26.42 | 0.0042 |
| projection | `index_wqi` | m=4, n=8192, k=1536 | 0.1531 | 0.66 | 82.75 | 0.0127 |
| projection | `index_wki` | m=4, n=128, k=7168 | 0.1548 | 0.05 | 6.50 | 0.0010 |
| projection | `index_weights` | m=4, n=64, k=7168 | 0.1549 | 0.02 | 3.53 | 0.0005 |
| indexer | `index_cache_update` | batch=4, record=132B | 0.0080 | 0.00 | 0.14 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=4, tokens=8193, heads=64, dim=128 | 0.0204 | 26.31 | 216.97 | 0.0044 |
| indexer | `index_topk` | batch=4, logits=(4, 8193), topk=2048 | 0.0370 | 0.00 | 3.54 | 0.0001 |
| indexer | `indexer_total` | batch=4, tokens=8193, topk=2048 | 0.0603 | 8.90 | 75.59 | 0.0046 |
| attention | `attn_cache_update` | batch=4, record=656B | 0.0070 | 0.00 | 0.76 | 0.0000 |
| attention | `sparse_mla_decode` | batch=4, heads=128, topk=2048, qk=576, v=512 | 0.1368 | 16.68 | 47.67 | 0.0065 |
| projection | `post_wv_b_grouped` | groups=128, m/group=4, n=128, k=512 | 0.1652 | 0.41 | 56.38 | 0.0093 |
| projection | `post_wo` | m=4, n=7168, k=16384 | 0.2156 | 4.36 | 546.02 | 0.1177 |
| attention | `attention_total_with_post` | batch=4, topk=2048, includes cache+sparse+W_VB+W_O | 0.4392 | 7.49 | 304.10 | 0.1336 |

## Batch 4, History KV 16384

Output shape: `4x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=4, n=1536, k=7168 | 0.1669 | 0.53 | 66.59 | 0.0111 |
| projection | `wq_b` | m=4, n=24576, k=1536 | 0.1514 | 1.99 | 250.84 | 0.0380 |
| projection | `wk_b_grouped` | groups=128, m/group=4, n=512, k=128 | 0.1590 | 0.42 | 57.31 | 0.0091 |
| projection | `wkv_a` | m=4, n=576, k=7168 | 0.1582 | 0.21 | 26.68 | 0.0042 |
| projection | `index_wqi` | m=4, n=8192, k=1536 | 0.1514 | 0.66 | 83.70 | 0.0127 |
| projection | `index_wki` | m=4, n=128, k=7168 | 0.1592 | 0.05 | 6.32 | 0.0010 |
| projection | `index_weights` | m=4, n=64, k=7168 | 0.1571 | 0.02 | 3.48 | 0.0005 |
| indexer | `index_cache_update` | batch=4, record=132B | 0.0081 | 0.00 | 0.14 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=4, tokens=16385, heads=64, dim=128 | 0.0210 | 51.18 | 420.39 | 0.0088 |
| indexer | `index_topk` | batch=4, logits=(4, 16385), topk=2048 | 0.0564 | 0.00 | 3.49 | 0.0002 |
| indexer | `indexer_total` | batch=4, tokens=16385, topk=2048 | 0.0758 | 14.16 | 118.96 | 0.0090 |
| attention | `attn_cache_update` | batch=4, record=656B | 0.0072 | 0.00 | 0.74 | 0.0000 |
| attention | `sparse_mla_decode` | batch=4, heads=128, topk=2048, qk=576, v=512 | 0.1371 | 16.65 | 47.57 | 0.0065 |
| projection | `post_wv_b_grouped` | groups=128, m/group=4, n=128, k=512 | 0.1668 | 0.40 | 55.85 | 0.0093 |
| projection | `post_wo` | m=4, n=7168, k=16384 | 0.2159 | 4.35 | 545.29 | 0.1177 |
| attention | `attention_total_with_post` | batch=4, topk=2048, includes cache+sparse+W_VB+W_O | 0.4271 | 7.70 | 312.71 | 0.1336 |

## Batch 4, History KV 32768

Output shape: `4x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=4, n=1536, k=7168 | 0.1651 | 0.53 | 67.29 | 0.0111 |
| projection | `wq_b` | m=4, n=24576, k=1536 | 0.1551 | 1.95 | 244.87 | 0.0380 |
| projection | `wk_b_grouped` | groups=128, m/group=4, n=512, k=128 | 0.1616 | 0.42 | 56.39 | 0.0091 |
| projection | `wkv_a` | m=4, n=576, k=7168 | 0.1559 | 0.21 | 27.07 | 0.0042 |
| projection | `index_wqi` | m=4, n=8192, k=1536 | 0.1478 | 0.68 | 85.75 | 0.0127 |
| projection | `index_wki` | m=4, n=128, k=7168 | 0.1553 | 0.05 | 6.48 | 0.0010 |
| projection | `index_weights` | m=4, n=64, k=7168 | 0.1564 | 0.02 | 3.49 | 0.0005 |
| indexer | `index_cache_update` | batch=4, record=132B | 0.0082 | 0.00 | 0.14 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=4, tokens=32769, heads=64, dim=128 | 0.0234 | 91.81 | 752.66 | 0.0176 |
| indexer | `index_topk` | batch=4, logits=(4, 32769), topk=2048 | 0.0525 | 0.00 | 6.24 | 0.0003 |
| indexer | `indexer_total` | batch=4, tokens=32769, topk=2048 | 0.0833 | 25.79 | 215.34 | 0.0179 |
| attention | `attn_cache_update` | batch=4, record=656B | 0.0072 | 0.00 | 0.74 | 0.0000 |
| attention | `sparse_mla_decode` | batch=4, heads=128, topk=2048, qk=576, v=512 | 0.1376 | 16.58 | 47.39 | 0.0065 |
| projection | `post_wv_b_grouped` | groups=128, m/group=4, n=128, k=512 | 0.1622 | 0.41 | 57.45 | 0.0093 |
| projection | `post_wo` | m=4, n=7168, k=16384 | 0.2161 | 4.35 | 544.73 | 0.1177 |
| attention | `attention_total_with_post` | batch=4, topk=2048, includes cache+sparse+W_VB+W_O | 0.4299 | 7.65 | 310.69 | 0.1336 |

## Batch 4, History KV 65536

Output shape: `4x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=4, n=1536, k=7168 | 0.1606 | 0.55 | 69.20 | 0.0111 |
| projection | `wq_b` | m=4, n=24576, k=1536 | 0.1530 | 1.97 | 248.12 | 0.0380 |
| projection | `wk_b_grouped` | groups=128, m/group=4, n=512, k=128 | 0.1586 | 0.42 | 57.45 | 0.0091 |
| projection | `wkv_a` | m=4, n=576, k=7168 | 0.1583 | 0.21 | 26.66 | 0.0042 |
| projection | `index_wqi` | m=4, n=8192, k=1536 | 0.1506 | 0.67 | 84.11 | 0.0127 |
| projection | `index_wki` | m=4, n=128, k=7168 | 0.1582 | 0.05 | 6.36 | 0.0010 |
| projection | `index_weights` | m=4, n=64, k=7168 | 0.1554 | 0.02 | 3.52 | 0.0005 |
| indexer | `index_cache_update` | batch=4, record=132B | 0.0082 | 0.00 | 0.14 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=4, tokens=65537, heads=64, dim=128 | 0.0317 | 135.58 | 1110.42 | 0.0352 |
| indexer | `index_topk` | batch=4, logits=(4, 65537), topk=2048 | 0.0523 | 0.00 | 11.28 | 0.0006 |
| indexer | `indexer_total` | batch=4, tokens=65537, topk=2048 | 0.0838 | 51.27 | 426.96 | 0.0358 |
| attention | `attn_cache_update` | batch=4, record=656B | 0.0072 | 0.00 | 0.73 | 0.0000 |
| attention | `sparse_mla_decode` | batch=4, heads=128, topk=2048, qk=576, v=512 | 0.1372 | 16.63 | 47.54 | 0.0065 |
| projection | `post_wv_b_grouped` | groups=128, m/group=4, n=128, k=512 | 0.1630 | 0.41 | 57.15 | 0.0093 |
| projection | `post_wo` | m=4, n=7168, k=16384 | 0.2157 | 4.36 | 545.72 | 0.1177 |
| attention | `attention_total_with_post` | batch=4, topk=2048, includes cache+sparse+W_VB+W_O | 0.4325 | 7.60 | 308.84 | 0.1336 |

## Batch 16, History KV 4096

Output shape: `16x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=16, n=1536, k=7168 | 0.1670 | 2.11 | 68.31 | 0.0114 |
| projection | `wq_b` | m=16, n=24576, k=1536 | 0.1604 | 7.53 | 240.73 | 0.0386 |
| projection | `wk_b_grouped` | groups=128, m/group=16, n=512, k=128 | 0.1636 | 1.64 | 68.98 | 0.0113 |
| projection | `wkv_a` | m=16, n=576, k=7168 | 0.1604 | 0.82 | 28.03 | 0.0045 |
| projection | `index_wqi` | m=16, n=8192, k=1536 | 0.1513 | 2.66 | 85.39 | 0.0129 |
| projection | `index_wki` | m=16, n=128, k=7168 | 0.1559 | 0.19 | 8.14 | 0.0013 |
| projection | `index_weights` | m=16, n=64, k=7168 | 0.1575 | 0.09 | 5.14 | 0.0008 |
| indexer | `index_cache_update` | batch=16, record=132B | 0.0082 | 0.00 | 0.55 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=16, tokens=4097, heads=64, dim=128 | 0.0209 | 51.29 | 426.17 | 0.0089 |
| indexer | `index_topk` | batch=16, logits=(16, 4097), topk=2048 | 0.0300 | 0.00 | 13.13 | 0.0004 |
| indexer | `indexer_total` | batch=16, tokens=4097, topk=2048 | 0.0585 | 18.34 | 159.20 | 0.0093 |
| attention | `attn_cache_update` | batch=16, record=656B | 0.0070 | 0.00 | 3.02 | 0.0000 |
| attention | `sparse_mla_decode` | batch=16, heads=128, topk=2048, qk=576, v=512 | 0.4363 | 20.92 | 59.78 | 0.0261 |
| projection | `post_wv_b_grouped` | groups=128, m/group=16, n=128, k=512 | 0.1667 | 1.61 | 72.54 | 0.0121 |
| projection | `post_wo` | m=16, n=7168, k=16384 | 0.2189 | 17.17 | 541.39 | 0.1185 |
| attention | `attention_total_with_post` | batch=16, topk=2048, includes cache+sparse+W_VB+W_O | 0.7017 | 18.75 | 223.31 | 0.1567 |

## Batch 16, History KV 8192

Output shape: `16x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=16, n=1536, k=7168 | 0.1668 | 2.11 | 68.39 | 0.0114 |
| projection | `wq_b` | m=16, n=24576, k=1536 | 0.1512 | 7.99 | 255.43 | 0.0386 |
| projection | `wk_b_grouped` | groups=128, m/group=16, n=512, k=128 | 0.1625 | 1.65 | 69.44 | 0.0113 |
| projection | `wkv_a` | m=16, n=576, k=7168 | 0.1587 | 0.83 | 28.32 | 0.0045 |
| projection | `index_wqi` | m=16, n=8192, k=1536 | 0.1494 | 2.69 | 86.49 | 0.0129 |
| projection | `index_wki` | m=16, n=128, k=7168 | 0.1577 | 0.19 | 8.05 | 0.0013 |
| projection | `index_weights` | m=16, n=64, k=7168 | 0.1565 | 0.09 | 5.17 | 0.0008 |
| indexer | `index_cache_update` | batch=16, record=132B | 0.0093 | 0.00 | 0.48 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=16, tokens=8193, heads=64, dim=128 | 0.0229 | 93.65 | 772.21 | 0.0177 |
| indexer | `index_topk` | batch=16, logits=(16, 8193), topk=2048 | 0.0385 | 0.00 | 13.60 | 0.0005 |
| indexer | `indexer_total` | batch=16, tokens=8193, topk=2048 | 0.0632 | 33.97 | 288.43 | 0.0182 |
| attention | `attn_cache_update` | batch=16, record=656B | 0.0072 | 0.00 | 2.96 | 0.0000 |
| attention | `sparse_mla_decode` | batch=16, heads=128, topk=2048, qk=576, v=512 | 0.4362 | 20.92 | 59.79 | 0.0261 |
| projection | `post_wv_b_grouped` | groups=128, m/group=16, n=128, k=512 | 0.1685 | 1.59 | 71.78 | 0.0121 |
| projection | `post_wo` | m=16, n=7168, k=16384 | 0.2189 | 17.17 | 541.36 | 0.1185 |
| attention | `attention_total_with_post` | batch=16, topk=2048, includes cache+sparse+W_VB+W_O | 0.7010 | 18.76 | 223.53 | 0.1567 |

## Batch 16, History KV 16384

Output shape: `16x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=16, n=1536, k=7168 | 0.1608 | 2.19 | 70.96 | 0.0114 |
| projection | `wq_b` | m=16, n=24576, k=1536 | 0.1530 | 7.89 | 252.35 | 0.0386 |
| projection | `wk_b_grouped` | groups=128, m/group=16, n=512, k=128 | 0.1584 | 1.69 | 71.22 | 0.0113 |
| projection | `wkv_a` | m=16, n=576, k=7168 | 0.1587 | 0.83 | 28.34 | 0.0045 |
| projection | `index_wqi` | m=16, n=8192, k=1536 | 0.1546 | 2.60 | 83.60 | 0.0129 |
| projection | `index_wki` | m=16, n=128, k=7168 | 0.1583 | 0.19 | 8.02 | 0.0013 |
| projection | `index_weights` | m=16, n=64, k=7168 | 0.1585 | 0.09 | 5.10 | 0.0008 |
| indexer | `index_cache_update` | batch=16, record=132B | 0.0081 | 0.00 | 0.55 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=16, tokens=16385, heads=64, dim=128 | 0.0320 | 134.32 | 1103.27 | 0.0353 |
| indexer | `index_topk` | batch=16, logits=(16, 16385), topk=2048 | 0.0548 | 0.00 | 14.36 | 0.0008 |
| indexer | `indexer_total` | batch=16, tokens=16385, topk=2048 | 0.0896 | 47.94 | 402.64 | 0.0361 |
| attention | `attn_cache_update` | batch=16, record=656B | 0.0071 | 0.00 | 2.99 | 0.0000 |
| attention | `sparse_mla_decode` | batch=16, heads=128, topk=2048, qk=576, v=512 | 0.4339 | 21.03 | 60.11 | 0.0261 |
| projection | `post_wv_b_grouped` | groups=128, m/group=16, n=128, k=512 | 0.1699 | 1.58 | 71.19 | 0.0121 |
| projection | `post_wo` | m=16, n=7168, k=16384 | 0.2197 | 17.10 | 539.23 | 0.1185 |
| attention | `attention_total_with_post` | batch=16, topk=2048, includes cache+sparse+W_VB+W_O | 0.6982 | 18.84 | 224.43 | 0.1567 |

## Batch 16, History KV 32768

Output shape: `16x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=16, n=1536, k=7168 | 0.1617 | 2.18 | 70.55 | 0.0114 |
| projection | `wq_b` | m=16, n=24576, k=1536 | 0.1517 | 7.96 | 254.55 | 0.0386 |
| projection | `wk_b_grouped` | groups=128, m/group=16, n=512, k=128 | 0.1605 | 1.67 | 70.31 | 0.0113 |
| projection | `wkv_a` | m=16, n=576, k=7168 | 0.1570 | 0.84 | 28.64 | 0.0045 |
| projection | `index_wqi` | m=16, n=8192, k=1536 | 0.1498 | 2.69 | 86.25 | 0.0129 |
| projection | `index_wki` | m=16, n=128, k=7168 | 0.1576 | 0.19 | 8.06 | 0.0013 |
| projection | `index_weights` | m=16, n=64, k=7168 | 0.1587 | 0.09 | 5.10 | 0.0008 |
| indexer | `index_cache_update` | batch=16, record=132B | 0.0080 | 0.00 | 0.56 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=16, tokens=32769, heads=64, dim=128 | 0.0829 | 103.65 | 849.72 | 0.0704 |
| indexer | `index_topk` | batch=16, logits=(16, 32769), topk=2048 | 0.0537 | 0.00 | 24.40 | 0.0013 |
| indexer | `indexer_total` | batch=16, tokens=32769, topk=2048 | 0.1336 | 64.32 | 537.15 | 0.0717 |
| attention | `attn_cache_update` | batch=16, record=656B | 0.0072 | 0.00 | 2.96 | 0.0000 |
| attention | `sparse_mla_decode` | batch=16, heads=128, topk=2048, qk=576, v=512 | 0.4353 | 20.97 | 59.93 | 0.0261 |
| projection | `post_wv_b_grouped` | groups=128, m/group=16, n=128, k=512 | 0.1646 | 1.63 | 73.48 | 0.0121 |
| projection | `post_wo` | m=16, n=7168, k=16384 | 0.2202 | 17.06 | 538.06 | 0.1185 |
| attention | `attention_total_with_post` | batch=16, topk=2048, includes cache+sparse+W_VB+W_O | 0.7006 | 18.77 | 223.64 | 0.1567 |

## Batch 16, History KV 65536

Output shape: `16x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=16, n=1536, k=7168 | 0.1700 | 2.07 | 67.10 | 0.0114 |
| projection | `wq_b` | m=16, n=24576, k=1536 | 0.1547 | 7.81 | 249.69 | 0.0386 |
| projection | `wk_b_grouped` | groups=128, m/group=16, n=512, k=128 | 0.1637 | 1.64 | 68.93 | 0.0113 |
| projection | `wkv_a` | m=16, n=576, k=7168 | 0.1621 | 0.82 | 27.74 | 0.0045 |
| projection | `index_wqi` | m=16, n=8192, k=1536 | 0.1521 | 2.65 | 84.95 | 0.0129 |
| projection | `index_wki` | m=16, n=128, k=7168 | 0.1583 | 0.19 | 8.02 | 0.0013 |
| projection | `index_weights` | m=16, n=64, k=7168 | 0.1562 | 0.09 | 5.18 | 0.0008 |
| indexer | `index_cache_update` | batch=16, record=132B | 0.0082 | 0.00 | 0.54 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=16, tokens=65537, heads=64, dim=128 | 0.1695 | 101.39 | 830.40 | 0.1407 |
| indexer | `index_topk` | batch=16, logits=(16, 65537), topk=2048 | 0.0530 | 0.00 | 44.51 | 0.0024 |
| indexer | `indexer_total` | batch=16, tokens=65537, topk=2048 | 0.2318 | 74.11 | 617.22 | 0.1431 |
| attention | `attn_cache_update` | batch=16, record=656B | 0.0070 | 0.00 | 3.02 | 0.0000 |
| attention | `sparse_mla_decode` | batch=16, heads=128, topk=2048, qk=576, v=512 | 0.4351 | 20.97 | 59.94 | 0.0261 |
| projection | `post_wv_b_grouped` | groups=128, m/group=16, n=128, k=512 | 0.1625 | 1.65 | 74.43 | 0.0121 |
| projection | `post_wo` | m=16, n=7168, k=16384 | 0.2192 | 17.14 | 540.57 | 0.1185 |
| attention | `attention_total_with_post` | batch=16, topk=2048, includes cache+sparse+W_VB+W_O | 0.7020 | 18.74 | 223.22 | 0.1567 |

## Batch 64, History KV 4096

Output shape: `64x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=64, n=1536, k=7168 | 0.1679 | 8.39 | 75.05 | 0.0126 |
| projection | `wq_b` | m=64, n=24576, k=1536 | 0.1566 | 30.86 | 263.11 | 0.0412 |
| projection | `wk_b_grouped` | groups=128, m/group=64, n=512, k=128 | 0.1620 | 6.63 | 123.20 | 0.0200 |
| projection | `wkv_a` | m=64, n=576, k=7168 | 0.1594 | 3.32 | 35.09 | 0.0056 |
| projection | `index_wqi` | m=64, n=8192, k=1536 | 0.1544 | 10.43 | 90.23 | 0.0139 |
| projection | `index_wki` | m=64, n=128, k=7168 | 0.1603 | 0.73 | 14.50 | 0.0023 |
| projection | `index_weights` | m=64, n=64, k=7168 | 0.1610 | 0.36 | 11.54 | 0.0019 |
| indexer | `index_cache_update` | batch=64, record=132B | 0.0083 | 0.00 | 2.16 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=64, tokens=4097, heads=64, dim=128 | 0.0320 | 134.34 | 1116.16 | 0.0357 |
| indexer | `index_topk` | batch=64, logits=(64, 4097), topk=2048 | 0.0311 | 0.00 | 50.59 | 0.0016 |
| indexer | `indexer_total` | batch=64, tokens=4097, topk=2048 | 0.0662 | 64.90 | 563.23 | 0.0373 |
| attention | `attn_cache_update` | batch=64, record=656B | 0.0071 | 0.00 | 11.95 | 0.0001 |
| attention | `sparse_mla_decode` | batch=64, heads=128, topk=2048, qk=576, v=512 | 1.6003 | 22.81 | 65.20 | 0.1043 |
| projection | `post_wv_b_grouped` | groups=128, m/group=64, n=128, k=512 | 0.1711 | 6.28 | 135.64 | 0.0232 |
| projection | `post_wo` | m=64, n=7168, k=16384 | 0.2322 | 64.75 | 523.63 | 0.1216 |
| attention | `attention_total_with_post` | batch=64, topk=2048, includes cache+sparse+W_VB+W_O | 1.9345 | 27.20 | 128.81 | 0.2492 |

## Batch 64, History KV 8192

Output shape: `64x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=64, n=1536, k=7168 | 0.1716 | 8.21 | 73.43 | 0.0126 |
| projection | `wq_b` | m=64, n=24576, k=1536 | 0.1682 | 28.73 | 245.00 | 0.0412 |
| projection | `wk_b_grouped` | groups=128, m/group=64, n=512, k=128 | 0.1632 | 6.58 | 122.26 | 0.0200 |
| projection | `wkv_a` | m=64, n=576, k=7168 | 0.1614 | 3.28 | 34.67 | 0.0056 |
| projection | `index_wqi` | m=64, n=8192, k=1536 | 0.1573 | 10.24 | 88.60 | 0.0139 |
| projection | `index_wki` | m=64, n=128, k=7168 | 0.1642 | 0.72 | 14.16 | 0.0023 |
| projection | `index_weights` | m=64, n=64, k=7168 | 0.1569 | 0.37 | 11.84 | 0.0019 |
| indexer | `index_cache_update` | batch=64, record=132B | 0.0082 | 0.00 | 2.19 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=64, tokens=8193, heads=64, dim=128 | 0.0902 | 95.24 | 785.26 | 0.0708 |
| indexer | `index_topk` | batch=64, logits=(64, 8193), topk=2048 | 0.0530 | 0.00 | 39.55 | 0.0021 |
| indexer | `indexer_total` | batch=64, tokens=8193, topk=2048 | 0.1468 | 58.54 | 497.08 | 0.0730 |
| attention | `attn_cache_update` | batch=64, record=656B | 0.0072 | 0.00 | 11.86 | 0.0001 |
| attention | `sparse_mla_decode` | batch=64, heads=128, topk=2048, qk=576, v=512 | 1.5989 | 22.83 | 65.25 | 0.1043 |
| projection | `post_wv_b_grouped` | groups=128, m/group=64, n=128, k=512 | 0.1665 | 6.45 | 139.39 | 0.0232 |
| projection | `post_wo` | m=64, n=7168, k=16384 | 0.2303 | 65.26 | 527.77 | 0.1216 |
| attention | `attention_total_with_post` | batch=64, topk=2048, includes cache+sparse+W_VB+W_O | 1.9287 | 27.28 | 129.20 | 0.2492 |

## Batch 64, History KV 16384

Output shape: `64x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=64, n=1536, k=7168 | 0.1637 | 8.61 | 76.97 | 0.0126 |
| projection | `wq_b` | m=64, n=24576, k=1536 | 0.1588 | 30.43 | 259.48 | 0.0412 |
| projection | `wk_b_grouped` | groups=128, m/group=64, n=512, k=128 | 0.1590 | 6.75 | 125.52 | 0.0200 |
| projection | `wkv_a` | m=64, n=576, k=7168 | 0.1625 | 3.25 | 34.43 | 0.0056 |
| projection | `index_wqi` | m=64, n=8192, k=1536 | 0.1527 | 10.55 | 91.25 | 0.0139 |
| projection | `index_wki` | m=64, n=128, k=7168 | 0.1642 | 0.72 | 14.15 | 0.0023 |
| projection | `index_weights` | m=64, n=64, k=7168 | 0.1605 | 0.37 | 11.57 | 0.0019 |
| indexer | `index_cache_update` | batch=64, record=132B | 0.0081 | 0.00 | 2.21 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=64, tokens=16385, heads=64, dim=128 | 0.1722 | 99.79 | 819.68 | 0.1411 |
| indexer | `index_topk` | batch=64, logits=(64, 16385), topk=2048 | 0.0532 | 0.00 | 59.19 | 0.0031 |
| indexer | `indexer_total` | batch=64, tokens=16385, topk=2048 | 0.2350 | 73.11 | 614.03 | 0.1443 |
| attention | `attn_cache_update` | batch=64, record=656B | 0.0070 | 0.00 | 12.15 | 0.0001 |
| attention | `sparse_mla_decode` | batch=64, heads=128, topk=2048, qk=576, v=512 | 1.6041 | 22.76 | 65.04 | 0.1043 |
| projection | `post_wv_b_grouped` | groups=128, m/group=64, n=128, k=512 | 0.1654 | 6.49 | 140.26 | 0.0232 |
| projection | `post_wo` | m=64, n=7168, k=16384 | 0.2329 | 64.56 | 522.07 | 0.1216 |
| attention | `attention_total_with_post` | batch=64, topk=2048, includes cache+sparse+W_VB+W_O | 1.9300 | 27.26 | 129.11 | 0.2492 |

## Batch 64, History KV 32768

Output shape: `64x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=64, n=1536, k=7168 | 0.1642 | 8.58 | 76.72 | 0.0126 |
| projection | `wq_b` | m=64, n=24576, k=1536 | 0.1559 | 30.99 | 264.24 | 0.0412 |
| projection | `wk_b_grouped` | groups=128, m/group=64, n=512, k=128 | 0.1605 | 6.69 | 124.33 | 0.0200 |
| projection | `wkv_a` | m=64, n=576, k=7168 | 0.1592 | 3.32 | 35.14 | 0.0056 |
| projection | `index_wqi` | m=64, n=8192, k=1536 | 0.1577 | 10.21 | 88.36 | 0.0139 |
| projection | `index_wki` | m=64, n=128, k=7168 | 0.1592 | 0.74 | 14.60 | 0.0023 |
| projection | `index_weights` | m=64, n=64, k=7168 | 0.1584 | 0.37 | 11.73 | 0.0019 |
| indexer | `index_cache_update` | batch=64, record=132B | 0.0098 | 0.00 | 1.83 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=64, tokens=32769, heads=64, dim=128 | 0.3273 | 104.99 | 860.71 | 0.2817 |
| indexer | `index_topk` | batch=64, logits=(64, 32769), topk=2048 | 0.0663 | 0.00 | 79.05 | 0.0052 |
| indexer | `indexer_total` | batch=64, tokens=32769, topk=2048 | 0.4067 | 84.49 | 705.60 | 0.2870 |
| attention | `attn_cache_update` | batch=64, record=656B | 0.0071 | 0.00 | 11.89 | 0.0001 |
| attention | `sparse_mla_decode` | batch=64, heads=128, topk=2048, qk=576, v=512 | 1.6002 | 22.81 | 65.20 | 0.1043 |
| projection | `post_wv_b_grouped` | groups=128, m/group=64, n=128, k=512 | 0.1640 | 6.55 | 141.51 | 0.0232 |
| projection | `post_wo` | m=64, n=7168, k=16384 | 0.2293 | 65.55 | 530.11 | 0.1216 |
| attention | `attention_total_with_post` | batch=64, topk=2048, includes cache+sparse+W_VB+W_O | 1.9336 | 27.21 | 128.87 | 0.2492 |

## Batch 64, History KV 65536

Output shape: `64x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=64, n=1536, k=7168 | 0.1625 | 8.67 | 77.51 | 0.0126 |
| projection | `wq_b` | m=64, n=24576, k=1536 | 0.1587 | 30.45 | 259.69 | 0.0412 |
| projection | `wk_b_grouped` | groups=128, m/group=64, n=512, k=128 | 0.1631 | 6.58 | 122.38 | 0.0200 |
| projection | `wkv_a` | m=64, n=576, k=7168 | 0.1631 | 3.24 | 34.30 | 0.0056 |
| projection | `index_wqi` | m=64, n=8192, k=1536 | 0.1522 | 10.58 | 91.55 | 0.0139 |
| projection | `index_wki` | m=64, n=128, k=7168 | 0.1574 | 0.75 | 14.77 | 0.0023 |
| projection | `index_weights` | m=64, n=64, k=7168 | 0.1562 | 0.38 | 11.89 | 0.0019 |
| indexer | `index_cache_update` | batch=64, record=132B | 0.0080 | 0.00 | 2.23 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=64, tokens=65537, heads=64, dim=128 | 0.6364 | 107.99 | 884.49 | 0.5628 |
| indexer | `index_topk` | batch=64, logits=(64, 65537), topk=2048 | 0.0944 | 0.00 | 99.97 | 0.0094 |
| indexer | `indexer_total` | batch=64, tokens=65537, topk=2048 | 0.7532 | 91.24 | 759.85 | 0.5723 |
| attention | `attn_cache_update` | batch=64, record=656B | 0.0071 | 0.00 | 12.00 | 0.0001 |
| attention | `sparse_mla_decode` | batch=64, heads=128, topk=2048, qk=576, v=512 | 1.6081 | 22.70 | 64.88 | 0.1043 |
| projection | `post_wv_b_grouped` | groups=128, m/group=64, n=128, k=512 | 0.1638 | 6.56 | 141.65 | 0.0232 |
| projection | `post_wo` | m=64, n=7168, k=16384 | 0.2310 | 65.09 | 526.36 | 0.1216 |
| attention | `attention_total_with_post` | batch=64, topk=2048, includes cache+sparse+W_VB+W_O | 1.9476 | 27.01 | 127.94 | 0.2492 |

