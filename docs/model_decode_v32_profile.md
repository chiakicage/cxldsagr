# DeepSeek V3.2 model decode profile

Date: 2026-07-06 20:24:22 UTC

This profile uses DeepGEMM for projection, indexer logits, W_VB, and W_O, and flash_mla_sm120/sparse_mla_sm120 for sparse MLA decode. All tensors stay on GPU; no KV offload path is used.

Environment:

```text
GPU: NVIDIA GeForce RTX 5080
PyTorch: 2.12.1+cu130
PyTorch CUDA runtime: 13.0
Compute capability: (12, 0)
DeepGEMM: 2.5.0
SMs used by DeepGEMM: 84
mode=fp8_fp4w, post_proj=deepgemm, warmups=1, iters=3
```

Model shape:

- dim=7168, heads=128, q_lora=1536, kv_lora=512
- MLA decode q_dim=576, sparse topk=2048
- Indexer heads=64, head_dim=128

Method:

- event_ms is CUDA event elapsed time for the segment and includes stream idle gaps between kernels.
- kernel_ms is the sum of CUDA kernel self times reported by torch.profiler for one profiled segment run.
- launch_gap_ms = max(event_ms - kernel_ms, 0). It approximates CPU launch overhead, stream idle time, and profiler/event mismatch; it is directional, not a hardware counter.
- cpu_wall_ms is wall time for the profiled run and includes profiler overhead, so use it mainly as a sanity check.

Summary:

- Batch 1, history 8192: e2e 37.403 ms, kernel self time 6.775 ms (18.1%), launch/gap 30.628 ms (81.9%). Largest gap segment: post_wv_b (14.329 ms, 2821 kernels).
- Batch 8, history 8192: e2e 37.601 ms, kernel self time 7.176 ms (19.1%), launch/gap 30.426 ms (80.9%). Largest gap segment: projection_all (15.424 ms, 2990 kernels).
- Batch 64, history 8192: e2e 37.806 ms, kernel self time 8.719 ms (23.1%), launch/gap 29.087 ms (76.9%). Largest gap segment: projection_all (15.544 ms, 2990 kernels).

Overall: the measured decode path is launch/scheduling dominated, not kernel-execution dominated. The main source is thousands of small kernels in projection/post-projection activation quantization and grouped GEMM setup, especially projection_all and post_wv_b. The sparse_mla_decode kernel itself is mostly kernel-execution time and has very small launch/gap overhead.

## Batch 1, History KV 8192

| Segment | event ms | kernel ms | launch/gap ms | kernels | CPU wall ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| projection_all | 13.6610 | 3.1802 | 10.4808 | 2993 | 893.3254 |
| index_cache_update | 0.1225 | 0.0171 | 0.1054 | 16 | 4.0431 |
| index_logits_deepgemm | 0.0286 | 0.0054 | 0.0232 | 3 | 2.1626 |
| index_topk | 0.0400 | 0.0336 | 0.0064 | 6 | 3.1226 |
| attn_cache_update | 0.4509 | 0.0707 | 0.3802 | 67 | 9.6052 |
| sparse_mla_decode | 0.0615 | 0.0526 | 0.0089 | 3 | 2.1438 |
| post_wv_b | 17.5797 | 3.2509 | 14.3289 | 2821 | 432.6847 |
| post_wo | 0.2140 | 0.1196 | 0.0944 | 23 | 4.2998 |
| e2e_decode_layer | 37.4033 | 6.7755 | 30.6278 | 5932 | 853.3881 |

Interpretation: launch/scheduling gaps are a major part of the measured e2e time for this case.

## Batch 8, History KV 8192

| Segment | event ms | kernel ms | launch/gap ms | kernels | CPU wall ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| projection_all | 18.9209 | 3.4973 | 15.4236 | 2990 | 418.4260 |
| index_cache_update | 0.1186 | 0.0218 | 0.0968 | 16 | 3.3867 |
| index_logits_deepgemm | 0.0290 | 0.0114 | 0.0176 | 3 | 1.5928 |
| index_topk | 0.0454 | 0.0379 | 0.0075 | 7 | 2.0032 |
| attn_cache_update | 0.4686 | 0.0819 | 0.3867 | 68 | 9.0197 |
| sparse_mla_decode | 0.2554 | 0.2477 | 0.0076 | 3 | 1.7229 |
| post_wv_b | 17.6717 | 3.1165 | 14.5552 | 2822 | 379.4340 |
| post_wo | 0.2116 | 0.1227 | 0.0889 | 23 | 4.3051 |
| e2e_decode_layer | 37.6012 | 7.1755 | 30.4257 | 5932 | 832.7319 |

Interpretation: launch/scheduling gaps are a major part of the measured e2e time for this case.

## Batch 64, History KV 8192

| Segment | event ms | kernel ms | launch/gap ms | kernels | CPU wall ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| projection_all | 18.9362 | 3.3923 | 15.5439 | 2990 | 420.2601 |
| index_cache_update | 0.1297 | 0.0212 | 0.1084 | 16 | 3.4296 |
| index_logits_deepgemm | 0.0954 | 0.0840 | 0.0114 | 3 | 1.6753 |
| index_topk | 0.0821 | 0.0432 | 0.0389 | 21 | 3.0848 |
| attn_cache_update | 0.4510 | 0.0802 | 0.3708 | 68 | 8.8286 |
| sparse_mla_decode | 1.6595 | 1.6529 | 0.0066 | 3 | 3.1106 |
| post_wv_b | 17.6175 | 3.2855 | 14.3320 | 2822 | 382.2769 |
| post_wo | 0.2180 | 0.1398 | 0.0782 | 23 | 4.2304 |
| e2e_decode_layer | 37.8056 | 8.7191 | 29.0865 | 5946 | 840.9200 |

Interpretation: launch/scheduling gaps are a major part of the measured e2e time for this case.

