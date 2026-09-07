# DeepSeek V3.2 detailed decode benchmark

Date: 2026-07-09 10:49:16 UTC

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
model_run/deepseek_v32_decode.py --detail --batch-sizes 1 --history-lens 32768 --warmups 1 --iters 3 --mode fp8 --post-proj deepgemm --output docs/model_decode_v32_detail_fp8_b1_l32768.md
```

## Batch 1, History KV 32768

Output shape: `1x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| projection | `wq_a` | m=1, n=1536, k=7168 | 0.2842 | 0.08 | 38.83 | 0.0110 |
| projection | `wq_b` | m=1, n=24576, k=1536 | 0.2531 | 0.30 | 149.41 | 0.0378 |
| projection | `wk_b_grouped` | groups=128, m/group=1, n=512, k=128 | 0.3043 | 0.06 | 28.17 | 0.0086 |
| projection | `wkv_a` | m=1, n=576, k=7168 | 0.2557 | 0.03 | 16.24 | 0.0042 |
| projection | `index_wqi` | m=1, n=8192, k=1536 | 0.2527 | 0.10 | 49.88 | 0.0126 |
| projection | `index_wki` | m=1, n=128, k=7168 | 0.2533 | 0.01 | 3.71 | 0.0009 |
| projection | `index_weights` | m=1, n=64, k=7168 | 0.2550 | 0.00 | 1.89 | 0.0005 |
| indexer | `index_cache_update` | batch=1, record=132B | 0.0119 | 0.00 | 0.02 | 0.0000 |
| indexer | `index_logits_deepgemm` | batch=1, tokens=32769, heads=64, dim=128 | 0.0310 | 17.32 | 142.00 | 0.0044 |
| indexer | `index_topk` | batch=1, logits=(1, 32769), topk=2048 | 0.1040 | 0.00 | 0.79 | 0.0001 |
| indexer | `indexer_total` | batch=1, tokens=32769, topk=2048 | 0.1394 | 3.85 | 32.16 | 0.0045 |
| attention | `attn_cache_update` | batch=1, record=656B | 0.0110 | 0.00 | 0.12 | 0.0000 |
| attention | `sparse_mla_decode` | batch=1, heads=128, topk=2048, qk=576, v=512 | 0.0876 | 6.51 | 18.62 | 0.0016 |
| projection | `post_wv_b_grouped` | groups=128, m/group=1, n=128, k=512 | 0.2829 | 0.06 | 30.48 | 0.0086 |
| projection | `post_wo` | m=1, n=7168, k=16384 | 0.2954 | 0.80 | 397.82 | 0.1175 |
| attention | `attention_total_with_post` | batch=1, topk=2048, includes cache+sparse+W_VB+W_O | 0.6442 | 1.28 | 198.36 | 0.1278 |

