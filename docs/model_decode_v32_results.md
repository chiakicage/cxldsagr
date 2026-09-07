# DeepSeek V3.2 model decode benchmark

Date: 2026-07-06 20:23:17 UTC

This benchmark runs one synthetic decode layer with full projection, sparse indexer, sparse MLA attention, per-head value projection, and final output projection. All tensors stay on GPU; no KV offload path is used.

Environment:

```text
GPU: NVIDIA GeForce RTX 5080
PyTorch: 2.12.1+cu130
PyTorch CUDA runtime: 13.0
Compute capability: (12, 0)
DeepGEMM: 2.5.0
SMs used by DeepGEMM: 84
mode=fp8_fp4w, post_proj=deepgemm, warmups=1, iters=3, quick=False
```

Model shape:

- `dim=7168`, `heads=128`, `q_lora=1536`, `kv_lora=512`
- MLA decode `q_dim=576` (`latent=512` + `rope=64`), `d_v=512`, sparse `topk=2048`
- Indexer `heads=64`, `head_dim=128`

Command:

```bash
model_run/deepseek_v32_decode.py --batch-sizes 1,4,8,16,32,64 --history-lens 4096,8192 --warmups 1 --iters 3 --output docs/model_decode_v32_results.md
```

| Batch | History KV len | Total len | Top-k | E2E ms | Projection ms | Indexer ms | Attention+post ms | Output |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 4096 | 4097 | 2048 | 26.9134 | 13.6922 | 0.1545 | 13.2924 | `1x7168` |
| 1 | 8192 | 8193 | 2048 | 26.7122 | 13.6090 | 0.1485 | 13.1776 | `1x7168` |
| 4 | 4096 | 4097 | 2048 | 26.9051 | 13.7035 | 0.1610 | 13.3457 | `4x7168` |
| 4 | 8192 | 8193 | 2048 | 26.9481 | 13.5626 | 0.1521 | 13.1565 | `4x7168` |
| 8 | 4096 | 4097 | 2048 | 27.0297 | 13.5823 | 0.1553 | 13.2260 | `8x7168` |
| 8 | 8192 | 8193 | 2048 | 27.0642 | 13.6015 | 0.1550 | 13.1914 | `8x7168` |
| 16 | 4096 | 4097 | 2048 | 27.1452 | 13.6301 | 0.1584 | 13.1805 | `16x7168` |
| 16 | 8192 | 8193 | 2048 | 27.0165 | 13.6151 | 0.1557 | 13.1523 | `16x7168` |
| 32 | 4096 | 4097 | 2048 | 27.0917 | 13.6775 | 0.1584 | 13.4152 | `32x7168` |
| 32 | 8192 | 8193 | 2048 | 27.1230 | 13.6270 | 0.1732 | 13.2309 | `32x7168` |
| 64 | 4096 | 4097 | 2048 | 27.4008 | 13.6167 | 0.1562 | 13.3388 | `64x7168` |
| 64 | 8192 | 8193 | 2048 | 27.1324 | 13.7087 | 0.2105 | 13.2079 | `64x7168` |

Notes:

- E2E includes activation quantization, current-token KV/index cache update, DeepGEMM paged-index logits, `torch.topk`, sparse MLA decode, per-head `W_VB`, and final `W_O`.
- The previous sparse-then-DeepGEMM illegal-address was caused by non-canonical strides on size-1 grouped-GEMM dimensions after transpose/contiguous; GroupedLinear now canonicalizes strides before launching DeepGEMM.
- Projection/indexer/attention columns are measured as isolated sections and therefore are not expected to add up exactly to E2E.
- The sparse MLA decode wrapper is used for batch sizes up to 64 decode tokens, matching the current decode kernel limit.
