# DeepSeek V3.2 detailed decode benchmark

Date: 2026-07-09 11:03:54 UTC

This benchmark reports concrete projection, indexer, and attention timings for the synthetic one-layer decode path. All tensors stay on GPU; no KV offload path is used.

Environment:

```text
GPU: NVIDIA GeForce RTX 5080
PyTorch: 2.12.1+cu130
PyTorch CUDA runtime: 13.0
Compute capability: (12, 0)
DeepGEMM: 2.5.0
SMs used by DeepGEMM: 84
mode=fp8, post_proj=deepgemm, detail_scope=iter, warmups=1, iters=3
```

Model shape:

- dim=7168, heads=128, q_lora=1536, kv_lora=512
- MLA q_dim=576, v_dim=512, sparse topk=2048
- Indexer heads=64, head_dim=128

Method:

- `detail_scope=modules` benchmarks isolated module calls only; it does not include aggregate indexer/attention/iteration rows.
- `detail_scope=iter` benchmarks only the full decode iteration (`project -> indexer -> attention/post`).
- Projection rows time the DeepGEMM wrapper call, including activation quantization, and exclude surrounding RMSNorm/split/cat/transposes unless the row name says total.
- TFlop/s is throughput computed as math FLOPs divided by CUDA-event time. Rows with no meaningful FLOP model show 0.00.
- Bandwidth is effective bytes divided by CUDA-event time. Bytes include the obvious tensor/cache input, quantized activation/scale buffers, persistent quantized weights/scales, and output; it is not a hardware counter.
- Cache update rows measure paged slot writes of prepacked current-token records. Current-token KV/index packing is excluded from these rows.

Command:

```bash
model_run/deepseek_v32_decode.py --detail --detail-scope iter --batch-sizes 1 --history-lens 32768 --warmups 1 --iters 3 --mode fp8 --post-proj deepgemm --output docs/model_decode_v32_detail_fp8_b1_l32768_iter.md
```

## Batch 1, History KV 32768

Output shape: `1x7168`

| Section | Name | Shape | ms | TFlop/s | GB/s | Effective GB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| iter | `decode_iter` | batch=1, history=32768, full project+indexer+attention | 3.0505 | 0.49 | 68.14 | 0.2079 |

