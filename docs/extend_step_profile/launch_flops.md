# Launch and throughput audit

new=4096, chunk=512; synthetic attention only. FMA=2 FLOPs.

| Step | 4K TFLOPS | 64K TFLOPS | 64K % of measured GEMM reference |
|---|---:|---:|---:|
| wq_a | 214.94 | 213.96 | 51.5 |
| wq_b | 301.39 | 298.94 | 71.9 |
| wkv_a | 174.53 | 173.13 | 41.7 |
| wk_b | 97.34 | 98.04 | 23.6 |
| index_wqi | 223.87 | 222.61 | 53.6 |
| index_wki | 66.44 | 66.11 | 15.9 |
| index_weights | 35.34 | 35.07 | N/A |
| wv_b | 172.26 | 177.57 | 42.7 |
| wo | 359.16 | 358.20 | 86.2 |
| index/mqa_logits | 171.45 | 177.76 | N/A |
| mla/prefill | 76.05 | 74.00 | N/A |

Reference: identical DeepGEMM FP8 API, 8192^3 GEMM, prequantized operands, CUDA Graph median of 10 replays: 415.65 TFLOPS. This is an achieved-throughput reference, not a theoretical peak. GEMM step times include API-internal scale packing/reduction but exclude activation quantization.

Indexer useful FLOPs = 2*64*128*(T*history + T*(T+1)/2), excluding masked/tile-padding work. MLA useful FLOPs = 2*T*128*2048*(576+512); QK BF16 and PV FP8, both FP32 accumulation. Softmax, reductions, top-k, quantization and layout operations are not counted as matmul FLOPs.

Overall useful matmul throughput: 87.70 / 109.91 TFLOPS (4K/64K), including all attention-step overhead in latency.

Launch check: 12 separately synchronized samples. Eager medians 50.123/77.509 ms; graph medians 50.020/78.049 ms. CPU submit medians 13.273/16.546 ms eager vs 0.0045/0.732 ms graph. No material end-to-end graph gain. No-step-annotation CUPTI traces contain 1232 device activities each; GPU interval-union busy fractions 96.20%/97.95%; gaps 1.947/1.611 ms. Profiler perturbs launches; busy fraction is not SM/Tensor utilization.

Nominal RTX 5080 whitepaper peaks: ordinary FP8 FP32 accumulation 225.1 TFLOPS, BF16 FP32 accumulation 112.6 TFLOPS at 2617 MHz. Driver maximum SM clock is 3090 MHz, giving at most 265.8 TFLOPS for the former by frequency-only scaling. This does not explain the measured block-scaled GEMM throughput. Exact block-scaled peak/instruction throughput remains unverified; no theoretical utilization is claimed for those GEMMs.

Nsight Compute failed with ERR_NVGPUCTRPERM; hardware counter utilization is unavailable. No driver settings were changed.

Source: https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf (Table 4).

WO precision verification: matched cached specialization N=7168, K=16384, M tile=128, split-K=1, FP4 flags all false to the profiled kernel. `cuobjdump --dump-sass` shows `QMMA.SF.16832.F32.E4M3.E4M3.E8`. Thus operands are E4M3 FP8, scales are E8, and accumulator is F32; output is BF16. Source PTX is `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0`. This rules out an explicit FP16-accumulator path. Higher MXFP8 instruction throughput is a plausible explanation, but an exact 2x theoretical peak is not established by this measurement.

Full projection audit (current FP8 extend profile): all eight DeepGEMM projections—WQ_A, WQ_B, WKV_A, WK_B, index_WQI, index_WKI, WV_B, WO—have matching cached kernel specializations whose SASS uses only `QMMA.SF.16832.F32.E4M3.E4M3.E8` for MMA. They all use the MXFP8 block-scaled instruction path with FP32 accumulators and BF16 outputs. This describes the instruction path; application quantization groups remain 128 elements, not independent native MXFP8 scales recalculated every 32 elements.

Exceptions: index_weights is a BF16 linear (the trace identifies a CUTLASS BF16 GEMM); the matched DeepGEMM index-logits cubin uses `QMMA.16832.F32.E4M3.E4M3` without SF, i.e. ordinary FP8 with FP32 accumulation. Current MLA source/default bf16_qk=True uses BF16 QK and ordinary FP8 PV (not the optional block-scaled QK branch). Full template/source/SASS evidence for the eight projections and index logits is in `gemm_precision.json`.
