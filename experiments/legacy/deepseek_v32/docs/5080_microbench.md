Device: NVIDIA GeForce RTX 5080 (sm120)
Clock: 2617.00 MHz, SMs: 84
GPU total available threads: 129024 (84 SM * 1536)
Benchmark launch config: blocks=252, threads=512, launched=129024

CUDA Core Benchmark
------------------------------------------------------------------------------------
| Instruction        | Core Computation                       | Peak Performance   |
|--------------------|----------------------------------------|--------------------|
| DFMA               | FMA(f64,f64,f64)                       | 0.82 TFLOPS        |
| FFMA               | FMA(f32,f32,f32)                       | 56.69 TFLOPS       |
| HFMA2              | FMA(f16,f16,f16)                       | 61.36 TFLOPS       |
| BFMA2              | FMA(bf16,bf16,bf16)                    | 61.86 TFLOPS       |
| IMAD               | MAD(s32,s32,s32)                       | 29.93 TOPS         |
| DP2A               | DP2A(s32,s16,s16)                      | 60.27 TOPS         |
| DP4A               | DP4A(s32,s8,s8)                        | 121.61 TOPS        |
------------------------------------------------------------------------------------

Tensor Core Benchmark
----------------------------------------------------------------------------------------------------
| Instruction        | MMA Shape     | Core Computation                       | Peak Performance   |
|--------------------|---------------|----------------------------------------|--------------------|
| HMMA               | m16n8k8       | MMA(f32,tf32,tf32)                     | 61.52 TFLOPS       |
| HMMA               | m16n8k16      | MMA(f32,f16,f16)                       | 123.05 TFLOPS      |
| HMMA               | m16n8k16      | MMA(f16,f16,f16)                       | 246.17 TFLOPS      |
| HMMA               | m16n8k16      | MMA(f32,bf16,bf16)                     | 123.04 TFLOPS      |
| QMMA               | m16n8k32      | MMA(f32,e4m3,e4m3)                     | 247.22 TFLOPS      |
| QMMA               | m16n8k32      | MMA(f16,e4m3,e4m3)                     | 492.63 TFLOPS      |
| IMMA               | m16n8k32      | MMA(s32,s8,s8)                         | 488.79 TOPS        |
| QMMA.SF            | m16n8k32      | MMA(f32,e4m3,e4m3,scale=ue8m0)         | 486.91 TFLOPS      |
| QMMA               | m16n8k32      | MMA(f32,e3m2,e3m2)                     | 247.19 TFLOPS      |
| QMMA               | m16n8k32      | MMA(f32,e2m1,e2m1)                     | 247.21 TFLOPS      |
| OMMA.SF            | m16n8k64      | MMA(f32,e2m1,e2m1,scale=ue8m0)         | 984.27 TFLOPS      |
| OMMA.SF            | m16n8k64      | MMA(f32,e2m1,e2m1,scale=ue8m0)         | 984.29 TFLOPS      |
----------------------------------------------------------------------------------------------------
