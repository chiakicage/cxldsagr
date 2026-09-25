# NOSA GR：instruction + 历史 65536，候选新 token 1024

## 实验目的与内容

测量 NOSA 在 **instruction + 历史合计 65,536 tokens，候选新 token 1,024 tokens**
下的前向性能、各模块 MFU，并通过 Nsight Systems 分析 GPU 模块开销与 CPU launch 瓶颈。

## 精确输入与执行边界

| 部分 | Token 数 | 在完整输入中的区间（左闭右开） |
| --- | ---: | --- |
| instruction / 固定模板前缀 | 28 | [0, 28) |
| 用户历史 | 65,508 | [28, 65,536) |
| 已缓存 prefix（instruction + 历史） | **65,536** | **[0, 65,536)** |
| 候选新 token（含候选段结束模板） | **1,024** | **[65,536, 66,560)** |
| 完整输入 | **66,560** | [0, 66,560) |

GR 原有 API 的 `item_tokens` 包含 instruction，因此实验适配为
`user_lengths=(65508,)`、`item_lengths=(1052,)`：1052 = instruction 28 + 候选 1024。
这只是生成器预算字段的约定；**实际稳定 prefix=65536、candidate suffix=1024**。
不移动任何历史 token 到 extend，也不在 66560 之外追加 instruction。
`--prefix-tokens` 和 `--new-tokens` 分别控制上述两个实际执行长度；instruction 长度由 tokenizer 计算。

运行时检查 GR 语义边界与执行边界相同，并逐层审计 attention：32 层均须满足
`Q=[1024,32,128]`、`K=V=[66560,2,128]`。进入 extend 前 KV 长度为 65536，
extend 为单次 1024-token forward，结束时 KV 长度为 66560。
完整请求、执行边界、逐层形状分别保存为 `request.json`、`execution.json`、`attention_shapes.json`。

模型为 NOSA-8B 全部 32 层，BF16，FlashInfer dense Full Attention；输出 normalized hidden states，
不执行 LM head、自回归 decode、sparse selection 或 offloading。
在实验进程内将模型上下文与 GR 检查上限覆盖到 66560；使用原有 LongRoPE factors。
磁盘 checkpoint 与生产默认上下文仍为 32768。本实验不评价超出训练上下文后的模型质量。

## 运行方式与调用模块

从仓库根目录运行：

```bash
bash experiments/nosa_gr_65536_1024/scripts/run.sh run_001
.venv/bin/python -m experiments.nosa_gr_65536_1024.src.capture --help
```

`scripts/run.sh` 固定 `--prefix-tokens 65536 --new-tokens 1024`，执行
nsys capture → SQLite export → analyze → MFU，保存各步骤 stdout/stderr，拒绝覆盖已有 run ID。
需可用的 NVIDIA 驱动、GPU 与 nsys。

- `src/capture.py` 调用 `GR.input_generator`、`models.nosa.model`，使用
  `executor.model_executor.run_chunks` 和本实验 `src/instrumentation.py` 的 `ModuleScopes`。
  cache 由模型 cache manager 分配/释放；RoPE 标注覆盖位置缓存访问与融合 Q/K 旋转。
  `src/sources.py` 记录模型、layers、cache、executor、serving、算子和 GR 的源码指纹。
- `src/analyze.py` 使用标准库 SQLite 读取 CUDA/NVTX 活动，计算活动并集、空档和模块归属。
- `src/mfu.py` 使用本次 execution、实际模型配置与 nsys 模块时间生成 `mfu.json`。
- `tests/` 检查语义/执行边界一致、实际 attention 形状、GPU 空档计算及分段 FLOPs 守恒。
- `output/log/<run_id>/` 保存 stdout/stderr；`output/data/<run_id>/` 保存请求、形状、metadata、
  源码快照、SQLite、analysis.json、mfu.json；`output/profile/<run_id>/` 保存原始 `.nsys-rep`。
  全部 output 默认不进 Git。

## 测量方法

平台：NVIDIA M403（CUDA 名称 H200，132 SM、SM90），GPU 0；PyTorch 2.10.0+cu132 / CUDA 13.2，
FlashInfer 0.6.18，驱动 570.124.06，Nsight Systems 2025.6.3。
每阶段预热 2 次、基准计时 5 次取中位数。加载、GR 生成、prefix 构建、形状审计和基准计时
在捕获区间外；nsys 进程仍已启动，因此“采集关闭”不表示完全没有 profiler 注入影响。
关闭 CPU IP sampling/context-switch tracing，不同时启用 PyTorch profiler。

NVTX 区间：

- `GR/light/full_prefill/0`：从空 KV 处理 66560 tokens，65 个 1024-token chunk。
- `GR/light/extend/0`、`/1`、`/2`：65536-token prefix 已就绪，执行 1024-token 候选，重复 3 次。
- `GR/detailed/full_prefill/0`、`GR/detailed/extend/0`：相同计算，加模块与层号标记，
  如 `nosa::extend/input_layernorm/0`。

GPU active 为 kernel/memcpy/memset 时间区间的并集，GPU span 为首个活动开始到最后一个结束。
active 占比不是 SM occupancy 或 MFU。Host 提交时间包含模型内部 CUDA 等待。
launch-to-kernel 表示 kernel 开始减对应 launch API 结束（负数截为 0），用于观察排队。

MFU 定义：H200 BF16 dense 标称 989 TFLOPS，
仅计有效矩阵 FLOPs。attention 使用因果有效对数 `T*P + T*(T+1)/2`；本次 extend 的
`P=65536,T=1024`。模块分母为详细采集 GPU 时间，整段分母为采集关闭区间的墙钟中位数。
非矩阵模块 MFU 不适用，报告其时间；不使用稀疏算力分母。

## 测量实现

CUDA BF16 推理使用 FlashInfer 普通层和 RoPE kernel：

- 首层独立 RMSNorm；跨层 residual、post-attention residual 和 final norm 使用 add+RMSNorm。
- Q/K/V 由一次 `qkv_proj` GEMM 生成，切片视图直接用于 RoPE 与 attention。
- gate/up 由一次 `gate_up_proj` GEMM 生成，随后执行融合 SiLU×up。
- 合并权重在模型加载时直接写入最终参数切片，前向不拼接权重，不为 RoPE 额外复制 Q/K。
  K/V 仍逐层写入 resident cache；这部分写入计入 `kv_cache_and_layout`。
- 静态 LongRoPE 预生成 FP32 cos/sin cache，按位置切片复用；每层 Q/K 使用一次
  `apply_rope_with_cos_sin_cache_inplace`，以 split-half 布局旋转。保持原 checkpoint 的
  逐频率 factors 和 attention_factor，旋转后才写回 BF16。
- 位置缓存不进入 checkpoint；首次构建发生在预热前，稳态 chunk 不重建常量或位置。

## 当前 GPU 结果（2026-09-25）

Run ID：`flashinfer_merged_gemm_65536_1024_20260925_02`。完成 32 层 GPU 前向、Nsight Systems 捕获、SQLite 导出与 MFU 分析。

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/nosa_gr_65536_1024/scripts/run.sh flashinfer_merged_gemm_65536_1024_20260925_02
```

本次使用上节硬件、依赖、BF16 和精确输入，测量期间没有并行 GPU 测试。
GPU UUID 为 `2522820c-89d9-aa17-f79c-ca8cc767fb77`；metadata 保存实际依赖版本、硬件、
模型配置、源码指纹和每次计时。32 层实际形状均为 `Q=[1024,32,128]`、
`K=V=[66560,2,128]`，extend 前后 cache 长度分别为 65536、66560。
完整前向和 prefix+extend 的末 token hidden 最大绝对差 **0**、余弦相似度 **1.0**，输出均有限。

### 墙钟与整段 MFU

| 阶段 | 墙钟中位数 ms | Host 提交中位数 ms | 整段 MFU |
| --- | ---: | ---: | ---: |
| full_prefill | 3498.293 | 3284.493 | 62.75% |
| extend | 81.271 | 9.442 | 63.44% |

Host 提交时间包含 CUDA API 内部等待，不与 GPU 时间相加。

### 合并投影与实际 kernel

每层有 4 次投影 GEMM：QKV、O、gate/up、down。QKV 输出为 `[1024,4608]`，
gate/up 输出为 `[1024,32768]`；原 checkpoint 在加载时合并到最终权重存储。
下表为 detailed extend 的 32 层合计，包含 final norm。

| 操作 | Kernel 数 | GPU ms |
| --- | ---: | ---: |
| Q/K/V 合并 GEMM | 32 | 1.812897 |
| gate/up 合并 GEMM | 32 | 12.323178 |
| RMSNorm + residual | 65 | 0.492129 |
| SwiGLU 激活 | 32 | 0.763969 |
| Q/K RoPE | 32 | 0.240096 |
| K/V cache 写入 | 64 | 0.170816 |
| RoPE 准备（缓存复用） | 0 | 0 |

QKV 与 gate/up 各层均只有一个 GEMM kernel，时间和 MFU 按合并模块统计。
Trace 同时确认 FlashInfer `RMSNormKernel`、`FusedAddRMSNormKernel`、
`activation::act_and_mul_kernel<…silu>` 与 `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`。

Q/K/V 是合并输出的行跨距视图，RoPE 直接在 Q/K 原位置旋转，V 保持不变。
K/V 各自从这些视图写入连续的 resident cache，因此每层有两个 `direct_copy_kernel_cuda`，
合计 64 个；它们属于必要的 cache 写入。没有独立的权重拼接、Q/K contiguous 或 activation
拼接 kernel。gate/up GEMM 另有 32 次 MEMSET，时间包含在其模块 GPU 时间中。
本次捕获没有 MEMCPY 活动；分析器支持 Nsight 省略零活动类型的表。

预生成的 RoPE cache 为 `[66560,128]` FP32，约 32.5 MiB，另有约 0.51 MiB 的 int64 positions。
稳态 `rope_prepare` 没有 GPU kernel、H2D 拷贝或同步；采集区间没有 `cudaStreamSynchronize`。
各段计时首尾的两次显式 `cudaDeviceSynchronize` 用于界定测量边界。

### GPU 活动与 CPU launch

| 低标注区间 | Kernel 数 | GPU span ms | GPU active ms | Gap ms | Active 占比 | Launch-to-kernel 中位数 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_prefill/0 | 23010 | 3487.119 | 3455.286 | 31.833 | 99.09% | 141.403 |
| extend/0 | 354 | 80.678 | 79.961 | 0.717 | 99.11% | 34.553 |
| extend/1 | 354 | 80.521 | 79.826 | 0.695 | 99.14% | 34.451 |
| extend/2 | 354 | 80.271 | 79.588 | 0.683 | 99.15% | 34.398 |

三次 extend 的 GPU gap 为 0.68–0.72 ms，其中下一次 CUDA API 尚未进入的部分为
0.18–0.20 ms；launch-to-kernel 中位数约 34.4–34.6 ms，说明 CPU 已提前提交工作。
本场景整体仍以 GPU 计算为主，不能把全部 GPU gap 归因于 CPU launch。

### 各模块时间与 MFU

GPU 时间来自详细采集的模块归因。非矩阵模块 MFU 不适用。

| 模块 | Extend kernel 数 | Extend GPU ms | Extend MFU | Full MFU |
| --- | ---: | ---: | ---: | ---: |
| rope_prepare | 0 | 0.000 | 不适用 | 不适用 |
| embedding | 1 | 0.005 | 不适用 | 不适用 |
| input_layernorm | 1 | 0.005 | 不适用 | 不适用 |
| input_layernorm_add_residual | 31 | 0.232 | 不适用 | 不适用 |
| qkv_proj | 32 | 1.813 | 68.99% | 67.29% |
| rope_apply | 32 | 0.240 | 不适用 | 不适用 |
| attention_core | 32 | 57.048 | 62.85% | 61.03% |
| o_proj | 32 | 1.652 | 67.31% | 65.41% |
| post_attention_layernorm_add_residual | 32 | 0.247 | 不适用 | 不适用 |
| gate_up_proj | 32 | 12.323 | 72.17% | 70.56% |
| swiglu_elementwise | 32 | 0.764 | 不适用 | 不适用 |
| down_proj | 32 | 6.043 | 73.59% | 71.84% |
| final_norm_add_residual | 1 | 0.007 | 不适用 | 不适用 |
| kv_cache_and_layout | 64 | 0.171 | 不适用 | 不适用 |

attention core 为 extend 的主要 GPU 开销：**57.048 ms**，
约占模块 GPU 时间合计的 **70.8%**。
当前基线使用合并 QKV/gate-up GEMM、FlashInfer 普通层与 Q/K RoPE；有效矩阵工作量为
full prefill **2,170,865,718,394,880 FLOPs**、extend **50,990,120,173,568 FLOPs**
（投影与 attention 合计）。合并减少投影调用次数，端到端性能仍主要由 attention 与矩阵计算决定。
本次没有测量 CUDA Graph、sparse attention 或 offloading。

原始 trace 为 `output/profile/flashinfer_merged_gemm_65536_1024_20260925_02/nosa_gr_65536_1024.nsys-rep`；
请求、逐层形状、源码快照、完整计时、SQLite、模块分析和 MFU 在
`output/data/flashinfer_merged_gemm_65536_1024_20260925_02/`，对应 stdout/stderr 在 `output/log/flashinfer_merged_gemm_65536_1024_20260925_02/`。
