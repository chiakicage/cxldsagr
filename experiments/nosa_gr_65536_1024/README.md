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

目标平台为 NVIDIA Hopper / H200（SM90），使用仓库环境中的 PyTorch、CUDA、FlashInfer
与 Nsight Systems。重跑时由 metadata 记录实际 GPU、驱动和依赖版本。
默认每阶段预热 2 次、基准计时 5 次取中位数。加载、GR 生成、prefix 构建、形状审计和基准计时
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

## 结果（2026-09-28）

本节来自完成的 run `dense_h200_gpu1_20260928_01`；32 层前向、Nsight Systems 捕获、SQLite 导出和分析均已完成。

```bash
CUDA_VISIBLE_DEVICES=1 bash experiments/nosa_gr_65536_1024/scripts/run.sh dense_h200_gpu1_20260928_01
```

设备上报 `NVIDIA H20Z`，132 SM、SM90；按本机已确认的 H200 配置，MFU 分母采用
**989 TFLOPS dense BF16 标称峰值**。使用物理 GPU 1，程序内为 `cuda:0`，PCI 地址
`0000:67:00.0`；CUDA 与 NVML 返回的 UUID 不同，两者原值均保留在 metadata，设备对应关系
在运行前按 PCI 地址核对。驱动为 570.124.06、功率上限 700 W。
本次运行使用 PyTorch **2.12.1+cu130**（CUDA 构建版本 **13.0**）、FlashInfer **0.6.18**，
Nsight Systems **2025.6.3.541**；metadata 记录时间为 2026-09-28 07:33:54 UTC。

输入仍为上文的 65536-token prefix 和 1024-token suffix，BF16、1024-token chunk。
每阶段预热 2 次、基准重复 5 次。32 层实际 attention 形状均为
`Q=[1024,32,128]`、`K=V=[66560,2,128]`，extend 前后 cache 长度分别为 65536、66560。
完整前向与 prefix+extend 的末 token hidden 均有限，最大绝对差为 **0**、余弦相似度为 **1**。
这项验收覆盖本次 dense 执行拆分，不是模型质量评估。

### 采集关闭区间的延迟与整段 MFU

表中时间来自 nsys 进程内、CUDA profiler capture 尚未启动的基准阶段。
**nsys 的注入仍可能影响计时；本次没有独立的无 profiler 进程计时。**
墙钟包括提交与完成等待；host submit 也可能包含 CUDA API 内部等待，不能与 GPU 时间相加。

| 阶段 | 墙钟中位数 ms [min, max] | Host submit 中位数 ms | 整段 MFU |
| --- | ---: | ---: | ---: |
| full_prefill | 3516.808 [3507.447, 3517.771] | 3301.988 | 62.41% |
| extend | 81.590 [81.155, 83.588] | 11.876 | 63.19% |

有效矩阵工作量为 full prefill **2,170,865,718,394,880 FLOPs**、extend
**50,990,120,173,568 FLOPs**。整段 MFU 的分母为上表墙钟中位数；后面的模块 MFU
来自独立的 detailed 捕获区间，两类时间不能相减来估计 launch 开销。

### 模块 GPU 时间与 MFU

每个阶段只有一次 detailed 捕获；GPU 时间为按 CUDA launch 关联到模块的活动 duration
之和。矩阵模块计算有效 MFU，其他模块保留时间并标为不适用。

| 模块 | Full GPU ms | Full MFU | Extend kernel 数 | Extend GPU ms | Extend MFU |
| --- | ---: | ---: | ---: | ---: | ---: |
| `embedding` | 0.303 | 不适用 | 1 | 0.005 | 不适用 |
| `input_layernorm` | 0.317 | 不适用 | 1 | 0.005 | 不适用 |
| `input_layernorm_add_residual` | 15.220 | 不适用 | 31 | 0.233 | 不适用 |
| `qkv_proj` | 120.480 | 67.48% | 32 | 1.819 | 68.77% |
| `rope_apply` | 15.744 | 不适用 | 32 | 0.240 | 不适用 |
| `kv_cache_and_layout` | 11.224 | 不适用 | 64 | 0.171 | 不适用 |
| `attention_core` | 1920.631 | 61.14% | 32 | 57.108 | 62.78% |
| `o_proj` | 110.462 | 65.42% | 32 | 1.661 | 66.94% |
| `post_attention_layernorm_add_residual` | 16.272 | 不适用 | 32 | 0.250 | 不适用 |
| `gate_up_proj` | 817.127 | 70.75% | 32 | 12.351 | 72.01% |
| `swiglu_elementwise` | 50.092 | 不适用 | 32 | 0.767 | 不适用 |
| `down_proj` | 401.781 | 71.94% | 32 | 6.064 | 73.33% |
| `final_norm_add_residual` | 0.493 | 不适用 | 1 | 0.007 | 不适用 |

每层 QKV、O、gate/up、down 各有一次 GEMM，extend 共 128 个投影 kernel；
`gate_up_proj` 另含 32 次 MEMSET，其耗时计入对应模块。K/V cache 写入为每层两个 kernel，
extend 共 64 个。RoPE 位置缓存复用，`rope_prepare` 没有归属的 GPU 活动。
详细 extend 总计 **354 个 kernel**，attention core 为 **57.108 ms**，占模块 GPU
活动时间的 **70.78%**，是本次 dense 路径最大的单项开销。

### GPU 活动与提交排队

下表来自 light 捕获；active 是 kernel/memcpy/memset 活动区间的并集。
“API 前 gap”只统计 GPU 已空闲且下一关联 CUDA API 尚未进入的空档部分，
launch-to-kernel 为 API 结束到 kernel 开始的等待中位数；均按上文定义计算。

| Light 区间 | Kernel 数 | GPU span ms | GPU active ms | Gap ms | Active 占比 | API 前 gap ms | Launch-to-kernel 中位数 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_prefill/0 | 23010 | 3474.530 | 3441.378 | 33.152 | 99.05% | 1.592 | 140.299 |
| extend/0 | 354 | 79.102 | 77.939 | 1.163 | 98.53% | 0.613 | 31.152 |
| extend/1 | 354 | 81.039 | 80.048 | 0.990 | 98.78% | 0.447 | 32.940 |
| extend/2 | 354 | 80.699 | 79.816 | 0.883 | 98.91% | 0.341 | 32.314 |

三次 extend 的 GPU gap 为 **0.883–1.163 ms**，其中 API 尚未进入的部分为
**0.341–0.613 ms**；launch-to-kernel 中位数为 **31.152–32.940 ms**。
本次 trace 中 GPU active 占比超过 98.5%，且已提交工作存在较长排队，支持主要时间花在
GPU 执行上的判断。不能把全部 gap 归因于 CPU launch，也不能将 active 占比解释为 SM occupancy。
本实验没有测量 CUDA Graph、sparse attention 或 offloading。

### 报告数据与复现

以下文件均来自 `dense_h200_gpu1_20260928_01`，没有沿用此前运行的数字。
[metadata.json](report/metadata.json)、[analysis.json](report/analysis.json) 和
[mfu.json](report/mfu.json) 从完成的 run 原样复制；[provenance.json](report/provenance.json)
记录输入与报告文件的 SHA256、生成方式和本次一致性检查。
44 份源码快照的 SHA256 与 metadata 及报告生成时的当前源码一致。

| 报告数据 | 生成方式 |
| --- | --- |
| [timings.csv](report/timings.csv) | 从 metadata 的 5 次 timings、medians 和 mfu 的 end_to_end 字段生成；保留完整精度 |
| [module_mfu.csv](report/module_mfu.csv) | analysis 的 detailed modules 与同阶段 mfu 按模块名关联；非矩阵 FLOPs/MFU 留空 |
| [launch.csv](report/launch.csv) | analysis 中全部 6 个 NVTX 根区间的活动与提交指标；CSV 的 launch 等待单位为微秒 |

CSV 使用 Python 标准库 `csv.DictWriter` 从上述 JSON 字段导出，完整副本同时保存在
`output/data/dense_h200_gpu1_20260928_01/`。分析 JSON 由 `src/analyze.py` 读取该 run 的 SQLite 生成，
MFU JSON 由 `src/mfu.py` 使用同次 metadata 和 analysis 生成，可按下面的入口重新计算：

```bash
.venv/bin/python -m experiments.nosa_gr_65536_1024.src.analyze \
  experiments/nosa_gr_65536_1024/output/data/dense_h200_gpu1_20260928_01/nsys.sqlite \
  --output experiments/nosa_gr_65536_1024/output/data/dense_h200_gpu1_20260928_01/analysis.json
.venv/bin/python -m experiments.nosa_gr_65536_1024.src.mfu \
  experiments/nosa_gr_65536_1024/output/data/dense_h200_gpu1_20260928_01 --peak-tflops 989
```

完整请求、逐层形状、源码快照和 SQLite 在 `output/data/dense_h200_gpu1_20260928_01/`；
原始 `.nsys-rep` 在 `output/profile/dense_h200_gpu1_20260928_01/`，stdout/stderr 在
`output/log/dense_h200_gpu1_20260928_01/`。请求文件 SHA256 为
`0bbabf07bc72f9804dbc2e9644cf708e21eafe237631f4a56c64b20f9cda7f2f`。
