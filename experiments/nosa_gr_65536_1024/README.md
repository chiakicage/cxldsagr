# NOSA GR：instruction + 历史 65536，候选新 token 1024

## 实验目的与内容

测量 NOSA 在 **instruction + 历史合计 65,536 tokens，候选新 token 1,024 tokens**
下的前向性能、各模块 MFU，并通过 Nsight Systems 检查未融合 kernel 和 CPU launch 瓶颈。

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
.venv/bin/python -m pytest experiments/nosa_gr_65536_1024/tests -q
```

`scripts/run.sh` 固定 `--prefix-tokens 65536 --new-tokens 1024`，执行
nsys capture → SQLite export → analyze → MFU，保存各步骤 stdout/stderr，拒绝覆盖已有 run ID。
需可用的 NVIDIA 驱动、GPU 与 nsys。

- `src/capture.py` 调用 `GR.input_generator`、`models.nosa.model`，使用
  `executor.model_executor.run_chunks` 和本实验 `src/instrumentation.py` 的 `ModuleScopes`。
  cache 由模型 cache manager 分配/释放；RoPE 标注跟随 `models.nosa.layers.apply_rotary`。
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

## 当前结果

Run ID：`semantic_65536_1024_20260925`，已完成 GPU 测量、nsys 捕获、SQLite 导出和 MFU 分析。
原始报告：`output/profile/semantic_65536_1024_20260925/nosa_gr_65536_1024.nsys-rep`。
所有以下数字均来自这次正确语义的新运行。

32 层实际形状均为 `Q=[1024,32,128]`、`K=V=[66560,2,128]`；
GR history span 为 `[28,65536)`，candidate span 为 `[65536,66560)`。
完整前向与 prefix+extend 的末 token hidden state 最大绝对差 **0**，余弦相似度 **1.0**，输出均有限。
采集时相关测试通过。删除多长度前向实验后，本实验 8 个测试通过，Ruff、CLI 与 shell 语法检查通过；
该整理只迁移分块执行和模块标注工具，没有重新测量 GPU，以下结果仍来自上述 run ID。
本次 framework 合并保留精确输入与审计，适配分块执行、RoPE 挂钩、cache 生命周期和源码指纹；
当前驱动不可用，未在合并后的 framework 上重新测量 GPU，历史性能数字未改写。

### 基准计时

| 阶段 | 墙钟中位数 ms | Host 提交中位数 ms | 整段 MFU |
| --- | ---: | ---: | ---: |
| full_prefill | 3943.353 | 3879.321 | 55.66% |
| extend | 88.415 | 25.295 | 58.31% |

### GPU 活动与 CPU launch

| 低标注区间 | Kernel 数 | GPU span ms | GPU active ms | Gap ms | Active 占比 | Launch-to-kernel 中位数 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_prefill/0 | 85020 | 3953.483 | 3782.341 | 171.142 | 95.67% | 15.398 |
| extend/0 | 1308 | 89.875 | 87.208 | 2.667 | 97.03% | 33.182 |
| extend/1 | 1308 | 89.277 | 85.249 | 4.028 | 95.49% | 32.361 |
| extend/2 | 1308 | 88.459 | 85.783 | 2.676 | 96.98% | 32.469 |

**该场景整体没有表现为主要受 CPU launch 限制。** 三次 extend 的 GPU 活动覆盖 95.49–97.03%，
launch-to-kernel 中位数 32.36–33.18 ms，表明 CPU 已提前提交大量工作。
extend 全部 GPU 空档为 2.67–4.03 ms，其中下一个 CUDA API 尚未进入的部分仅 0.38–1.72 ms；
后者与 host 提交迟到一致，不能把全部 gap 都归因于 CPU。
第二次有约 1.30 ms 的单次空档，另外两次最大空档约 0.10–0.11 ms。
这些结果不排除局部 launch 开销，也没有测量 CUDA Graph 或融合优化后的实际收益。

### 各模块时间与 MFU

下表 kernel 数与 GPU ms 为 32 层、1024-token extend 的详细采集；Full MFU 对应 66560-token
完整前向的详细采集。各模块 GPU 时间包括归属到该模块的 memcpy/memset。

| 模块 | Extend kernel 数 | Extend GPU ms | Extend MFU | Full MFU |
| --- | ---: | ---: | ---: | ---: |
| model_misc | 1 | 0.001 | 不适用 | 不适用 |
| rope_prepare | 17 | 0.025 | 不适用 | 不适用 |
| embedding | 1 | 0.005 | 不适用 | 不适用 |
| input_layernorm | 288 | 2.441 | 不适用 | 不适用 |
| q_proj | 32 | 1.589 | 69.95% | 71.27% |
| k_proj | 32 | 0.340 | 20.43% | 20.68% |
| v_proj | 32 | 0.323 | 21.54% | 21.60% |
| rope_apply | 320 | 2.292 | 不适用 | 不适用 |
| attention_core | 32 | 54.362 | 65.95% | 65.37% |
| o_proj | 32 | 1.581 | 70.32% | 71.27% |
| residual | 64 | 0.396 | 不适用 | 不适用 |
| post_attention_layernorm | 288 | 2.442 | 不适用 | 不适用 |
| gate_proj | 32 | 6.149 | 72.31% | 73.29% |
| swiglu_elementwise | 64 | 1.381 | 不适用 | 不适用 |
| up_proj | 32 | 5.993 | 74.21% | 75.30% |
| down_proj | 32 | 5.771 | 77.06% | 78.31% |
| final_norm | 9 | 0.076 | 不适用 | 不适用 |
| kv_cache_and_layout | 0 | 0.097 | 不适用 | 不适用 |

矩阵投影和 attention core 共 256 个 kernel；其他模块 **1052 个（80.4%）**，
合计约 **9.15 ms**，其中 KV cache/layout 为 64 次 memcpy，kernel 数为 0。
attention core 本身 **54.36 ms**，占主要 GPU 时间。

未融合证据明确：两组 RMSNorm 各每层 9 个 kernel，RoPE apply 每层 10 个，
SwiGLU 每层 2 个，residual 每层 2 个。实现中的 cast/square/mean/rsqrt/multiply、
RoPE 的 neg/multiply/add/cat，以及 SiLU/multiply 分别启动。可以优先评估融合这些操作，
但 kernel 数量占比不能作为耗时占比。

### 同步点

详细 full prefill 的 65 次 `cudaStreamSynchronize` 全部在 `rope_prepare`，CPU API 累计
2199.984 ms；低标注 full prefill 对应累计 2401.613 ms。`rotary_cos_sin()` 每个 chunk
通过 `torch.tensor(..., device=...)` 重建 LongRoPE factors，伴随小拷贝和同步。
CPU 时间包含等待此前 GPU 队列，不能再加到 GPU 时间上，也不能当成消除同步后的墙钟收益。
本次没有修改或测量常量缓存/算子融合优化。

请求、执行审计、各层形状、全部重复计时、CUDA 活动明细、MFU 和同步归属检查分别保存在
`output/data/semantic_65536_1024_20260925/` 下的 `request.json`、`execution.json`、`attention_shapes.json`、
`metadata.json`、`analysis.json`、`mfu.json`、`sync_audit.json`。

## 历史错误记录

- `20260925` 实际执行 65564 prefix + 996 suffix，不满足本实验要求；旧 README 已保存在
  `output/data/20260925/report_before_execution_fix.md`，原始数据与报告保留供追溯。
- `exact_65536_1024_20260925` 曾强行切成 65536 + 1024，却将 28 个历史 token 放入 extend；
  同样不满足要求，已停止运行并标为无效，不作为本实验结果。
