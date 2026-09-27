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

## 结果与当前状态

**改动后未运行。** `NosaKVCache` 的共享生命周期随 indexer 优化调整，dense forward 的
cache 调用路径也发生变化。本实验受影响的结果、性能结论与对应运行产物已清理；当前没有
可报告的 dense 前向耗时、MFU 或 launch 分析结果，需按上述命令重新测量。

本实验每次通过 GR 生成完整请求并保存到新 run 的 `request.json`，不依赖已删除的运行目录。
