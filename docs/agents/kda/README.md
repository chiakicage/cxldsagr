# KDA agent 内部组件文档

这里集中保存供执行 agent 使用的 NOSA kernel 任务约束、实现检查点和候选调查记录。
面向人的研究主线见[研究思路](../../research.md)，当前判断与缺口见
[研究状态](../../status.md)，后续研究工作见[路线图](../../roadmap.md)。
本次整理只调整文档，不产生新的正确性或性能结果。

| 组件 | 任务与计划 | 已验收实现 | 候选记录 |
| --- | --- | --- | --- |
| Resident indexer | [任务](nosa_indexer/task.md)、[计划](nosa_indexer/implementation_plan.md) | [2026-09-29 检查点](nosa_indexer/checkpoint.md) | [调查记录](nosa_indexer/investigation_log.md) |
| Resident block sparse attention | [任务](nosa_sparse_attention/task.md)、[计划](nosa_sparse_attention/implementation_plan.md) | [2026-09-29 检查点](nosa_sparse_attention/checkpoint.md) | [调查记录](nosa_sparse_attention/investigation_log.md) |
| Offload fetch / attention | [任务](nosa_offload_attention/task.md)、[计划](nosa_offload_attention/implementation_plan.md) | [2026-09-30 检查点](nosa_offload_attention/checkpoint.md) | 失败诊断原保存在 `/tmp`，未迁为实验结果 |

各组件检查点使用中文；任务、执行约束和历史技术记录保留英文。历史记录中
`current`、`active`、`latest` 指原记录时点，不能据此判断当前状态；以各组件检查点为准。
`/tmp` 路径是原证据位置，不保证临时文件仍然存在，也不作为仓库文档链接。

## Resident 两个组件的共用测量契约

目标是完整 indexer 和完整 block sparse attention 在 captured L0、L15、L31 上分别达到
40% useful MFU。负载为 65536-token prefix + 1024 queries，BF16、32 Q heads、2 KV heads、
D128、64-token blocks，使用完整 NOSA 33/64 selection、CIS bias、精确因果遮罩和稳定并列排序。
原 GPU 名称报告为 NVIDIA H20Z / SM90、132 SM；旧 run ID 中的 `h200` 不改变实际设备记录。
峰值分母沿用 989 TFLOPS，PCI identity 和时钟随原运行记录保存。

Useful FLOPs 不计 padding、被 mask 的计算或 QK 重算。Indexer 的 QK useful FLOPs 为
34,616,115,200，40% 对应 87.503 µs；attention 为 68,190,994,432，对应 172.374 µs。
校验、增量派生缓存准备、selection 和全部 helper 均计入各自完整模块。
这组 append-module 测量不包含 prefix 构建、普通 K/V/CIS append 写入和 cache transaction
setup；不属于 offloading 测量。完整 kernel 时间总和、CUDA-event API 区间和 wall completion
区间分别报告，不能相互替代；score-only 也不能代替完整 indexer。

## Resident 联合验收与报告状态

以下是原 run `kda_main_bf16_pair_v3_development` 的完整模块 kernel 时间总和，使用实际
sparse-model L0/L15/L31 captures、GPU0、10 次 warmup、30 次 eager samples 和 3 次 profiler
repeats。完整数据见[模块检查点报告](../../../experiments/nosa_kernel_mfu/README.md#完整模块检查点)。

| Layer | Indexer µs | Indexer MFU | Attention µs | Attention MFU |
| --- | ---: | ---: | ---: | ---: |
| 0 | 141.471 | 24.741% | 175.391 | 39.312% |
| 15 | 137.854 | 25.390% | 173.247 | 39.798% |
| 31 | 137.504 | 25.455% | 170.879 | 40.350% |

只有 L31 attention 达到 40%，两个完整模块的共同目标仍未完成。CUDA-event API 和 wall
时间、源文件哈希及更早开发测量保存在[indexer 调查记录](nosa_indexer/investigation_log.md)。
这些未经配对的开发检查点不能替代候选的 paired A/B 证据或其他论文实验。

原验收包含 7 个 native 组件重建、361 项 GPU 回归、18 组完整 kernel 归因检查、独立压缩
和选块一致性检查，三个 captured layer 的选择 ID 均未改变。发布前重新运行的 GPU 全局
入口为 361 项通过，测量与归因工具 CPU 测试为 560 项通过，Ruff 检查和格式检查通过。
当时全局 CPU 入口两次都在既有 pattern 分析的 NumPy 数组排序阶段长时间未完成，限制
线程数后仍未消除，已停止；不能将这次 resident 检查点写成全局 CPU/GPU 全部通过。
其他日期或组件的 CPU 测试结果也不能自动补足这条记录。

发布时只对 attention Python adapter 做等价格式化，原 measurement metadata 保留原哈希。
`git_commit` 指测量时未提交工作树的基线，实际实现身份由源码哈希、快照及已登记的
AST 等价格式化变体确定。

| 实验 | 原 run / 实现 | 状态与边界 |
| --- | --- | --- |
| [完整模型 native / Triton](../../../experiments/indexer_block_sparse_profile/README.md) | `sparse_native_h200_gpu1_20260929_01`、`sparse_triton_h200_gpu1_20260929_01`，`94bf521` | 2026-09-29 已按独立 sparse 轨迹补测 64K+1K full-prefill / extend；同源、设备、请求、完整归因及输出验收通过，旧 sparse 报告和运行产物已替换。该次报告记录 553 项实验 CPU 测试、361 项全局 GPU 测试通过。 |
| [Synthetic operator 对照](../../../experiments/nosa_kernel_mfu/README.md) | `kernel_mfu_h200_gpu1_20260928_071840`，`020961b` | 补测未完成；原 468.68 / 216.17 µs attention / score 数字只描述旧实现。完整模块检查点另列，不把两类结果混用。 |
| [Full-NOSA pattern](../../../experiments/nosa_indexer_pattern_65536_1024/README.md) | 以原实验 README 中的 run ID 和实现记录为准 | 受影响的 sparse / full-NOSA 部分待补测；旧结果保留原边界，未受影响的 dense-only 部分不因此重跑。 |

上述状态描述原 resident 检查点。2026-09-30 新增 offload 分支后，模型/cache 源码图
已扩展，当前分支的完整模型实验尚未补测；2026-10-01 算子目录迁移后的性能也未重新
测量。当前研究进度统一见[研究状态](../../status.md)，原 run 不作为这些改动后的实测结果。

正确性测试用于代码验收，不能替代模型质量或论文实验结论。保留待替换的报告和运行产物，
直到新实现的正确性、来源稳定性与测量完整性通过；发布新 run 后，再同步替换和清理受影响的
旧素材。失败候选只保留调查结论与原临时证据路径，不作为有效实验对照。

## 复验入口

以下是原 resident 任务使用的入口；先选择空闲 GPU，并为新运行使用新的 run ID。
本次文档整理没有执行这些命令。

```bash
bash scripts/run_tests.sh cpu
bash scripts/run_tests.sh gpu
bash experiments/nosa_kernel_mfu/scripts/capture.sh <new-input-run> --kernel-backend native
bash experiments/nosa_kernel_mfu/scripts/run.sh <new-run> --peak-tflops 989 --reference-all
bash experiments/nosa_kernel_mfu/scripts/run.sh <new-real-run> --input-dir <captured-input-dir> --peak-tflops 989 --reference-all
bash experiments/indexer_block_sparse_profile/scripts/run.sh <new-native-run> --kernel-backend native --peak-tflops 989
bash experiments/indexer_block_sparse_profile/scripts/run.sh <new-triton-run> --kernel-backend triton --peak-tflops 989
```

NCU full / PM sampling / source counters 使用隔离的公开算子 harness、现有 `-lineinfo` 构建及
`ncu_report` 解析。有效运行采用实验目录的 `output/{data,log,profile}/<run_id>`，需要呈现的
素材放 `report/`；项目 `AGENTS.md` 优先于通用 KDA 产物建议。
