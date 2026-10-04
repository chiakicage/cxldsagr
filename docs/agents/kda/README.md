# KDA agent 内部组件文档

这里集中保存供执行 agent 使用的 kernel 任务约束、实现检查点和候选调查记录。
当前研究判断与缺口见[研究状态](../../status.md)，接下来要做的研究任务见
[路线图](../../roadmap.md)。

| 组件 | 任务与计划 | 已验收实现 | 候选记录 |
| --- | --- | --- | --- |
| Resident indexer | [任务](nosa_indexer/task.md)、[计划](nosa_indexer/implementation_plan.md) | [2026-09-29 检查点](nosa_indexer/checkpoint.md) | [调查记录](nosa_indexer/investigation_log.md) |
| Resident block sparse attention | [任务](nosa_sparse_attention/task.md)、[计划](nosa_sparse_attention/implementation_plan.md) | [2026-09-29 检查点](nosa_sparse_attention/checkpoint.md) | [调查记录](nosa_sparse_attention/investigation_log.md) |
| Offload fetch / attention | [任务](nosa_offload_attention/task.md)、[计划](nosa_offload_attention/implementation_plan.md) | [2026-09-30 检查点](nosa_offload_attention/checkpoint.md) | 失败诊断原保存在 `/tmp`，未迁为实验结果 |
| NOSA Q128 offload | [任务](nosa_q128_overlap/task.md)、[执行索引](nosa_q128_overlap/implementation_plan.md) | [检查点](nosa_q128_overlap/checkpoint.md)：候选均未通过提升门槛，未合入；对齐已减少 sector，但首个重叠样本仍未达标 | [当前 64-CTA NCU 诊断](nosa_q128_overlap/independent_warps_ncu_findings.md)、[对齐对照](nosa_q128_overlap/aligned_host_diagnostic_findings.md)、[调查记录](nosa_q128_overlap/investigation_log.md) |
| DeepSeek indexer quantization | [任务](deepseek_quantization/task.md)、[计划](deepseek_quantization/implementation_plan.md) | [2026-10-03 算子检查点](deepseek_quantization/checkpoint.md) | [调查记录](deepseek_quantization/investigation_log.md) |
| DeepSeek cache metadata | [任务](deepseek_cache_metadata/task.md)、[计划](deepseek_cache_metadata/implementation_plan.md) | [C3b 算子检查点](deepseek_cache_metadata/checkpoint.md) | 端到端验收见 [C3 集成记录](../system/deepseek_motivation_c3_integration.md) |
| DeepSeek indexer causal tail | [任务](deepseek_indexer_tail/task.md)、[计划](deepseek_indexer_tail/implementation_plan.md) | [C4a 算子检查点](deepseek_indexer_tail/checkpoint.md) | [候选草案](deepseek_indexer_tail/draft.md) |
| DeepSeek official top-k wrapper | [任务](deepseek_topk_wrapper/task.md)、[计划](deepseek_topk_wrapper/implementation_plan.md) | [C5 算子检查点](deepseek_topk_wrapper/checkpoint.md) | 完整请求见 [C6 验收](../system/deepseek_motivation_c6_integration.md) |
| DeepSeek sparse recall metadata | [任务](deepseek_sparse_recall/task.md)、[计划](deepseek_sparse_recall/implementation_plan.md) | [C6 算子检查点](deepseek_sparse_recall/checkpoint.md) | [失败顺序审查](deepseek_sparse_recall/lifecycle_review.md)；完整请求见 C6 验收 |
| DeepSeek dense prefetch | [任务](deepseek_dense_prefetch/task.md)、[计划](deepseek_dense_prefetch/implementation_plan.md) | [C7a 组件检查点](deepseek_dense_prefetch/checkpoint.md)，已通过组合数值验收 | [限制 gather CTA 数的候选计划](deepseek_dense_prefetch/c7b_implementation_plan.md) |
| DeepSeek dense MLP packing | [任务](deepseek_mlp_packing/task.md)、[计划](deepseek_mlp_packing/implementation_plan.md) | 组件 eager / Graph 测量与生产正确性验收已通过；接入后的 serving 性能未测量 | [接入计划](deepseek_mlp_packing/integration_plan.md)、[源码与验证记录](deepseek_mlp_packing/checkpoint.md) |
| DeepSeek linear activation quantization | [任务](deepseek_linear_quantization/task.md)、[计划](deepseek_linear_quantization/implementation_plan.md) | 前两版数值通过但 Graph 性能回退；整数归约候选已通过离线检查，待 GPU 验收 | [检查点](deepseek_linear_quantization/checkpoint.md)、[整数归约计划](deepseek_linear_quantization/redux_candidate_plan.md) |
| DeepSeek ECHO prefetch hint | [任务](deepseek_prefetch_hint/task.md)、[计划](deepseek_prefetch_hint/implementation_plan.md) | [精确阈值更新检查点](deepseek_prefetch_hint/checkpoint.md) | [组合验收](../system/deepseek_motivation_c7_hint_validation.md)，正式测量进行中 |
| DeepSeek norm I/O | [任务](deepseek_norm_io/task.md)、[计划](deepseek_norm_io/implementation_plan.md) | [原型检查点](deepseek_norm_io/checkpoint.md)，尚未合入生产实现 | [候选草案](deepseek_norm_io/draft.md) |

组件检查点、任务、执行约束和技术记录可使用英文。历史记录中
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
repeats。这条历史记录已由 2026-10-04 的配对测量替换；原调查语义保留，
当前数据见[完整模块报告](../../../experiments/nosa_kernel_mfu/README.md#完整模块检查点)。

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

当前报告导航更新于 2026-10-05；下表以已发布的 run 和各自源码快照为准。
上面的 2026-09-29 数值与验收过程仅记录当时组件任务，不作为当前报告或配对基线。
Dense/sparse 目录已合并，check/bench/profile 入口已拆分；本次整理没有新增测量。

| 实验 | 已发布 run | 状态与边界 |
| --- | --- | --- |
| [完整模型 native / Triton](../../../experiments/nosa_baseline_performance/README.md) | `sparse_flags_native_20261004_01`、`sparse_flags_triton_20261004_01` | 64K+1K、完整 32 层 resident 前向；独立墙钟和独立 nsys/module 采集，同请求、109 份源码及完整归因通过。原始 SQLite 重算和 364 项独立报告检查通过，两个旧 run 已清理。 |
| [Synthetic / captured operator](../../../experiments/nosa_kernel_mfu/README.md) | `kernel_flags_synthetic_20261004_01`、`kernel_flags_real_20261004_01` | 全部 28 行检查 1024-query FP32 reference。Synthetic 与固定真实输入分别报告；真实输入保留 `kda_inputs_baseline_20260928_1345`，不声称它是新模型轨迹。 |
| [完整模块 native / Triton](../../../experiments/nosa_kernel_mfu/README.md#完整模块检查点) | `modules_flags_native_20261004_01`、`modules_flags_triton_20261004_01` | L0/L15/L31 的完整 indexer/attention，12 行及 36 份 trace 通过复核。Kernel 总时间、event 与 wall 分开报告；只有 L31 attention 超过 40%，两个完整模块的共同目标仍未完成。四个 kernel/module run 共 477 项独立报告检查通过，三个旧测量 run 已清理。 |
| [Dense 全模型](../../../experiments/nosa_baseline_performance/README.md) | `dense_wrapper_20261004_01` | 完整 32 层 dense 前向，SQLite/MFU 重算和 80 项独立报告检查通过。基准在 nsys 进程内关闭 capture 时计时，没有独立无 profiler 进程。旧 dense 仅在下游分析替换后清理。 |
| [Full-NOSA pattern](../../../experiments/nosa_indexer_pattern_65536_1024/README.md) | `nosa_pattern_flags_20261004_01` | 三组实际选择、QA32/QA64、分解与分布保留；假定 dense MFU/带宽的时间、overlap 和阈值分析已退出。历史复核数量包含已退出支线，不作为当前范围的验收数量。 |
| [A1024 offload 算子](../../../experiments/nosa_offload_overlap/README.md) | `nosa_cached_fetch_20261004_01`、`nosa_cached_fetch_confirm40_20261004_01`、`nosa_cached_fetch_profile_20261004_01` | 冻结真实输入，默认调用每次重取完整稀疏并集。20/40 次完整 API 配对均支持融合收益，9 个内部样本双比率均过 90%；普通 owned-cache 完整 32 层数值另行通过。657 项 timing 与 27 个 profile range 复核后发布，三个旧 run 已清理；不证明固定 P/NH 命中或 serving 性能。 |

Resident 各项保持 A1024 测量边界，offload 行单独报告冷稀疏并集的算子回放。
它们不能替代固定 P/NH、A128 或 budget-serving 的验收。当前研究进度仍见
[研究状态](../../status.md)；本表只维护实验入口。

正确性测试用于代码验收，不能替代模型质量或论文实验结论。保留待替换的报告和运行产物，
直到新实现的正确性、来源稳定性与测量完整性通过；发布新 run 后，再同步替换和清理受影响的
旧素材。失败候选只保留调查结论与原临时证据路径，不作为有效实验对照。

## 复验入口

当前入口的参数与独立验收要求见各脚本帮助。本次整理只运行 CPU 回归和帮助检查。

```bash
bash scripts/run_tests.sh cpu
bash experiments/nosa_kernel_mfu/scripts/run.sh --help
bash experiments/nosa_kernel_mfu/scripts/modules.sh --help
bash experiments/nosa_baseline_performance/scripts/dense.sh --help
bash experiments/nosa_baseline_performance/scripts/sparse.sh --help
```

NCU full / PM sampling / source counters 使用隔离的公开算子 harness、现有 `-lineinfo` 构建及
`ncu_report` 解析。有效运行采用实验目录的 `output/{data,log,profile}/<run_id>`，需要呈现的
素材放 `report/`；项目 `AGENTS.md` 优先于通用 KDA 产物建议。
