# KDA agent 内部组件文档

这里集中保存供执行 agent 使用的 kernel 任务约束、实现检查点和候选调查记录。
当前研究判断与缺口见[研究状态](../../status.md)，接下来要做的研究任务见
[路线图](../../roadmap.md)。下表用于定位组件材料，表内状态属于所链接记录的日期；
当前是否已集成、是否完成补测，应结合对应实验 README 判断。

| 组件 | 任务与计划 | 检查点及原记录状态 | 调查与阶段记录 |
| --- | --- | --- | --- |
| DeepSeek indexer quantization | [任务](deepseek_quantization/task.md)、[计划](deepseek_quantization/implementation_plan.md) | [2026-10-03 算子检查点](deepseek_quantization/checkpoint.md) | [调查记录](deepseek_quantization/investigation_log.md) |
| DeepSeek cache metadata | [任务](deepseek_cache_metadata/task.md)、[计划](deepseek_cache_metadata/implementation_plan.md) | [C3b 算子检查点](deepseek_cache_metadata/checkpoint.md) | 端到端验收见 C3 集成记录（Git `934485b:docs/agents/system/deepseek_motivation_c3_integration.md`） |
| DeepSeek indexer causal tail | [任务](deepseek_indexer_tail/task.md)、[计划](deepseek_indexer_tail/implementation_plan.md) | [C4a 算子检查点](deepseek_indexer_tail/checkpoint.md) | [候选草案](deepseek_indexer_tail/draft.md) |
| DeepSeek official top-k wrapper | [任务](deepseek_topk_wrapper/task.md)、[计划](deepseek_topk_wrapper/implementation_plan.md) | [C5 算子检查点](deepseek_topk_wrapper/checkpoint.md) | 完整请求见 C6 验收（Git `934485b:docs/agents/system/deepseek_motivation_c6_integration.md`） |
| DeepSeek sparse recall metadata | [任务](deepseek_sparse_recall/task.md)、[计划](deepseek_sparse_recall/implementation_plan.md) | [C6 算子检查点](deepseek_sparse_recall/checkpoint.md) | [失败顺序审查](deepseek_sparse_recall/lifecycle_review.md)；完整请求见 C6 验收 |
| DeepSeek dense prefetch | [任务](deepseek_dense_prefetch/task.md)、[计划](deepseek_dense_prefetch/implementation_plan.md) | [C7a 组件检查点](deepseek_dense_prefetch/checkpoint.md)，已通过组合数值验收 | [限制 gather CTA 数的候选计划](deepseek_dense_prefetch/c7b_implementation_plan.md) |
| DeepSeek dense MLP packing | [任务](deepseek_mlp_packing/task.md)、[计划](deepseek_mlp_packing/implementation_plan.md) | 组件 eager / Graph 测量与生产正确性验收已通过；接入后的 serving 性能未测量 | [接入计划](deepseek_mlp_packing/integration_plan.md)、[源码与验证记录](deepseek_mlp_packing/checkpoint.md) |
| DeepSeek linear activation quantization | [任务](deepseek_linear_quantization/task.md)、[计划](deepseek_linear_quantization/implementation_plan.md) | 前两版数值通过但 Graph 性能回退；整数归约候选已通过离线检查，待 GPU 验收 | [检查点](deepseek_linear_quantization/checkpoint.md)、[整数归约计划](deepseek_linear_quantization/redux_candidate_plan.md) |
| DeepSeek ECHO prefetch hint | [任务](deepseek_prefetch_hint/task.md)、[计划](deepseek_prefetch_hint/implementation_plan.md) | [精确阈值更新检查点](deepseek_prefetch_hint/checkpoint.md) | 组合验收（Git `934485b:docs/agents/system/deepseek_motivation_c7_hint_validation.md`），正式测量进行中 |
| DeepSeek norm I/O | [任务](deepseek_norm_io/task.md)、[计划](deepseek_norm_io/implementation_plan.md) | [原型检查点](deepseek_norm_io/checkpoint.md)，尚未合入生产实现 | [候选草案](deepseek_norm_io/draft.md) |

组件检查点、任务、执行约束和技术记录可使用英文。历史记录中
`current`、`active`、`latest` 指原记录时点，不能据此判断当前状态；以各组件检查点为准。
`/tmp` 路径是原证据位置，不保证临时文件仍然存在，也不作为仓库文档链接。
