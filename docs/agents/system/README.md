# 当前系统工程任务

| 任务 | 入口与范围 |
| --- | --- |
| T-007：NOSA H64K indexer workspace / dispatch | [待实施计划](nosa/nosa_indexer_dispatch_plan.md)；直接依据为[元数据复用结果](nosa/nosa_copy_descriptor_result.md)、[匹配拷贝对照](nosa/nosa_guarded_copy_control.md)和[预提交诊断](nosa/nosa_hbm_prequeue_diagnostic.md) |

当前架构与接口见[执行器](../../../executor/README.md)、[缓存](../../../cache/README.md)、
[serving](../../../serving/README.md)及模型 README；性能结果见[实验索引](../../../experiments/README.md)。
独立正确性验收与冻结对照的最小证据见
[验收索引](../acceptance/unified_runtime_20261005/evidence.json)，其中旧验收只适用于原源码。

已被替代的计划、检查点、交接和 patch 已删除。原文件可从 Git `934485b` 的
`docs/agents/system/<原文件名>` 回查；不以历史计划启动已取消或延期的任务。
统一框架重构的过程材料已删除，接口、测量和独立验收分别保留在上述入口。
目录维护规则见[根 AGENTS](../../../AGENTS.md)。

研究判断和任务取舍见[状态表](../../status.md)与[研究待办](../../roadmap.md)；
算子工作见 [KDA](../kda/README.md)。
