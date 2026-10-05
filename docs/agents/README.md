# Agent 工作材料

本目录保存任务计划、实现检查点、验收依据和必要的历史修正。全仓开发约定从
[根 AGENTS](../../AGENTS.md)读取，再按任务进入对应模型、算子、实验和依赖规则。

| 入口 | 内容 |
|---|---|
| [系统工程](system/README.md) | T-007 的必要计划与依据；已完成重构的过程材料已删除 |
| [独立验收](acceptance/unified_runtime_20261005/evidence.json) | 冻结源码、原始验收收据与证据索引；范围以各记录为准 |
| [KDA 组件](kda/README.md) | 按算子组件保存任务、候选调查和验收检查点 |
| [Supervisor 来源](research-supervisor/sources.md)、[理解修正](research-supervisor/updates.md) | 研究判断的内部依据与研究者修正 |

当前研究判断和待办分别维护在 [status](../status.md) 与 [roadmap](../roadmap.md)；
模型能力查模块 README，性能结果查[实验索引](../../experiments/README.md)。
历史材料中的“当前”“进行中”按记录日期理解，不自动成为新的执行任务。

系统材料只保留在办任务，完成或被重构替代后按根 AGENTS 删除；算子材料按组件定位。
仍需保留的阶段依据须有可核验的源码、run ID、命令和测量边界，不改写冻结证据。
