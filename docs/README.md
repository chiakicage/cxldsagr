# 项目文档导航与维护分工

## 人阅读的入口

| 文档 | 内容 | 维护者 |
|---|---|---|
| [系统架构](architecture.md) | 模块职责、请求与缓存生命周期、模型接入 | 编码执行者 |
| [status.md](status.md) | 实验室四环节的当前理解、已有依据和具体缺口，可直接修改 | Supervisor 与研究者 |
| [roadmap.md](roadmap.md) | 简短的下一步任务及对应研究条目；完成或取消后移出 | Supervisor 与研究者 |
| [实验索引](../experiments/README.md)及各实验报告 | 实验方法、结果和结论 | 实验分析执行者 |
| 各模块 README，如[模型](../models/README.md)、[算子](../operators/README.md)、[服务入口](../serving/README.md) | 模块能力和使用说明 | KDA / 编码执行者 |

论文与组会的叙事组织由其他 agent 后续完成，Supervisor 只提供研究状态、缺口与依据。
项目如何探索、为何从某个设计开始，不能充当呈现给读者的任务或问题定义。
实验室原始流程保存在技能的 [Research.md](../skills/research-supervisor/references/Research.md)，继续作为四环节核对依据。

## Agent 的工作材料

`docs/` 下除 `agents/` 外的文档面向人阅读。详细任务、实现计划、验收和内部记录放在 `docs/agents/`：

- Supervisor：[材料依据](agents/research-supervisor/sources.md)、[重要理解修正](agents/research-supervisor/updates.md)。
- KDA：[组件文档](agents/kda/README.md)，分别维护 indexer、sparse attention 和 offload attention 的任务与检查点。
- 系统编码与实验执行者：[在办工程任务](agents/system/README.md)；已完成重构的接口见各模块 README，结果见对应实验报告。
- 独立正确性验收：[冻结源码与证据索引](agents/acceptance/unified_runtime_20261005/evidence.json)，原始收据保留各自的源码和验证边界。

完整入口见 [agent 文档导航](agents/README.md)。工程方案只保留当前任务所需材料，
验收记录按原始身份独立保留，不另维护一份与研究状态表并行的“当前实现状态”。

具体实验报告和素材留在 `experiments/<experiment>/`，模块说明留在模块内。
职责与目录约束见 [AGENTS.md](../AGENTS.md)；[Supervisor 技能](../skills/research-supervisor/SKILL.md)在项目内更新，直接读取使用。

## 工作怎样回到研究状态

每项研究工作关联状态表中的具体条目。执行者维护详细材料，返回发现、受影响的条目和仍然未知的内容；
Supervisor 更新研究判断和下一步任务。完成的工作从 roadmap 移除，必要历史留在内部记录中。
研究发现可以支持、修正或否定原判断；四环节可交叉推进。
