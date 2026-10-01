# 项目文档导航与维护分工

当前研究从 sparse KV fetching 与 sparse attention 计算的重叠设计出发，探索
sparse attention GR serving 场景；存储范围先限定为 HBM 与 CPU DRAM。
场景、问题和贡献仍可修改，已有实现不代表这些研究判断已经成立。

## 先读哪份文档

| 文档 | 回答什么 | 主要维护者 |
|---|---|---|
| [research.md](research.md) | 如何按“任务与场景 → Baseline 与 Motivation → Challenge 与 Design → Evaluation”讲清研究 | Supervisor 根据研究者修正维护；需要写作时按需调用写作技能 |
| [status.md](status.md) | 四个环节分别知道什么、还缺什么、具体卡在哪里 | Supervisor 维护，人可以直接修改；这是当前研究判断与 checklist 的唯一状态表 |
| [roadmap.md](roadmap.md) | 下一步做什么、对应哪些研究条目、为何优先 | Supervisor 提出并更新任务，研究者调整方向与优先级 |
| [实验索引](../experiments/README.md)及各实验报告 | 实际做了什么实验，结果支持什么结论 | 实验分析执行者维护；Supervisor 引用结果并更新其研究含义 |
| 各模块 README，例如 [模型](../models/README.md)、[算子](../operators/README.md)、[服务入口](../serving/README.md) | 当前模块能力、使用入口及必要说明 | KDA / 编码执行者随工程变化维护 |

准备组会或论文时从 `research.md` 提炼叙事，研究讨论时从 `status.md` 定位问题，
安排工作时使用 `roadmap.md`。三者分别维护叙事、判断和行动，不重复保存完整实验报告。
已测结果以相应实验报告为依据；模块能力以代码和模块说明为依据。

## Agent 的工作材料

`docs/` 下除 `agents/` 外的文档面向人阅读。任务契约、实现计划、检查细节和内部回查
放在 `docs/agents/`，通过链接按需查看：

- Supervisor 的[材料依据](agents/research-supervisor/sources.md)与
  [重要理解修正](agents/research-supervisor/updates.md)。它们帮助回查，不另建一套研究状态。
- [KDA 组件文档](agents/kda/README.md)：按 resident indexer、resident sparse attention
  和 offload attention 分别维护任务、实现计划、检查点及候选调查。
- 系统编码与实验执行者维护[工程背景](agents/system/research-context.md)、
  [实现状态](agents/system/implementation-status.md)和
  [候选实现方案](agents/system/implementation-roadmap.md)。它们保留具体能力和工程约束，
  不另定研究方向；候选方案的取舍回到研究状态与任务中讨论。

具体实验报告和素材继续留在 `experiments/<experiment>/`，模块说明继续留在模块内，
无需为这次分层复制到 `docs/`。项目约束见 [AGENTS.md](../AGENTS.md)；
[Supervisor 技能](../skills/research-supervisor/SKILL.md)在项目内持续更新，直接读取使用。

## 怎样分工并回到研究主线

Supervisor 负责研究理解、缺口识别和任务组织。KDA / 编码执行者负责实现，实验分析执行者
负责测量与解释结果；文献核验、场景调研和写作按需交给相应 agent 或 skill。
这些是建议的职责，不表示项目已经部署了所有角色。

每项工作在 `roadmap.md` 中关联 `status.md` 的研究条目，说明它准备回答什么问题。
执行者将详细产物写入所属目录，交回“得到什么发现、影响哪个条目、还有什么不确定”。
Supervisor 据此更新研究判断、四环节叙事和后续任务；研究者的修正进入下一轮理解。

例如，场景与数据探索主要推进第一环，也影响容量动机和整体评测；baseline 调研与分析
推进第二环，并帮助凝练第三环的挑战；实现和机制实验推进第三、四环。
探索无论得到支持、反例还是排除一种方案，都应说明改变了哪项研究理解。
四环节是呈现与核对框架，工作可以交叉推进。
