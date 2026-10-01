# 项目文档

本项目研究生成式推荐中的 sparse KV on-demand loading：复用固定 user history，
将历史 KV 保存在 host，按本次 candidate 的稀疏选择加载，并让加载与 attention
重叠。当前已有 NOSA 算子原型和局部性能证据，跨请求复用与有限 HBM serving 仍待实现。

| 文档 | 内容 |
| --- | --- |
| [研究思路](research.md) | 场景、问题、async sparse KV fetching attention 与待验证假设 |
| [系统状态](status.md) | 当前实现、已验证范围、实验版本和主要缺口 |
| [后续计划](roadmap.md) | 从算子原型推进到多用户 serving 的步骤和验收条件 |
| [KDA 组件文档](kda/README.md) | 按组件整理的任务、实现计划、检查点和优化记录 |
| [实验入口](../experiments/README.md) | 每个实验的命令、run ID、测量边界和正式报告 |
| [DeepSeek / SM120 历史资料](../experiments/legacy/deepseek_v32/README.md) | 已移除执行路径的有效历史报告与复现说明 |

中文概览面向项目研究与进度讨论，KDA 中的英文任务和计划供 agent 执行时参考。
目录职责与维护规则统一见 [AGENTS.md](../AGENTS.md)，运行和环境准备见
[项目 README](../README.md)。
