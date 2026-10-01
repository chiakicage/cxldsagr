# 初始研究状态的材料依据

更新：2026-10-01。本轮读取用户输入和仓库现有说明，未重新测量、重建报告或核验所有论文。
下面的性能和实现状态是**已有材料记载**。它们用于界定初始判断，不能代替未来任务所需的实际核验。
本文件供 agent 回查；给人的[研究状态](../../status.md)和[下一步任务](../../roadmap.md)在 `docs/`。

## S-001：研究者当前输入与实验室流程

- **来源：**本任务中研究者关于呈现分层、非线性推进、可编辑状态表、设计先行和 CPU DRAM 范围的说明；
  [实验室 Research.md 副本](../../../skills/research-supervisor/references/Research.md)。
- **对应：**四环节结构及全部当前候选解释。
- **支持：**用户希望以已有 sparse KV fetching/attention overlap 设计寻找场景；GR serving 仍待完善；
  ECHO 是重点对比；项目先关注 HBM/CPU DRAM。
- **边界：**“容量限制—offload 等待—dense 冗余—sparse 重叠困难”是用户提出的研究线索，
  不因写入状态表而成为测量结论。先构造哪种模型/数据路线尚未由用户决定。

## S-002：GR 请求构造工具

- **来源：**[GR/README.md](../../../GR/README.md)，重点是开头、请求内容、热度/到达和适用边界说明。
- **对应：**1.2、1.3、1.4、4.3。
- **支持：**有固定历史、候选变化、长度预算、热度和模拟到达时间的生成工具。
- **边界：**默认 synthetic，catalog 模式也使用合成行为与候选；没有真实下一商品标签。
  生成工具不能证明真实推荐任务、质量或线上负载代表性。

## S-003：本地 serving 的能力

- **来源：**[serving/README.md](../../../serving/README.md)，当前范围及请求生命周期。
- **对应：**1.4、1.5、2.2、4.1。
- **支持：**有单 GPU 串行请求执行入口，NOSA 可使用 resident/offload。
- **边界：**每条请求独立 cache；没有跨请求 KV 复用、batch scheduling 或到达时间回放。
  生成的 QPS/时间戳不等于已运行的 serving 吞吐，不能据此证明容量限制服务效率。

## S-004：NOSA GR 长上下文工程实验

- **来源：**[nosa_gr_65536_1024/README.md](../../../experiments/nosa_gr_65536_1024/README.md)，
  精确输入与执行边界、结果及模块耗时。
- **对应：**1.2、1.4、2.2、4.3。
- **支持：**已有 32 层 dense forward，prefix 65,536、suffix 1,024 的输入和模块分析基础。
- **边界：**该实验不执行 sparse selection、offload、LM head 或 decode；实验进程扩大了默认
  32,768-token context，未评价长上下文任务质量。attention 计算占比较大不能证明 KV 容量动机。

## S-005：NOSA sparse fetch/attention overlap

- **来源：**[nosa_offload_overlap/README.md](../../../experiments/nosa_offload_overlap/README.md)，
  实验目的、完整调用性能、完整模型数值检查和范围限制；[项目 README](../../../README.md)。
- **对应：**3.2、4.1、4.2、4.3。
- **支持：**在 64K+1K 固定单层输入回放中，L0/L15/L31 相对整批 sparse union 一次 fetch 后
  attention 的串行对照，完整调用中位延迟下降 23.80% / 23.30% / 19.87%；已有独立复测。
  完整模型 resident/offload 数值检查另有记录。
- **边界：**性能属于固定输入单层回放，不是完整模型或 serving 性能。NOSA HBM staging 仍
  覆盖一层完整逻辑地址，尚无有限 slots/eviction；cache 分配不等于进程峰值显存。
  本轮未复核原始运行，也未将数值测试作为性能证据。

## S-006：ECHO 本地实现和报告

- **来源：**[deepseek_v32_echo_prefill/README.md](../../../experiments/deepseek_v32_echo_prefill/README.md)，
  报告状态、测量边界和已验收结果。
- **对应：**2.1、2.3、2.5、3.3、4.1。
- **支持：**有 DeepSeek V3.2 61 层的 ECHO 相关实现，indexer 内融合 prefetch，CPU DRAM
  backing、有限 HBM pool 与 residual recall。旧报告显示降低主 KV 的 HBM 分配，同时增加单请求延迟。
- **边界：**2026-10-01 gather 对齐修复后完整模型性能待补测；旧报告只对应其版本与输入。
  该路径为单请求执行器，不代表 continuous batching serving。它与 NOSA 单层回放的模型、
  执行范围和硬件安排不同，不能直接形成设计优劣排名，也不能用它断言 ECHO 在所有场景都慢。

## S-007：并行整理的系统工程与 KDA 材料

- **来源：**提交 `1971047` 的材料，现整理为[工程背景](../system/research-context.md)、
  [实现状态](../system/implementation-status.md)、[候选实现方案](../system/implementation-roadmap.md)
  与 [KDA 组件文档](../kda/README.md)。
- **对应：**1.1、1.3、1.4、2.2、2.5、3.1、3.2、4.1–4.3；T-002、T-004、T-005。
- **支持：**补充固定 history / 变化 candidate 的具体候选负载、生成器与 serving 的区别，
  保留跨请求复用、有限 HBM、派生状态和请求流评测的工程约束；KDA 按组件维护。
- **边界：**原文中的确定场景和开发顺序不覆盖研究者尚未确定场景、非线性探索的要求。
  NOSA 完整地址 staging 与 DeepSeek 有限 pool 不混用；完整数值验收不证明服务性能或任务质量。
  Dense 与完整 NOSA 的差别含 CIS，不能把差异都归因于 offload；完整 query batch 的强串行
  对照及拆 query 时的同拆分对照继续适用。旧 run 不代表目录迁移或实现更新后的新测量。

## 初次覆盖范围

仅核对上述代表性材料，未声称穷尽项目实验或 SOTA。初始状态的已有支持限于对应来源允许的范围；
下一次研究任务涉及具体结论时再核对相应材料和当前版本。研究任务与条目映射见[研究路线](../../roadmap.md)。
