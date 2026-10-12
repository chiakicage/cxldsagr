# 下一步任务

更新：2026-10-12。GR 与正常 agentic serving 均为候选；Prefill/Extend 与 Decode 分开，
当前优先 Q1 Decode 和 Batch=128 Decode。无 offload 单层观测已分列 GEMM、non-GEMM
与完整 API，并补齐独立实际 HBM 指标；完整 Decode 对照及管理/重叠设计仍待开展。
以下只列待做任务；既有 A128 Extend 不作为 Batch=128 Decode，原 motivation、
4K/16K、容量填满与后续审阅、simulation 仍暂缓。

| 任务 | 下一步 | 主研究条目 |
|---|---|---|
| T-009 Decode Cache Manager | 分别量化 Q1 与 Batch=128 的管理及其引入的开销，探索减少逐层决策的方式，以完整单步净收益检验 | 2.6（关联 3.4、4.4） |
| T-010 Indexer Cache offload/overlap | 检验 Indexer Cache 常驻对 batch 的限制，设计 Indexer 计算与 Indexer Cache 读取重叠，量化容量、新增 IO 和效率 | 2.7（关联 3.5、4.5） |
| T-011 Main Cache overlap | 设计 Indexer、Attention 与 Main Cache 读取重叠，将剩余 Recall 纳入流水；与管理策略分开消融 | 2.8（关联 3.6、4.5） |
| T-002 场景与模型数据 | 明确 GR、agentic serving 的具体任务及模型/数据适配，分别定义 Prefill/Extend、Q1 Decode 和 Batch=128 Decode | 1.1（关联 1.2–1.4） |
| T-003 Decode 可比范围 | 以拆分后的驻留算子观测和独立 HBM 指标为依据，明确两档 Decode 的 batch、历史、驻留、输出和模型适配，建立共同完整单步 baseline 范围 | 2.1（关联 1.4、4.3） |
