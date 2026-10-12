# cxldsagr 研究状态

> **2026-10-11 研究范围与优先级调整：**除 GR 外，将正常 agentic serving 纳入候选场景。
> Prefill/Extend 与 Decode 分开考虑，当前优先 Q1 Decode 和 Batch=128 Decode。
> 从 Cache Manager 与 Overlap 两个角度分析 ECHO；Cache Manager 引入的开销也归入管理成本。
> 当前关注管理开销仍大、Indexer Cache 未 offload 对 batch 与重叠效率的限制，以及剩余
> Recall 未纳入重叠。对应设计是减少逐层决策、重叠 Indexer 计算与 Indexer Cache 读取，
> 并让 Indexer、Attention 与 Main Cache 读取重叠。这些是研究者本轮提出的问题与设计方向，
> 因果占比和设计收益仍待验证；已有 A128 Extend 不作为 Batch=128 Decode 结果。依据见 S-040。

> **2026-10-12 无 offload 算子拆分与 HBM 补测完成：**DeepSeek Decode 覆盖 Q=1、
> B=1/32/64/128/256/512/1024，Extend 覆盖 B=1、Q=32/64/128/256/512/1024，
> 两类均测 H=4K/16K/32K/64K；公共算子与 MoE 按 S=BQ 分组测量。
> 新 run `observation_20261012_01` 完成 66 个负载、2,616 项独立观测，分列纯 GEMM、
> non-GEMM 和完整 API 聚合；Indexer 与 Attention 内核保留融合边界。Clean 时间/MFU
> 与独立 NCU 实际 HBM 指标分别报告，替换此前混合 API 观测。Q1 与 Batch=128
> 已有局部算子依据，完整 Decode 对照仍待建立；该观测不执行 cache 事务或 offload，
> 不验证 Cache Manager 成本或 Overlap 收益，也不能相加为模型或 serving 延迟。依据见 S-041。

> **2026-10-09 Q1 准备融合已完成接入与补测：**新批次的 HBM 三层 GPU 窗口为
> 1.014466 ms，官方参考为 1.014975 ms；输入、驻留与框架仍不同，不能据此声称
> 等价实现加速。ECHO 的三层准备节点由 9 个减到 3 个，完整同步 step 中位数为
> 2.562300 ms，仍慢于 serial sparse 的 2.401064 ms。优化收益来自私有 500 对
> 对照的约 1.32% 中位数下降，部分均值与 p99 变差；不以两轮正式中位数相减估计收益。
> 同输入官方 core 的冷态／完全驻留 NCU 对照仍说明 miss 路径有长尾，尚不能拆分
> 预约和搬运的贡献。T-008 本轮接入与独立补测已完成；C10、NOSA、GR 场景及
> serving 判断不变。依据为 S-038、S-039。

> **2026-10-08 DeepSeek Q1 效率复测：**按用户要求，继续优化真实前三层 H64K+A1
> 的 local HBM 与 ECHO 路径。四方法改为各自独立进程准备和测量后，HBM 的
> L0–L2 GPU 窗口为 1.013090 ms，与 SGLang 参考的 1.014975 ms 接近。
> 此前 dense 同进程预热会影响随后 HBM 的节点间隙；这是测量流程问题，不能据此
> 将原差距全部归因于模型计算。ECHO 的完整同步 step 仍为 2.588716 ms，慢于
> serial sparse 的 2.395402 ms。准备阶段融合候选尚未获得可发布的性能结论。
> 两侧输入、驻留和框架不同，官方跨实现数值验收仍未通过；本次不改变 GR 场景、
> C10 矩阵或 NOSA 的判断。依据见 S-037；仅恢复用户指定的 Q1 优化范围。

> **2026-10-07 DeepSeek motivation 矩阵已发布：**用户指定的
> H=[4K,16K,64K] × A=[128,256,512,1024] 共 12 点已完成独立验收与正式计时。
> 实际沿用 P=65,536、NH=16,777,216、U=16、R=2、history chunk=1,024、seed=42；
> 模型为 C10，使用 v4 纯计算图。H4K 的 HBM 历史复访全部命中，四档 A 的复访均值
> 均为 HBM 最低；H16K 均为 dense 最低；H64K 的 A128/256 为 serial sparse 最低，
> A512/1024 则为 dense 略低。ECHO 在全部 12 点均慢于 serial sparse。
> 历史保留收益和 candidate 成本随 H/A 改变，不能把原 H64K/A128 排名推广到所有配置。
> 每个配置只测一条正式轨迹，每方案含两轮用户访问，不证明稳定排名或真实 GR 场景质量。
> T-006 已完成，依据见 S-034。NOSA motivation、容量填满与后续审阅、simulation
> 仍暂缓，广泛优化未恢复。
> 旧单点数值与计时依据保留；撤下 timeline 展示不使其局部 profile 证据失效，
> 也不把旧 profile 改称新矩阵的机制验收。

> **2026-10-07 矩阵前的 DeepSeek 单点重跑：**按用户“只跑 DeepSeek V3.2”的选择，
> C10 当前 DMA 与 ECHO 修正路径已完成独立数值验收、四方案正式计时和匹配 profile 审查。
> HBM、ECHO、serial sparse、dense prefetch 的复访请求均值分别为
> 2163.789、21.133、15.865、19.798 ms；历史命中分别为 0/16、16/16、16/16、16/16。
> Serial sparse 的复访最快，dense 的完整 32 请求总耗时最低；这是一轮轨迹，
> 不提供重复运行置信区间。C10 只捕获部分计算，不能套用真实三层完整图的执行边界。
> 依据见 S-033；该次重跑只覆盖一个配置，后续 H/A 矩阵及其独立来源见上方说明。

> **2026-10-06 DeepSeek 完整 extend 图：**[真实三层 A128 MFU](../experiments/deepseek_v32_mfu/README.md)
> 已完成用户指定的 dense 等待位置调整，并补齐四方法独立计时和逐算子 MFU。
> HBM、ECHO、serial sparse、dense prefetch 的 extend 分别为
> 3.326、5.716、4.265、5.706 ms。Dense 的原生预取重叠已验证，完整延迟与 ECHO
> 接近，仍高于 serial sparse；局部重叠不能单独决定完整延迟。
> 广泛优化仍已停止，本轮只完成这项指定调整及补测，不追加 gap 优化任务。
> [C10 motivation](../experiments/deepseek_v32_motivation/README.md)已按上方用户选择完成单点及矩阵重跑，
> 容量后续审阅及其他 motivation 实验继续暂缓。直接依赖已撤回旧 MFU
> 输入的单层 simulation 已撤回，没有用新图数字替换旧模拟。
> 三层普通追加与 C10 分别验收；本次结果不证明完整 serving 收益。
> 旧本地官方 ECHO 适配及其结果已删除；[独立 SGLang 复现](../experiments/deepseek_v32_echo_official/README.md)
> 保留真实前三层的 performance-only 计时。ECHO 与 HBM-only 均已完成，数值验收未通过；
> 两种配置的容量不同，不作为等容量对照。
> 三层完整图及 MFU 依据见 S-032，两条执行路径分别解释。

> **2026-10-05 NOSA 安排：**按用户继续推进目标的要求，恢复 H64K NOSA 性能优化。
> 目标仍是相同配置下的 HBM-only、dense prefetch、sync sparse、async sparse 对照，
> 验证 HBM candidate 效率、cache 正确性，并减少计算和 IO 之外的开销。
> 单次校验内元数据复用候选已完成测量与独立审计，两请求各有约 0.1 ms 的局部改善，
> 尚未证明四方案完整 serving 收益，暂不接入。
> 已有正式结果保持原实现边界，4K/16K 继续暂停。

> **2026-10-04 测量范围调整：**按研究者最新要求，后续性能测量先只做 64K history，
> 暂停追加 4K 和 16K。本轮这两档的正式测量和 profile 已在调整前完成并通过核验，
> 保留其有效结果；整理报告不意味着继续测量这两档。

> **2026-10-04 固定 P/NH 实测：** [deepseek_v32_motivation](../experiments/deepseek_v32_motivation/README.md)
> 已完成 P=65,536、NH=16,777,216、16 用户两轮的四方案对照。HBM-only 只能保留一个
> history，复访 0/16 命中；三个 offload 方案均为 16/16。复访均值依次为 HBM
> 2874.51 ms、ECHO 70.04 ms、sparse fetch 64.63 ms、dense prefetch 70.55 ms。
> ECHO 与 sparse fetch 的 candidate H2D 相同，dense 为其 9.688 倍；本轮未显示
> ECHO 的融合预取收益。dense/sparse 均复用 P 槽，candidate 仅在 GPU 临时执行。
> 该结果支持固定容量下的历史保留与搬运对照；总 HBM 字节数并不相同，也未建立真实
> GR 质量或场景代表性。96 条 offload 输出与 HBM 逐位一致。下方已撤回的旧预算、W
> 与 chunk 判断仍不作为证据；详细归因及 NOSA 的共同模型比较未由本次运行完成。

> **2026-10-03 用户修正：** 下文引用的 DeepSeek 4 GiB / W / chunk 对照及其默认
> 配置、容量和性能判断已撤回，相关实验产物已清理，不再作为当前有效证据。
> 受影响的 1.3–1.5、2.1–2.2、2.5、4.1、4.3 及对应卡点保留原文，待研究状态重新核对。
> [固定 P/NH 容量报告](../experiments/cache_management/README.md)现按 GPU 临时
> candidate 重新规划：NH 只保留 history，candidate 整批执行，用完丢弃，不写回 DRAM。
> 512 GiB DRAM 下的 NH 静态边界对应 455 个 64K history；P 还受这些用户的 resident
> indexer、共享 workspace 和 allocator 余量约束。新结果是静态估算，未跑满用户；短
> checkpoint 检查仅验证数值与存储语义。旧 candidate 写回 DRAM 的容量结果已替换，
> 不作为新实现的显存测量或性能排名。NOSA 与独立前三层计算实验不在此次撤回范围内。

更新：2026-10-12（GR 与 agentic serving 均为候选，优先 Decode，按 Cache Manager 与 Overlap 分析 ECHO；驻留算子已分列 GEMM、non-GEMM 与独立 HBM 指标）。
既有 GR 对照保留研究者选择的固定用户顺序 loop。两模型的统一框架重构结果
与同机 P0 分别比较，原已发布数字保留历史来源。NOSA 三轮完整 trace 的中位总耗时
比 P0 增加 406.132 ms（0.19664%）；准入、清理和部分 candidate／复访阶段仍有耗时
增加，不能声称重构没有性能代价。A128 的 96 个适用内部样本均未达到两种 90% 门槛；
A1024 冷并集单层回放的 9 个样本均通过，最低 ratio 为 0.91081246。
两者回答不同执行路径的问题。两模型使用相同 token 配额，但物理分配方式不同；
本轮没有跑满 NH，也不证明真实 GR 场景、任务质量或物理容量上限。
T-007 的 indexer 执行 workspace 复用仍是未实施的独立方案，重构没有实现或验证它。
此前元数据复用的局部改善与诊断边界继续保留，4K/16K 暂停。

当前候选场景包括 **sparse attention GR serving 与正常 agentic serving**，内存范围仍为
**HBM 与 CPU DRAM**。Prefill/Extend 与 Decode 分开考虑，当前优先 **Q1 Decode 和
Batch=128 Decode**。Q1 关注单序列单 token 步；Batch=128 关注 128 个序列各执行一个
decode token，不能用单序列 A128 Extend 或 128 次串行请求代替。

研究重点是分别识别 ECHO 的 **Cache Manager 开销与 Overlap 局限**，检验减少管理
决策、offload Indexer Cache 及扩展计算/读取重叠能否改善 Decode。具体 agentic 任务、
模型、数据和请求轨迹仍待明确。已有 GR 历史保留测量与真实前三层 Q1 局部结果各有
边界；Q1 与 Batch=128 已有无 offload 单层算子观测，完整 Decode 对照尚未建立，
**五类方案的共同模型比较仍未建立**。
固定相同的共享 cache 实现后，官方计算后端与非矩阵算子适配进一步改善了前三层计算效率；
算子 kernel 利用率与完整阶段的端到端利用率分开报告。计算优化与本次 cache/serving 结果分别解读。
旧短轨迹没有建立有效的用户规模与缓存容量压力对照；16K 的少量复访重建差异只能描述
原轨迹。既有 GR 对照保留固定 U 个用户、按同一顺序遍历 R 轮的选择，用户覆盖与复用距离据此解释。
两模型原控制点均为 U=16、R=2、H=65536、A=128、P=65536、NH=16777216；
DeepSeek 后续在相同 U/R 与 P/NH 下补齐了 12 点 H/A 矩阵，NOSA 的范围不随之扩大。
比较的是指定 token 配额，不是相同总 HBM/DRAM 字节预算。已撤回的 4 GiB / W / chunk
数据不再支持当前容量或 chunk 选择；其他规模与真实场景代表性仍未验证。

可以直接修改下面的表述、状态和卡点；下次 Supervisor 会据此更新相关判断和下一步任务。
四环节可以同时推进，已有设计不要求场景先定稿。

[下一步任务](roadmap.md) · [维护分工](README.md) ·
[项目内 Supervisor](../skills/research-supervisor/SKILL.md)

## 1. 任务、场景与研究问题

- [ ] 能清楚说明研究什么服务需求、采用什么研究对象，以及希望改善什么。

| 条目 / 要回答的问题 | 当前理解 | 状态与具体缺口 |
|---|---|---|
| **1.1 研究什么任务与场景？** | GR serving 与正常 agentic serving 均为候选；Prefill/Extend 与 Decode 分开考虑，当前优先 Q1 Decode 和 Batch=128 Decode。既有 GR 方案仍是固定 history、变化 candidate 的 Extend | **范围与优先级已选择，场景仍待具体化。** 需明确 agentic 任务、服务需求、上下文增长及模型适配；GR 代表性也尚未成立 |
| **1.2 使用什么模型？** | 既有 GR 对照使用完整 NOSA-8B 和约 8B 的 DeepSeek C10 checkpoint 工作负载替身；另有真实 DeepSeek 前三层 Q1 对照。后续开发优先 NOSA，Decode 的具体模型适配仍需明确 | **已有执行对象，新增范围未验收。** C10 未经训练，前三层不等于完整 61 层；NOSA 与 DeepSeek 输出范围不同，不能直接比较跨模型延迟或任务质量 |
| **1.3 使用什么数据？** | 既有 GR 对照保留固定用户合成 loop、固定 history 与变化 candidate，不要求热度；agentic serving 的输入、生成轨迹和上下文分布另行确定 | **已有 GR 受控矩阵，Decode 数据待明确。** DeepSeek 的 16 用户、12 点 H/A 矩阵保留原范围；固定 token 的 Q1 局部对照不能证明真实 agentic 负载代表性 |
| **1.4 怎样形成 serving 负载？** | 当前分别考察单序列单 token 的 Q1 Decode，以及 128 个序列各一个 token 的 Batch=128 Decode。Prefill/Extend 单列；既有 GR 仍按同一用户顺序重复 R 轮 | **Q1 与 Batch=128 已有局部算子依据，完整 serving 负载待建立。** 52 点驻留矩阵使用独立合成 KV，不包含多步事务或请求调度；仍需明确各序列历史、驻留、步间追加与批处理方式。既有串行 loop 和 A128 Extend 均不覆盖 Batch=128 Decode |
| **1.5 主要优化目标是什么？** | 在相同 HBM/DRAM 硬预算下检验 Decode 的 Cache Manager 开销与计算/读取重叠；拟分别报告 Q1 单步延迟、Batch=128 单步延迟与吞吐，以及对应内存占用 | **目标方向已明确，量化要求待定。** 既有 GR 仍分首访、复访命中与重建；固定 P/NH 配额不等于相同总字节预算。Decode 的延迟/吞吐目标和真实到达过程尚未确定 |

既有 GR loop 的请求与统计定义保留：总请求 UR，首访 U，复访 U(R−1)，相邻两次
访问同一用户之间有 U−1 个不同用户；淘汰后重建仍计为复访。首访比例 1/R 会影响
全请求均值。这些定义不直接用作 Decode 的 batch 或调度语义。

## 2. SOTA baseline 与 Motivation

- [ ] 覆盖主要相关方案，讲清它们的取舍，并量化目标场景中的关键问题。

当前从 **Cache Manager 与 Overlap** 两个角度寻找 ECHO 的问题。Cache Manager 包括
决策、分配/淘汰、映射与发布、缓存状态准备/维护，以及这些操作引入的调度、同步和额外
搬运开销；即使它们出现在 Recall 或融合 kernel 中，也按管理原因归类。Overlap 关注
必要的 Indexer Cache、Main Cache 读取与计算能否重叠，以及仍暴露多少取数等待。
耗时归属与流水依赖分别分析，避免重复计入；尚未归因的 gap 保留为未知。

此前的“全 HBM 容量约束 → DRAM 取数等待 → dense 冗余 → sparse 重叠困难”仍是
待验证的线索。已有 Extend 结果可提供局部依据，Decode 的三个重点问题分别列在 2.6–2.8。

| 条目 / 要回答的问题 | 当前理解 | 状态与具体缺口 |
|---|---|---|
| **2.1 主要 baseline 是否覆盖？** | 候选集合仍为 HBM-only、ECHO、full prefetch、sync sparse loading、async sparse loading；当前优先明确 Q1 与 Batch=128 Decode 中 ECHO 的可比路径 | **已有驻留算子基线，共同 Decode 对照未建立。** 两模型的 Extend 对照和 DeepSeek 前三层 Q1 保留原范围；新增 Q1、Batch=128 等形状的无 offload 算子观测不能补齐五类完整 serving baseline。ECHO 未接入同一 NOSA 对照，不拼接跨模型排名；SOTA 覆盖仍待补 |
| **2.2 全 HBM 的问题是什么？** | 固定复用距离下，history 配额影响复访是否重建；实际物理容量还受模型布局、workspace、allocator 和其他分配约束 | **指定配额下的历史保留差异已测。** DeepSeek 固定 P=65,536 时，H4K 的 HBM 保留全部 16 用户、复访 16/16 命中；H16K/H64K 分别只能保留 4/1 个用户，顺序 loop 的复访均为 0/16。三个 offload 方案始终 16/16。H4K 没有避免历史重建的收益，不能沿用 H64K 的解释；这些配额结果不证明物理容量上限，也不恢复已撤回的 C1024/C2048 结论 |
| **2.3 CPU DRAM offload 的问题是什么？** | NOSA 固定容量 loop 中，保留历史避免了重新构建，但引入了首访分配、写回及复访取数成本 | **重构后已测量。** 各轮先求 16 次复访均值，再取三轮中位数，HBM/dense/sync/async 分别为 2342.511/71.980/29.568/31.952 ms；三个 offload 的首访均慢于 HBM。容量保留收益与取数代价须分开，不能把完整差值归为串行等待 |
| **2.4 Dense prefetch 的问题是什么？** | 整层历史预取仍使用 sparse attention；搬运量较多不必然意味着请求更慢。DeepSeek 三层完整图与 C10 单点均有连续内存 DMA 和计算的局部重叠证据 | **C10 取舍随 H/A 改变。** H16K 四档 A 的 dense 复访均值最低；H64K 的 A128/256 为 serial sparse 最低，A512/1024 为 dense 略低。ECHO/sparse 的 H2D 非零时，dense payload 为它们的 1.832–9.688 倍。不能仅据字节数或旧单点局部重叠解释排序；流量是软件 payload，矩阵未新增 profile |
| **2.5 与 ECHO 如何比较？** | 当前优先在 Decode 中分别分析 Cache Manager 和 Overlap。ECHO 的 Prefill/Extend、官方正常 Q1 Decode 与本地冷态 Q1 路径保留各自语义 | **局部差距已观察，归因仍有缺口。** 本地前三层 cold Q1 的 ECHO 完整 step 仍慢于 serial sparse；C10 的 12 点 Extend 矩阵也均较慢。两类结果不互相代替，官方与本地输入、容量及驻留不同，不能据此推断 Batch=128 或完整 serving 收益 |
| **2.6 Cache Manager 的开销在哪里？** | 研究者判断 ECHO 的管理开销仍大；分析范围包含管理操作本身及其引入的准备、调度、同步和额外搬运成本 | **有局部诊断，Decode 总成本待量化。** 现有 manager 指标只覆盖 A128 Extend 的 top-k→MLA 区间，Q1 的准备还混合了输入打包与缓存准备。新增驻留 `indexer_prepare` 与 scheduler metadata 不包含 Cache Manager 事务，不能直接计作管理成本。仍需按原因拆分完整流程，不能将完整差值或混合准备整段归入管理成本 |
| **2.7 Indexer Cache 如何限制 batch 与重叠？** | ECHO 当前 Indexer Cache 常驻 HBM，未考虑其 offload；研究者提出这限制 Batch Size，进而使 Overlap 效率不佳 | **常驻是实现事实，因果链待验证。** 新增矩阵提供不同 batch 的单层 resident Indexer/MLA 依据，但单层 B=1024 能执行不证明完整 61 层可准入同样 batch。仍需量化完整模型的 indexer 占用、可准入 batch，以及 offload 后新增 IO 和计算/读取重叠的净影响 |
| **2.8 剩余 Recall 是否也能重叠？** | ECHO 已融合 Indexer 与部分 Main Cache 预取，但剩余 Recall 仍在精确 top-k 后、Attention 前完成；研究者希望将它也纳入重叠，减少暴露开销 | **流水局限已识别，收益待测。** 本地 cold Q1 每层预取 64 条，随后 recall 1,983–2,041 条；官方正常 Q1 的驻留与召回路径不同。需区分 Recall 内的管理成本、实际搬运和暴露等待 |

ECHO cache baseline 的官方逐层共享 pool、实际领取才驱逐、
`indexer/prefetch → top-k → 主 KV append → exact recall` 的顺序及时间戳 priority
已核对并修正原本地差异。旧实现与 chunk 选择计划（Git `934485b:docs/agents/system/echo_cache_implementation_plan.md`）
只保留历史回查用途，不恢复已撤回的预算、chunk 和性能判断。当前取舍以
[固定 P/NH motivation](../experiments/deepseek_v32_motivation/README.md)与本地 MFU
各自的报告为准。旧本地官方适配及其结果已删除；[官方 ECHO 实验](../experiments/deepseek_v32_echo_official/README.md)
保留独立 SGLang 真实前三层复现，新 Q1 源码桥接及本地适配另按 S-037 验收。
按用户要求继续测量性能，ECHO 与 HBM-only
均已完成 performance-only 计时；两种配置容量不同，不作为等容量对照，
不据此宣称数值验收通过。

2026-10-07 的[官方 Prefill/Extend recall 诊断](../experiments/deepseek_v32_echo_official/report/recall_diagnosis.md)
确认了独立 SGLang 的一项实现开销：召回每层扫描全部 host pool 的标记，零 miss 时
仍耗时约 1.53 ms。相同二进制的隔离实验与 NCU 将主要成本定位到扫描中的整数与
逻辑运算。因此图中长 recall 条不能作为纯取数等待或 ECHO 重叠能力不足的证据。
本地实现使用不同的召回路径，这一原因不能直接解释 C10 矩阵中的 ECHO 排名。
该轮只完成诊断，未修改实现；详细依据见 S-035。官方正常 Q1 Decode 使用另一条
召回路径，不能把上述扫描耗时作为 Decode 的瓶颈证据。

后续用户指定的 Q1 优化已有独立结果：本地 HBM 的 GPU 窗口接近官方参考，
ECHO 在本地 cold 条件下仍慢于 serial sparse。准备流程会影响节点间隙，
因此 GPU gap 不能未经控制就归为可移除的计算开销。官方 ECHO 与本地使用不同
输入和自然／cold 驻留；其融合 indexer 时长差不能单独解释为实现效率差异。
这些结果只对应真实三层单 token 路径，当前生产结果见 S-039；本轮 Decode 范围和
两类开销的划分见 [S-040](agents/research-supervisor/sources.md#s-040decode-优先级与-echo-问题划分)。

非 GR 计算对照使用真实 checkpoint 第 0–2 层，hidden/residual 依次传播，包含
embedding、三个 dense MLP、final norm 与末 token LM head。当前
[四方法 MFU](../experiments/deepseek_v32_mfu/README.md)采用 H64K+A128、history
chunk=1K、完整 extend chunk=128、每层 P=65,664、cold extend 和计算图。Offload
在 extend 前清除历史主 KV 的 HBM 驻留，保留 DRAM 与 resident indexer；HBM
保留主 KV。该普通持久追加路径与 C10 serving 替身分开，不据此推断完整 61 层、
GR 场景或推荐质量。Extend 的 GPU 主体使用一次完整图 replay，输入准备、事务与
同步提交仍计入独立墙钟。下表来自 dense 延后同步后的四方法独立测量。

| 方法 | Prefill 中位延迟 ms | Prefill 最终 MFU | Extend 中位延迟 ms | Extend 最终 MFU |
| --- | ---: | ---: | ---: | ---: |
| HBM | 644.938 | 44.90% | 3.326 | 20.30% |
| ECHO | 648.375 | 44.66% | 5.716 | 11.81% |
| serial sparse | 640.701 | 45.19% | 4.265 | 15.83% |
| dense prefetch | 645.800 | 44.84% | 5.706 | 11.83% |

最终 MFU 将相同 useful matrix 工作量按精度换算为理论计算时间，再除以独立同步
墙钟；逐算子 MFU 的分母是对应 API 的 kernel duration sum。两者都不是 Tensor
Core 活跃率。Prefill MFU 覆盖全部 64 个 chunk；主 timeline 只显示最后一个
chunk 的 L0–L2，extend 主图也取 L0–L2。Dense 的起点包括 L0 首个计算或预取中
更早的一项，终点仍是 L2 最后计算结束；另附含启动阶段的 extend 对比图。
图的裁剪不改变完整阶段 MFU 的分母。
逐算子分母保留其 API 内的量化和融合预取，非矩阵操作不填写 MFU。

本轮将 dense extend 的历史 KV 等待延至 indexer/top-k 后，原生 profile 已验证
预取与前置计算重叠；prefill 和其他方法的调度未变。新旧运行的 native 构建身份
也有变化，因此旧 dense 6.166 ms 与本轮 5.706 ms 不构成单变量调度对照。
旧 v3 MFU 和依赖其输入的 simulation 已撤回。Profile 与独立计时的差值不直接
等于可移除 CPU 成本，本次三层结果也不证明完整 serving 收益。新结果见
[四方法汇总](../experiments/deepseek_v32_mfu/report/full_extend_graph/mfu/summary.json)与
[S-032](agents/research-supervisor/sources.md#s-032deepseek-a128-四方法-mfu-与局部重叠)。

C10 的 12 点矩阵说明，历史保留收益和 candidate 成本需要随 H/A 分别解释。
H4K 时 HBM 的 16 次复访全部命中，ECHO 与 serial sparse 也不需要 candidate
历史 H2D；四档 A 均由 HBM 取得最低复访均值。H16K/H64K 时，HBM 复访全部重建，
三个 offload 方案均保留历史；其中 H16K 的四档 A 均由 dense 最低，H64K 则在
A128/256 由 serial sparse 最低、A512/1024 由 dense 略低。不能把避免 history
重建的差值归为 attention 或预取加速，也不能由搬运量单独推出请求延迟。

矩阵的独立数值、内存与计量审查已完成。每个配置只测一条正式轨迹，每方案含
两轮用户访问、共 32 请求，首访占一半；首访、复访和完整轨迹分别报告，不证明
重复运行的稳定排名或单项改动的因果收益。C10 使用 v4 纯计算图，cache 事务、
选择、召回与 IO 留在图外，
与真实三层的完整 extend 图分开。完整结果与验收边界见
[S-034](agents/research-supervisor/sources.md#s-034deepseek-motivation-ha-矩阵安排)。

矩阵没有新增 profile。此前 H64K/A128 单点的数值、计时与局部 DMA 重叠证据
仍按 [S-033](agents/research-supervisor/sources.md#s-033deepseek-c10-当前路径重跑)
保留，撤下旧 timeline 展示不把该 profile 扩展到其他 H/A 点。容量报告引用的
既有 DMA 内存来源独立保留；独立 SGLang 计时也不构成本次矩阵的验收依据。

## 3. Challenge 与 Design

- [ ] 能解释真实困难、设计与困难的对应关系，以及相对已有方法的技术区别。

| 条目 / 要回答的问题 | 当前理解 | 状态与具体缺口 |
|---|---|---|
| **3.1 Sparse fetch 为什么难以重叠？** | NOSA 既有 A128 Extend 的 96 个适用内部样本均未达到两种 90% 重叠门槛；无 host-copy 的样本不计为通过。Decode 还需分析 batch、Indexer Cache 供数和剩余 Recall 对流水的限制 | **Extend 局部证据保留，Decode 待分析。** 先前 NCU 诊断提示 host 数据依赖和 consumer K/V 就绪等待，同二进制对照确认对齐可消除高于逻辑 payload 的 12.5% sector 访问；这些结果不能分解当前请求延迟或替代 Batch=128 Decode，也未证明计算或 IO 主导 |
| **3.2 当前设计如何应对？** | NOSA 融合唯一历史读取与 sparse attention；固定 P/NH 专用路径使用有限逐层容量、逻辑位置直接映射和 session 身份标签，candidate 只在 GPU 临时执行 | **固定路径已实现，方法收益未成立。** 当前 async 在首访、复访和完整轨迹均慢于 sync。多个 Q128 候选未通过数值或重叠门槛，均未合入；通用 budget 共享 staging 仍不等于通用有限槽缓存或热点淘汰策略 |
| **3.3 技术区别和研究贡献是什么？** | 当前拟分别减少 Cache Manager 开销，并让 Indexer、Attention 的计算与相应 Cache 读取重叠 | **设计方向已明确，贡献仍待验证。** 与 ECHO 和其他方案的实质区别、适用范围及净收益尚未建立，最终方法与贡献表述未确定 |
| **3.4 怎样进一步减少管理开销？** | 探索无需逐层决策的管理方式，减少重复决策与由管理流程引入的开销 | **设计方向，尚未实现或验证。** 需说明不同层的选择与驻留如何支持统一决策，并保持完整稀疏选择、容量约束和事务语义；操作次数减少不等于端到端收益 |
| **3.5 怎样重叠 Indexer 计算与 Indexer Cache？** | 将 Indexer Cache offload 纳入设计，让 Indexer 计算与 Indexer Cache 的读取/搬入重叠，探索扩大 batch 后的流水效率 | **设计方向，尚未实现或验证。** 需确定供数粒度、缓冲容量和计算依赖；offload 是否扩大可行 batch、能否隐藏新增读取成本仍待测 |
| **3.6 怎样重叠 Main Cache 读取？** | 同时考虑用 Indexer 与 Attention 的计算重叠 Main Cache 读取，并将剩余 Recall 纳入流水 | **设计方向，尚未实现或验证。** 需明确精确选择就绪前后的可读范围、供数/消费依赖及缓冲复用；管理成本与重叠收益分别归因，再检验完整 Decode 的净收益 |

NOSA 固定 P/NH 路径已将历史保留与 candidate 临时执行分开。P 限制逐层历史 HBM
容量，NH 是按页对齐的 host history 准入额度，host backing 随 session 分配。
DeepSeek 则持有共享全局 host arena，并按 pinned allocator 档位预留存储；相同 P/NH
不意味着两模型分配了相同字节数。NOSA 的 NH 不表示已分配全部 backing，也不保证
填满后满足 512 GiB DRAM。NOSA dense 路径当前要求 H≤P
且 history/chunk 按 64 tokens 对齐。这是有明确限制的直接映射方案，不能称为通用
热点缓存或淘汰问题已经解决；通用 budget 路径的 backend 共享 staging 另有边界。

重构后的数值验收、正式计时、匹配 profile 与独立 API 参考分别保留 run ID、收据和
冻结源码，按实际执行路径核对身份，不能假定不同采集范围的完整源码清单相同。
最终证据见 [NOSA motivation](../experiments/nosa_motivation/README.md)与
[材料依据 S-031](agents/research-supervisor/sources.md#s-031统一框架补测与研究边界)。
正确性和测量完整性通过，不表示重叠或效率目标已经完成。两模型的静态规划、cache
账本与完整请求内存观测见[容量报告](../experiments/cache_management/README.md)。
逻辑 payload、cache 计费及预留、进程占用分别解释；计费包含 graph private reservation，
不等于 tensor payload 之和。Allocated、reserved 和设备已用量分开。本次 DeepSeek
账本更新与完整请求内存观测已发布，五项静态 P/NH 边界未变；固定 16 用户轨迹仍
不证明装满 NH 或离线最大容量。

## 4. Evaluation

- [ ] 用整体比较、机制分析和消融回答前面提出的研究问题，并明确适用范围。

| 条目 / 要回答的问题 | 当前理解 | 状态与具体缺口 |
|---|---|---|
| **4.1 整体系统收益是什么？** | 两模型分别比较完整 loop、首访和复访；DeepSeek H16K/H64K 的 offload 复访优势首先来自保留历史，H4K 全命中时没有这项重建差值。真实三层 MFU 单独检查 baseline 执行效率 | **容量、执行效率与方法收益分开。** NOSA 的 async 复访均值比 sync 慢 8.06%，重构后完整 trace 中位数比 P0 增加 406.132 ms；原两模型重构对照的 cleanup 仍有增加。C10 矩阵的最低复访均值随 H/A 在 HBM、dense 和 serial sparse 间变化；一轮结果不证明稳定排名，也不建立共同模型对照或场景质量 |
| **4.2 哪些设计带来收益？** | 独立矩阵/attention API、resident A1024、冷稀疏并集回放与 DeepSeek 局部 profile 分别提供局部证据；主机编排诊断不隔离可移除成本 | **收益依赖输入与范围。** NOSA A1024 融合 API 延迟下降 23.02%–29.47%，9 个样本双比率过 90%，但 A128 serving 的 96 个适用样本仍未过。真实三层与 C10 单点均有 dense DMA/计算重叠证据；新 C10 矩阵中 dense 在 6 点复访最低，但没有新增 profile 来解释这些点。不能沿用旧单点作跨配置机制归因。Q1 独立进程复测的 HBM 窗口接近官方参考，cold ECHO 完整 step 仍慢于 serial sparse。准备融合的私有 500 对中位数下降约 1.32%，尾部未一致改善；默认路径接入与独立补测已完成，三层准备由 9 个节点减到 3 个（S-038、S-039）；不据此推广 serving 收益或恢复广泛优化 |
| **4.3 结论适用于哪些条件？** | DeepSeek 已有 C10 的 12 点 H/A Extend 矩阵及真实前三层 Q1，另有包括 Batch=128 Decode 的 52 点无 offload 单层算子观测；NOSA 固定容量范围仍为 64K+128 Extend。Prefill/Extend 与 Decode 分别报告 | **驻留算子范围已扩展，完整 agentic serving 尚未覆盖。** 新矩阵使用合成 activation 与独立序列 KV，未运行上下文增长、多步追加或完整 61 层。既有 Q1 本地冷态与官方自然驻留，以及 C10、NOSA 的模型/输出、配额和统计边界保留，不外推场景代表性 |
| **4.4 如何检验 Cache Manager 设计？** | 在 Q1 与 Batch=128 Decode 下量化完整管理流程及其引入的开销，用匹配对照检验减少逐层决策的收益 | **待开展。** 已有 post-top-k 指标不覆盖全部管理成本；需同时保留完整单步延迟、实际搬运和未归因 gap，避免把管理次数或局部区间减少直接作为净收益 |
| **4.5 如何检验 Overlap 设计？** | 分别检验 Indexer Cache offload、Indexer 计算与 Indexer Cache 读取的重叠，以及 Indexer、Attention 计算与 Main Cache 读取的重叠；单独检查剩余 Recall | **待开展。** Q1 与 Batch=128 各自建立匹配串行对照，报告容量、实际流量、暴露等待和完整单步延迟/吞吐；管理策略与重叠策略分开消融，并测量组合后的净收益 |

DeepSeek 的[无 offload 算子观测](../experiments/deepseek_v32_observation/report/results.md)
已将纯 GEMM 与量化、norm、RoPE、cast、layout、routing 等 non-GEMM 分开测量，
完整 API 聚合另列。H64K 的 B128 Decode 中，融合 Indexer 的有效 MFU 为 24.98%，
独立 profile 的 HBM 利用率为 84.71%；融合 MLA core 对应为 49.70%/52.84%，
包含 ID 转换等准备的完整 MLA API MFU 为 44.50%。Q1/B1 的 Indexer、MLA core
MFU 分别为 7.03%/5.05%，不能与完整 Decode 单步效率混用。

S1024 的 dense MLP 完整 API MFU 为 50.10%；grouped MoE 的 gate/up/down 纯 GEMM
useful MFU 为 8.97%/8.99%/9.54%，独立 profile 的 HBM 利用率为
80.82%/79.79%/79.59%，完整 MoE MFU 为 6.62%。Useful FLOPs 只计实际 selected
routes，静态 packing 与实际 HBM 流量另计，不能仅凭低 useful MFU 判断官方
kernel 的算力效率。Clean MFU 与 HBM 利用率使用不同的独立测量分母；小操作可能
受到明显的 profile 扰动，warm-cache 的低 HBM 流量也可能来自 L2 命中。
该观测不执行 Cache Manager 事务、offload 或 Recall，也未检验计算/IO 重叠；
各 API 与聚合区间不能相加为完整模型延迟。来源与支持边界见
[S-041](agents/research-supervisor/sources.md#s-041deepseek-无-offload-算子观测)。

完整 resident A1024 对照中，native sparse 的 full/extend 墙钟中位数为
2374.539/36.208 ms，有效 MFU 为 50.38%/52.56%；dense 为 3506.118/81.807 ms，
MFU 为 62.61%/63.02%。两者的有效矩阵工作量不同，不能只按 MFU 判断谁更快。
独立 sparse profile 的 extend indexer/attention MFU 分别为 23.05%/38.03%，
尚未同时达到 40%。这些是 ordinary owned resident 路径的结果，不替代 A128 serving
效率验收，详见[统一 MFU 报告](../experiments/nosa_mfu/README.md)。

当前 pattern 捕获中，Dense QA-only64、Dense full NOSA64 和实际 sparse NOSA64
的全模型去重 K/V payload 分别为 788.34375/583.46875/492.28125 MiB。当前 QA64
选块、有效位与并集数组均与保留数组逐元素相同；这只支持该请求的复现，不建立通用
跨环境保证。Pattern 未测时间、物理流量或模型质量，见
[模式报告](../experiments/nosa_indexer_pattern_65536_1024/README.md)。

独立 GEMM/BMM 与 FA3 API 中位数之和是组合参照，不是实际整请求或理论性能上限。
完整请求与 candidate 使用各自的 FLOPs 和无 profiler 墙钟；诊断计时不替代正式延迟。
两者差值包含参考未计入的必要非矩阵工作、host 工作、执行相互影响及计时差异，不能
直接当作 CPU launch 开销或可消除的浪费。长历史主导的接近度不能证明 candidate
高效或计算/IO 主导。最终分子、分母和取样见[NOSA motivation](../experiments/nosa_motivation/README.md)。

Clean bench01 的 HBM 请求 0/16 均重建 history。完整请求墙钟为
2353.533/2344.728 ms，MFU 为 50.10%/50.28%，相对独立 FA3 与矩阵 API 组合
的 MFU 比率为 99.733%/100.111%。Candidate extend 为
16.097616/16.817585 ms，MFU 为 14.764%/14.131%；对应组合为
12.129760/12.201056 ms，MFU 比率为 75.351%/72.549%。Candidate 分母不含
admission、history 构建及单列的 runner cleanup，仍包含 backend 内部 discard
与 lease drain。完整请求接近组合参照不消除 candidate 的效率差距，也不证明
计算和 IO 已主导；数值、profile 与测量完整性通过，效率结论仍须解释。详见
[当前 API 对照](../experiments/nosa_motivation/report/final/api_comparison.json)。

以下 prequeue、图内拷贝、元数据复用、扫描和独立 indexer API 数字均来自重构前诊断，
只按各自冻结源码解释。它们支持 T-007 的候选动机，不是重构后性能或 workspace
复用收益的证据；历史比较不能代替新路径验收。

H64K HBM 请求 0/16 的独立进程诊断中，提前提交相同 candidate 计算，使校验前
event 区间中位数从 14.721/14.713 ms 降至 12.918/12.905 ms，14 组配对均缩短。
80 份输出精确一致，两项预定的 1.05 倍中位数检查均通过。该干预支持测试减少
图输入拷贝与提交的具体实现；它没有隔离 CPU 成本，也不测普通 serving 收益。
Event 区间不含有限值决策、事务完成和 lease 归还；两个 repeat-5 样本在校验入口
之后仍比相邻同组样本多约 21–22 ms，其中一个来自 prequeue 组。扫描计数只标出
所在 candidate，未测扫描时长，不能据此归因。独立重建的 history 与额外检查副本
也使条件不同于正式用户轨迹。详见[预提交诊断](agents/system/nosa/nosa_hbm_prequeue_diagnostic.md)。

后续 H64K 外部原型把层间拷贝的提交移入已有计算图，保留 GPU 拷贝。请求 0/16
的普通完整 candidate 中位数分别从 15.416903/15.330098 ms 变为
15.258893/15.332270 ms，40 份保存的输出经独立重开均精确一致。该原型未取得
一致的中位数改善，暂不接入生产。基线长尾显著影响均值；扫描计数没有提供时长，
独立进程的 GC 与记录条件也不同，不能据此声称消除长尾或获得 serving 净收益。
该诊断保留原冻结结果，详见图内拷贝诊断（Git `934485b:docs/agents/system/nosa_captured_copy_pilot.md`）。

后续匹配对照共用运行检查和初始化，分别在图外、图内提交同样的拷贝。图内模式
在两个请求中的 candidate 中位数分别少 0.257480/0.038759 ms；相对原始实现
只少 0.076418/0.015620 ms，尚不足以建立稳定净收益。66 份保存输出（含六份
初始构建输出）经独立核对均逐位一致。两个匹配模式的短循环都没有 filtered pool
调用，包括图外拷贝，因此不能把调用缺席归因于图内提交，也不能声称消除长尾。
前期对照见[匹配拷贝记录](agents/system/nosa/nosa_guarded_copy_control.md)。

恢复优化后，单次输入校验内元数据复用在请求 0/16 的完整 candidate 中位数上，
比匹配图内拷贝对照少 0.105163/0.106095 ms；整个外部原型比原始实现少
0.133753/0.125243 ms。66 份保存输出经独立重开均逐位一致。该结果支持这项
局部复用，但顺序进程、重复 candidate 和额外 history 检查副本仍不同于正式用户
轨迹，不能据此声称四方案 serving 加速、长尾消除或 MFU 目标完成。原型暂不接入，
详见[元数据复用结果](agents/system/nosa/nosa_copy_descriptor_result.md)。

当前准备的[indexer 执行 workspace 方案](agents/system/nosa/nosa_indexer_dispatch_plan.md)
保留原有打分和选块 kernel，在执行 lease 内复用选择输出与 normalizer，并减少
重复的临时分配和视图构造。方案仍未实施，统一框架重构没有实现或验证这项候选。
实施前须按验收后的布局重新核对分配与分派假设；是否采用取决于完整 candidate
的净收益、cache 生命周期与四方案验收，不能用分配次数减少代替性能结果。

接入前的 dense request ID 20 与两种 sparse request ID 22 长尾未做 GC/pool 插桩。
后续两次普通复跑重现这些位置，匹配探针在三处 candidate 中均捕获约 36 ms 的 pool
引用扫描；这段扫描不同于 GC collection 耗时。外部原型首轮将三个长尾各缩短约
14 ms，但仍比复访中位数高约 22–24 ms，48 个 offload 首访同 ID 比较中有 47 个变慢。
反序第二轮再次缩短三个长尾约 13.35–13.84 ms；offload 复访均值在两轮中都降低，
首访和典型请求仍有好有坏。这些顺序对照不能排除漂移，同一批请求也不是独立样本。

重构前的扫描实现接入时已完成补测。该次正式轨迹中的 dense20/sync22/async22 仍是各方案
candidate 最大值，分别为 90.496/46.396/48.827 ms。该轮整 case 路由计数确认使用
native 扫描，但没有逐请求扫描区间，不能把旧探针的 36 ms 倒填为当前长尾的解释。
前期诊断与当前结果的身份、保留映射和限制见
发布记录（Git `934485b:docs/agents/system/nosa_pool_scan_publication.md`）。

接入前的完整 A128 indexer 在请求 0/16 的真实输入上复放了全部 32 层，wall 中位数分别为
4.792/5.058 ms，选择和 attention 输出精确一致。这是连续独立 API 调用，包含两次
score pass、有限值检查与选块，但不含模型层间的 staging；不能从完整 candidate
延迟中直接相减来计算可消除的开销。该时期的 Nsight graph/node 诊断均通过输出检查，
但 candidate 分别比普通观测中位数慢约 7.9%/26.4%，数据库还提示事件可能不完整，
因而没有建立完整 GPU 活动或空闲时间的证据。证据边界见
[S-026](agents/research-supervisor/sources.md#s-026nosa-固定-pnh-当前实现的完整测量与效率边界)。

旧通用 budget 热度短轨迹已作为完整范围退出，不再作为当前容量或 baseline 效率的
实验入口。该范围曾有有效局部结果，但未建立用户规模或物理容量结论；退出原因及
共享工具去向见整理记录（Git `934485b:docs/agents/system/experiment_organization.md`）。

研究者将实验按目的分为四类：论文中的 motivation；检验 baseline 实现效率是否合理的
性能实验；用于后续设计、暂不属于论文主线的 sparse pattern 分析；检验自有设计性能的
microbenchmark。是否直接进入论文不是实验的唯一保留标准，目录合并或撤销应服从这些
目的。Baseline 性能检查须对应实际采用的实现和负载，microbenchmark 不替代系统收益。

数值正确性是四类实验共同的前提，本轮按独立验收、正式性能、侵入式 profile 整理入口，
已通过的相关实现与配置不必在每次性能重复中重跑完整参考对照。历史测量中的输出比较大多在
计时外，仍增加实验总耗时与样本间工作；其占比尚未单独测量。运行时 finite 检查、
allocator 校验与 numerical repair 另按实际作用评估，不能把改变后的执行语义称为
原实现的纯计时优化。最初入口整理只完成 CPU 验证；本轮已分别补独立数值、正式计时
与 profile，具体覆盖范围见各最终报告及 S-031，不再沿用“新入口未运行 GPU”的旧状态。
入口整理的历史范围见整理记录（Git `934485b:docs/agents/system/experiment_organization.md`）。

按用户授权，旧短轨迹已清理；请求构造、来源和内存审计已迁出，通用 budget 实现
与必要回归仍保留。Dense/sparse resident 报告迁入共同 baseline 入口，保持原运行身份。

研究者进一步将 NOSA resident baseline 与算子 MFU 合并为一份
[NOSA MFU 报告](../experiments/nosa_mfu/README.md)，后续新增算子的效率结果继续汇入此处。
算子、完整模块与完整模型保留各自的测量范围；A128 固定容量 serving 的效率仍由
motivation 检验。合并报告不改变 4.2–4.3 的证据强度，也不产生新的性能结论。

## 当前最主要的卡点

| 卡在哪里 | 影响哪里 | 下一步建议 |
|---|---|---|
| **ECHO 的完整 Cache Manager 成本尚未量化。** 有准备与 post-top-k 局部证据，但 Q1 差距不能全归入管理；新增 Batch=128 驻留算子观测不包含管理事务 | 2.6、3.4、4.4 | **T-009：Decode 管理开销与决策粒度**，识别管理引入的成本，探索减少逐层决策 |
| **Indexer Cache 常驻对 batch 与 Overlap 的因果影响尚未验证。** 缺少 Batch=128 Decode 的容量与供数对照 | 2.7、3.5、4.5 | **T-010：Indexer Cache offload 与重叠**，检验容量收益、新增 IO 与流水效率 |
| **剩余 Recall 尚未纳入 Attention 重叠。** 冷态 Q1 仍需较多残余召回，官方自然驻留另有边界 | 2.8、3.6、4.5 | **T-011：Main Cache 读取与计算重叠**，联合考虑 Indexer、Attention 与剩余 Recall |
| **Decode 的场景和共同对照仍需建立。** Q1 与 Batch=128 已有驻留单层观测；agentic 任务/数据、多步事务与完整 baseline 仍未建立 | 1.1–1.5、2.1、4.3 | **T-002、T-003：场景适配与 Decode 可比范围**，明确两档负载的语义与执行边界 |

既有 NOSA H64K workspace 方案 T-007 保留为独立的 Prefill/Extend 任务，排在当前
Decode 主线之后；既有元数据复用等局部改善尚未形成四方案完整 serving 收益。既有 motivation、
4K/16K、容量填满与后续审阅、simulation 不因本轮状态更新自动恢复。新 Decode
任务所需的容量分析另按其负载定义，固定 token 配额仍不等于相同物理预算。

DeepSeek 当前有效结论以上方用户修正与[固定容量 motivation](../experiments/deepseek_v32_motivation/README.md)
为准，不恢复已撤回的 4 GiB/W/chunk 排名。NOSA 的
[固定容量结果](../experiments/nosa_motivation/README.md)、
[resident A1024 报告](../experiments/nosa_mfu/README.md)与 offload 算子结果
分别保留来源和边界，不能混成同一实现或跨模型排名。旧 budget-serving 实验已退出；
candidate 效率、因果归因和物理预算证明仍有缺口。
具体下一步见[待办清单](roadmap.md)。

内部回查：[材料依据](agents/research-supervisor/sources.md)、[重要理解修正](agents/research-supervisor/updates.md)。
运行细节不作为这张表的研究主线。
