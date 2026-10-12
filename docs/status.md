# cxldsagr 研究状态

> **2026-10-12 模型范围调整：**用户明确要求移除 NOSA 的代码、实验及共享接入。
> 当前实现与实验只保留 DeepSeek V3.2，T-007 取消。下文较早的 NOSA 提及仅记录当时
> 范围，不再支持当前研究判断或执行安排；移除不是对其历史实验有效性的重新判定。依据见 S-042。

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
既有 DeepSeek GR 对照保留研究者选择的固定用户顺序 loop，并分别报告指定 token 配额、
实际内存占用和测量边界；这些结果不证明真实场景质量或物理容量上限。

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
DeepSeek 原控制点为 U=16、R=2、H=65536、A=128、P=65536、NH=16777216；
DeepSeek 后续在相同 U/R 与 P/NH 下补齐了 12 点 H/A 矩阵。
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
| **1.2 使用什么模型？** | 当前使用 DeepSeek V3.2；既有 GR 对照为约 8B 的 C10 checkpoint 工作负载替身，另有真实前三层 Q1 对照。NOSA 已按用户要求移除 | **已有执行对象，新增范围未验收。** C10 未经训练，前三层不等于完整 61 层；Decode 的模型适配与任务质量仍需明确 |
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
| **2.1 主要 baseline 是否覆盖？** | 候选集合仍为 HBM-only、ECHO、full prefetch、sync sparse loading、async sparse loading；当前优先明确 Q1 与 Batch=128 Decode 中 ECHO 的可比路径 | **已有驻留算子基线，共同 Decode 对照未建立。** DeepSeek Extend 与前三层 Q1 保留原范围；无 offload 算子观测不能补齐五类完整 serving baseline，SOTA 覆盖仍待补 |
| **2.2 全 HBM 的问题是什么？** | 固定复用距离下，history 配额影响复访是否重建；实际物理容量还受模型布局、workspace、allocator 和其他分配约束 | **指定配额下的历史保留差异已测。** DeepSeek 固定 P=65,536 时，H4K 的 HBM 保留全部 16 用户、复访 16/16 命中；H16K/H64K 分别只能保留 4/1 个用户，顺序 loop 的复访均为 0/16。三个 offload 方案始终 16/16。H4K 没有避免历史重建的收益，不能沿用 H64K 的解释；这些配额结果不证明物理容量上限，也不恢复已撤回的 C1024/C2048 结论 |
| **2.3 CPU DRAM offload 的问题是什么？** | 保留历史可以避免复访重建，同时引入分配、写回和取数成本。DeepSeek C10 矩阵表明取舍随 H/A 改变 | **容量保留收益与取数代价须分开。** H4K 的 HBM 复访全部命中，四档 A 均由 HBM 取得最低复访均值；H16K/H64K 的对照另有历史保留收益。不能把完整延迟差值都归为串行等待 |
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
| **3.1 Sparse fetch 为什么难以重叠？** | Decode 需分析 batch、Indexer Cache 供数和剩余 Recall 对流水的限制 | **待分析。** 现有 DeepSeek 局部 profile 不能分解完整 Decode 请求成本，也不能替代 Batch=128 的对照 |
| **3.2 当前设计如何应对？** | 基于 DeepSeek 检验减少管理决策、offload Indexer Cache，以及让 Indexer、Attention 与 Main Cache 读取重叠 | **设计方向，收益待验证。** 需保持完整选择、固定历史与临时候选语义、容量限制和事务边界；NOSA 实现已退出当前方案 |
| **3.3 技术区别和研究贡献是什么？** | 当前拟分别减少 Cache Manager 开销，并让 Indexer、Attention 的计算与相应 Cache 读取重叠 | **设计方向已明确，贡献仍待验证。** 与 ECHO 和其他方案的实质区别、适用范围及净收益尚未建立，最终方法与贡献表述未确定 |
| **3.4 怎样进一步减少管理开销？** | 探索无需逐层决策的管理方式，减少重复决策与由管理流程引入的开销 | **设计方向，尚未实现或验证。** 需说明不同层的选择与驻留如何支持统一决策，并保持完整稀疏选择、容量约束和事务语义；操作次数减少不等于端到端收益 |
| **3.5 怎样重叠 Indexer 计算与 Indexer Cache？** | 将 Indexer Cache offload 纳入设计，让 Indexer 计算与 Indexer Cache 的读取/搬入重叠，探索扩大 batch 后的流水效率 | **设计方向，尚未实现或验证。** 需确定供数粒度、缓冲容量和计算依赖；offload 是否扩大可行 batch、能否隐藏新增读取成本仍待测 |
| **3.6 怎样重叠 Main Cache 读取？** | 同时考虑用 Indexer 与 Attention 的计算重叠 Main Cache 读取，并将剩余 Recall 纳入流水 | **设计方向，尚未实现或验证。** 需明确精确选择就绪前后的可读范围、供数/消费依赖及缓冲复用；管理成本与重叠收益分别归因，再检验完整 Decode 的净收益 |

DeepSeek 的共享全局 host arena 按 pinned allocator 档位预留，固定 P/NH 配额与实际
字节数分别报告。静态规划、cache 账本与完整请求内存观测见
[容量报告](../experiments/cache_management/README.md)。
逻辑 payload、cache 计费及预留、进程占用分别解释；计费包含 graph private reservation，
不等于 tensor payload 之和。Allocated、reserved 和设备已用量分开。本次 DeepSeek
账本更新与完整请求内存观测已发布，五项静态 P/NH 边界未变；固定 16 用户轨迹仍
不证明装满 NH 或离线最大容量。

## 4. Evaluation

- [ ] 用整体比较、机制分析和消融回答前面提出的研究问题，并明确适用范围。

| 条目 / 要回答的问题 | 当前理解 | 状态与具体缺口 |
|---|---|---|
| **4.1 整体系统收益是什么？** | DeepSeek 分别比较完整 loop、首访和复访；H16K/H64K 的 offload 复访优势首先来自保留历史，H4K 全命中时没有这项重建差值。真实三层 MFU 单独检查 baseline 执行效率 | **容量、执行效率与方法收益分开。** C10 矩阵的最低复访均值随 H/A 在 HBM、dense 和 serial sparse 间变化；一轮结果不证明稳定排名，也不建立完整 Decode 对照或场景质量 |
| **4.2 哪些设计带来收益？** | DeepSeek 独立算子与局部 profile 提供局部证据；主机编排诊断不隔离可移除成本 | **收益依赖输入与范围。** 真实三层与 C10 单点均有 dense DMA/计算重叠证据；新 C10 矩阵中 dense 在 6 点复访最低，但没有新增 profile 来解释这些点。不能沿用旧单点作跨配置机制归因。Q1 独立进程复测的 HBM 窗口接近官方参考，cold ECHO 完整 step 仍慢于 serial sparse。准备融合的私有 500 对中位数下降约 1.32%，尾部未一致改善；默认路径接入与独立补测已完成，三层准备由 9 个节点减到 3 个（S-038、S-039）；不据此推广 serving 收益或恢复广泛优化 |
| **4.3 结论适用于哪些条件？** | DeepSeek 已有 C10 的 12 点 H/A Extend 矩阵及真实前三层 Q1，另有包括 Batch=128 Decode 的 52 点无 offload 单层算子观测。Prefill/Extend 与 Decode 分别报告 | **驻留算子范围已扩展，完整 agentic serving 尚未覆盖。** 新矩阵使用合成 activation 与独立序列 KV，未运行上下文增长、多步追加或完整 61 层。既有 Q1 本地冷态与官方自然驻留，以及 C10 的模型/输出、配额和统计边界保留，不外推场景代表性 |
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

## 当前最主要的卡点

| 卡在哪里 | 影响哪里 | 下一步建议 |
|---|---|---|
| **ECHO 的完整 Cache Manager 成本尚未量化。** 有准备与 post-top-k 局部证据，但 Q1 差距不能全归入管理；新增 Batch=128 驻留算子观测不包含管理事务 | 2.6、3.4、4.4 | **T-009：Decode 管理开销与决策粒度**，识别管理引入的成本，探索减少逐层决策 |
| **Indexer Cache 常驻对 batch 与 Overlap 的因果影响尚未验证。** 缺少 Batch=128 Decode 的容量与供数对照 | 2.7、3.5、4.5 | **T-010：Indexer Cache offload 与重叠**，检验容量收益、新增 IO 与流水效率 |
| **剩余 Recall 尚未纳入 Attention 重叠。** 冷态 Q1 仍需较多残余召回，官方自然驻留另有边界 | 2.8、3.6、4.5 | **T-011：Main Cache 读取与计算重叠**，联合考虑 Indexer、Attention 与剩余 Recall |
| **Decode 的场景和共同对照仍需建立。** Q1 与 Batch=128 已有驻留单层观测；agentic 任务/数据、多步事务与完整 baseline 仍未建立 | 1.1–1.5、2.1、4.3 | **T-002、T-003：场景适配与 Decode 可比范围**，明确两档负载的语义与执行边界 |

NOSA 及 T-007 已按用户要求移除。DeepSeek 的 GR/agentic 候选、Q1 与 Batch=128
优先级、Cache Manager 与 Overlap 的研究划分继续保留。既有 motivation、容量填满、
后续审阅和 simulation 不因这次清理自动恢复；固定 token 配额仍不等于相同物理预算。

DeepSeek 当前有效结论以上方用户修正与[固定容量 motivation](../experiments/deepseek_v32_motivation/README.md)
为准，不恢复已撤回的 4 GiB/W/chunk 排名。
具体下一步见[待办清单](roadmap.md)。

内部回查：[材料依据](agents/research-supervisor/sources.md)、[重要理解修正](agents/research-supervisor/updates.md)。
运行细节不作为这张表的研究主线。
