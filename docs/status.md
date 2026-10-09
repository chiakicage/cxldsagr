# cxldsagr 研究状态

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

更新：2026-10-09（Q1 准备融合已完成生产验收与独立补测）。保留研究者选择的固定用户顺序 loop。两模型的统一框架重构结果
与同机 P0 分别比较，原已发布数字保留历史来源。NOSA 三轮完整 trace 的中位总耗时
比 P0 增加 406.132 ms（0.19664%）；准入、清理和部分 candidate／复访阶段仍有耗时
增加，不能声称重构没有性能代价。A128 的 96 个适用内部样本均未达到两种 90% 门槛；
A1024 冷并集单层回放的 9 个样本均通过，最低 ratio 为 0.91081246。
两者回答不同执行路径的问题。两模型使用相同 token 配额，但物理分配方式不同；
本轮没有跑满 NH，也不证明真实 GR 场景、任务质量或物理容量上限。
T-007 的 indexer 执行 workspace 复用仍是未实施的独立方案，重构没有实现或验证它。
此前元数据复用的局部改善与诊断边界继续保留，4K/16K 暂停。

研究从 **sparse KV fetching 与 sparse attention 计算的重叠设计**出发，正在探索
**sparse attention GR serving** 是否是合适的应用场景。当前范围是 **HBM 与 CPU DRAM**。
最主要的未决问题是：如何建立有依据的 GR 任务、模型、数据和服务负载，并在其中说明
KV 容量约束、offload 开销及相对 ECHO 的设计价值。已有单卡、跨请求保留用户历史的
串行测量；**DeepSeek 替身与完整 NOSA 各有固定容量四方案控制点，五类方案的共同模型比较仍未建立**。
固定相同的共享 cache 实现后，官方计算后端与非矩阵算子适配进一步改善了前三层计算效率；
算子 kernel 利用率与完整阶段的端到端利用率分开报告。计算优化与本次 cache/serving 结果分别解读。
旧短轨迹没有建立有效的用户规模与缓存容量压力对照；16K 的少量复访重建差异只能描述
原轨迹。固定 U 个用户、按同一顺序遍历 R 轮的选择不变，用户覆盖与复用距离据此解释。
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
| **1.1 研究什么任务与场景？** | sparse attention GR serving 是候选场景；当前讨论固定 history 的构建与变化 candidate 的 extend，不含自回归 decode，跨请求保留有效历史 | **候选。** 执行生命周期已明确，仍需明确具体推荐任务、服务方式和适用范围 |
| **1.2 使用什么模型？** | 现有执行对象为完整 NOSA-8B，以及 DeepSeek V3.2 前三层权重和对应输入独立复制构成的约 8B dense 工作负载替身；后续开发优先 NOSA | **执行对象已确定，任务模型仍待选择。** DeepSeek 替身未经训练，不等于完整 61 层验证；NOSA 只输出 hidden，DeepSeek 另执行末 token LM head，不能直接比较跨模型延迟或任务质量 |
| **1.3 使用什么数据？** | 用户选择固定 U 个 user 的合成 loop 数据，不需要热度；沿用每用户固定 history、每次访问变化 candidate 的请求内容 | **已有受控矩阵。** DeepSeek 已测 16 用户、H=[4K,16K,64K] × A=[128,256,512,1024] 的 12 点矩阵。真实推荐标签及场景代表性仍未验证 |
| **1.4 怎样形成 serving 负载？** | 同一用户顺序完整重复 R 轮：总请求 UR，首访 U，复访 U(R−1)；相邻两次访问之间有 U−1 个不同用户 | **已有两模型控制点及同机重复。** U=16/R=2、H=65536/A=128、P=65536/NH=16777216；两模型各有三轮 P0 与三轮重构后测量。重复次数不增加用户规模覆盖；Beauty 短轨迹不替代完整用户覆盖，同参数也不等于跨模型同计算 |
| **1.5 主要优化目标是什么？** | 相同 HBM/DRAM 硬预算下比较完整 loop 的逐请求延迟，分开首轮、后续复访命中和复访重建；淘汰后重建仍计入复访 | **指标方向已明确。** 固定 P/NH 实验比较 token 额度，不等于相同总 HBM 字节预算；首访比例 1/R 影响全请求均值，仍需分轮报告。真实服务的吞吐/到达目标待确定 |

## 2. SOTA baseline 与 Motivation

- [ ] 覆盖主要相关方案，讲清它们的取舍，并量化目标场景中的关键问题。

当前候选论证是：**全 HBM 的容量约束 → CPU DRAM offload 的取数等待 → dense prefetch
的冗余 → sparse fetch 的重叠困难**。这是待分析和验证的思路线索，各环节尚未整体成立。

| 条目 / 要回答的问题 | 当前理解 | 状态与具体缺口 |
|---|---|---|
| **2.1 主要 baseline 是否覆盖？** | motivation 目标集合为 HBM-only、ECHO、full prefetch、sync sparse loading、async sparse loading；full prefetch 搬运完整历史，attention 仍稀疏 | **两模型各有四类对照。** 统一资源与执行契约已接入，正式计时、独立数值依据和 profile 分别记录。ECHO 尚未接入同一 NOSA 对照，不能拼接跨模型排名；SOTA 覆盖仍待补 |
| **2.2 全 HBM 的问题是什么？** | 固定复用距离下，history 配额影响复访是否重建；实际物理容量还受模型布局、workspace、allocator 和其他分配约束 | **指定配额下的历史保留差异已测。** DeepSeek 固定 P=65,536 时，H4K 的 HBM 保留全部 16 用户、复访 16/16 命中；H16K/H64K 分别只能保留 4/1 个用户，顺序 loop 的复访均为 0/16。三个 offload 方案始终 16/16。H4K 没有避免历史重建的收益，不能沿用 H64K 的解释；这些配额结果不证明物理容量上限，也不恢复已撤回的 C1024/C2048 结论 |
| **2.3 CPU DRAM offload 的问题是什么？** | NOSA 固定容量 loop 中，保留历史避免了重新构建，但引入了首访分配、写回及复访取数成本 | **重构后已测量。** 各轮先求 16 次复访均值，再取三轮中位数，HBM/dense/sync/async 分别为 2342.511/71.980/29.568/31.952 ms；三个 offload 的首访均慢于 HBM。容量保留收益与取数代价须分开，不能把完整差值归为串行等待 |
| **2.4 Dense prefetch 的问题是什么？** | 整层历史预取仍使用 sparse attention；搬运量较多不必然意味着请求更慢。DeepSeek 三层完整图与 C10 单点均有连续内存 DMA 和计算的局部重叠证据 | **C10 取舍随 H/A 改变。** H16K 四档 A 的 dense 复访均值最低；H64K 的 A128/256 为 serial sparse 最低，A512/1024 为 dense 略低。ECHO/sparse 的 H2D 非零时，dense payload 为它们的 1.832–9.688 倍。不能仅据字节数或旧单点局部重叠解释排序；流量是软件 payload，矩阵未新增 profile |
| **2.5 与 ECHO 如何比较？** | ECHO prefill fetch 适用于后续 chunk 或已有 offloaded prefix 的 extend；无 decode 不排除适用。已复现层内共享 token cache，GR session 保留与容量/异步安全适配单独说明 | **本轮矩阵未显示 ECHO 的复访优势。** ECHO 在 12 个 H/A 点的复访均值均高于 serial sparse，两者 candidate H2D 总量逐点相同。三层 cold A128 完整图也单独观察到 ECHO 较慢；一轮矩阵不证明稳定排序，融合 kernel 的计算与 IO 不从 trace 内部分离，独立 SGLang 计时不替代共同模型上的 NOSA/ECHO 比较 |

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

2026-10-07 的[官方 recall 诊断](../experiments/deepseek_v32_echo_official/report/recall_diagnosis.md)
确认了独立 SGLang 的一项实现开销：召回每层扫描全部 host pool 的标记，零 miss 时
仍耗时约 1.53 ms。相同二进制的隔离实验与 NCU 将主要成本定位到扫描中的整数与
逻辑运算。因此图中长 recall 条不能作为纯取数等待或 ECHO 重叠能力不足的证据。
本地实现使用不同的召回路径，这一原因不能直接解释 C10 矩阵中的 ECHO 排名。
本轮只完成诊断，未修改实现或恢复广泛优化；详细依据见 S-035。

后续用户指定的 Q1 优化已有独立结果：本地 HBM 的 GPU 窗口接近官方参考，
ECHO 在本地 cold 条件下仍慢于 serial sparse。准备流程会影响节点间隙，
因此 GPU gap 不能未经控制就归为可移除的计算开销。官方 ECHO 与本地使用不同
输入和自然／cold 驻留；其融合 indexer 时长差不能单独解释为实现效率差异。
这些结果只对应真实三层单 token 路径，详见 S-037。

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
| **3.1 Sparse fetch 为什么难以重叠？** | NOSA 当前固定容量测量的 96 个适用内部样本均未达到两种 90% 重叠门槛；无 host-copy 的样本不计为通过 | **A128 重叠门槛未满足。** 先前候选的 NCU 诊断提示 host 数据依赖和 consumer K/V 就绪等待，同二进制对照确认对齐可消除高于逻辑 payload 的 12.5% sector 访问；这些局部结果不能分解当前请求延迟，也未证明计算或 IO 主导 |
| **3.2 当前设计如何应对？** | NOSA 融合唯一历史读取与 sparse attention；固定 P/NH 专用路径使用有限逐层容量、逻辑位置直接映射和 session 身份标签，candidate 只在 GPU 临时执行 | **固定路径已实现，方法收益未成立。** 当前 async 在首访、复访和完整轨迹均慢于 sync。多个 Q128 候选未通过数值或重叠门槛，均未合入；通用 budget 共享 staging 仍不等于通用有限槽缓存或热点淘汰策略 |
| **3.3 技术区别和研究贡献是什么？** | 希望在减少搬运的同时获得重叠收益 | **待总结。** 与 ECHO 和其他方案的实质区别尚未建立；系统名称、最终技术划分和贡献表述均未确定 |

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
| **4.3 结论适用于哪些条件？** | DeepSeek 已测 SM90/Hopper、16 用户两轮、固定 P/NH 的 H=[4K,16K,64K] × A=[128,256,512,1024]；NOSA 原固定容量范围仍为 64K+128。两模型计算与输出范围不同 | **12 点矩阵已发布，外推范围仍有限。** 每个配置只测一条正式轨迹，每方案含两轮用户访问，不混入原重构对照的三轮统计。DeepSeek 是独立复制 block 的 C10 替身且执行末 token LM head，NOSA 使用完整 32 层 checkpoint；其他 U/R、物理字节预算、真实到达过程与 GR 质量未验证。NOSA pair04、请求 0/16 诊断、resident A1024 与 pattern 仍有各自范围 |

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
| **ECHO 相对 serial sparse 的加速尚未成立。** C10 的 12 点矩阵均观察到 ECHO 复访较慢；预测/预取、搬运方式与等待的因果归因仍有缺口 | 2.5、3.1、4.1、4.2 | **T-003：共同模型适配与可比范围。** 矩阵任务已完成，不把一轮排序视为稳定规律；用户指定的 Q1 优化由 T-008 单独推进，不扩大 C10 或 NOSA 的实验范围 |
| **缺少可以直接采用的 sparse attention GR 场景。** 已有生成器和模型代码，但模型、数据与服务需求如何对应仍不清楚 | 1.1–1.5；影响 2.2 的容量动机和 4.1 的整体评测 | **T-002：场景构造方案比较**，先列出具体可行路线和各自能回答的问题 |
| **五类 baseline 尚未在共同模型上形成完整可比集合。** 共享资源与执行契约已接入，但 ECHO 与 async 仍分属不同模型入口，模型和输出范围也不同 | 2.1、2.5、3.1、3.3 | **T-003：共同模型适配与可比范围**，明确剩余适配与可回答的问题 |
| **固定 token 配额不等于相同物理预算。** DeepSeek 已补齐 12 点 H/A 矩阵，但两模型的计算和分配不同，其他 U/R、P/NH、字节预算范围仍未验证 | 1.4、2.2、4.1、4.3 | **当前不追加容量测量。** 保留历史命中与 candidate 成本的区别；容量填满和后续审阅仍暂缓，不跑满 NH，不扩展 NOSA 范围 |
| **NOSA 当前固定容量 loop 未显示 async 收益。** H64K prequeue 显示提交敏感，但 candidate 效率差距、剩余长尾和供数/流水线等待仍需归因；外部原型尚未形成四方案完整 serving 收益 | 2.1、2.3、2.4、3.1、3.2、4.1、4.2 | **T-007：H64K 效率与归因**，用完整 candidate 对照检验 indexer workspace 复用，并保持四方案正确性和测量验收 |

DeepSeek 当前有效结论以上方用户修正与[固定容量 motivation](../experiments/deepseek_v32_motivation/README.md)
为准，不恢复已撤回的 4 GiB/W/chunk 排名。NOSA 的
[固定容量结果](../experiments/nosa_motivation/README.md)、
[resident A1024 报告](../experiments/nosa_mfu/README.md)与 offload 算子结果
分别保留来源和边界，不能混成同一实现或跨模型排名。旧 budget-serving 实验已退出；
candidate 效率、因果归因和物理预算证明仍有缺口。
具体下一步见[待办清单](roadmap.md)。

内部回查：[材料依据](agents/research-supervisor/sources.md)、[重要理解修正](agents/research-supervisor/updates.md)。
运行细节不作为这张表的研究主线。
