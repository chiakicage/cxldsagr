# cxldsagr 研究状态

> **2026-10-05 最新安排：**按用户继续推进目标的要求，恢复 H64K NOSA 性能优化。
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

更新：2026-10-05。保留研究者选择的固定用户顺序 loop。两模型的统一框架重构结果
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
当前两模型控制点均为 U=16、R=2、H=65536、A=128、P=65536、NH=16777216；
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
| **1.3 使用什么数据？** | 用户选择固定 U 个 user 的合成 loop 数据，不需要热度；沿用每用户固定 history、每次访问变化 candidate 的请求内容 | **已有受控数据点。** DeepSeek 使用 16 用户、64K history 与 128 candidate；真实推荐标签、其他规模及场景代表性仍未验证 |
| **1.4 怎样形成 serving 负载？** | 同一用户顺序完整重复 R 轮：总请求 UR，首访 U，复访 U(R−1)；相邻两次访问之间有 U−1 个不同用户 | **已有两模型控制点及同机重复。** U=16/R=2、H=65536/A=128、P=65536/NH=16777216；两模型各有三轮 P0 与三轮重构后测量。重复次数不增加用户规模覆盖；Beauty 短轨迹不替代完整用户覆盖，同参数也不等于跨模型同计算 |
| **1.5 主要优化目标是什么？** | 相同 HBM/DRAM 硬预算下比较完整 loop 的逐请求延迟，分开首轮、后续复访命中和复访重建；淘汰后重建仍计入复访 | **指标方向已明确。** 固定 P/NH 实验比较 token 额度，不等于相同总 HBM 字节预算；首访比例 1/R 影响全请求均值，仍需分轮报告。真实服务的吞吐/到达目标待确定 |

## 2. SOTA baseline 与 Motivation

- [ ] 覆盖主要相关方案，讲清它们的取舍，并量化目标场景中的关键问题。

当前候选论证是：**全 HBM 的容量约束 → CPU DRAM offload 的取数等待 → dense prefetch
的冗余 → sparse fetch 的重叠困难**。这是待分析和验证的思路线索，各环节尚未整体成立。

| 条目 / 要回答的问题 | 当前理解 | 状态与具体缺口 |
|---|---|---|
| **2.1 主要 baseline 是否覆盖？** | motivation 目标集合为 HBM-only、ECHO、full prefetch、sync sparse loading、async sparse loading；full prefetch 搬运完整历史，attention 仍稀疏 | **两模型各有四类对照。** 统一资源与执行契约已接入，正式计时、独立数值依据和 profile 分别记录。ECHO 尚未接入同一 NOSA 对照，不能拼接跨模型排名；SOTA 覆盖仍待补 |
| **2.2 全 HBM 的问题是什么？** | 固定复用距离下，history 配额影响复访是否重建；实际物理容量还受模型布局、workspace、allocator 和其他分配约束 | **指定配额的控制点支持。** P=H 时，两个模型的 HBM-only 均只保留一个 history、复访 0/16 命中，三个 offload 方案均为 16/16。这不证明 HBM 物理上只能容纳一个用户，也不恢复已撤回的 C1024/C2048 容量结论 |
| **2.3 CPU DRAM offload 的问题是什么？** | NOSA 固定容量 loop 中，保留历史避免了重新构建，但引入了首访分配、写回及复访取数成本 | **重构后已测量。** 各轮先求 16 次复访均值，再取三轮中位数，HBM/dense/sync/async 分别为 2342.511/71.980/29.568/31.952 ms；三个 offload 的首访均慢于 HBM。容量保留收益与取数代价须分开，不能把完整差值归为串行等待 |
| **2.4 Dense prefetch 的问题是什么？** | 整层历史预取仍使用 sparse attention。NOSA 固定容量轨迹中，dense 的复访 candidate H2D payload 多于两种 sparse，复访延迟也高于同步 sparse | **该控制点有冗余搬运观测。** P0 与重构后保留相同的输入、命中分类、配额及 H2D；这些是软件 payload，不是物理总线流量。流量、预取机会、非矩阵工作与调度成本尚未完成因果分解，不能外推为通用排名 |
| **2.5 与 ECHO 如何比较？** | ECHO prefill fetch 适用于后续 chunk 或已有 offloaded prefix 的 extend；无 decode 不排除适用。已复现层内共享 token cache，并单独标明 GR session 保留与容量/异步安全适配 | **本地对照成立，方法收益未成立。** 当前固定 P/NH、C10 替身对照中，ECHO 的复访延迟仍高于 serial sparse，候选 H2D 相同；不能宣称融合预取加速。旧 W/chunk 排名已撤回，该结果也不替代共同模型上的 NOSA/ECHO 比较 |

ECHO cache baseline 的官方逐层共享 pool、实际领取才驱逐、
`indexer/prefetch → top-k → 主 KV append → exact recall` 的顺序及时间戳 priority
已核对并修正原本地差异。旧实现与 chunk 选择计划（Git `934485b:docs/agents/system/echo_cache_implementation_plan.md`）
只保留历史回查用途，不恢复已撤回的预算、chunk 和性能判断。当前取舍以
[固定 P/NH motivation](../experiments/deepseek_v32_motivation/README.md)与
[官方适配对照](../experiments/deepseek_v32_echo_official/README.md)各自的最终报告为准。
官方未覆盖的容量/异步边界与本地适配单独验收，不称为完整 SGLang serving 复现。

非 GR 计算对照使用真实前三层 64K+1K，算子 kernel 利用率与完整阶段墙钟利用率分别报告。
历史跨版本输出并非逐元素等价，稀疏选择也有变化，未测推荐质量；不把历史数值一致范围
扩展到新版本。最终延迟、输入、数值依据和 profile 边界见
[计算报告](../experiments/deepseek_v32_echo_prefill/README.md)。这些局部结果不证明 cache
或 ECHO 方法收益。

本轮真实前三层的 resident/offload prefix 墙钟中位数为 642.340/1190.485 ms，
extend 为 13.954/27.632 ms。按各精度 dense 峰值归一化的完整阶段有效计算利用率，
prefix 为 45.08%/24.32%，extend 为 38.70%/19.54%；这些不是单个 kernel 的 MFU。
独立 check02 与 profile 数值检查通过，本轮没有新增 NCU。该普通三层缓存路径
与固定 P/NH 的 C10 serving 替身分开，不由这组数值推断完整模型或 GR 质量。

C10 固定 loop 的 clean bench01 中，ECHO/serial sparse/dense 的复访请求均值为
24.530/18.084/33.197 ms；ECHO 与 serial sparse 的 16 次复访 candidate H2D
均为 1.161186 GiB。该控制点未显示 ECHO 的整体加速收益；单凭相同搬运量与延迟
差异，仍不能判定融合预取的因果作用或是否发生重叠。独立矩阵活动与正式墙钟的
差值也不是已隔离的可移除 CPU 成本。

最终数据分别见[C10 诊断](../experiments/deepseek_v32_motivation/report/diagnosis/aggregate_mfu.json)
与[真实三层汇总](../experiments/deepseek_v32_echo_prefill/report/layers3/summary.json)；
两者的 run、数值依据与发布来源在 S-031 分别绑定。

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
不等于 tensor payload 之和。Allocated、reserved 和设备已用量分开，固定 16 用户
轨迹不证明装满 NH 或离线最大容量。

## 4. Evaluation

- [ ] 用整体比较、机制分析和消融回答前面提出的研究问题，并明确适用范围。

| 条目 / 要回答的问题 | 当前理解 | 状态与具体缺口 |
|---|---|---|
| **4.1 整体系统收益是什么？** | 两模型分别比较完整 loop、首访和复访；offload 的复访优势首先来自保留历史、避免 HBM 重建 | **容量、重构成本与方法收益分开。** NOSA 的复访均值三轮中位数中，async 比 sync 慢 8.06%；重构后的完整 trace 中位总耗时比 P0 增加 406.132 ms。两模型各自全部 128 个匹配请求的 cleanup 中位数均增加，完整 trace 汇总不能抵消局部开销。元数据复用只有历史局部改善；五类共同模型对照与场景质量仍未建立 |
| **4.2 哪些设计带来收益？** | 独立矩阵/attention API、resident A1024 与冷稀疏并集 offload 回放分别提供局部证据；历史 H64K prequeue 与元数据复用诊断支持继续检验主机编排，但不隔离可移除成本 | **收益依赖输入与范围。** 本轮 A1024 三层算子回放的融合 API 延迟下降 23.02%–29.47%，9 个内部样本双比率均过 90%，最低为 0.91081246；A128 固定 serving 的 96 个适用样本仍全部未过门槛。单层收益不证明 serving 收益；candidate 效率、剩余长尾和重叠失败仍待归因 |
| **4.3 结论适用于哪些条件？** | 固定容量控制点覆盖 SM90/Hopper、16 用户两轮、64K+128；NOSA 使用完整 32 层 checkpoint，DeepSeek 使用独立复制 block 的替身且另执行末 token LM head | **范围有限。** 两模型各有三轮同机 P0 与重构后正式 trace；NOSA pair04 是另一次复核，不混入三轮统计。匹配诊断只选请求 0/16。重复测量不扩展到其他 U/R、物理字节预算、真实到达过程或 GR 质量；resident A1024 与 pattern 仍有各自范围 |

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
| **当前 ECHO 融合预取没有取得相对串行 sparse 的复访收益。** 候选 H2D 相同，预测/预取与等待的具体成本仍需按本轮 profile 分析 | 2.5、3.1、4.1、4.2 | **T-007：loop motivation 测量与归因**，区分预测/预取开销、取数重叠和用户历史保留 |
| **缺少可以直接采用的 sparse attention GR 场景。** 已有生成器和模型代码，但模型、数据与服务需求如何对应仍不清楚 | 1.1–1.5；影响 2.2 的容量动机和 4.1 的整体评测 | **T-002：场景构造方案比较**，先列出具体可行路线和各自能回答的问题 |
| **五类 baseline 尚未在共同模型上形成完整可比集合。** 共享资源与执行契约已接入，但 ECHO 与 async 仍分属不同模型入口，模型和输出范围也不同 | 2.1、2.5、3.1、3.3 | **T-003：共同模型适配与可比范围**，明确剩余适配与可回答的问题 |
| **两模型固定容量结果仍只覆盖一个共同参数点。** NOSA 与 DeepSeek 的计算和物理分配不同，其他 U/R、P/NH 及字节预算范围尚未验证 | 1.4、2.2、4.1、4.3 | **T-006：loop 数据与容量压力**，在 H64K、不跑满 NH 的约束下选择有区分力的范围并保持完整轮次 |
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
