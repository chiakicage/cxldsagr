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
> [固定 P/NH 容量报告](../experiments/deepseek_v32_echo_cache/README.md)现按 GPU 临时
> candidate 重新规划：NH 只保留 history，candidate 整批执行，用完丢弃，不写回 DRAM。
> 512 GiB DRAM 下的 NH 静态边界对应 455 个 64K history；P 还受这些用户的 resident
> indexer、共享 workspace 和 allocator 余量约束。新结果是静态估算，未跑满用户；短
> checkpoint 检查仅验证数值与存储语义。旧 candidate 写回 DRAM 的容量结果已替换，
> 不作为新实现的显存测量或性能排名。NOSA 与独立前三层计算实验不在此次撤回范围内。

更新：2026-10-05。保留研究者选择的固定用户顺序 loop。NOSA 固定 P/NH 的正式轨迹、匹配 profile 和独立 API 已验收；旧通用 budget 短轨迹已结束独立实验维护。固定 loop 的 async 仍慢于 sync，96 个适用内部样本也都未达到两种 90% 门槛。完整请求接近独立 API 组合，不表示 candidate 已高效或计算、IO 已主导。恢复 H64K 优化后，元数据复用候选在两个请求中各有约 0.1 ms 的局部改善，尚未接入；下一步准备 indexer 执行 workspace 复用原型，以完整 candidate 对照检验收益，该原型尚未运行 GPU。生产实现与已有正式结果不变。DeepSeek 的用户修正与上方固定容量结果保持原边界，五类方案的共同模型比较仍未建立。

研究从 **sparse KV fetching 与 sparse attention 计算的重叠设计**出发，正在探索
**sparse attention GR serving** 是否是合适的应用场景。当前范围是 **HBM 与 CPU DRAM**。
最主要的未决问题是：如何建立有依据的 GR 任务、模型、数据和服务负载，并在其中说明
KV 容量约束、offload 开销及相对 ECHO 的设计价值。已有单卡、跨请求保留用户历史的
串行测量；**DeepSeek 替身与完整 NOSA 各有固定容量四方案控制点，五类方案的共同模型比较仍未建立**。
固定相同的共享 cache 实现后，官方计算后端与非矩阵算子适配进一步改善了前三层计算效率；
算子 kernel 利用率与完整阶段的端到端利用率分开报告。计算优化与本次 cache/serving 结果分别解读。
旧短轨迹也没有建立有效的用户规模与缓存容量压力对照；16K 的少量复访重建差异只能描述
原轨迹。当前计划固定 U 个用户，按同一顺序遍历 R 轮，使用户覆盖与复用距离可直接解释；
已完成的 DeepSeek 控制点固定 U=16、R=2、HBM 4 GiB / DRAM 64 GiB；C1024 的
workspace 可保留 16 个 session，而 C2048 只能保留 15 个，复访分别全命中与全重建。
这支持该控制点的容量边界解释；尚未证明真实 GR 场景的代表性，也不确定其他规模的结论。

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
| **1.4 怎样形成 serving 负载？** | 同一用户顺序完整重复 R 轮：总请求 UR，首访 U，复访 U(R−1)；相邻两次访问之间有 U−1 个不同用户 | **已有两模型控制点。** NOSA 接入扫描优化后，已完成 U=16/R=2、H=65536/A=128、P=65536/NH=16777216 的新四方案轨迹与匹配诊断；DeepSeek 控制点见上方更新。其他 loop 规模仍待验证；Beauty 短轨迹不替代完整用户覆盖，同参数也不等于跨模型同计算 |
| **1.5 主要优化目标是什么？** | 相同 HBM/DRAM 硬预算下比较完整 loop 的逐请求延迟，分开首轮、后续复访命中和复访重建；淘汰后重建仍计入复访 | **指标方向已明确。** 固定 P/NH 实验比较 token 额度，不等于相同总 HBM 字节预算；首访比例 1/R 影响全请求均值，仍需分轮报告。真实服务的吞吐/到达目标待确定 |

## 2. SOTA baseline 与 Motivation

- [ ] 覆盖主要相关方案，讲清它们的取舍，并量化目标场景中的关键问题。

当前候选论证是：**全 HBM 的容量约束 → CPU DRAM offload 的取数等待 → dense prefetch
的冗余 → sparse fetch 的重叠困难**。这是待分析和验证的思路线索，各环节尚未整体成立。

| 条目 / 要回答的问题 | 当前理解 | 状态与具体缺口 |
|---|---|---|
| **2.1 主要 baseline 是否覆盖？** | motivation 目标集合为 HBM-only、ECHO、full prefetch、sync sparse loading、async sparse loading；full prefetch 搬运完整历史，attention 仍稀疏 | **两模型各有四类对照。** NOSA 当前 HBM/dense/sync/async 的正式测量、profile 与独立 API 均已验收；接入前的长尾诊断另保留来源。ECHO 尚未接入同一 NOSA 对照，不能拼接跨模型排名；SOTA 覆盖仍待补 |
| **2.2 全 HBM 的问题是什么？** | 固定复用距离下，实际 session 容量决定复访是否重建；workspace 本身也会改变可保留用户数 | **一个控制点支持。** DeepSeek C1024 的 ECHO/serial 可保留 16 用户且复访全命中，HBM 复访全重建；ECHO C2048 容量降到 15 后也全重建。其他用户数、预算与两者均命中的区间仍待覆盖 |
| **2.3 CPU DRAM offload 的问题是什么？** | NOSA 固定容量 loop 中，HBM 复访 0/16 命中，三个 offload 方案均为 16/16；保留历史避免了重新构建，但引入了首访分配、写回及复访取数成本 | **当前实现已测量。** HBM/dense/sync/async 复访均值为 2321.743/71.594/29.321/31.405 ms；三个 offload 的首访均慢于 HBM。容量保留收益与取数代价须分开，不能把完整差值归为串行等待 |
| **2.4 Dense prefetch 的问题是什么？** | 整层历史预取仍使用 sparse attention。NOSA 当前固定容量轨迹中，dense 的复访 candidate H2D 计数约为两种 sparse 的 7.283 倍，复访均值也高于同步 sparse | **该控制点有冗余搬运观测。** 这些计数是软件 payload，不是物理总线流量；流量、预取机会、非矩阵工作与调度成本尚未完成因果分解，不能外推为通用排名 |
| **2.5 与 ECHO 如何比较？** | ECHO prefill fetch 适用于后续 chunk 或已有 offloaded prefix 的 extend；无 decode 不排除适用。已复现层内共享 token cache，并单独标明 GR session 保留与容量/异步安全适配 | **本地对照成立，方法收益未成立。** 当前控制点选择 C1024/W1024；两次完整 ECHO 轨迹均慢于同配置 serial sparse，不能宣称融合预取加速。该结果不替代共同模型上的 NOSA/ECHO 比较 |

ECHO cache baseline 的实现与受控测量已完成。官方逐层共享 pool、实际领取才驱逐、
`indexer/prefetch → top-k → 主 KV append → exact recall` 的顺序及时间戳 priority
已核对并修正原本地差异。[实现与 chunk 选择计划](agents/system/echo_cache_implementation_plan.md)
覆盖真实前三层独立空前缀的 64K+1K 数值门禁、内存审计、三层 chunk 扫描及独立十 block
GR 对照。C1024 两次 ECHO 完整轨迹约 77–78 秒，C2048 约 128–130 秒；前者保住全部
用户历史，但冷请求和 p95 更慢。这是预定两轮负载的取舍，不是通用最优 chunk。
同 C1024 的 ECHO 比串行 sparse 慢约 15%，本轮没有建立融合预取的整体加速收益。
官方未覆盖的容量/异步边界与本地适配单独验收，不称为完整 SGLang serving 复现。

最新计算对照使用非 GR 的真实前三层 64K+1K，在同一 GPU、同一共享 cache 实现上测量；
MLA 算子利用率约 66%，resident extend 端到端利用率约 29%。每版内部的 resident/offload
及插桩检查均逐位一致；跨版本输出并非逐元素等价，稀疏选择也有变化，未测推荐质量。
具体延迟、输入、对照与数值边界见[计算报告](../experiments/deepseek_v32_echo_prefill/README.md)。
这些结果支持当前负载上的计算实现效率改善，不证明 cache 或方法收益。

## 3. Challenge 与 Design

- [ ] 能解释真实困难、设计与困难的对应关系，以及相对已有方法的技术区别。

| 条目 / 要回答的问题 | 当前理解 | 状态与具体缺口 |
|---|---|---|
| **3.1 Sparse fetch 为什么难以重叠？** | NOSA 当前固定容量测量的 96 个适用内部样本均未达到两种 90% 重叠门槛；无 host-copy 的样本不计为通过 | **A128 重叠门槛未满足。** 先前候选的 NCU 诊断提示 host 数据依赖和 consumer K/V 就绪等待，同二进制对照确认对齐可消除高于逻辑 payload 的 12.5% sector 访问；这些局部结果不能分解当前请求延迟，也未证明计算或 IO 主导 |
| **3.2 当前设计如何应对？** | NOSA 融合唯一历史读取与 sparse attention；固定 P/NH 专用路径使用有限逐层容量、逻辑位置直接映射和 session 身份标签，candidate 只在 GPU 临时执行 | **固定路径已实现，方法收益未成立。** 当前 async 在首访、复访和完整轨迹均慢于 sync。多个 Q128 候选未通过数值或重叠门槛，均未合入；通用 budget 共享 staging 仍不等于通用有限槽缓存或热点淘汰策略 |
| **3.3 技术区别和研究贡献是什么？** | 希望在减少搬运的同时获得重叠收益 | **待总结。** 与 ECHO 和其他方案的实质区别尚未建立；系统名称、最终技术划分和贡献表述均未确定 |

当前固定 P/NH 路径已将历史保留与 candidate 临时执行分开。P 限制逐层历史 HBM
容量，NH 是按页对齐的 host history 准入额度，按 session 实际分配；NH 数值不表示
已分配全部 backing，也不保证填满后满足 512 GiB DRAM。Dense 路径当前要求 H≤P
且 history/chunk 按 64 tokens 对齐。这是有明确限制的直接映射方案，不能称为通用
热点缓存或淘汰问题已经解决；通用 budget 路径的 backend 共享 staging 另有边界。

当前固定路径已完成完整 32 层、16 用户两轮的新四方案正式测量，128 份完整
candidate hidden 验收通过；匹配 profile 和独立 API 参考也已验收。三组证据绑定
同一运行源码，具体身份和限制见 [NOSA motivation](../experiments/nosa_motivation/README.md)
及[材料依据 S-026](agents/research-supervisor/sources.md#s-026nosa-固定-pnh-当前实现的完整测量与效率边界)。
正确性和测量完整性通过，不表示重叠或效率目标已经完成。

## 4. Evaluation

- [ ] 用整体比较、机制分析和消融回答前面提出的研究问题，并明确适用范围。

| 条目 / 要回答的问题 | 当前理解 | 状态与具体缺口 |
|---|---|---|
| **4.1 整体系统收益是什么？** | DeepSeek 的固定 P/NH 结果见上方修正；NOSA 当前已比较完整 loop、首访和复访，offload 的复访优势来自保留历史、避免 HBM 重建 | **容量与方法收益分开。** NOSA 固定 loop 的 sync/async 延迟比为完整轨迹 0.965545、复访 0.933633，均表示 async 更慢；没有建立融合取数收益。扫描实现已接入并补测，前期两轮原型对照的部分长尾改善仍不能证明普遍或独立净收益。最新 H64K 元数据复用只有两个请求的局部改善，尚未接入或验证完整 serving 收益。五类共同模型对照与场景质量仍待完成 |
| **4.2 哪些设计带来收益？** | 当前固定容量的独立矩阵/attention API 参考已验收；resident A1024 和冷稀疏并集 offload 算子报告分别提供局部证据；H64K prequeue 显示提交敏感，元数据复用的匹配对照支持约 0.1 ms 的局部改善 | **收益依赖输入与范围。** Resident native 的 full/extend 加速为 1.536×/1.689×；A1024 三层算子回放的融合 API 延迟下降 22.71%–29.34%，9 个内部样本双比率均过 90%。这些不证明 A128 serving 收益；prequeue 和元数据复用对照也未隔离可移除的 CPU 成本。Candidate 效率差距、剩余长尾和重叠失败仍待归因 |
| **4.3 结论适用于哪些条件？** | 固定容量控制点覆盖 SM90/Hopper、16 用户两轮、64K+128；NOSA 使用完整 32 层 checkpoint，DeepSeek 使用独立复制 block 的替身且另执行末 token LM head | **范围有限。** 当前 NOSA 固定容量证据是一条完整轨迹，匹配诊断选请求 0/16。固定容量、resident A1024 与 pattern 分别保留测量范围，不代表其他 loop 规模、完整物理字节预算、真实到达过程或 GR 质量 |

当前固定容量测量的 HBM request ID 0/16 都重建历史。完整请求 MFU 相对“独立
GEMM/BMM 与 FA3 API 中位数之和”对应的 MFU 比率为 100.227% / 102.021%，
candidate-only 比率为 75.118% / 73.599%。独立组合不是实际整请求或理论性能上限，
因此比率可以超过 100%。Candidate extend 为 16.203399/16.642680 ms，组合为
12.171712/12.248864 ms；差值包含参考未计入的必要非矩阵工作、host 工作、执行
相互影响及计时差异，不能直接当作 CPU launch 开销或可消除的浪费。长历史主导的
接近度不能证明 candidate 高效或计算/IO 主导。具体分子、分母和取样见
[当前 API 对照](../experiments/nosa_motivation/report/nosa_motivation_poolscan_sm90_20261004_01/api_comparison.json)。

H64K HBM 请求 0/16 的独立进程诊断中，提前提交相同 candidate 计算，使校验前
event 区间中位数从 14.721/14.713 ms 降至 12.918/12.905 ms，14 组配对均缩短。
80 份输出精确一致，两项预定的 1.05 倍中位数检查均通过。该干预支持测试减少
图输入拷贝与提交的具体实现；它没有隔离 CPU 成本，也不测普通 serving 收益。
Event 区间不含有限值决策、事务完成和 lease 归还；两个 repeat-5 样本在校验入口
之后仍比相邻同组样本多约 21–22 ms，其中一个来自 prequeue 组。扫描计数只标出
所在 candidate，未测扫描时长，不能据此归因。独立重建的 history 与额外检查副本
也使条件不同于正式用户轨迹。详见[预提交诊断](agents/system/nosa_hbm_prequeue_diagnostic.md)。

后续 H64K 外部原型把层间拷贝的提交移入已有计算图，保留 GPU 拷贝。请求 0/16
的普通完整 candidate 中位数分别从 15.416903/15.330098 ms 变为
15.258893/15.332270 ms，40 份保存的输出经独立重开均精确一致。该原型未取得
一致的中位数改善，暂不接入生产。基线长尾显著影响均值；扫描计数没有提供时长，
独立进程的 GC 与记录条件也不同，不能据此声称消除长尾或获得 serving 净收益。
原正式结果继续有效，详见[图内拷贝诊断](agents/system/nosa_captured_copy_pilot.md)。

后续匹配对照共用运行检查和初始化，分别在图外、图内提交同样的拷贝。图内模式
在两个请求中的 candidate 中位数分别少 0.257480/0.038759 ms；相对原始实现
只少 0.076418/0.015620 ms，尚不足以建立稳定净收益。66 份保存输出（含六份
初始构建输出）经独立核对均逐位一致。两个匹配模式的短循环都没有 filtered pool
调用，包括图外拷贝，因此不能把调用缺席归因于图内提交，也不能声称消除长尾。
前期对照见[匹配拷贝记录](agents/system/nosa_guarded_copy_control.md)。

恢复优化后，单次输入校验内元数据复用在请求 0/16 的完整 candidate 中位数上，
比匹配图内拷贝对照少 0.105163/0.106095 ms；整个外部原型比原始实现少
0.133753/0.125243 ms。66 份保存输出经独立重开均逐位一致。该结果支持这项
局部复用，但顺序进程、重复 candidate 和额外 history 检查副本仍不同于正式用户
轨迹，不能据此声称四方案 serving 加速、长尾消除或 MFU 目标完成。原型暂不接入，
详见[元数据复用结果](agents/system/nosa_copy_descriptor_result.md)。

当前准备的[indexer 执行 workspace 方案](agents/system/nosa_indexer_dispatch_plan.md)
保留原有打分和选块 kernel，在执行 lease 内复用选择输出与 normalizer，并减少
重复的临时分配和视图构造。候选尚未运行 GPU；是否采用取决于完整 candidate
的净收益、cache 生命周期与四方案验收，不能用分配次数减少代替性能结果。

接入前的 dense request ID 20 与两种 sparse request ID 22 长尾未做 GC/pool 插桩。
后续两次普通复跑重现这些位置，匹配探针在三处 candidate 中均捕获约 36 ms 的 pool
引用扫描；这段扫描不同于 GC collection 耗时。外部原型首轮将三个长尾各缩短约
14 ms，但仍比复访中位数高约 22–24 ms，48 个 offload 首访同 ID 比较中有 47 个变慢。
反序第二轮再次缩短三个长尾约 13.35–13.84 ms；offload 复访均值在两轮中都降低，
首访和典型请求仍有好有坏。这些顺序对照不能排除漂移，同一批请求也不是独立样本。

扫描实现现已接入并完成新测量。新正式轨迹中的 dense20/sync22/async22 仍是各方案
candidate 最大值，分别为 90.496/46.396/48.827 ms。本轮整 case 路由计数确认使用
native 扫描，但没有逐请求扫描区间，不能把旧探针的 36 ms 倒填为当前长尾的解释。
前期诊断与当前结果的身份、保留映射和限制见
[发布记录](agents/system/nosa_pool_scan_publication.md)。

接入前的完整 A128 indexer 在请求 0/16 的真实输入上复放了全部 32 层，wall 中位数分别为
4.792/5.058 ms，选择和 attention 输出精确一致。这是连续独立 API 调用，包含两次
score pass、有限值检查与选块，但不含模型层间的 staging；不能从完整 candidate
延迟中直接相减来计算可消除的开销。该时期的 Nsight graph/node 诊断均通过输出检查，
但 candidate 分别比普通观测中位数慢约 7.9%/26.4%，数据库还提示事件可能不完整，
因而没有建立完整 GPU 活动或空闲时间的证据。证据边界见
[S-026](agents/research-supervisor/sources.md#s-026nosa-固定-pnh-当前实现的完整测量与效率边界)。

旧通用 budget 热度短轨迹已作为完整范围退出，不再作为当前容量或 baseline 效率的
实验入口。该范围曾有有效局部结果，但未建立用户规模或物理容量结论；退出原因及
共享工具去向见[整理记录](agents/system/experiment_organization.md#retired-gr-serving)。

研究者将实验按目的分为四类：论文中的 motivation；检验 baseline 实现效率是否合理的
性能实验；用于后续设计、暂不属于论文主线的 sparse pattern 分析；检验自有设计性能的
microbenchmark。是否直接进入论文不是实验的唯一保留标准，目录合并或撤销应服从这些
目的。Baseline 性能检查须对应实际采用的实现和负载，microbenchmark 不替代系统收益。

数值正确性是四类实验共同的前提，本轮按独立验收、正式性能、侵入式 profile 整理入口，
已通过的相关实现与配置不必在每次性能重复中重跑完整参考对照。当前输出比较大多在
计时外，仍增加实验总耗时与样本间工作；其占比尚未单独测量。运行时 finite 检查、
allocator 校验与 numerical repair 另按实际作用评估，不能把改变后的执行语义称为
原实现的纯计时优化。入口拆分与 CPU 验证已完成，未运行新实验；新入口尚未完成
GPU 验收或性能测量。实现和验证范围见[整理记录](agents/system/experiment_organization.md)。

按用户授权，旧短轨迹已清理；请求构造、来源和内存审计已迁出，通用 budget 实现
与必要回归仍保留。Dense/sparse resident 报告迁入共同 baseline 入口，保持原运行身份。

## 当前最主要的卡点

| 卡在哪里 | 影响哪里 | 下一步建议 |
|---|---|---|
| **当前 ECHO 融合预取没有取得相对串行 sparse 的整体收益。** 采样复访虽命中 session，所需历史 HBM token 在预取前却全 miss，实际 H2D 总字节与串行方案相同 | 2.5、3.1、4.1、4.2 | **T-007：loop motivation 测量与归因**，区分预测/预取开销、取数重叠和用户历史保留 |
| **缺少可以直接采用的 sparse attention GR 场景。** 已有生成器和模型代码，但模型、数据与服务需求如何对应仍不清楚 | 1.1–1.5；影响 2.2 的容量动机和 4.1 的整体评测 | **T-002：场景构造方案比较**，先列出具体可行路线和各自能回答的问题 |
| **五类 baseline 尚未在共同模型上形成完整可比集合。** ECHO 与 async 当前分属不同模型入口，模型和输出范围也不同 | 2.1、2.5、3.1、3.3 | **T-003：共享资源接入与可比范围**，明确适配工作与可回答的问题 |
| **两模型固定容量结果仍只覆盖一个共同参数点。** NOSA 与 DeepSeek 的计算不同，其他 U/R、P/NH 及字节预算范围尚未验证 | 1.4、2.2、4.1、4.3 | **T-006：loop 数据与容量压力**，按实际容量选择有区分力的范围并保持完整轮次 |
| **NOSA 当前固定容量 loop 未显示 async 收益。** H64K prequeue 显示提交敏感，但 candidate 效率差距、剩余长尾和供数/流水线等待仍需归因；外部原型尚未形成四方案完整 serving 收益 | 2.1、2.3、2.4、3.1、3.2、4.1、4.2 | **T-007：H64K 效率与归因**，用完整 candidate 对照检验 indexer workspace 复用，并保持四方案正确性和测量验收 |

DeepSeek 当前有效结论以上方用户修正与[固定容量 motivation](../experiments/deepseek_v32_motivation/README.md)
为准，不恢复已撤回的 4 GiB/W/chunk 排名。NOSA 的
[固定容量结果](../experiments/nosa_motivation/README.md)、
[resident A1024 报告](../experiments/nosa_baseline_performance/README.md)与 offload 算子结果
分别保留来源和边界，不能混成同一实现或跨模型排名。旧 budget-serving 实验已退出；
candidate 效率、因果归因和物理预算证明仍有缺口。
具体下一步见[待办清单](roadmap.md)。

内部回查：[材料依据](agents/research-supervisor/sources.md)、[重要理解修正](agents/research-supervisor/updates.md)。
运行细节不作为这张表的研究主线。
