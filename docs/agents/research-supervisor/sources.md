# 研究状态的材料依据

> **2026-10-06 DeepSeek A128 更新：**当前真实三层结果见 S-032，默认已由 A1024 改为
> A128。旧 `deepseek_v32_echo_prefill` 导航统一指向 `deepseek_v32_mfu`；旧 `layers3`
> 报告已由 `four_methods` 替换，新链接只提供当前入口，不是旧 run 的证据。S-031 的
> 历史命令、receipt 与原始产物路径不改写；NOSA 结果仍按其原来源解释。
> 研究者要求暂缓 motivation 实验，C10 补测与容量后续审阅随之暂停；真实三层结果
> 不替代该执行路径。旧本地官方 ECHO 适配及其结果已删除；
> [官方 ECHO 入口](../../../experiments/deepseek_v32_echo_official/README.md)仅保留独立 SGLang
> 真实前三层的 performance-only 复现。ECHO 与 HBM-only 计时均已完成，数值验收未通过；
> 两种配置的容量不同，不作为等容量对照。

> **2026-10-05 统一框架补测：**该次结果由 S-031 记录。S-022–S-030 中的“当前”与
> “本轮”仍按各条记录日期理解，历史 run ID、源码、数值和命令不改写。已退休的报告
> 路径仅作历史定位；同路径的新报告不能反过来充当旧记录的证据。

> **2026-10-05 范围调整：**旧 `gr_serving` 短轨迹实验已整体退出，相关报告和运行产物
> 已删除。S-009、S-010、S-027 及其他条目中的旧路径、验收和保留说明记录当时状态，
> 不表示产物仍可用或存在待补测任务。当前范围见[实验索引](../../../experiments/README.md)。

> **2026-10-03 撤回说明：** 本文涉及的旧 DeepSeek 4 GiB / W / chunk 对照已按用户要求撤回，
> 相关实验源码与运行产物已清理；下文仅保留当时的工程过程，不再证明当前容量或性能。
> 当前入口为[固定 P/NH 容量实验](../../../experiments/cache_management/README.md)。

> **工程来源导航：**已被替代的过程文件以 `934485b:<原路径>` 标注，可从该 Git
> 版本回查。S-021 所记旧内存验收的“11 个配置 / 352 个请求 / 172 个阶段”原始产物
> 已不在所记路径；这些数字只保留历史范围，不表示目前可重新核验。独立共享资源验收的
> 原始索引另存于[验收目录](../acceptance/unified_runtime_20261005/shared_cache_integration_evidence.json)，
> 它有自己的冻结源码与覆盖范围，不能替代缺失产物。

更新：2026-10-06。S-001–S-007 记录初始仓库材料；其旧算子结果未在本轮重新测量。
S-008–S-010 记录本轮实际完成并审计的 GR serving 实验及用户约束；S-011–S-012 记录
用户修正后的负载候选与数据可用性检查；S-013 记录热度抽样选择，S-014 记录本轮
MFU/cache 与 baseline 判断修正。最新选择见 S-019：固定用户 loop 替代热度 IID；
S-020 记录当时 cache 局部实现与公共系统验收的边界；S-018 是共享 cache 下的计算对照，
S-021 补入本轮已验收的 ECHO 受控 loop、内存证据及 chunk 选择。
S-022–S-025 增补 NOSA 固定容量原版实测、最新源码正确性、Q128 负结果及 A1024 补测。
S-029 记录四类实验整理、旧范围退出与独立验收入口的实现边界；S-031 记录统一框架
补测、残余开销与物理分配边界，不覆盖历史条目的原始身份；S-032 记录当前 DeepSeek
A128 四方法 MFU、局部 dense 重叠及依赖实验补测边界。
其余历史条目保留各自当时的范围。本次增量同步既有证据，未重新核验全部论文或 SOTA。
本文件供 agent 回查；给人的[研究状态](../../status.md)和[下一步任务](../../roadmap.md)在 `docs/`。

## S-001：研究者当前输入与实验室流程

- **来源：**本任务中研究者关于呈现分层、非线性推进、可编辑状态表、设计先行和 CPU DRAM 范围的说明；
  [实验室 Research.md 副本](../../../skills/research-supervisor/references/Research.md)。
- **对应：**四环节结构及全部当前候选解释。
- **支持：**用户希望以已有 sparse KV fetching/attention overlap 设计寻找场景；GR serving 仍待完善；
  ECHO 是重点对比；项目先关注 HBM/CPU DRAM。
- **边界：**“容量限制—offload 等待—dense 冗余—sparse 重叠困难”是用户提出的研究线索，
  不因写入状态表而成为测量结论。初次未选择模型/数据路线；本轮执行对象的明确选择见 S-008，
  仍不等于确认真实推荐模型和数据路线。

## S-002：GR 请求构造工具

- **来源：**[GR/README.md](../../../GR/README.md)，重点是开头、请求内容、热度/到达和适用边界说明。
- **对应：**1.2、1.3、1.4、4.3。
- **支持：**有固定历史、候选变化、长度预算、热度和模拟到达时间的生成工具。
- **边界：**默认 synthetic，catalog 模式也使用合成行为与候选；没有真实下一商品标签。
  生成工具不能证明真实推荐任务、质量或线上负载代表性。

## S-003：本地 serving 的能力

- **来源：**[serving/README.md](../../../serving/README.md)，当前范围及请求生命周期。
- **对应：**1.4、1.5、2.2、4.1。
- **支持：**原 `serving.runner` 仍逐请求创建释放 cache；本轮新增 `serving.persistent` 与
  `cache.prefix_pool`，按全局用户 LRU 跨请求保留固定历史，双预算准入、执行后截短，
  以访问次数区分首访与复访。独立入口为 `serving.run_multi_user`。
- **边界：**没有 batch scheduling、到达时间回放或网络服务；生成的 QPS/时间戳不等于
  已运行吞吐。whole-session LRU 不等于 NOSA 已有有限 token/page slots。测量依据见 S-009。

## S-004：NOSA GR 长上下文工程实验

- **来源：**[NOSA resident baseline](../../../experiments/nosa_mfu/README.md) 的 dense 部分，
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

**后续修正：**研究者指出 ECHO MFU/cache 策略与 DeepSeek MFU 存在问题，见 S-014。
旧延迟、MFU 和归因不能用于认定 baseline 有效或判断 ECHO 方法优劣。

- **来源：**[DeepSeek MFU 当前入口](../../../experiments/deepseek_v32_mfu/README.md)，
  报告状态、测量边界和已验收结果。
- **对应：**2.1、2.3、2.5、3.3、4.1。
- **支持：**有 DeepSeek V3.2 61 层的 ECHO 相关实现，indexer 内融合 prefetch，CPU DRAM
  backing、有限 HBM pool 与 residual recall。旧报告显示降低主 KV 的 HBM 分配，同时增加单请求延迟。
- **边界：**2026-10-01 gather 对齐修复后完整模型性能待补测；旧报告只对应其版本与输入。
  该路径为单请求执行器，不代表 continuous batching serving。它与 NOSA 单层回放的模型、
  执行范围和硬件安排不同，不能直接形成设计优劣排名，也不能用它断言 ECHO 在所有场景都慢。

## S-007：并行整理的系统工程与 KDA 材料

- **来源：**提交 `1971047` 的材料，现整理为工程背景（Git `934485b:docs/agents/system/research-context.md`）、
  实现状态（Git `934485b:docs/agents/system/implementation-status.md`）、候选实现方案（Git `934485b:docs/agents/system/implementation-roadmap.md`）
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

## S-008：本轮用户选择与执行范围

- **来源：**本轮用户要求实现 GR serving，先 DeepSeek V3.2 前三层及对应输入独立复制到约 8B，
  不用 MoE；之后完整 NOSA-8B。后续指定保留 1/8/32 并加入 64/128/256/512 用户，
  按热度控制复访、每用户最多 8 次，固定历史 4K/16K/64K 分开运行，时长约 30 分钟。
  执行契约（Git `934485b:docs/agents/system/gr_serving_task.md`）、最终审查（Git `934485b:docs/agents/system/gr_serving_review.md`）。
- **对应：**1.2、1.4、1.5、2.1、2.5、4.1、4.3。
- **选择与假设分开：**用户明确选择七档用户、最大复访数和三档历史。每档约 30 分钟、
  128-token candidate、4 GiB / 16 GiB 缓存配额，以及 64K 每档请求上限 6，属于执行者
  为控制时长所作并已告知的安排；关于时长是否三档合计的可选澄清未获回答，不能写成用户确认。
- **边界：**4K/16K 每档上限 32，N=1 因复访上限实际 9；64K 每档 6。
  热度为 Beauty interaction_count、seed 42；不强制首访或补齐复访。DeepSeek 是
  7,827,793,408 参数的 checkpoint 工作负载替身，NOSA 执行完整 32 层但无 LM head；
  执行对象不等于经过验证的 GR 任务模型。

## S-009：三档固定历史的完整串行 serving 测量

**后续修正：**用户否定其名义人数扩展与零 miss 的容量实验意义。以下记录仍是原轨迹的
数值/计量记录，不能支持用户规模扩展或作为新容量主实验；当前设计修正见 S-011。
另据 S-014，ECHO/DeepSeek 基线本身待修正，其旧性能排名暂停采用；原预算审计只核对
当时接口申报与边界采样，不证明临时 cache scratch 的完整峰值已被覆盖。

- **来源：**旧 GR 实验（已结束）（Git `934485b:docs/agents/system/experiment_organization.md`）、
  独立验收记录（Git `934485b:docs/agents/system/gr_serving_review.md`），以及报告中各历史档的 metadata、audit、
  summary、per-request 和源码 manifest。原始记录在
  `experiments/gr_serving/output/data/<run_id>/`，未复制到 docs。
- **Run ID：**`gr_serving_h200_20261002_h4k_01`、`gr_serving_h200_20261002_h16k_01`、
  `gr_serving_h200_20261002_h64k_01`。分别 56/56/56 组、1608/1608/336 请求，
  程序总时长 8.258/31.983/32.797 分钟；全部 2664 次非 HBM 输出比较逐位一致。
  4K 源码 digest 为 `67f91c34ead43d4056500661caa1c67e8ef8a50bc25327e081a0561bbab43060`；
  16K/64K 为 `45511acf6b7ec0e0a9bf3e21a4f9b1c9b74f88a46f03303f3f93cc9460dfd44c`。
  两者仅 profile 诊断入口及其测试不同，4K profile 记录了允许的差异；性能代码未变。
- **对应：**主条目 2.2；同时影响 1.4、1.5、2.1–2.5、3.2、4.1–4.3。工作用于区分
  “容量导致复访重建”和“offload 命中执行开销”；若相对 HBM 的全部请求仍变慢，不能将
  相对串行 sparse 的改善表述为相对 HBM 的整体收益。
- **支持：**16K NOSA HBM/offload 准入容量为 7/31 个 session；HBM 有 108 次淘汰、
  63 次复访中的 13 次 miss，三种 offload 均无复访 miss。N=64 的复访均值为
  HBM 187.88 ms、serial sparse 41.15 ms，但全部七档的全请求均值仍是 HBM 最低。
  DeepSeek 16K HBM 有 34 次淘汰，但未导致复访 miss。4K 与 64K 的所有实测复访均命中。
  Dense 预取并不总是慢于 sparse。NOSA overlap 相对 serial sparse 的全请求均值在
  4K/16K/64K 分别 6/7、7/7、7/7 档下降，合计 20/21；复访均值仅分别 2/7、1/7、0/3
  个非空档下降，未形成稳定复访优势。两者不能互相替代。
- **不足以支持：**H200 全物理显存耗尽、512 个实际活跃用户、真实推荐质量、并发吞吐、
  ECHO 与 NOSA 跨模型设计排名，或普通 activation 的隔离峰值。配额包含缓存附属开销，
  权重/输出 payload 与进程 allocator 峰值分开报告；首次 NOSA case 峰值含模型切换时的旧权重。
  64K 四档没有复访，其他档仅 5/2/1 个；4K/16K 大用户档也只有 4/2/3 个。
  不同请求上限、DeepSeek slots 和首访比例使跨历史总均值不是单变量比较。
- **验收：**原始数据、数值记录、真实 CPU 参考张量、heat/cap 重放、请求身份、逐请求 LRU、
  双预算、源码/依赖和报告重建均通过独立审计。失败和中止运行未作为有效实验保存。

## S-010：实际 GR 复访中的 NOSA 内部区间

- **来源：**上述报告的三档 profile metadata/analysis；run 为
  `gr_serving_h200_20261002_h4k_profile_02`、`gr_serving_h200_20261002_h16k_profile_01`、
  `gr_serving_h200_20261002_h64k_profile_01`。使用已保存正式请求，所有层执行，
  只采 layer 31；4K/16K 为 population 8 的请求 2/3/7，64K 为 population 1 的请求 1/2/3。
- **对应：**主条目 3.1；同时影响 3.2、4.2、4.3。用于核对局部重叠证据能否外推到
  128-token candidate 的 serving，不能以完整 kernel 窗口替代实际工作区间。
- **支持：**唯一读取与完整 hidden 正确性通过。Page-envelope 与 stripe-copy 两个比率
  在每个样本数值相同；4K 为 72.87/74.34/75.24%，16K 为 78.37/79.83/81.21%，
  64K 为 76.87/73.18/79.27%。全部九个样本低于 90%，作为有效负结果保留。
- **边界：**交集只针对实际 consumer softmax 区间，不代表完整 attention 隐藏率或物理链路
  字节数；serial 未采集对应 math 区间，比率为 null。未把诊断开销混入正式 serving 延迟。
  这些输入不同于旧 64K+1K 单层实验，不撤销其有效结果，也不沿用它的 90% 主张。

## S-011：实际复访用户与容量压力的协议重构

**后续范围：**用户最终明确采用 S-013 的独立有放回热度抽样。以下 4N 配额方案已撤回，
不作为当前对照或待实现任务；没有相应 GPU 结果。

- **来源：**用户本轮两次修正；候选协议（Git `934485b:docs/agents/system/gr_serving_workload_redesign.md`）、
  可行性分析（Git `934485b:docs/agents/system/gr_serving_workload_feasibility.md`）。本轮只做 CPU 调度/LRU 预演和
  基于原数据的成本估计，没有模型或算子性能测量，不向 experiments 添加新结果。
- **对应：**主条目 1.4；同时影响 2.2、4.1、4.3。要澄清的是：N 是否对应实际复访用户，
  以及同一请求序列是否同时覆盖 HBM 与 offload 的保留容量，而非是否能产生几个命中请求。
- **候选规则：**期望复访数为 `clip(lambda * heat_weight, 1, 8)`，调整 lambda 使总和为
  3N，按最大余数整数化，首访加复访构成 4N 个请求并全局打散。实际用户数和复访用户数
  都等于 N，首访比例固定 25%，各用户次数仍随热度变化。上下界和整数化改变原始热度，
  必须报告生成后的频次，不能称为真实访问流量回放。
- **压力依据：**固定历史、等大小 session 的 LRU 中，复访前的不同其他用户数 D 满足
  `D >= C` 时容量 C 会 miss。NOSA 16K 原配额对应 C=7/31；seed 42 的 CPU 预演中，
  N=16 的 48 次复访有 HBM/offload miss 21/0，N=64 的 192 次为 167/84。
  这只是调度与准入模型预测；新模型运行尚未验证它们。
- **边界：**不挑选有利 seed，不仅补齐首次访问，不借离线快照免除重建成本，不把 CPU
  命中率或旧均值推算的时间当作 serving 实测。旧七档全覆盖的 8008 次首访/历史只是
  请求计数下界；耗时估算还取决于真实 miss、模型初始化和计量开销。

## S-012：从已有行为数据采样复访

**后续范围：**真实时间戳路线未被采用；用户选择直接使用已有热度，见 S-013。原始日志
不再作为构造请求流的前置条件。

- **来源：**用户询问“这个复访能否用已有的数据集来采样”；
  数据轨迹核对与协议（Git `934485b:docs/agents/system/gr_serving_dataset_trace.md`）、
  [已有热度说明](../../../GR/analysis/README.md)、`GR/heat.py` 与 `GR/dataset.py`。
- **对应：**主条目 1.4；同时影响 1.3、2.2、4.1、4.3。数据来源与跨用户顺序决定复用
  距离，边际热度不能替代时序证据。
- **已核实：**Amazon timestep_map 是用户内部交互排名，当前热度加载只累加次数；
  industrial 来自合成生成流程。原始文件所记 `/mnt/nfs/.../Bi-KV/data` 当前不可访问，
  限定相邻项目检查未找到替代源，不能断言原 pickle 没有其他字段。
- **公开候选：**[天池 UserBehavior](https://tianchi.aliyun.com/dataset/649) 官方页面元信息
  确认是淘宝行为数据、用于隐式反馈推荐。正文和元数据接口访问受限，原始字段、日期范围、
  下载可用性本轮未核验；没有导入数据或完成回放。具体数据集尚未由用户选定。
- **建议：**固定时间窗及事件映射，从真实复访用户中抽 N 人，保留原全局顺序，每人最多
  首访加 8 次复访；总数允许 2N–9N，不再强制 4N 或 25% 首访。先核对采样后的不同
  用户复用距离及两类 miss，再按时间预算分批执行完整 GPU 点。
- **边界：**用户筛选、同时间事件合并和封顶会改变流量及复用距离；不能按 miss 挑选窗口。
  行为是请求代理，固定 history/candidate 内容仍需单独适配；原时间只用于排序时不测
  原 QPS 或排队。此次只有数据核对和协议讨论，没有真实 trace 统计或新模型结果。

## S-013：用户明确简单热度概率抽样

**后续范围：**2026-10-03 研究者改为固定用户多轮顺序 loop，见 S-019。本条保留原选择，
不再作为当前 motivation 负载要求；热度文件与已有工具不因此失效。

- **来源：**用户要求用现有热度构造分布、取消最多 8 次限制，并明确“从用户访问次数
  转化成概率分布，再随机采样”。当前协议（Git `934485b:docs/agents/system/gr_serving_workload_redesign.md`）、
  `GR/heat.py` 与 `GR/scheduling.py`。
- **选择：**p_i=c_i/sum(c)，给定 T 次独立有放回抽样；实际用户数、复访用户数和
  r_i=max(n_i-1,0) 均为输出。不强制每用户出现或复访，不分配 4N 配额。
- **已有材料：**累计曲线 C(x) 可按 p_i=C(i/N)-C((i-1)/N) 近似展开。N 是底层池
  大小，现有 Beauty 统计可提供 22,332；不能把该维度当本轮实际用户数。
- **改动：**测量和单模型入口默认无复访 cap，manifest 补充 returning_users 与逐用户
  revisits，独立审计兼容 None；原有 weighted 抽样算法保持。新配额实现已撤回。
- **边界：**曲线近似不恢复逐用户身份与真实时序；IID 序列可能复访少或接近全 miss，
  按完整固定种子结果分析，不据此重挑样本。没有新增 GPU 测量或性能结论。
- **对应：**1.3、1.4、2.2、4.1、4.3；T-006 根据总请求数、实际人数和预算重新选择实验点。

## S-014：ECHO/DeepSeek MFU 与 cache 策略修正

- **来源：**研究者本轮明确指出 ECHO 实现的 MFU 和 cache 策略有问题，因此 baseline
  对比不太正确；DeepSeek 当前实现的 MFU 也不太正确。代码核对与边界见
  系统状态（Git `934485b:docs/agents/system/implementation-status.md`）、三层诊断（Git `934485b:docs/agents/system/deepseek_echo_three_layer_profile.md`）
  和 GR 审查（Git `934485b:docs/agents/system/gr_serving_review.md`）。
- **判断修正：**此前“工程对照已建立”“模型内可比”过强，改为接口与原测量记录已有，
  baseline 有效性待修正。数值一致、计数/归因自洽及相同预算配置均不能替代计算实现、
  MFU 解释和 cache 策略的审计。受影响性能排名不得用于认定 ECHO 设计局限。
- **代码事实与未知分开：**ECHO 为预取主动回收槽位，serial sparse 按实际缺失量回收；
  在默认 4096 slots / limit 8192 下，前者会回收当前 chunk 外的全部槽。临时 mapping、
  排序和 remap scratch 未在保留 tensor 统计中显式覆盖。两项是审计起点，不是已测定的
  全部根因或开销占比；本轮也未证明某个 FLOPs 公式必错。
- **保留范围：**旧 run ID、数字、报告和产物保留至修正复测验收后替换；数值记录仅支持
  原实现及输入的一致性。独立 NOSA 数值与局部 overlap 证据不因本次修正自动失效，
  其真实 serving 容量与复访收益仍受旧负载边界限制。
- **补测状态：**16 用户两轮顺序访问的 run 01 因源码身份检查失败，没有有效替换结果；
  run 02 仅有计划。本轮只核对文档和代码、运行 CPU 回归并整理提交，没有修复 MFU/cache
  或新增 GPU 性能结果。
- **对应：**2.1、2.4、2.5、3.3、4.1；新增优先任务 T-008，T-003 的设计比较以可信基线为依据。

## S-015：ECHO prefill fetch 与 GR 生命周期策略

- **来源：**研究者猜测 prefill fetch 用于 chunked prefill，并询问只有 prefill/extend
  的 GR offload 策略。[原论文](https://www.usenix.org/system/files/osdi26-liu-guangda.pdf)
  §3、§5.2、§6.1、§6.4.3；上游 ECHO commit `bc1b75c1000010d0ac6f032ebaac283255c050b1`。
  项目依据为 `serving/persistent.py`、`cache/prefix_pool.py`、
  `models/nosa/serving.py` 与 `cache/host_backing.py`。具体源码位置和推理见
  [生命周期分析](gr_prefill_extend_offload.md)。
- **已核实：**chunked prefill 是典型用途，但已有 host prefix 的单次 extend 同样
  可使用 ECHO inter-query。它跨同次调用的 Q blocks 隐藏历史读回，新 KV 直接写 HBM
  而无需先从 host 取回；精确调用顺序由 S-016 补充，主 KV append 位于 prefetch 之后。
  没有历史 miss 则不需 fetch。PD 专用 P 实例禁用 offload 不能泛化到所有 prefill。
  原论文 PD 主吞吐使用预计算 KV，不提供这条 prefill 路径的端到端收益证据。
- **候选建议：**将冷历史构建、跨请求历史保留和 candidate 取数分开选择策略；预算内
  resident build、DRAM 历史与 HBM 热点分级、candidate 请求内临时状态均待评估。
  比较整层提前预取、整批稀疏并集串行 fetch 和重叠，以实际并集与总延迟判断。
- **当前边界：**整用户 LRU 会释放两级资源，未实现独立冷热迁移；NOSA 每 session
  保留派生记录及 full-address staging，候选目前仍写 host 后再 truncate。
  建议不是用户已选择或已经实现的策略。本轮无代码修改或新测量，不恢复 S-014 中
  待修正 baseline 的有效性，也不改变原实验报告。
- **对应：**1.1、2.2–2.5、3.1–3.2、4.1–4.3；细化 T-003/T-007，保留 T-008 优先级。

## S-016：官方 ECHO cache 语义与复现计划

- **来源与范围：**研究者要求先只考虑 ECHO baseline、结合官方 SGLang 制定 cache
  实现与 chunk 选择计划，指定放在 `docs/agents/system/`。交付为
  实现计划（Git `934485b:docs/agents/system/echo_cache_implementation_plan.md`），含固定 commit 的源码位置。
  只读核对，未实施 cache 改造、运行 chunk sweep 或改变旧性能报告。
- **已核实：**官方每层共享 pool，使用全局 host IDs；indexer/prefetch 和 exact top-k
  先于主 KV append，随后 guaranteed recall/MLA。预取在实际 miss claim 时才驱逐，
  priority 为按事件写入的时间戳，`EARLY_EVICT` 关闭；不是频率计数。
  本地先写 host 再读新 KV、预先清 slots、每 session 独立 pool 与之不同。
- **正确性边界：**官方 extend 的部分空池处理、超 pool 选集与异步 host 写依赖需
  明确限制或补足；本地 query 消费拆分不是官方能力。计划将正常域对齐与这些适配
  分开验收，不以复现为由照抄潜在错误，也不加入新 GR policy。
- **Chunk：**论文写 2048，artifact mixed 脚本为 -1，伪 PD P 节点为 16384；不是
  同一 offload 条件的结果。当前 DeepSeek serving 只在 attention 内分块，模型级
  outer chunk 还需实现并协调 GEMM 形状变化。计划以 2048 起步、保留 1024 控制，
  同预算扫描 256–4096；128 candidate 保持真实长度，最终 C 尚未测定。
- **对应：**2.5、2.1、2.2、4.1、4.3；T-008 具备实施契约，上一轮广泛策略建议暂不展开。

## S-017：ECHO 与 NOSA/dense 的 cache 复用边界

**后续进度：**以下为提出建议与计划时的状态。NOSA 共享执行资源现已有实现与局部验证，
见 S-020；公共 serving/完整硬预算与新性能结果仍待完成。

- **来源：**研究者说明另一 agent 正在修改 ECHO，询问 NOSA sparse async loading 与
  dense prefetch 能否复用。本轮只读核对工作区中的 shared serving 契约、PrefixPool、
  ECHO 共享池及 NOSA offload/serving；详细来源与建议见[复用边界](cache_reuse_boundaries.md)。
- **已核实：**ECHO 目标是 backend/device 持有逐层有限 pool；NOSA 当前仍由每 session
  持有一层 sparse 或两层 dense staging，前者只有逻辑地址 staging、没有有限 slots。
  新 shared plan/准入接口已出现在工作区，但 ECHO 仍在开发，本轮未独立验收。
- **建议：**复用资源规划、预算、所有权和异步生命周期，优先将 NOSA 执行 workspace
  改为 backend 共享；serial sparse/overlap 保持同一缓存条件，dense 保持完整历史预取。
  NOSA 有限页缓存需要适配 head/block 布局、物理寻址和消费者协议，单独评估。
- **研究边界：**共享 staging 不等于保留热块；改变资源所有权可能改变容量与排名，
  不能将影响全归于 overlap。建议未被选择或实施，没有新性能结果，不改变旧报告。
  对应 2.1、2.4、2.5、3.2、4.1、4.2，细化 T-003，保留 T-008 优先级。
- **后续计划交付：**按研究者要求制定修改计划（Git `934485b:docs/agents/system/nosa_shared_cache_implementation_plan.md`），
  明确 NOSA 主线与 ECHO 接口冻结后的 DeepSeek dense 批次、共享 scratch 上限、
  backend 生命周期、完整 checkpoint 验收与受影响实验替换。新增源码核查发现 dense
  device-only FA3 scratch 需显式覆盖，计划让 dense/hbm 使用同一预留机制。
  本轮仍仅交付计划，未修改执行路径、运行测试或补测，不恢复原 baseline 有效性。

## S-018：当前共享 cache 下的前三层计算对照

- **来源：**[实验报告](../../../experiments/deepseek_v32_mfu/README.md)及
  非矩阵优化验收（Git `934485b:docs/agents/system/deepseek_nonmatrix_optimization.md`）。本轮 candidate 为
  `20261003_echo_layers3_nonmatrix_candidate_02`，重新测量的 control 为
  `20261003_echo_layers3_nonmatrix_control_01`；替换此前共享 cache 改造前的计算结果。
  实现身份以各 run 的源码快照为准，不将当前工作区整体视为该次测量版本。
- **执行与对照：**非 GR workload 仅真实第 0–2 层，64K prefix + 1K extend，包含
  embedding、三个 dense block、final norm 与末 token LM head。两版在同一物理 GPU 5
  的 H200、同一共享 cache 实现和预算条件下运行；矩阵后端仍为官方 DeepGEMM main /
  FlashMLA。本轮变更包括 FlashInfer 非矩阵适配、移除 Hadamard 与量化优化。
- **支持：**resident prefix/extend 中位延迟由 1176.202/24.300 ms 降至
  850.664/18.344 ms；offload 由 2025.688/51.511 ms 降至 1773.963/36.245 ms。
  MLA 算子利用率约 66%，resident extend 端到端精度归一化利用率为 29.44%；
  kernel 与完整阶段的指标不能混用。单请求、默认时钟，不提供总体置信区间。
- **数值与语义边界：**两版各自八项 resident/offload 及插桩检查均逐位一致，新版三层
  全部 MLA 输出通过独立 FP32 参考。跨版本 hidden/logits 均有元素超原容差；真实传播
  下三层选集平均交集占 2048 slots 的 98.56%/98.86%/94.21%，没有完全相同的 query
  选集行。这描述整个变更集的差异，不能归因单项。移除量化前旋转不保证逐元素等价，
  argmax 相同也不证明任务质量；未放宽容差或测量推荐质量。
- **系统边界：**相同 cache 代码不代表相同搬运工作量，选集变化也改变 cache 行为。
  这些结果支持当前输入上新计算路径的延迟改善，不证明 cache 优化、ECHO 方法收益、
  完整 61 层、GR loop、完整硬预算或 chunk 选择已验收。对应 2.1、2.5、4.1；
  T-008/T-003 的系统验收依赖与原系统排名暂停采用的判断不变。

## S-019：固定用户 loop 与五类 motivation baseline

- **来源：**研究者本轮明确固定 user 数量、重复顺序遍历多次，不再需要热度数据；
  用于 HBM-only、ECHO、full prefetch、sync sparse loading、async sparse loading
  的 motivation 实验，并说明 cache 系统仍在实现、相关数据尚未出来。
  规则与交接见 [loop_motivation.md](loop_motivation.md)。
- **选择与建议分开：**固定 U、同序 R 轮及五类 baseline 是用户选择。具体 U/R、
  长度与预算组合未定；按实际容量跨越命中/miss 区间、保持各规模相同 R 是后续建议。
  此前 U=16/R=2 的失败尝试不能当新配置已确定或实验已完成。
- **只读核对：**`GR/scheduling.py` 已用取模支持 sequential；
  `experiments/gr_serving/src/workload.py` 为该模式构造显式用户 ID、不读热度，
  并检查固定 history 与变化 candidate。`measure.py` 可传 sequential/users/requests，
  尚无 rounds/完整轮数约束，各用户规模共用 requests。普通 `serving.run_multi_user`
  仍硬编码 weighted。既有双遍测试代码存在，本轮未运行。
- **可推导而非实测：**完整 loop 共 UR 请求、U 首访、U(R−1) 复访、复用距离 U−1。
  等大小 session/严格 LRU/固定容量且无额外失效时，U≤C 的后续轮全命中，U>C 全 miss。
  这是 session 层预测，不推断 ECHO token 命中率；实际 C 须计入共享资源、session
  与 pending 开销并在新系统核验。
- **比较边界：**当前 DeepSeek 为 `hbm/echo/serial_sparse/dense_prefetch`，NOSA 为
  `hbm/serial_sparse/dense_prefetch/overlap`；五类尚未在同一模型全部接齐。Full prefetch
  仅改变历史搬运范围，attention 仍稀疏。不能通过跨模型延迟拼接得到方法排名。
- **研究含义：**规则简化了用户覆盖与复用距离，能推进受控容量和执行开销实验；
  不证明真实 GR、热度或到达规律。没有新 GPU 数据或性能结论；不替换旧报告与产物。
- **对应：**1.3–1.5、2.1–2.5、4.1、4.3；T-006 改为 loop 构造，T-007 负责新 motivation
  测量与归因，T-003 明确模型内比较域，T-008 的 cache/系统验收仍是实验依赖。

## S-020：cache 局部实现与系统验收边界

- **来源：**ECHO core checkpoint（Git `934485b:docs/agents/system/echo_cache_core_checkpoint.md`）、
  native checkpoint（Git `934485b:docs/agents/system/echo_cache_native_checkpoint.md`）、
  NOSA 总检查点（Git `934485b:docs/agents/system/nosa_shared_cache_checkpoint.md`）、
  workspace 检查点（Git `934485b:docs/agents/system/nosa_shared_workspace_checkpoint.md`）、
  dense 检查点（Git `934485b:docs/agents/system/nosa_dense_staging_checkpoint.md`）和
  公共入口审计及后续记录（Git `934485b:docs/agents/system/nosa_shared_entrypoint_audit.md`）。本轮只读这些材料
  并少量核对当前 `serving/persistent.py`、`measure.py`，没有独立重跑其测试。
- **ECHO 已有局部证据：**共享 host IDs、逐层有限 HBM pool、session 视图、实际领取
  驱逐、exact recall 与异步 append 生命周期已有实现/检查。Core 最新附录记载真实
  checkpoint 前三层 64K+1K，resident/offload 各自独立空 cache 构建 sparse prefix，
  全部 extend hidden 与 logits 逐位一致。这是数值验证，不是 loop serving 性能。
  Implementation checkpoint 仍列该数值项待做，当前以 core 的具体完成记录限定其状态。
- **NOSA 已有局部证据：**backend 共享 sparse 单层 workspace、dense 双 staging 与
  hbm/dense FA3 scratch 已实现，含 C/A/Q plan、执行 lease 与 session 生命周期检查。
  总检查点记载完整 32 层、四方案、两个独立用户及交错复访，在 64K+1K 与 64K+128
  的全部 candidate hidden 与 HBM 逐位一致。Profile 迁移已有代码与 CPU 检查；
  新 GPU profile 未发布。总检查点尚列旧 profile 待办，需结合入口审计的后续记录阅读。
- **系统缺口：**当前公共 runner 仍无准入 owner 绑定；measure 的 A 规划及 scheme
  更换生命周期未完整接入。独立 backend 的 hooks、资源测试不能替代公共 runner 的
  双预算与异常清理验收。稳定源码下的全局回归、完整预算/瞬时分配审计、ECHO chunk
  选择、DeepSeek dense 共享适配和新 loop 正式结果仍待完成；不能称全局测试已通过。
- **研究边界：**修正“NOSA 尚未实施”“staging 仍按用户持有”“共享计划无执行验证”
  的过时表述，但不升级为系统 baseline 已成立。NOSA 仍没有有限 HBM page/token slots
  或淘汰。共享所有权会改变实际容量，原数字不能代表新实现；旧报告按替换规则保留。
- **对应：**2.1、2.4、2.5、3.2、4.1、4.2；T-008/T-003 聚焦剩余集成和系统验收，
  已完成的局部工作不重复排成待实现。

## S-021：ECHO 共享 token cache、受控 loop 与默认 chunk

- **来源：**本任务实际完成 实现计划（Git `934485b:docs/agents/system/echo_cache_implementation_plan.md`） 的
  代码、数值与性能执行，证据分别见 冻结验收（Git `934485b:docs/agents/system/echo_cache_freeze_gate.md`）、
  内存账本（Git `934485b:docs/agents/system/memory_acceptance_summary.json`）、
  独立选值评审（Git `934485b:docs/agents/system/echo_default_independent_assessment.json`）、
  正式选择记录（Git `934485b:docs/agents/system/echo_cache_default_review.json`）及
  [ECHO cache 实验](../../../experiments/cache_management/README.md)。原始数据仍在实验 output 中。
- **源码和范围：**非 GR 的真实 checkpoint 0–2 层顺序传播，独立空 cache 构建 64K
  prefix 加 1K extend，全部 extend hidden 和末 token logits 逐位一致；不跑 61 层。
  三层 chunk 扫描另用 64K+128，接受 C256/1024/2048，C512 数值不通过、C4096 超预算。
  GR 单独使用十个独立 dense block 的 source-input replay，包含 candidate 全 hidden
  和末 token LM head；不能称为训练后的 DeepSeek 8B 或完整模型性能。
- **正式数据：**首轮 `20261003_echo_gr_chunks_01_*` 五个配置加
  `20261003_echo_gr_chunks_repeat_01_deployment_c1024/c2048` 两次重复，共七 run、
  28 个方案配置、896 个请求。C2048/W2048 在两协议中复用同一物理 run，不增加重复数。
  四方案全部输出与同配置 HBM 逐位一致。每条轨迹 16 用户顺序两遍，64K+128、seed42、
  NH1050624/P32768、HBM4GiB/DRAM64GiB、H200/SM90 GPU0 默认时钟。
- **支持的判断：**部署 C1024 的 ECHO 两次总时间为 76.982–78.437 秒，C2048 为
  128.349–129.513 秒；最坏与最好之间仍差 38.89%。共享 workspace 改变容量，分别
  保留 16/15 用户，对应全复访命中/全重建。按已固定的完整轨迹目标选择 C1024/W1024，
  保留其首访和 p95 较慢的代价。C256 同样保留 16 用户，但总时间 227.802 秒。
- **不支持的判断：**同 C1024 ECHO 的两次平均总时间仍比 serial sparse 慢 15.08%，
  不建立融合预取的整体加速；比同 C HBM 少 15.50% 涉及容量和重建成本。诊断只采
  请求0/15/16/31，采样复访的 session hit 与历史 HBM token hit 不同；ECHO 预取加
  residual recall 与串行 recall 的 H2D 总字节相同。未采样请求流量未知，不外推。
- **核验边界：**内存有四组工程门禁、11 个直接配置，其他配置保留解析覆盖界限；
  模型权重、普通 activation、cache 预算与进程峰值分开。memory 与 formal 的生成上限
  元数据不同，但完整请求文件逐字节一致，见 身份绑定（Git `934485b:docs/agents/system/echo_memory_formal_workload_binding.json`）。
  首个 C256 测量完成后修复后置审计器，原运行时、数据和时间未改，原 wrapper exit1
  保留；恢复记录（Git `934485b:docs/agents/system/echo_gr_auditor_recovery.md`）不把重新审计当新性能样本。
- **研究边界：**只建立这一 DeepSeek 替身控制点的四方案对照；不推出真实 GR 质量、
  其他输入/预算/GPU 的最优值或 NOSA 性能。旧 NOSA 资料继续保留其原实现/轨迹局限，
  独立前三层 MFU 结果不受本次 GR 子集替换影响。对应 1.3–1.4、2.1–2.2、2.5、4.1–4.3。

## S-022：NOSA 固定容量原版实测与最新源码验收分开

> 下列记录保留原 `_02` 的测量边界与当时判断。当前 `_03` 及匹配 profile/API 已完成，
> 现行结果和来源见 S-026；旧数字不用于描述当前实现。

- **来源：**[NOSA motivation](../../../experiments/nosa_motivation/README.md)、
  工程检查点（Git `934485b:docs/agents/system/nosa_motivation_plan.md`）、
  独立 API 分析（Git `934485b:docs/agents/system/nosa_motivation_hbm_audit.md`）。
  正式表、summary 和 acceptance 位于
  `experiments/nosa_motivation/output/data/nosa_motivation_sm90_20261004_02/report/`。
- **已测范围：**原源码 `6e3dfd17a86dd87be4ec89f0bfccc9bb25a52c5af773f5a5f2ed0753ba5c3e0c`，
  H65536/A128/C1024、U16/R2、P65536/NH16777216，完整 NOSA 32 层、BF16，四方案
  各 32 个正式请求、独立空 cache，128 份全部 candidate hidden 逐位一致。
  NOSA 不执行 LM head，与 DeepSeek 替身的输出计算不同。
- **容量与延迟：**HBM 复访 0/16 命中，三个 offload 均为 16/16；HBM/dense/sync/async
  复访均值为 2465.616814/73.241589/39.532961/42.948851 ms。三个 offload 首访均慢于
  HBM；async 相对 sync 的完整轨迹、首访和复访延迟门槛均失败，完整轨迹慢约 2.8381%。
  Dense 与 sparse 的复访 candidate H2D 软件计数为 34,363,932,672 / 4,718,657,536 B，
  后者 sync/async 相同；这是 payload 计数，不是物理总线流量或总延迟归因。
- **资源边界：**P/NH 是 token 容量，实际总 HBM 并不因此相等。固定路径使用逻辑位置
  直接映射与 session 标签；dense prefetch 要求 H≤P，后端要求 history/chunk 按
  64 tokens 对齐。Candidate 临时留在 GPU。NH 按 session 惰性分配，只是 host
  准入额度，填满还须满足物理 DRAM。
  它不使通用 budget 的共享全地址 staging 自动具备通用有限槽或热点淘汰能力。
- **诊断与参考：**原版 `nosa_motivation_current_diagnostic_20261004_02` 的请求 16 全部
  32 层样本均未达到 page/stripe 两种 90% 门槛；请求 0 无 host-copy 样本。独立
  `nosa_attention_reference_sm90_20261004_01` 审计了 2,112 个 attention 调用，并核对
  请求 16 复用 2,048 个相同 history 调用的依据。HBM 请求 0/16 整请求 MFU 相对独立
  GEMM/BMM+FA3 中位数之和的 MFU 为 98.2195%/95.6393%；对应组合时间为
  2386.912/2387.013 ms，runner 为 2430.182/2495.851 ms。组合是独立 API 中位数之和，
  不是实际整请求或理论上限，也不能代表所有请求。
- **Candidate 限制：**同两请求的 candidate 比率为 63.3831%/17.8166%，extend 为
  19.6079/70.3229 ms；后者异常保留、原因未隔离。长历史主导的接近度不证明 candidate
  高效。物化 QK/PV 的独立参考也保留，不能仅因模型快于重复物化的参考便宣布高效。
- **最新源码：**`02b5d0f9589c5e49257b40b73a5ddce4b2811f0614cc79d4afe242a2aa09f08b`
  在 register8/private_cpp 下通过两用户两轮、完整 32 层四方案 graph/eager 逐位验收、
  host miss、输出生命周期与 owner 清理；register8 是已验收候选配置。正式性能及其
  匹配 profile/API 尚未补测，原 `_02` 不能充当该实现的性能证据。
- **对应：**1.4–1.5、2.1、2.3–2.4、3.1–3.2、4.1–4.3；T-003/T-006/T-007。

## S-023：Q128 候选拒绝与当前诊断范围

- **来源：**[Q128 检查点](../kda/nosa_q128_overlap/checkpoint.md)、
  [64-CTA 独立 warp 诊断计划](../kda/nosa_q128_overlap/independent_warps_ncu_plan.md)及
  [已完成诊断](../kda/nosa_q128_overlap/independent_warps_ncu_findings.md)及
  [同二进制对齐对照](../kda/nosa_q128_overlap/aligned_host_diagnostic_findings.md)。
- **否定结果：**Group4 的完整 hidden 数值门槛未通过；V-first、shared-union halves、
  双 stripe ILP、earliest-pair、frontier fanout、independent-warps 和四组 K/V 读取
  ILP 均未通过重叠
  门槛，未合入。某候选的局部正确性或 API 中位数不能代替其失败的提升条件。
- **独立 warp 筛选：**`nosa_q128_independent_warps_screen_gpu0_20261004_01` 的代表性
  数值、payload、唯一读取与生命周期通过，但首个内部样本双比率均为 0.8005504587，
  立即停止；没有后续样本、完整模型或完整 API 延迟验收。未覆盖 copy 包括 20.928 µs
  启动与 48.64 µs 内部空档，没有尾部空档；单样本不能构成候选间统计性能排名。
- **NCU 边界：**旧 `nosa_q128_ncu_baseline_20261004_02` 是 32-CTA 原版诊断，支持
  小 grid 与供数/流水线等待的局部解释，不能直接解释最新 64-CTA 实现。当前 64-CTA
  NCU run `nosa_q128_independent_warps_ncu_gpu0_20261004_01` 已完成 full/source
  两份报告及独立解析。Full 的 long-scoreboard 基础采样中，host 依赖 store 和
  consumer K/V transaction wait 分别占 43.60%/42.23%，producer acquire 占 0.138%；
  这些是同一计数器内的采样比例，不能相加为实际延迟分解。
- **对齐影响已确认：**相同二进制对照保存了实际映射地址。Host 基址从模 32 余 16
  改为 32 字节对齐后，两条 load 均从每 warp 18 sectors 降到理想的 16 sectors；
  TEX sysmem read-miss sectors 从 517,248 降到 459,776，减少 11.11%，消除了
  高于逻辑 payload 的 12.5% 访问放大。输出、指令数、payload 与 HBM store 不变。
  诊断 padded owners 的 allocator 容量为 128 MiB，保留原 buffers 后共 192 MiB，
  不能直接作为满足容量预算的生产改动。随后首个对齐内部样本双比率为 84.08%，
  未达 90% 后立即停止，完整 API 延迟未测；这些结果也未证明带宽饱和或 spill 主导。
- **四组读取候选：**[隔离准备与验收](../kda/nosa_q128_overlap/four_pair_ilp_plan.md)
  保留原 host 分配，将每 warp 四组 K/V 读取提前到依赖 store 之前，并提高 halves2
  的 producer 寄存器额度。20 次短页调用和代表性数值、唯一读取、生命周期检查通过，
  但首个内部样本双比率仅 70.28%，随即停止；没有后续 API 延迟或完整模型验收。
  所选主函数的 stack 也从 32 bytes 变为 0，因此这一候选同时改变了读取调度与寄存器
  分配；不能从单个筛选样本推导独立的 ILP 效果或候选间统计性能排名。
- **对应：**3.1–3.2、4.1–4.2；T-007。所有候选/NCU 材料为实验目录外的工程诊断，
  不作为新的完整请求实验或正向 overlap 结论。

## S-024：Resident A1024 补测与 pattern 时间链

- **来源：**[NOSA MFU 报告](../../../experiments/nosa_mfu/README.md)中的全模型 native/Triton、
  算子、完整模块与 dense 结果，以及
  [pattern](../../../experiments/nosa_indexer_pattern_65536_1024/README.md)及各自 publication/audit。
- **全模型：**`sparse_flags_native_20261004_01` / `sparse_flags_triton_20261004_01`，
  H65536/A1024、完整 32 层、独立无 profiler 墙钟；native full/extend 为
  2394.454843/36.752254 ms，Triton 为 3678.514021/62.075100 ms，加速 1.536×/1.689×。
  两组 109 份源码、请求与输入身份一致；各组 full/extend 的全部 candidate hidden
  及插桩输出核对通过，不要求跨后端逐位一致。完整模块归因通过审计。
- **局部限制：**四组算子与模块补测已发布：`kernel_flags_synthetic_20261004_01`、
  `kernel_flags_real_20261004_01`、`modules_flags_native_20261004_01` 和
  `modules_flags_triton_20261004_01`。完整 native indexer kernel MFU 约 24.7%–25.6%，
  attention 约 39.1%–40.2%，只有 L31 attention 达到 40%；完整 API 的成本更高。
  Synthetic attention 在 16K/32K/64K 仍慢于 Triton，不能把真实 selection 上的局部
  优势推广到所有输入。固定真实输入来自 `kda_inputs_baseline_20260928_1345`，
  沿用当时的 sparse 轨迹，并未重新采集最新源码轨迹。
- **Dense 与 pattern：**`dense_wrapper_20261004_01` 的 full/extend 为
  3540.283663/82.792723 ms，计时仍处于 nsys 进程内、capture 关闭阶段。
  `nosa_pattern_flags_20261004_01` 三组 dense QA-only/full NOSA/sparse full NOSA
  并集为 788.34375/583.46875/492.28125 MiB；保留的 QA64 数组与新 dense QA64 逐元素
  相同。用新 dense 时间重算了 estimate、overlap 与 threshold；它们仍是单请求离线
  模型，没有执行 DMA，也不证明 A128 offload 收益。
- **发布状态：**报告通过独立检查后替换，旧 resident 受影响产物已清理；有效 QA 数组与
  分解/分布继续保留。这些完成项不再列为最新源码 motivation 的前置待办。
- **对应：**4.2–4.3；辅助限定 3.1 的输入依赖与局部机制证据，不替代 T-007。

## S-025：当前 A1024 offload 冷并集算子补测

- **原来源：**当时的 offload 报告及发布收据，原路径为
  `experiments/nosa_offload_overlap/report/publication.json`。该路径只定位历史文件，
  不将同路径下的替换报告用作本条证据；新结果另见 S-031 与
  [当前 offload 入口](../../../experiments/nosa_offload_overlap/README.md)。
- **新结果：**`nosa_cached_fetch_20261004_01` 的 L0/L15/L31 完整 API 中位延迟较
  同轮串行对照下降 29.34%/24.53%/22.71%；独立 40 次复测确认收益。全部 9 个内部
  样本的 page-envelope 与 stripe-copy 比率均达 0.9，最低 0.9062054934。
- **验收：**全部 query 的 FP32 参考、serial/fused 逐位比较、每次唯一 payload、
  stripe 身份和内部窗口通过；当前普通 owned-cache 的完整 32 层 64K+1K 检查从
  独立空 prefix 构建，全部 extend hidden 逐位一致。657 项 timing 和 27 个 profile
  range 独立核验后发布，并清理三个被替换的旧 run。
- **边界：**算子输入仍来自冻结的历史 L0/L15/L31 轨迹；默认无 cache tag，每次重取
  整批稀疏并集，不测 fixed P/NH 命中或 serving 延迟。不同日期/GPU 之间不作代码
  提速归因。A1024 的正向结果与 Q128 失败分别成立，支持继续检查输入几何的影响，
  不能将前者推广成 async serving 收益。
- **对应：**3.1、4.2–4.3；不替代 T-007 的最新源码完整请求验收。


## S-026：NOSA 固定 P/NH 当前实现的完整测量与效率边界

- **原来源：**当时的 NOSA motivation；历史发布清单、正式验收、profile 验收与独立 API
  文件分别为以下路径。它们只定位原记录，不表示替换后仍可从原目录重开：
  `experiments/nosa_motivation/report/nosa_motivation_poolscan_sm90_20261004_01/publication.json`、
  `experiments/nosa_motivation/report/nosa_motivation_poolscan_sm90_20261004_01/measurement_review.json`、
  `experiments/nosa_motivation/report/nosa_motivation_poolscan_sm90_20261004_01/profile_review.json`、
  `experiments/nosa_motivation/report/nosa_motivation_poolscan_sm90_20261004_01/api_comparison.json`。
  新结果单独见 S-031 与[NOSA motivation 当前入口](../../../experiments/nosa_motivation/README.md)。
  本条吸收已接受的运行和独立核验，不重复执行 GPU 测量。
- **身份与范围：**正式 `nosa_motivation_poolscan_sm90_20261004_01`、profile
  `nosa_motivation_poolscan_profile_sm90_20261004_01`、API
  `nosa_attention_poolscan_sm90_20261004_01` 绑定 118 文件运行源码摘要
  `93061ceb297bfd27ec0bbf6ac21de13c75cc612ece1cb8b5855ec86d3548bb79`。
  原始数据分别保留于 `experiments/nosa_motivation/output/data/<run_id>/`。
  H65536/A128/C1024、U16/R2、P65536/NH16777216，完整 NOSA 32 层、BF16；
  四方案各 32 个正式请求，从独立空 cache 开始，128 份全部 candidate hidden 验收通过。
  不执行 LM head，不与 DeepSeek 替身作跨模型延迟排名。
- **容量与延迟：**HBM 复访 0/16 命中，三个 offload 均为 16/16。HBM/dense/sync/async
  复访均值为 2321.742774/71.594266/29.320605/31.404843 ms；三个 offload 的首访均慢于
  HBM。Sync/async 的完整轨迹延迟比为 0.965545，复访为 0.933633，均表示 async 更慢。
  Dense/sparse 的复访 candidate H2D 为 34,363,932,672 / 4,718,657,536 B，后者在
  sync/async 中相同；这些是软件 payload，不是物理总线流量或因果延迟分解。
- **内部 overlap：**匹配 profile 的 request ID 0/16 覆盖 32 层。96 个有 fetch 的
  async 内部样本均未达到两种 90% 门槛，两种比率的 min/median/max 均为
  0.621540/0.713207/0.790677；另 96 个 async 样本没有 fetch，保留 null。
  Profile 的 24 个 timeline、12 个内部记录、36 份输出及 384 个层样本均通过各自检查。
  有效的低比率样本仍保留，不能以数值正确、窗口相交或 A1024 结果替代门槛。
- **效率边界：**HBM request ID 0/16 的完整请求 MFU 相对独立 GEMM/BMM+FA3
  API 中位数之和对应的 MFU 比率为 100.227379%/102.020880%；candidate-only 为
  75.118264%/73.599107%。Candidate extend 为 16.203399/16.642680 ms，组合为
  12.171712/12.248864 ms。组合不是实测整请求或理论上限，比率超过 100% 不违反该定义。
  差值包括参考未计入的必要非矩阵工作、host 工作、执行相互影响及计时差异，不等于
  孤立 CPU launch 开销或可消除的浪费。物化 QK/PV 参考仍保留；没有计算或 IO 主导的证明。
  独立审计核对 2,112 个 attention 调用及精确 FA3 输出，raw QK/PV 每调用抽查三行。
- **新正式长尾与路由：**dense20/sync22/async22 仍是各方案 candidate 最大值，extend
  为 90.496473/46.396237/48.826513 ms。目标进程记录了 native pool provider，case
  内 unfiltered/fallback/audit-error 增量均为 0；这些计数没有逐请求扫描位置或时长。
  它也不证明 allocator-snapshot 使用同一 provider。新旧正式轨迹的差异不隔离漂移，
  不能将下述旧探针的扫描时长倒填为本轮长尾的因果解释。
- **当前 H64K 预提交诊断：**`nosa_hbm_prequeue_r0_20261004_01` 与
  `nosa_hbm_prequeue_r16_20261004_01` 在两个独立进程中使用上述正式 HBM 请求 0/16，
  保持同一 118 文件源码身份。各自从空 cache 构建 history 后，重复相同 candidate；
  四组各三次预热、七次测量。提前提交相同计算后，control/prequeue 的校验前 event
  中位数分别为 14.721472/12.917824 ms 与 14.712832/12.905472 ms，14 组配对均缩短。
  Event-only/ordinary wall 和两种 delay 的匹配检查按预定 1.05 倍中位数门槛均通过；
  这不要求每个样本都只有很小开销。80 份 BF16 hidden 经独立重开后与正式输出精确一致。
  该结果支持所测输入对提前提交敏感，不隔离可移除 CPU 成本，不测普通 serving 加速，
  也不证明计算或 IO 主导。数据、计时定义、driver 与审核摘要见
  [预提交诊断记录](../system/nosa/nosa_hbm_prequeue_diagnostic.md)。
- **预提交诊断的尾部与边界：**Event 区间在 `DeferredValidation.check` 入口结束，
  不含 finite 归约/host 决策、事务完成、lease 归还及退出时的 allocator 检查。
  请求 0 的 repeat-5 prequeue 与请求 16 的 repeat-5 event-only 在该入口之后，
  仍比相邻同组样本多约 21–22 ms；所有样本均保留。两进程各记录三次 filtered
  pool-referrer 调用，其他路由与错误为零；逐 candidate 计数只标出扫描所在区间，
  没有测扫描的起止或耗时，不能把这两处长尾单独归因于扫描。56 个计时样本的设备
  allocation/free/retry/OOM 计数变化均为零。诊断没有复跑完整四方案用户轨迹，
  额外的 2,223,619,840 B GPU history 检查副本虽在计时外、各组共用，仍改变分配与
  cache 条件。History 的保留状态由目标进程逐次核验，payload 未另存供独立重开。
- **H64K 图内拷贝原型：**外部 `nosa_copy_{baseline,prototype}_r{0,16}_20261004_01`
  四个独立进程保持上述 118 文件运行源码，顺序为 baseline0、prototype0、prototype16、
  baseline16；各自从空 cache 构建 H65536，再运行 A128 三次预热、七次计时样本。
  原型将 62 次层间拷贝提交移入已有 projection graphs，保留 GPU 拷贝、64 次 replay、
  独立 owned inputs 与原事务检查。普通完整 `extend_candidate` wall 包含有限值
  决策、输出 clone、discard、lease drain 与退出 allocator 检查。请求 0/16 的
  baseline/prototype 中位数分别为 15.416903/15.258893 ms 与 15.330098/15.332270 ms，
  差值为 −0.158010/+0.002172 ms，未取得一致的典型延迟改善，暂不接入生产。
  全部 28 个计时样本、12 次预热保留；40 份保存的 candidate hidden 经独立重开后
  与正式 HBM 输出逐位一致，每次运行的 118 份源码快照均通过 hash 检查。
- **原型长尾与判断边界：**两组 baseline 的 measured_00/measured_02 分别为
  37.079517/123.045208 ms 与 37.023268/118.773616 ms，各自的 candidate 计数区间
  都包含一次 filtered pool 调用；两组 prototype 的计时区间内没有此调用。计数未测
  扫描时长，独立进程、重复 candidate、记录引起的 GC 阶段及 cache 条件也有差异，
  因而均值差异不证明原型消除长尾或获得 serving 净收益。两臂各保留
  2,223,619,840 B 额外 history 检查副本，图的 static/private 计费相同，40 次
  设备 allocation/free/retry/OOM 增量均为零；这些不等于全进程物理 HBM 验收。
  初始构建输出与 history/pointer 一致性仍是目标进程断言。记录在外部
  `/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_captured_input_copy_prototype_20261004_01/`；
  独立审计位于同级 `nosa_captured_copy_pilot_review_20261004_01/`，
  `consolidated.json` SHA-256 为
  `bd0119032757e4a4f184f777703e7d7bb348f368034d9554c9b79388e949b8c3`。
  具体原型身份与剩余验收见图内拷贝诊断（Git `934485b:docs/agents/system/nosa_captured_copy_pilot.md`）。
- **H64K 匹配拷贝对照：**`nosa_control_{baseline,guarded_eager,captured}_r{0,16}_20261005_01`
  共六个独立进程，实际时间戳确认 B/E/C0、C/E/B16 顺序且无重叠。E/C 共用
  guard、source references、owned inputs、初始化与失败清理，仅改变 62 次前驱
  拷贝的提交位置；B 保留原实现。C−E 中位数为 −0.257480/−0.038759 ms，
  E−B 为 +0.181062/+0.023139 ms，C−B 为 −0.076418/−0.015620 ms。
  净差异小且幅度变化明显，暂不接入；E/B 也改变初始化和引用条件，不能隔离
  guard CPU 时间。66 份保存输出（含六份构建输出）独立重开后逐位一致；每次
  运行的 118 文件快照、60 个样本的零设备 allocation/free/retry/OOM 增量、
  graph 计费及其他检查均通过。History/pointer 一致性仍是目标进程断言。
  B 的 measured_00/measured_02 各包含一次 filtered 调用；E/C 样本循环均无
  此调用，图外拷贝也出现缺席，不能将其归因于 captured placement，更不提供
  扫描时长或长期避免扫描的证明。所有 42 个计时样本与 18 次预热均保留。
  独立结果位于外部 `nosa_guarded_copy_control_review_20261005_01/consolidated.json`，
  SHA-256 为 `ab27da91d1348d4b87705368e779f32eb9e161f39dbdec3b09f237b0d6be1403`。
  具体身份、数字与边界见[匹配拷贝对照](../system/nosa/nosa_guarded_copy_control.md)。
  后续准备的单次校验内元数据复用候选保留全部字段与校验位置，不跨 attention
  缓存元数据；2026-10-05 整理报告时曾停止后续优化，当时没有启动该候选的 GPU
  测量。最新继续要求与完成后的结果见 S-028。
- **停止前的候选准备：**外部
  `/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_copy_descriptor_reuse_20261005_01/`
  保留源码、diff、CPU mocked 控制流检查和源码审查。原型 SHA-256 为
  `6d70546ba905e1e7eb0a09e51cd9b5740ccb950353a4102464590f9752facde4`；
  `cpu_static_receipt.json` 记录 63 个 CPU 用例通过，SHA-256 为
  `38c132e1483e7a0d7f6615e5b7e3af5bb62c9e5d044e5f6cdd56e0f9a39cad2c`。
  这些材料不证明 CUDA 正确性、checkpoint 等价、性能改善或生产可用；未接入生产，
  当时未运行 GPU。后续 `_02` 测量与独立审计另见 S-028，不能由这些准备材料代替。
- **前期证据与保留范围：**下述普通复跑、GC/pool 探针、扫描原型、完整 A128 indexer
  API 与 Nsight 结果均来自接入前的源码。原 `nosa_motivation_sm90_20261004_03`
  性能家族已由本轮结果替换；后续诊断需要的 433 个输入按精确 hash 保留在外部
  `/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_poolscan_retained_followup_inputs_20261004_01/`。
  其中 `mapping.json` 的 SHA-256 为
  `b65e0efbdc6e12a1a6c5d1d8bbe4cb2a69199484a91464f0b3c3f512d93d020b`，
  `reopen_receipt.json` 为
  `c635940898f65db8c5c8fe9d4bce40e84f50b572c8cefa60db6876446653f10b`。
  原 ID、路径和收据不改写；保留子集不等于重新验收旧完整 capture。原正式长尾未做
  GC/pool 插桩，后续 trace 不能倒填成旧样本事件。具体映射和边界见
  当前发布记录（Git `934485b:docs/agents/system/nosa_pool_scan_publication.md`）。
- **接入前的匹配长尾证据：**两次普通完整轨迹复跑均重现 dense20/sync22/async22；随后
  `nosa_gc_pool_probe_20261004_01` 保持原四方案顺序、预热和全部请求，在三处
  candidate 中分别捕获 36.471005/36.012608/36.528546 ms 的 referrer scan。
  对应直接 GC 回调区间为 1.162762/0/0 ms，不能把约 36 ms 写成 GC collection
  耗时；外层 pool 检查包含内部扫描，时间不能相加。该探针关闭 Nsight/PyTorch
  profiler，128 份 hidden 与正式结果逐位一致；独立复算的 99 项候选区间统计全部匹配。
  这支持优先检查扫描成本；该探针本身不测量可移除收益，后续干预另列如下。
- **诊断来源与限制：**实验目录外的 `/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/` 下，
  `nosa_candidate_repeat_20261004_01/` 的 `repeat_1/`、`repeat_2/` 保存普通复跑，
  `nosa_nsys_followup_20261004_01/gc_probe_01/` 保存探针原始数据，
  `nosa_gc_probe_seal_20261004_01/receipt.json` 为封存收据，SHA-256 为
  `682ae6868a778918492be9bbbfbd6e89394837ef372e881929253ba0ba13324b`。
  收据绑定同一 114 文件运行源码、driver、原始数据及独立复算；128 条同 ID wall
  均与两次普通复跑比较，差异不能单独识别探针开销。只有 11 个请求带绝对区间标记，
  未标记的 HBM 复访最大值不能归因。详细区间和计时边界保留在工程材料。
- **接入前的外部扫描原型：**前几个版本分别暴露 frozen、审计、trace/signal 和动态 getter
  语义问题，未采用。最新 `_03` 通过有限范围的 CPU 语义核查和九种真实 CUDA 空
  MemPool 检查，当时尚未合入生产。首轮 baseline/native 各执行完整四方案轨迹，分别有
  128 份输出与正式结果精确一致；6,118 项完整性检查和 9,692 项独立统计复核通过。
  Dense20/sync22/async22 extend 从 104.600/60.356/63.451 ms 降至
  90.800/46.532/49.468 ms，仍为各组最大值。Offload 复访均值下降，中位数与首访
  extend 均值上升，48 个首访 candidate 同 ID 比较中有 47 个变慢；HBM request 24
  也成为新的最大值。单个顺序对照不能排除运行顺序与系统漂移。
  完整样本和收据在外部 `nosa_pool_scan_intervention_analysis_20261004_01/`，
  `analysis_receipt.json` SHA-256 为
  `2a6a7a86fc5198d4296eb59519474b72ba00e5247f6551984d60843b97880fd7`。
  具体原型身份、限制和验收见扫描优化记录（Git `934485b:docs/agents/system/nosa_pool_scan_optimization.md`）。
- **反序对照：**第二轮保持相同 driver、原型、各臂内部顺序与预热，先 native 后
  baseline，再次通过各 128 份输出检查。上述三个长尾分别缩短
  13.350957/13.840763/13.351112 ms；offload 复访 extend 均值在两轮均降低。
  第二轮首访 extend 均值变化为 -0.240027/-0.075973/+0.056798 ms，未重现首轮
  普遍变慢；两轮描述性合并的首访均值仍为 +0.177255/+0.252951/+0.303609 ms。
  全部 512 条观测、256 对同 ID 比较均保留，12,256 项完整性与 31,625 项独立统计
  检查通过。反序不等于消除漂移，同 ID 重复也不是独立样本。这些前期证据支持了后续接入，
  不构成普遍加速或独立净收益证明。外部
  `nosa_pool_scan_intervention_analysis_20261004_02/analysis_receipt.json` SHA-256 为
  `cc918be091ba55e607c1f434eac7a75de0835db5a5db002a7a1f44cdad41f93f`。
- **接入前的完整 A128 indexer API：**请求 0/16 的真实 Q/K/CIS 复放全部 32 层 indexer，保留两次
  score pass、有限值检查、稳定选块与输出分配，全部选择及 attention 输出精确一致。
  Wall 中位数为 4.792001/5.058357 ms，event 为 4.769504/5.034816 ms。
  三次预热、七次重复的三类 API 共保留 42 个样本。连续独立 API 调用不包含模型层间
  staging，不从整请求计时中扣除；新 raw-QK 参考的 FP32 输出每 pass 逻辑写入
  2.003418 GiB，而 fused score 不物化该输出，不能把它称为 native 下界或实测流量。
  外部 `nosa_indexer_api_analysis_20261004_01/results/receipt.json` SHA-256 为
  `24b11330345327ebc1054ff1333d8ea3e6a1f158a6871c15b2ffb38266318915`。
  完整边界见候选效率分析（Git `934485b:docs/agents/system/nosa_candidate_efficiency_followup.md`）。
- **接入前的 Nsight 边界：**HBM 请求 0–16 的两次独立运行仅采集请求 16，34 份输出检查通过。
  Graph/node candidate wall 为 17.979575/21.060208 ms，相对两次普通观测中位数
  为 +7.92%/+26.41%，均超出预先选择的 5% 诊断容差。两份 SQLite 均提示可能缺失
  CUDA/NVTX 事件。Node 的已记录活动并集为 13.498574 ms；graph envelope 另计，
  没有完整活动清单或 GPU 空闲时间证明。外部
  `nosa_nsys_followup_20261004_01/capture_pair_analysis_01/final_receipt.json` SHA-256 为
  `092a879359199f89fd45d3ace55eed8da966a2ec77e5180021d1047d966e5512`。
  不从普通 wall 中扣除这些区间，也不因未建立完整空闲时间证据而自动重跑。
- **资源与研究边界：**固定 P/NH 是 token 额度，实际总 HBM 不因此相等；NH 按 session
  惰性分配，不代表填满后的物理容量已验收。通用 budget 共享 staging 仍没有有限
  token/page slots 或热点淘汰能力。单条完整 loop 与少量诊断不能证明真实 GR 质量、
  多规模结论或所有 candidate 高效。
- **对应与下一步：**主关联 2.1/4.1，同时关联 1.4、2.3–2.4、3.1–3.2、4.2–4.3。
  T-007 移除已完成的扫描接入及新正式/profile/API 补测；前期复跑、探针、原型对照、
  indexer API、有限范围的 Nsight、H64K prequeue、图内拷贝原型与匹配对照也不再列为待办。
  拷贝诊断主关联 4.1，关联 2.1/4.2；暂不接入，不改变当前正式报告的实现或结论。
  2026-10-05 报告整理阶段曾停止 NOSA 优化，单次校验内元数据复用候选移出待办。
  [motivation 报告](../../../experiments/nosa_motivation/README.md)已完成中文正文与
  汇总图整理，沿用已验收数据，保留正式四方案、独立 API 与外部诊断各自的 run ID、
  长尾和结论边界，报告整理本身没有新增 GPU 测量。用户随后明确要求继续推进目标，
  恢复 H64K 优化，见 S-028；已完成的报告整理不再列为待办。检查的 CPU 成本、
  完整请求收益和因果归因仍有缺口。其他既有研究问题与 ECHO 归因保留，
  4K/16K 暂停安排不变。

## S-027：NOSA 通用 budget 短轨迹补测与物理预算边界

- **后续状态：**该范围已于 2026-10-05 整体退出，见
  整理记录（Git `934485b:docs/agents/system/experiment_organization.md`）。以下记录原测量及其限制，
  不再作为当前实验交付。
- **原来源：**三档历史的正式与 profile 报告。
  正式 run 为 `gr_nosa_poolscan_h4k_20261004_01`、`gr_nosa_poolscan_h16k_20261004_01`、
  `gr_nosa_poolscan_h64k_20261004_01`；对应 profile 在 `_20261004_01` 前插入 `_profile`。
  原始数据当时位于 `experiments/gr_serving/output/data/<run_id>/`，现已随范围清理。
- **源码与请求：**正式清单 286 文件，摘要
  `58f60c7939b1e6b5bd2fc58770a9548bb16c4f967194ddf92d4bbe345f5a8868`；profile 清单
  299 文件，摘要 `98ba72a339f8f593a205df6b1c5d431262f016ed52218ba5d7857d1ec91c1961`。
  六组启动的源码、设备、CPU/NUMA 策略和目标进程 native pool 映射均已核验。
  444 条完整 workload 记录供四方案共用，合计 1776 条正式请求；1332 条非 HBM
  candidate hidden 与相同请求的 HBM 输出逐元素一致，全部保存的输出已在 CPU 重开。
- **旧请求依赖：**42 个旧 workload 输入与新文件逐字节相同，按
  旧 GR 实验（已结束）（Git `934485b:docs/agents/system/experiment_organization.md`）复核。
  新文件保留原请求内容，不复制旧 payload，不改写原审核脚本或旧 ID；这些别名只支持
  workload 复核，不保留或重新验收旧性能家族。复现工具及原始收据在
  `experiments/gr_serving/output/data/gr_nosa_poolscan_publication_20261004_01/`。
- **测量范围：**完整 32 层 NOSA，H4096/16384/65536、A128、C1024；统一准入额度
  HBM 4 GiB / DRAM 16 GiB。三种 offload 复用 backend 共享 workspace，candidate
  执行后 truncate，不使用固定 P/NH 的临时候选 discard 路径，也未启用 compute graphs。
  每方案独立预热，正式每请求一次；H4K/H16K 每档最多 32 请求，H64K 为 6 请求。
- **结果：**HBM 的全请求均值在 21/21 个历史长度/用户数组合最低；overlap 的全请求
  均值在 21/21 组低于 serial sparse，复访均值只在 3/17 个有复访的组合中更低。
  H16K 的 HBM 有 13 次复访重建，三个 offload 均无；H4K/H64K 未观察到复访重建。
  三档 layer-31 的 18 条 serial/overlap 记录通过诊断检查，9 个 overlap 样本的两种
  ratio 均低于 90%；这些比率只覆盖实际 consumer softmax 区间，不代表完整 attention
  或整体延迟隐藏率。空复访组仍为 null，短轨迹的分位数只作描述统计。
- **预算与路由限制：**验收支持活跃分配准入账本，不覆盖 inactive cached blocks、
  碎片、库直接分配及完整进程瞬时峰值。Allocated、reserved 与设备已用量仍须区分，
  不能声称全进程 cache/执行已满足物理 4 GiB 预算。通用模式的全逻辑地址 staging
  不具备固定路径的有限 pool。六次目标进程均记录 native pool provider，unfiltered、
  fallback 和 audit-error 为 0；case 总计数不说明逐请求扫描时长或 allocator-snapshot 路径。
- **研究者修正继续适用：**后续性能测量先只做 64K history，暂停追加 4K/16K。
  本轮两档已在调整前完成并验收，继续保留；报告整理不意味着继续运行这两档。
  Beauty 热度短轨迹只描述所测请求，不恢复真实 GR、用户规模扩展或容量主实验结论。
  配置用户数不是实际活跃数，披露覆盖不足不能替代有效复访 miss 与完整用户覆盖；
  固定 loop 的选择不变，数值一致也不证明推荐质量。
- **对应：**1.4–1.5、2.1–2.4、3.1–3.2、4.1–4.3。T-003 移除已完成的共享路径补测，
  保留共同模型适配与物理预算证据；T-006/T-007 保留 64K 范围内有区分力的容量及归因。

## S-028：H64K 恢复优化与元数据复用诊断

- **用户安排：**最新要求继续推进原目标，取代此前停止优化、只整理报告的安排。
  恢复 H64K 的 HBM-only、dense prefetch、sync sparse、async sparse 效率与
  cache 正确性工作；4K/16K 继续暂停。目标仍包含 HBM MFU 接近矩阵 API 参考，
  以及减少其他方案计算和 IO 之外的开销，不能以单项诊断完成代替整体目标。
- **来源：**[元数据复用结果](../system/nosa/nosa_copy_descriptor_result.md)与
  [indexer 执行 workspace 计划](../system/nosa/nosa_indexer_dispatch_plan.md)。本次读取
  完成记录及独立汇总收据，未重复运行 GPU。Run IDs 为
  `nosa_descriptor_{baseline,captured_control,descriptor_reuse}_r{0,16}_20261005_02`；
  独立收据位于外部
  `/mnt/ssd-wlcb/chenkaiqi/.cxldsagr-tmp/nosa_copy_descriptor_results_review_20261005_02/consolidated.json`，
  SHA-256 为 `c46d1343ede8b9815f7e28150d5e2aaeae4541072d11c12643e8b7162d4d8283`。
- **设计与结果：**六个独立进程按 B/C/D0、D/C/B16 顺序运行；B 是原实现，C 是
  图内拷贝匹配对照，D 只在单次输入校验内复用元数据，保留全部检查及其位置。
  各进程独立构建完整 32 层 H65536 history，再执行三次预热、七次 A128 candidate。
  D−C 的完整 candidate wall 中位数差为 −0.105163/−0.106095 ms，D−B 为
  −0.133753/−0.125243 ms。全部样本和长尾保留；局部差异支持继续检查主机编排，
  尚不支持生产接入或四方案完整 serving 收益。
- **验收边界：**66 份保存输出（含六份构建输出）独立重开后与正式 HBM 输出
  逐位一致，42 个计时样本的设备 allocation/free/retry/OOM 计数不变。History
  bytes/pointer、事务状态及关闭检查通过的是目标进程断言，history payload
  未另存供独立重开。顺序进程、重复 candidate、额外 history 副本与正式轨迹
  条件不同；无 filtered pool 调用不证明消除扫描或长尾。没有新的独立 API
  MFU 参考，不能把本次 wall 差异写成已隔离的 getter/CPU 成本。
- **实现与下一步：**原型保留在外部，生产路径和正式实验数字不变。T-007 移除
  已完成的元数据复用诊断，继续准备 lease 内复用选择输出与 normalizer 的
  indexer workspace 原型，保留原 kernel、检查、提交和回滚语义。该原型尚未
  运行 GPU；先检验完整 candidate 净收益，再决定接入和受影响四方案的补测。
- **对应：**主条目 4.1，关联 2.1/4.2；candidate 效率、剩余长尾、A128 async
  收益与计算/IO 主导均未证明。场景代表性、共同模型比较及其他研究者修正保留。

## S-029：四类实验整理与独立数值验收

- **来源：**用户要求按四类用途整理实验、不运行新实验，随后要求整理完成后提交。
  [实验索引](../../../experiments/README.md)与整理记录（Git `934485b:docs/agents/system/experiment_organization.md`）
  记录最终范围、工具迁移及验证边界。
- **实现：**两个 motivation、resident baseline、算子效率与 offload microbench
  分离 check/bench/profile。成功验收绑定执行身份，bench 复用匹配记录；源码、输入、
  后端或配置不匹配时拒绝复用。模型运行所需的检查和同步保留。
- **范围：**旧 `gr_serving` 热度短轨迹整体退出，通用 serving API 与必要回归保留。
  Dense/sparse resident 合并；pattern 保留实际观测，只结束理想时间和 overlap 支线。
- **证据边界：**CPU 回归、目录与文件哈希检查通过；新入口未运行 GPU，旧报告仍
  属于原实现及计时边界。没有测出数值检查的占比，也没有新的性能或方法收益结论。
- **对应：**4.2–4.3；T-008 完成并移出 roadmap。H64K 优化及其他未决研究任务保持。

## S-030：NOSA 合并 MFU 报告

- **来源：**研究者要求每组实验保留一个算子 MFU 入口，并明确合并
  `nosa_baseline_performance` 与 `nosa_kernel_mfu`，后续新增算子进入同一报告。
  当前入口见 [NOSA MFU](../../../experiments/nosa_mfu/README.md)，维护约定见
  [实验规则](../../../experiments/AGENTS.md)。
- **范围：**统一保留算子、完整模块和完整 resident 模型的有效结果，原 run ID、
  report 数据、源码快照及运行产物不改写。合并清单位于
  `experiments/nosa_mfu/output/data/mfu_layout_migration_20261005_01/migration.json`。
- **边界：**本轮仅合并入口与报告，不产生 GPU 测量。单算子 MFU、模块 API 成本、
  完整模型效率和 A128 serving 的结论仍分别解释；原来的来源和补测限制继续适用。
- **对应：**4.2–4.3；报告维护方式已确定，H64K 优化和现有研究缺口不变。

## S-031：统一框架补测与研究边界

**2026-10-06 后续状态：**本条保留 2026-10-05 的运行与回查记录。真实三层旧报告已
由 S-032 的 A128 四方法结果替换；C10 新补测另记。旧本地官方适配的代码与结果
已按用户要求删除，不把独立 SGLang 的新报告反用于证明本条旧数字。

- **来源与核验：**本条汇总执行者提供的冻结测量与独立复核；Supervisor 重开报告
  核对发布文件与引用数值，没有重新运行模型。当前入口为
  [NOSA motivation](../../../experiments/nosa_motivation/README.md)、
  [DeepSeek motivation](../../../experiments/deepseek_v32_motivation/README.md)、
  [两模型 cache 管理](../../../experiments/cache_management/README.md)、
  [NOSA MFU](../../../experiments/nosa_mfu/README.md)、
  [NOSA pattern](../../../experiments/nosa_indexer_pattern_65536_1024/README.md)、
  [A1024 offload](../../../experiments/nosa_offload_overlap/README.md)与
  [DeepSeek 三层计算](../../../experiments/deepseek_v32_mfu/README.md)。
  各组报告分别绑定原运行、验收与后续发布来源，具体见下表。
- **统一框架与验收：**两个模型共用容量计划、owner/lease 与 token 执行契约，保留
  各自的计算与事务边界。错误直接传播，必要清理的多重异常按各自验收记录核对。
  独立数值、正式计时和 profile 分开，捕获时的 execution/native 身份保持原样；
  捕获后的报告生成器及 helper 修改单独记录，不声称此前捕获执行过后来的发布清理修复。
- **NOSA 重构对照：**三轮当前完整 trace 中位总耗时为 206,944.563 ms，P0 为
  206,538.430 ms，增加 406.132 ms（0.19664%）。两组极差分别为 409.043 与
  223.016 ms。全部 128 个匹配请求的 cleanup 中位数增加，HBM 首访 candidate
  extend 增加 0.110724 ms。相邻 pair04 的总差值为 +388.599 ms（0.18777%），
  单独保留，不纳入预定三轮统计。见[三版本对照](../../../experiments/nosa_motivation/report/three_version/results.md)
  与[pair04 独立复核](../acceptance/unified_runtime_20261005/nosa_pair04_independent_review.json)。
  同机重复不隔离因果，CPU mock 也未覆盖真实 storage scan、lease 或 CUDA，不能
  将全部差值归为某一项检查成本。
- **DeepSeek 重构对照：**完整 trace 中位总耗时变化为 −162.908 ms（−0.09076%），
  但 HBM 首访 candidate、ECHO/serial/dense 复访，以及准入和清理仍有残余增加。
  全部 128 个匹配请求的 cleanup 中位数增加，整体汇总下降不表示无局部回退。
  原始 896 行的身份、分类、配额、计费与完整 cache diagnostics 复核见
  [独立对照](../acceptance/unified_runtime_20261005/deepseek_final_independent_review.json)，
  原报告 `experiments/deepseek_v32_motivation/report/three_version/results.md`
  曾保留各阶段及三轮范围；这些不构成统计置信度或因果归因，当前入口见
  [DeepSeek motivation](../../../experiments/deepseek_v32_motivation/README.md)。
- **异步收益的适用范围：**NOSA A128 的 96 个适用内部样本全部未通过两种 90%
  overlap 门槛，复访均值三轮中位数中 async 比 sync 慢 8.06%。A1024 冷并集
  单层回放的 9 个样本全部通过，最低 ratio 为 0.91081246；同轮串行/融合完整 API
  对照的延迟下降为 23.02%–29.47%。历史 2026-10-04 kernel 审计仍属于旧二进制。
  单层、完整 resident 模型与固定 serving 的计时和验收范围不能互相替代。
- **容量解释：**相同 P/NH 是 token 配额；NOSA 随 session 分配 host backing，
  DeepSeek 使用全局 host arena 与 pinned allocator 档位，物理存储不同。
  逻辑 KV payload、cache 计费、容量预留与进程占用分别报告；cache HBM 计费包含
  graph private reservation，不等于 tensor payload 总和。Allocated、reserved
  和设备已用量分开，普通 activation 未单独测峰。455 个 64K history 的边界仍是
  DeepSeek 静态估算；16 用户的完整请求观测不证明填满 NH 或离线最大 P/NH 能装入机器。
  规划与字节公式的独立复核见
  [容量算术复核](../acceptance/unified_runtime_20261005/cache_unified_independent_arithmetic_review.json)。
- **保留与可复现边界：**两模型 `report/three_version/` 的选用数据支持重算比较表格，
  不足以在原始依赖退休后重做完整历史源码/数值审计。离散 GPU 进程样本只说明采样
  时未见其他计算进程，不能证明连续独占，也不记录 CPU 活动、时钟或利用率。
- **T-007 继续独立：**[workspace 计划](../system/nosa/nosa_indexer_dispatch_plan.md)
  尚未实施，统一框架重构没有实现或验证它。实施前须按验收后的源码重查分配和分派
  假设，再检验完整 candidate 净收益与四方案正确性。S-028 的历史元数据复用数字
  不改写；冻结 reviewer 所需的六份原始参考按
  [保留与重开说明](../acceptance/unified_runtime_20261005/t007_reference.md)回查。
- **对应与研究判断：**1.4、2.1–2.5、3.1–3.2、4.1–4.3。完成的重构和报告工作
  不再列入 roadmap；T-006、T-003、T-007、T-002 保留。H64K-only、不跑满 NH、
  固定 loop、五类共同模型对照仍未建立、真实场景与质量未验证等修正均不变。

- **已发布的容量、NOSA 局部与 DeepSeek 证据：**下表各 manifest 分别绑定所选报告文件、原运行
  身份及后续报告生成来源。该轮 Supervisor 重开并核对所列报告的发布文件和 README
  的 SHA256，全部匹配；原始数组、数值与 native 验收沿用执行者的独立审计，未再跑 GPU。

| 报告 | 当前运行与选用范围 | 发布依据 |
| --- | --- | --- |
| NOSA motivation | `refactor_final_nosa_publication_20261005_02`；clean bench01、profile02 与独立 attention reference01 分开，三版本对照保留原始选择 | [最终 manifest](../../../experiments/nosa_motivation/report/publication_manifest.json)、[API 对照](../../../experiments/nosa_motivation/report/final/api_comparison.json) |
| 容量 | `cache_unified_20261005_02`；五份 DeepSeek 静态计划、四种 NOSA 分配声明，两模型各自 clean bench01 的八行完整请求观测 | [publication](../../../experiments/cache_management/report/unified/publication.json)；静态最大值不等于实测物理容量 |
| NOSA MFU | 真实算子、native/Triton 模块及 sparse 模型的 bench/profile 为 `refactor_mfu_*_20261005_01`；dense 为 fresh check02 后的 `refactor_mfu_dense_{bench,profile}_20261005_02` | [publication](../../../experiments/nosa_mfu/report/publication.json)列出全部 12 个当前 run；synthetic 与冻结输入保持原身份 |
| NOSA pattern | `refactor_nosa_pattern_capture_20261005_01`；独立 observer check 分开保存；`refactor_qa64_environment_compare_20261005_01` 比较当前与保留 QA64 数组 | [publication](../../../experiments/nosa_indexer_pattern_65536_1024/report/publication.json)、[QA64 比较](../../../experiments/nosa_indexer_pattern_65536_1024/report/qa64/environment_comparison.json) |
| NOSA overlap | `refactor_nosa_overlap_{bench,confirm40,profile}_20261005_01`；主测、40 次确认和独立 profile 分开 | [publication](../../../experiments/nosa_offload_overlap/report/publication.json)、[当前 profile 审计](../../../experiments/nosa_offload_overlap/report/current_profile_integrity.json) |
| DeepSeek C10 motivation（该次来源） | `refactor_final_deepseek_publication_20261005_01`，选用 `refactor_final_deepseek_{bench,profile}_20261005_01`；三轮对照只保留本条历史范围 | 后续修改的补测另记 S-032；[当前实验入口](../../../experiments/deepseek_v32_motivation/README.md)不反向证明此处旧 run |
| DeepSeek 真实三层（已替换） | 历史运行 `refactor_three_layers_publication_20261005_01`，独立 check02，`refactor_three_layers_{bench,profile}_20261005_01` | 旧 `layers3` 已替换；[当前四方法 publication](../../../experiments/deepseek_v32_mfu/report/four_methods/publication_manifest.json)与[当前汇总](../../../experiments/deepseek_v32_mfu/report/four_methods/summary.json)只对应 S-032，不能作为本行旧运行的证据 |

- **MFU 结果：**native sparse full/extend 为 2374.539186/36.207893 ms、
  50.379877%/52.557023%；dense 为 3506.117727/81.806672 ms、
  62.605166%/63.023282%。分母取独立无 profiler benchmark；两种 attention 的有效
  FLOPs 不同，不以 MFU 大小替代延迟比较。Sparse 独立 profile 的 extend
  indexer/attention 为 23.054795%/38.028264%，尚未同时达到 40%。
- **Pattern 结果：**三个当前选择分支的全模型去重 K/V payload 为
  788.34375/583.46875/492.28125 MiB。QA64 的 ids、valid、union 与保留数组均为
  0 个不匹配元素，只支持这一请求的复现。Capture metadata 的 hidden check 字段
  仍为 null，独立 observer check 未回写成 capture 自带的数值验收；该实验不测时间、
  带宽、物理传输或质量。
- **发布与历史保留：**三个 NOSA 报告的 `source_delta.json` 与 `publication.json`
  将捕获源码、后续 navigation/identity 修正和实际执行的 CPU 报告生成器分开。
  `snapshot_report_helpers` 只保存报告进程已加载的仓库 Python 文件当前磁盘字节及
  pyproject/lock，不证明原捕获 runtime/native 身份或全部可能源码。对应
  `retirement.json` 记录替换范围；synthetic、QA32/QA64、分解与分布继续保留。
  Overlap 的历史 kernel 审计原文件保留，只选 12 个按原路径核对哈希的静态来源；
  其历史 runtime-binding 和动态 profile 字段不支持本轮结论。当前九个样本的 P=F
  与唯一 stripe 验收来自当前 profile。

- **NOSA 最终选择与清理：**最终 manifest 绑定 33 个报告文件与 README，并包含
  `report/three_version/` 的原样选定数据。复制到 `report/final/` 的 publication、
  report/helper 清单保持原字节，其中 `source/` 路径仍从原 publication02 output
  解析，不从 Git 报告目录解析。37 份 helper 快照在原 output 保留；轨迹图的源码
  复制记录不声称采集了完整已加载模块清单。17 个枚举旧路径已清理，P0/当前控制
  和 T-007 的六份数值参考继续保留；不将保留的比较表格等同于完整历史运行可重审。

- **NOSA 完整请求与 candidate API 参照：**正式墙钟只取
  `refactor_final_nosa_bench_20261005_01`，独立数值依据为
  `refactor_final_nosa_check_20261005_01`，诊断为
  `refactor_final_nosa_profile_20261005_02`，attention 参考为
  `refactor_final_nosa_attention_reference_20261005_01`。发布输出
  `experiments/nosa_motivation/output/data/refactor_final_nosa_publication_20261005_02`
  的 14 个资产哈希、37 份已加载 helper 源文件快照与四组独立预计算比较均已核对。
  当前 benchmark 源码集合摘要为
  `4665ccce226af23f631e619f748369cea28d4a09b99c1947cbb6b030e58f120c`，
  numerical receipt 文件 SHA256 为
  `34ee0a120712d353bd57dc350c3667f5ce2f2fc2b8a1607aa1f1a9f00c405bc5`。
  不把这个完整集合摘要代入另一采集范围的源码或 native 身份。
- **API 数值与边界：**HBM 请求 0/16 的完整墙钟为
  2353.533356/2344.728021 ms，MFU 为 50.096669%/50.284801%；独立 FA3
  与矩阵 API 中位数之和为 2347.256745/2347.328041 ms，对应 MFU 比率
  99.733311%/100.110888%。Candidate 墙钟为 16.097616/16.817585 ms，
  MFU 为 14.763505%/14.131472%；组合参照为 12.129760/12.201056 ms，
  比率为 75.351281%/72.549394%。完整请求与 candidate 分别覆盖各自全部有效
  matrix FLOPs；candidate 的 `extend_ms` 不含 admission、prefix 与独立
  runner cleanup，但包括 backend 内部 discard/lease drain。
- **两种独立参考均保留：**raw QK/PV 将按 query 选中的 KV 物化并重复，完整请求
  组合为 4230.536234/4230.075306 ms；candidate 为 12.648416/12.187488 ms。
  FA3 组合保留优化后的 sparse 访问与 attention helpers。两者都是独立 API 中位数
  之和，不是实际请求或理论上限，不因较慢参考更容易通过就据此判定高效。Attention
  参考重开 2,112 个调用，并在 32 层 final-prefix 一致的依据下复用 2,048 个
  history 调用；其 operand、数值与计时边界由独立审计记录，不声称重新运行 GPU。
- **验收结论：**numerical、profile、measurement integrity 通过；async 的完整
  trace/首访/复访 speedup 均失败，96 个适用内部样本的双重 90% overlap 门槛全部
  未过。MFU 接近度、compute/IO dominance 仍需解释。当前数值、分母与诊断分别见
  [API 对照](../../../experiments/nosa_motivation/report/final/api_comparison.json)、
  [验收结果](../../../experiments/nosa_motivation/report/final/acceptance.json)及
  [profile 摘要](../../../experiments/nosa_motivation/report/final/profile_summary.json)。

- **DeepSeek C10 诊断：**`refactor_final_deepseek_profile_20261005_01` 的
  `analysis/completion_receipt.json` 接受全部 13 个成功的 CPU 分析阶段；正式墙钟取
  `refactor_final_deepseek_bench_20261005_01`。矩阵归因独立复核覆盖 45,933 个
  matrix calls/primary kernels、195 组、267,562 个 GPU activities 与 6,560 次
  graph replay。HBM/ECHO/serial sparse/dense 的请求有效计算利用率，首访为
  44.13%/42.15%/43.42%/43.20%，复访为 44.23%/9.16%/12.42%/6.77%；
  按精度分别用 dense 峰值归一化，不能直接与另一模型的单一 BF16 MFU 排名。
  HBM 复访重建历史，三个 offload 方案命中，因此两类请求工作量不同。
  ECHO/serial sparse/dense 的复访均值为 24.530059/18.083835/33.196589 ms。
  ECHO 与 serial sparse 的 16 次复访 candidate H2D 均为 1.161186 GiB，仍未
  建立融合预取加速。API activity 与正式墙钟是独立测量；差值不等于 CPU 净成本。
- **真实前三层计算：**`refactor_three_layers_{bench,profile}_20261005_01`
  复用独立 `check_02` receipt，canonical `receipt_sha256` 为
  `9ac8ab2d3b695d32e9fd81b269ca4f54130023fab4f63ce89df6ee077c18f9a5`。
  Profile 的 `postrun_audit.json` 与 `publication/summary.json` 已接受；resident/
  offload prefix 中位数为 642.340201/1190.485267 ms，extend 为
  13.953853/27.631516 ms。完整阶段按精度归一化的有效计算利用率分别为 prefix
  45.077263%/24.321963%、extend 38.699821%/19.543322%。Resident/offload
  candidate hidden、末 token logits 与对应插桩对照逐位一致；prefix 的输出检查
  属于运行时范围，postrun 不重新证明未保存的 prefix tensor。该验收不独立解析
  native trace、检查 kernel 或验证 native 库哈希，相关来源按各自执行记录回查。
  本轮没有 NCU，不沿用旧 66% MLA / 29% resident-extend 数值。
- **当时回查的数据：**C10 接受记录位于
  `experiments/deepseek_v32_motivation/output/data/refactor_final_deepseek_profile_20261005_01/analysis/completion_receipt.json`，
  聚合值位于同目录 `aggregate_mfu/aggregate_mfu.json`；真实三层汇总位于
  `experiments/deepseek_v32_echo_prefill/output/data/refactor_three_layers_profile_20261005_01/publication/summary.json`。
  该次 Supervisor 重开上述文件核对 run ID、接受状态和引用数值，未重复 GPU 或原始
  trace 数值审计。最终报告发布清单已在上表绑定，三层汇总当时 run 为
  `refactor_three_layers_profile_20261005_01`；当前结果另见 S-032，旧路径不表示产物仍保留。
- **DeepSeek 后续发布来源：**三份 `source_bindings.json` 明确注明，在后续 CPU
  发布进程中收集的是已加载仓库模块对应文件当时的磁盘字节，不代表已加载的
  bytecode，也不能据此声称此前 13 阶段分析已加载同一完整清单。`evaluation/provenance.py` 后加的 snapshot helper 不属于此前执行；
  既有函数 AST 相同的核对与原始执行/native 身份分别保留。真实三层的实际
  `--publish` 重开校验以退出码 0 完成；只选报告文件，源码快照留在 output。
  三份发布 manifest 的 retirement 均已完成，P0/当前控制保留；最终跨文档导航由
  root 统一检查。


## S-032：DeepSeek A128 四方法 MFU 与局部重叠


- **研究条目与变化：**主条目 2.4、2.5、4.1–4.2。研究者将默认 extend 改为 A=128，
  并要求 dense 使用连续 Host DRAM/HBM 与 `cudaMemcpyAsync`。真实前三层已完成
  新实现的四方法独立验收、正式计时和 profile；dense 完整 extend 低于 serial sparse，
  仍高于 HBM。随后研究者要求“先不要管 motivation 实验”，C10 DMA 补测、容量
  后续审阅及其他 motivation 实验暂缓，既有报告和产物保留当前状态。
- **正式来源：**[实验 README](../../../experiments/deepseek_v32_mfu/README.md)、
  [结果](../../../experiments/deepseek_v32_mfu/report/four_methods/results.md)、
  [汇总](../../../experiments/deepseek_v32_mfu/report/four_methods/summary.json)、
  [运行验收](../../../experiments/deepseek_v32_mfu/report/four_methods/run_acceptance.json)、
  [原生 SQL overlap 复核](../../../experiments/deepseek_v32_mfu/report/four_methods/dense_overlap_sql_audit.json)和
  [发布清单](../../../experiments/deepseek_v32_mfu/report/four_methods/publication_manifest.json)。
  Check、bench、profile 分别为 `deepseek_mfu_dma_a128_check_20261006_01`、
  `deepseek_mfu_dma_a128_bench_20261006_01`、`deepseek_mfu_dma_a128_profile_20261006_01`。
- **实际范围：**真实 checkpoint 第 0–2 层依次传播 hidden/residual，含 embedding、
  三个 dense MLP、final norm 和末 token LM head；普通持久追加，不使用 C10 source-input
  replay。H=65,536、A=128、P=65,664，history chunk=1,024，extend 为完整 128-token
  batch；cold 只清除 offload 历史主 KV 的 HBM 驻留，DRAM/indexer 保留，HBM 方法
  保留主 KV。主 KV 为 BF16 512 latent + 64 RoPE。GPU3 经 PCI/SM90 核验为 H200
  SXM，CPU24–31、8 线程。Cache HBM/DRAM 预算为 24/64 GiB，权重和普通 activation
  另计；没有验证多用户填满 NH 或物理容量上限。
- **阶段结果：**下表为独立无 profiler bench 的中位延迟。MFU 分子来自完整实际调用
  账本的 useful matrix FLOPs，按 FP8/BF16/FP32 名义 dense peak 归一化。四方法
  useful 工作量相同，prefill/extend 理想计算时间分别为 289.549384/0.675229 ms。

| 方法 | Prefill ms | Prefill MFU | Extend ms | Extend MFU |
| --- | ---: | ---: | ---: | ---: |
| HBM | 649.381321 | 44.588499% | 3.761362 | 17.951720% |
| ECHO | 686.326987 | 42.188256% | 8.434403 | 8.005655% |
| serial sparse | 665.076367 | 43.536261% | 6.534333 | 10.333559% |
| dense prefetch | 664.751053 | 43.557567% | 5.985946 | 11.280242% |

- **DMA 与剩余 gap：**所选第 1 层完整 extend 窗口中，下一层 H2D DMA 的完整时长
  为 1.374430 ms，与当前层独立计算的交集为 0.690078 ms，占 50.208305%。原生
  CUPTI 记录为 stream 45 上的一次 `cudaMemcpyAsync_v3020`；独立只读 SQL 核对
  源内存类型为 Pinned、目标为 Device、copyCount=1。三个层的 H2D memcpy 均为
  75,497,472 B，与 65,536 records 的 cache 计数一致，dense scope 内没有 mapped-host
  gather。四方法该层 extend GPU gap 分别为 0.300639/2.078591/1.830013/0.008064 ms，
  prefill 最后 chunk 分别为 0.075424/0.358751/0.071232/0.071712 ms。Dense 最大
  单段 gap 为 0.002144 ms。Gap 是全部 GPU 活动区间并集的补集，不是单独归因的
  CPU 成本；局部重叠也不替代完整 serving 收益。ECHO 融合 kernel 保留为 Compute + IO。
- **持久追加与资源：**cold extend 每层还写回新增 token 的 147,456 B；C10 的
  transient candidate 没有这项写回，不能互换语义。Prefill 每层 H2D 为 0、D2H 为
  75,497,472 B。Dense ticket 借用已有 span，不新增 ID/count tensor，本次三层
  资源计划中的 ticket workspace 为 0 B。Graph static storage/allocated/private
  reserved 仍为 195,863,088/197,275,136/1,684,013,056 B；规划上限与实际分配分开。
- **数值与归属验收：**check 13 项、profile 25 项比较全部逐位一致；发布时额外重开
  四方法 hidden/logits 的 8 个保存 tensor，与 check controls 逐位一致。Q=128/1024
  在三层各有 projection/finish，共 12 个模板；八组阶段共 1,560 次 replay。逐层
  shape、capture/runtime graph ID、GPU 节点及每个 chunk 的调用次数通过核验，
  kernel count/time、API 归属、每 query 的 indexer/MLA 覆盖与四方法工作量守恒。
  Graph setup 加八组阶段共九份 NSYS capture。本轮未采集 NCU。
- **身份与复核边界：**receipt 为
  `/tmp/cxldsagr-checks/deepseek_v32_mfu/data/deepseek_mfu_dma_a128_check_20261006_01/receipt.json`，
  canonical SHA256 为 `94f396fde2c4dca6bab58b0baf7b0b2b0726e4af09882a77021d933d5111aa02`；
  canonical execution identity 为
  `bc72b3bf8ec9a2daa2996f705d5a537ee87b66aeb542358f1b1e4a635df4b310`。
  三次运行的执行、源码、native 和输入身份相同，显式绑定 dense transport 与实际 pool
  布局。当前文件核验覆盖所记录 FlashInfer native 库与 build metadata，不扩大为所有
  二进制的独立审计。报告 helper 的实际哈希单独记录，18 项发布文件和 README 绑定核验。
- **监测条件：**三个测量进程和 observer wrapper 均退出 0，check/bench/profile
  分别有 13/18/32 次离散样本，最大间隔为 2.199/2.202/2.846 秒。独立原始记录复核
  没有发现外来 GPU 进程、退出竞态或查询错误；三份监测与复核均已绑定。离散样本
  不能证明连续隔离、全机独占或 CPU 独占。
- **替换边界：**本次 DMA 报告替换 A128 mapped-host gather 版本。旧 query、pool、
  驻留和计算图条件不同的 A1024 结果不作同配置加速对照。历史命令与 run ID 保留
  各自含义；受影响旧素材和运行产物在新发布验收后清理，不保留平行旧报告。
- **C10 与容量边界：**[C10 报告](../../../experiments/deepseek_v32_motivation/README.md)
  当前保留的 mapped-gather 来源为 `deepseek_mfu_c10_bench_20261006_01` 与
  `deepseek_mfu_c10_profile_20261006_02`。复访均值为 HBM/ECHO/serial/dense
  2178.992/23.520/17.188/25.462 ms；16 次候选 H2D 合计分别为
  0/1.161186/1.161186/11.25 GiB。它们不代表 DMA；C10 独立补测现已暂缓。
  [容量材料](../../../experiments/cache_management/README.md)保留已发布状态，后续审阅暂停。
  455 个 64K history 仍是静态估算，真实三层测量不验收填满 NH；NOSA 的有效结果
  保留各自原始范围。
- **其他任务边界：**旧本地官方 ECHO 适配已删除，独立 SGLang 计时另行报告。T-006 容量范围探索及
  C10 DMA、剩余 gap、ECHO 预测/预取与历史保留归因随 motivation 暂缓，不列为当前
  补测依赖。T-007 保留尚未实施的 NOSA indexer workspace 独立方案；这次 DeepSeek
  改动没有完成该方案。固定 loop、H64K-only、
  4K/16K 暂停、不跑满 NH、共同模型与 GR 场景仍待建立等选择不变。
