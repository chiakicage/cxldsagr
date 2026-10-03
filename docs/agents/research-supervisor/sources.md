# 研究状态的材料依据

更新：2026-10-02。S-001–S-007 记录初始仓库材料；其旧算子结果未在本轮重新测量。
S-008–S-010 记录本轮实际完成并审计的 GR serving 实验及用户约束；S-011–S-012 记录
用户修正后的负载候选与数据可用性检查；S-013 记录热度抽样选择，S-014 记录本轮
MFU/cache 与 baseline 判断修正；未重新核验全部论文或 SOTA。
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

**后续修正：**研究者指出 ECHO MFU/cache 策略与 DeepSeek MFU 存在问题，见 S-014。
旧延迟、MFU 和归因不能用于认定 baseline 有效或判断 ECHO 方法优劣。

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

## S-008：本轮用户选择与执行范围

- **来源：**本轮用户要求实现 GR serving，先 DeepSeek V3.2 前三层及对应输入独立复制到约 8B，
  不用 MoE；之后完整 NOSA-8B。后续指定保留 1/8/32 并加入 64/128/256/512 用户，
  按热度控制复访、每用户最多 8 次，固定历史 4K/16K/64K 分开运行，时长约 30 分钟。
  [执行契约](../system/gr_serving_task.md)、[最终审查](../system/gr_serving_review.md)。
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

- **来源：**[GR serving 报告](../../../experiments/gr_serving/README.md)、
  [独立验收记录](../system/gr_serving_review.md)，以及报告中各历史档的 metadata、audit、
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

- **来源：**用户本轮两次修正；[候选协议](../system/gr_serving_workload_redesign.md)、
  [可行性分析](../system/gr_serving_workload_feasibility.md)。本轮只做 CPU 调度/LRU 预演和
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
  [数据轨迹核对与协议](../system/gr_serving_dataset_trace.md)、
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

- **来源：**用户要求用现有热度构造分布、取消最多 8 次限制，并明确“从用户访问次数
  转化成概率分布，再随机采样”。[当前协议](../system/gr_serving_workload_redesign.md)、
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
  [系统状态](../system/implementation-status.md)、[三层诊断](../system/deepseek_echo_three_layer_profile.md)
  和 [GR 审查](../system/gr_serving_review.md)。
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
