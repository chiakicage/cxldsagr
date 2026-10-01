# 系统工程候选实现方案

供系统编码与实验执行者维护。内容合并自 `1971047` 的工程材料，整理于 2026-10-02；
原实验日期、run ID 和测量边界保留，本次未运行新实验。固定 history、变化 candidate、
prefill-only 是候选方案，其代表性与模型、数据适配仍待确定。研究判断与任务取舍以
[四环节状态表](../../status.md)、[研究叙事](../../research.md)和[研究路线](../../roadmap.md)为准。

原方案日期：2026-10-01。若采用固定 history 的 GR 候选场景，可将 NOSA 的 async sparse
KV fetching attention 推进到有限 HBM 下的多用户 prefill serving。以下保留候选开发拆分，
不代表已选定路线、已执行或必须按此顺序推进；研究探索可交叉进行。
研究问题见[research.md](research-context.md)，已完成范围见[系统状态](implementation-status.md)。

相关工作核对与开发并行，计划在系统方案固化前完成首轮比较，明确与 sparse attention、
prefix caching、KV offloading 及 prefetch 的差异。

## 1. 建立当前实现的完整模型性能基线

先回答单层收益能否保留到整个 NOSA 模型。对同一 checkpoint 和请求，比较 resident
sparse、串行 sparse offload、async sparse offload。三组分别从独立空 cache 构建
sparse prefix，每次重复恢复相同的起始状态。

分别报告冷启动的 prefix 构建成本和已有 prefix 下的 candidate extend 成本。
计时覆盖 embedding、所有层、indexer/CIS、派生缓存、fetch、attention、KV 写回、
final norm 与完成同步；加载权重和编译不计入推理。Cache 分配和恢复若在计时外，应
明确记录；后续请求级测量要包含请求实际承担的这些成本。

验收要求是全部 extend hidden 的数值检查通过，并发布带新 run ID、源码指纹和
完整计时边界的结果。延迟结论来自未启用 profiler 的运行，profile 负责解释开销。
同时核对 resident sparse 完整模型、synthetic MFU 和 full-NOSA pattern 的补测
范围。DeepSeek gather 修复的性能补测单列维护，不阻塞 NOSA 系统开发。

## 2. 实现跨请求固定 history 复用

目前 `serving` 在每条请求后释放 cache。下一步增加用户 prefix 的长期持有与查找，
先在串行请求下把生命周期做正确，再扩展调度。目标流程是：

```text
首次访问：校验请求 → 构建固定 prefix → 保存可复用状态 → 执行 candidate
后续访问：核对 prefix 兼容性 → 复用已有 history KV → 执行新的 candidate
请求结束：等待相关操作完成 → 丢弃或回退 candidate 状态 → 保留固定 prefix
```

Cache key 至少需要区分实际 token 前缀及模型兼容信息，不能只使用 `user_id`。
固定边界采用 `stable_prefix_tokens`；checkpoint、位置编码、dtype 和 sparse policy
变化时须拒绝不兼容复用。K/V、CIS、压缩记录与稳定 pool 应一致回退，失败请求不能
破坏已经提交的 history。派生窗口跨越 prefix 边界时也要覆盖到。

这部分需要共同调整 `cache/` 的持有和事务接口、NOSA 的派生状态处理、`serving/`
的请求生命周期，保持 `executor/` 不读取 GR 请求。验收覆盖冷启动、重复访问、不同
candidate、交错用户、失败回滚和缓存失效；复用执行与独立完整重算保持约定的数值一致性。

## 3. 为 NOSA 加入有限 HBM 页池

将一层完整逻辑地址 staging 替换为受预算约束的 slots，建立逻辑页到物理 slot 的
映射，明确准入、替换和正在加载或消费页的保护。模型继续提供 KV 布局和选择语义，
共享 cache 管理存储与生命周期。DeepSeek 的有限 token pool 可提供接口经验，
复用前仍需适配 NOSA 的 GQA 页布局和 attention 访问方式。

精确工作集超过池容量时，需要分段加载与消费，同时保留每个 query 的完整选择，
不能用裁剪 top-k 隐藏容量不足。Slots 的覆盖和复用必须等待相关异步操作完成。
同时预算 CIS、压缩记录、workspace、临时 candidate 状态与 host backing，避免
主 KV 移出 HBM 后，其他常驻状态成为新的用户容量限制。

验收需要在小于单请求完整工作集的预算下运行，证明峰值受控、无过早覆盖或死锁、
结果正确；分别记录逻辑 payload、重复加载、命中率、cache 分配和进程峰值。
有限池中的分段复用若引入重复搬运，应如实统计并解释，不沿用当前完整 staging
路径的每层 read-once 结论。

## 4. 接入请求流，测量服务能力

复用 `GR/` 的用户热度、固定 history 和候选变化，用相同请求序列对比方案。
增加到达时间回放、等待队列和完成时间记录，先建立单 GPU 串行服务基线，再评估
并发或批调度是否值得加入。生成器的目标 QPS 只是输入压力，不能当作已达到的吞吐。

| 变化因素 | 主要观察量 |
| --- | --- |
| 用户数、热度分布、复访间隔 | 可保留的 history 数、缓存命中、冷启动比例 |
| History 长度、candidate 长度 | 整批选择并集、host payload、indexer/fetch/attention 开销 |
| HBM 与 host 预算 | 页替换、重复加载、内存峰值、可持续服务规模 |
| 请求到达率 | 服务时间、排队时间、p50/p95/p99 请求延迟、完成 QPS、队列是否持续增长 |

对照至少包括相同容量与复用策略下的串行 sparse offload 和 async sparse offload；
另用 resident sparse、history 重算说明 HBM 容量与前缀复用的影响。区分冷请求和
复访请求，固定 warmup 与统计区间。验收目标是在相同资源和输出要求下，确认 async
方案在哪些负载下降低请求延迟或提高稳定吞吐，也记录没有收益的区域。

## 5. 补齐机制分析与适用范围

按待澄清的问题安排针对性消融，可与完整模型和 serving 探索交叉开展：任务粒度、fetch 并行度、
fetch/compute 顺序、去重和 HBM 替换策略。每次只改变可解释的因素。若拆分 query
计算，要增加相同拆分的串行对照，并始终保留整批并集一次 fetch 的强串行基线。

Overlap 继续使用 kernel 内实际工作区间；page envelope 与非空 stripe-copy
window 分别计算。每页 envelope 必须等于本页全部非空 stripe 的
`min(start)/max(end)`；每个 profiled sample 的两种 ratio 都须 `≥0.9`。
不能用完整 kernel 窗口或只看中位数替代验收。机制指标解释性能，整体延迟决定方案
是否值得采用。

Resident indexer 和 attention 的 40% useful MFU 目标继续按
[KDA 组件计划](../kda/README.md)维护，它是模块优化目标，不是系统研究完成标准。
模型推广优先检验 DeepSeek V3.2 的不同访问语义；CXL/RDMA、可变 history、跨 GPU
或网络服务在当前 local DRAM 路径完成后另行确定范围。

若需要推荐质量主张，另选有真实目标标签的数据和评价协议；
当前合成 GR 请求不承担这一结论。

## 报告更新

性能优化或数值修正后，先补测受影响实验，验收正确性、源码身份和计时完整性，再
发布新 run 并替换旧报告与产物。补测期间保留旧 run 和测量边界，明确“改动后未运行”。
实现状态更新到[系统实现状态](implementation-status.md)，正式结果保存在各自实验中；测试仍只用于
代码验收。具体维护规则见 [AGENTS.md](../../../AGENTS.md)。
