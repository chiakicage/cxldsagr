# NOSA offload fetch / attention 实现检查点

本页记录 2026-09-30 已验收的融合实现，build key 为 `484e32532ba74fb6`。完整源码、依赖
和输入身份保存在各正式 run 中。本次整理没有重新编译、执行验收或生成性能结果。

## 实现范围

目标输入为 BF16、65536 prefix + 1024 queries、32 Q heads、2 KV heads、D128、64-token /
64-block selection。历史 K/V 存在 pinned local DRAM，CIS 和压缩记录保留在 HBM。
各层共享一层完整逻辑地址范围的 HBM staging，尚无有限 slots 或淘汰策略。

Planner 按 `(KV head, block)` 去重，每个唯一的 64-token page 划分为 8 个互不重叠的
8-token stripe。所有 CTA 保留 persistent FA3 计算，默认最多 96 个 CTA 额外利用 producer
warpgroup 的 warp 1–3 搬运，warp 0 保留 TMA，两个 consumer warpgroup 计算 attention。
历史向量只有一个 owner 和一次 `.cv` host load；stripe writer 的 fence/barrier 与跨 CTA
acq_rel ready 链完成后，TMA acquire / async-proxy fence 才允许读取 HBM。

在两个 KV heads 且 `ceil(queries / 8) * KV_heads == 256` 时，fetch 与 compute 同为 head 1
再 head 0；其他几何保持原调度。producer / consumer 使用 24 / 240 寄存器预算，并检查
64512-register CTA pool 和 cooperative occupancy。完整 ordering、空尾 stripe、初始化、
计数与测量规则见[任务契约](task.md)。

模型显式接口为 `attention_mode="sparse"`、`cache_backend="offload"`，默认
`offload_query_tile_size=128`、`offload_fetch_ctas=96`、`offload_overlap=True`。
Query tile size 仅分组统计首次读取字节，不拆分 attention。CUDA 支持 native SM90
BF16 / D128 / GQA16；不支持的路径和 Graph capture 明确失败，CPU 用于正确性参考。
增量压缩读取 device append，至多补读 31 个历史边界 token。

## 正确性验收

原全局 CPU 回归记录为 1339 passed、674 skipped、34 subtests passed；GPU 环境下的
SM90 offload 算子及 cache/model 专项共 62 passed，其中包含 CPU 检查。三次重复的
增强 trace 检查未重复计数。CPU skip 不算 GPU 验证。
完整 32 层 checkpoint 测试记录为 1 passed：resident / offload 从各自独立的空 cache
构建 64K sparse prefix，然后执行 1K extend，全部 normalized extend hidden 逐位一致，
`max_abs=0`。测试耗时不代表完整模型性能。

提交后的 cache 分配为 resident HBM 2,262,627,840 bytes，offload HBM 150,825,168 bytes，
offload pinned host 2,181,038,080 bytes。这些数字不含模型权重，也不是进程峰值显存。

## 算子重放性能

| 用途 | 原 run ID |
| --- | --- |
| 主计时，20 次重复 | `nosa_fused_stripe8_head1_20260930_01` |
| 独立确认，40 次重复 | `nosa_fused_stripe8_head1_confirm40_20260930_01` |
| 独立 profile，每层 3 次 | `nosa_fused_stripe8_head1_nsys_20260930_01` |

三个 replay 路径分别通过 FP32 reference 检查；serial / fused 输出逐位一致，resident 对照
使用原 BF16 容限。这与上述完整 checkpoint 的逐位一致检查是两组独立证据。

| Layer | 主计时 serial / fused ms | 延迟降低 | 独立确认 serial / fused ms | 延迟降低 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.472336 / 0.359904 | 23.80% | 0.460464 / 0.351344 | 23.70% |
| 15 | 0.544144 / 0.417376 | 23.30% | 0.542016 / 0.415440 | 23.35% |
| 31 | 0.542784 / 0.434944 | 19.87% | 0.550112 / 0.431776 | 21.51% |

表中为完整调用的中位数。两轮 serial control 都使用相同的新初始化，一次 fetch 完整 query
batch 的稀疏并集，再执行原整批 FA3。计时包含 initialization、planning、compaction、
prepare / sort / main / repair、launch gaps、输出和完成依赖，没有从 profiler kernel
时长拼接出完整延迟。每层主运行的 20 个样本和确认运行的 40 个样本，在 complete-call
CUDA time 与同步 wall time 上都由 fused 获胜；三种轮转 mode order 的 serial / fused
中位数比也均大于 1。

## 内部重叠证据

独立 profile 的 page-envelope 与非空 stripe-copy 两种 ratio 中位数，在 L0/L15/L31
均分别为 95.0938% / 95.3356% / 94.9440%。九个样本的两种 ratio 全部不低于 90%，
最小值为 92.2007%。每页 envelope 由其非空 stripe 的 min(start) / max(end) 构成，
不能把 envelope 中的空隙当作真实 copy。

本次样本的 page-envelope-only 全局 union 时长、额外 softmax intersection 和带符号的
page-minus-stripe fraction 都为零。这是全局区间并集的观测，不能推断每页 envelope
内部都没有空隙。指标分母是实际 stripe-copy 或 page-envelope 的区间并集，分子是它们
与内部 softmax-update 窗口的交集；softmax 窗口不包含 QK/PV MMA。因此该比例不等于
全部 attention hiding，也不代表 PCIe/CXL 链路占用。逻辑 KV 字节不证明物理链路字节。

独立 runtime/source 和 compiled-kernel review 通过；正式复算确认三个 run 的身份、原始
计时和 profile 证据，两组报告 JSON/CSV 都可逐字节重建。Nsight 还检查每个样本只有一个
fused main、准备操作分类正确，以及 serial fetch / attention 没有 kernel overlap。

## 结论边界

这些性能结果来自 NVIDIA H20Z（SM90、132 SM）上的固定输入单层重放。它们支持目标几何下
融合 fetch / attention 相对完整稀疏并集串行方案的延迟收益；完整模型和 serving 性能仍未测量。
计时排除 KV cache writeback、indexer / CIS 计算、输入加载、host registration、编译和 warmup，
没有 50 GB/s 限速。有限 HBM slots / eviction、跨请求 residency、CUDA Graph capture、
CXL/RDMA 仍未接入或验证。

输入、依赖、重复次数、数据和生成方式见[实验报告](../../../../experiments/nosa_offload_overlap/README.md)。
后续改动须按[执行与验收计划](implementation_plan.md)重新验收并生成新的 run ID，原数字不能
作为改动后实现的结果。
