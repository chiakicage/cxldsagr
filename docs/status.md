# 当前系统状态

核对日期：2026-10-01。本文根据当前代码、模块 README 和已发布实验整理，没有运行
新的性能测量。报告数字保留原 run ID 和实现边界；目录迁移后的当前性能不能直接由
旧数字代替。

项目已完成 NOSA sparse offload 的算子原型、完整模型数值验证和单层性能验证。
距离研究设想中的多用户 serving，主要还缺跨请求 history KV 复用、NOSA 有限 HBM
页池，以及包含请求排队的延迟与吞吐测量。

## 请求如何经过系统

```text
GR 请求生成器 → serving 请求适配 → executor 分块执行 → models/layers → operators
                                      ↕
                              cache 请求事务与存储
```

| 模块 | 当前职责与实现 | 当前边界 |
| --- | --- | --- |
| [GR](../GR/README.md) | 固定 history、每轮变化的 candidate、精确 token 预算、热度曲线和模拟到达时间 | 不持有 KV、不执行模型；合成内容无真实推荐标签 |
| [serving](../serving/README.md) | 单进程、单 GPU 串行处理请求，拆分 stable prefix 与 candidate suffix | 每请求分配、prefill、extend、释放；没有跨请求复用、批调度或到达时间回放 |
| [executor](../executor/README.md) | 对 token tensor 分块执行所有模型层，选择输出 | 不读取 GR 请求；当前只保留最后一个 chunk 的输出 |
| [models](../models/README.md) / [layers](../layers/README.md) | 模型配置、权重、位置编码、选择语义、KV 布局及共享普通层 | 共享执行分层当前接入 NOSA；DeepSeek ECHO 保持独立路径 |
| [cache](../cache/README.md) | 请求分配、逐层写入、提交、回滚、截短和释放；管理原始及派生记录 | 没有按用户长期持有的 prefix 目录；有限 token pool 当前用于 DeepSeek |
| [operators](../operators/README.md) | 按模型提供 indexer、attention、linear 等后端，共享通用 record 搬运 | 当前主要平台为 SM90/Hopper；SM120 执行代码已移除 |

当前 serving 返回 candidate 末 token 的 normalized hidden，不运行 LM head 或
文本生成。研究场景只有 prefill；模型 CLI 具备 decode 能力，不代表本项目的 serving
负载包含 decode。生成器的 `common_prefix_tokens` 是共同文本前缀长度，不是实际
缓存命中量；模拟 `timestamp` 也不是系统已服务的请求时间。

## NOSA

### 模型与 resident 路径

默认模型模式为 dense，显式 `attention_mode="sparse"` 启用完整 NOSA。NOSA-8B
使用 32 层、32 Q heads、2 KV heads、D128。完整 sparse policy 采用 64-token block：
含 sink/local 的 query-aware 阶段保留 33 块，再由 query-agnostic CIS 补满 64 块，
attention 同时使用 CIS 加性 bias。

QA-only pattern 分析采用另一套 policy，不能与完整 NOSA 混用。Resident indexer
和 block sparse attention 已接入 SM90 CUDA/CuTe 与 Triton；增量压缩 K/CIS 和
稳定 CIS pool 随 KV 一起提交、回滚和截短。代码与形状范围见
[NOSA 模型](../models/nosa/README.md)和[算子](../operators/nosa/README.md)。

### Offload 路径

显式 offload 目前支持 native SM90、BF16、D128、GQA16。历史 K/V 在 pinned local
DRAM，CIS、压缩 K/CIS 和稳定 CIS pool 仍常驻 GPU。Indexer 更新压缩记录时只补读
必要的历史边界，不为选块重新加载整个 K prefix。

每层调用按 `(KV head, logical block)` 去重，每个 64-token 页拆成 8 个不重叠的
8-token stripe。同一个 cooperative 主 kernel 内，默认最多 96 个 attention CTA
使用 producer warpgroup 中的三个 warp 读取 host，同时保留 persistent FA3
计算。页完成同步保证 attention 只消费已就绪数据；每个选中的历史向量只执行一次
host load，后续复用来自 HBM。完整协议与资源约束见
[offload 实现](../operators/nosa/attention/offload/api.py)和[KDA 文档](kda/README.md)。

`overlap=False` 的对照一次加载完整 query batch 的精确稀疏并集，再执行整批
attention。当前 `query_tile_size` 只用于首次读取流量统计，不拆分 attention。

HBM staging 覆盖一层完整逻辑地址范围，并在层间复用。它减少了各层完整 K/V 的
常驻需求，但尚无有限 slots、地址映射与 eviction。也没有跨请求复用已取回的块；
多用户场景下，仍 resident 的派生记录如何受容量约束尚待设计。CUDA Graph capture、
CXL/RDMA 不属于已支持或验证的 offload 路径。

## DeepSeek V3.2

[SM90 ECHO](../models/deepseek_v32/README.md)支持完整 checkpoint 的 61 层、
embedding、dense/grouped MoE、final norm 和 LM head，按 token chunk 执行所有层。
主 KV 使用 BF16 512 latent + 64 RoPE record，indexer FP8 K/scales 仍 resident。

其 offload 复用[有限 HBM token pool](../cache/sparse_token_cache.py)和 pinned DRAM
backing，在 indexer 中融合 prefetch，再用精确 top-k/residual recall 补齐实际需求。
工作集超过 pool 时拆分 query 消费，不裁剪每个 query 的选择。该路径的有限 pool
能力不能归于 NOSA；ECHO 的 indexer/prefetch 融合也不同于 NOSA 的 fetch/attention
融合。当前优先推进 NOSA，DeepSeek 保留为另一模型的实现和验证基础。

## 已有证据与补测状态

### NOSA offload

完整 32 层数值检查从独立空 resident/offload cache 构建各自的 64K sparse prefix，
再执行 1K extend，全部 normalized hidden 逐位一致，`max_abs=0`。这是正确性
验收，不是完整模型 offload 性能测量，也不等同于推荐质量验收。

单层性能报告在 NVIDIA H20Z、132 SM、SM90 上回放固定 L0/L15/L31 输入。以下是
2026-09-30 原实现的完整调用中位数，包含准备、fetch、attention、repair 和 launch
gaps，排除 indexer/CIS、KV 写回、host pinning、编译与预热。

| 层 | 整批稀疏并集串行方案 | 融合方案 | 延迟下降 |
| --- | ---: | ---: | ---: |
| L0 | 0.472336 ms | 0.359904 ms | 23.80% |
| L15 | 0.544144 ms | 0.417376 ms | 23.30% |
| L31 | 0.542784 ms | 0.434944 ms | 19.87% |

主 run 为 `nosa_fused_stripe8_head1_20260930_01`，5 次预热、20 次重复；独立 40 次
确认 run 为 `nosa_fused_stripe8_head1_confirm40_20260930_01`。独立 profile run
`nosa_fused_stripe8_head1_nsys_20260930_01` 每层 3 个样本，全部 9 个样本的
stripe-copy 与 page-envelope 两种 copy/softmax 相交比例都超过 90%，最低 92.20%。
该比例不表示 90% 的完整传输延迟已被隐藏，实际提速由上表独立计时判断。

原测量 fused build key 为 `484e32532ba74fb6`，具体源码身份以该 run 的指纹和快照
为准。2026-10-01 算子目录迁移后未重新性能测量。环境、误差容限、字节核验和全部
数据见[正式 offload 报告](../experiments/nosa_offload_overlap/README.md)。

该次 checkpoint 检查记录的 resident cache HBM 为 2,262,627,840 B，offload cache
HBM 为 150,825,168 B，host 为 2,181,038,080 B。这些是 cache 分配量，不含权重，
也不是进程峰值显存或多用户容量测试。

### 其他实验

| 实验 | 可以支持的结论 | 当前状态 |
| --- | --- | --- |
| [NOSA dense 64K+1K](../experiments/nosa_gr_65536_1024/README.md) | Full Attention 的单请求计算成本 | 已有报告，不包含 sparse offload；计时在 nsys 进程内，不能与 sparse 的独立进程计时直接拼成严格加速比 |
| [Resident sparse 完整模型](../experiments/indexer_block_sparse_profile/README.md) | `94bf521` 下 native/Triton 的完整模型前向及模块分解 | 9/29 两组 sparse run 已发布；9/30 扩展模型/cache 源码图后的当前分支尚未补测 |
| [Kernel MFU](../experiments/nosa_kernel_mfu/README.md) | `kda_main_bf16_pair_v3_development` 的完整模块检查点 | 三层 indexer 约 24.7–25.5% useful MFU；attention 仅 L31 达 40%，目标未全部完成；synthetic 对照待补测 |
| [选块 pattern](../experiments/nosa_indexer_pattern_65536_1024/README.md) | QA-only 分析与旧 full-NOSA 轨迹的选择统计 | full-NOSA 受数值修复影响，待补测；不能据旧轨迹推断当前模型选择 |
| [DeepSeek ECHO](../experiments/deepseek_v32_echo_prefill/README.md) | 原完整 61 层、64K+1K resident/offload 的末 token logits 一致性及原延迟 | KV gather 对齐修复后的完整模型性能待补测；旧测量输入未触发该非对齐问题 |
| [CPU DRAM](../experiments/cpu_dram_bandwidth/README.md) | 本机 CPU 内存访问的带宽背景 | 不能替代 GPU-host 稀疏加载、CXL 或 serving 测量 |

详细 run ID 与限制以各实验 README 为准。NOSA 的 64K 实验扩大了运行时上下文上限，
没有评价超出 checkpoint 默认 32768 上下文后的推荐质量。旧 DeepSeek/SM120
测量集中在[历史归档](../experiments/legacy/deepseek_v32/README.md)。

## 下一阶段的完成标准

需要把算子结果推进到真实请求生命周期：同一用户复访时直接复用固定 prefix，
candidate 后缀不污染下一轮；NOSA 的 HBM 和 host 占用受明确预算约束；完整模型
offload 测量覆盖 indexer、缓存更新和同步；最后在包含到达与排队的请求流上验证
延迟和吞吐。具体拆分见[后续计划](roadmap.md)。
