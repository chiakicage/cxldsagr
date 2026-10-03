# GR serving 的两级 KV cache 与 motivation 实验

日期：2026-09-29。本文是实现规格，不是性能报告。

2026-09-30 主实验配置更新：固定 embedding + 前五层（3 dense FFN + 2 MoE）、64K 历史、
1K 候选，扫描 64/128/256/384/512 用户，64 为低压力控制点。增加有限 HBM-only 用户 LRU
对照，复访延迟包含淘汰后的重算。默认三层和下文三层算术保留为前期设计背景，不能套用
于五层容量；当前五层配置、预算与运行状态见 [实验 README](../experiments/gr_cache_serving/README.md)。
目前完成 64/128 用户四组和 256 用户 HBM-only；完整扫描被 pinned Host 内存容量阻塞。
本机总主存中的 256 GiB 位于 node 3 的 `ZONE_MOVABLE`，小规模 CUDA 注册探测会将页面迁移
到普通节点；不能将总主存视作可 pinned 容量。另须计入 Torch pinned allocator 的 2 次幂取整。

## 要回答的问题

GR 请求使用已有 `GR` 生成器：按用户热度有放回采样，同一用户的历史固定，
候选随 `visit_index` 更新。首次请求构建历史 KV，复访复用历史，只计算新的候选。
请求处理到 candidate extend 完成，不执行 LM head 或自回归生成。

这项工作属于 `cxldsagr` 的论文研究。`cxl-recsys` 的 Scheduler / Worker / MemPool、
CXL / RDMA 集成是另一项交付，本实验不修改或自动同步该仓库。

本轮三个 baseline 都使用 sparse attention；dense prefetch 指全量搬运 KV，
不是 dense attention。因此本轮直接检验的是 offload 的容量价值、选择性搬运的收益
和尚未隐藏的传输开销。若要单独论证 sparse attention 比 dense attention 更合适，
还需要同一 workload 的 dense attention 对照。本轮也不据合成 GR 文本评价推荐质量。

## 单卡实验对象

使用 DeepSeek-V3.2 的 embedding 和完整第 0、1、2 层，输出第三层 hidden states。
这是三层截断模型，不是完整 DeepSeek serving；不能把延迟乘以 61/3 当成全模型结果。
默认三层，保留一层与两层选项用于检查，不复制随机层，也不跳过第三层 FFN。

`first_k_dense_replace=3` 指前三层使用 dense FFN，attention 仍包含 DSA indexer。
主 attention 为 128 heads、MLA latent 512 + RoPE 64；indexer 为 64 heads、D128、top-k 2048。
三组使用相同权重、精度、RoPE、归一化 Hadamard、FP8 indexer 和稀疏选择语义。
旧 legacy 的 no-Hadamard 路径不作为原版 ECHO 的数值参考。
实际加载原 checkpoint 的 FP8 权重量化配置与 128x128 block scales，保留上游权重后处理；
activation 与 MLA KV 为 BF16，index K 为 FP8、scale 为 FP32。
“BF16 activation/KV”不表示所有模型权重均转成 BF16。

本机为 H100 PCIe 80GB、约 381GiB 总主存，其中约 125.50 GiB 为 managed non-Movable 内存。
只读 checkpoint header 的参数容量如下，
不含激活、KV、临时解量化及工作区，也不含最终 norm 和 LM head：

| 范围，均含 embedding | 原始权重和 scales | 假设 FP8 解量化 BF16、原 F32 参数保留 |
| --- | ---: | ---: |
| 前一层 | 2,451,455,424 B | 3,048,276,992 B |
| 前两层 | 3,049,552,768 B | 4,243,195,904 B |
| 前三层 | 3,647,650,112 B | 5,438,114,816 B |

第三列只是可选存储方案的容量估算，不是当前模型权重精度或实测分配量。
所需张量全部位于第一分片，按 checkpoint index 选择读取；不加载完整模型的其余权重。
保留源 checkpoint，不原地改写其层数或配置。正式运行记录张量清单、源码和环境指纹。
初版统一使用 BF16 MLA KV：每 token 每层 1152 B；FP8 index K 和 scale 另计 132 B。
不套用旧 SM120 的 656 B packed record。64K 历史、三层对应每用户 216MiB MLA KV
和 24.75MiB index；128 用户分别为 27GiB 和 3.09375GiB，元数据另外计费。

## 缓存怎么管理

### 生命周期

持久对象是不可变的 stable prefix，包括完整指令与历史。请求 session 只借用 prefix，
拥有自己的 candidate suffix 和临时工作区。完成一次请求不释放 prefix，不把候选接到
下一次复访的历史中。同一 prefix 的请求先串行执行；初版不做 continuous batching。

```text
到达请求 -> 查找 prefix
             未命中 -> prefill 历史 -> 写 host -> 等待完成 -> 发布 prefix
             已命中 -> 取得 prefix lease
          -> 准备 index -> 逐层 candidate extend -> 返回 hidden
          -> 释放 candidate/lease，保留 prefix 和允许保留的 HBM 副本
```

缓存身份由模型权重版本、计算/缓存精度、attention 语义、位置编码配置和完整 prefix
token 摘要共同确定；`user_id` 用于热度归属，不能单独证明命中。`history_sha256` 不包含
完整指令，也不能替代 token 摘要。初版不做跨用户公共指令的 radix 去重。

host 为 prefix 的主 attention KV 保存权威副本；HBM 中这些 KV 是可丢弃的副本。
首轮独立的 index K 仅保存在 HBM，它是复用 prefix 的必要状态，不能随意丢弃。
首访的各层 KV 在 GPU 产生后
写入 pinned host pool，全部层及 D2H 完成才能发布。冷 prefill 采用分块执行，其工作区
也受总 HBM 预算约束；不够则拒绝该配置，不暗中突破预算。热复访的 index 恢复、临时
candidate 写入、选块和 fetch 均属于请求成本。

candidate 的 KV 不必写入持久 host pool；但它在本次 suffix 的所有 chunk 间必须可见。
请求失败只丢弃私有 suffix；失败的首次构建不发布 prefix。DMA 或直接读取 host 的 GPU
kernel 完成，并且所有使用者释放前，
不能重用 slot；lease 防止活动 prefix 被 host 淘汰。host 容量不足时淘汰未被使用的完整
prefix，并同步失效其 HBM/index 映射；下次访问重新 prefill，计为容量 miss。

### HBM 放哪些数据

总预算拆成权重/激活工作区、index、持久 KV 副本、活动 suffix、fetch 双缓冲和元数据。
对照必须报告并限制总量，不能只限制名为 cache 的 pool。每层维护独立的逻辑 token 到
物理 slot 映射；NOSA 扩展时再加入 KV head 和 block 维度，不把 MLA 布局固化到共享接口。

首轮保持 ECHO 的 index K 常驻 HBM，显式计入全部已保留 prefix 的 index 容量；host
prefix 淘汰时一并释放。容量不足时减少保留用户或产生 miss，不把 index 放在预算外。
后续再单独评估 index 的 host 存档/按用户恢复，不将这项改变混入首轮 ECHO 对照。

目标性能对照中，三组跨请求 prefix 保留统一使用按字节计费的 LRU，host 以完整 prefix 为淘汰单位。
原版 ECHO 的 HBM 使用其原生优先级与 admission：fused prefetch 会直接更新 GPU pool，
不能称为完全相同的普通 LRU。`sparse_sync` 优先复用该管理器的原有非预取 recall 路径；
当前 dense 原型使用独立 staging/双缓冲并保留原 write pool，搬入全部既有 prefix，
不扣除 HBM hit，也不调用 sparse recall 做 admission；新 suffix 仍由原 pool 方法写入。
不能把它表述为“仅 attention 选中条目进入持久缓存”。主结果须同时记录各组
HBM policy 和预取造成的 admission/eviction，不能把全部差异单独归因于 overlap。
另在相同初始 HBM 映射上比较一次 extend，并用 ECHO 自身开关预取作消融。

热度策略先在 `sparse_sync` 内与明确实现的 LRU 独立对照：只观察已到达请求，
每 1024 次到达将频次衰减一半，当前用户加一。
host 用用户衰减频次 / prefix 字节数决定保留优先级；HBM 使用用户频次乘以该用户历史
请求中该层 KV 单元的选中比例，再除以条目字节数。每请求每条目最多计一次访问，同分按
LRU，仍同分按稳定 ID。初次选中的条目可 admission，但不能读取下一条请求或当前尚未
计算的选择。`user_heat_weight` 只用于生成输入和结果分组，不用于在线 policy 决策。
该策略是待验证的简单候选，不预设优于 LRU。

### 代码职责与最小接口

`serving` 增加用户 prefix 注册表、lease 和 FIFO 请求回放；`executor` 接收绑定 prefix
的 session，仍只认识 token tensor；`cache` 增加 host store、HBM pool、预算和 policy。
模型声明 record 布局与 index 状态，算子负责实际搬运与消费，不在实验脚本复制模型数学。

注册表提供 `acquire(prefix_key)`、`publish(prefix_key, snapshot)` 和 `release(lease)`；
pool 提供按字节预留、逻辑 ID 解析以及带完成事件的 fetch 结果。`CacheAccess` 增加
逻辑长度、index 访问和分层 KV 访问能力；resident `layer_view` 保持仅属于 resident 后端。
模型计算设备与 backing 位置分别声明，不能用 `cache.device == model.device` 排除 host。

未来 NOSA 接入还必须让 indexer 从压缩 K/CIS 和追加边界数据运行，不能因为原始 KV 在
host 就把整份 K 搬回 GPU。压缩窗口、稳定 pool、边界 tail 和 suffix 的提交/回滚一起管理。
本轮三层 DSA 实验不会自动完成这项 NOSA 接口改造。

## 三个 baseline 怎么比较

下表定义比较路径；当前原型的全量搬运消融和预算限制见后文，统一预算的性能对照尚未完成。

| 组别 | KV 获取 | 可重叠部分 |
| --- | --- | --- |
| `echo` | ECHO 原始 extend 预取 + 最终 top-k 后补取遗漏项 | fetch 与 indexer |
| `dense_prefetch` | 双缓冲逐层搬入全部既有 prefix，不扣 HBM hit；当前 suffix 由 GPU 生成，仍执行相同 sparse attention | 下一层搬运与当前层计算，实际重叠待测 |
| `sparse_sync` | 当前层完成选块后，对查询集合取并集，命中解析、按需搬运，全部完成再 attention | 不重叠 fetch 与计算 |

三组在同一个三层执行器中使用相同 prefix/suffix、模型后端、选择算法和 attention
kernel。ECHO 后端调用固定版本原始 fused indexer/recall，不能用 Python 预测替代后
继续命名原版 ECHO。`sparse_sync` 是借鉴 HiSparse 的阻塞消融，不称为完整 HiSparse。
先统一 prefix 保留策略、输入顺序、HBM/host 总预算，逐组标注 HBM policy；再单独
扫描在线热度 policy。修改 ECHO admission 的派生版本必须另命名 `echo_gr_adapted`，
记录补丁，并保留原管理器参考，不能静默替换 ECHO。

一次 1024-token extend 的访问工作集是逐 query 选择的并集，不能直接按 top-k=2048
计算请求所需容量。首轮三组均保持完整 1024-query extend，记录每层实际并集；
不额外拆小 query 来替 ECHO 改写其 inter-query 流水。空间不够时先淘汰非活动副本，
仍无法容纳必要工作集则报告配置不可运行，不能丢弃已选 KV、悄悄扩容或改短请求。
后续 attention overlap 方案再研究 tile 大小、跨 tile 复用和流水粒度，并单列消融。

当前 `dense_prefetch` 实现记录为 `dense_all_prefix_staging_v1`：不论原 write pool
是否命中，都将完整已有 prefix 经 pinned staging 搬入双缓冲；新 suffix 由 GPU 计算后补入。
它仍保留原 write pool，而不是用双缓冲替换全部持久 pool。额外 scratch、mapping、
原 pool、resident index 和临时空间都必须计费；统一总 HBM 字节预算尚未完成。
因此它现在是全量搬运消融，不是已经完成统一 admission/完整字节预算的最终 baseline。

全 HBM resident sparse 用作各组正确性参考和传输开销参照；HBM-only prefix LRU
用作是否值得 offload 的容量对照。这两项是必要控制组，不以预设输赢构造 motivation。
ECHO 重叠的是 fetch/indexer，不是同层 fetch/main-attention 的分块流水。
后者尚未实现；dense 的下一层 H2D 是否与前一层计算实际重叠也须由时间线验证，
不能由双 stream 推出。理想重叠下界只能写估计，不能写实测收益。

## Workload、测量和验收

沿用 `GR` 的 Beauty 默认曲线、weighted sampling、Poisson 到达、seed 42。
用户数默认 128，prefix=65536、candidate=1024，按实际 tokenizer 指令长度转换预算。
初版保存 512 条请求，检查稳定 prefix 和真实复访；完整测量建议 512 条缓存预热后
继续同一流的 2048 条请求。冷启动另从空缓存回放；不得用同一批候选预先填好 HBM。
输入仍是根据热度曲线生成的合成流，不是原始线上访问轨迹。

实验先做串行、尽快处理的机制测量，再按真实时钟回放 Poisson 时间戳并保留 FIFO 排队。
请求只串行，初版不宣称测试了并行 batch 的吞吐扩展。QPS 扫描按一次独立饱和率校准，
统一选择该率的 25%、50%、75%、90%、110%，三组使用同一组到达时间。过载下报告
队列增长、排空时间和超时，不只统计成功请求的延迟。

主 sweep 为 KV 副本、suffix、staging 和映射元数据合计预算 1/2/4/8GiB，另加相同的
index 预留，并报告完整总量；权重和 activation/workspace 单独实测。host 预算 32GiB，
128 个 64K 用户可完整 host 保留。另将 host 缩小至 8GiB，观察
淘汰后的 re-prefill；扩展用户数前先运行容量审计。控制 prefix=16K/64K、suffix=128/1024
和 uniform/Beauty 的敏感性，每次只改变一个轴。与原始 trace 长度不符时重新生成输入。

每个配置独立进程、固定预热边界、三个种子 42/43/44；无 profiler 回放用于主延迟，独立
重放用于 Nsight 时间线。报告首访、host hit、HBM hit、host eviction miss 各自的数量和
p50/p95/p99，另报 requests/s、排队/执行/冷 prefill 时间、实际 H2D/D2H 字节、selected
union、重复搬运、预取准确率、残余 miss、index 与元数据容量和峰值 HBM/host。

正确性验收覆盖：同一候选输入在重新 prefill 与复用 prefix 两条路径下结果一致、
候选变化不污染 prefix、内容或模型版本变化不误命中、
非对齐 prefix、长 suffix 分 chunk、host/HBM 淘汰后重取、预算耗尽、异常回滚、DMA 生命周期。
三种传输路径必须消费相同选择且 KV 内容一致；输出按相同后端确定性检查，浮点容差与参考
实现保持一致。不能用随机张量回放或单层 kernel 时间替代三层 GR 请求测量。

ECHO 的 `extend_logits_offsets` 是逐层、按 batch slot 存放的预取预测状态，不是
数学 index K/scale。原实现只在首个 cold chunk 清零；GR 复访绕过 cold prefill 后，
不能沿用上一用户在该 slot 上留下的阈值。适配器在 prefix 完成时保存阈值快照，
每次候选分支开始恢复，分支内允许更新、结束后丢弃。该状态计入元数据容量，
独立于不变的 prefix KV；这个 GR 生命周期适配需要记录，不能称为未修改的原版 serving。

### 数值控制与 kernel 派生修复

原生 top-k 的原子输出位置分配可能只改变同一 selected multiset 的排列，从而改变
FlashMLA 浮点归约结果；边界同分也可能改变 membership。正确性测试默认使用
`test_only_logical_topk_order_v1`，在原 indexer 返回后按逻辑 token 次序重排，逐行
验证 multiset 不变，保留重复项和 `-1`，不改预测阈值、不重新选择 token。
这个控制覆盖所有 prefill chunks/extends，不解决边界 tie，并含额外同步，不能进入
性能路径或被表述为原生确定性保证。`SPARSEGR_ECHO_TEST_TOPK_ORDER=native` 可禁用。
临时参考记录完整输入、checkpoint 元数据、ECHO revision/patch SHA 与控制 ID/源码 SHA。

64K 原版 ECHO fused prefetch 还存在另一项共享 phase flag 竞态：不同 warp 对
`s_prefetch_enabled` 的分支读取与下一阶段更新可交错，导致同步路径不一致。
`models/deepseek_v32/echo_kernel.py` 的 `echo_sm90_prefetch_phase_snapshot_v1`
显式把 flag 快照到线程局部变量，在同步后用快照判断分支；这是 derived kernel，
不是对 top-k membership 的修改，也不是 logical 次序控制可以代替的修复。
单 header overlay 不写回上游 checkout 或安装包；记录 source/patched header SHA、
installed include manifest SHA、C++ 扩展和初始化器源码 SHA，以区分原版与派生结果。
未改动 include 仍链接原安装树，使用期间不允许更新该包。

`open_echo_runner(..., kernel_patch=None)` 默认使用原 kernel；显式完整 patch ID 才启用
overlay。GPU 测试脚本仅为 `echo_gr_adapted` 默认选择 `phase_snapshot`，
`SPARSEGR_ECHO_TEST_KERNEL_PATCH=native` 可关闭。该选择先于 DeepGEMM/SGLang 导入，
覆盖整个 runner 生命周期；编译器初始化不可逆，每进程仅一次，切换版本须新进程。

当前 4K + 1K 与 64K + 1K 均在上述 logical 次序控制下完成四路径 `all` 正确性检查，
包括 phase snapshot 修复 ECHO 与同批 resident 文件交叉参考，容差保持
`rtol=0.02, atol=0.02`。64K 使用 V2 Beauty 128 用户池、512 请求中抽取的三用户子序列，
不是完整回放。此前 4K native 次序四路径检查也通过；64K native 次序不宣称通过，
原版 ECHO kernel 仍观察到挂起。尚无三组性能回放、统一 HBM 预算结果或
fetch/main-attention overlap 的实测结论。

## ECHO 接入事实与交付顺序

上游固定为 `sjtu-zhao-lab/ECHO@bc1b75c1000010d0ac6f032ebaac283255c050b1`，源码放
`3rdparty/ECHO/` 合适，暂为被忽略的独立 checkout。实验工具记录 commit 和补丁；
ECHO 自带修改版 DeepGEMM 必须隔离，不能替换顶层共享 DeepGEMM 或盲目共享不同版本头文件。
需要固定依赖的额外 checkout 放到该外部实验环境中，不扩散为本仓库通用依赖。

[ECHO 论文](https://www.usenix.org/system/files/osdi26-liu-guangda.pdf) 的重叠对象是 indexer；
它明确讨论 prefill 多 query 并集可能扩大工作集。公开配置还要求显式开启 extend 预取，
默认开关不能代表启用了完整 ECHO。[作者复现说明](https://github.com/sjtu-zhao-lab/ECHO/tree/bc1b75c1000010d0ac6f032ebaac283255c050b1)
使用多卡完整模型，其 mix 脚本关闭 radix cache；因此 GR 跨请求历史复用需要另外验收。
PD 脚本中有 fake prefill/transfer 配置，不直接用它报告真实冷启动与 KV 搬运。
[HiSparse](https://arxiv.org/abs/2608.07009) 也包含特定条件下的跨层预取，不能笼统称为无 overlap 系统。

按以下顺序交付，每步通过后再产生下一步性能数据：

1. 固定源码、核算三层参数/缓存、生成并审计真实 GR 请求；这些都是准备材料。
2. 跑通原始 ECHO 依赖和 SM90 kernel，构造三层无 LM head 前向，验证完整 indexer 数学。
3. 完成跨请求 prefix/host/HBM 生命周期，先用 resident 与阻塞 fetch 检查数值。
4. 接入逐层全量预取和 ECHO extend，跑同请求、同预算的三组机制对照。
5. 回放完整热度请求流、收集容量/延迟/传输证据，再评估热度 policy 与 attention overlap。

当前执行状态以 [实验 README](../experiments/gr_cache_serving/README.md) 为准；未执行步骤
明确写未运行。源码准备和容量算术不构成模型正确性或论文性能结果。
