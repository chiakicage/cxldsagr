# ECHO baseline cache 管理复现与 chunk size 实现计划

> **2026-10-03 撤回说明：** 本文涉及的旧 DeepSeek 4 GiB / W / chunk 对照已按用户要求撤回，
> 相关实验源码与运行产物已清理；下文仅保留当时的工程过程，不再证明当前容量或性能。
> 当前入口为[固定 P/NH 容量实验](../../../experiments/deepseek_v32_echo_cache/README.md)。

> 迁移定位：本轮实验已独立到 [deepseek_v32_echo_cache](../../../experiments/deepseek_v32_echo_cache/README.md)。
> 下文执行历史中的原命令、源码路径和哈希保持原样；产物现位置通过
> [迁移记录](echo_cache_experiment_migration.md)回查，目录迁移不构成新的测量。

日期：2026-10-03（验收层数已按当前 AGENTS.md 和用户确认修正为前三层）。状态：**P0–P6 完成。运行时、内存门禁、采样诊断、独立扫描及正式 GR 7 轮、896 请求均已验收；默认 C1024/W1024 已通过评审，报告已发布，受影响旧结果已清理**。
研究位置：T-008 / 2.5，关联 2.1、2.2、4.1、4.3。

当前证据入口为[实施 checkpoint](echo_cache_implementation_checkpoint.md)、
[内存验收](echo_cache_memory_audit.md)和
已发布的独立扫描（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/chunk_sweep/chunk_report.md`）。
前三层 64K+1K 独立空 cache 数值门禁、279 项 GPU 与 2,338 项 CPU 回归均已通过。
正式 GR 的五组首轮配置与 C1024/C2048 独立重复均已通过完整源码、产物和输出审计；
run ID 与首轮 C256 外部 auditor 恢复边界见[监控记录](echo_cache_freeze_gate.md)。
默认选择与清理 receipt 见[发布账本](echo_cache_publication_ledger.json)，最终结果见
GR 报告（已撤回，原路径：`../../../experiments/deepseek_v32_echo_cache/report/echo_chunks/results.md`）。独立三层扫描
与十 block GR 对照保持各自验收范围。

## 1. 目标、范围与源码身份

用户要求先复现 ECHO baseline，依据官方 SGLang 实现制定 cache 管理计划，并明确
chunked prefill 的 chunk size 如何选择。本计划交付正确的 cache 行为、资源归属、
预算与验收顺序，不扩展 NOSA 或上一轮提出的其他 GR offload 策略。

- 官方参考：`https://github.com/sjtu-zhao-lab/ECHO`，commit
  `bc1b75c1000010d0ac6f032ebaac283255c050b1`；本地 `3rdparty/ECHO/` 仅作只读参考，
  不纳入父项目 Git，不初始化它的嵌套依赖。
- 论文：[USENIX 最终版](https://www.usenix.org/system/files/osdi26-liu-guangda.pdf)，
  §3–5、§6.1、§6.4.3。下面把论文、官方代码和本地适配分别标明。
- 本地阅读基于 HEAD `1a9aa455ef101a64da255cb1828bfa8f6d9a3f0c` 及当前工作区。
  同时存在独立 MFU/MLA/linear/indexer 优化工作；此 HEAD **不是包含这些改动的源码快照**。
  实施前必须冻结实际源码及本地 include 指纹，协调 `echo.py` / `echo_logits.cuh` ABI，
  不覆盖正在进行的数值/寄存器优化。协调入口见
  [MFU 工作](deepseek_mfu_optimization.md)与[已有 cache 审计](deepseek_mfu_echo_audit.md)。
- 保留模型主 KV 的 BF16 512 latent + 64 RoPE 布局、resident FP8 index-K/scales、
  精确 top-2048、位置编码、causal mask、MLA 数学和输出范围。通用 cache 接受显式
  record width/dtype；576 维限制留在 DeepSeek 模型/算子适配。
- 保留 host backing，包括 candidate 的写回；**本轮不加入 candidate ephemeral、
  resident-build 后切换 offload、冷热用户整段晋升或新的替换算法**。
- GR 的固定 history 跨请求保留、candidate 后 truncate 是本项目适配。它不属于
  官方脚本已验证的 prefix cache 功能；官方相关脚本使用 `--disable-radix-cache`。
  本轮复现其层内 token cache，不声称复现整个 SGLang scheduler、PD 或 decode graph。

## 2. 官方行为与本地差异

以下官方源码路径相对 `3rdparty/ECHO/`。表中的本地现状记录制定计划时的起点和待改
差异，不代表当前实现仍有这些缺口；实施完成状态见上面的 checkpoint。实施者按固定
commit 核对，不能仅凭论文名称或本地 API 名称替代行为比较。

| 项目 | 官方事实与位置 | 制定计划时的本地现状 / 要求 |
|---|---|---|
| 资源归属 | `sglang/python/sglang/srt/mem_cache/memory_pool_host.py:856–899`：全局 host ID，按 layer 管理共享 device pool、allocator、priority 与双向映射 | 当前每 session × layer 独立分配 slots；改为 backend/model 拥有逐层共享池 |
| 执行顺序 | `models/deepseek_v2.py:1646,1684`（位于 `sglang/python/sglang/srt/`）：先 indexer，再 main attention；`nsa_indexer.py:716,742` 写 resident index-K 后运行选块 | 当前主 KV append 先于 indexer；按下节恢复官方顺序 |
| 新主 KV | `memory_pool_host.py:1126–1212`：直接分配 device slots、写 HBM，同时写 host backing | 本地 `append` 只 D2H，随后 `ensure(current)` H2D；消除这次强制往返 |
| 预取 | `layers/attention/nsa/nsa_indexer.py:520–559`：排序低 priority slots，kernel 实际 miss 才领取，结束后 `post_alloc` 与 priority finalize | 本地 `prepare_prefetch` 先淘汰最多 8192 个槽；取消预先清空 |
| 替换 priority | `memory_pool_host.py:1432–1444` 使用递增 `fifo_counter` 写时间戳；free 阶段会给 protected selection 更新时间戳 | 按事件复现；不是频率计数，不把本地“每次 ensure 更新全部 selected 的 age”当成等价实现 |
| 低分淘汰 | `nsa_indexer.py:53` 中 `EARLY_EVICT=False`，调用只见 decode 支路 | 不启用、不宣传为当前 baseline 行为；论文 §4.2 称 LRU-like，不能据此推定其他 policy |
| 精确 recall | `layers/attention/nsa_backend.py:638–656`：整批 exact top-k 中的实际 miss 去重、recall、再整批 MLA | 保留精确语义；官方没有 query-union 超 pool 的自动拆分，现有拆分必须标为本地容量适配 |
| 预取上限 | `DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh:1239,1355–1356`：`min(8192, P-Q)` | P 为可用 device slots，Q 为本次调用总 query 数；留出随后 append 的容量，不等于必须驱逐这么多记录 |
| extend offset | `nsa_indexer.py:502,551–555`：首次 chunk 清零、使用最后至多 4 行 logits 的 mean | 不是论文 §5.2 的 kth-score EMA；保留此算法身份。本地 finite-value guard 与按 session 保存 hint 是适配，单独记录，不偷偷换预测算法 |
| 预算 | 当前源码含 int64 reverse map、allocator bitmap/free stack、排序/去重临时空间 | 不能直接套论文 `4NH+13NG` 公式，也不能仅在 extend 返回后数 retained tensors |

官方在默认 flags 下不启用 fused prefetch；复现 ECHO inter-query 时必须明确设置
`SGLANG_NSA_FUSE_LOGITS_RECALL_EXTEND=1` 的等价本地开关，并记录全部相关 flags。

### 2.1 每层的目标执行顺序

1. 投影当前 Q、主 KV 和 index-K；写入 resident index-K/scales，声明 pending logical range。
2. 等待本次可能读取的旧 host KV 写回完成；准备 priority 排序和清零 allocation log。
3. indexer 与历史 prefetch 融合执行。此时当前 chunk 的主 KV 还未安装到 device pool；
   新 KV 投影 tensor 必须保持有效。预取排除当前 suffix，不能读取尚未初始化的 host 行。
4. kernel 完成后，依据实际 allocation log 更新 allocator/free-state 和 priority。
5. 完成精确 top-k，并按源码算法维护当前 session 的 offset hint。
6. 为当前 chunk 主 KV 分配 slots，直接从投影 tensor 写 HBM；并向 host backing 写入，
   保存源 tensor 的生命周期与完成事件。
7. 计算精确选择中的实际 miss，保护本次消费者所需记录，完成 guaranteed recall 与 remap。
   每次 recall（含后续消费组）都等待所读 host IDs 的写回事件，包含本次 append 的 suffix。
8. 执行 MLA。全部相关 GPU 操作完成后才允许消费者涉及的 slots 复用。

这恢复官方的 `prefetch → top-k → main-KV append → exact recall → attention`。
不能仅修 `append` 的复制方向，却保留不同的 cache 时序并宣称已经对齐官方。

### 2.2 必要的正确性补足与复现边界

官方源码本身存在未充分覆盖的状态，计划不复制潜在错误：

- **partial-free recall：**官方 extend recall 以 miss 数作为 free 参数，并在 debug 检查中
  假设有 miss 时 pool 已满。按安全通用语义计算
  `to_evict=max(0, misses-existing_free)`，空槽优先；在其满池正常域中行为一致。
- **选集保护：**官方保护是提升 priority，不是不可驱逐锁。执行前检查整个消费集合是否
  能装入 pool；可行时保证选择中的已有记录不被覆盖。若保护检查改变正常域的 victim
  集合，必须定位原因，不直接当作性能优化接受。
- **异步 host write：**官方部分事件等待位于后续 `set_mla_kv_buffer` 内，不能据此证明
  前面的 fused prefetch 已等待旧写回。必须在首次 host 读取前建立依赖，session 释放与
  host ID 复用前 drain；不依赖默认 stream 或“通常已写完”。
- **超容量：**官方正常域按整批 union 执行；本地可保留精确 query 消费拆分以满足硬预算，
  每次触发都计数并在结果中标注。它不能被描述为官方已实现能力。单 query 的 exact
  selection 都装不下时明确失败，不裁剪 top-k。
- **hint 与 nondeterminism：**正常有限输入核对官方 hint 更新；异常/非有限值的 guard
  作为修正披露。相同 priority 的 victim 次序以及并发预取赢家可能不唯一，不把不同
  合法 map 排列当作错误，也不能以此掩盖额外驱逐或重复读。
- **allocator 与计数寿命：**原 allocator 在多 CTA 同一 kernel 中读/写 available-size
  scalar，不能假定存在 grid barrier；移植为明确分阶段的 reserve/copy/finalize，
  actual allocation 小于 requested 时必须失败，不能默默截短。官方 victim helper 的
  priority 搜索范围约为 `[0,10000000]`，也未见 clock rollover；到边界前在全局静止点
  做保持次序与同分关系的 rank 压缩，保留 free/sentinel。此项作为寿命正确性适配单列。

## 3. 资源与接口设计

### 3.1 选定的结构

采用最接近官方的 **全局 host token ID + 每层共享 HBM pool**。不采用跨 owner
指针目录方案，不将每层 pool 合并为单层 staging。

| 所有者 | 状态 |
|---|---|
| backend / 完整模型，各 device | 固定容量 host arena（按该 device 上的模型层）、逐层 HBM records、逐层全局双向映射、priority/clock、allocator；共享 cache workspace 与受控 transfer streams |
| session | host token/page allocation、logical→global host page table、长度/兼容签名、独立 resident index-K/scales、offset、pending transaction 与事件 |
| 一次 layer operation | 对共享 workspace 的排他 lease、实际 allocation log、consumer 保护集合及完成事件；不跨调用累积 tensor 列表 |

逻辑 token ID 先经 session page table 转成全局 host ID，再查该层 host→device map。
反向表也使用同一全局 ID 域。当前 `echo.py` 内构造 `arange(columns)` 的做法必须移除，
page table 由 cache lease 提供。只共享 `records` 指针会在跨用户 eviction 时清错映射。

host arena 容量 NH 与每层 pool 容量 P 显式配置并在初始化前核算；采用与 page table
匹配的 64-token 分配粒度，尾页 padding 和预留 candidate 容量均计费。可保留各 session
连续的 resident index-K/scales 以复用现有计算接口，这是存储布局适配，不共享用户状态。
NH 不只由 DRAM 上限决定，还受全局 maps、resident indexer 与执行 workspace 的 HBM
约束。不能假设把主 KV 放到 host 后，就能无限保留用户。

session/layer view 显式分开 `committed_length`、`indexer_visible_end` 与
`kv_written_end`。prefetch 时 indexer 已看见当前 Q 的 index-K，而 main KV 尚未 append；
当前 host IDs 已预留但数据不可读。旧 `prepare_prefetch` 要求当前 chunk 已落在
`cache.written` 范围的断言须替换为阶段检查，并验证实际 prefetch 只访问已写历史。

device token eviction 只失效映射，不释放 host 历史；host 写已完成时不再写回 victim。
只有 host token 容量或 per-session HBM 索引预算不足，才通过现有 session LRU 释放
整个用户。固定历史身份变更与容量不足仍按原契约重建。
释放仅失效属于该 session 的全局 IDs，等待全部旧操作结束后才能复用 IDs；session view
带 owner/epoch 检查防止释放后使用。当前串行执行以严格 quiescence 防止 ABA，不引入
未经验证的并发服务。

### 3.2 文件与最小接口改造

| 文件 / 位置 | 计划职责 |
|---|---|
| `cache/sparse_token_cache.py`，必要时拆出 `cache/sparse_token_pool.py` | 分离 shared pool 与 session cache view；保留通用 width/dtype；处理 allocation lease、映射、保护、事务及计费 |
| `operators/deepseek_v32/indexer/echo.py`、`csrc/echo_indexer.cu`、`csrc/echo_logits.cuh` | 接收全局 page table、priority 排序与可复用 allocation log；按实际 claim 驱逐，返回实际分配事实；不改 GEMM 数学 |
| 同目录可新增 `cache_ops.py` 与就近 C++/CUDA helper | 封装官方 allocate/free/protect/post_alloc/mark-misses 对应 GPU 操作；由模型显式注入 cache，避免通用 cache 反向导入模型配置 |
| `models/deepseek_v32/echo_attention.py` | 注入 session view；恢复官方 layer 顺序，分离 index-K 写入与主 KV append；精确消费与 overflow 标记 |
| `models/deepseek_v32/echo_infer.py`、`serving_backend.py` | 分别持有模型/worker 共享 pool；协调逐层事务、输出、chunk、snapshot 与 GR truncate |
| `executor/serving_backend.py`、`serving/persistent.py`、`cache/prefix_pool.py` | 增加共享资源计划/一次性计费、host token 准入和 active workspace 预留；其他 backend 走零共享资源的兼容实现 |
| `experiments/deepseek_v32_echo_prefill/`、`experiments/gr_serving/` 的既有入口/审计 | 记录 pool scope、NH/P、chunk、scratch、cache policy revision、copy/eviction 计数与容量 fallback；不新造根实验脚本 |

建议接口顺序为 `plan_resources(budgets, limits)`（纯估算）→ `allocate_shared(plan)`
→ `acquire_session(token_capacity, reservation)` → `begin_step` / layer lease
→ `commit` 或 `abort`。`shared_bytes()` 与 `session_bytes()` 不重叠；
`release_session()` 不释放 shared pool，backend close 才统一释放。

`PrefixSessionPool` 的准入同时检查字节配额与 host token 空间：先判断请求是否能在
空池中成立，再逐个 LRU 释放直到两者都可满足，最后分配。不能在 backend 分配失败后
无条件清空用户，更不能为了 global arena 把 per-session byte reservation 假装成零成本。
host arena 已实际分配的 DRAM 只在 shared footprint 计一次；各 session 的 host token
占用另报，不再重复累加实际 DRAM 字节。

最小实现可给 `PrefixSessionPool` 增加 fixed/shared footprint 和 host-page 单位预算，
entry 保存占用页数；无需通用多资源调度框架。host allocator 分配非连续 64-token
pages，page table 负责映射，剩余页数足够即可分配，不额外制造连续大区间碎片问题。

旧 `--deepseek-slots` 表示每 session 的 pool。新共享模式使用明确的 pool-scope 配置
与新参数（建议 `--sparse-pool-tokens`），不得悄悄沿用旧语义。`serial_sparse` 复用
同一共享 pool/append/recall，仅关闭 fused prefetch；它是 ECHO 的必要消融。
其他方案只做共享接口兼容，不在本计划扩展策略。

### 3.3 预算账本

对每个 device 独立满足：

```text
HBM_cache = shared_records + global_maps/priority/allocator
          + shared_cache_workspace
          + sum(session_indexer/page_tables/hints/transaction_state)
          + active_pending_append_and_copy_sources
DRAM_cache = allocated_host_arenas + other_cache_owned_host_storage
```

所有项按实际 dtype、capacity、alignment 和 shared storage identity 统计。workspace
覆盖排序、scan、unique/bitmap、free stack、保护 mask、allocation log、remap、计数器、
cache/索引执行需要的临时数组；本轮保守将 indexer logits/top-k 临时空间纳入执行预留，
不能事后改名为普通 activation 以避开限制。模型权重及其余普通 activation 另报，物理
显存也须容得下完整执行峰值。
分配前预留；惰性增长计旧新 buffer 共存峰值；共享 workspace 按串行层执行复用。
active copy sources 则合计所有尚未完成的层/操作，不能因计算串行就假设只存一层；
设置有界 in-flight 写回数量，超过窗口先等待并释放 source，再继续分配。
禁止继续按 chunk 保存 GPU counter tensor 列表，使用有界计数器/累计统计。

## 4. 预取、recall 与事务不变量

### 4.1 融合预取

- prepare 只排序候选 slot，不清旧映射。空槽 priority=-1，保留 padding slot 0
  的 sentinel 语义；P 表示真实可分配数，不把额外 sentinel 行算成可用 slot。
- 按实时 global host→device 做 `MISSING→CLAIMED` CAS，独立 atomic rank 从排序结果
  领取唯一 slot。实际获批时才失效旧 owner 的映射并填写新映射/alloc log；预算溢出的
  claim 回退为 MISSING。不能用初始 h2d 快照替代实时 miss 判定。
- kernel 独占该层 pool，每个排序 slot 最多领取一次。其他 stream、session release、
  exact recall 和 attention 必须等 kernel 完成；提前发布物理 map 不代表数据已可读。
- 保留官方全池候选排序；不加入“仅空槽预取”“排除当前预测选集”等不同策略。
  `P-Q` 只是 cap；Q>P 必须在 launch 前拒绝，避免 unsigned underflow。
- finalize 只更新实际写入的 slots，padding 不能变成普通 slot。counter 可能 overshoot，
  实际 copy 数来自有效 allocation log/准确计数，不能直接使用 counter 原值。

priority 的具体事件不能被合并成“每次访问都更新 LRU”：

| 事件 | 对齐官方的更新 |
|---|---|
| 新主 KV 分配、prefetch finalize、exact recall 分配 | 对实际分配 slots 写当前 clock，再推进 clock；buffer 中 padding 0 最后恢复 sentinel。空批次的 clock 推进按参考调用点记录 |
| exact recall 前保护已有 selected slots | 每次写当前 clock 并推进，包括全命中、miss=0、to_evict=0；后续 recall finalize 即使 alloc log 全零也按参考推进 clock |
| 单独的 hit 查询 | 不额外加一次本地 age 更新 |
| eviction / release | 失效双向映射、priority=-1、更新 free state；不增加访问频次 |

append 分配后时间戳较新不意味着永远不会被后续 exact recall 淘汰；只保证当前消费者
的完整选择受到保护。超容量消费适配中可能发生新 KV 写回后的合法再次读回，须与原先
无条件 round trip 区分计数。

### 4.2 新 KV 与 guaranteed recall

- 当前投影 KV 直接写 HBM，host 写有明确事件和源 storage 生命周期；无条件的
  `GPU→host→GPU` 往返必须消失。后续真的 eviction 导致重读则合法，但须能从事件中解释。
- 精确 top-k 先映射到 global IDs，再做整批 union，忽略 invalid/padding IDs。
  仅实际 miss 分配；已有选中 records 在消费者完成前不可覆盖。
  容量拆分若需要重读本次新 KV，同样先等对应 D2H 完成，不能只等待旧 history。
- 对每次消费集合 U，检查 `|U|≤P`。不足时按确定的 query 顺序拆分消费并保留完整
  每-query 选择；该路径记 `capacity_split`。同一消费组内唯一读取，跨组重读如实计数，
  不把分组后的总流量说成整批唯一读取。
- 正常路径尽量保持 GPU-side count/scan/alloc，避免逐 query `.item()` / CPU ID
  往返；容量 fallback 的 host 同步如保留，必须计时并披露。

### 4.3 事务与可复现状态

- 保存本次开始的 session/layer transaction，所有层、输出和 GPU 完成后统一 commit。
  `echo_infer.py` 的失败路径补齐与 serving 一致的 drain，再 rollback。
- rollback / truncate 仅失效该 session 后缀的主 KV、indexer 可见范围、hint/pending
  状态，不 reset 全局池。已被合法驱逐的历史仍可从 host recall；不承诺失败后自动
  恢复所有用户的原 HBM residency。
- 若 CUDA 错误使状态无法安全恢复，将 backend 标为不可继续使用，不能声称 rollback
  成功后继续接请求。
- snapshot/restore 按全局 pool 保存一次 records、双向表、allocator/free-state、
  priority/clock，加 session page tables、有效长度和 hints。仅在全局静止时恢复。
  完整模型重复 extend 的状态恢复在计时外；真实 serving trace 不逐请求恢复热度状态。
  诊断快照容量单独披露，不作为 serving 可用 cache 容量或隐式额外 backing。
  快照重放限定相同 session 集合与不变的 host/indexer prefix 内容；发生 host ID 重用
  或历史变化后必须重建快照，不能只恢复 maps 就假定旧 backing 内容也已恢复。

## 5. Chunk size：约束、起点与选择协议

### 5.1 三种粒度不得混用

| 粒度 | 本轮定义 |
|---|---|
| 模型 prefill chunk C | 一批 token 顺序执行全部层，控制投影/MLP/indexer 临时空间和写回节奏；这是要选择的 chunk size |
| indexer kernel Q tile BQ | 官方与本地 DSA 的 64 index heads 对应 `BQ=128/64=2`；是 kernel 内部切片，不是 C，不在本轮调 tile |
| attention 消费组 G | 只有 exact union 超 P 时才拆分；不会重新投影、重新计算 index scores 或减少 top-k；另记 fallback 开销 |

本地完整模型外层与 attention runner 都有 chunk 循环；但当前 DeepSeek serving 的
`prefill` 直接把整段 history 送入 `_forward`，只有 attention runner 内部分块，
embedding/MLP 仍处理整段。这不等价于官方模型级 chunked prefill，需要新增外层
“每 chunk 顺序执行全部层”的调度，并与 MFU 工作协调由此改变的 GEMM 形状。
统一责任：outer prefill 决定 C，传入层的 token batch 就是该轮 indexer query
batch，避免不透明的二次 scoring 切分。extend 独立配置，当前 GR 128 candidate 默认
整批执行，不因 prefill C 改变而重切。64K+1K 模型验收同样显式记录 extend chunk。

不能简单循环现有 `_forward`：它会每次执行末 token LM head 并提交事务。应提取层执行
循环，一次 prefill 只 begin/commit 一次，LM head 仅执行原输出范围要求的位置；每个
chunk 内重新保存源前三层输入，10-block 替身仍复制该 chunk 对应 source 的输入。
不得跨 chunk 复用激活，或将副本串接成新的未验证轨迹。

### 5.2 官方数值不是通用最优值

- 论文 §6.1 写所有框架的 chunked prefill=2048。
- 当前 artifact 的 `sglang/e2e_test_mix_sharegpt.sh:175,199` 为 `-1`，关闭 serving
  chunked prefill；`e2e_test_pd_infini.sh:122,161` 的伪 P 节点为 16384，且它不启用
  offload。不能拿这些数值当同一 prefill-offload 条件下的测量结论。
- **2048 是复现起点，1024 是现有实验的必要连续性对照**；计划主扫描
  `C∈{256,512,1024,2048,4096}`。4096 仅在预算检查通过时进入，失败点报告约束原因。
  本轮依据完整轨迹及重复结果选择 C1024/W1024，限定于第 7 节的工作负载与预算。

### 5.3 可行性筛选

1. Q≤P，预取 cap 为 `min(8192,P-Q)`；Q=P 时可执行无预取的合法边界。
2. 每 query 的 exact selection 必须 fit P；整批 U≤P 才属于官方整批消费域。
   C≤P **不保证** U≤P，`C×2048` 也不等于实际 union。
3. logits、selection、投影与 pending source、metadata helpers 的峰值须 fit 同一预算。
   以下只是 `N=65536` 时单个 FP32 logits 张量与 int32 top-k IDs 的尺寸，**不是峰值**：

   | C | logits | top-2048 IDs |
   |---:|---:|---:|
   | 256 | 64 MiB | 2 MiB |
   | 512 | 128 MiB | 4 MiB |
   | 1024 | 256 MiB | 8 MiB |
   | 2048 | 512 MiB | 16 MiB |
   | 4096 | 1024 MiB | 32 MiB |

   还需算 causal mask、masked clone、top-k values/workspace、对齐和复制源；实际 N
   随 chunk 增长。分 chunk 能减小临时峰值，但不消除累计 history 与 resident indexer。
4. C 小会增加模型/管理 launch、host 往返机会和跨 chunk 重读；C 大会增加 union、
   排序/score 空间和 pool churn。更大的 Q-block 流水线窗口不自动带来整体加速。

还要核对实际 launch：当前 native `grid=min(SM数,ceil(Q/2))`，每 CTA 顺序处理其
分配到的 Q blocks。在 132-SM 几何下，128 candidate 只有 64 个 Q blocks，每个活跃
CTA 只有一个 block，缺少该 CTA 内“计算下一个 Q block”的持续窗口；256 query
同样通常只有一轮。跨 CTA 的工作区间仍可能相交，不能将其直接等同 §5.2 的持续流水
或端到端收益。512/1024/2048 的更多轮次是要测量的结构差异，不是加速保证。
prefill C 的调大不能改变真实 128-token extend 的 Q；不得填充伪 query、重复 candidate
或未经授权改成跨请求 batching 来制造重叠窗口。

### 5.4 测量与选值

以下为本轮预先确定的测量与选值协议；实际验收状态见文首和第 7 节。

**先固定正确实现和 NH/P。** 改成共享池后，P 的意义已变化；不能把旧 per-user
32768 slots 的容量结论原封不动搬过来。初始可保留 P=32768 作为明确标注的全局层池
控制点，预算允许且实验目的需要时另评估 P。比较纯 C 效果时，workspace 统一按扫描
最大 C 预留，保证变化 C 不悄悄改变 pool 或可保留 session 数。

**先完成每个 C 的数值检查，再比较完整冷 prefill。** 使用相同 64K history、真实
checkpoint 输入、独立空 cache；计入 cache write、排序/规划、indexer、prefetch、
exact recall、attention、MLP 和最后同步。只看 indexer 时间不能选 C。
JIT/权重加载不计时；计划至少 2 次独立 warmup、5 次 fresh-cache 冷构建测量，按
固定交错顺序运行 C，保留所有样本。硬件波动明显时扩大重复数，不能删除慢点选赢家。

**再检验 128-token extend 与缓存状态。** 对每个 C 构建后的自然 prefix residency
测量首次 extend，观察 C 通过残留 cache 状态带来的影响；另用完全相同的已验证 prefix
快照比较取数路径，防止把更热的起始状态误当成 kernel 收益。prefix 数学/选择的差异
必须先完成归因及输出验收：同一 C 的 resident/offload 从独立空 cache 构建，完整
prefix/extend hidden、末 token logits 与精确选集要求逐位一致；跨 C 的全部 extend
hidden 与末 token logits 保持既定 `rtol=0.01 / atol=0.02`，失败候选不进入性能排名。
跨 C 的完整 prefix hidden 与选择差异另外完整披露，不以额外的全 prefix 逐元素门槛
替代用户指定的 extend 输出范围。该区分源于固定输入诊断：FP32 index-head 线性层的
batch 几何可改变近并列 score 的排序，相同选集的不同消费顺序会改变 MLA 舍入；
只替换 index weights 已重现最早的差异。保留官方 baseline 的消费顺序，不修改精确
selection、不提高阈值，也不将跨 C prefix 宣称为无差异。每个固定状态的 extend
至少 5 次 warmup、20 次计时重复，恢复
在计时外，恢复后的全部 cache 状态都核对。

**最后选择服务配置。** 当前已有受控工作负载是 64K+128、16 用户顺序两遍、4 GiB
HBM / 64 GiB DRAM；若实施时该范围仍有效，以同一完整轨迹验证候选 C，不重启已被
否定的旧规模扫描。报告首访、复访重建、host hit、实际 HBM token hit 的数量与延迟。
现有 `prefix_hit_tier` 标签不足以说明部分 token 驻留，需另报比例。
固定池扫描先判断执行成本；部署选择另按每个 C 的实际 workspace 重新核算准入，
报告容量改变后的结果，二者不混成一个结论。

选择规则：通过正确性与预算门槛，在预先固定的完整轨迹上比较总执行时间/平均请求
延迟，同时报告复访延迟、冷构建时间和 p95。改善小于重复波动时不宣称最优，保留
较小 workspace 的候选；始终报告 1024 与 2048 对照。若冷构建最快的 C 损害复访，
保留两者取舍，不用自选首访比例拼出统一赢家。最终值连同 workload、NH/P、硬件和
implementation revision 固定下来，不宣称对所有 GPU/输入通用。

每层诊断至少记录：unique selection、fused 前命中、实际 prefetch、remaining miss、
append D2H、H2D、eviction、容量拆分/跨组重读、prepare/finalize/recall 时间。
随机调度下实际预取身份可不同，但总字节和 map 必须可核验。时间与流量诊断另跑，
不把 intrusive instrumentation 混入正式请求延迟。

## 6. 分阶段实施与验收

| 阶段 | 具体交付 | 完成门槛 |
|---|---|---|
| P0 行为契约与接口冻结 | 固定上述官方调用链、flags、priority 更新点、正常域/修正域；与 MFU 工作对齐 native ABI 和源码 | 对每项差异给源位置；本计划实施版本可追溯 |
| P1 共享资源与预算 | global host IDs/arena、逐层 pool、session views、shared/per-session/workspace 账本、host token 准入 | 两用户共享 pool 只计一次；跨用户同 local ID 不混淆；不可能请求无副作用失败；峰值预留先于分配 |
| P2 官方 layer 顺序与新写 | 分离 index-K / main-KV append，直接 HBM 写、host 事件与源生命周期 | 无 compulsory 新 KV H2D；copy 未完成不能复用；非默认 stream 与立即下一个 chunk 均正确 |
| P3 融合预取与 priority | 动态领取才 eviction、真实 allocation log、post_alloc/finalize、GPU metadata helpers | 零候选/零预算不驱逐；map 双向一致；无残留 claim；overflow counter 不被当实际 copy |
| P4 精确 recall 与事务 | partial-free 修复、consumer 保护、overflow 标签、全层 commit/abort、session-only truncate/global snapshot | cold host、满池、部分空池、A→B→A、host ID 重用、失败后继续或明确 poison 均符合契约 |
| P5 Chunk 配置与测量接口 | outer/extend/consume 粒度明确；NH/P 和 scratch 可行性检查；trace/metadata 新字段 | 默认不隐式二次切分；完整 candidate 输出与原范围一致；预算、scope、样本可审计 |
| P6 验收、补测、发布 | 独立数值验收、C 扫描、模型与 serving 对照、新 run ID 报告 | 新结果验证并发布后才替换受影响旧产物；旧结果不代表新 cache |

### 必须覆盖的有意义检查

- CPU reference：allocator/free/priority 事件、全局 ID 归属、字节与 token 双重准入、
  session truncate/release。不能把写成相同算法的测试当作官方行为证明。
- GPU metadata/transport：empty、full、partial-free、重复选择、全部 resident、零
  prefetch、cap=0/耗尽、同分 priority、无效 IDs、padding sentinel、host 写未结束、
  非默认 stream；逐记录验证复制内容与 map。正常域对照官方 helpers 或记录的输入/输出，
  并发赢家不要求 bitwise 相同，但 selection、ownership 与实际 copy 不变量必须相同。
- 压力测试：在小 P 上强制跨用户替换、超过整批容量但单 query 可容纳、单 query
  不可容纳、host ID 反复释放复用；确认不会误清其他 session，也不会裁剪精确选择。
- 模型：resident/offload 从独立空 cache 构建 sparse prefix，比较全部 extend hidden；
  按当前项目约束仅运行真实 checkpoint 第 0–2 层顺序传播的 64K+1K 验收，包含
  embedding、三个 dense MLP、final norm 与末 token LM head；不要求 61 层验收。
  10-block GR checkpoint 替身与非 GR 前三层 benchmark 分别标注，不相互替代。
  chunk 导致不同计算几何时先检查选择与数值，不能为了通过而直接放宽阈值。
- 回归位置：`cache/tests/`、`operators/deepseek_v32/indexer/tests/`、
  `models/deepseek_v32/tests/`，跨模块放 `tests/integration/`；不向测试目录写报告。
  显式 GPU 检查缺硬件/依赖须失败，不能用 skip 当验收。

## 7. 报告替换与最终验收

受影响消费者至少为 `experiments/deepseek_v32_echo_prefill` 与 `experiments/gr_serving`。
对应 measure/audit/snapshot 的 cache 格式与数据字段都要更新，不能只改内核后复用旧
prefetch/recall 计数。`serial_sparse` 随同共享基础设施重验；`dense_prefetch` 的继承
接口做兼容回归，NOSA 不进入本轮新策略比较。

性能/正确性修改后，旧 README、report 素材和对应 output 在新结果验收发布前保留，
标明 implementation/run ID 与“cache 改动后未运行”或实际补测状态。新结果通过后按影响范围替换，
不启动旧 4K/16K 或已撤回的规模矩阵，也不删除仍承载未受影响 NOSA 证据的混合产物。
详细范围沿用[已授权顺序负载记录](gr_serving_sequential_rerun.md)和实施时最新用户指令。
完整原始数据放各实验 `output/{data,log,profile}/<run_id>`，报告选定素材放 `report/`。
官方方法域与本地容量 fallback 的数字分开说明，不用旧 MFU 优化验证代替新 cache 验收。

共享 API/资源迁移、native 正常域差分、最终 CPU/GPU 回归和独立空 cache 的真实前三层
64K+1K 数值门禁已完成。四组内存门禁共接受 11 cases、352 requests、172 phases；
直接采样、解析形状上界及进程内存的区别见[内存验收](echo_cache_memory_audit.md)。
C1024/C2048 两组采样诊断也已验收，每组完整执行四方案 128 requests，详细流量只覆盖
预声明的 16 requests。权重从 `/preset-models` 读取，源码身份见
[实施 checkpoint](echo_cache_implementation_checkpoint.md)。

独立扫描 `20261003_echo_shared_chunks_02` 已重新筛选全部预算可行 C，并完成验收发布：
C256/C1024/C2048 通过数值门禁后计时，C512 因跨 C extend hidden 的 1 个元素超出原阈值
排除，C4096 因预算排除；C512 resident 内存检查通过不改变其数值排除。
原始样本、自然首次 extend 和公共 prefix 快照重复均保留于对应 run。

正式 GR 固定/部署 workspace 矩阵与 C1024/C2048 预声明重复共 7 轮、896 请求已接受，
完整源码、产物及全部 candidate hidden / logits 审计均通过。首轮 C256 未领先，
无需为领先配置额外补跑 C256；C2048 fixed/deployment 是同一共享控制，不重复计样本。
默认 C1024/W1024 已通过评审，四组 memory 通过
[workload 身份记录](echo_memory_formal_workload_binding.json)与正式请求绑定。
`20261003_echo_gr_chunks_publication_01` 的最终报告和 README 已发布；prepared03
清理已实际 apply（root session `35272` exit 0）：93 文件写入、486 个 DeepSeek-only
文件删除、1,590 个 retained NOSA 文件核验，receipt 记录在
[发布账本](echo_cache_publication_ledger.json)。报告保留 C2048 冷请求/p95 更低的取舍，
以及 ECHO 相比同 C serial sparse 慢 15.08% 的结果，不宣称融合预取获得整体收益。

报告层 workload 身份绑定修复的两组定向测试分别 37 项、20 项通过；修正后的外部
auditor 通过 111 项 CPU 测试。以上沿用执行 agent/root 的验证记录，无新 runtime
改动。详细运行边界见[实施 checkpoint](echo_cache_implementation_checkpoint.md)。
**P0–P6 已完成，结论限定于已发布的工作负载、预算、硬件和源码身份。**
