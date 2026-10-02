# KV cache 管理

[prefix_pool.py](prefix_pool.py) 提供 serving 使用的 `PrefixSessionPool`：以用户固定
历史的 token 身份管理跨请求 session，按 LRU 整用户淘汰。分配前同时检查 HBM / DRAM
峰值预留，执行后核验实际 cache tensor 容量；模型提供布局、分配、统计与同步释放。
预算包含 KV、索引派生记录、映射、cache scratch、staging 和待提交 append，模型权重及
普通计算 activation 另计。候选 suffix 由 serving 在执行完成后 truncate，保留用户历史。
这一层是有限预算的用户 session 管理，不改变 NOSA 内部仍采用完整逻辑地址 staging、
没有页级有限 slots / eviction 的事实。运行入口见 [serving](../serving/README.md)。

当前预算核验依赖模型的预留与 tensor 统计。DeepSeek 在 recall、预取槽位回收和 remap
过程中分配的临时 cache scratch 尚需完整峰值审计，执行后采样不能证明硬预算覆盖了
所有瞬时分配；ECHO 的主动预取淘汰策略也待修正核查。见
[系统审计范围](../docs/agents/system/implementation-status.md)。

[manager.py](manager.py) 提供 `CacheManager` 与 `ResidentCache`。
模型通过 `CacheSpec` 声明层数、上下文容量上限、命名 record shape 和兼容信息；
共享管理器不假设 NOSA GQA 或 DeepSeek packed MLA 布局。
[NOSA 适配](../models/nosa/cache.py) 保留独立 K/V，形状仍为
`[layer, capacity, kv_head, head_dim]`，K 为 post-RoPE，V 保留原值。
NOSA sparse 模式额外声明 `[layer, capacity, kv_head]` 的 `cis_scores`，与 K/V
使用同一层写入、提交、重置和释放边界。dense 与 sparse cache 布局不能混用。

manager 通过 `allocate` / `release` 管理请求 session。一次模型 forward 内执行
`begin_step` → 各层 `write_layer` → `commit_step`；resident `layer_view` 提供包含本次写入的
连续前缀视图。全部层和输出成功后才推进有效长度，异常通过 `abort_step` 保留旧长度。
`reset` 重用分配好的存储，`release` 使 session 失效并放弃其 buffer 所有权。
`stats` 返回容量、resident/host 字节数和有效长度。

辅助状态归请求 session 按 layer 保存；CIS 的模型语义由 NOSA 声明，管理器仅管理命名 record。
manager 通过弱引用记录 session，兼容既有 `model.new_cache()` 使用方式。

[indexer_cache.py](indexer_cache.py) 提供 `IndexerCache`：接受显式 record spec，惰性分配
逐层派生数据，并保存 committed/pending 有效长度和请求共享 workspace。它不导入模型
配置或计算压缩语义。NOSA sparse cache 持有该模块，SM90 indexer 首次访问时补齐前缀，
后续只追加完整压缩窗口和稳定 pooled CIS；未使用 indexer 的层仍可提交原始 KV。
派生数据与 KV 一起 commit/abort，计入 session 内存统计。`NosaKVCache.truncate(length)`
同步回退所有有效长度；旧 `length=` 只允许合法倒退，存在 opaque state 的非零倒退会拒绝。

`CacheManager` 接受模型提供的 session allocator；默认 NOSA 使用 `ResidentCache`，
显式 `cache_backend="offload"` 使用下述 host backing 适配。CPU placement 用于独立
数学测试。`layer_view` 的连续张量保证仅属于 resident 后端，main sparse attention
通过 cache access 获取实际布局。

## NOSA host backing

[host_backing.py](host_backing.py) 提供 `HostBackingCache`：按模型声明的命名 record
shape 分配 pinned CPU backing，持有本次 append 的独立 device 副本，并在私有
transfer stream 写回主存。它不提供完整 resident `layer_view`；提交、回滚、截短、
重置和释放先等待 GPU 使用结束，全部层成功后才发布新长度。

[NosaOffloadCache](../models/nosa/offload_cache.py) 保存 pinned 历史 K/V，CIS 与
压缩 K/CIS、稳定 CIS pool 留在 HBM。增量压缩只读取本次 device append 和最多
31 个历史边界 token，不为 indexer 重载整个 K prefix。KV、CIS 和派生记录统一
commit/abort/truncate；CPU reference 保留相同事务语义。

请求共享 [SM90 NosaFetchWorkspace](../operators/nosa/attention/offload/api.py)，为一层的完整
逻辑 token 地址范围分配 HBM staging，按 `(KV head, block)` 去重并稀疏读取。当前
融合主 kernel 的所有 CTA 都执行 attention，启用 fetch 的 CTA 以三个空闲
producer warp 动态领取唯一页中的 8-token stripe；每个历史 K/V 向量只读一次。
每 stripe 完成写入后以 acq_rel RMW 累计 ready，TMA acquire 观察到 8 后复用 HBM。
尾页空 stripe 不读取 host，仍参与完成协议。跨层 staging、queue 与 scratch 均须
等待前次调用完成后复用，不依赖 host memory 的 L2 缓存。
该分配不是有限 slots / eviction cache，CIS 和派生记录仍按层常驻。
`stats()` 分别计入 host backing、resident
append、CIS、派生记录与已分配 staging / queue / scratch，不能将它当成进程峰值显存。

CUDA offload 当前要求 SM90、BF16、D128、GQA16 和 native attention。融合版本中
`offload_query_tile_size=128` 仅控制首次读取流量的 query 分组，attention 处理完整
query batch；`offload_fetch_ctas=96` 限制同时参与 fetch 的 attention CTA 数，实际
数量不超过 cooperative grid 大小，不单独保留 fetch CTA。串行与融合共用 native
initialization，合并 metadata reset、page-0 padding 与 strided suffix staging；
首用规划仍为后续独立 launch。
完整 32 层 checkpoint 检查 1 passed：resident/offload 分别从独立空 cache
构建 64K sparse prefix，再执行 1K extend，全部 normalized hidden 逐位相同，max_abs=0。
提交后的 cache 分配为 resident HBM `2262627840 B`、offload HBM
`150825168 B`、offload pinned host `2181038080 B`；不包含模型权重，也不是进程峰值显存。
单层性能与 stripe / page-envelope overlap 结果见 [NOSA 模型](../models/nosa/README.md)及
[offload 实验](../experiments/nosa_offload_overlap/README.md)。此实现使用本机 pinned
DRAM，未验证 CXL/RDMA，也不提供跨请求 prefix residency。

## DeepSeek ECHO token cache

[sparse_token_cache.py](sparse_token_cache.py) 提供独立的 `SparseTokenCache`，由模型显式
提供 record 宽度与 dtype。支持完整 resident 分配，或 pinned CPU DRAM backing 加有限
HBM slots；维护 logical/physical ID、淘汰、当前工作集保护、精确 recall 与传输计数。
[DeepSeek SM90](../models/deepseek_v32/README.md) 为主 KV 使用 BF16 512 latent + 64 RoPE
record，indexer K/scales 仍常驻 GPU。ECHO indexer 使用预先腾出的 slots 融合预取，
attention 前精确补取剩余 miss；选择并集超过 pool 时拆分 query 消费，不改变每 query 的选择。

完整模型为各层调用 `begin_step`，逐 chunk `append`；全部 61 层、输出与 GPU 同步成功
后统一 `commit`。失败只回滚本次启动的 step，保留已提交历史；支持 `truncate` / `reset`。
实验在计时外保存与恢复 prefix 的 HBM residency，避免把重复 extend 的热缓存当成基准。
这条路径使用本地 DRAM，不包含 CXL/RDMA，也未接入 NOSA 的 `CacheManager`。
修复前完整 61 层的 64K + 1K resident/offload 测量已验收；KV gather 对齐修复后的
完整模型性能待补测，版本与结果见
[ECHO 实验](../experiments/deepseek_v32_echo_prefill/README.md)。
