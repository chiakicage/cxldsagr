# KV cache 管理

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

当前只有模型设备上的 resident 后端：生产推理使用 HBM，CPU 用于独立数学测试。
未来完整 KV 可保存在 local DRAM，HBM 缓存当前访问块；本轮不分配 DRAM backing，
不实现复制、淘汰、预取或跨请求前缀保留。`layer_view` 的连续张量保证仅属于 resident
后端，不能成为未来 main sparse attention 的通用前置要求。
