# KV cache 管理

[manager.py](manager.py) 提供 `CacheManager` 与 `ResidentCache`。
模型通过 `CacheSpec` 声明层数、上下文容量上限、命名 record shape 和兼容信息；
共享管理器不假设 NOSA GQA 或 DeepSeek packed MLA 布局。
[NOSA 适配](../models/nosa/cache.py) 保留独立 K/V，形状仍为
`[layer, capacity, kv_head, head_dim]`，K 为 post-RoPE，V 保留原值。

manager 通过 `allocate` / `release` 管理请求 session。一次模型 forward 内执行
`begin_step` → 各层 `write_layer` → `commit_step`；resident `layer_view` 提供包含本次写入的
连续前缀视图。全部层和输出成功后才推进有效长度，异常通过 `abort_step` 保留旧长度。
`reset` 重用分配好的存储，`release` 使 session 失效并放弃其 buffer 所有权。
`stats` 返回容量、resident/host 字节数和有效长度。

辅助状态归请求 session 按 layer 保存；本轮仅提供状态对象的访问和提交边界，不维护 CIS。
manager 通过弱引用记录 session，兼容既有 `model.new_cache()` 使用方式。

当前只有模型设备上的 resident 后端：生产推理使用 HBM，CPU 用于独立数学测试。
未来完整 KV 可保存在 local DRAM，HBM 缓存当前访问块；本轮不分配 DRAM backing，
不实现复制、淘汰、预取或跨请求前缀保留。`layer_view` 的连续张量保证仅属于 resident
后端，不能成为未来 main sparse attention 的通用前置要求。
