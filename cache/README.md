# KV cache 管理

[capacity.py](capacity.py) 定义共享资源和 session 的容量计划；
[lifecycle.py](lifecycle.py) 管理准入 owner、session 注册、执行 lease 和故障状态。
模型负责实际分配、事务提交、异步完成检查和释放，公共状态机不重复执行这些操作。

`allocator/` 保存 [allocation bounds](allocator/budget.py)、
[fresh snapshot provider](allocator/snapshot.py) 和
[Python MemPool provider](allocator/pool_referrers.py)。当前由 NOSA 显式调用，
迁移到公共目录没有给 DeepSeek 增加 allocator 扫描。原有 rounding、每次调用的新 snapshot、
CPython/头文件认证、动态 getter 和 GC 变化后的重新扫描保持不变；native 源码及 ABI
白名单位于 `allocator/csrc/`。按 let it crash 要求，provider 初始化、编译或认证
失败会直接报错，不再自动切换实现。错误传播、双故障和资源保留的验收见
[错误处理证据](../docs/agents/acceptance/unified_runtime_20261005/let_it_crash_summary.md)；
完整请求结果见两个模型的 motivation 报告。

[prefix_pool.py](prefix_pool.py) 提供 serving 使用的 `PrefixSessionPool`：以用户固定
历史的 token 身份管理跨请求 session，按 LRU 整用户淘汰。分配前同时检查 HBM / DRAM
峰值预留，执行后核验实际 cache tensor 容量；固定 P/NH 模式按对应的 host 页或
HBM history-token 配额准入。模型提供布局、分配、统计与同步释放。
预算包含 KV、索引派生记录、映射、cache scratch、staging 和待提交 append，模型权重及
普通计算 activation 另计。DeepSeek GR 的 `echo/serial_sparse` 只为 history 准入，
候选使用共享 GPU 临时空间，正常结束后 discard，失败直接报错终止；该公共入口的其他后端
（包括普通 NOSA budget 后端）仍由 serving 在执行后 truncate 候选 suffix，保留用户历史。
普通 NOSA budget 路径采用完整逻辑地址 staging 和整用户 session LRU，尚无页级
有限 slots / eviction。独立的 NOSA 固定 P/NH 入口见下文。运行入口见
[serving](../serving/README.md)。

[staging.py](staging.py) 为模型提供两个固定容量的命名 record buffer 及串行 lease，
不选择预取内容。槽位身份包含 session、generation 和 layer；写入等待旧 consumer，
消费等待 copy ready，异常归还也等待未消费的预取。无法确认完成时保留所有权并禁用
后续复用。NOSA 与 DeepSeek dense backend 均已接入；两者的模型布局和资源计划分别由
对应 backend 提供。
完整生命周期与预算入口的早期集成验收见
[原始验收索引](../docs/agents/acceptance/unified_runtime_20261005/shared_cache_integration_evidence.json)，
其结论只适用于记录中的冻结源码。旧热度短轨迹
已结束独立实验维护，通用预算实现和回归保留；当前性能对照见
[固定容量 motivation](../experiments/nosa_motivation/README.md)。

DeepSeek 分别预留共享 pool、session 私有状态和执行 workspace。分配后 tensor 统计
不能单独证明执行峰值；PyTorch allocated、reserved 与设备已用量分别报告。
当前容量声明、静态规划和完整请求观测见
[两模型 cache 管理报告](../experiments/cache_management/README.md)。固定用户轨迹
成功执行，不代表用满 NH 配额或采用离线最大 P/NH 时仍能装入机器。

[manager.py](manager.py) 提供 `CacheManager` 与 `ResidentCache`。
模型通过 `CacheSpec` 声明层数、上下文容量上限、命名 record shape 和兼容信息；
共享管理器不假设 NOSA GQA 或 DeepSeek packed MLA 布局。
[NOSA 适配](../models/nosa/cache/resident.py) 保留独立 K/V，形状仍为
`[layer, capacity, kv_head, head_dim]`，K 为 post-RoPE，V 保留原值。
NOSA sparse 模式额外声明 `[layer, capacity, kv_head]` 的 `cis_scores`，与 K/V
使用同一层写入、提交、重置和释放边界。dense 与 sparse cache 布局不能混用。

manager 通过 `allocate` / `release` 管理请求 session。一次模型 forward 内执行
`begin_step` → 各层 `write_layer` → `commit_step`；resident `layer_view` 提供包含本次写入的
连续前缀视图。全部层和输出成功后才推进有效长度，异常通过 `abort_step` 保留旧长度。
`reset` 重用分配好的存储，`release` 使 session 失效并放弃其 buffer 所有权。
`stats` 返回容量、resident/host 字节数和有效长度。

辅助状态由请求 session 按 layer 保存；CIS 的模型语义由 NOSA 声明，管理器仅管理命名 record。
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
遵循[模型 attention 契约](../models/attention_contracts.py)，通过 cache access 获取实际布局。

`PrefixSessionPool` 还支持与 host pages 独立的 HBM history token 配额，用于固定
P 的 HBM-only serving。准入、LRU 和实际占用审计分别检查两种配额。resident
`SparseTokenCache` 可带独立 candidate 容量：候选只在 GPU 临时写入，结束后丢弃，
不推进历史长度，也不占 history token 配额。

## NOSA host backing

[host_backing.py](host_backing.py) 提供 `HostBackingCache`：按模型声明的命名 record
shape 分配 pinned CPU backing，持有本次 append 的独立 device 副本，并在私有
transfer stream 写回主存。它不提供完整 resident `layer_view`；提交、回滚、截短、
重置和释放先等待 GPU 使用结束，全部层成功后才发布新长度。

[NosaOffloadCache](../models/nosa/cache/offload.py) 保存 pinned 历史 K/V，CIS 与
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
上述为普通模型 owned 模式。普通 NOSA budget serving 注入 backend resources 后，session 只计自身
历史/派生/append/indexer scratch，共享 staging/queue/FA3 scratch 单列计费；跨用户
和跨层借用都在同一执行 lease 内，不跨用户共享历史数值。

独立的 [NOSA fixed cache](../models/nosa/cache/fixed.py) 与
[fixed resources](../models/nosa/execution/fixed_resources.py) 为固定 P/NH 入口保留逐层有限 P 槽，
以逻辑页偏移和 session tag 直接映射。HBM-only 按 P 个 history token 做 session LRU，
offload 按 NH 准入并随 session 分配 host backing；当前要求 H<=P，history 与 prefill
chunk 按 64 token 对齐。候选整批在 GPU 执行后 discard，主 K/V 不写回 host history，
CIS/indexer 候选尾部仍按实际 session storage 计费。该路径复用已取回的历史块，
不提供任意容量下的 token LRU。当前正式测量、匹配 profile、独立 API 对照及重构前后比较
见[固定容量实验](../experiments/nosa_motivation/README.md)。报告分别说明数值验收和未达到的
性能门槛；cache 账本不等于进程物理峰值。

CUDA offload 当前要求 SM90、BF16、D128、GQA16 和 native attention。融合版本中
`offload_query_tile_size=128` 仅控制首次读取流量的 query 分组，attention 处理完整
query batch；`offload_fetch_ctas=96` 限制同时参与 fetch 的 attention CTA 数，实际
数量不超过 cooperative grid 大小，不单独保留 fetch CTA。串行与融合共用 native
initialization，合并 metadata reset、page-0 padding 与 strided suffix staging；
首用规划仍为后续独立 launch。
原 owned 路径的完整 32 层 64K+1K 数值检查、cache 分配观测和单层
stripe / page-envelope overlap 结果见 [NOSA 模型](../models/nosa/README.md)及
[offload 实验](../experiments/nosa_offload_overlap/README.md)，按各自源码与 run ID 解读。
固定 P/NH 的当前验收与请求观测另见[固定容量实验](../experiments/nosa_motivation/README.md)，
两种路径的记录不互相替代。此实现使用本机 pinned
DRAM，未验证 CXL/RDMA；上述 owned/普通 budget 路径不复用跨请求已取回的 HBM 块。
用户固定 history 的跨请求保留由 `serving.persistent` / `prefix_pool` 管理。

## DeepSeek ECHO token cache

[sparse_token_pool.py](sparse_token_pool.py) 提供由模型/backend 持有的共享 host arena 和
逐层有限 HBM pool，host 按 64-token page 分配。各 session 持有独立 page table、
history indexer 状态与长度；持久 token 经 page table 映射为全局 host ID，逐层双向映射维护
其 HBM residency。[sparse_token_cache.py](sparse_token_cache.py) 提供 session/layer
view、完整 resident 存储、工作集保护、精确 recall 与缓存事务，record 宽度和 dtype
由模型显式提供。用户级 LRU 释放 host pages，token eviction 只失效 HBM 映射。
[host_allocation.py](host_allocation.py) 按 PyTorch pinned allocator 的 2 的幂次
档位逐层预留 host backing，并让逻辑视图的 storage 显式保留完整档位以便分配后核验。
普通 CPU reference 不做该取整；传输字节始终按实际 record 计算。DeepSeek dense
backing 使用同一规则。当前容量与验证范围见
[cache 管理报告](../experiments/cache_management/README.md)。
[DeepSeek SM90](../models/deepseek_v32/README.md) 为主 KV 使用 BF16 512 latent + 64 RoPE
record，indexer K/scales 仍常驻 GPU。每个外层 query batch 先写 index-K，再执行
融合历史预取与精确 top-k；实际 miss 领取槽位时才淘汰。当前主 KV 直接写 HBM，并
异步写回 host；这是普通持久 append 的行为。精确 recall 等待对应写回完成，attention
消费完成前保护其 slots。选择并集超过 pool 时只拆分 query 消费，保留完整的每-query
选择并统计跨组重读。

DeepSeek GR 的 `echo/serial_sparse` 使用显式 transient step。session 容量与 NH
配额只覆盖 `padded(H)=ceil(H/64)×64`，candidate 不申请 host pages，也不写回 DRAM。
candidate 一次整批进入各层，history prefill 的 chunk 调度保持不变。
每层 records 在 P 个历史槽和 sentinel 后额外保留 Amax 个 GPU 行；候选逻辑 ID
`H+i` 直接对应物理行 `P+1+i`。host→device 映射仍长 NH，device→host、priority
和 free bitmap 仍长 P+1，候选不进入这些历史映射或淘汰状态。混合选择只对历史部分
精确 recall；候选保持在 GPU 尾部，是否拆分消费也只由选中的历史并集与 P 决定。

`begin_transient(A)` 保持 committed length 为 H，只推进本次可见的 written 范围；
`commit` 拒绝提交临时候选。正常结束调用 `discard_transient`，等待相关 GPU 操作
完成后清除候选可见范围与所有权。候选共享存储跨用户复用，
同层仅允许一个 owner；history 的内容、页所有权与已提交 indexer 状态保持独立。
候选执行失败直接报错终止，runner 释放对应 session，不恢复或重试。
当前层的候选 indexer 直接与 history 拼入 backend 共享的一层 `[H+A,d]` K/scales
workspace，不另存逐层候选 indexer。这份 workspace 和每层候选 KV 尾部一起计入
共享资源，不按用户数重复计费。完整请求的 H+A 上下文和 A 执行上限仍须在准入前检查。

普通持久模型为所选各层调用 `begin_step`，每个 token chunk 顺序执行全部所选层；全部层、
输出与 GPU 同步成功后统一 `commit`。失败先 drain，再回滚本次 step；无法安全 drain
时禁用 backend。`truncate` 只失效该 session 的 suffix，不重置共享池。
实验在计时外保存与恢复全局 pool、session 和 prefix 状态；快照要求 host ownership
及 prefix 内容不变，其额外诊断存储单独披露。
这条路径使用本地 DRAM，不包含 CXL/RDMA，也未接入 NOSA 的 `CacheManager`。
当前非 GR 验收限定真实 checkpoint 第 0–2 层的 64K + 1K；模型保留完整层数能力。
直接 `backend.extend` 和非 GR 模型保持上述持久事务。通用字节预算 serving 的
`hbm/dense_prefetch` 使用持久 append 后 truncate；`echo/serial_sparse` 使用显式 GPU
transient candidate。固定 P/NH 的四个方案均在 GPU 临时执行 candidate 后 discard。
各路径的独立验收、静态规划与完整请求观测见
[两模型 cache 管理报告](../experiments/cache_management/README.md)，不能用短请求正确性
检查或旧实现的容量记录证明当前物理容量。
不能将用户历史命中等同于所选历史 token 已驻留 HBM。
