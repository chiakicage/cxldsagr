# DeepSeek V3.2 实现与验收规则

本文件适用于 DeepSeek V3.2 模型及其关联算子、cache、serving 和实验任务。
代码路径均相对仓库根目录；同时遵守[根规则](../../AGENTS.md)和
[实验规则](../../experiments/AGENTS.md)。

当前模块能力见 [README](README.md)。容量结果见
[统一 cache management 实验](../../experiments/cache_management/README.md)，固定容量
四方案见 [motivation](../../experiments/deepseek_v32_motivation/README.md)。
下列容量公式、执行语义和验证边界仍是开发约束，不由历史结果替代。

## 固定 P/NH、候选与容量规划

当前 DeepSeek ECHO 容量实验按用户指定的 `P`（每层 HBM token pool）与 `NH`
（全局 host token arena）运行，检查给定容量是否能完成完整请求，并估算二者的联合
可行边界。该模式不以 4 GiB 等 cache 字节子预算或独立 `W` 预留决定准入；实际
chunk/candidate、权重、indexer、映射、执行临时空间和 pinned DRAM 仍须满足物理容量。
DeepSeek GR 的 `echo/serial_sparse` 通过 `retained_session_capacity` 按固定 history
长度 H 创建 session，并通过 `extend_candidate` 在共享 GPU 临时空间执行候选。
NH 只为按 64 token 对齐的 H 分配 host pages；候选主 KV 不写回 DRAM。candidate
一次整批执行，history prefill 仍按 chunk 执行。候选 indexer 与 history 直接拼入
backend 共享的一层 `[H+A,d]` K 及对应 scales workspace，不另存逐层候选 indexer，
不按用户数重复计费。请求仍须满足 H+A 的上下文上限和 A 的执行上限；在这些上限内
改变 A 不得因候选容量变化重建相同 history。
另提供离线或模型加载后的容量规划：本轮可用于 cache 与执行 workspace 的 HBM
预算为 `总 HBM × 0.9 − 模型加载占用`，不能改成 `加载后剩余 HBM × 0.9`；
CPU DRAM 规划预算为 512 GiB。规划须计入填满 NH 所需的全部 session indexer、
映射、allocator 实际容量及活跃执行 workspace；离线估算与实际完整请求验收分别报告。
HBM 额度评估须区分 PyTorch allocated、reserved 和设备已用量；allocator 缓存仍占用
HBM，不能只因 allocated 未超额就声称满足实际预算。旧轨迹的占用差额是观测值；
若将其用于未来规划，只能标为人为选择的额外扣减假设，不能称为实际预分配或当前
实现的必需开销。无需把每个离线容量候选都跑满才交付分析。
旧 4 GiB / W / chunk 对照已按用户要求撤回并清理。其他任务可继续使用根规则所述的通用
HBM / DRAM budget 模式，不能将两种模式的容量或命中结果混用。

`deepseek_v32_motivation` 单独测量固定 P/NH 的四方案端到端对照。HBM-only 通过
独立 HBM history token 配额按 session LRU 准入；三个 offload 方案均复用逐层 P 槽的
历史 cache，NH 只负责 DRAM 历史容量。固定模式 dense prefetch 要求 H<=P，提前在
独立 stream 将下一层全部历史中的 miss 搬入该层 P 槽；命中不重搬，不额外分配两层
完整 staging。消费、回滚或释放前等待对应拷贝完成。四方案 candidate 均整批 GPU
临时执行，结束后 discard。通用 budget 模式的原 dense 双 staging 路径另行保留。
观测到的 allocator 差额、规划时人为扣除的额度与实现预分配的 storage 分别描述；
不能把旧运行差额称为当前实现的固定预留，也不能默认从指定 P/NH 实验中扣除它。

## ECHO token pool 与事务

`cache/sparse_token_pool.py` 为 DeepSeek SM90 ECHO 提供 backend/model 所有的全局
pinned local DRAM pages 与逐层有限 HBM pool；session 保留独立 page table 和
history indexer 状态。GR `echo/serial_sparse` 为每层主 KV storage 增加独立的候选
尾部，P 个历史槽位及一个 sentinel 仍使用原映射，候选没有 host ID、page table
项或淘汰元数据。`cache/sparse_token_cache.py` 管理 session/layer view、精确 recall
与事务；普通 `begin_step/commit` 保留持久 append，显式 transient step 只允许
discard/rollback，候选始终从 GPU 尾部读取。record 宽度和 dtype 由模型提供。
host page 释放遵守 session LRU，token eviction 只失效 HBM 映射；不能将这套 DeepSeek
token pool 的有限 HBM pool 能力或候选临时存储策略归于 NOSA。

## 非 GR benchmark 与 GR 工作负载边界

- DeepSeek V3.2 的非 GR benchmark 仅使用真实 checkpoint 第 0–2 层依次传播
  hidden/residual，包含 embedding、三个 dense MLP、final norm 与末 token LM head；
  不复制 block，不称为独立训练的三层模型，也不外推为完整模型性能。
  `deepseek_v32_echo_prefill` 的 `measure` / `run.sh` 与 `profile_layers` 共用这一范围。
  模型实现保留完整 61 层及 grouped MoE 能力，但当前 benchmark 不要求运行完整 61 层。
  按 token chunk 依次执行选定的全部层，限制临时 hidden 显存。主 KV
  使用 BF16 512 latent + 64 RoPE record，indexer FP8 K/scales 仍 resident。融合
  indexer prefetch 后必须执行精确 top-k / residual recall；工作集超过 HBM pool 时
  拆分 query 消费，不裁剪每 query 的精确选择。全部层和 GPU 同步成功后统一提交；
  失败只回滚本次启动的事务。前三层 resident/offload 对照从独立空 cache 构建 prefix，
  每次 extend 恢复相同 prefix HBM residency；权重加载、编译和状态恢复不计入执行时间。
  比较全部 extend hidden；单层或单算子正确性检查不能替代前三层完整 64K + 1K 测量。
  非矩阵操作优先复用 FlashInfer；norm 保留 checkpoint FP32 权重和舍入前 FP32
  residual sum。Indexer RoPE 后直接量化，不执行 Hadamard。量化 kernel 优化遵循
  KDA，和已编译官方 DeepGEMM helper 比较完整 API 成本，逐位验收 FP8 数据与 scale；
  移除 Hadamard 引起的选择/输出变化另行报告，不据此声称任务质量等价。
- 单卡 GR serving 的 DeepSeek 对照另用 `models/deepseek_v32/execution/adapter.py`：
  按用户要求将真实 checkpoint 前三层独立复制成 10 个 dense block，不执行 MoE；
  每个副本复制对应 source block 的 hidden 与 residual 输入，不串接出未经验证的深层
  激活轨迹。独立权重、KV 与 indexer 状态不可因输入相同而共享；含 embedding、final
  norm 与 LM head 共 7,827,793,408 参数，明确称为 checkpoint 工作负载替身，不能
  表述为经过训练的 DeepSeek 8B 或完整 61 层验证。比较 `hbm`、`echo`、`serial_sparse`
  与 `dense_prefetch`；通用 budget 模式的后者由 backend 持有一份双 layer staging 和
  独立 stream，逐层预取完整历史主 KV；固定 P/NH 模式遵守本文件前述历史 pool 规则。
  session 独立持有 host records、映射和 indexer，只在执行 lease
  内借用当前层 view；stage 复用须等待前一个 consumer，归还前等待未消费的预取。
  DeepSeek serving 计算全部 candidate hidden
  和最后 token LM head，NOSA serving 当前只计算 hidden，跨模型延迟不能忽略这一区别。
  GR `echo/serial_sparse` 的 history-only session 与共享 GPU candidate 路径通过
  `retained_session_capacity/extend_candidate` 接入；直接 `backend.extend` 和非 GR
  模型继续持久提交新增 token，`hbm/dense_prefetch` 的通用 budget 对照仍使用原
  session/截短接口；固定 P/NH 模式的 candidate 按前述规则 discard。
  候选路径的正确性验收与旧 ECHO 容量结果分别标记，不能以旧运行宣称新实现已验收。
  此特定 GR 工作负载与上述非 GR 前三层 benchmark 分别报告。
