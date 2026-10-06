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
历史 cache，NH 只负责 DRAM 历史容量。固定模式 dense prefetch 要求 H<=P，在
独立 stream 用 `cudaMemcpyAsync` 将下一层连续历史搬入该层 P 槽。每个 dense
session 的 host 页必须连续递增，HBM 使用逻辑 token i 对应 slot i+1 的连续布局。
已认证的连续完整命中不重搬；未认证驻留须按实际执行的完整拷贝计费，不能称为
只搬 miss。不额外分配两层完整 staging。消费、回滚或释放前等待对应拷贝完成。
四方案 candidate 均整批 GPU
临时执行，结束后 discard。通用 budget 模式的原 dense 双 staging 路径另行保留。
真实前三层模型的四方案入口使用普通持久 append；其 dense prefetch 要求
`P >= session capacity`，不能套用仅覆盖固定 history 的 H<=P 条件。
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

受支持的 native 精确召回在 H<=P 时以 GPU miss count 驱动有界 gather，随后在同一
stream 发布映射，不逐层读取 host scalar。当前 sparse gather 上限为 128 CTAs。
H>P 时仍检查完整精确并集，超出 pool 后拆分 query 消费，不裁剪单 query 的选择。
已认证的全命中路径、时钟边界、显式 reference 和不满足 native 条件的正常分派
继续保留；不能将局部去同步表述为整条路径没有 host 同步。没有读取 miss count
的未认证全命中调用可以保守失效 CPU append/residency 证明，但必须保留 GPU 映射、
record、priority/FIFO、free bitmap、clock 和流量统计的等价语义。

FIFO 分配仍须先用空槽、再淘汰较低优先级的槽，并保护当前精确选择。相同优先级的
slot 可以任意选取，不要求稳定顺序；这项放宽只适用于 cache 分配与淘汰，不改变
indexer 的精确 top-k 或其并列规则。验收须检查实际选中 KV、双向映射、free bitmap、
clock、计数和淘汰优先级是否合法，不能仅因并列 slot 的物理编号不同判错。若新的
并列选择改变后续命中或搬运量，须按实际轨迹重新测量，不能沿用旧流量与性能结论。

融合 ECHO indexer 的共享 KV stage 须等 WGMMA 和 scalar scale 读取都完成后才能
归还给 TMA producer。WGMMA wait 不能单独作为 scale 读取完成的证据。调整释放
位置时须核对实际编译指令中的数据依赖，并用跨多轮 query 的 stage 复用测试逐位
验收分数、预取 KV、映射与计数；不能只检查单轮 query 或放宽分数容差。

`sparse_selection_allocate_free` 只用于已有独占操作内、池中仅有一个活跃 session、
且 `host_written_end<=P` 的精确召回；provider 未提供该入口时保留原分派。合法状态下，
该 session 的已驻留记录数 L 与选中 miss 数 M 满足 `L+M<=H<=P`，因此空槽足够。
这是容量证明，不能据此认证历史已驻留。GPU 分配前仍核验 `free>=M`，核验失败直接
报错，不切换分配器。多 session 保留原 FIFO 分配；session 释放须清除全部相关映射并
确认 CUDA 完成后，才能从活跃集合移除。执行失败后遵守原清理契约，不能绕过清理继续召回。

普通持久 append 的 `sparse_append_free` 仅用于同样的单 session 独占操作，且新尾部
结束位置不超过 P。它先并行核验全池 priority 并生成空槽 mask，再选择足够的实际
空槽，沿用原 planned append 发布 record 与映射。容量条件不能代替驻留证明；
GPU 须检查选中槽的 free bitmap、owner 与空槽数量，失败直接报错。多 session、
超出 P 或 provider 不具备该入口时保留原正常分派。显式候选和 dense 连续槽位不走
这条普通 append 分配路径。

完整模型独占执行一个 prefill step 时，若全部待写入 token 都能放入 P，且历史已
认证驻留，可以将中间 chunk 的 ECHO hint 更新推迟到最后一个 chunk。期间不消费
这些 hint；最终 hint 仍按最后最多四行有限 logits 的原 FP32 归约生成。普通逐层
调用、H>P、未认证驻留、候选及显式诊断继续即时更新，失败时退出延后更新作用域并
按原事务规则回滚。不能仅凭 H<=P 推断历史已经驻留。

每个 session/layer 的计数 slab 为 8 个 int64，共 64 B：前 3 项保存融合 prefetch
统计，接着 4 项保存 native selection/eviction 统计，最后 1 项累计已成功发布的
native recall。读取指标时合并这些计数，不重复累计；session 规划、离线容量、
reset、释放和实际分配账本必须覆盖完整 slab。

`cache/prefetch.py` 的 dense lookahead 使用连续 pinned host/HBM span 和
`cudaMemcpyAsync`，不得静默退回 mapped-host gather。native 映射失效与发布按
stream 顺序分开执行，避免旧映射清除覆盖新映射；copy event 被消费 stream 等待后
才能认证连续驻留。同次执行的 ticket 使用同一 caller stream，消费前 wait，回滚
或释放前 drain；异步未完成时保留 host/device storage。提交或完成失败后禁用复用。
DMA ticket 借用已有 storage，不分配 GPU ID/count scratch，`dense_ticket_reservation`
为 0；新增实际 storage 仍须计入容量。检查实际 H2D 活动、字节数、地址连续性和
compute/IO 交集，不能仅由创建 stream 或异步 API 推断重叠或性能收益。普通 sparse
pool 保留原页分配与 FIFO 策略，不套用 dense 的连续布局。

## 计算图与模型生命周期

纯计算图策略为 `deepseek-compute-islands-v4-bound-inputs`，用于 prefill 和现有 C10
路径，只捕获 projection 和
finish 的纯计算，cache 事务、选择、召回与 IO 留在图外。projection 用同一个动态
整数位置生成 RoPE 与 exclusive causal ends，并持有不可变的零 starts；
indexer 借用这些 bounds。同一 query shape 的层共享 positions、零 starts
和动态 start；位置未变时不重复写入 start。第 1、2 层直接借用前一层 finish 的
输出，后续 C10 副本仍复制 source 输入到各自独立的 graph storage。借用的输出
计入其所属 graph private pool，不重复计入 static storage；规划上限与实际
allocated/reserved 分别报告。真实模型 graph bank 仅支持单 GPU 的 dense 第 0–2
层，跨方法和 cache 重建复用同一权重的计算图，不称为完整 offload graph capture。
replay 必须核验 capture 时的精度策略、权重身份、query shape 和 residual 分支。
持久 offload 的异步 D2H 写回源须独立持有，不能被后续 replay 覆盖。

真实前三层另提供显式准备的 `deepseek-full-extend-graph-v2-dense-late-wait`：一个图覆盖 embedding、
全部层的计算、cache 操作、实际 IO、ECHO hint 备份、final norm 和 LM head。
输入校验与 staging、事务开始、完成同步和主机提交留在图外。该入口只支持单 GPU
的 dense 层前缀、单个 session 和一次完整 query batch；offload 要求 H+A<=P。
图绑定固定 H/A、输出模式、cache generation/storage、clock、驻留及追加证明。
每次 replay 前须恢复匹配的 prefix；cold/warm 状态或其他绑定变化时直接报错，
不能自动切换到图外执行。不得据此声称支持任意增长的历史、C10 或 NOSA 完整图。

该完整图的 dense 路径允许本层 projection、indexer 和 top-k 与本层历史 H2D 重叠。
在 append 修改映射及主 attention 消费 KV 前，须等待 H2D 与映射发布完成，随后发起
下一层预取。跨层预取前须释放当前层 cache lease，之后重新取得当前层 lease 才能
append、recall 和执行 MLA；不能在未完成预取时读取或修改主 KV 映射。
这项调度仅由真实三层 dense 完整图及其准备阶段显式启用，普通 forward、prefill、
C10 与其他方法保留原调度。图结束前仍须汇合全部 H2D/D2H，保留所有被引用的 storage。

共享 pool 的 capture 只在显式授权作用域内开放。结束 capture 前须汇合全部 IO
stream，保留图引用的 host/device storage 和写回源，并计入 graph private pool；
选择的规划上限、实际 allocated/reserved 和设备已用量分别报告。Replay 同步成功
后才应用捕获的主机状态增量，再统一提交所有层；GPU 计数不能重复累计。
Cache 重建或释放前先销毁图。完成状态不明时保留 owner/storage 并禁用复用。

模型 poisoned 或 closed 后，graph 准备与执行、cache 分配或切换、prefix
eviction、snapshot/restore 和 forward 均须在使用资源前失败。`synchronize()` 和
`close()` 保留清理入口；close 先 drain owner，再释放资源并清除各层 cache/attention
引用。无法确认异步完成时保留 owner/storage 并禁用复用；原始异常与清理异常均须
保留，不能恢复执行或自动重试。

## 非 GR benchmark 与 GR 工作负载边界

- DeepSeek V3.2 的非 GR benchmark 仅使用真实 checkpoint 第 0–2 层依次传播
  hidden/residual，包含 embedding、三个 dense MLP、final norm 与末 token LM head；
  不复制 block，不称为独立训练的三层模型，也不外推为完整模型性能。
  `deepseek_v32_mfu` 的 `measure` / `run.sh` 与 `profile_layers` 共用这一范围。
  模型实现保留完整 61 层及 grouped MoE 能力，但当前 benchmark 不要求运行完整 61 层。
  按 token chunk 依次执行选定的全部层，限制临时 hidden 显存。主 KV
  使用 BF16 512 latent + 64 RoPE record，indexer FP8 K/scales 仍 resident。融合
  indexer prefetch 后必须执行精确 top-k / residual recall；工作集超过 HBM pool 时
  拆分 query 消费，不裁剪每 query 的精确选择。全部层和 GPU 同步成功后统一提交；
  失败只回滚本次启动的事务。前三层四方法对照从独立空 cache 构建 prefix，每次
  extend 恢复相同 prefix，再按声明的 cold/warm 设置处理 HBM residency；cold 只清除
  offload 主 KV 驻留，保留 DRAM 与 resident indexer。两种设置分别验收和测量。
  权重加载、编译和状态恢复不计入执行时间。
  比较全部 extend hidden；当前 MFU 范围为 H=65,536、A=128，其他 A 须单独声明并
  验收。单层或单算子正确性检查不能替代声明范围内的前三层完整测量。
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
