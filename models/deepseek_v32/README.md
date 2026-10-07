# DeepSeek V3.2

真实前三层支持四种 cache 实现和显式准备的完整 extend CUDA Graph；独立验收、
正式计时与 profile 的当前状态见[四方案 MFU 实验](../../experiments/deepseek_v32_mfu/README.md)。
十 block C10 GR 工作负载已完成固定 P/NH 的 12 点四方案独立 check 和正式 bench，见
[motivation 实验](../../experiments/deepseek_v32_motivation/README.md)。本轮没有新采集
profile；保留的 profile 只对应旧 H65536/A128 单点。
固定历史的临时候选语义与前三层 MFU 的持久追加分开验收。

独立 SM90 checkpoint 推理实现，支持 ECHO prefill/extend。完整结构包含 61 层、
embedding、3 个 dense MLP、58 个 MoE、final norm 和 LM head。硬件算子按功能位于
[operators/deepseek_v32](../../operators/deepseek_v32/README.md)。
当前非 GR benchmark 仅执行真实 checkpoint 第 0–2 层，包含 embedding、final norm
和末 token LM head；GR 对照另用十个独立 dense block 的输入重放工作负载。
完整层数入口仍保留，本轮验收不等于完整 61 层验证。两条工作负载分别报告，
真实三层的数值、性能和 profile 边界见
[四方案 MFU 实验](../../experiments/deepseek_v32_mfu/README.md)。

| 模块 | 用途 |
| --- | --- |
| [model.py](model.py)、[infer.py](infer.py) | 完整模型与 CLI：跨 GPU 放置权重、按 token chunk 执行全部层、统一缓存事务及 LM head |
| [layers.py](layers.py) | 双路 hidden/residual、RMSNorm、dense / grouped MoE 与 checkpoint 权重加载 |
| [attention.py](attention.py) | 执行外层 query batch 的投影、融合 prefetch、精确 top-k / recall、物理 ID remap 与 sparse MLA |
| [config.py](config.py)、[checkpoint.py](checkpoint.py) | checkpoint 配置、tensor 读取与 BF16/FP8 线性权重 |
| [projections.py](projections.py)、[rotary.py](rotary.py) | MLA/indexer 投影、YaRN 与 RoPE 配对 |
| [nonmatrix.py](nonmatrix.py) | FlashInfer RMSNorm、残差归一化与 SiLU 适配，保留 FP32 checkpoint norm 权重 |
| [adapter.py](execution/adapter.py) | 单卡 GR serving 的 10 个 dense block / 输入复制工作负载与可复用用户 cache session |
| [replay.py](replay.py) | checkpoint 副本的独立加载、source 层映射及 hidden/residual 输入克隆；物理层数由入口显式指定 |
| [session.py](cache/session.py) | 用户 session 状态与借用双缓冲的 dense cache view |
| [prefetch.py](cache/prefetch.py) | 逐层 pool 的连续历史 DMA、异步 ticket 与拷贝 drain |
| [planning.py](execution/planning.py) | 根据模型维度和执行上限计算固定 P/NH 或字节预算的资源计划 |
| [compute_graphs.py](execution/compute_graphs.py) | projection/finish 计算图、动态 causal bounds 与 graph storage 审计 |
| [extend_graph.py](execution/extend_graph.py) | 固定 prefix 的完整 extend 图、输入与 cache 身份检查、主机状态提交及图内 IO 所有权 |
| [pipeline.py](execution/pipeline.py) | 本地四种 serving 方案的 attention/cache factory 与指标汇总 |
| [tests/](tests) | ECHO checkpoint / block / 全模型调度事务参考测试 |

## 运行

从仓库根目录执行；第三方源码准备见 [依赖说明](../../3rdparty/README.md)。SM90 使用基础环境：

```bash
python3 scripts/prepare_3rdparty.py --init
uv sync
source .venv/bin/activate
python -m models.deepseek_v32.infer --help
python -m models.deepseek_v32.infer \
  --model /preset-models --num-layers 3 --devices 0 \
  --input-ids /path/to/input_ids.json --history 65536 --chunk-size 1024 --offload --slots 16384
```

`input_ids.json` 是单个请求的 token ID 列表，64K + 128 场景共 65,664 项；省略
`--offload` 使用 resident 主 KV。权重常驻各 GPU，hidden/residual 在层放置边界传输。
每个 token chunk 顺序经过全部层；所有 chunk、输出与 GPU 同步成功后才提交请求长度。
默认只计算最后 token 的 LM head。前三层 benchmark 的输入生成与可复现命令见
[四方案 MFU 实验](../../experiments/deepseek_v32_mfu/README.md)。

显式 `--num-layers 3 --devices 0` 只加载并顺序执行 checkpoint 第 0–2 层；实验入口
固定使用这一范围。模型 CLI 省略 `--num-layers` 仍执行完整模型。Python 接口为
`DeepSeekEchoModel(..., num_layers=3)`；`forward(..., return_hidden=True)` 另返回
全部输入 token 经 final norm 后的 hidden，供数值比较。前三层顺序传播是 checkpoint
工作负载，不能称为独立训练的三层模型或完整 61 层输出。

Python 接口 `set_cache_method(method)` 接受 `hbm`、`echo`、`serial_sparse` 和
`dense_prefetch`，每次切换创建独立空 cache。真实前三层的单 GPU dense 路径可预先调用
`prepare_compute_graphs(query_sizes)`，跨方法和 cache 重建复用同一组 projection/finish
计算图；cache、选择、召回与事务仍在图外。纯计算图策略为
`deepseek-compute-islands-v4-bound-inputs`，projection 用同一动态位置生成 RoPE
和 indexer causal ends，第 1、2 层直接使用前一层 finish 图的输出。

完整 extend 图通过 `prepare_extend_graph(token_ids, *, all_logits=False,
return_hidden=False, capture_scope=None)` 显式启用，策略为
`deepseek-full-extend-graph-v2-dense-late-wait`。准备时预热并捕获一次完整 query batch，随后恢复
匹配的 prefix。之后相同输出模式的 `forward(token_ids)` 使用一次 graph replay，
覆盖 embedding、三层计算、indexer、cache 写入与精确召回、H2D/D2H、ECHO hint
备份、final norm 和 LM head。输入校验与 staging、事务开始、同步和主机提交在图外；
同步成功后先应用 cache 的主机状态增量，再统一提交所有层。

该入口支持单 GPU 的 dense 层前缀和单个 session，offload 要求 `H+A<=P`。
图绑定 H/A、输出模式、权重与精度、cache generation/storage、clock 及驻留状态。
每次 replay 前须恢复同一 prefix，并设置与 capture 一致的 cold/warm 状态；这些
条件变化会直接报错。默认 logits、`return_hidden=True` 和 `all_logits=True`
分别准备图。返回的 tensor 借用 graph storage，需跨后续 replay 保留时由调用者复制。
`forward(..., use_extend_graph=False)` 可显式执行原路径作对照。完整图当前只用于
固定 prefix 的真实层执行，不支持任意增长的 history，也未接入 C10 或 NOSA。

真实三层的 dense 完整图让本层 projection、indexer 和 top-k 与历史 H2D 同时推进；
在 append 和主 attention 前等待数据及映射就绪，再发起下一层预取。
跨层预取时释放当前层 cache lease，随后重新取得它完成 append、recall 和 MLA。
Prefill 与普通 forward 保留原来的层前等待。新调度的验收与测量状态见 MFU 实验。

两类图分别记录规划上限、static allocated、private reserved 和设备已用量。
完整图引用 cache storage，重建或释放 cache 前先销毁图；其异步写回源保留至图完成。
模型关闭或异步失败进入 poisoned 状态后拒绝新执行和资源变更，保留
`synchronize()`、`close()` 用于清理。

MFU 默认 H=65,536、A=128、chunk=1,024、P=65,664，四方案均持久提交新增 token。
dense 要求 `P >= session capacity`。独立 correctness 通过 `return_hidden=True`
比较完整 hidden，正式 bench/profile 使用默认输出边界，包含阶段末的 final norm
与末 token LM head。默认 cold extend 恢复 prefix 后仅清除 offload 主 KV 的 HBM
驻留，保留 DRAM 与 resident indexer；warm 模式另行验收和测量。

模型 CPU 回归：

```bash
.venv/bin/python -m pytest \
  models/deepseek_v32/tests/test_echo_model.py \
  models/deepseek_v32/tests/test_echo_block.py \
  models/deepseek_v32/tests/test_echo_infer.py -q
```

## KV 与 RoPE 语义

- SM90 ECHO 主 KV 每 token 1152 B：512 个 BF16 latent 加 64 个 BF16 RoPE key。
  Indexer FP8 K 与 FP32 scale 常驻 GPU；主 KV 使用 resident 存储，或
  [共享 token pool](../../cache/sparse_token_pool.py) 的 pinned DRAM backing 与逐层有限 HBM slots。
  模型/backend 持有 pool，session 持有独立 host page table 与 indexer 状态。
  持久 append 在 indexer 历史预取后执行精确 top-k，再直接写入当前主 KV 并异步写回
  host，补齐剩余 miss；GR 临时候选另按下节使用共享 GPU 尾部，不写回 host。
  过大的 query 选中并集拆分消费，
  不截短 query 的精确选择。此路径使用本地 DRAM，不包含 CXL/RDMA。
- H<=P 的 native 精确召回由 GPU miss count 驱动最多 128 CTAs 的 gather，再在
  同一 stream 发布映射，省去逐层 host count 读取。H>P 保留完整并集检查和 query
  拆分。每个 session/layer 使用 8 个 int64 计数器，共 64 B，分别记录融合预取、
  selection/eviction 和成功 recall；容量计划与实际分配均包含这些计数器。
- Offload attention 入口消费已召回的 HBM records 与物理 ID，复用 device-only MLA。
  融合 prefetch 属于 indexer；当前 attention kernel 本身不读取 host backing。
- 主 MLA 的 64 维位置分量用 interleaved 配对（`is_neox=False`），indexer 前 64 维用
  split-half 配对（`is_neox=True`）。Attention scale 按原始 192 维 QK 计算并应用 YaRN mscale。
- Indexer RoPE 后直接进行 FP8 量化，不执行 Hadamard；精确 top-k 选择当前量化 logits
  中的最大值。该变更可能改变量化值与选择，不能视为与旧 Hadamard 输出逐位等价。
  RoPE、norm、SiLU 与 top-k 复用 FlashInfer；量化按 KDA 流程独立验证和测量。

## 相关实验

真实前三层的 dense prefetch 使用连续布局和 `cudaMemcpyAsync`。纯计算图与完整
extend 图的独立数值验收、正式计时和 profile 分别记录在
[SM90 四方案 prefill/extend MFU](../../experiments/deepseek_v32_mfu/README.md)，
以该页的实现版本、输入、计时边界和来源为准。完整图的准备及 prefix 恢复不计入
执行时间；完整请求时延、三层窗口、逐算子 MFU 和局部 IO 重叠分别解释。

C10 已完成 12 点四方案的独立 check、正式 bench 和请求内存观测，本轮未采集
profile；前三层结果与 C10 验收分别报告。
十 block GR 工作负载的固定 P/NH 容量检查见
[统一 cache management 实验](../../experiments/cache_management/README.md)。
原有 DeepSeek / SM120 实验的有效历史结果保存在
`local/experiments/legacy/deepseek_v32/`，不进 Git，仍按原 run ID 和测量环境解读。

共享 GR 请求生成使用 [request_format.py](request_format.py)：DeepSeek 历史请求模板与长上下文预算适配。
用法见 [GR 生成器](../../GR/README.md)。

## 单卡 GR Serving 工作负载

共享 ECHO cache 与模型级 chunk 调度已接入。固定 P/NH 的静态规划、实际分配和
完整请求验收分别报告。用于 cache management 的十 block 本地四方案观测仍为
H65,536/A128、16 用户两轮的 DMA check、正式 bench 和请求内存数据，来源 run ID 为
`deepseek_dma_c10_bench_20261006_01`。这些运行没有填满 NH，也不能代表任意用户数
或 H/A 配置。容量范围与可核验结果见
[统一 cache management 实验](../../experiments/cache_management/README.md)。
旧 4 GiB / W / chunk 对照已撤回，不再据此指定默认 chunk 或给出性能排名。

[adapter.py](execution/adapter.py) 提供本次 GR serving 对照使用的单卡工作负载：
从真实 checkpoint 前三层重复加载 10 个独立的 dense block，source 顺序为
`[0,1,2,0,1,2,0,1,2,0]`，不执行 MoE。每个副本复制其 source block 的 hidden 与
residual 输入，并使用独立权重、主 KV 和 indexer cache。加上真实 embedding、final
norm 和 LM head 共 7,827,793,408 参数；其中 dense backbone 为 5,974,428,160 参数。
这个工作负载用于控制计算和缓存大小，不能作为经过训练的 DeepSeek 8B 模型；它与
上述非 GR 前三层 benchmark 分别报告。

`DeepSeekServingBackend.runtime_driver(policy)` 接入公共 token runtime 契约，每个用户 session 在多次请求
间保留固定 prefix。策略包括 `hbm`、`echo`、`serial_sparse` 和 `dense_prefetch`：
前两种复用 resident / ECHO 路径，串行 sparse
在 indexer 与精确 top-k 完成后召回缺失 records。通用预算模式的 dense 使用两块
完整 layer staging 和独立 CUDA stream，在当前 block 执行时预取下一层的全部
历史主 KV；固定 P/NH 的 dense 直接使用逐层 pool，见下文。四者均执行相同的
稀疏选择与 MLA。所有用户 cache 的主 KV、indexer、映射、ECHO counter 及 dense staging
在 budget 模式下都计入 cache budget；权重和临时激活另计。固定 P/NH 模式按指定
pool/arena 容量执行，并记录物理显存和 DRAM 占用；cache 统计仍不等于进程峰值。

模型装配已拆分为 replay、session、资源规划和 pipeline。资源与 session 计划列出具名
分配项，创建 session 时消费同一份不可变计划；共享生命周期管理 owner、generation、
执行 lease 与失败后的资源保留。逻辑 token ID 通过 `TokenSelection` 传给 attention，
不新增 tensor 搬运或改变 indexer/prefetch 融合顺序。`num_layers` 由调用入口显式
传入；GR 入口与当前实验使用十个物理 block。执行时把 session 的 attention runner 和
chunk 大小作为参数传给 block，模型不再临时挂载用户 attention/cache。
数值验收、正式计时和诊断 profile 使用独立入口；
报告中的 run ID、源码和测量边界共同标识结果。当前批次为
`deepseek_motivation_matrix_20261007_01`，固定 P=65,536、NH=16,777,216，覆盖
H∈{4096,16384,65536}、A∈{128,256,512,1024} 的 12 个点。每点四方案各执行
32 个请求，使用 v4 局部计算图；逐点配对的独立 check、正式 bench 及验收依据见
[motivation 实验](../../experiments/deepseek_v32_motivation/README.md)。原
`deepseek_dma_c10_bench_20261006_01` 仍作为统一 cache management 报告的独立
内存来源保留，不用其旧计时代表本轮结果。
容量与内存说明见[统一 cache management](../../experiments/cache_management/README.md)。

本轮矩阵没有新采集 profile。保留的 `deepseek_motivation_rerun_profile_20261006_01`
只匹配旧 H65536/A128 单点 `deepseek_motivation_rerun_bench_20261006_01`；
局部 DMA/计算重叠的适用范围见 motivation 报告。该 profile 不代表新矩阵，
每点的一轮请求轨迹也不证明稳定性能排序或真实 GR 任务质量。

GR 的 `echo/serial_sparse` 在 session 计划中将保留容量设为 H，
host pages 与私有 history indexer 都按 H 保留；`extend_candidate` 使用 backend
共享的 GPU 临时空间处理候选。每层主 KV storage 为 `[P+1+Amax,576]`，前 P+1 行
包括 P 个历史槽和一个 sentinel，保留原映射；尾部只存本次候选，不分配 host ID 或淘汰元数据。
candidate 一次整批进入各层。当前层的候选 indexer 与该用户 history 直接拼入一层
共享的 `[H+A,d]` K 和对应 scales workspace，容量按执行上限预留；不保留逐层候选
indexer。候选相关共享空间不乘以用户数；各层和各用户的历史数值仍独立。

候选正常结束后 discard 临时 KV/indexer 状态，恢复 history 的有效长度和 indexer
hint；相关异步使用结束后才能复用共享空间。候选失败直接报错终止，runner 释放该
用户 session，不恢复或重试。NH 按 `ceil(H/64)×64` 准入，
因此在已规划的 A 和 H+A 上限内改变 A 不会重建相同 history。请求仍执行全部候选
hidden 和末 token logits，history-only 只改变保留容量。
通用 budget 模式的 `hbm/dense_prefetch` 保留原 session 容量及 extend 后 truncate 的接口；
直接 `backend.extend` 与上述非 GR `DeepSeekEchoModel` 继续持久提交新增 token，并
保留原有 chunk 执行语义。

固定 P/NH 模式已接入四方案：HBM-only 按 H 扣除独立 HBM token 配额 P，
`echo/serial_sparse/dense_prefetch` 按 H 扣除 NH 的 host 页配额，并复用相同的逐层
P 槽历史 cache。固定模式 dense 要求 H<=P，使用
[cache/prefetch.py](cache/prefetch.py) 在独立 stream 通过 `cudaMemcpyAsync` 提前
搬入下一层完整历史。每个 session 使用连续递增的 host 页，逻辑 token i 对应
HBM slot i+1；每层 pool 直接提供连续目标，不另分配完整 layer staging。
已认证的连续完整命中直接复用；其他驻留状态拷贝全部已初始化历史，并按实际
字节数记账。映射失效和拷贝后的发布仍由 native kernel 执行，不能将整个 dense
路径称为不使用 SM。ticket 借用已有 host/device storage，不分配 GPU ID/count
scratch，对应预留为 0；消费前等待 copy event，回滚或释放前 drain，无法确认完成
时保留 storage 并禁用复用。GPU trace 的实际交叠和完整阶段延迟分别见对应实验。
HBM-only 的 candidate 紧接 resident history，offload
的 candidate 使用 pool 尾部；四方案都使用共享的合并 indexer workspace，结束后
丢弃候选。端到端对照见 [motivation 实验](../../experiments/deepseek_v32_motivation/README.md)。

通用 budget 模式的 dense 将双缓冲及其 stream/events 归于 backend，每个用户仍独立持有历史
host records、映射和 indexer 状态。一次完整 prefill/extend 借用双缓冲，槽位覆盖前
等待前一个 consumer，归还前等待包括未消费预取在内的全部异步操作。
runner 从构造到 close 绑定唯一准入 owner；其关闭只释放 session，最外层再关闭
backend。以上 C10 容量、临时候选和输入重放边界与真实前三层 MFU 分别报告。

C10 矩阵的 12 份独立 check 共保存 1,536 份完整候选 hidden/logits，
1,152 组 offload/HBM 对照逐字节一致；逐点记录见 motivation 报告的验收索引。
模型测试与这些数值 check 不替代固定 P/NH 容量填满实验、正式计时或未运行的配置。

`sparse_pool_tokens` 配置 backend 每层共享容量，`host_arena_tokens` 配置全局 host
容量；原 per-session `slots` 参数已移除。固定 P/NH 模式按实际 prefill chunk 与
candidate 大小执行，不使用独立 `workspace_query_tokens` 准入。通用 budget 模式仍
可显式指定该预留上限。prefill 的每个 chunk 依次执行十个 block，一次请求只提交一次；
GR candidate 固定整批执行，不随 prefill chunk 再拆分投影、MLP 或 indexer。资源计划
分别保留 H、A 和 H+A 的上限；候选临时执行不提交持久 history。

输出包含全部 candidate normalized hidden，同时执行最后 token 的 LM head；当前 NOSA
serving backend 只输出 hidden，其执行边界不同。跨方案数值对照使用同一模型的相同边界。

单元测试及可选的真实 checkpoint 正确性检查：

```bash
source .venv/bin/activate
python -m pytest models/deepseek_v32/tests/test_serving_backend.py -q
DEEPSEEK_SERVING_CHECKPOINT=/preset-models \
  python -m pytest models/deepseek_v32/tests/test_serving_checkpoint.py -q
```

真实 checkpoint 检查使用独立空 cache 构建 2,304-token prefix，并比较 16 / 23-token
candidate 与截短后的复访；所有策略的全部 candidate hidden，以及每个副本相对其 source
的 hidden / residual 均通过 bitwise 对照。这是正确性验证，serving 性能结果由对应实验
单独报告。GPU candidate 的短检查同样比较 2,304-token history 与 16 / 23-token
候选的全部 hidden/logits，确认与 HBM 逐位一致、候选 D2H 为零、history KV 与 indexer
不变；该工程检查不提供完整多用户容量或显存峰值结果。实现契约见
[模型规则](AGENTS.md)，完整请求与测量边界见上述对应实验。
